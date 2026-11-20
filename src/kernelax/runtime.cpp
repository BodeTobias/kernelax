// Copyright (C) 2026 Tobias Bode
// SPDX-License-Identifier: MIT

// Runtime of the kernels: the choice among their variants (interleaving K) and the XLA FFI handler.

#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "xla/ffi/api/ffi.h"

#if defined(__x86_64__)
#include <xmmintrin.h>
static inline uint64_t fpmode() { return _mm_getcsr(); }  // the floating-point mode: MXCSR
static inline void set_fpmode(uint64_t c) { _mm_setcsr((unsigned)c); }
#else
static inline uint64_t fpmode() { return __builtin_aarch64_get_fpcr64(); }  // FPCR
static inline void set_fpmode(uint64_t c) { __builtin_aarch64_set_fpcr64(c); }
#endif

extern "C" {

typedef void (*kernelax_fn)(int nt, int64_t n, void** p);

// A kernel: its variants (emit.emit_map with K = 1, 2, ...) and the chosen one (-1: not yet chosen), chosen_nt: the
// threads of the call that chose it; calls, ns and last_n: its calls through XLA so far, their time and the samples of
// the last (cpu.Kernel.stats); per, next: the seconds per sample of the variants timed on long calls so far, the
// variant of the next one (0: none yet); runs, work: the variants' runs, choices included, and the samples they ran
// (tests).
struct kernelax_kernel {
  int nvar, chosen;
  int64_t tune_min;
  double tune_time, tune_long, tune_slice;
  kernelax_fn fn[4];
  int64_t calls, ns, last_n, chosen_nt;
  double per[4];
  int64_t runs, work;
  int next;
};

// Runs kernel k on n samples (p: inputs, then outputs). Until a variant is chosen, the first call with
// n >= tune_min chooses one on its own data: variant 0 runs on all samples (the outputs, and its time), then the
// variants are compared on the first m samples, m such that a run takes about tune_slice seconds, at least 64 per
// thread (not tune_min or more: at n near tune_min that was the whole batch, the choice 14 times the call): rounds over
// the variants, >= 3 and until each ran tune_time seconds, the least total time wins (the typical time: a variant that
// is only sometimes fast loses, which matters as the choice is kept, see cpu.Kernel.persist). The chosen variant then
// writes the outputs of all samples (the comparison overwrote the first m, and sums are partial): the choice costs about
// one call and 3 x nvar slices. A call of variant 0 longer than tune_long is not repeated for the choice (that cost a
// long call 2 to 14 times its time): the next calls with n >= tune_min run the other variants, one each, then variant 0
// once more (its first time includes the first call's warm-up), each timed per sample over the whole call, and the
// fastest is chosen after that: no samples run twice.

static double timed(kernelax_fn f, int nt, int64_t n, void** p) {
  const auto t0 = std::chrono::steady_clock::now();
  f(nt, n, p);
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// New variants from a build thread (cpu.Kernel.complete): their fn[i] written first, then nvar and chosen published
// here (release; kernelax_run reads them with acquire: no call through a variant not yet written, also on aarch64).
void kernelax_publish(kernelax_kernel* k, int nvar, int chosen) {
  k->next = 0;  // (a choice over long calls starts again with the new variants)
  __atomic_store_n(&k->chosen, chosen, __ATOMIC_RELAXED);
  __atomic_store_n(&k->nvar, nvar, __ATOMIC_RELEASE);
}

void kernelax_run(kernelax_kernel* k, int nt, int64_t n, void** p) {
  const int nvar = __atomic_load_n(&k->nvar, __ATOMIC_ACQUIRE), chosen = __atomic_load_n(&k->chosen, __ATOMIC_ACQUIRE);
  if (chosen >= 0 || n < k->tune_min) {
    k->fn[chosen < 0 ? 0 : chosen](nt, n, p);
    return;
  }
  ++k->runs;
  if (k->next > 0 && k->next <= nvar) {  // a choice over long calls: the next variant on this call (variant 0 again last)
    const int v = k->next < nvar ? k->next : 0;
    k->work += n;
    k->per[v] = timed(k->fn[v], nt, n, p) / n;
    k->next += 1;
    if (k->next > nvar) {
      int c = 0;
      for (int u = 1; u < nvar; ++u) c = k->per[u] < k->per[c] ? u : c;
      k->chosen_nt = nt;
      __atomic_store_n(&k->chosen, c, __ATOMIC_RELEASE);
    }
    return;
  }
  k->work += n;
  const double t_all = timed(k->fn[0], nt, n, p);
  if (t_all > k->tune_long) {  // (its outputs are this call's; the other variants on the next calls)
    k->per[0] = t_all / n;
    k->next = 1;
    return;
  }
  const int64_t m = t_all <= k->tune_slice ? n : std::min<int64_t>(n, std::max<int64_t>(64 * (int64_t)nt, (int64_t)(n * (k->tune_slice / t_all))));
  double sum[4] = {0, 0, 0, 0};
  for (int round = 0;; ++round) {
    double least = 1e300;
    for (int v = 0; v < nvar; ++v) {  // (the variants as at the start: one added meanwhile is not half timed)
      ++k->runs;
      k->work += m;
      sum[v] += timed(k->fn[v], nt, m, p);
      least = std::min(least, sum[v]);
    }
    if (round >= 2 && least >= k->tune_time) break;
  }
  int c = 0;
  for (int v = 1; v < nvar; ++v) c = sum[v] < sum[c] ? v : c;
  k->chosen_nt = nt;
  __atomic_store_n(&k->chosen, c, __ATOMIC_RELEASE);
  ++k->runs;
  k->work += n;
  k->fn[c](nt, n, p);  // the outputs of all samples by the chosen variant
}

// Calls through XLA run on one thread, one after the other: libgomp keeps a pool of threads per thread that starts a
// parallel region, and XLA calls custom calls from several of its threads; their idle pools spun on the cores the
// working one needed (an FE residual on 8 cores: 28.7 instead of 11.7 ms), and independent kernels started at once
// oversubscribed the cores. The caller's floating-point mode (XLA's threads flush subnormals) goes with each call.
// (~10 us a call for the two wake-ups; calls from Python run on the calling thread.) The pool's thread t runs
// on the t-th CPU of the process's affinity (unless OMP_PROC_BIND or GOMP_CPU_AFFINITY decide): unbound, the scheduler
// left two of them on one core for tens of calls, each barrier waiting a time slice (3 ms) beside a background gcc or a
// foreign process; a kernel of 0.3 ms on 8 cores took 0.53-0.82 instead of 0.37-0.41 ms beside a gcc,
// 7.4-8.4 instead of 0.74-0.79 ms beside a process on one core.
struct kernelax_runner {
  std::mutex call, m;
  std::condition_variable cv;
  kernelax_kernel* k;
  int nt;
  int64_t n;
  void** p;
  uint64_t csr;
  uint64_t posted, finished;
  std::vector<int> cpus;  // the process's affinity when the thread started
  bool restricted;        // whether it is fewer than the online CPUs (taskset, MPI's or a scheduler's binding)
  int pinned;             // threads of the pool bound so far
};

static void pin(kernelax_runner* r, int nt) {  // (on the kernel thread: its pool's first nt threads, once)
  static const bool own = !getenv("OMP_PROC_BIND") && !getenv("GOMP_CPU_AFFINITY");
  if (!own || !r->restricted || nt <= r->pinned || nt < (int)r->cpus.size()) return;  // (not bound where the process
  // may run anywhere, nor with fewer threads than its CPUs: several such processes, e.g. under a container's or a
  // scheduler's mask, would all bind to the same first CPUs; 3 processes of 1 thread on 3 CPUs: 2.7x slower)
#pragma omp parallel num_threads(nt)
  {
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(r->cpus[omp_get_thread_num() % r->cpus.size()], &one);
    sched_setaffinity(0, sizeof one, &one);
  }
  r->pinned = nt;
}

static kernelax_runner* runner() {
  static kernelax_runner* r = [] {
    kernelax_runner* r = new kernelax_runner{};  // (never destroyed: its thread waits on it until the process ends)
    cpu_set_t all;
    if (sched_getaffinity(getpid(), sizeof all, &all) == 0)
      for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &all)) r->cpus.push_back(c);
    r->restricted = r->cpus.size() < (size_t)sysconf(_SC_NPROCESSORS_ONLN);
    std::thread([r] {
      pthread_setname_np(pthread_self(), "kernelax_pool");  // (at most 15 characters)
      std::unique_lock<std::mutex> l(r->m);
      for (;;) {
        r->cv.wait(l, [r] { return r->posted != r->finished; });
        l.unlock();
        pin(r, r->nt);
        set_fpmode(r->csr);
        kernelax_run(r->k, r->nt, r->n, r->p);
        l.lock();
        r->finished = r->posted;
        r->cv.notify_all();
      }
    }).detach();
    return r;
  }();
  return r;
}

static void run_on_kernel_thread(kernelax_kernel* k, int nt, int64_t n, void** p) {
  kernelax_runner* r = runner();
  std::lock_guard<std::mutex> one(r->call);
  std::unique_lock<std::mutex> l(r->m);
  r->k = k, r->nt = nt, r->n = n, r->p = p, r->csr = fpmode();
  ++r->posted;
  r->cv.notify_all();
  r->cv.wait(l, [r] { return r->finished == r->posted; });
}

// Kernels by key (hash of the source): compiled XLA programs refer to kernels by key, not by address,
// so that a program from another process (persistent compilation cache) fails instead of crashing.
static std::mutex registry_mutex;
static std::vector<std::pair<uint64_t, kernelax_kernel*>> registry;

void kernelax_register(uint64_t key, kernelax_kernel* k) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  for (auto& e : registry)
    if (e.first == key) {
      e.second = k;
      return;
    }
  registry.emplace_back(key, k);
}

static kernelax_kernel* lookup(uint64_t key) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  for (auto& e : registry)
    if (e.first == key) return e.second;
  return nullptr;
}

}  // extern "C"

namespace ffi = xla::ffi;

static ffi::Error Call(ffi::RemainingArgs args, ffi::RemainingRets rets, int64_t key, int64_t nt, int64_t n) {
  kernelax_kernel* k = lookup(static_cast<uint64_t>(key));
  if (!k) return ffi::Error::Internal("kernelax: kernel not built in this process (persistent compilation cache?)");
  std::vector<void*> p;
  for (size_t i = 0; i < args.size(); ++i) {
    auto b = args.get<ffi::AnyBuffer>(i);
    if (b.has_error()) return b.error();
    p.push_back(b->untyped_data());
  }
  for (size_t i = 0; i < rets.size(); ++i) {
    auto b = rets.get<ffi::AnyBuffer>(i);
    if (b.has_error()) return b.error();
    p.push_back((*b)->untyped_data());
  }
  const auto t0 = std::chrono::steady_clock::now();
  run_on_kernel_thread(k, static_cast<int>(nt), n, p.data());
  const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  __atomic_fetch_add(&k->calls, 1, __ATOMIC_RELAXED);
  __atomic_fetch_add(&k->ns, ns, __ATOMIC_RELAXED);
  __atomic_store_n(&k->last_n, n, __ATOMIC_RELAXED);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kernelax_call, Call,
                              ffi::Ffi::Bind().RemainingArgs().RemainingRets().Attr<int64_t>("key").Attr<int64_t>(
                                  "nt").Attr<int64_t>("n"));
