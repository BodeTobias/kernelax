// Copyright (C) 2026 Tobias Bode
// SPDX-License-Identifier: MIT
//
// kernelax's GPU runtime (gpu.py): the kernels (cubins compiled per traced function, kernel kernelax_k) registered by key and
// run by the XLA FFI handler kernelax_gpu_call on XLA's CUDA stream: one thread per sample, blocks of the kernel's size,
// the arguments (sample count, inputs, outputs) as kernel parameters; outputs that are sums zeroed before. A kernel
// may come in variants (gpu.py: ptxas -O1, dense layers as loops, where the kernel spills): the first call with at least TUNE_MIN samples
// times variant 0 on all of them (its outputs); a call up to tune_long then times each variant twice more (CUDA events on
// the stream) and keeps the fastest; a longer one is not repeated (that cost 2 V + 1 times the call): the next calls run
// the other variants, one each, then variant 0 again (its first run loaded it), and the fastest is kept after that
// (kernelax_gpu_choice: stored in the index). Built once.
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int64_t TUNE_MIN = 4096;
double tune_long = 0.02;  // seconds (gpu.TUNE_LONG, kernelax_gpu_tuning)

struct Variant {
  std::string cubin;
  int block;  // threads per block
  cudaLibrary_t lib = nullptr;
  cudaKernel_t fn = nullptr;
};

struct Kernel {
  std::vector<Variant> variants;
  int chosen = -1;  // (-1: not chosen yet; one variant: 0)
  std::vector<int> zero;  // outputs to zero before a launch (sums over the samples)
  int next = 0;  // a choice over long calls: the variant of the next call (0: none started)
  std::vector<float> per;  // its milliseconds per sample so far
  int64_t runs = 0, work = 0;  // launches of a choice and their samples (tests)
};

std::mutex mu;
std::unordered_map<uint64_t, Kernel> registry;

cudaError_t load(Variant& v) {
  if (v.fn) return cudaSuccess;
  cudaError_t e = cudaLibraryLoadData(&v.lib, v.cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  return e == cudaSuccess ? cudaLibraryGetKernel(&v.fn, v.lib, "kernelax_k") : e;
}

ffi::Error Call(cudaStream_t s, ffi::RemainingArgs args, ffi::RemainingRets rets, int64_t key, int64_t n) {
  std::lock_guard<std::mutex> lock(mu);  // (also serializes a first call's timing)
  auto it = registry.find((uint64_t)key);
  if (it == registry.end()) return ffi::Error::Internal("kernelax: GPU kernel not built in this process (persistent compilation cache?)");
  Kernel& k = it->second;
  for (auto& v : k.variants)
    if (cudaError_t e = load(v); e != cudaSuccess)
      return ffi::Error::Internal(std::string("kernelax: loading a GPU kernel failed: ") + cudaGetErrorString(e) +
                                  " (built for another GPU? KERNELAX_CUDA_ARCH)");
  std::vector<void*> ptr;
  std::vector<size_t> bytes;
  for (size_t i = 0; i < args.size(); ++i) {
    auto b = args.get<ffi::AnyBuffer>(i);
    if (b.has_error()) return b.error();
    ptr.push_back(b->untyped_data());
  }
  const size_t nin = ptr.size();
  for (size_t i = 0; i < rets.size(); ++i) {
    auto b = rets.get<ffi::AnyBuffer>(i);
    if (b.has_error()) return b.error();
    ptr.push_back((*b)->untyped_data());
    bytes.push_back((*b)->size_bytes());
  }
  long long nn = n;
  std::vector<void*> params{&nn};
  for (auto& p : ptr) params.push_back(&p);
  auto launch = [&](int v) -> cudaError_t {
    const int b = k.variants[v].block;
    for (int j : k.zero) cudaMemsetAsync(ptr[nin + j], 0, bytes[j], s);
    return n > 0 ? cudaLaunchKernel((const void*)k.variants[v].fn, dim3((unsigned)((n + b - 1) / b)), dim3(b), params.data(), 0, s)
                 : cudaSuccess;
  };
  const int V = (int)k.variants.size();
  if (k.chosen < 0 && n >= TUNE_MIN) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    auto timed = [&](int v) -> float {  // milliseconds of variant v on all samples (1e30: it does not launch, e.g. for its
      ++k.runs, k.work += n;            // resources: never chosen; its error, not a sticky one, cleared)
      cudaEventRecord(a, s);
      const cudaError_t le = launch(v);
      cudaEventRecord(b, s);
      cudaEventSynchronize(b);
      float ms = 1e30f;
      if (le != cudaSuccess) cudaGetLastError();
      else cudaEventElapsedTime(&ms, a, b);
      return ms;
    };
    k.per.resize(V, 1e30f);
    bool done = false;  // (this call's outputs written)
    if (k.next > 0) {  // a choice over long calls: the next variant on this call (variant 0 again last)
      const int v = k.next < V ? k.next : 0;
      k.per[v] = timed(v);
      done = k.per[v] < 1e30f;
      if (++k.next > V) {
        int c = 0;
        for (int u = 1; u < V; ++u) c = k.per[u] < k.per[c] ? u : c;
        k.chosen = c;
      }
    } else {
      const float t0 = timed(0);  // (the outputs, unless it is short and another variant wins)
      done = t0 < 1e30f;
      if (t0 * 1e-3 > tune_long) {
        k.per[0] = t0, k.next = 1;
      } else {  // a short call: each variant twice (the first may load it; variant 0's: the run above), the fastest kept
        float best = 1e30f;
        for (int rep = 0; rep < 2; ++rep)
          for (int v = rep == 0; v < V; ++v) {
            const float ms = timed(v);
            if (rep == 1 && ms < best) best = ms, k.chosen = v;
          }
        done = false;
      }
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    if (done) {
      const cudaError_t e = cudaGetLastError();
      return e == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(std::string("kernelax: ") + cudaGetErrorString(e));
    }
    if (k.chosen >= 0) ++k.runs, k.work += n;
  }
  launch(k.chosen < 0 ? 0 : k.chosen);  // (the outputs: from the chosen variant, last)
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(std::string("kernelax: ") + cudaGetErrorString(e));
}

}  // namespace

// A kernel's variant (cubin, its threads per block); the first registration of a key sets its zeroed outputs.
// chosen >= 0: the variant to use (loaded from the index), no timing.
extern "C" void kernelax_gpu_register(uint64_t key, const char* cubin, int64_t size, int block, int nzero, const int* zero, int chosen) {
  std::lock_guard<std::mutex> lock(mu);
  Kernel& k = registry[key];
  if (k.variants.empty()) k.zero.assign(zero, zero + nzero);
  k.variants.push_back(Variant{std::string(cubin, (size_t)size), block});
  k.chosen = chosen >= 0 ? chosen : k.variants.size() == 1 ? 0 : -1;  // (a second variant: to be timed)
}

// Free memory of the current device (bytes), its multiprocessors and their resident threads at most: the driver
// reserves a kernel's local memory (bytes per thread) for every thread that can be resident (gpu.py, _fitting).
extern "C" int kernelax_gpu_memory(unsigned long long* free_bytes, int* sms, int* threads) {
  size_t f = 0, t = 0;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaMemGetInfo(&f, &t) != cudaSuccess ||
      cudaDeviceGetAttribute(sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
      cudaDeviceGetAttribute(threads, cudaDevAttrMaxThreadsPerMultiProcessor, dev) != cudaSuccess)
    return 1;
  *free_bytes = f;
  return 0;
}

// The chosen variant (-1: not timed yet).
extern "C" int kernelax_gpu_choice(uint64_t key) {
  std::lock_guard<std::mutex> lock(mu);
  auto it = registry.find(key);
  return it == registry.end() ? -1 : it->second.chosen;
}

// The seconds above which a first call is not repeated for the choice (gpu.TUNE_LONG).
extern "C" void kernelax_gpu_tuning(double long_seconds) { tune_long = long_seconds; }

// Launches of a choice of key's variant and their samples (tests).
extern "C" long long kernelax_gpu_work(uint64_t key, long long* runs) {
  std::lock_guard<std::mutex> lock(mu);
  auto it = registry.find(key);
  if (it == registry.end()) return -1;
  *runs = it->second.runs;
  return it->second.work;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kernelax_gpu_call, Call,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().RemainingRets()
                                  .Attr<int64_t>("key").Attr<int64_t>("n"));
