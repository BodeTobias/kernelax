# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Efficiency of the kernels of benchmarks/cases.py against the machine and against JAX.

    python benchmarks/run.py [case ...] [--threads N]

Machine: FMA peak (hand-written C, 16 independent chains per thread) and triad bandwidth (the better of plain and
non-temporal stores, measured per case for the case's amount of data, so that the bound sees the same caches).
Cost model (calibrate): every vector instruction one slot of the FMA ports (a mul whose only use is an add/sub
fuses with it into one FMA), special operations (recip, sqrt, exp, ...) at their measured throughput as the
generator emits them, a table lookup as measured.
Per case: the dynamic operation count per sample (loop bodies times the mean over SIMD blocks of the
largest iteration count in the block, as the masked loops run), from which
    compute bound = max(sum of count x cost on the FMA ports, on the divider),   memory bound = bytes in + out at that bandwidth,
and the efficiency = max(bounds) / kernel time (outputs preallocated and warm).
"""

import argparse
import ctypes
import functools
import os
import sys
import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cases import CASES, example  # noqa: E402

from kernelax import dag, emit, vec  # noqa: E402
from kernelax.build import compile_c, simd_width  # noqa: E402
from kernelax.reference import evaluate  # noqa: E402
from kernelax.vmap import _aligned, cvmap  # noqa: E402

ARITH = {"add", "sub", "mul"}  # flops (an FMA counts 2)
SKIP = {"in", "carry", "carrymask", "loopout", "loopmask"}
DIV = {"recip", "sqrt"}  # on the divider, in parallel to the FMA ports
_MACHINE = r"""#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>
#define STREAM(p, v) {STREAM}(p, ({REG})(v))
#define W {W}
typedef double vd __attribute__((vector_size(8 * W)));
/* FMA peak: n rounds of 16 independent FMAs per thread */
void kernel(int nt, const int64_t* n, double* out, void** unused) {{
  #pragma omp parallel num_threads(nt)
  {{
    vd acc[16];
    const vd m = (vd){{}} + 0.9999999, c = (vd){{}} + 1e-7;
    for (int i = 0; i < 16; ++i) acc[i] = (vd){{}} + (2.0 + i);  /* (not 1.0: the fixed point of acc m + c, which gcc folds) */
    for (int64_t it = 0; it < *n; ++it) {{
      #pragma GCC unroll 16
      for (int i = 0; i < 16; ++i) acc[i] = acc[i] * m + c;
    }}
    double s = 0.0;
    for (int i = 0; i < 16; ++i) for (int l = 0; l < W; ++l) s += acc[i][l];
    out[omp_get_thread_num()] = s;
  }}
}}
/* a = b + 3 c (a aligned, n a multiple of W), with plain or non-temporal stores into a */
void triad(int nt, int64_t n, double* a, const double* b, const double* c, int64_t stream) {{
  #pragma omp parallel num_threads(nt)
  {{
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < n; i += W) {{
      vd x, y;
      memcpy(&x, b + i, sizeof x);
      memcpy(&y, c + i, sizeof y);
      if (stream) STREAM(a + i, x + 3.0 * y);
      else *(vd*)(a + i) = x + 3.0 * y;
    }}
    _mm_sfence();
  }}
}}
"""


def best(fn, rep=5):
    ts = []
    for _ in range(rep):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t)
    return min(ts)


@functools.cache
def _machine_lib():
    W = simd_width()
    lib = compile_c(_MACHINE.format(W=W, STREAM="_mm512_stream_pd" if W == 8 else "_mm256_stream_pd",
                                    REG="__m512d" if W == 8 else "__m256d"))[0]
    lib.kernel.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 3
    lib.triad.argtypes = [ctypes.c_int, ctypes.c_int64] + [ctypes.c_void_p] * 3 + [ctypes.c_int64]
    return lib


def machine(nt):
    """FMA peak in flop/s."""
    W, lib = simd_width(), _machine_lib()
    n, out = np.array([20_000_000], np.int64), np.zeros(nt)
    t = best(lambda: lib.kernel(nt, n.ctypes.data, out.ctypes.data, None), 3)
    return nt * n[0] * 16 * W * 2 / t


def bandwidth(nt, nbytes):
    """Triad bandwidth in byte/s for nbytes moved, the better of plain and non-temporal stores (useful bytes:
    3 x 8 per entry, as STREAM counts): the same amount of data as a case, so that the bound sees the same
    caches (a large L3 holds part of it)."""
    n = max(nbytes // 24 // 64 * 64, 64)
    a, b, c = (_aligned((n,)) for _ in range(3))
    b[:], c[:] = 1.0, 2.0
    lib = _machine_lib()
    return max(24 * n / best(lambda: lib.triad(nt, n, a.ctypes.data, b.ctypes.data, c.ctypes.data, s)) for s in (0, 1))


def body(loop):
    """Nodes of a loop body: those depending on the loop's carries (a nested loop node included)."""
    dep, out = {p.id for p in loop.args[0][0]}, []
    for n in dag.schedule(dag.loop_roots(loop)):
        if n.id not in dep and any(isinstance(a, dag.Node) and a.id in dep for a in n.args):
            dep.add(n.id)
            out.append(n)
    return out


def dynamic_ops(call, case, args, W, n=4096):
    """Operations per sample as the kernel runs them (static count, dynamic histogram); loops run as long as
    their slowest lane in a block of K W samples (W: the block's lanes)."""
    vals = [np.asarray(a)[:n] if ax == 0 else np.broadcast_to(a, (n,) + np.shape(a))
            for a, ax in zip(args, case.in_axes)]
    return kernel_ops(call.kernel.outs, vals, W)


def kernel_ops(outs, vals, W):
    """dynamic_ops of a kernel's outputs (Kernel.outs, its DAG) on probe values of its inputs (vals[k]: (n, *leaf),
    the kernel's input leaves in its order, as it reads them)."""
    n = len(vals[0])
    roots = [x for o in outs for x in np.asarray(o, dtype=object).flat if isinstance(x, dag.Node)]
    iters = {}
    with np.errstate(all="ignore"):
        evaluate(roots, vals, iters)
    blockmax = {k: v[:n // W * W].reshape(-1, W).max(1).mean() for k, v in iters.items()}
    hist, static, seen = Counter(), Counter(), []

    def walk(nodes, mult):
        for x in nodes:
            if x.op == "while":
                walk(body(x), blockmax.get(x.id, 0.0))  # absolute: already summed over the enclosing iterations (none:
                # a nested loop whose enclosing one never ran)
            elif x.op not in SKIP:
                hist[x.op] += mult
                static[x.op] += 1
                seen.append((x, mult))
    walk(dag.schedule(roots), 1.0)
    # FMAs: an add/sub that has a mul of a single use among its operands takes it in (one instruction for both)
    uses = Counter(a.id for x, _ in seen for a in x.args if isinstance(a, dag.Node))
    uses.update(r.id for r in roots)
    hist["fused"] = sum(m for x, m in seen if x.op in ("add", "sub") and any(
        isinstance(a, dag.Node) and a.op == "mul" and uses[a.id] == 1 for a in x.args))
    return sum(static.values()), hist


def calibrate(nt, peak, reps=1000):
    """Seconds per (sample x op) at nt threads, all threads busy: add/sub/mul one flop at the FMA peak, other
    single instructions (neg, compares, select, ...) one FMA slot (2 flops); the special ops (recip, sqrt, exp,
    libmvec functions, pow, atan2) and the table lookup measured as the generator emits them, on 64 independent
    vectors in L1 (throughput, not latency), on one thread (best of 15 short runs; with all threads the
    VM's measurements jump by 2x between runs), scaled to nt threads as the FMA peak scales."""
    W = simd_width()
    ops = ["recip", "sqrt", *vec.MVEC, "pow", "atan2"]
    x = dag.mk("in", 0, 0)
    src = [emit.prelude(W, 8 * W, 8)]
    src.append("void kernel(int nt, const void* a, const void* b, void** p) {}")
    src.append(f"static const double TABX[64] = {{{', '.join(str(float(i)) for i in range(64))}}};")
    for op in ops + ["table"]:  # table: a lookup in a constant table (emit: lookup, a vector gather)
        expr = "lookup_d(TABX, x * 63.0, 64)" if op == "table" else \
            emit._c_expr(dag.mk(op, *([x, x] if op in ("pow", "atan2") else [x])), lambda n: "x", 8 * W)
        src.append(f"""double run_{op}(int nt, int64_t reps) {{
  double s = 0.0;
  #pragma omp parallel num_threads(nt) reduction(+:s)
  {{
    vd v[64], y[64];
    for (int i = 0; i < 64; ++i) v[i] = bc_d(0.5 + i / 64.0);
    for (int64_t r = 0; r < reps; ++r) {{
      for (int i = 0; i < 64; ++i) {{ const vd x = v[i]; y[i] = {expr}; }}
      __asm__ volatile("" : : "r"(v), "r"(y) : "memory");  /* no hoisting out of the reps loop */
    }}
    for (int i = 0; i < 64; ++i) s += y[i][0];
  }}
  return s;
}}""")
    lib = compile_c("\n".join(src) + "\n")[0]
    cost = dict.fromkeys(ARITH, 1 / peak)
    cost["other"] = 2 / peak
    for op in ops + ["table"]:
        fn = getattr(lib, f"run_{op}")
        fn.restype, fn.argtypes = ctypes.c_double, [ctypes.c_int, ctypes.c_int64]
        cost[op] = best(lambda: fn(1, reps), 15) / (nt * reps * 64 * W)
    cost["iwrap"] = 0.0  # integer ops: one slot ("other"), the wraps mostly merge into neighbouring masks and shifts
    cost["threefry"] = 81 * cost["other"]  # (dag.threefry: threefry_v's vector instructions in 32-bit lanes, 20 rounds
    # of add, rotate and xor, 5 key injections, the conversions in and out)
    return cost


def work_per_sample(hist, cost):
    """Seconds per sample at the compute bound: every vector instruction one slot of the FMA ports (an FMA, a mul
    fused into an add/sub, one for both), special ops at their cost; the divider works in parallel."""
    slots = (sum(hist[o] for o in ARITH) - hist["fused"]) * cost["other"]
    use = lambda div: sum(c * cost.get(o, cost["other"]) for o, c in hist.items()
                          if (o in DIV) == div and o not in ARITH and o != "fused")
    return max(use(False) + slots, use(True))


def run(case, nt, peak, cost):
    W, rng = simd_width(), np.random.default_rng(0)
    probe = case.make(rng, 4096)
    t0 = time.perf_counter()
    call = cvmap(case.f, example(case, probe), case.in_axes, n_threads=nt)
    t_build = time.perf_counter() - t0
    N = int(np.clip(2e9 / max(sum(dynamic_ops(call, case, probe, W)[1].values()), 1), 2 ** 14, 2 ** 20)) // (W * nt) * W * nt
    args = case.make(rng, N)
    nbytes = 8 * (sum(np.size(a) for a, ax in zip(args, case.in_axes) if ax == 0)
                  + N * sum(np.size(o) for o in call.kernel.outs))
    jf = jax.jit(jax.vmap(case.f, in_axes=case.in_axes))
    jargs = [jnp.asarray(a) for a in args]
    jf(*jargs)
    t_jax = best(lambda: jf(*jargs))
    t_call = best(lambda: call(*args))  # the first call chooses K
    static, hist = dynamic_ops(call, case, probe, W * call.kernel.K())  # loops: as long as the slowest of K W lanes
    flops, work = sum(c for o, c in hist.items() if o in ARITH), work_per_sample(hist, cost)
    out = call.empty(N)
    t_warm = best(lambda: call(*args, out=out))
    bw = bandwidth(nt, nbytes)
    bound = max(N * work, nbytes / bw)
    special = sum(c for o, c in hist.items() if o not in ARITH and o != "fused")
    return dict(case=case.name, K=call.kernel.K(), N=N, static=static, flops=flops, special=special, kib=len(call.kernel.sources[0]) // 1024,
                build=t_build, gcc=call.kernel.compile_time, jax=t_jax * 1e3, call=t_call * 1e3, warm=t_warm * 1e3,
                speedup=t_jax / t_call, gflops=N * flops / t_warm / 1e9, peak=N * flops / t_warm / peak * 100,
                cbound=N * work * 1e3, mbound=nbytes / bw * 1e3, eff=bound / t_warm * 100)


COLS = [("case", "{}"), ("K", "{}"), ("N", "{}"), ("static", "{}"), ("flops", "{:.0f}"), ("special", "{:.0f}"), ("kib", "{}"),
        ("build", "{:.2f}"), ("gcc", "{:.2f}"), ("jax", "{:.2f}"), ("call", "{:.2f}"), ("warm", "{:.2f}"),
        ("speedup", "{:.1f}"), ("gflops", "{:.1f}"), ("peak", "{:.0f}"), ("cbound", "{:.2f}"), ("mbound", "{:.2f}"),
        ("eff", "{:.0f}")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)))
    a = ap.parse_args()
    nt = a.threads
    peak, bw = machine(nt), bandwidth(nt, 2 ** 30)
    cost = calibrate(nt, peak)
    print(f"threads {nt}, W {simd_width()}: FMA peak {peak / 1e9:.1f} GFLOP/s, triad (1 GB) {bw / 1e9:.1f} GB/s")
    print("cost per op in flops at the FMA peak: " + ", ".join(f"{o} {c * peak:.0f}" for o, c in cost.items()
                                                                if o not in ARITH))
    print("\n| " + " | ".join(c for c, _ in COLS) + " |\n|" + "---|" * len(COLS))
    for case in CASES:
        if case.xfail or (a.cases and case.name not in a.cases):
            continue
        r = run(case, nt, peak, cost)
        print("| " + " | ".join(fmt.format(r[c]) for c, fmt in COLS) + " |", flush=True)
