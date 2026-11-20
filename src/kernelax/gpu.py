# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""GPU backend: a traced per-sample function (frontend.trace) as one CUDA kernel over a batch, one thread per sample.

- emit: the scalar DAG (as for the C kernels: CSE, constant folding, small linear algebra unrolled) as CUDA C. While
  loops are loops per thread (no masks: a thread's loop ends with its sample). A sample's input rows are read where
  used (the rows of a warp's samples go through L1 once), entries the same for all samples are one load per warp.
  Outputs per sample go through shared memory and are written coalesced; sums over the samples (reduce) are added per
  warp, then atomically; sums at sample-dependent positions (frontend._Scatter) atomically. Integers that kernelax
  carries as exact doubles become long long where the DAG proves them integral (_ints): on many GPUs float64 runs at a
  fraction of float32. __launch_bounds__(B, 1): with B alone ptxas aims at full occupancy and spills large kernels.
  Dense layers (frontend._dense: a uniform matrix times per-sample values) unrolled where the code fits the
  registers (LIVE), else loops over the matrix rows, products of one matrix in one loop, only the columns used.
- build: nvcc -cubin (device code only, ~0.3 s), cached on disk by source (KERNELAX_CACHE_DIR/gpu); a second variant
  (ptxas -O1) where -O3 spills, timed against it at the first call; a kernel in kernelax's index on disk (cpu.py) loads
  in a later process without trace, emit or nvcc.
- run: gpu_runtime.cu's XLA FFI handler kernelax_gpu_call on XLA's stream (built once), see primitive.py.
"""

import atexit
import collections
import ctypes
import functools
import hashlib
import os
import pathlib
import re
import shutil
import signal
import subprocess
import threading
import time

import jax
import numpy as np

from kernelax.build import _COMPILING, CACHE, env, intact, locked, no_gc
from kernelax import dag
from kernelax.dag import Node, loop_roots, schedule
from kernelax.vec import nan_bits
from kernelax import frontend
from kernelax.frontend import _flat, trace

NVCC = os.environ.get("KERNELAX_NVCC") or "nvcc"
ARCH = os.environ.get("KERNELAX_CUDA_ARCH")  # (default: the first GPU's, sm_XY)


@functools.cache
def arch():
    return ARCH or "sm_" + jax.devices("gpu")[0].compute_capability.replace(".", "")


@functools.cache
def nvcc_version():  # (in the cache keys of the cubins and the runtime: a CUDA 12 process does not load CUDA 13's)
    return subprocess.run([NVCC, "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
SMEM = 48 * 1024  # static shared memory of a block (bytes)
BLOCK = 128  # threads per block at most (fewer where shared memory needs it)
ROUND = {"floor", "ceil", "trunc", "rint"}
_CT = {"f": "float", "d": "double", "i": "int", "u": "unsigned", "h": "__half", "b": "__nv_bfloat16"}  # elements of an array in memory, by kind (cpu._kind)
_BIN = {"add": "+", "sub": "-", "mul": "*"}
_CMP = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}
_IBIN = {"iadd": "+", "isub": "-", "imul": "*", "iand": "&", "ior": "|", "ixor": "^"}
_MATH = {"sin", "cos", "tan", "exp", "log", "tanh", "sinh", "cosh", "asin", "acos", "atan", "asinh", "acosh", "atanh", "log1p",
         "expm1", "exp2", "cbrt", "erf", "erfc", "log2", "log10", "sqrt", "floor", "ceil", "trunc", "rint"}

_PRELUDE16 = r"""#include <cuda_fp16.h>
#include <cuda_bf16.h>
// entry O of a 16-bit float input the same for all samples (read as such: no conversion pass before the kernel)
template <int O> __device__ __forceinline__ float ldu(const __half* p) { return __half2float(__ldg(p + O)); }
template <int O> __device__ __forceinline__ float ldu(const __nv_bfloat16* p) { return __bfloat162float(__ldg(p + O)); }
"""

_HELPERS = r"""// jax.random's threefry2x32 (dag.threefry): 32-bit words, packed into one 64-bit integer (x's word low)
__device__ __forceinline__ long long kx_threefry(long long a, long long b, long long c, long long d) {
  const unsigned k0 = (unsigned)a, k1 = (unsigned)b, k2 = k0 ^ k1 ^ 0x1BD11BDAu;
  unsigned x = (unsigned)c + k0, y = (unsigned)d + k1;
  x += y; y = x ^ ((y << 13) | (y >> 19)); x += y; y = x ^ ((y << 15) | (y >> 17)); x += y; y = x ^ ((y << 26) | (y >> 6)); x += y; y = x ^ ((y << 6) | (y >> 26)); x += k1; y += k2 + 1u; x += y; y = x ^ ((y << 17) | (y >> 15)); x += y; y = x ^ ((y << 29) | (y >> 3)); x += y; y = x ^ ((y << 16) | (y >> 16)); x += y; y = x ^ ((y << 24) | (y >> 8)); x += k2; y += k0 + 2u; x += y; y = x ^ ((y << 13) | (y >> 19)); x += y; y = x ^ ((y << 15) | (y >> 17)); x += y; y = x ^ ((y << 26) | (y >> 6)); x += y; y = x ^ ((y << 6) | (y >> 26)); x += k0; y += k1 + 3u; x += y; y = x ^ ((y << 17) | (y >> 15)); x += y; y = x ^ ((y << 29) | (y >> 3)); x += y; y = x ^ ((y << 16) | (y >> 16)); x += y; y = x ^ ((y << 24) | (y >> 8)); x += k1; y += k2 + 4u; x += y; y = x ^ ((y << 13) | (y >> 19)); x += y; y = x ^ ((y << 15) | (y >> 17)); x += y; y = x ^ ((y << 26) | (y >> 6)); x += y; y = x ^ ((y << 6) | (y >> 26)); x += k2; y += k0 + 5u;
  return (long long)(((unsigned long long)y << 32) | x);
}
template <class T> __device__ __forceinline__ T mx(T a, T b) { return a != a ? a : b != b ? b : (a > b ? a : b); }
template <class T> __device__ __forceinline__ T mn(T a, T b) { return a != a ? a : b != b ? b : (a < b ? a : b); }
__device__ __forceinline__ long long mx(long long a, long long b) { return a > b ? a : b; }
__device__ __forceinline__ long long mn(long long a, long long b) { return a < b ? a : b; }
template <class T> __device__ __forceinline__ T sgn(T x) { return x > 0 ? T(1) : x < 0 ? T(-1) : x; }
// an index into n entries from a value (rounded to the nearest; clamped; NaN: 0), as the C kernels' lookup
__device__ __forceinline__ long long ix(double x, long long n) { return x != x ? 0 : (long long)fmin(fmax(rint(x), 0.0), (double)(n - 1)); }
__device__ __forceinline__ long long ix(float x, long long n) { return ix((double)x, n); }
__device__ __forceinline__ long long ix(long long x, long long n) { return x < 0 ? 0 : x >= n ? n - 1 : x; }
__device__ __forceinline__ long long ix(int x, long long n) { return ix((long long)x, n); }
__device__ __forceinline__ int mx(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int mn(int a, int b) { return a < b ? a : b; }
"""

_PRELUDE = "#include <cstdint>\n" + _HELPERS + r"""// entry O of an input the same for all samples, loaded where used: volatile (not merged with other loads of it into a
// register held across the kernel, nor hoisted), the offset an immediate (no address register per entry: LLVM keeps
// those live from where it computes them, thousands in a large kernel); from L1, one broadcast per warp
template <int O> __device__ __forceinline__ float ldu(const float* p) { float r; asm volatile("ld.global.nc.f32 %0, [%1+%2];" : "=f"(r) : "l"(p), "n"(O * 4)); return r; }
template <int O> __device__ __forceinline__ double ldu(const double* p) { double r; asm volatile("ld.global.nc.f64 %0, [%1+%2];" : "=d"(r) : "l"(p), "n"(O * 8)); return r; }
template <int O> __device__ __forceinline__ int ldu(const int* p) { int r; asm volatile("ld.global.nc.s32 %0, [%1+%2];" : "=r"(r) : "l"(p), "n"(O * 4)); return r; }
template <int O> __device__ __forceinline__ unsigned ldu(const unsigned* p) { unsigned r; asm volatile("ld.global.nc.u32 %0, [%1+%2];" : "=r"(r) : "l"(p), "n"(O * 4)); return r; }
// a spill slot (_spill) at byte offset O from the thread's first slot (shared address b), likewise volatile
template <int O> __device__ __forceinline__ void spst(unsigned b, float v) { asm volatile("st.shared.f32 [%0+%1], %2;" :: "r"(b), "n"(O), "f"(v)); }
template <int O> __device__ __forceinline__ float spld(unsigned b) { float r; asm volatile("ld.shared.f32 %0, [%1+%2];" : "=f"(r) : "r"(b), "n"(O)); return r; }
__device__ __forceinline__ void kx_add(float* p, float v) { atomicAdd(p, v); }
__device__ __forceinline__ void kx_add(double* p, double v) {  // (atomicAdd of doubles: from sm_60 on)
#if __CUDA_ARCH__ >= 600
  atomicAdd(p, v);
#else
  unsigned long long *a = (unsigned long long*)p, old = *a, prev;
  do { prev = old; old = atomicCAS(a, prev, __double_as_longlong(v + __longlong_as_double(prev))); } while (old != prev);
#endif
}
template <class T> __device__ __forceinline__ T wsum(T v) {
#pragma unroll
  for (int k = 16; k > 0; k >>= 1) v += __shfl_xor_sync(0xffffffffu, v, k);
  return v;
}
"""


_CPU_PRELUDE = r"""// kernelax: one sample per call, the scalar code of the GPU emitter (gpu.emit) compiled for the CPU
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(a, b)
static inline double __drcp_rn(double x) { return 1.0 / x; }
static inline float __frcp_rn(float x) { return 1.0f / x; }
static inline long long __double_as_longlong(double x) { long long r; std::memcpy(&r, &x, 8); return r; }
static inline double __longlong_as_double(long long x) { double r; std::memcpy(&r, &x, 8); return r; }
static inline int __float_as_int(float x) { int r; std::memcpy(&r, &x, 4); return r; }
static inline float __int_as_float(int x) { float r; std::memcpy(&r, &x, 4); return r; }
#if !defined(__GLIBC__)
static inline void sincos(double x, double* s, double* c) { *s = sin(x); *c = cos(x); }
static inline void sincosf(float x, float* s, float* c) { *s = sinf(x); *c = cosf(x); }
#endif
template <int O, class T> static inline T ldu(const T* p) { return p[O]; }  // (an input the same for all samples)
static const struct { long long x; } blockIdx = {0}, threadIdx = {0};
"""
_GPU_ONLY = ("__shfl", "atomicAdd", "kx_add", "__shared__", "__syncthreads", "__ldg", "float4", "spst", "spld", "__cvta", "wsum",
             "asm ")  # (constructs of the GPU code without a CPU counterpart in the point mode: refused there)


def _lit(c, ct):
    """C of constant c as type ct."""
    if ct == "bool":
        return "true" if c else "false"
    if ct == "int":
        ct = "long long"  # (an int literal; converted where used)
    if ct == "long long" and not (isinstance(c, (int, np.integer)) or float(c).is_integer() and -2 ** 63 <= float(c) < 2 ** 63):
        ct = "double"  # (an integer compared with a fraction, or with a float beyond int64: binomial's i < finfo.max)
    if ct == "long long":
        c = int(c) % 2 ** 64
        c = c - 2 ** 64 if c >= 2 ** 63 else c
        return "(-9223372036854775807LL - 1)" if c == -2 ** 63 else f"{c}LL"
    c = float(c)
    if ct == "float":
        with np.errstate(over="ignore"):
            c = float(np.float32(c))  # (beyond float32: inf, as XLA rounds such a constant)
    if np.isnan(c):  # (constant expressions: also in a __device__ table's initializer; doubles with their bits: those
        return '__builtin_nanf("")' if ct == "float" else nan_bits(c)  # of 64-bit integers, vec.nan_bits)
    if np.isinf(c):
        return ("" if c > 0 else "-") + ("__builtin_huge_valf()" if ct == "float" else "__builtin_huge_val()")
    return f"{np.float32(c).item().hex()}f" if ct == "float" else c.hex()


def _all_nodes(roots, cols=None):
    """Every node the roots need, loop bodies included; of a dense node's operand (frontend._dense) the columns of the
    results needed (into cols: dense id -> those columns)."""
    out, seen, stack, cols = [], set(), list(roots), {} if cols is None else cols
    while stack:
        order = schedule(stack, seen)
        seen.update(n.id for n in order)
        out += order
        stack = [r for n in order if n.op == "while" for r in loop_roots(n)]
        for n in order:
            if n.op == "denseout":
                d, T = n.args[0], n.args[0].args[0][0][6]
                if n.args[1] % T not in cols.setdefault(d.id, set()):
                    cols[d.id].add(n.args[1] % T)
                    stack += [x for x in d.args[0][1][n.args[1] % T::T] if x.id not in seen]
    return out


def _operand(d, cols):  # the entries of a dense node's operand in the columns used
    T = d.args[0][0][6]
    return [x for i, x in enumerate(d.args[0][1]) if i % T in cols[d.id]]


def _reach(roots, done):
    """Ids of the nodes not in done the roots need (through dense nodes' operands too; not into loop bodies: a loop's
    outside values are its arguments)."""
    ids, stack = set(), list(roots)
    while stack:
        order = schedule(stack, set(done) | ids)
        ids.update(n.id for n in order)
        stack = [x for n in order if n.op == "dense" for x in n.args[0][1]]
    return ids


def _bounds(x):
    """(lo, hi) of a float32 value clamped by max and min with constants after NaN was replaced (frontend._ftoi), else
    None: it converts to an integer only so (others may be inf, NaN or beyond 64 bits; a user's clip keeps NaN)."""
    if not (isinstance(x, Node) and x.op in ("min", "max")):
        return None
    c, y = (x.args[0], x.args[1]) if not isinstance(x.args[0], Node) else (x.args[1], x.args[0])
    if isinstance(c, Node) or not isinstance(y, Node) or y.op != {"min": "max", "max": "min"}[x.op]:
        return None
    d = next((v for v in y.args if not isinstance(v, Node)), None)
    z = next((v for v in y.args if isinstance(v, Node)), None)  # (select(z != z, value, constant))
    nan_free = z is not None and z.op == "select" and isinstance(z.args[0], Node) and z.args[0].op == "ne" and \
        z.args[0].args[0] is z.args[0].args[1] and not isinstance(z.args[2], Node)
    lo, hi = (d, c) if x.op == "min" else (c, d)
    return None if d is None or not lo <= hi or not nan_free else (float(lo), float(hi))


_I32, _I53 = (-2.0 ** 31, 2.0 ** 31 - 1), 2.0 ** 53  # (int's range; doubles hold integers exactly within +-2^53)
_WIDEN = (None, _I32, (-2.0 ** 52, 2.0 ** 52))  # (the steps of a loop carry's range: its values', int's, +-2^52)


def _ints(roots, kin, nodes=None, E=8):
    """{id: (lo, hi)} of the double nodes ("d") that hold integers, long long in the kernel (int where E == 4 and the
    range is int's): inputs of integer dtypes, the sample index, conversions of integers and masks and of float32
    integers clamped to a range (_bounds), arithmetic, selects and rounding of integers, integral tables, loop carries
    whose initial and next values are integers. Only within +-2^53, where the C kernels' doubles hold them exactly:
    beyond (21! in a loop, a float64 integer times 1e10) a value stays a double, as there (as long long it wrapped).
    A carry's range (optimistic first) grows in steps (_WIDEN) to hold its initial and next values; at the last a
    next value within +-2^53 suffices (a counter), else it is no integer (a product). A fixpoint."""
    nodes = _all_nodes(roots) if nodes is None else nodes
    triples = [(p, o, x) for L in (n for n in nodes if n.op == "while") for p, o, x in zip(L.args[0][0], L.args[0][1], L.args[1:])
               if p.t == "d"]
    carry, level = {p.id: (0.0, 0.0) for p, _, _ in triples}, {p.id: 0 for p, _, _ in triples}
    whole = lambda c: not isinstance(c, (bool, np.bool_)) and float(c).is_integer() and abs(float(c)) <= _I53 and \
        not (c == 0 and np.signbit(c))  # (-0.0: a float, its sign kept)
    hull = lambda *rs: None if any(r is None for r in rs) else (min(r[0] for r in rs), max(r[1] for r in rs))
    tables = {}  # id of a table's values -> their range if all are integers (tables are shared by their lookups)

    def table(t):
        if id(t) not in tables:
            tables[id(t)] = (float(min(t)), float(max(t))) if all(whole(x) for x in t) else None
        return tables[id(t)]

    def wrapped(x):  # the range of an integer node's value (iwrap: its width's; f32bits: int32's; else int64's)
        if not isinstance(x, Node):
            return (float(x), float(x))
        if x.op == "iwrap":
            return (-2.0 ** (x.args[1] - 1), 2.0 ** (x.args[1] - 1) - 1) if x.args[2] else (0.0, 2.0 ** x.args[1] - 1)
        return _I32 if x.op == "f32bits" else (0.0, 1.0) if x.op == "b2i" else (-2.0 ** 63, 2.0 ** 63)

    for _ in range(64):
        R, fl = {}, set()  # (fl: float32 values that are integers, the results of rounding)
        rng = lambda a: R.get(a.id) if isinstance(a, Node) else ((float(a), float(a)) if whole(a) else None)
        for n in nodes:
            op, a = n.op, n.args
            if n.t == "s":
                if op in ROUND or op in ("max", "min", "select") and all(
                        (isinstance(x, Node) and x.id in fl) or (not isinstance(x, Node) and whole(x)) for x in (a[1:] if op == "select" else a)):
                    fl.add(n.id)
                continue
            if n.t != "d":
                continue
            r = None
            if op in ("in", "load") and kin[a[0]] in "iu":
                r = _I32 if kin[a[0]] == "i" else (0.0, 2.0 ** 32 - 1)
            elif op == "b2f":
                r = (0.0, 1.0)
            elif op == "sample":
                r = (0.0, _I32[1] if E == 4 else _I53)  # (a 4-byte program's sample index: int, fewer than 2^31 samples)
            elif op == "itof" and not a[1]:
                r = wrapped(a[0])
            elif op == "tof64" and isinstance(a[0], Node) and a[0].id in fl:
                r = _bounds(a[0])
            elif op == "carry":
                r = carry.get(n.id)
            elif op == "loopout":
                r = carry.get(a[0].args[0][0][a[1]].id)
            elif op in ("add", "sub", "mul") and rng(a[0]) and rng(a[1]):
                (p, q), (u, v) = rng(a[0]), rng(a[1])
                r = (p + u, q + v) if op == "add" else (p - v, q - u) if op == "sub" else (
                    min(p * u, p * v, q * u, q * v), max(p * u, p * v, q * u, q * v))
            elif op == "neg" and rng(a[0]):
                r = (-rng(a[0])[1], -rng(a[0])[0])
            elif op == "abs" and rng(a[0]):
                p, q = rng(a[0])
                r = (0.0 if p <= 0 <= q else min(abs(p), abs(q)), max(abs(p), abs(q)))
            elif op in ("max", "min") and rng(a[0]) and rng(a[1]):
                r = tuple((max if op == "max" else min)(rng(a[0])[k], rng(a[1])[k]) for k in (0, 1))
            elif op == "select":
                r = hull(rng(a[1]), rng(a[2]))
            elif op in ROUND:
                r = rng(a[0])
            elif op == "table":
                r = table(a[0])
            elif op == "arr":
                r = hull(*map(rng, a))
            elif op == "pick" and isinstance(a[0], Node):
                r = rng(a[0])
            if r is not None and -_I53 <= r[0] and r[1] <= _I53:
                R[n.id] = r
        grown = False
        for p, o, x in triples:
            have, want = carry[p.id], hull(rng(o), rng(x))
            if have is None or want is not None and (have[0] <= want[0] and want[1] <= have[1] or level[p.id] == len(_WIDEN)):
                continue  # (dropped, holding, or at the last step: within +-2^53, as any integer node)
            grown, level[p.id] = True, level[p.id] + 1
            carry[p.id] = None if want is None or level[p.id] > len(_WIDEN) else \
                hull(want, have) if _WIDEN[level[p.id] - 1] is None else hull(want, _WIDEN[level[p.id] - 1])
        if not grown:
            return R
    raise AssertionError("kernelax GPU: the integers' ranges did not settle")


def _exact(roots, ints, nodes=None):
    """Ids of the double nodes ("d") that are not integers but must stay doubles in a float32 kernel: those computed
    from integers alone (integer division's quotient estimates, frontend._idivrem; the bits of 64-bit integers,
    bitsf64: keys in a loop's carries), and loop carries whose initial and next values are such, one of them neither
    an integer nor a constant (a fixpoint, as _ints; integral carries stay float32: small integers, a result code). The
    others are float32 values the DAG types as doubles (e.g. a select between constants): computed in float32, as XLA
    does."""
    nodes = _all_nodes(roots) if nodes is None else nodes
    loops = [n for n in nodes if n.op == "while"]
    carries = {p.id for L in loops for p in L.args[0][0] if p.t == "d" and p.id not in ints}
    while True:
        exact = set(carries)  # (assumed exact first)
        ok = lambda a: not isinstance(a, Node) or a.t in ("i", "m") or a.id in ints or a.id in exact  # (masks: a select's
        # condition, i // 7's correction)
        for n in nodes:
            if n.t != "d" or n.id in ints or n.op in ("table", "carry"):  # (tables: of the program's float32 constants)
                continue
            if n.op == "loopout":
                if n.args[0].args[0][0][n.args[1]].id in carries:
                    exact.add(n.id)
                continue
            args = [a for a in n.args if isinstance(a, Node)]
            if args and all(ok(a) for a in args) and any(a.t != "m" for a in args):  # (a mask alone: a select of
                exact.add(n.id)  # constants, float32)
        bits = lambda a: isinstance(a, Node) and a.id in exact and a.op != "carry"  # (exact, not integral)
        need = {p.id for L in loops for p, o, x in zip(L.args[0][0], L.args[0][1], L.args[1:])
                if ok(o) and ok(x) and (bits(o) or bits(x))}  # (a placeholder that sibling loops share, frontend._loop:
        drop = carries - need  # exact where one of them needs it; the others hold their float32 values in it, exactly)
        if not drop:
            return exact
        carries -= drop


ARRAY_MIN = 16  # picks among more options than this read an array per thread (local memory), else a chain of selects
# (8 -> 16: jumanji's Tetris 60.1 -> 11.3 ms, Snake 7.5 -> 5.4, xland-minigrid 3.35 -> 2.52, pgx 2048 +3 %)
LIVE = 160  # values live in registers at most (top level), beyond: _spill
_DEF = re.compile(r"^  const (float|int|bool|long long|double) (t\d+) = ")
_NAME = re.compile(r"\b(?:t\d+|c\d+_\d+)\b")


def _spill(lines, limit):
    """(lines, slots): straight-line code at the top level with values spilled to shared memory (SP[slot * B + thread])
    where more than `limit` would be live: the one used last of the live 32-bit values, reloaded before its next use.
    (ptxas spills such code to local memory, which misses L1 at 255 registers per thread; e.g. a PINN's MLP: 4 vectors
    of 32 per layer in and out.) Loops and blocks count as one statement; their values stay in registers."""
    if limit is None:
        return lines, 0
    stmts, nxt = _parse(lines)
    live, ptr = {}, {}  # (spillable values in registers -> type; position in their use lists)
    if max_live(stmts, nxt) <= limit:
        return lines, 0
    out, spilled, cur, free, slots = [], {}, {}, [], 0
    for k, (body, defs, uses) in enumerate(stmts):
        for u in sorted(uses & spilled.keys()):  # reloads
            slot, t = spilled.pop(u)
            free.append(slot)
            cur[u] = f"{u}_r{k}"
            v = f"spld<{slot} * BLK * 4>(spb)"
            out.append(f"  const {t} {cur[u]} = {v if t == 'float' else f'__float_as_int({v})' if t == 'int' else f'__float_as_int({v}) != 0'};")
            live[u] = t
        if any(u in cur for u in uses):
            pat = re.compile(r"\b(" + "|".join(re.escape(u) for u in uses if u in cur) + r")\b")
            body = [pat.sub(lambda mm: cur[mm.group(1)], ln) for ln in body]
        out += body
        for u in uses:
            while ptr.get(u, 0) < len(nxt[u]) and nxt[u][ptr.get(u, 0)] <= k:
                ptr[u] = ptr.get(u, 0) + 1
            if ptr[u] >= len(nxt[u]):
                live.pop(u, None)
        for d, t in defs.items():
            if d in nxt:
                live[d] = t
        while len(live) > limit:
            victim = max(live, key=lambda v: nxt[v][ptr.get(v, 0)] if ptr.get(v, 0) < len(nxt[v]) else 1 << 60)
            t = live.pop(victim)
            slot = free.pop() if free else slots
            slots = max(slots, slot + 1)
            r = cur.get(victim, victim)
            out.append(f"  spst<{slot} * BLK * 4>(spb, {r if t == 'float' else f'__int_as_float((int){r})'});")
            spilled[victim] = (slot, t)
    return out, slots


def _parse(lines):
    """(statements: (lines, defs (name -> type), uses), next uses per name: statement indices) of straight-line code at
    the top level; loops and blocks count as one statement."""
    stmts, i = [], 0
    while i < len(lines):
        j = i + 1
        if lines[i].endswith("{"):  # a block: to its closing line at this indentation
            ind = len(lines[i]) - len(lines[i].lstrip())
            while not (lines[j].startswith(" " * ind + "}") and len(lines[j]) - len(lines[j].lstrip()) == ind):
                j += 1
            j += 1
        body = lines[i:j]
        m = _DEF.match(body[0]) if len(body) == 1 else None
        inner = {d for ln in body for d in re.findall(r"(?:float|int|bool|long long|double) (t\d+|[cu]\d+_\d+)", ln)}
        uses = {u for ln in body for u in _NAME.findall(ln)} - inner
        stmts.append((body, {m.group(2): m.group(1)} if m and m.group(1) in ("float", "int", "bool") else {}, uses))
        i = j
    nxt = {}
    for k in range(len(stmts) - 1, -1, -1):  # next uses, per statement: name -> its uses' statement indices
        for u in stmts[k][2]:
            nxt.setdefault(u, []).append(k)
    for v in nxt.values():
        v.reverse()
    return stmts, nxt


def max_live(stmts, nxt):
    """The most values (spillable ones) live at once in straight-line code."""
    last = {u: v[-1] for u, v in nxt.items()}
    live, most = set(), 0
    for k, (_, defs, uses) in enumerate(stmts):
        live -= {u for u in uses if last[u] == k}
        live |= {d for d in defs if d in nxt}
        most = max(most, len(live))
    return most


def emit(inputs, outputs, batched, reduce, mem, loops=None, nodes=None, smem=None, live=None, cpu=False):
    """(CUDA source of `kernelax_k(n, inputs..., outputs...)`, threads per block, the outputs the runtime zeroes). loops:
    dense layers (frontend._dense) as loops or unrolled; None: unrolled where the straight-line code keeps at most LIVE
    values live. smem, live: SMEM and LIVE (default: the module's). cpu: C++ for the CPU instead (the export's point
    mode, smem 0: no staging, no spills): a static `kernelax_k` after _CPU_PRELUDE; code without a CPU counterpart
    (reduce outputs, 16-bit floats) raises NotImplementedError."""
    smem, live = SMEM if smem is None else smem, LIVE if live is None else live
    E, kin, kout = mem
    reduce = reduce or [False] * len(outputs)
    scat = [hasattr(a, "entries") for a in outputs]
    roots = [x for o in outputs for x in _flat(o) if isinstance(x, Node)]
    plain = _all_nodes(roots) if nodes is None else nodes  # (every node the outputs need: nodes, given or found)
    ints = _ints(roots, kin, plain, E)
    exact = _exact(roots, ints, plain) if E == 4 and not dag._F64 else None  # (float32 programs: doubles only where
    # exactness needs them; with float64 values, x64's intermediates of float32 arrays: all of them)
    ET = "float" if E == 4 else "double"
    narrow = lambda n: E == 4 and _I32[0] <= ints[n.id][0] and ints[n.id][1] <= _I32[1]  # (4-byte programs: int, where
    # the range is int's)
    ctype = lambda n: {"s": "float", "m": "bool", "i": "long long"}.get(n.t) or (
        ("int" if narrow(n) else "long long") if n.id in ints else "double" if exact is None or n.id in exact else "float")
    atype = lambda a: "int" if a.op == "iarr" and all(  # an array's elements: int for integer options of int's range
        isinstance(o, Node) and o.op == "iwrap" and (o.args[1] < 32 or o.args[1] == 32 and o.args[2]) or
        not isinstance(o, Node) and -2 ** 31 <= o < 2 ** 31 for o in a.args) else ctype(a)  # (wrapped to 32 bits signed
    # or fewer: half the local memory of long long; jumanji's Tetris 5.5 of 7.3 KB of stack per thread)
    sizes = [int(np.size(a)) for a in inputs]
    rows = [int(np.size(a)) if not (s or r) else 0 for a, s, r in zip(outputs, scat, reduce)]  # outputs per sample
    staged = [r > 1 and 32 * (r | 1) * E <= smem for r in rows]  # (rows too long for 32 threads: written directly)
    stage = max((r | 1 for r, st in zip(rows, staged) if st), default=0)  # (odd strides: no bank conflicts)
    names, tables, mode = {}, {}, {"loops": False}  # (loops: dense layers as loops, else unrolled)
    trig = {}  # (sin or cos, argument id) -> node: both of one argument by one sincos
    cols = {}  # dense id -> the columns of its results used
    nodes = _all_nodes(roots, cols)
    rounded = dag.exact_products(nodes)  # (products error-free transformations need rounded: not fused into their uses)
    dense = {}  # (matrix: input, offset, steps, rows, columns) -> its dense nodes (frontend._dense)
    for n in nodes:
        if n.op == "dense":
            dense.setdefault(n.args[0][0][:6], []).append(n)
    arrays = {n.id: n for n in nodes if n.op in ("arr", "iarr") and len(n.args) > ARRAY_MIN and not any(  # options of
        isinstance(o, Node) and o.op in ("carry", "carrymask") for o in n.args)}  # picks: an array (a loop's carries:
    # not emitted where defined, so not stored)
    slots = {}  # per thread (local memory; else a chain of selects per read): option id -> (array id, position)
    for a in arrays.values():
        for j, o in enumerate(a.args):
            if isinstance(o, Node):
                slots.setdefault(o.id, []).append((a.id, j))
    for n in nodes:
        if n.op in ("sin", "cos") and isinstance(n.args[0], Node):
            trig[(n.op, n.args[0].id)] = n
    guards, opened, refused, inside = dag.guards(roots), set(), set(), []  # (work only one side of selects needs: in a
    # block run where the thread takes that side, dag.guards; refused: (work, scope) where not, inside: the work of the
    # open blocks)

    def cv(x, ct):  # C of operand x (node or constant) as type ct
        return names[x.id] if isinstance(x, Node) else _lit(x, ct)

    def expr(n):
        op, a, ct = n.op, n.args, ctype(n)
        oct = next((ctype(x) for x in a if isinstance(x, Node)), ct)  # (the operands' type: compares, conversions)
        f = "f" if ct == "float" else ""
        if op in _BIN:  # (integers: the operands in the result's type, int + int computes in int)
            c = f"({ct})" if ct in ("int", "long long") else ""
            if n.id in rounded and ct in ("float", "double"):  # (nvcc fuses no __*mul_rn into an fma; the export's C++:
                return (f"__builtin_assoc_barrier({cv(a[0], ct)} * {cv(a[1], ct)})" if cpu else  # gcc's barrier)
                        f"__{ct[0]}mul_rn({cv(a[0], ct)}, {cv(a[1], ct)})")
            return f"{c}{cv(a[0], ct)} {_BIN[op]} {c}{cv(a[1], ct)}"
        if op == "neg":
            return f"-({ct}){cv(a[0], ct)}" if ct in ("int", "long long") else f"-{cv(a[0], ct)}"
        if op == "recip":
            return f"__frcp_rn({cv(a[0], ct)})" if ct == "float" else f"__drcp_rn({cv(a[0], ct)})"
        if op == "rsqrt":
            return f"__frcp_rn(sqrtf({cv(a[0], ct)}))" if ct == "float" else f"__drcp_rn(sqrt({cv(a[0], ct)}))"
        if op in _MATH and ct in ("int", "long long"):  # (an integer result: rounding, of an integer operand exact)
            if op in ("floor", "ceil", "trunc", "rint") and oct in ("int", "long long"):
                return f"({ct}){cv(a[0], oct)}"
            ft = oct if oct in ("float", "double") else "double"
            return f"({ct}){op}{'f' if ft == 'float' else ''}({cv(a[0], ft)})"
        if op in _MATH:
            return f"{op}{f}({cv(a[0], ct)})"
        if op == "abs":
            return f"llabs({cv(a[0], ct)})" if ct == "long long" else f"abs({cv(a[0], ct)})" if ct == "int" else f"fabs{f}({cv(a[0], ct)})"
        if op == "sign":
            return f"sgn<{ct}>({cv(a[0], ct)})"
        if op in ("pow", "atan2", "fmod"):
            return f"{op}{f}({cv(a[0], ct)}, {cv(a[1], ct)})"
        if op in ("max", "min"):
            return f"{'mx' if op == 'max' else 'mn'}(({ct}){cv(a[0], ct)}, ({ct}){cv(a[1], ct)})"
        if op in _CMP:
            return f"{cv(a[0], oct)} {_CMP[op]} {cv(a[1], oct)}"
        if op in ("select", "iselect"):
            return f"{cv(a[0], 'bool')} ? ({ct}){cv(a[2], ct)} : ({ct}){cv(a[1], ct)}"
        if op in ("and", "or"):
            return f"{cv(a[0], 'bool')} {'&&' if op == 'and' else '||'} {cv(a[1], 'bool')}"
        if op == "not":
            return f"!{cv(a[0], 'bool')}"
        if op in ("b2f", "b2i"):
            return f"({ct}){cv(a[0], 'bool')}"
        if op in ("tof16", "tobf16"):  # (frontend._round: rounded to the format, as a conversion to it and back)
            return f"__half2float(__float2half_rn({cv(a[0], 'float')}))" if op == "tof16" else \
                f"__bfloat162float(__float2bfloat16_rn({cv(a[0], 'float')}))"
        if op in ("tof32", "tof64"):
            return f"({ct}){cv(a[0], oct)}"
        if op == "itof":
            return f"({ct})(unsigned long long){cv(a[0], 'long long')}" if a[1] else f"({ct}){cv(a[0], 'long long')}"
        if op == "ftoi":
            x = cv(a[0], oct)
            return f"(long long){x}" if oct in ("int", "long long") else f"({x} != {x} ? 0LL : (long long){x})"  # (integers:
            # no NaN)
        if op == "fbits":
            return f"__double_as_longlong({cv(a[0], 'double')})"
        if op == "bitsf64":
            return f"__longlong_as_double({cv(a[0], 'long long')})"
        if op == "f32bits":
            return f"(long long)__float_as_int({cv(a[0], 'float')})"
        if op == "bitsf32":
            return f"__int_as_float((int){cv(a[0], 'long long')})"
        if op in _IBIN:  # (wraparound defined: unsigned; nvcc has no -fwrapv, and LLVM folds signed overflow away)
            return f"(long long)((unsigned long long){cv(a[0], ct)} {_IBIN[op]} (unsigned long long){cv(a[1], ct)})"
        if op in ("ishl", "ishr"):
            return f"(long long)((unsigned long long){cv(a[0], ct)} << {cv(a[1], ct)})" if op == "ishl" else \
                f"{cv(a[0], ct)} >> {cv(a[1], ct)}"
        if op == "ishru":
            return f"(long long)((unsigned long long){cv(a[0], ct)} >> {cv(a[1], ct)})"
        if op == "iwrap":
            x, bits, signed = a
            if signed and bits in (8, 16, 32):  # (sign extension of the low bits, as the shifts below; one conversion)
                narrow = {8: "signed char", 16: "short", 32: "int"}[bits]
                return f"(long long)({narrow}){cv(x, oct if oct in ('int', 'long long') else ct)}"
            if signed:
                return f"(long long)((unsigned long long){cv(x, ct)} << {64 - bits}) >> {64 - bits}"
            return f"{cv(x, ct)} & {_lit((1 << bits) - 1, 'long long')}"
        if op == "threefry":
            return f"kx_threefry({', '.join(cv(x, 'long long') for x in a)})"
        if op == "icmp":
            c, x, y, u = a
            cast = "(unsigned long long)" if u else ""
            return f"{cast}{cv(x, 'long long')} {_CMP[c]} {cast}{cv(y, 'long long')}"
        raise NotImplementedError(f"kernelax: '{op}' is not supported in CUDA kernels")

    def mem_load(k, at, ct):  # entry `at` of input k as type ct
        v = f"in{k}[{at}]"
        return f"({ct}){v}" if _CT[kin[k]] != ct else v

    def value(x, out, done, ct="double"):
        """Name of x (constants: literals of type ct) after emitting into `out` what it needs and `done` lacks."""
        if not isinstance(x, Node):
            return _lit(x, ct)
        for n in schedule([x], done):
            if n.id in done:  # (the cos of a sincos emitted with its sin, or the other way round; a guarded block's)
                continue
            if len(inside) < dag.GUARD_DEPTH and any(guard(g, out, done) for g in sorted(  # (the work of a select's
                    (g for g in guards.get(n.id, ()) if g[3] not in opened and (g[3], id(done)) not in refused),
                    key=lambda g: -len(g[3]))):  # side alone: the outermost group first)
                continue
            done.add(n.id)
            nm = names[n.id] = f"t{n.id}"
            ct_n = ctype(n)
            if n.op == "in" and not batched[n.args[0]]:  # (an entry the same for all samples: loaded where used, one
                k, i = n.args  # broadcast load per warp from L1, instead of a register for the whole kernel)
                v = f"ldu<{i}>(in{k})"
                names[n.id] = f"(({ct_n}){v})" if _CT[kin[k]] != ct_n else v
            elif n.op == "in":
                k, i = n.args
                out.append(f"  const {ct_n} {nm} = {mem_load(k, f'sr * {sizes[k]} + {i}', ct_n)};")
            elif n.op == "sample":
                out.append(f"  const {ct_n} {nm} = ({ct_n})s;")
            elif n.op == "while":
                loop(n, out, done)
            elif n.op in ("sin", "cos") and isinstance(n.args[0], Node) and (
                    other := trig.get(("cos" if n.op == "sin" else "sin", n.args[0].id))) is not None and other.id not in done \
                    and all((other.id in w) == (n.id in w) for w in inside):  # (in the same guarded block, or none)
                sn, cn = (n, other) if n.op == "sin" else (other, n)  # (the other one: done in this scope)
                names[sn.id], names[cn.id] = f"t{sn.id}", f"t{cn.id}"
                done.add(other.id)
                out.append(f"  {ct_n} t{sn.id}, t{cn.id}; sincos{'f' if ct_n == 'float' else ''}({names[n.args[0].id]}, &t{sn.id}, &t{cn.id});")
                out += [f"  A{a}[{j}] = ({atype(arrays[a])}){names[other.id]};" for a, j in slots.get(other.id, ())]
            elif n.op in ("loopout", "loopmask"):
                names[n.id] = f"c{n.args[0].id}_{n.args[1]}"
            elif n.op == "dense" and not mode["loops"]:
                pass  # (its results unrolled: each where it is used)
            elif n.op == "denseout" and not mode["loops"]:
                (k, c0, sj, sk, M, K, T), xs = n.args[0].args[0]
                j, t = divmod(n.args[1], T)
                col = [value(xs[i * T + t], out, done, ct_n) for i in range(K)]  # (the operand's column: dag.hidden)
                w = lambda i: f"ldu<{c0 + j * sj + i * sk}>(in{k})" if _CT[kin[k]] == ct_n else f"(({ct_n})ldu<{c0 + j * sj + i * sk}>(in{k}))"
                out.append(f"  const {ct_n} {nm} = {' + '.join(f'{w(i)} * {c}' for i, c in enumerate(col))};")
            elif n.op == "dense":  # (frontend._dense) a loop over the matrix rows, not unrolled: per row K x (used
                # columns of T) FMAs, the matrix entries from the read-only cache (the same in a warp: one broadcast),
                # the results to local memory; with the other products of this matrix that neither depend on these nor
                # these on them (a layer's value and its derivatives: the matrix loaded once)
                (k, c0, sj, sk, M, K, _), _ = n.args[0]
                group, need = [n], {n.id: _reach(_operand(n, cols), done)}
                for m in dense.get(n.args[0][0][:6], []):
                    if m.id in done or m is n or ctype(m) != ct_n:
                        continue
                    req = _reach(_operand(m, cols), done - {n.id})  # (n: in done already)
                    if not any(g.id in req or m.id in need[g.id] for g in group):
                        group.append(m)
                        need[m.id] = req
                done.update(m.id for m in group)  # (before their operands: a dense node emitted for them must not
                for m in group:  # take one of these)
                    for y in _operand(m, cols):
                        value(y, out, done, ct_n)
                T = {m.id: m.args[0][0][6] for m in group}
                acc = {(m.id, t): f"a{m.id}_{t}" for m in group for t in sorted(cols[m.id])}
                out += [f"  {ct_n} D{m.id}[{M * T[m.id]}];" for m in group]
                out += [f"  _Pragma(\"unroll 1\") for (int j = 0; j < {M}; ++j) {{",
                        f"    const {_CT[kin[k]]}* w = in{k} + {c0} + j * {sj}; {ct_n} {', '.join(f'{a} = 0' for a in acc.values())};"]
                x_of = {m.id: m for m in group}
                fma = lambda i, v: " ".join(f"{a} += {v} * {cv(x_of[mid].args[0][1][i * T[mid] + t], ct_n)};" for (mid, t), a in acc.items())
                if _CT[kin[k]] == ct_n == "float" and sk == 1 and K % 4 == 0 and c0 % 4 == 0 and sj % 4 == 0:  # (rows
                    out += [f"    {{ const float4 v = __ldg(reinterpret_cast<const float4*>(w + {i})); "  # contiguous and
                            + " ".join(fma(i + d, f"v.{c}") for d, c in enumerate("xyzw")) + " }" for i in range(0, K, 4)]  # aligned:
                else:  # four entries per load; XLA's buffers are aligned)
                    out += [f"    {{ const {ct_n} v = ({ct_n})__ldg(w + {i * sk}); " + fma(i, "v") + " }" for i in range(K)]
                out += [f"    D{mid}[j * {T[mid]} + {t}] = {a};" for (mid, t), a in acc.items()] + ["  }"]
            elif n.op == "denseout":
                out.append(f"  const {ct_n} {nm} = D{n.args[0].id}[{n.args[1]}];")
            elif n.op == "load":
                k, i = n.args
                out.append(f"  const {ct_n} {nm} = {mem_load(k, f'ix({names[i.id] if isinstance(i, Node) else _lit(i, 'double')}, {sizes[k]}LL)', ct_n)};")
            elif n.op == "table":
                vals, i = n.args
                t = tables.setdefault((tuple(vals), ct_n), len(tables))
                out.append(f"  const {ct_n} {nm} = TAB{t}[ix({cv(i, 'double')}, {len(vals)}LL)];")
            elif n.op in ("arr", "iarr"):
                pass  # (the options of a pick: taken there, or stored where computed, below)
            elif n.op in ("pick", "ipick") and n.args[0].id in arrays:
                out.append(f"  const {ct_n} {nm} = A{n.args[0].id}[ix({cv(n.args[1], 'double')}, {len(n.args[0].args)}LL)];")
            elif n.op in ("pick", "ipick"):
                arr, i = n.args
                opts = [cv(o, ct_n) for o in arr.args]
                out.append(f"  {ct_n} {nm} = {opts[0]};")  # (no block in braces: _spill's reloads before a line of
                out.append(f"  const long long q{n.id} = ix({cv(i, 'double')}, {len(opts)}LL);")  # the chain stay visible)
                out += [f"  {nm} = q{n.id} == {j} ? ({ct_n}){o} : {nm};" for j, o in enumerate(opts[1:], 1)]
            else:
                out.append(f"  const {ct_n} {nm} = {expr(n)};")
            out += [f"  A{a}[{j}] = ({atype(arrays[a])}){names[n.id]};" for a, j in slots.get(n.id, ())]  # (as an option
        return names[x.id]  # of an array: stored where computed, not live until the array's reads)

    def guard(grp, out, done):
        """The work of a group of selects' side (dag.guards) in a block run only where the thread takes that side;
        its results into values declared before (zero where it does not: the selects take the other side there)."""
        p, side, outs, work = grp
        pre = [n for n in schedule([p, *outs], done) if n.id not in work]  # (what the work reads from outside it: before
        if any(n.op in ("carry", "carrymask") for n in pre):  # the block; a carry not defined here: emit.guard)
            refused.add((work, id(done)))
            return False
        opened.add(work)
        m = value(p, out, done, "bool")
        for n in pre:
            if n.id not in done:
                value(n, out, done)
        out += [f"  {ctype(o)} g{o.id} = ({ctype(o)})0;" for o in outs]
        inner = []
        inside.append(work)
        for o in outs:
            inner.append(f"  g{o.id} = {value(o, inner, done, ctype(o))};")
        inside.pop()
        out += [f"  if ({'' if side == 2 else '!'}({m})) {{"] + ["  " + ln for ln in inner] + ["  }"]
        for o in outs:
            names[o.id] = f"g{o.id}"
        return True

    def loop(L, out, done):
        (phs, nxt, cond), inits = L.args[0], L.args[1:1 + len(L.args[0][0])]
        scope = set(done)
        for i, (p, x) in enumerate(zip(phs, inits)):
            v = value(x, out, done, ctype(p))
            names[p.id] = f"c{L.id}_{i}"
            scope.add(p.id)
            out.append(f"  {ctype(p)} c{L.id}_{i} = {v};")
        body = []
        c = value(cond, body, scope, "bool")
        body.append(f"  if (!({c})) break;")
        new = [value(o, body, scope, ctype(p)) for p, o in zip(phs, nxt)]
        body += [f"  const {ctype(p)} u{L.id}_{i} = {v};" for i, (p, v) in enumerate(zip(phs, new))]  # (then all at once)
        body += [f"  c{L.id}_{i} = u{L.id}_{i};" for i in range(len(phs))]
        out += ["  for (;;) {"] + ["  " + b for b in body] + ["  }"]

    pos = {n.id: k for k, n in enumerate(schedule(roots))}
    ready = lambda x: max((pos.get(a.id, -1) for a in x.args if isinstance(a, Node)), default=-1) if isinstance(x, Node) else -1
    direct = [not (scat[j] or reduce[j] or staged[j]) for j in range(len(outputs))]

    def body():  # the statements computing and storing the outputs: the entries written directly each where it is
        names.clear()  # ready (the latest of its operands, in emission order), so that it is not live to the end
        opened.clear()
        refused.clear()
        lines, done = [], set()  # (the ys of an unrolled scan: per step, not all of them; the final carry last)
        for j, a in enumerate(outputs):
            if not direct[j]:
                store(j, a, lines, done)
        entries = [(j, i, x) for j, a in enumerate(outputs) if direct[j] for i, x in enumerate(np.asarray(a, dtype=object).ravel())]
        for j, i, x in sorted(entries, key=lambda e: ready(e[2])):
            lines.append(f"  if (live) out{j}[s * {int(np.size(outputs[j]))} + {i}] = {written(j, value(x, lines, done, OT(j)))};")
        return lines

    OT = lambda j: "long long" if E == 4 and kout[j] in "iu" else ET  # (constants of output j: integer literals)

    def written(j, v):  # the value as output j's element (int32 bits; 16-bit floats rounded at the store)
        out16 = {"h": "__float2half_rn", "b": "__float2bfloat16_rn"}.get(kout[j], "")
        return f"{out16}(__int_as_float((int)(long long)({v})))" if E == 4 and kout[j] in "iu" else f"{out16}(({ET})({v}))"

    def store(j, a, lines, done):
        if scat[j]:  # (positions outside, -1 or beyond the array, e.g. from a gather in fill mode: dropped)
            for f, x in a.entries:
                v, p = value(x, lines, done, ET), value(f, lines, done, "long long")
                lines.append(f"  if (live && (unsigned long long)({p}) < {a.size}ULL) kx_add(out{j} + (long long)({p}), ({ET})({v}));")
            return
        flat = list(np.asarray(a, dtype=object).ravel())
        vals = [value(x, lines, done, OT(j)) for x in flat]
        if reduce[j]:
            for i, v in enumerate(vals):
                lines += [f"  {{ const {ET} w = wsum<{ET}>(live ? ({ET})({v}) : ({ET})0);",
                          f"    if ((threadIdx.x & 31) == 0) kx_add(out{j} + {i}, w); }}"]
            return
        st = lambda v: f"__int_as_float((int)(long long)({v}))" if E == 4 and kout[j] in "iu" else f"({ET})({v})"  # (int32 bits)
        out16 = {"h": "__float2half_rn", "b": "__float2bfloat16_rn"}.get(kout[j], "")  # (16-bit floats: rounded at the store)
        R, Rs = len(vals), len(vals) | 1  # (staged: the rows through shared memory, written by the block together)
        lines += ["  __syncthreads();"] + [f"  SM[threadIdx.x * {Rs} + {i}] = {st(v)};" for i, v in enumerate(vals)]
        lines += ["  __syncthreads();", f"  for (int k = threadIdx.x; k < cnt * {R}; k += BLK) out{j}[s0 * {R} + k] = {out16}(SM[k / {R} * {Rs} + k % {R}]);"]
    mode["loops"] = bool(loops)
    lines = body()
    if loops is None and any(n.op == "dense" for n in nodes) and max_live(*_parse(lines)) > live:  # (unrolled they
        mode["loops"] = True  # would not fit the registers: ptxas spills such code, also with _spill's slots; e.g. a
        lines = body()  # PINN's MLP)
    spilled, slots = _spill(lines, live if E == 4 else None)
    if 32 * (stage * E + slots * 4) > smem:  # (with the staged rows no room for the slots: ptxas spills)
        spilled, slots = lines, 0
    lines = spilled
    B = BLOCK  # threads per block: the staging rows and the spill slots in shared memory
    while B > 32 and B * (stage * E + slots * 4) > smem:
        B //= 2
    lines = [ln.replace("BLK", str(B)) for ln in lines]
    tabs = "".join(f"__device__ const {ct} TAB{t}[{len(v)}] = {{{', '.join(_lit(c, ct) for c in v)}}};\n" for (v, ct), t in tables.items())
    params = ["long long n"] + [f"const {_CT[k]}* __restrict__ in{i}" for i, k in enumerate(kin)] + [f"{_CT[kout[j]] if kout[j] in 'hb' else ET}* __restrict__ out{j}" for j in range(len(outputs))]
    head = [(f"static void kernelax_k({', '.join(params)}) {{" if cpu else
             f"extern \"C\" __global__ void __launch_bounds__({B}, 1) kernelax_k({', '.join(params)}) {{"),
            f"  const long long s0 = (long long)blockIdx.x * {B}, s = s0 + threadIdx.x, sr = s < n ? s : n - 1;  // (sr: rows of the last sample past the end)",
            f"  const bool live = s < n; const int cnt = (int)(n - s0 < {B} ? n - s0 : {B}); (void)cnt; (void)live;"]
    for a in arrays.values():  # (options of picks, per thread; constants set here, the others where computed)
        head.append(f"  {atype(a)} A{a.id}[{len(a.args)}];")
        head += [f"  A{a.id}[{j}] = {_lit(o, atype(a))};" for j, o in enumerate(a.args) if not isinstance(o, Node)]
    if any(staged):
        head.append(f"  __shared__ {ET} SM[{B * stage}];")
    if slots:
        head += [f"  __shared__ float SP[{B * slots}];  // (values spilled by _spill)",
                 "  const unsigned spb = (unsigned)__cvta_generic_to_shared(SP + threadIdx.x);"]
    half = set(kin + kout) & set("hb") or any(n.op in ("tof16", "tobf16") for n in nodes)
    if cpu:
        bad = sorted({c for c in _GPU_ONLY for ln in head + lines if c in ln} | ({"16-bit floats"} if half else set()))
        if bad:
            raise NotImplementedError(f"kernelax.export: the point mode has no CPU code for {', '.join(bad)}")
        return _CPU_PRELUDE + _HELPERS + tabs + "\n".join(head + lines) + "\n}\n", B, []
    src = (_PRELUDE16 if half else "") + _PRELUDE + tabs + "\n".join(head + lines) + "\n}\n"
    return src, B, [j for j in range(len(outputs)) if reduce[j] or scat[j]]


MAX_LIVE = 8192  # 32-bit words live at once at most (live_words): beyond,
# a kernel is refused before nvcc (ptxas spills such code for minutes, into tens of KB of local memory per thread;
# e.g. a local Newton solve per quadrature point unrolled with all tangent directions)
_DECL = re.compile(r"^\s*(?:const )?(double|long long|float|int|unsigned|bool|__half|__nv_bfloat16) ([A-Za-z_]\w*) = ")
_WORDS = re.compile(r"\b[A-Za-z_]\w*\b")


def live_words(src):
    """The most 32-bit words of named values live at once in a kernel's source (double and long long: two), at every
    level: a loop or block is a statement whose own peak adds to the values live around it."""
    lines = src.split("\n")
    return _live(lines[next(i for i, ln in enumerate(lines) if "kernelax_k(" in ln) + 1:])


def _live(lines):
    stmts, i = [], 0
    while i < len(lines):  # (statements at this level: a line, or a block to its closing line at its indentation)
        j = i + 1
        if lines[i].endswith("{"):
            ind = len(lines[i]) - len(lines[i].lstrip())
            while j < len(lines) and not (lines[j].lstrip().startswith("}") and len(lines[j]) - len(lines[j].lstrip()) == ind):
                j += 1
            j += 1
        stmts.append(lines[i:j])
        i = j
    words, last = {}, {}
    for k, st in enumerate(stmts):
        if len(st) == 1 and (m := _DECL.match(st[0])):
            words[m.group(2)] = (k, 2 if m.group(1) in ("double", "long long") else 1)
    for k, st in enumerate(stmts):
        for ln in st:
            for u in _WORDS.findall(ln):
                if u in words:
                    last[u] = k
    dying = collections.defaultdict(list)
    for u, k in last.items():
        dying[k].append(u)
    born = collections.defaultdict(list)
    for u, (k, _) in words.items():
        if last.get(u, k) > k:
            born[k].append(u)
    cur = most = 0
    for k, st in enumerate(stmts):
        if len(st) > 1:
            most = max(most, cur + _live(st[1:-1]))
        cur += sum(words[u][1] for u in born[k]) - sum(words[u][1] for u in dying[k] if words[u][0] < k)
        most = max(most, cur)
    return most


TUNE_LONG = 0.02  # seconds: a first call (variant 0, all its samples) longer than this is not repeated for the choice of
# a variant; the next calls run the other variants, one each (gpu_runtime.cu; timed in the call: 2 V + 1 times it)
NVCC_TIMEOUT = env("KERNELAX_NVCC_TIMEOUT", 120.0, float, 0.001)  # seconds for a kernel's nvcc (its variants at once);
# beyond, a kernel is too large and its region stays in XLA (as build.GCC_TIMEOUT: MAX_NODES bounds the operations,
# not ptxas' time; e.g. a Newton solve per quadrature point unrolled, 8 MB of source)


class NvccTimeout(NotImplementedError):
    """nvcc ran out of time (in the index with its limit: final after two tries at that limit, _failed)."""


def _failed(entry):
    """Whether an index entry's error stands (kernel._failed's rule, for NVCC_TIMEOUT)."""
    return "timeout" not in entry or entry["tries"] >= 2 and entry["timeout"] >= NVCC_TIMEOUT


VARIANTS = ("-O3", "-O1")  # ptxas levels: the second built only where the first spills (its scheduler raises the
# register pressure of large straight-line kernels), the faster one chosen by timing at the first call (gpu_runtime.cu)


def build(src, alts=()):
    """The cubins of a kernel's variants, (cubin, index of its source: 0 src, i alts[i - 1]) (device code only, cached on
    disk by source and ptxas level): src at ptxas -O3; where that spills also src at -O1 and the alternative sources
    (dense layers as loops) at -O3. Each source through cicc once (nvcc -ptx), its PTX then through ptxas per level;
    all at once (the wall time of the slowest), the others stopped unless needed. Within NVCC_TIMEOUT: else an
    NvccTimeout, or (beyond the first variant) the variants built so far. Built once at a time (build.locked: another
    process or thread building the same source is waited for, its cubins then found)."""
    d = pathlib.Path(CACHE) / "gpu"
    d.mkdir(parents=True, exist_ok=True)
    first = _cubin(src, VARIANTS[0], d)
    if intact(first[0]) and not _spills(first[1]):  # (cached without spills: done)
        return [(first[0], 0)]
    with locked(first[0].with_suffix(".lock")):
        return _build(src, alts, d, first)


def _build(src, alts, d, first):  # (build, under its lock)
    deadline = time.monotonic() + NVCC_TIMEOUT
    if intact(first[0]) and not _spills(first[1]):
        return [(first[0], 0)]
    wanted = [(src, VARIANTS[0], 0)] + [(src, opt, 0) for opt in VARIANTS[1:]] + [(a, VARIANTS[0], i) for i, a in enumerate(alts, 1)]
    ptx = {}  # source -> (its PTX file, cicc started or None)
    for source, opt, _ in wanted:
        if source not in ptx and not intact(_cubin(source, opt, d)[0]):
            ptx[source] = _ptx(source, d)
    started, out = [], []
    too_long = lambda: NvccTimeout(f"kernelax GPU: nvcc took more than {NVCC_TIMEOUT} s for a kernel of {len(src) // 1024} KB: "
                                   "too large (KERNELAX_NVCC_TIMEOUT raises the limit)")
    try:
        for n, (source, opt, i) in enumerate(wanted):
            if n and not _spills(first[1]):
                break
            cubin, info = _cubin(source, opt, d)
            if not intact(cubin):
                for src_n, (file, p) in ptx.items():  # (the PTX: of every source wanted, then ptxas on each level)
                    if p is not None and p.returncode is None:
                        try:
                            _finish(p, file, file.with_suffix(".ptx.log"), deadline)
                        except subprocess.TimeoutExpired:
                            if n:
                                return out
                            raise too_long() from None
                        for s2, o2, _ in wanted:  # (ptxas on every level of a source whose PTX is ready)
                            if s2 == src_n and not intact(_cubin(s2, o2, d)[0]) and (s2, o2) not in [x[0] for x in started]:
                                started.append(((s2, o2), _ptxas(s2, o2, file, d)))
                if (source, opt) not in [x[0] for x in started]:
                    started.append(((source, opt), _ptxas(source, opt, ptx[source][0], d)))
                p = dict(started)[(source, opt)]
                try:
                    _finish(p, cubin, info, deadline)
                except subprocess.TimeoutExpired:
                    if n:
                        break
                    raise too_long() from None
            out.append((cubin, i))
    finally:
        for p in [p for _, (_, p) in ptx.items()] + [p for _, p in started]:
            if p is not None and p.returncode is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                pathlib.Path(p.args[p.args.index("-o") + 1]).unlink(missing_ok=True)
            _COMPILING.pop(p, None)
    return out


def _cubin(src, opt, d):
    """The cubin of src at ptxas level opt and its ptxas -v output (cache paths: by source, architecture and level)."""
    key = hashlib.sha256((src + arch() + opt + nvcc_version()).encode()).hexdigest()[:20]
    return d / f"{key}.cubin", d / f"{key}.ptxas"


def _ptx(src, d):
    """(src's PTX file, cicc (nvcc -ptx) started, or None if cached)."""
    key = hashlib.sha256((src + arch() + nvcc_version()).encode()).hexdigest()[:20]
    ptx, cu = d / f"{key}.ptx", d / f"{key}.ptx.cu"
    if ptx.exists():
        return ptx, None
    cu.write_text(src)
    p = subprocess.Popen([NVCC, "-O3", f"-arch={arch()}", "-ptx", "-o", str(ptx) + ".tmp", str(cu)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE, text=True, start_new_session=True)
    _COMPILING[p] = [str(ptx) + ".tmp"]  # (stopped at exit, its output removed: build._stop)
    return ptx, p


def _ptxas(src, opt, ptx, d):
    """ptxas started on src's PTX at level opt (its cubin's source written beside it: Kernel reads it from there)."""
    cubin, _ = _cubin(src, opt, d)
    cubin.with_suffix(".cu").write_text(src)
    p = subprocess.Popen([str(pathlib.Path(shutil.which(NVCC) or NVCC).with_name("ptxas")), f"-arch={arch()}", opt, "-v", "-o",
                          str(cubin) + ".tmp", str(ptx)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    _COMPILING[p] = [str(cubin) + ".tmp"]
    return p


def _finish(p, out, log, deadline):
    """Waits for p (until deadline: TimeoutExpired), its output moved to `out`, its stderr into `log`."""
    err = p.communicate(timeout=max(deadline - time.monotonic(), 0.0))[1]
    if p.returncode:
        raise RuntimeError(f"kernelax GPU: {pathlib.Path(p.args[0]).name} failed on {p.args[-1]}:\n{err[-2000:]}")
    log.write_text(err)
    os.replace(str(out) + ".tmp", out)


def _spills(info):
    """Whether a cubin's kernel spills (its ptxas -v output: bytes of spill stores)."""
    m = re.search(r"(\d+) bytes spill stores", info.read_text())
    return bool(m and int(m.group(1)))


@functools.cache
def runtime():
    """gpu_runtime.cu built (nvcc against jaxlib's FFI headers, cached by source) and its handler registered."""
    src = pathlib.Path(__file__).with_name("gpu_runtime.cu")
    d = pathlib.Path(CACHE) / "gpu"
    d.mkdir(parents=True, exist_ok=True)
    so = d / f"runtime_{hashlib.sha256(src.read_bytes() + (jax.__version__ + nvcc_version()).encode()).hexdigest()[:16]}.so"
    with locked(str(so) + ".lock"):  # (another process building it: waited for)
        if not intact(so):
            r = subprocess.run([NVCC, "-O2", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "--diag-suppress=940,2473", "-Xcompiler",
                                "-Wno-return-type,-Wno-attributes", f"-I{jax.ffi.include_dir()}", "-o", str(so) + ".tmp", str(src)],
                               capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"kernelax GPU: nvcc failed on the runtime {src}:\n{(r.stdout + r.stderr)[-2000:]}")
            os.replace(str(so) + ".tmp", so)
    lib = ctypes.CDLL(str(so))
    lib.kernelax_gpu_register.argtypes = [ctypes.c_uint64, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.kernelax_gpu_choice.argtypes, lib.kernelax_gpu_choice.restype = [ctypes.c_uint64], ctypes.c_int
    lib.kernelax_gpu_memory.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.kernelax_gpu_tuning.argtypes = [ctypes.c_double]
    lib.kernelax_gpu_work.argtypes, lib.kernelax_gpu_work.restype = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_longlong)], ctypes.c_longlong
    lib.kernelax_gpu_tuning(TUNE_LONG)
    jax.ffi.register_ffi_target("kernelax_gpu_call", jax.ffi.pycapsule(lib.kernelax_gpu_call), platform="CUDA")
    return lib


class LocalMemory(NotImplementedError):
    """A kernel's local memory (its stack: spills, arrays per thread) beyond what the GPU has free for all the threads
    it can hold (the driver reserves it for each of them at the launch; XLA preallocates most of the memory): not
    noted in the index (another process may have more free)."""


def _stack(cubin):
    """Bytes of local memory per thread of a cubin's kernel (ptxas -v: its stack frame)."""
    return max(map(int, re.findall(r"(\d+) bytes stack frame", pathlib.Path(cubin).with_suffix(".ptxas").read_text())), default=0)


_RESERVED = [0]  # local memory the kernels of this process have needed (bytes, the largest: the driver keeps it)


def _fitting(cubins):
    """The cubins whose local memory fits for all the threads the GPU can hold (the free memory, less what XLA's
    allocator may still take up to its limit before its first array, its preallocation; once that pool is taken, XLA
    grows beyond it only as needed (JAX >= 0.12.1 by default up to the whole GPU) and the two share the rest, as
    without preallocation, where only its use counts; plus what the driver holds already for earlier kernels);
    LocalMemory if none does (the launch would fail: out of memory)."""
    free, sms, threads = ctypes.c_ulonglong(), ctypes.c_int(), ctypes.c_int()
    if runtime().kernelax_gpu_memory(ctypes.byref(free), ctypes.byref(sms), ctypes.byref(threads)):
        raise RuntimeError("kernelax GPU: cudaMemGetInfo failed")
    st = jax.local_devices(backend="gpu")[0].memory_stats() or {}
    shared = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() in ("false", "0") or \
        os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") == "platform"  # (XLA takes memory as needed: shared, its use counts)
    xla = 0 if shared or st.get("pool_bytes", 0) else max(0, st.get("bytes_limit", 0) - st.get("bytes_in_use", 0))
    avail = free.value - xla + _RESERVED[0]
    need = {c: _stack(c) * sms.value * threads.value for c in cubins}
    ok = [c for c in cubins if need[c] == 0 or need[c] <= avail]
    if not ok:
        c = min(cubins, key=need.get)
        raise LocalMemory(f"kernelax GPU: the kernel needs {_stack(c)} bytes of local memory per thread, {need[c] / 2 ** 30:.1f} GiB "
                          f"for the GPU's {sms.value * threads.value} threads; {max(avail, 0) / 2 ** 30:.1f} GiB are free next "
                          "to XLA's memory (XLA_PYTHON_CLIENT_MEM_FRACTION, or XLA_PYTHON_CLIENT_PREALLOCATE=false to share it)")
    _RESERVED[0] = max(_RESERVED[0], *(need[c] for c in ok))
    return ok


BUILT = []  # the kernels of this process, in order
_REGISTERING = threading.RLock()  # (kernels build in threads; the first registration loads the runtime)
_BUILDING = (threading.Lock(), {})  # per kernel key, a lock: the same source built once at a time
NVCC_JOBS = env("KERNELAX_NVCC_JOBS", 0) or len(os.sched_getaffinity(0))  # kernels built at once (kernelax.jit:
# trace and emit one at a time, their nvcc in parallel)
_NVCC_SLOTS = threading.BoundedSemaphore(NVCC_JOBS)
_REGISTERED = {}  # key: the cubins registered under it (its variants, in order)
MAX_NODES = env("KERNELAX_GPU_MAX_NODES", 300000, int, 1)  # frontend.MAX_NODES while tracing a GPU kernel (as
# for the C kernels; nvcc: gymnax' 262 000 nodes in 18 s), beyond: the region stays in XLA
DENSE_MIN = 64  # frontend.DENSE_MIN while tracing a GPU kernel: dense layers (a uniform matrix times per-sample values)
# as loops over the rows (unrolled, an MLP's layers are one straight-line block that ptxas spills)
SCAN_SHORT = 16  # scans of at most this many steps unrolled (frontend.SCAN_SHORT): a short loop's dynamic indices
# into tables and buffers cost more on the GPU than its unrolled code (e.g. the stages of a Runge-Kutta step)
ROLL_XS = 50000  # frontend.ROLL_XS: scans rolled despite their xs where unrolled larger (gymnax: 200 steps over keys
# computed in the kernel, 262 000 nodes unrolled, nvcc 18 s; rolled 13 % slower)
PICK_INTS = True  # frontend.PICK_INTS: picks among integers too as arrays (the keys above, without LAZY_KEYS)
PICK_MIN = 4  # frontend.PICK_MIN: picks among more options than this as pick nodes (emitted as chains up to ARRAY_MIN
# options; the frontend's chains at 16, as on the CPU: Asterix and xland-minigrid 12 % faster, Snake 43 % slower)
LAZY_KEYS = True  # frontend.LAZY_KEYS: keys of random.split computed where read (gymnax' keys: at the step)
XLA_GPU = True  # frontend.XLA_GPU: special functions as XLA lowers them on the GPU (float32 lgamma in float32)
ROUND16 = True  # frontend.ROUND16: float16 / bfloat16 rounding as the hardware's conversion (2 instructions, not ~20)
SHORT_NODES = 20000  # frontend.SHORT_NODES: short loops unrolled within this many new nodes each (nvcc ~0.2 ms per
# statement; blackjax' 10 NUTS steps unrolled: 117 000 nodes, ~40 s to build and slower than rolled)


def _index(f, example, batched, reduce, E, keep_nans):
    from kernelax import cpu  # (the index: cpu.py's)
    return cpu._index_key(f, example, batched, f"gpu:{arch()}:{nvcc_version()}:{SCAN_SHORT}:{SHORT_NODES}:{ROLL_XS}:{PICK_INTS}:{PICK_MIN}:{LAZY_KEYS}:{ROUND16}:{ARRAY_MIN}:{LIVE}:{MAX_NODES}:{DENSE_MIN}:{XLA_GPU}:{E}", None, reduce, keep_nans)[0]


def indexed(f, example, batched, reduce, E, keep_nans=False):
    """Whether Kernel(...) finds the kernel in the index (loads it without trace, emit or nvcc, or raises at once)."""
    from kernelax import cpu
    entry = cpu._read(_index(f, example, batched, reduce, E, keep_nans))
    return bool(entry) and (_failed(entry) if "error" in entry else all(cpu._in_cache(c) and intact(c) for c in entry["cubins"]))


class Kernel:
    """A traced function's GPU kernel, registered with the runtime under `key`; with the index on disk (cpu.py)
    loaded without trace, emit or nvcc, the variant chosen in an earlier process only."""

    def __init__(self, f, example, batched, reduce, mem, keep_nans=False, cached=True, defer=False):
        """defer: nvcc and the registration in a thread after the emit (the key known at once, e.g. for compiling
        the program that calls the kernel meanwhile; wait() for the cubin)."""
        from kernelax import cpu  # (the index: cpu.py's)
        self._thread = self._error = None
        with cpu._LOCK, no_gc():  # (the frontend's DAG is global: trace and emit one kernel at a time; nvcc after, in parallel
            self.index = _index(f, example, batched, reduce, mem[0], keep_nans) if cached else None  # to other builds)
            entry = cpu._read(self.index) if self.index else None
            if entry and "error" in entry and _failed(entry):
                raise NotImplementedError(cpu._noted(entry, self.index))
            if entry and "error" not in entry and all(cpu._in_cache(c) and intact(c) for c in entry["cubins"]):
                self.key, self.blocks, self.zero, self.cubins = entry["key"], entry["blocks"], entry["zero"], entry["cubins"]
                choice = entry.get("choice")
                self.source = pathlib.Path(self.cubins[0]).with_suffix(".cu").read_text()
                self._register(*((self.cubins, self.blocks) if choice is None else ([self.cubins[choice]], [self.blocks[choice]])),
                               -1 if choice is None else 0)
                self.saved = self.stored = choice is not None or len(self.cubins) == 1
                BUILT.append(self)
                return
            try:
                alts, blocks = self._emit(f, example, batched, reduce, mem, keep_nans)
            except NotImplementedError as e:
                if self.index:  # (not a kernel: noted, so that later processes know at once)
                    cpu._put(self.index, {"error": str(e)})
                raise
        if defer:
            self._thread = threading.Thread(target=self._finish, args=(alts, blocks, entry), daemon=True)
            self._thread.start()
            return
        self._finish(alts, blocks, entry)
        self.wait()

    def _finish(self, alts, blocks, entry):  # nvcc, the registration, the index entry (an error: kept for wait())
        from kernelax import cpu
        try:
            self._compile(alts, blocks)
        except NotImplementedError as e:  # (a timeout only after two; local memory: per process)
            if self.index and not isinstance(e, LocalMemory):
                tries = (entry or {}).get("tries", 0) + 1 if isinstance(e, NvccTimeout) else None
                cpu._put(self.index, {"error": str(e)} | ({"timeout": NVCC_TIMEOUT, "tries": tries} if tries else {}))
            self._error = e
            return
        except Exception as e:  # noqa: BLE001 (raised by wait())
            self._error = e
            return
        if self.index:
            cpu._put(self.index, {"key": self.key, "blocks": self.blocks, "zero": self.zero, "cubins": self.cubins})

    def wait(self):
        """Once the kernel is built and registered (deferred: its nvcc may still run); raises its error."""
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise self._error

    def _emit(self, f, example, batched, reduce, mem, keep_nans):
        """Traces f and emits the kernel's source (and the alternatives' sources, their blocks)."""
        from kernelax import cpu  # (f's jaxpr: traced once, cpu.traced)
        names = ("SCAN_SHORT", "SHORT_NODES", "ROLL_XS", "PICK_INTS", "PICK_MIN", "LAZY_KEYS", "ROUND16", "MAX_NODES", "DENSE_MIN",
                 "XLA_GPU")  # (the frontend's
        # settings for GPU kernels)
        saved = [getattr(frontend, k) for k in names]
        for k in names:
            setattr(frontend, k, globals()[k])
        try:
            inputs, outs, _ = trace(f, *example, closed=cpu.traced(f, example), uniform=[not b for b in batched], reduce=reduce,
                                    keep_nans=keep_nans)
        finally:
            for k, v in zip(names, saved):
                setattr(frontend, k, v)
        nodes = _all_nodes([x for o in outs for x in _flat(o) if isinstance(x, Node)])
        self.source, block, self.zero = emit(inputs, outs, batched, reduce, mem, nodes=nodes)
        if 2 * self.source.count("\n") > MAX_LIVE and (live := live_words(self.source)) > MAX_LIVE:  # (at most two
            # words per line: shorter sources stay below)
            raise NotImplementedError(f"kernelax GPU: a kernel with {live} words (32-bit) live at once, far beyond the 255 "
                                      "registers: ptxas would spill most of them, for minutes")
        self.key = int(hashlib.sha256(self.source.encode()).hexdigest()[:15], 16)
        alts = [emit(inputs, outs, batched, reduce, mem, loops=True, nodes=nodes)[:2]] if '_Pragma("unroll 1")' not in self.source \
            and any(n.op == "dense" for n in nodes) else []
        return [a for a, _ in alts], [block] + [b for _, b in alts]  # (dense layers unrolled that ptxas spills
        # nonetheless: as loops too, timed against them at the first call)

    def _compile(self, alts, blocks):  # nvcc (one build of a source at a time), then registered
        with _BUILDING[0]:
            lock = _BUILDING[1].setdefault(self.key, threading.Lock())
        with lock, _NVCC_SLOTS:
            built = build(self.source, alts)
        self.cubins, self.blocks = [str(c) for c, _ in built], [blocks[i] for _, i in built]
        self._register(self.cubins, self.blocks, -1)
        self.saved = self.stored = len(self.cubins) == 1  # (stored: nothing (more) to store in the index)
        BUILT.append(self)

    def _register(self, cubins, blocks, chosen):
        with _REGISTERING:
            self._register_now(cubins, blocks, chosen)

    def _register_now(self, cubins, blocks, chosen):
        if self.key in _REGISTERED:  # (the same source again: its variants are registered)
            return
        ok = _fitting(cubins)  # (the variants whose local memory fits; LocalMemory if none)
        cubins, blocks = [c for c in cubins if c in ok], [b for c, b in zip(cubins, blocks) if c in ok]
        _REGISTERED[self.key] = cubins
        z = (ctypes.c_int * max(1, len(self.zero)))(*self.zero)
        for c, b in zip(cubins, blocks):
            data = pathlib.Path(c).read_bytes()
            runtime().kernelax_gpu_register(self.key, data, len(data), b, len(self.zero), z, chosen)

    def chosen(self):
        """The cubin run (the variant chosen at the first call; None before)."""
        c = runtime().kernelax_gpu_choice(self.key)
        return _REGISTERED[self.key][c] if c >= 0 else None

    def resources(self):
        """(registers, bytes of spill stores) of the cubin run (before a choice: the first variant's), from ptxas -v."""
        info = pathlib.Path(self.chosen() or self.cubins[0]).with_suffix(".ptxas").read_text()
        return int(re.search(r"Used (\d+) registers", info).group(1)), int(re.search(r"(\d+) bytes spill stores", info).group(1))

    def persist(self):
        """The variant chosen at the first call (of at least TUNE_MIN samples) into the index entry, so that later
        processes load only it; called after a call and at exit. saved: called (kernelax.jit need not wait for the choice:
        a later one is stored at exit)."""
        from kernelax import cpu
        self.saved = True
        if not self.stored and self.index and len(_REGISTERED[self.key]) > 1 and (c := self.chosen()) and cpu._read(self.index):
            cpu._update(self.index, lambda e: dict(e, choice=e["cubins"].index(c)) if e and e.get("choice") is None and
                        c in e.get("cubins", ()) else e)
            self.stored = True


@atexit.register
def _persist():
    for k in BUILT:
        k.persist()
