# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""cvmap: a per-sample JAX function as one C/SIMD kernel over a batch, called on numpy arrays (internal: tests,
benchmarks, the kernel's sources)."""

import jax
import jax.numpy as jnp
import numpy as np

from kernelax.frontend import _wide
from kernelax.cpu import NT, _kind, elements, kernel
from kernelax.realify import complex_leaves


def cvmap(f, example, in_axes, *, n_threads=None, W=None, K=None, keep_nans=False):
    """Kernel of ``jax.vmap(f, in_axes)`` for per-sample arguments shaped like ``example``.

    example: the per-sample arguments (pytrees of arrays); in_axes: per argument 0 (batched along the
    leading axis) or None (the same for all samples). Returns call(*args, out=None) -> the outputs of f
    with a leading sample axis, in f's output tree: float64 (64-bit integers in their dtype), or for a kernel of
    4-byte elements (cpu.elements: float32 and 32-bit integers on AVX-512) in f's dtypes; `out`: preallocated
    output leaves of the kernel's elements (``call.empty(n)``). ``call.kernel``: the cpu.Kernel (sources, chosen K, traced DAG).
    W: SIMD width, by default the CPU's (8: AVX-512, 4: AVX2; W=4 also runs on AVX-512 CPUs).
    keep_nans: keep x * 0, x - x and 0 / x as operations, so that NaN and inf propagate as in IEEE (and XLA). By
    default they fold to 0, which keeps the code of derivatives small (the zeros of jacfwd's basis vectors).
    K: sub-blocks of W samples interleaved per block (hides the latency of dependent operations, emit.emit_map).
    By default the fastest of K = 1..4, measured on the first call with at least cpu.TUNE_MIN samples.
    n_threads: by default cpu.NT.
    """
    if len(example) != len(in_axes) or any(a not in (0, None) for a in in_axes):
        raise ValueError("kernelax: cvmap: in_axes needs 0 or None per argument")
    batched = [ax == 0 for x, ax in zip(example, in_axes) for _ in jax.tree.leaves(x)]
    if not any(batched):
        raise ValueError("kernelax: cvmap: no batched argument")
    leaves, tree = jax.tree.flatten(list(example))
    cin = [np.iscomplexobj(x) for x in leaves]
    cout = [o.dtype if jnp.iscomplexobj(o) else None for o in jax.tree.leaves(jax.eval_shape(f, *example))]
    if any(cin) or any(cout):  # (complex values in and out: [re, im] on a trailing axis)
        f, example = complex_leaves(f, tree, cin), [_split(x) if c else x for x, c in zip(leaves, cin)]
    k = kernel(f, example, batched, W, K, keep_nans=keep_nans)
    nt = n_threads or NT
    outs = jax.tree.leaves(jax.eval_shape(f, *example))  # (a kernel loaded from the index has no traced outputs)
    shapes, dtypes = [o.shape for o in outs], [o.dtype for o in outs]
    E = elements([jax.typeof(x) for x in jax.tree.leaves(example)], outs)
    el = np.float64 if E == 8 else np.float32
    empty = lambda n: [_aligned((n,) + sh, el) for sh in shapes]

    def call(*args, out=None):
        arrs = [np.require(_to(_split(a) if c else np.asarray(a)), requirements="C") for a, c in zip(jax.tree.leaves(args), cin)]
        n = next(a.shape[0] for a, b in zip(arrs, batched) if b)
        if [a.shape for a in arrs] != [((n,) if b else ()) + x.shape for x, b in zip(k.inputs, batched)]:
            raise ValueError("kernelax: cvmap: argument shapes differ from the example")
        res = empty(n) if out is None else out
        if not all(r.flags.c_contiguous and r.dtype == el and r.shape == (n,) + sh for r, sh in zip(res, shapes)):
            raise ValueError(f"kernelax: cvmap: out needs C-contiguous {np.dtype(el)} arrays of shapes {[(n,) + sh for sh in shapes]} "
                             "(call.empty(n))")
        k(nt, n, arrs, res)
        res = [r.view(d) if _wide(d) else r if E == 8 or _kind(d) == "f" and d == el else
               r.view(np.uint32 if _kind(d) == "u" else np.int32).astype(d) if _kind(d) != "f" else r.astype(d) for r, d in zip(res, dtypes)]
        return jax.tree.unflatten(k.out_tree, [_join(r, c) if c else r for r, c in zip(res, cout)])
    call.kernel, call.empty = k, empty
    return call


def _split(a):  # a complex array as [re, im] on a trailing axis
    a = np.asarray(a)
    return np.stack([a.real, a.imag], -1)


def _join(r, dtype):  # [re, im] on the trailing axis as complex values (set, not re + 1j im: 1j inf is nan + inf j)
    out = np.empty(r.shape[:-1], dtype)
    out.real, out.imag = r[..., 0], r[..., 1]
    return out


def _to(a):
    """An argument as the kernel reads it (as primitive.to_kernel; its dtype as JAX's: without x64, 32 bits)."""
    a = np.asarray(a, jax.dtypes.canonicalize_dtype(a.dtype))
    if a.dtype.itemsize == 8:
        return a.view(np.float64) if _wide(a.dtype) else np.asarray(a, np.float64)
    if jax.dtypes.issubdtype(a.dtype, np.floating):  # (also bfloat16 and float8: numpy does not know them as floats)
        return np.asarray(a, np.float32)
    return a if a.dtype == np.uint32 else a.astype(np.int32)


def _aligned(shape, dtype=np.float64):
    """Array at a 64-byte boundary (the kernels stream whole vectors into aligned outputs)."""
    size = np.dtype(dtype).itemsize
    raw = np.empty(int(np.prod(shape)) + 64 // size, dtype)
    start = (-raw.ctypes.data % 64) // size
    return raw[start:start + int(np.prod(shape))].reshape(shape)
