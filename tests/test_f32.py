# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""float32 programs: computed in float32 (kernels of 4-byte elements, 16 lanes on AVX-512), against jax.vmap in
float32. The coverage cases (test_coverage) with float32 and int32 arguments, and the API paths (kernelax, kernelax.jit)."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kernelax import kernel, jit
from kernelax.vmap import cvmap
from kernelax.build import simd_width
from kernelax.cpu import elements

from test_coverage import C, GAPS, TOL32

jax.config.update("jax_enable_x64", True)
F32_GAPS = dict(GAPS, **{"float32 math": None})
SKIP = {"random"}  # (keys from float seeds: the seeds differ in float32)


def _f32(a):
    a = np.asarray(a)
    return a.astype(np.float32) if np.issubdtype(a.dtype, np.floating) else a.astype(np.int32)


def _check(out, ref, tol=2e-5):
    assert len(out) == len(ref)
    for o, x in zip(out, ref):
        x = np.asarray(x, np.float64)  # (tol: float32 rounding, XLA's elementary functions differ by ulps)
        np.testing.assert_allclose(np.asarray(o, np.float64), x, rtol=tol, atol=tol * (1 + np.abs(np.where(np.isfinite(x), x, 0)).max(initial=0)),
                                   equal_nan=True)


@pytest.mark.parametrize("group, name, f, args", [
    pytest.param(*c, id=f"{c[0]}: {c[1]}", marks=[pytest.mark.xfail(strict=True, reason=F32_GAPS[c[1]])] if F32_GAPS.get(c[1]) else [])
    for c in C if c[0] not in SKIP])
def test_coverage_f32(group, name, f, args):
    args = [_f32(a) for a in args]
    try:
        ref = jax.tree.leaves(jax.vmap(f)(*args))
    except (TypeError, ValueError) as e:  # (the case mixes in float64 constants: no float32 function for JAX either)
        pytest.skip(f"jax: {str(e).splitlines()[0][:80]}")
    out = jax.tree.leaves(cvmap(f, [a[0] for a in args], (0,) * len(args), K=1)(*args))
    _check(out, ref, TOL32.get(name, 2e-5))


def test_elements():
    """4-byte elements exactly where all inputs and outputs fit in 4 bytes: 16 lanes on AVX-512, 8 on AVX2."""
    f32, f64, i32 = (jax.ShapeDtypeStruct((3,), d) for d in (jnp.float32, jnp.float64, jnp.int32))
    assert elements([f32, i32], [f32]) == 4 and elements([f32], [f64]) == 8 and elements([f32], [f32]) == 4
    for W in (None, 4):
        c = cvmap(lambda x: jnp.sin(x) * 2, (np.zeros(3, np.float32),), (0,), W=W)
        assert f"#define W {2 * (W or simd_width())}" in c.kernel.sources[0]


def test_rounding_as_xla():
    """Every float32 operation rounds to float32: sums whose float64 value differs (cancellation), integer conversions."""
    f = lambda x, i: ((x + 1e8) - 1e8, (x * 3).astype(jnp.int32), i.astype(jnp.float32) + x, jnp.sqrt(x) * jnp.float32(1.1))
    x = np.random.default_rng(0).uniform(0, 10, (1000, 4)).astype(np.float32)
    i = np.random.default_rng(1).integers(-2 ** 30, 2 ** 30, (1000, 4)).astype(np.int32)
    out = cvmap(f, (x[0], i[0]), (0, 0))(x, i)
    for o, r in zip(out, jax.vmap(f)(x, i)):
        assert o.dtype == r.dtype
        np.testing.assert_array_equal(o, r)


def test_mixed_and_api():
    """float32 next to float64 (an 8-byte kernel, float32 values still in float32), int32 indices, derivatives through
    the primitive (kernelax, kernelax.jit)."""
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0.5, 2, (200, 3)).astype(np.float32), rng.uniform(0.5, 2, (200, 3))
    f = lambda x, y: (jnp.log(x) * 2, jnp.log(y) + x.sum())
    for o, r in zip(cvmap(f, (x[0], y[0]), (0, 0))(x, y), jax.vmap(f)(x, y)):
        np.testing.assert_allclose(o, r, rtol=1e-6)  # (the float32 sum: in another order)
        assert o.dtype == r.dtype or o.dtype == np.float64
    g = kernel(lambda v: jnp.sum(jnp.sin(v) ** 2))
    gr = jax.jit(jax.vmap(jax.grad(g)))(x)
    assert gr.dtype == np.float32
    np.testing.assert_allclose(gr, jax.vmap(jax.grad(lambda v: jnp.sum(jnp.sin(v) ** 2)))(x), rtol=2e-6)
    idx = rng.integers(0, 50, (20000,)).astype(np.int32)
    h = lambda t, i: jax.vmap(lambda j: jnp.tanh(t[j]) * t[(j + 1) % 50])(i)
    t = jnp.asarray(rng.standard_normal(50).astype(np.float32))
    J = jit(h)
    for _ in range(3):
        out = J(t, idx)
    J.wait()
    out = J(t, idx)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, h(t, idx), rtol=2e-6, atol=1e-7)


def test_without_x64():
    """JAX's default precision (no x64): float32 kernels under kernelax.jit, with run-time values (PRM: passed by their
    bits) and two sample axes (the sample index as int32): gathers and sums into arrays by index."""
    with jax.enable_x64(False):
        rng = np.random.default_rng(5)
        N, M = 64, 3000
        pos = jnp.asarray(rng.uniform(0, 30, (N, 2)), jnp.float32)
        idx = jnp.asarray(rng.integers(0, N, (M, 2)), jnp.int32)

        def f(pos, idx):
            d = pos[idx[:, 0]] - pos[idx[:, 1]]
            e = jnp.sum(d * d, axis=1)
            return jnp.sum(e), jnp.zeros(N).at[idx[:, 0]].add(e), jax.vmap(lambda r: jax.vmap(lambda c: jnp.sin(r * c))(pos[:8, 1]))(pos[:, 0])
        J = jit(f)
        J(pos, idx)
        J.wait()
        out, ref = J(pos, idx), jax.jit(f)(pos, idx)
        assert J.report()[0]["status"] in ("trial", "kernels")
        for o, r in zip(out, ref):
            assert o.dtype == r.dtype == jnp.float32
            np.testing.assert_allclose(o, r, rtol=1e-5, atol=1e-5 * float(np.abs(np.where(np.isfinite(np.asarray(r)), np.asarray(r), 0)).max(initial=0)))


def test_derivatives_of_mixed_kernels():
    """Reverse mode through a kernel with float32 outputs next to 8-byte values (the cotangents in the kernel's
    widths), cotangents in the inputs' dtypes, weak-typed Python scalars kept weak (float32 stays float32, int32
    stays int32), static arguments told apart by type (2 and 2.0), as jax."""
    rng = np.random.default_rng(8)
    x64, x32 = rng.uniform(0.2, 1.5, 7), rng.uniform(0.2, 1.5, 7).astype(np.float32)
    f = lambda x: jnp.sin(x).astype(jnp.float32)
    loss = lambda g: lambda x: jnp.sum(jax.vmap(g)(x))
    np.testing.assert_allclose(jax.grad(loss(kernel(f)))(x64), jax.grad(loss(f))(x64), rtol=1e-6)
    h = lambda x, y: jnp.sin(x) * y.astype(jnp.float32) + y ** 2  # (float32 x next to a float64 y)
    for k in (0, 1):
        got = jax.grad(lambda x, y: jnp.sum(jax.vmap(kernel(h))(x, y)), argnums=k)(x32, x64)
        ref = jax.grad(lambda x, y: jnp.sum(jax.vmap(h)(x, y)), argnums=k)(x32, x64)
        assert got.dtype == ref.dtype
        np.testing.assert_allclose(got, ref, rtol=1e-6)
    s = lambda x, c: x * c
    for args in ((x32[0], 0.5), (np.int32(7), 3), (x32, 2)):
        assert kernel(s)(*args).dtype == s(*args).dtype, args
    np.testing.assert_allclose(jax.grad(lambda x: kernel(lambda x, n: jnp.sin(x) * n)(x, 2))(x32[0]), 2 * np.cos(x32[0]), rtol=1e-6)
    ks, js = kernel(s, static_argnums=1), jit(s, static_argnums=1)
    i = np.int32(2 ** 30)
    for g in (ks, js):
        assert g(i, 2).dtype == s(i, 2).dtype and g(i, 2.0).dtype == s(i, 2.0).dtype
        assert float(g(i, 2.0)) == 2.0 ** 31


def _narrow_reloads(asm):
    """Places in gcc's assembly where a mask stored to the stack a byte wide (kmovb) is read back wider (movzwl, kmovw,
    ...) while the bytes above it were not written by mask stores: GCC PR target/117159 (gcc 12-14), lanes 8-15 from old
    stack contents. Conservative: gcc 15 reads such slots two bytes wide too, harmlessly (the upper byte shifted out
    of the mask), and is counted as well."""
    import re

    def width(op, regs):  # bytes moved to or from memory
        if op.startswith("kmov"):
            return {"b": 1, "w": 2, "d": 4, "q": 8}[op[4]]
        if op.startswith("v"):
            return 64 if "%zmm" in regs else 32 if "%ymm" in regs else 16 if "%xmm" in regs else 64
        if m := re.fullmatch(r"mov[sz]([bw])[wlq]", op):
            return {"b": 1, "w": 2}[m.group(1)]
        return {"b": 1, "w": 2, "l": 4, "q": 8}.get(op[-1], 8)
    last, bad = {}, 0  # stack byte -> its last store: "kb" (kmovb), "k" (another mask store), "o"
    for line in asm.splitlines():
        parts = line.split(None, 1)
        if len(parts) < 2 or parts[0].startswith((".", "#")):
            continue
        op, args = parts[0], [a.strip() for a in re.split(r",(?![^(]*\))", parts[1])]
        at = lambda a: int(m.group(1)) if (m := re.fullmatch(r"(-?\d+)\(%rsp\)", a)) else None
        for a in args[:-1]:  # (reads)
            if (x := at(a)) is not None and not op.startswith("v") and (w := width(op, parts[1])) > 1 \
                    and last.get(x) == "kb" and any(last.get(x + i) not in ("kb", "k") for i in range(1, w)):
                bad += 1
        if len(args) >= 2 and (x := at(args[-1])) is not None:  # (a store)
            for i in range(width(op, parts[1])):
                last[x + i] = ("kb" if op == "kmovb" else "k") if op.startswith("kmov") else "o"
    return bad


@pytest.mark.skipif(os.uname().machine != "x86_64", reason="x86-64 only")
def test_16_lane_masks_under_older_gcc(monkeypatch):
    """The masks of 16 lanes from two compares of 8 (doubles and int64 in float32 kernels: counters, indices) by kunpackb:
    gcc 12-14 with Intel's and x86-64-v4's tuning (the export's AVX-512 variant, -march=native on Intel) spilled
    `lo | hi << 8` a byte wide and reloaded it two bytes wide (GCC PR target/117159; lanes 8-15 from old stack
    contents; znver4's tuning did not spill there). kernelax builds with gcc >= 12 and exports with the host's: the
    kernels' assembly by every gcc on the PATH, both tunings, has no such reload; the old join has, with gcc 12-14 (the
    check sees the bug). The kernel of 8 doubles is emitted (not built) on any x86-64 CPU: one without AVX-512 runs
    kernels of 4, without such masks."""
    import re
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    from kernelax import cpu, vec
    from kernelax.build import CC
    from kernelax.emit import emit_map
    ccs = sorted({CC} | {f for d in os.environ.get("PATH", "").split(os.pathsep) if os.path.isdir(d)
                         for f in os.listdir(d) if re.fullmatch(r"gcc-1[2-9]", f)})
    old = [cc for cc in ccs if re.fullmatch(r"gcc-1[234]", cc)]
    asm = lambda src, cc, t: subprocess.run([cc, "-O3", f"-march={t}", "-fno-math-errno", "-ffp-contract=fast", "-fwrapv", "-fopenmp",
                                             "-w", "-S", "-o", "-", "-x", "c", "-"], input=src, capture_output=True, text=True,
                                            check=True).stdout
    a = np.random.default_rng(5).uniform(size=(5, 5)).astype(np.float32)

    def source():
        with cpu._LOCK:
            inputs, outs, _, mem = cpu._trace(lambda m: jnp.linalg.eig(m)[1].real, [a], [True])
            return emit_map(inputs, outs, [True], 8, 1, None, mem)[0]  # (cvmap's source with W=8, K=1)
    with jax.enable_x64(False):
        src = source()
        monkeypatch.setattr(vec, "_kjoin", lambda p: f"((vm)({p[0]}))" if len(p) == 1 else f"((vm)({p[0]}) | (vm)({p[1]}) << 8)")
        before = source() if old else None
    jobs = [(s, cc, t) for cc in ccs for t in ("x86-64-v4", "sapphirerapids") for s in ([src, before] if cc in old else [src])]
    with ThreadPoolExecutor(len(jobs)) as pool:  # (gcc processes in parallel)
        found = list(pool.map(lambda j: _narrow_reloads(asm(*j)), jobs))
    for (s, cc, t), n in zip(jobs, found):
        assert (n > 0) if s is before else (n == 0), (cc, t, "old join" if s is before else "kernelax' join", n)
