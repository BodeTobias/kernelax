# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""The GPU backend (kernelax.gpu): every case of benchmarks/cases.py with float32 arguments (cases that close over float64
constants: float64 kernels), jax.vmap(kernel(f)) on the GPU against jax.vmap(f) by XLA on the CPU (no TF32 matmuls).
Integer outputs may differ in 1 % of the samples: loop counts that end where a float32 residual is exactly zero
(newton_linesearch) depend on rounding (FMA contraction). Skipped without a CUDA build of jaxlib."""

import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not any(d.platform == "gpu" for d in jax.devices()), reason="no GPU")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
import cases  # noqa: E402

from kernelax import kernel  # noqa: E402


@pytest.mark.parametrize("case", [c for c in cases.CASES if not c.xfail], ids=lambda c: c.name)
def test_matches_xla(case):
    args = [np.asarray(a) for a in case.make(np.random.default_rng(0), 1000)]
    args = [a.astype(np.float32) if a.dtype.kind == "f" else a.astype(np.int32) for a in args]
    ref = jax.jit(jax.vmap(case.f, case.in_axes), backend="cpu")(*args)
    out = jax.jit(jax.vmap(kernel(case.f), case.in_axes))(*args)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(ref)):
        if np.asarray(b).dtype.kind in "iu":
            assert np.mean(np.asarray(a) != np.asarray(b)) <= 0.01
            continue
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4 * max(1.0, float(np.max(np.abs(b[np.isfinite(b)]), initial=0))))



def test_rounding_of_integers():
    """floor, ceil, trunc and rint of integers (integer variables in the kernel: a conversion to an integer of a value
    that already is one, as in evosax) compile and are exact."""
    from kernelax import dag, gpu
    k = dag.mk("in", 0, 0)  # (an int32 entry per sample: an integer node; mk, so that nothing folds)
    outs = [np.array([dag.mk("tof32", dag.mk(op, k))], dtype=object) for op in ("floor", "ceil", "trunc", "rint")]
    src, block, _ = gpu.emit([np.array([k], dtype=object)], outs, [True], None, (4, ["i"], ["f"] * 4))
    assert all(f"{op}(" not in src.split("kernelax_k(")[1] for op in ("floor", "ceil", "trunc", "rint"))  # (integers: no call)
    gpu.build(src)  # (nvcc: trunc(int) does not compile for the device)


_DENSE = {  # products of one uniform matrix: merged into one loop where independent (kernelax.gpu, frontend._dense)
    "one": lambda W, x, y: W @ jnp.tanh(x),
    "independent": lambda W, x, y: (W @ jnp.tanh(x), W @ jnp.sin(y)),
    "dependent": lambda W, x, y: W @ jnp.tanh(W @ x) + (W @ y).sum(-1),
    "chain": lambda W, x, y: (W @ jnp.tanh(W @ jnp.tanh(x)), W @ y),
    "transposed": lambda W, x, y: (jnp.tanh(x) @ W, jnp.einsum("kt,jk->tj", jnp.sin(y), W)),
    "laplacian": lambda W, x, y: jnp.trace(jax.hessian(lambda v: jnp.sum(jnp.tanh(W @ jnp.tanh(W @ v))))(x)),  # (of the
}  # second derivatives' columns only the diagonal's used: the others not computed)
_LOOPS = {"one": 1, "independent": 1, "dependent": 2, "chain": 2, "transposed": 2, "laplacian": 4}


@pytest.mark.parametrize("loops", [False, True], ids=["unrolled", "loops"])
@pytest.mark.parametrize("name", list(_DENSE))
def test_dense_layers(name, loops, monkeypatch):
    """Dense layers (a uniform matrix times per-sample values): unrolled, or as loops over the rows where the unrolled
    code would not fit the registers (forced here: LIVE = 4)."""
    from kernelax import gpu
    monkeypatch.setattr(gpu, "LIVE", 4 if loops else 160)
    r = np.random.default_rng(0)
    W, x, y = (r.standard_normal(s).astype(np.float32) for s in ((16, 16), (512, 16), (512, 16, 3)))
    f = lambda *a: _DENSE[name](*a)  # (a new function per test: its own kernel)
    out = jax.jit(jax.vmap(kernel(f), (None, 0, 0)))(W, x, y)
    assert gpu.BUILT[-1].source.count('_Pragma("unroll 1")') == (_LOOPS[name] if loops else 0)
    ref = jax.jit(jax.vmap(f, (None, 0, 0)), backend="cpu")(W, x, y)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(ref)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4 * float(np.abs(np.where(np.isfinite(b), b, 0)).max(initial=0)))


def test_short_scan_rolled():
    """A short scan (10 steps) whose steps unrolled would exceed gpu.SHORT_NODES stays a loop (blackjax' NUTS steps)."""
    from kernelax import gpu
    def f(x):
        def step(c, _):
            for _ in range(600):
                c = jnp.sin(c) * 1.01 + 0.1
            return c, c.sum()
        return jax.lax.scan(step, x, None, length=10)
    x = np.random.default_rng(0).uniform(-1, 1, (512, 2)).astype(np.float32)
    out = jax.jit(jax.vmap(kernel(f)))(x)
    assert "for (;;)" in gpu.BUILT[-1].source
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(jax.jit(jax.vmap(f), backend="cpu")(x))):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reverse", [False, True])
def test_rolled_over_keys(reverse):
    """A long scan over keys of random.split (gymnax' rollout): rolled (gpu.ROLL_XS), each key computed at its step
    (frontend._Split, gpu.LAZY_KEYS), not all of them before the loop."""
    from kernelax import gpu
    f = lambda key, x: jax.lax.scan(lambda c, k: (c * 0.99 + jax.random.normal(k, dtype=x.dtype), None), x,
                                    jax.random.split(key, 1000), reverse=reverse)[0]
    keys = jax.random.split(jax.random.PRNGKey(0), 512)
    x = np.random.default_rng(0).uniform(-1, 1, 512).astype(np.float32)
    out = jax.jit(jax.vmap(kernel(f)))(keys, x)
    src = gpu.BUILT[-1].source
    assert "for (;;)" in src and src.count("\n") < 2000  # (all keys in advance: 237 000 lines)
    np.testing.assert_allclose(np.asarray(out), np.asarray(jax.jit(jax.vmap(f), backend="cpu")(keys, x)), rtol=1e-4, atol=1e-4)


def test_rolled_over_integers():
    """A long scan over integers computed in the kernel: rolled (gpu.ROLL_XS), the integers read at the step from an
    array per thread (integer picks, gpu.PICK_INTS)."""
    import re
    from kernelax import gpu

    def f(s, x):
        xs = (jnp.arange(1000, dtype=jnp.uint32) * s) ^ (s >> 3)
        def step(c, k):
            for j in range(12):
                c = jnp.sin(c * (1.0 + j / 16) + (k % (j + 7)).astype(x.dtype) * 1e-3)
            return c, None
        return jax.lax.scan(step, x, xs)[0]
    s = jnp.asarray(np.random.default_rng(1).integers(0, 2 ** 31, 512), jnp.uint32)
    x = np.random.default_rng(0).uniform(-1, 1, 512).astype(np.float32)
    out = jax.jit(jax.vmap(kernel(f)))(s, x)
    src = gpu.BUILT[-1].source
    assert "for (;;)" in src and re.search(r"long long A\d+\[1000\];", src)
    np.testing.assert_allclose(np.asarray(out), np.asarray(jax.jit(jax.vmap(f), backend="cpu")(s, x)), rtol=1e-4, atol=1e-4)


def test_rolled_over_int32_in_int_arrays():
    """Integer picks of 32-bit signed values (or narrower) read from an array of int per thread, not long long: half
    the local memory (jumanji's Tetris: 65 arrays, 5.5 of its 7.3 KB of stack per thread). uint32 values stay in long
    long (test_rolled_over_integers)."""
    import re
    from kernelax import gpu

    def f(s, x):
        xs = (jnp.arange(1000, dtype=jnp.int32) * s) ^ (s >> 3)
        def step(c, k):
            for j in range(12):
                c = jnp.sin(c * (1.0 + j / 16) + (k % (j + 7)).astype(x.dtype) * 1e-3)
            return c, None
        return jax.lax.scan(step, x, xs)[0]
    s = jnp.asarray(np.random.default_rng(2).integers(-2 ** 31, 2 ** 31, 512), jnp.int32)
    x = np.random.default_rng(0).uniform(-1, 1, 512).astype(np.float32)
    out = jax.jit(jax.vmap(kernel(f)))(s, x)
    src = gpu.BUILT[-1].source
    assert "for (;;)" in src and re.search(r"\bint A\d+\[1000\];", src), re.findall(r"\w+ A\d+\[\d+\];", src)
    np.testing.assert_allclose(np.asarray(out), np.asarray(jax.jit(jax.vmap(f), backend="cpu")(s, x)), rtol=1e-4, atol=1e-4)


def test_picks_among_few_as_chains():
    """A pick among up to 16 options (gpu.ARRAY_MIN) is a chain of selects in registers, not an array per thread in
    local memory, which it fills and reads at each pick: jumanji's Tetris (65 arrays of 10 options) 60.1 -> 11.3 ms
    (XLA 49.9), Snake 7.5 -> 5.4, xland-minigrid 3.35 -> 2.52."""
    import re
    from kernelax import gpu

    def f(i, x):
        return jnp.stack([jnp.sin(x * (k + 1)) for k in range(10)])[i % 10]
    i, x = np.arange(301, dtype=np.int32), np.linspace(-1, 1, 301).astype(np.float32)
    out = jax.jit(jax.vmap(kernel(f)))(i, x)
    src = gpu.BUILT[-1].source
    assert not re.search(r"\w+ A\d+\[10\];", src), re.findall(r"\w+ A\d+\[\d+\];", src)
    np.testing.assert_allclose(np.asarray(out), np.asarray(jax.jit(jax.vmap(f), backend="cpu")(i, x)), rtol=1e-5, atol=1e-5)


def test_spills_around_chains_of_selects(monkeypatch):
    """A pick among few options (a chain of selects, at most gpu.ARRAY_MIN) in straight-line code that spills (_spill):
    a value reloaded for the chain stays visible after it. Before, the chain was a block in braces that _parse did not
    see as one: the reload was declared inside it, and a later use did not compile (xland-minigrid with ARRAY_MIN 16)."""
    from kernelax import gpu
    monkeypatch.setattr(gpu, "LIVE", 4)

    def f(i, x):
        v = [jnp.sin(x * (k + 1)) for k in range(6)]
        a = jnp.stack(v)[i % 6]  # (six options: a chain)
        b = jnp.stack(v[::-1])[(i + 1) % 6]
        return a * b + sum(v)
    i, x = np.arange(301, dtype=np.int32), np.linspace(-1, 1, 301).astype(np.float32)
    out = jax.jit(jax.vmap(kernel(f)))(i, x)
    src = gpu.BUILT[-1].source
    assert "spld<" in src, "no spills: the test does not test the chains next to them"
    np.testing.assert_allclose(np.asarray(out), np.asarray(jax.jit(jax.vmap(f), backend="cpu")(i, x)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("name", ["horner", "kepler", "transcendental", "rk4"])
def test_16_bit_floats(name, dtype):
    """float16 / bfloat16 arguments and results read and written as such by the GPU kernel, each equation's value
    rounded by the hardware's conversion (frontend.ROUND16), against the C kernels (float32 in memory, rounding by
    special.round_float): within an ulp of the format."""
    case = next(c for c in cases.CASES if c.name == name)
    dt = jnp.dtype(dtype)
    args = [np.asarray(a) for a in case.make(np.random.default_rng(0), 1000)]
    args = [jnp.asarray(a, dt) if a.dtype.kind == "f" else jnp.asarray(a, jnp.int32) for a in args]
    g = jax.vmap(kernel(case.f), case.in_axes)
    out = jax.jit(g)(*args)
    ref = jax.jit(g, backend="cpu")(*[jax.device_put(a, jax.devices("cpu")[0]) for a in args])
    eps = {"float16": 2.0 ** -10, "bfloat16": 2.0 ** -7}[dtype]
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(ref)):
        assert a.dtype == b.dtype
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        np.testing.assert_allclose(a, b, rtol=eps, atol=eps * max(1.0, float(np.max(np.abs(b[np.isfinite(b)]), initial=0))))


@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
def test_binomial(x64):
    """jax.random.binomial and multinomial per sample, exactly as XLA: the loops run while i < finfo.max (an integer
    compared with a float beyond int64: compared as doubles, not wrapped to 0), keys in the loops' carries (their bits
    as doubles, also in a float32 kernel), multinomial's keys of random.split lazily read by its scan."""
    jax.config.update("jax_enable_x64", x64)
    try:
        key = lambda v: jax.random.fold_in(jax.random.key(0), (v[0] * 1e3).astype(jnp.int32))
        fs = [lambda v: jax.random.binomial(key(v), 5.0, 0.3),  # (inversion)
              lambda v: jax.random.binomial(key(v), 50.0, v[1] / 3),  # (BTRS)
              lambda v: jax.random.multinomial(key(v), 5.0, jnp.array([0.2, 0.3, 0.5]))]
        x = np.random.default_rng(0).uniform(0.2, 2, (37, 4)).astype(np.float64 if x64 else np.float32)
        for f in fs:
            out, ref = jax.jit(jax.vmap(kernel(f)))(x), jax.jit(jax.vmap(f), backend="cpu")(x)
            np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
    finally:
        jax.config.update("jax_enable_x64", True)


def test_nvcc_timeout_in_index(tmp_path, monkeypatch):
    """An nvcc beyond KERNELAX_NVCC_TIMEOUT: NotImplementedError (the region stays in XLA), entered in the index with its
    limit; the second process tries again, after two the kernel is known as too large at once, until the limit is
    raised (as the C kernels' gcc timeout)."""
    from kernelax import gpu, cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    c = float(np.random.default_rng().uniform(1, 2))  # (a source not compiled before: nvcc runs)
    f, ex, mem = (lambda x: jnp.sin(x) * c), [np.zeros(3, np.float32)], (4, ["f"], ["f"])
    index = gpu._index(f, ex, [True], None, 4, False)
    monkeypatch.setattr(gpu, "NVCC_TIMEOUT", 0.01)
    for tries in (1, 2, 2):  # (the third request fails from the index, without nvcc)
        with pytest.raises(NotImplementedError, match="nvcc took more"):
            gpu.Kernel(f, ex, [True], None, mem)
        assert ck._read(index)["tries"] == tries
    assert gpu.indexed(f, ex, [True], None, 4)
    monkeypatch.setattr(gpu, "NVCC_TIMEOUT", 120.0)  # a larger limit: tried again
    assert not gpu.indexed(f, ex, [True], None, 4)
    k = gpu.Kernel(f, ex, [True], None, mem)
    assert "error" not in ck._read(index) and k.cubins


def test_live_words():
    """live_words: the most 32-bit words live at once, a loop's own peak added to the values live around it."""
    from kernelax import gpu
    src = """extern "C" __global__ void kernelax_k(long long n) {
  const double t1 = 1.0;
  const float t2 = 2.0f;
  double c3_0 = t1;
  for (;;) {
    const double t4 = c3_0 * t1;
    const double t5 = t4 + 1.0;
    if (!(t5 < 9.0)) break;
    c3_0 = t5 * t4;
  }
  const float t6 = t2 + (float)c3_0;
}"""
    assert gpu.live_words(src) == 2 + 1 + 2 + 2 + 2  # (t1, t2, the carry; in the loop t4 and t5)


def test_refused_before_nvcc(tmp_path, monkeypatch):
    """A kernel with more values live at once than MAX_LIVE (32-bit words): NotImplementedError before nvcc runs (the
    region stays in XLA), final in the index."""
    from kernelax import gpu, cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    monkeypatch.setattr(gpu, "MAX_LIVE", 16)
    monkeypatch.setattr(gpu, "build", lambda *a: pytest.fail("nvcc ran"))
    c = np.arange(1.0, 65.0, dtype=np.float32)
    f, ex, mem = (lambda x: jnp.sum(jnp.sin(x[0] * c) * jnp.cos(x[1] * c)[::-1])), [np.zeros(2, np.float32)], (4, ["f"], ["f"])
    for _ in range(2):  # (the second from the index)
        with pytest.raises(NotImplementedError, match="live at once"):
            gpu.Kernel(f, ex, [True], None, mem)
    assert gpu.indexed(f, ex, [True], None, 4)


@pytest.mark.parametrize("logdet", ["slogdet", "det"])
def test_map_axis_kernels(tmp_path, monkeypatch, logdet):
    """An element tangent (jacfwd of a residual summed over quadrature points, log J by slogdet: its derivative's LU
    solves fold the directions into their right-hand sides, _unfold) whose point loop would carry more accumulators
    than MAP_CARRIED: the element's function
    partitioned along the tangent directions (_Mapped), kernels over elements x directions, the rest batched over
    the elements and planned on its own; the same values as XLA. With log(det(F)), whose derivative the kernels take
    through the cofactors, its LU solve only in XLA's own program (counted by the cost model: 3000 rows without it
    stayed in XLA, 4.6 ms instead of 0.69 with kernels), the same."""
    from kernelax import auto, jit, cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    monkeypatch.setattr(auto, "MAP_CARRIED", 100)
    r = np.random.default_rng(36)
    E, Q = 3000, 4
    u, dphi, w = 0.05 * r.standard_normal((E, 8, 3)), r.standard_normal((Q, 8, 3)), r.uniform(size=Q)

    def psi(F):
        C, lJ = F.T @ F, jnp.linalg.slogdet(F)[1] if logdet == "slogdet" else jnp.log(jnp.linalg.det(F))
        return 0.5 * (jnp.trace(C) - 3) - lJ + 0.5 * lJ ** 2 + 0.1 * jnp.sum(C ** 2)
    residual = lambda ue: jnp.sum(jax.vmap(lambda d, wq: wq * jax.grad(lambda ue: psi(jnp.eye(3) + ue.T @ d))(ue))(dphi, w), 0)
    f = lambda u: jax.vmap(jax.jacfwd(residual))(u)
    g = jit(f, build_first=True)
    out = g(u)
    rep = g.report()[0]
    assert rep["status"] == "trial" and rep["regions"][0]["mapped"]["M"] == 24, rep
    ref = jax.jit(f)(u)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-10, atol=1e-10 * np.abs(np.where(np.isfinite(np.asarray(ref)), np.asarray(ref), 0)).max(initial=0))


def test_donation(monkeypatch):
    """donate_argnums on the GPU (kernelax.jit): the program with kernels takes the donated buffer (deleted after the call;
    no trials: XLA's program of the trials keeps it)."""
    from kernelax import auto, jit
    monkeypatch.setattr(auto, "TRIAL_MAX_CALL", 0.0)
    x = jax.device_put(np.random.default_rng(41).standard_normal((4096, 3)), jax.devices("gpu")[0])
    f = lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda c: c[0] < 8, lambda c: (c[0] + 1, jnp.sin(c[1]) + r), (0, r))[1])(x)
    g, ref = jit(f, donate_argnums=0, build_first=True), jax.jit(f)(x)
    for _ in range(3):
        a = jnp.array(x)
        np.testing.assert_allclose(np.asarray(g(a)), np.asarray(ref), rtol=1e-12, atol=1e-12)
        assert a.is_deleted() and g.report()[0]["status"] in ("trial", "kernels"), g.report()
    assert g.report()[0]["status"] == "kernels", g.report()


def test_kernel_failing_after_the_program_compiled(tmp_path, monkeypatch):
    """A new program compiles while its kernels' nvcc runs (their keys known after the emit); a kernel whose nvcc then
    fails (here a timeout) leaves the program compiled again without it, the values XLA's."""
    from kernelax import jit, gpu, cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    monkeypatch.setattr(gpu, "NVCC_TIMEOUT", 0.01)
    c = float(np.random.default_rng().uniform(1, 2))  # (a source not compiled before: nvcc runs)
    f = lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda y: y < 4.0, lambda y: y * 1.1 + r[0] * c, r[1]))(x)
    x = np.random.default_rng(38).uniform(0.5, 1.5, (2000, 3))
    g = jit(f, build_first=True)
    np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-12)
    rep = g.report()[0]
    assert rep["status"] == "xla" and any("nvcc took more" in e for e in rep["errors"]), rep


def test_outputs_written_where_ready():
    """An unrolled scan's ys written per step (each entry of an output written directly where its operands are
    ready), not all at the end: no spills (before, 2.7 KB per thread at 400 steps; at 6000, 85 KB, more local memory
    than the driver could reserve next to XLA's preallocation: the launch failed)."""
    from kernelax import gpu
    x = np.random.default_rng(2).uniform(-1, 1, (37, 4))
    f = lambda v: jax.lax.scan(lambda c, a: (c * 0.5 + a, c * a), v[0], jnp.tile(v, 100))
    out = jax.jit(jax.vmap(kernel(f)))(x)
    assert gpu.BUILT[-1].resources()[1] == 0
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(jax.jit(jax.vmap(f), backend="cpu")(x))):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


def test_local_memory_beyond_the_free(tmp_path, monkeypatch):
    """A kernel whose local memory (a per-sample array read at a dynamic index in a loop) the GPU could not reserve
    for all the threads it holds next to XLA's memory: LocalMemory (the region stays in XLA) instead of a launch that
    fails (out of memory), not noted in the index (another process may have more free); with the memory there, run."""
    from kernelax import gpu, cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    f = lambda v: jax.lax.fori_loop(0, 50, lambda i, c: c * 0.9 + jnp.sin(v[0] * jnp.arange(64.0))[(i * 7 + (c > 0).astype(int)) % 64], 0.0)
    x = np.random.default_rng(0).uniform(-1, 1, (300, 2))
    monkeypatch.setattr(gpu, "_RESERVED", [-2 ** 60])  # (as if nothing were free)
    with pytest.raises(NotImplementedError, match="local memory per thread"):
        jax.jit(jax.vmap(kernel(f)))(x)
    monkeypatch.setattr(gpu, "_RESERVED", [0])
    out = jax.jit(jax.vmap(kernel(f)))(x)
    assert gpu._stack(gpu.BUILT[-1].cubins[0]) > 0
    np.testing.assert_allclose(np.asarray(out), np.asarray(jax.jit(jax.vmap(f), backend="cpu")(x)), rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tables_with_inf_and_nan(dtype):
    """A constant table read at a sample's index with infinite and NaN entries: its __device__ initializer of constant
    expressions (nvcc refuses the bit casts there: dynamic initialization), the same values as XLA's."""
    from kernelax import gpu
    c = np.array([1.5, np.inf, -np.inf, np.nan, -2.0, 0.25] * 4, dtype)
    f = lambda i, x: jnp.asarray(c)[i] * x + 1.0
    i = np.random.default_rng(0).integers(0, len(c), 500).astype(np.int32)
    x = np.random.default_rng(1).uniform(0.5, 2, 500).astype(dtype)
    with jax.enable_x64(dtype == np.float64):
        out = jax.jit(jax.vmap(kernel(f)))(i, x)
        ref = jax.jit(jax.vmap(f), backend="cpu")(i, x)
    assert "TAB0" in pathlib.Path(gpu.BUILT[-1].cubins[0]).with_suffix(".cu").read_text()
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_no_runtime_without_kernels(tmp_path, monkeypatch):
    """A program kernelax.jit plans no kernels for (elementwise work, XLA's): no GPU runtime loaded (in a new cache nvcc
    would build it first, ~4 s)."""
    from kernelax import jit, gpu, cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    monkeypatch.setattr(gpu, "runtime", lambda: pytest.fail("runtime loaded"))
    f = lambda x: jnp.sin(x) * 2.0 + 1.0
    x = np.random.default_rng(39).uniform(size=5000)
    g = jit(f, build_first=True)
    np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-12)
    assert g.report()[0]["status"] == "xla"


@pytest.mark.parametrize("loop", [False, True])
def test_platform_dependent_on_the_gpu(loop, tmp_path, monkeypatch):
    """jax.lax.platform_dependent with a branch for the GPU (a library's CUDA kernel; here a host callback): kernelax.jit's
    kernels take the default branch (fused with its neighbours), also one with loops."""
    from kernelax import jit, cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    calls = []
    host_f = (lambda a: np.sin(np.sin(np.sin(a)))) if loop else (lambda a: np.sin(a) * 2.0)
    impl = (lambda v: jax.lax.fori_loop(0, 3, lambda i, y: jnp.sin(y), v)) if loop else (lambda v: jnp.sin(v) * 2.0)

    def f(x):
        host = lambda v: jax.pure_callback(lambda a: calls.append(1) or host_f(a), jax.ShapeDtypeStruct(v.shape, v.dtype), v,
                                           vmap_method="broadcast_all")
        return jnp.cos(jax.lax.platform_dependent(x, cuda=host, default=impl) * x) + x
    x = np.random.default_rng(44).standard_normal((1 << 16, 64))
    g = jit(f, build_first=True)
    np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-12, atol=1e-14)
    calls.clear()
    g(x)
    assert not calls, g.report()[0]


def test_other_architectures(tmp_path):
    """The kernels compile for every NVIDIA architecture the installed nvcc knows (device code: no such GPU needed;
    run and tuned on an L40S only)."""
    import subprocess
    from kernelax import gpu
    from kernelax.build import thread_map
    archs = [a.replace("compute_", "sm_") for a in subprocess.run([gpu.NVCC, "--list-gpu-arch"], capture_output=True, text=True,
                                                                   check=True).stdout.split()]
    for name in ("neo_hooke", "kepler", "interp", "mlp"):  # (built here, or by the tests before: then 24 of all built)
        case = next(c for c in cases.CASES if c.name == name)
        args = [np.asarray(a, np.float32) for a in case.make(np.random.default_rng(0), 300)]
        jax.block_until_ready(jax.jit(jax.vmap(kernel(case.f), case.in_axes))(*args))
    built = list(dict.fromkeys(k.source for k in gpu.BUILT))
    sources = {f"k{i}": s for i, s in enumerate(built[::-(-len(built) // 24)])}
    assert sources and len(archs) > 1
    for k, s in sources.items():
        (tmp_path / f"{k}.cu").write_text(s)

    def nvcc(job):
        k, a = job
        r = subprocess.run([gpu.NVCC, "-O3", f"-arch={a}", "-cubin", "-o", str(tmp_path / f"{k}_{a}.cubin"), str(tmp_path / f"{k}.cu")],
                           capture_output=True, text=True)
        return None if r.returncode == 0 else f"{k} {a}: {r.stderr[-300:]}"
    errors = [e for e in thread_map(nvcc, [(k, a) for k in sources for a in archs], 16) if e]
    assert not errors, errors


def test_scatter_positions_beyond_the_array():
    """A sum over the samples into a shared array at sample-dependent positions (the cotangent of a table read in fill
    mode): positions beyond the array are dropped, as on the CPU and in XLA (not added past its end)."""
    T, i = jnp.arange(1.0, 9.0, dtype=jnp.float32), jnp.array([0, 3, 8, 8, 2, 100, 7, 3] * 40, jnp.int32)
    g = lambda T, f: jnp.sum(jax.vmap(f, (0, None))(i, T) ** 2)
    read = lambda i, T: T.at[i].get(mode="fill", fill_value=0.0)
    out = jax.jit(jax.grad(lambda T: g(T, kernel(read))))(T)
    ref = jax.jit(jax.grad(lambda T: g(T, read)), backend="cpu")(T)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-5)


def test_uint32_beyond_int32():
    """uint32 values from 2^31 on in a 4-byte kernel (whose integers are int32 otherwise): exact."""
    x = jnp.array([4_000_000_000, 5, 2 ** 31, 2 ** 32 - 1, 3_000_000_001, 0, 2 ** 31 - 1] * 20, jnp.uint32)
    f = lambda x: (x.astype(jnp.float32), (x > jnp.uint32(3_000_000_000)).astype(jnp.float32), x // jnp.uint32(3))
    out = jax.jit(jax.vmap(kernel(f)))(x)
    ref = jax.jit(jax.vmap(f), backend="cpu")(x)
    for a, b in zip(out, ref):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_conversions_zeros_and_two_loops():
    """float32 to int32 / uint32 saturated, uint32 from conversions beyond int32, remainders keeping the sign of zero,
    and two while loops side by side (their carries were typed together: a key's bits next to a float32 rounded, 99.9 %
    of the gamma samples wrong), against XLA on the CPU."""
    sys.path.insert(0, os.path.dirname(__file__))
    from test_export import POINT_I, POINT_X, _point_cases
    n = 1000
    rng = np.random.default_rng(0)
    x = np.concatenate([POINT_X, rng.uniform(-1e10, 1e10, n - len(POINT_X))]).astype(np.float32)
    i = np.concatenate([POINT_I, rng.integers(-2 ** 31, 2 ** 31, n - len(POINT_I))]).astype(np.int32)
    s, a = np.arange(n, dtype=np.uint32) * 7 + 3, rng.uniform(0.2, 5, n).astype(np.float32)
    out = jax.jit(jax.vmap(kernel(_point_cases)))(x, i, s, a)
    ref = jax.jit(jax.vmap(_point_cases), backend="cpu")(x, i, s, a)
    for k, (o, r) in enumerate(zip(out, ref)):
        o, r = np.asarray(o), np.asarray(r)
        if r.dtype.kind in "iu":
            np.testing.assert_array_equal(o, r, err_msg=str(k))
        elif k >= 8:  # (samples: a rounding apart can take another branch of the rejection)
            assert np.mean(~np.isclose(o, r, rtol=1e-4)) <= 0.01, k
        else:
            np.testing.assert_array_equal(o, r, err_msg=str(k))
            np.testing.assert_array_equal(np.signbit(o)[~np.isnan(r)], np.signbit(r)[~np.isnan(r)], err_msg=str(k))


def test_integer_ranges():
    """The GPU kernels' integers by value ranges (test_export's point code cases): no long long beyond 2^53, float64
    intermediates of float32 programs as doubles, int64 ones not cut to int, constant int32 outputs exact, positions
    near INT32_MAX, a clipped NaN; against XLA on the CPU."""
    sys.path.insert(0, os.path.dirname(__file__))
    from test_export import range_cases
    for name, f, args, x64 in range_cases():
        with jax.enable_x64(x64):
            out = jax.jit(jax.vmap(kernel(f)))(*args)
            ref = jax.jit(jax.vmap(f), backend="cpu")(*args)
        for o, r in zip(jax.tree.leaves(out), jax.tree.leaves(ref)):
            np.testing.assert_array_equal(np.asarray(o), np.asarray(r), err_msg=name)


def test_variant_choice_at_the_cost_of_the_calls():
    """A kernel with variants (here a second copy of its cubin) chooses one at its first call with TUNE_MIN samples or
    more. A first call longer than gpu.TUNE_LONG is not repeated for it (before: every variant twice on all samples,
    then the chosen one again, 2 V + 1 times the call): the next calls run the other variants, one each, then variant 0
    again, each call launching once, and the choice follows the last. A short call times the variants within: 2 V + 1
    launches."""
    import ctypes
    from kernelax import gpu
    rt = gpu.runtime()
    x = np.random.default_rng(3).uniform(-1, 1, (8192, 3)).astype(np.float32)
    for long in (True, False):
        c = float(np.random.default_rng().uniform(1, 2))  # (a source not compiled before)
        f = lambda r, c=c: jnp.sin(r * c) * r + c
        g = jax.jit(jax.vmap(kernel(f)))
        g(x[:64])  # (built and registered; fewer samples than TUNE_MIN: no choice)
        k = gpu.BUILT[-1]
        data = pathlib.Path(k.cubins[0]).read_bytes()
        rt.kernelax_gpu_register(k.key, data, len(data), k.blocks[0], len(k.zero), (ctypes.c_int * max(1, len(k.zero)))(*k.zero), -1)
        rt.kernelax_gpu_tuning(0.0 if long else 1e9)
        try:
            runs, ref = ctypes.c_longlong(), jax.jit(jax.vmap(f))(x)
            for i in range(3 if long else 1):
                np.testing.assert_allclose(np.asarray(g(x)), np.asarray(ref), rtol=2e-6, atol=2e-6)
                work = rt.kernelax_gpu_work(k.key, ctypes.byref(runs))
                if long:
                    assert runs.value == i + 1 and work == len(x) * (i + 1), (i, runs.value, work)
                    assert (rt.kernelax_gpu_choice(k.key) < 0) == (i < 2)
            if not long:
                assert runs.value == 5 and rt.kernelax_gpu_choice(k.key) >= 0
        finally:
            rt.kernelax_gpu_tuning(gpu.TUNE_LONG)


def test_guarded_branches():
    """A select's operand that needs much work nothing else needs (an environment's reset where its episode ended, the
    untaken branch of lax.cond) is computed in a block only threads taking it run, also in loops (dag.guards); the
    results as before, NaN of the guarded work in threads that do not take it included."""
    import re
    from kernelax import gpu

    def heavy(x):
        for k in range(100):
            x = jnp.sin(x * 1.1 + k)
        return x

    def f(x):
        c = x > 0.9
        h = heavy(jnp.log(x - 0.95))  # (NaN below 0.95: threads that do not take it)
        y = jax.lax.while_loop(lambda v: v[0] < 2.0, lambda v: (v[0] + 0.5 + v[1], jnp.where(c, heavy(v[1] * v[0]), v[1])),
                               (x, x))[1]
        return (jnp.where(c, h, x * 2.0), jax.lax.cond(x < 0.05, lambda: heavy(x * 3.0), lambda: -x), y,
                jnp.where(c, heavy(y * 3.0), y))
    x = np.random.default_rng(88).uniform(0.0, 1.0, 999).astype(np.float32)
    out = jax.jit(jax.vmap(kernel(f)))(x)
    src = gpu.BUILT[-1].source
    assert len(re.findall(r"\n  +(?:float|double) g\d+ = ", src)) >= 1 and "for (;;)" in src  # (in the loop)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(jax.jit(jax.vmap(f), backend="cpu")(x))):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4)


def test_threefry_in_32_bit_words():
    """jax.random's threefry2x32 per sample as one node (dag.threefry), computed in 32-bit words (kx_threefry); the
    bits as jax.vmap's."""
    from kernelax import gpu

    def f(seed, x):
        k1, k2 = jax.random.split(jax.random.fold_in(jax.random.key(0), seed))
        return jax.random.uniform(k1, (5,)) * x, jax.random.randint(k2, (3,), 0, 1000), jax.random.bits(k2, (2,))
    seed, x = np.arange(301, dtype=np.int32), np.linspace(0.5, 1.5, 301).astype(np.float32)
    out = jax.jit(jax.vmap(kernel(f)))(seed, x)
    assert "kx_threefry(" in gpu.BUILT[-1].source.split("kernelax_k(")[1]
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(jax.jit(jax.vmap(f), backend="cpu")(seed, x))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
