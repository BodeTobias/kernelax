# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

import collections
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from cases import CASES  # noqa: E402

from kernelax import kernel, jit  # noqa: E402
from kernelax import cpu as ck  # noqa: E402
from kernelax.vmap import cvmap  # noqa: E402

BY_NAME = {c.name: c for c in CASES}


def close(a, b, rtol=1e-10):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        y = np.asarray(y, np.float64)
        np.testing.assert_allclose(np.asarray(x, np.float64), y, rtol=rtol, atol=rtol * (1 + np.abs(np.where(np.isfinite(y), y, 0)).max(initial=0)))


@pytest.mark.parametrize("case", [pytest.param(c, id=c.name, marks=[pytest.mark.xfail(strict=True, reason=c.xfail)]
                                               if c.xfail else []) for c in CASES])
def test_jit_vmap(case, monkeypatch):
    """jax.jit(jax.vmap(kernel(f))) against jax.vmap(f): the batch folded into the kernel's sample axis."""
    monkeypatch.setattr(ck, "FIX_K", 1)
    args = case.make(np.random.default_rng(0), 37)
    close(jax.jit(jax.vmap(kernel(case.f), case.in_axes))(*args), jax.vmap(case.f, case.in_axes)(*args))


@pytest.mark.parametrize("name", ["black_scholes", "lennard_jones", "log_strain", "cholesky4", "solve6", "neo_hooke",
                                  "quaternion", "mlp", "bilinear"])
def test_derivatives(name, monkeypatch):
    """Forward and reverse derivatives (reverse: of a sum over the batch, also with respect to the inputs
    that are the same for all samples, whose cotangents are summed), Hessian of a scalar output."""
    case = BY_NAME[name]
    args = case.make(np.random.default_rng(1), 13)
    monkeypatch.setattr(ck, "FIX_K", 1)
    g = kernel(case.f)
    first = lambda h: lambda *a: jax.tree.leaves(h(*a))[0]
    for h in (jax.jacfwd, jax.jacrev):
        close(jax.vmap(h(first(g)), case.in_axes)(*args), jax.vmap(h(first(case.f)), case.in_axes)(*args))
    loss = lambda h: lambda *a: sum(jnp.sum(jnp.sin(x)) for x in jax.tree.leaves(jax.vmap(h, case.in_axes)(*a)))
    every = tuple(range(len(args)))
    close(jax.jit(jax.grad(loss(g), every))(*args), jax.grad(loss(case.f), every)(*args))
    scalar = lambda h: lambda *a: jnp.sum(first(h)(*a))
    close(jax.vmap(jax.hessian(scalar(g)), case.in_axes)(*args), jax.vmap(jax.hessian(scalar(case.f)), case.in_axes)(*args))


def test_while_forward(monkeypatch):
    """Forward derivatives through while loops (reverse mode of a while loop is not defined in JAX either)."""
    monkeypatch.setattr(ck, "FIX_K", 1)
    for name in ("j2", "kepler", "newton_linesearch"):
        case = BY_NAME[name]
        args = case.make(np.random.default_rng(2), 21)
        first = lambda h: lambda *a: jax.tree.leaves(h(*a))[0]
        close(jax.vmap(jax.jacfwd(first(kernel(case.f))), case.in_axes)(*args),
              jax.vmap(jax.jacfwd(first(case.f)), case.in_axes)(*args), rtol=1e-8)


def test_single_and_nested(monkeypatch):
    """A single call (one sample), nested vmaps, vmap over an argument that is first the same for all."""
    f = lambda x, c: {"y": jnp.sin(x) * c[0] + x ** 2, "s": jnp.sum(jnp.exp(-x)) * c[1], "i": jnp.argmax(x)}
    monkeypatch.setattr(ck, "FIX_K", 2)
    g = kernel(f)
    x, c = np.random.default_rng(3).standard_normal((30, 3)), np.array([2.0, 3.0])
    close(g(x[0], c), f(x[0], c))
    assert g(x[0], c)["i"].dtype == f(x[0], c)["i"].dtype
    close(jax.vmap(jax.vmap(g, (0, None)), (0, None))(x.reshape(5, 6, 3), c),
          jax.vmap(jax.vmap(f, (0, None)), (0, None))(x.reshape(5, 6, 3), c))
    cs = np.random.default_rng(4).standard_normal((4, 2))
    close(jax.vmap(jax.vmap(g, (0, None)), (None, 0))(x, cs), jax.vmap(jax.vmap(f, (0, None)), (None, 0))(x, cs))


def test_without_x64(monkeypatch, tmp_path):
    """Without x64: float32 kernels of 4-byte arrays, 16 lanes on AVX-512 and 8 on AVX2 (W=4), also in kernelax.jit."""
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))  # (a program measured before, with other threads: trials again)
    jax.config.update("jax_enable_x64", False)
    try:
        x = np.linspace(0.1, 3, 1003, dtype=np.float32)
        for W in (None, 4):
            monkeypatch.setattr(ck, "FIX_W", W)
            y = jax.vmap(kernel(jnp.sin))(x)
            assert y.dtype == jnp.float32
            np.testing.assert_allclose(y, jax.vmap(jnp.sin)(x), rtol=2e-6)  # (libmvec's sinf: ulps from XLA's)
        monkeypatch.setattr(ck, "FIX_W", None)
        f = lambda x: jnp.sin(x) * jnp.exp(-x)
        g = jit(f, build_first=True)
        np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=2e-6)
        assert g.report()[0]["status"] == "trial"  # (ran with its kernels; trials from the next calls)
    finally:
        jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("K", [3, None])
def test_reduced_cotangents(K, monkeypatch):
    """Gradients with respect to arguments shared by all samples: summed inside the kernel (map-reduce), with a
    partial last block (padding lanes masked out) and, for K=None, the choice of K on >= 4096 samples."""
    case = BY_NAME["mlp"]
    args = case.make(np.random.default_rng(5), 5003)
    loss = lambda h: lambda *a: jnp.sum(jnp.cos(jax.vmap(h, case.in_axes)(*a)[0]))
    every = tuple(range(len(args)))
    monkeypatch.setattr(ck, "FIX_K", K)
    close(jax.jit(jax.grad(loss(kernel(case.f)), every))(*args), jax.grad(loss(case.f), every)(*args), rtol=1e-9)


def test_options(monkeypatch):
    """Options checked when a function is made (ValueError): n_threads, cpu.FIX_K and FIX_W (tests and benchmarks
    set these; no K or W in the API). The threads of a call by default (cpu.NT): OMP_NUM_THREADS (its first entry),
    at most the CPUs of the process's affinity."""
    for bad in (0, 1.5):
        with pytest.raises(ValueError, match="n_threads"):
            kernel(jnp.sin, n_threads=bad)
        with pytest.raises(ValueError, match="n_threads"):
            jit(jnp.sin, n_threads=bad)
    for name, bad in (("FIX_K", 0), ("FIX_W", 16), ("FIX_W", 3)):
        monkeypatch.setattr(ck, name, bad)
        with pytest.raises(ValueError, match=name):
            kernel(jnp.sin)
        with pytest.raises(ValueError, match=name):
            jit(jnp.sin)
        monkeypatch.setattr(ck, name, None)
    cpus = len(os.sched_getaffinity(0))
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert ck._threads() == cpus
    for env, nt in (("1", 1), ("2,1", min(2, cpus)), (f" {cpus + 5} ", cpus)):
        monkeypatch.setenv("OMP_NUM_THREADS", env)
        assert ck._threads() == nt
    for env in ("0", "x", "-2"):
        monkeypatch.setenv("OMP_NUM_THREADS", env)
        with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
            ck._threads()


def test_keyword_and_static_arguments(monkeypatch):
    """Keyword arguments (under jax.vmap mapped along their leading axis, as there), static_argnums and static_argnames
    as in jax.jit (either way; a Python int in range()), derivatives through them; one trace per signature; a static
    argument traced by an enclosing jax.jit, or not hashable: ValueError."""
    monkeypatch.setattr(ck, "FIX_K", 1)
    calls = []

    def f(x, n, *, scale, shift=0.0):
        calls.append(1)
        return sum(jnp.sin(x * scale) ** k for k in range(n)) + shift
    r = np.random.default_rng(6)
    x, s = r.standard_normal((37, 3)), r.uniform(0.5, 1.5, 37)
    ref = lambda n, shift, s: jax.vmap(lambda x, s: f(x, n, scale=s, shift=shift))(x, s)
    for g in (kernel(f, static_argnums=1, static_argnames="shift"), kernel(f, static_argnames=("n", "shift"))):
        close(jax.vmap(lambda x, s: g(x, 3, scale=s, shift=0.5))(x, s), ref(3, 0.5, s))
        close(jax.jit(lambda x, s: jax.vmap(g, (0, None))(x, 4, scale=s))(x, s), ref(4, 0.0, s))  # (n: unmapped, as is)
        want = ref(3, 0.5, s)
        before = len(calls)
        close(jax.vmap(lambda x, s: g(x, 3, scale=s, shift=0.5))(x, s), want)
        assert len(calls) == before  # (the Spec of this signature: cached, no new trace)
        close(g(x[0], 2, scale=1.5), f(x[0], 2, scale=1.5))
        loss = lambda h: lambda s: jnp.sum(jax.vmap(h, (0, None))(x, 3, scale=s) ** 2)
        close(jax.grad(loss(g))(s), jax.grad(loss(f))(s))
    g = kernel(f, static_argnums=1)
    with pytest.raises(ValueError, match="static argument is traced"):
        jax.jit(lambda x, n: jax.vmap(g, (0, None))(x, n, scale=s))(x, 3)
    with pytest.raises(ValueError, match="hashable"):
        kernel(lambda x, c: x * c[0], static_argnums=1)(x[0], [2.0])


def test_builds_locked(tmp_path):
    """Two processes building the same new kernels at once (MPI ranks): each source goes through gcc once
    (build.locked), the other process waits and loads it."""
    import subprocess
    log, cc = tmp_path / "gcc.log", tmp_path / "cc.sh"
    cc.write_text(f'#!/bin/sh\necho "$@" >> {log}\nexec gcc "$@"\n')
    cc.chmod(0o755)
    script = tmp_path / "rank.py"
    script.write_text("""import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from kernelax import kernel
x = np.linspace(0, 1, 64 * 3).reshape(64, 3)
print(float(jax.vmap(kernel(lambda v: jnp.sin(v) * jnp.exp(v[0]) + 0.123456))(x).sum()))
""")
    env = dict(os.environ, KERNELAX_CACHE_DIR=str(tmp_path / "cache"), KERNELAX_CC=str(cc), JAX_PLATFORMS="cpu")
    procs = [subprocess.Popen([sys.executable, str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True) for _ in range(2)]
    outs = [p.communicate(timeout=600) for p in procs]
    assert all(p.returncode == 0 for p in procs) and outs[0][0] == outs[1][0], outs
    built = [a[a.index("-o") + 1].rsplit(".", 2)[0] for a in (line.split() for line in log.read_text().splitlines()) if "-o" in a]
    assert built and len(built) == len(set(built)), built  # (outputs: path.<pid>.<thread>)


def test_jax_internals_in_one_place():
    """JAX's internals (jax._src) are used in kernelax/internals.py only, each with no public equivalent; JAX's legacy
    ClosedJaxpr (merged into Jaxpr since JAX 0.11, its shims to be removed) nowhere."""
    d = os.path.dirname(ck.__file__)
    src = {f: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d)) if f.endswith(".py")}
    assert [f for f, s in src.items() if f != "internals.py" and "jax._src" in s] == []
    assert [f for f, s in src.items() if "ClosedJaxpr" in s] == []


def test_size_limit():
    """A function that unrolls into too many scalar operations stops with the place that causes it."""
    def big(a, b):
        return jnp.tanh(a @ b)  # 64 x 64 x 64 multiplications per sample
    a = np.ones((3, 64, 64))
    with pytest.raises(NotImplementedError, match=r"more than \d+ scalar operations, here at 'dot_general' \(.*test_kernel.py"):
        jax.vmap(kernel(big))(a, a)


def test_floating_point_mode():
    """The kernel's threads take the caller's floating-point mode: XLA's (subnormals flushed) through kernelax, IEEE
    through cvmap, whichever thread created OpenMP's threads."""
    x = np.full(20011, 1e-310)
    via_xla = np.asarray(jax.jit(jax.vmap(kernel(jnp.tanh)))(x))
    np.testing.assert_array_equal(via_xla, jax.jit(jnp.tanh)(x))
    np.testing.assert_array_equal(np.ravel(cvmap(jnp.tanh, [x[0]], (0,))(x)), x)


def test_tiled_sums():
    """A loop over a counter whose other carries only accumulate (a quadrature: per point shared work, then the
    contributions to many sums) runs in tiles: the shared values once per step, then a loop per tile of sums. Rows of
    a table at the counter (the same in all lanes) are loads, not gathers."""
    from kernelax import emit
    w = np.random.default_rng(0).standard_normal((20, 8))

    def f(x):
        def body(c):
            h = jnp.tanh(x * jnp.asarray(w)[c[0]] + jnp.sin(x[0] * c[0]))
            return c[0] + 1, c[1] + jnp.outer(h, h * x[1])
        return jax.lax.while_loop(lambda c: c[0] < 20, body, (0, jnp.zeros((8, 8))))[1]
    x = np.random.default_rng(1).standard_normal((1003, 8))
    tiled, orig = [], emit._tiles
    emit._tiles = lambda L, o: (lambda r: (tiled.append(r is not None), r)[1])(orig(L, o))
    try:
        c = cvmap(f, (x[0],), (0,))
        close(c(x), jax.jit(jax.vmap(f))(x))
    finally:
        emit._tiles = orig
    assert any(tiled) and "lookup1_d(TAB" in c.kernel.sources[0] and "lookup_d(TAB" not in c.kernel.sources[0]


def test_wide_integers():
    """64-bit integers in and out of kernels by their bits: exact also beyond 2^53 (random bits, as in jax.random)."""
    x = np.random.default_rng(0).integers(0, 2 ** 63, (64, 2), dtype=np.uint64)
    for g in (lambda b: b * jnp.uint64(6364136223846793005) + jnp.uint64(1),
              lambda b: (jax.lax.shift_right_logical(b, jnp.uint64(12)) | jnp.uint64(0x3FF0000000000000)).view(jnp.float64)):
        a, r = cvmap(g, (x[0],), (0,))(x), jax.vmap(g)(x)
        assert a.dtype == r.dtype and np.array_equal(a, r)


def test_long_scan_as_loop():
    """A long scan with xs and ys small next to its body runs as a loop (the kernel holds its body once): xs read
    and ys written at the step, also in reverse; with ys as large as the body unrolled."""
    from kernelax import dag, frontend
    x = np.random.default_rng(2).uniform(-1, 1, (37, 4))
    body = lambda c, a: (sum(jnp.sin(c + a * k * 0.01) for k in range(1, 300)) * 0.5 / 300, c * a)  # (contracting)
    small = lambda c, a: (c * 0.5 + a, c * a)
    for body, rev, n, loop in ((body, False, 10, True), (body, True, 10, True), (small, False, 1500, False)):
        f = lambda v: jax.lax.scan(body, v[0], jnp.tile(v, n), reverse=rev)
        _, outs, _ = frontend.trace(f, x[0], uniform=[False])
        roots = [n for o in outs for n in np.asarray(o, dtype=object).flat if isinstance(n, dag.Node)]
        assert any(n.op == "while" for n in dag.schedule(roots)) == loop
        close(cvmap(f, (x[0],), (0,))(x), jax.vmap(f)(x))


def test_semantics_edges():
    """As jax.vmap: NaN in max/min/argmax/sort/sign, sums at sample positions returned per sample (also out of
    cond, scan and while), scatters with several data-dependent indices (in order), eigh of 2 x 2 matrices,
    x / 0, fmod near multiples, float32 rounding (reduce_precision), loops of accumulators that return their
    counter and run to a non-integral bound; complex numbers (sqrt of negative numbers, inf and NaN too)."""
    r = np.random.default_rng(5)
    x = r.normal(size=(37, 4))
    x[0, 0], x[1, 2], x[3] = np.nan, np.nan, [np.inf, -np.inf, 2.0, np.nan]
    A = r.normal(size=(37, 2, 2))
    A = A + A.transpose(0, 2, 1)
    acc = lambda v: jax.lax.while_loop(lambda c: c[0] < 32.5, lambda c: (c[0] + 1, c[1] + jnp.outer(jnp.tile(v, 2), v) * c[0]), (0, jnp.zeros((8, 4))))
    cases = [
        (lambda v: (jnp.maximum(v[0], v[1]), jnp.max(v), jnp.argmax(v), jnp.argmin(v), jnp.sort(v), jnp.sign(v),
                    jax.lax.cummax(v), jax.nn.relu(v)), x),
        (lambda v: (jnp.zeros(5).at[(jnp.abs(v[0]) * 3).astype(int) % 5].add(v[1]),
                    jax.lax.cond(v[2] > 0, lambda: jnp.zeros(3).at[(v[3] > 0).astype(int)].add(1.0), lambda: jnp.ones(3)),
                    jax.lax.scan(lambda c, a: (c, jnp.zeros(3).at[(a > 0).astype(int)].add(a)), 0.0, v)[1],
                    jnp.zeros(6).at[jnp.argsort(v)].set(jnp.arange(4.0)),
                    jnp.searchsorted(jnp.linspace(-2, 2, 5), v, method="sort")), np.nan_to_num(x)),
        (lambda a: (jnp.linalg.eigvalsh(a), a / 0.0, jnp.fmod(a, 0.3), jnp.mod(a * 10, 0.1)), A),
        (lambda v: jax.lax.reduce_precision(v * 1.1, 8, 23), np.nan_to_num(x)),
        (lambda v: acc(v[:4]), np.nan_to_num(x)),
    ]
    for f, a in cases:
        for o, q in zip(jax.tree.leaves(cvmap(f, (a[0],), (0,))(a)), jax.tree.leaves(jax.vmap(f)(a))):
            q = np.asarray(q, np.float64)
            np.testing.assert_allclose(np.asarray(o, np.float64), q, rtol=1e-12, atol=1e-12 * (1 + np.nanmax(np.abs(np.where(np.isfinite(q), q, 0)))), equal_nan=True)
    for f in (lambda v: jnp.abs(jnp.sqrt(v + 0j)), lambda v: jnp.sqrt(v + 0j)):
        np.testing.assert_allclose(cvmap(f, (x[0],), (0,))(x), jax.vmap(f)(x), rtol=1e-12, equal_nan=True)


def test_keep_nans():
    """By default x * 0, x - x and 0 / x fold to 0 (documented deviation); keep_nans keeps them, NaN and inf propagate
    as in XLA (also under kernelax.jit)."""
    from kernelax import jit
    f = lambda x: (x * 0.0, x - x, 0.0 / x, jax.grad(lambda y: jnp.sum(y[0] * y))(x))
    x = np.array([[1.0, np.nan, np.inf], [-2.0, 0.0, -np.inf]] * 600)
    ref = [np.asarray(r) for r in jax.vmap(f)(x)]
    out = cvmap(f, (x[0],), (0,))(x)
    assert not all(np.array_equal(o, r, equal_nan=True) for o, r in zip(out, ref))  # (the folded zeros)
    for o, r in zip(cvmap(f, (x[0],), (0,), keep_nans=True)(x), ref):
        np.testing.assert_array_equal(o, r)
    J = jit(lambda x: jax.vmap(f)(x), keep_nans=True)
    J(x)
    J.wait()
    for o, r in zip(J(x), ref):
        np.testing.assert_array_equal(o, r)


def test_large_scatter_targets():
    """Scatter-adds into large arrays: the operand is added as it is (no matrix over its unit vectors), and a symbolic
    index into a large per-sample array is refused (NotImplementedError: the region stays in XLA) instead of binding
    a candidate per position (both took memory quadratic in the size: an FE assembly in a time step) or comparing
    the entry with each position (beyond frontend.SCATTER_MAX)."""
    f = lambda x: jnp.zeros(40000).at[jnp.array([3, 7])].add(x[:2]).sum() + x[2]
    x = np.random.default_rng(6).standard_normal((50, 3))
    np.testing.assert_allclose(cvmap(f, (x[0],), (0,))(x), jax.vmap(f)(x), rtol=1e-12)
    g = lambda x: jnp.zeros(40000).at[(jnp.abs(x[0]) * 9).astype(int)].add(x[1])
    with pytest.raises(NotImplementedError, match="too large"):
        cvmap(g, (x[0],), (0,))


def test_sums_with_inputs_in_memory():
    """A sum over the samples stays one (an output of the kernel's reduction, not a per-sample array) also where the
    function reads an input in memory and where it lies in a cond with a constant index (an FE tangent inside a
    time-stepping loop: before, the sum became a dense array per sample)."""
    from kernelax import frontend
    t = np.arange(100.0)
    f = lambda t, i: jax.lax.cond(True, lambda: jnp.zeros(50000).at[i[0]].add(t[i[1]]), lambda: jnp.zeros(50000))
    _, outs, _ = frontend.trace(f, t, np.array([3, 4], np.int32), uniform=[True, False], reduce=[True])
    assert isinstance(outs[0], frontend._Scatter)


def test_loops_over_inputs_in_memory():
    """scan, while and cond whose only operand not constant is an input in memory (frontend._Mem): traced as symbolic,
    not bound by JAX on the lazy value."""
    t = jnp.linspace(0, 1, 100)
    fs = [lambda x, t: jax.lax.scan(lambda c, i: (c * 0.5 + t[i], c), 0.0, jnp.arange(5))[1] + x[0],
          lambda x, t: jax.lax.while_loop(lambda c: c[0] < 5, lambda c: (c[0] + 1, c[1] + t[c[0]]), (0, 0.0))[1] + x[0],
          lambda x, t: jax.lax.cond(True, lambda: t[3] * 2, lambda: t[4]) + x[0]]
    x = np.random.default_rng(0).uniform(0, 1, (5, 2))
    for f in fs:
        close(cvmap(f, (x[0], t), (0, None))(x, t), jax.vmap(f, (0, None))(x, t))


def test_rows_of_inputs_in_memory_at_sample_indices():
    """A row of an input in memory at a per-sample index (jnp.take: e.g. an element's data, read by its kernel over
    elements x directions), then indexed at a loop's counter: one load per entry read (_Mem with a symbolic offset),
    not the row's entries picked per counter value; also of bool inputs (masks)."""
    from kernelax import frontend, gpu
    r = np.random.default_rng(35)
    W = r.standard_normal((500, 27, 9))
    mask = W[..., 0] > 0

    def f(x, W, mask):
        n = jnp.clip(x[0].astype(jnp.int32), 0, 499)
        Wn, mn = jnp.take(W, n, axis=0), jnp.take(mask, n, axis=0)
        body = lambda c: (c[0] + 1, c[1] + jnp.where(mn[c[0]], jnp.sum(Wn[c[0]]) * x[1], 1.0))
        return jax.lax.while_loop(lambda c: c[0] < 27, body, (0, 0.0))[1]
    x = np.stack([r.uniform(0, 499, 64), r.standard_normal(64)], 1)
    close(cvmap(f, (x[0], W, mask), (0, None, None))(x, W, mask), jax.vmap(f, (0, None, None))(x, W, mask))
    _, outs, _ = frontend.trace(f, x[0], W, mask, uniform=[False, True, True])
    ops = {n.op for n in gpu._all_nodes([x for o in outs for x in frontend._flat(o) if isinstance(x, frontend.Node)])}
    assert "load" in ops and not ops & {"pick", "arr"}


def test_gcc_timeout_in_index(tmp_path, monkeypatch):
    """A gcc timeout is entered in the index with its limit: the second process tries again (one may come from a busy
    machine), after two the kernel is known as too large at once (the error says where it comes from), until the limit
    is raised (KERNELAX_GCC_TIMEOUT)."""
    from kernelax import build, cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    c = float(np.random.default_rng().uniform(1, 2))  # (a source not compiled before: gcc runs)
    f, ex = (lambda x: jnp.sin(x) * c), [np.zeros(3)]
    calls = []
    orig = build.compile_c.__wrapped__

    def slow(src, *a, **k):
        calls.append(1)
        raise build.GccTimeout("kernelax: gcc took more than the limit (test)")
    monkeypatch.setattr(ck, "compile_c", slow)
    monkeypatch.setattr(build, "GCC_TIMEOUT", 1.0)
    for expect in (1, 2, 2):  # (gcc calls so far: the third request fails from the index, and says so)
        ck._KERNELS.clear()
        with pytest.raises(NotImplementedError, match="gcc took more") as e:
            ck.kernel(f, ex, [True], K=1, cached=True)
        assert len(calls) == expect
    assert "noted in the index by an earlier call" in str(e.value) and ".json to try again" in str(e.value)
    monkeypatch.setattr(build, "GCC_TIMEOUT", 5.0)  # a larger limit: tried again
    monkeypatch.setattr(ck, "compile_c", lambda src, *a, **k: orig(src, *a, **k))
    k = ck.kernel(f, ex, [True], K=1, cached=True)
    assert k.Ks == [1]


def test_cpu_seconds_of_a_process_tree():
    """A background gcc runs at idle priority, which a busy program may starve for minutes: its timeout counts the CPU
    seconds of the process and its descendants (build._cpu_seconds; reaped children included), not wall time."""
    import subprocess, sys, time
    from kernelax import build
    busy = "import time\nt = time.process_time()\nwhile time.process_time() - t < 0.6: pass"
    p = subprocess.Popen(["sh", "-c", f"{sys.executable} -c '{busy}'; sleep 60"])
    time.sleep(0.3)
    during = build._cpu_seconds(p.pid)  # (the child still running, perhaps not yet on a CPU: a busy machine)
    end = time.monotonic() + 60
    while build._cpu_seconds(p.pid) < 0.6 and time.monotonic() < end:  # (until the child is done and reaped by sh)
        time.sleep(0.1)
    time.sleep(0.3)
    after = build._cpu_seconds(p.pid)  # (sh sleeping: its reaped child's seconds)
    p.kill()
    p.wait()
    assert 0.0 <= during < 0.6 and 0.55 < after < 1.0, (during, after)
    assert build._cpu_seconds(p.pid) == 0.0  # (gone)


def test_background_gcc_timeouts(monkeypatch):
    """A background gcc's limit counts its CPU seconds. One that does not get the CPU (here sleep) times out as starved
    after BACKGROUND_WALL times the limit, one that uses it (a busy loop) as too slow."""
    import sys
    from kernelax import build
    monkeypatch.setattr(build, "GCC_TIMEOUT", 0.5)
    build.BACKGROUND.on = True
    try:  # (the busy loop gets a wide window: at idle priority on a busy machine its 0.5 s may take long)
        for cmd, starved, wall in ((["sleep", "30"], True, 4), ([sys.executable, "-c", "while True: pass"], False, 120)):
            monkeypatch.setattr(build, "BACKGROUND_WALL", wall)
            with pytest.raises(build.GccTimeout) as e:
                build._gcc(cmd[0], cmd[1:], 1)
            assert e.value.starved == starved
    finally:
        build.BACKGROUND.on = False


def test_starved_variants_not_in_the_index(tmp_path, monkeypatch):
    """Variants whose background gcc was starved by the program leave the kernel out of the index (else later
    processes would load K = 1 only, for good); a later process builds them."""
    from kernelax import build, cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    c = float(np.random.default_rng().uniform(1, 2))  # (a source not compiled before)
    f, ex = (lambda x: jnp.sin(x) * c + jnp.cos(x * c)), [np.zeros(3)]
    orig = ck.compile_c

    def starved(src, *a, **k):
        raise build.GccTimeout("kernelax: starved (test)", starved=True)
    ck._KERNELS.clear()
    k = ck.kernel(f, ex, [True], cached=True, first=True)
    assert k.Ks == [1] and k.rest[0]
    monkeypatch.setattr(ck, "compile_c", starved)
    k.complete()
    assert k.Ks == [1] and ck._read(k.index) is None
    monkeypatch.setattr(ck, "compile_c", orig)
    ck._KERNELS.clear()
    k = ck.kernel(f, ex, [True], cached=True, first=True)
    k.complete()
    assert len(k.Ks) > 1 and ck._read(k.index)["Ks"] == k.Ks


def test_cond_with_large_operands_in_memory():
    """A cond passes inputs in memory (_Mem) into its branches as they are, as calls do: only what a branch reads
    becomes nodes (before, all their entries did: beyond MAX_NODES for an FE solver's state)."""
    u = np.arange(400000.0)
    f = lambda x, u: jax.lax.cond(u[0] >= 0, lambda x, u: x * u[5], lambda x, u: x, x, u)
    x = np.random.default_rng(13).standard_normal((100, 3))
    np.testing.assert_allclose(cvmap(f, (x[0], u), (0, None))(x, u), jax.vmap(f, (0, None))(x, u), rtol=1e-14)


def test_svd_edges():
    """svd by one-sided Jacobi: tall, wide and square (full_matrices) matrices, rank deficiency (zero singular values,
    pinv), the derivative of the singular values; against jax.vmap (u, vt compared through products: their signs
    are free)."""
    r = np.random.default_rng(12)
    for shape in ((5, 3), (3, 5), (4, 4)):
        a = r.standard_normal((300,) + shape)
        f = lambda a: (lambda u, s, vt: (s, (u * s) @ vt, u.T @ u, vt @ vt.T))(*jnp.linalg.svd(a, full_matrices=shape[0] == shape[1]))
        close(cvmap(f, (a[0],), (0,))(a), jax.vmap(f)(a))
    x, y = r.standard_normal((300, 4)), r.standard_normal((300, 3))
    low = np.einsum("ni,nj->nij", x, y) + np.einsum("ni,nj->nij", y[:, :1] * x[:, ::-1], y[:, ::-1])  # rank 2
    g = lambda a: (jnp.linalg.svd(a, compute_uv=False), jnp.linalg.pinv(a), jnp.linalg.matrix_rank(a))
    out, ref = cvmap(g, (low[0],), (0,))(low), jax.vmap(g)(low)
    np.testing.assert_allclose(out[0], ref[0], atol=1e-10)
    np.testing.assert_allclose(out[1], ref[1], rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(out[2], ref[2])
    h = jax.grad(lambda a: jnp.sum(jnp.linalg.svd(a, compute_uv=False) ** 1.5))
    a = r.standard_normal((300, 4, 3))
    close(cvmap(h, (a[0],), (0,))(a), jax.vmap(h)(a))


def test_narrow_floats():
    """float16 and bfloat16 values are rounded to their format at every equation (computed in float32): to nearest
    even, subnormals kept, overflow to inf, NaN kept; reduce_precision as XLA (subnormals of the format flushed),
    also of float64. Bit-equal to jax.vmap."""
    r = np.random.default_rng(13)
    edge = np.array([0.0, -0.0, 1e-6, -2.5e-7, 3e-8, 6.0e-5, 6.1e-5, 65519.0, 65520.0, 1e5, 1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11,
                     np.nan, np.inf, -np.inf, 1e-39, 3e-41, 0.1])
    x = np.concatenate([edge, r.standard_normal(200) * 10.0 ** r.uniform(-9, 6, 200)])
    y = np.roll(x, 7)
    f = lambda x, y: ((x.astype(jnp.float16) * y.astype(jnp.float16) + 1.5).astype(jnp.float32),
                      (x.astype(jnp.bfloat16) / 3 + y.astype(jnp.bfloat16)).astype(jnp.float32),
                      jax.lax.reduce_precision(x.astype(jnp.float32) * 1.1, 5, 10), jax.lax.reduce_precision(x * 1.1, 5, 10),
                      jax.lax.reduce_precision(x * 1.1, 8, 7), jax.lax.reduce_precision(x * 1.1, 8, 23))
    for o, ref in zip(cvmap(f, (x[0], y[0]), (0, 0))(x, y), jax.vmap(f)(x, y)):
        np.testing.assert_array_equal(np.asarray(o), np.asarray(ref))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_16_bit_bits_and_expansions(dtype):
    """16-bit floats: bitcasts to and from (u)int16 (signbit, copysign, nextafter, ldexp go through them) bit-equal to
    jax.vmap, also at zeros, subnormals, inf and NaN; primitives kernelax expands (lgamma in gamma, erf_inv) computed in
    float32 and rounded once, as XLA (rounded at each step, gamma was 12 ulps off): within an ulp of the format."""
    dt, r = jnp.dtype(dtype), np.random.default_rng(14)
    edge = np.array([0.0, -0.0, 1e-7, -6e-8, 6.0e-5, 6.1e-5, 65504.0, 1.0, -2.5, np.inf, -np.inf, np.nan, 3e-5, 1e-3, 0.3, 7.0])
    x = np.concatenate([edge, r.standard_normal(48) * 10.0 ** r.uniform(-6, 4, 48)]).reshape(-1, 4).astype(dt)
    y, i = np.roll(x, 3), r.integers(-32768, 32767, x.shape).astype(np.int16)
    bits = lambda a, b, i: (jnp.signbit(a), jnp.copysign(a, b), jnp.nextafter(a, b), jnp.ldexp(a, (i % 5).astype(jnp.int32)),
                            jax.lax.bitcast_convert_type(a, jnp.int16), jax.lax.bitcast_convert_type(b, jnp.uint16),
                            jax.lax.bitcast_convert_type(i, dt))  # (times 2: XLA flushes subnormal results)
    for o, ref in zip(cvmap(bits, (x[0], y[0], i[0]), (0, 0, 0))(x, y, i), jax.vmap(bits)(x, y, i)):
        np.testing.assert_array_equal(np.asarray(o).astype(np.float32), np.asarray(ref).astype(np.float32))
    expanded = lambda a: (jax.scipy.special.gamma(jnp.abs(a) % 6 + 0.1), jax.scipy.special.erfinv(jnp.tanh(a)))
    for o, ref in zip(cvmap(expanded, (x[0],), (0,))(x), jax.vmap(expanded)(x)):
        np.testing.assert_allclose(np.asarray(o, np.float64), np.asarray(ref, np.float64), rtol=float(jnp.finfo(dt).eps), atol=0)


def test_float8_refused():
    """Computing in float8 (formats other than float16 / bfloat16 / float32 / float64) is refused with the reason
    (kernelax.jit: the region stays in XLA), not done unrounded; float8 and bfloat16 inputs are read as floats (numpy does
    not know them as such: before, cvmap read them as integers)."""
    x = np.random.default_rng(0).uniform(-2, 2, (64, 4)).astype(jnp.float8_e4m3fn)
    with pytest.raises(NotImplementedError, match="float8_e4m3fn"):
        cvmap(lambda v: jnp.sin(v) * v, (x[0],), (0,))(x)
    with pytest.raises(NotImplementedError, match="float8_e5m2"):  # (fake quantization: rounded on the way)
        cvmap(lambda v: (v.astype(jnp.float32) * 1.1).astype(jnp.float8_e5m2).astype(jnp.float32), (x[0],), (0,))(x)
    f = lambda v: v.astype(jnp.float32) * 1.5 + v[::-1].astype(jnp.float32)  # (dequantized)
    np.testing.assert_array_equal(cvmap(f, (x[0],), (0,))(x), jax.vmap(f)(x))
    b = x.astype(jnp.bfloat16)
    g = lambda v: v * 2 + 1
    np.testing.assert_array_equal(np.asarray(cvmap(g, (b[0],), (0,))(b), np.float32), np.asarray(jax.vmap(g)(b), np.float32))


def same(a, b, rtol=1e-12):  # (NaN and inf at the same places, the rest relatively close)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(np.asarray(x, np.float64), np.asarray(y, np.float64), rtol=rtol, atol=0, equal_nan=True)


def test_nan_by_intent():
    """x - x and 0 * x of a choice between constants, one of them infinite, stay operations also without keep_nans:
    such code makes its NaN on purpose (JAX 0.12.1's gamma: inf - inf at its poles and at NaN; it gave 0)."""
    x = np.array([np.nan, -2.0, -1.0, 0.0, 0.5, 3.0, 171.5, np.inf, -np.inf])
    f = lambda x: (lambda c: (c - c, 0.0 * c))(jnp.where(x > 1, jnp.inf, 1.0))
    same(cvmap(f, (x[0],), (0,))(x), jax.vmap(f)(x))
    for dt in (np.float64, np.float32):
        g, xd = jax.scipy.special.gamma, x.astype(dt)
        same(cvmap(g, (xd[0],), (0,))(xd), jax.vmap(g)(xd), rtol=1e-6 if dt == np.float32 else 1e-12)


def test_special_edges():
    """zeta (poles: inf for even integers x, else NaN; x = 1, x < 1, negative q, q = inf: 0, as XLA from jaxlib 0.12.1),
    polygamma (orders 0 to 4, poles), i1e (0, signs, the switch to the asymptotic expansion at 20, large and infinite
    x) against jax.vmap."""
    xs = np.array([1.0, 0.5, 2.0, 2.0, 2.0, 3.0, 2.5, 2.0, np.inf, 2.0, 50.0, 1.0000001, 2.0, np.nan, 2.0, 4.5, 3.0, 0.0,
                   -2.0, 0.0, 3.0])
    qs = np.array([2.0, 2.0, 0.0, -1.0, -1.5, -1.5, -1.5, np.inf, 2.0, 1e-8, 1.0, 3.0, 1e10, 1.0, -0.0, 0.3, 0.0, -1.0,
                   -2.0, 2.0, -np.inf])
    same(cvmap(jax.scipy.special.zeta, (xs[0], qs[0]), (0, 0))(xs, qs), jax.vmap(jax.scipy.special.zeta)(xs, qs))
    x = np.array([0.5, 2.0, -1.5, 0.0, -2.0, 7.3, 1e-3])
    for m in range(5):
        g = lambda x, m=m: jax.scipy.special.polygamma(m, x)
        same(cvmap(g, (x[0],), (0,))(x), jax.vmap(g)(x))
    x = np.array([0.0, -0.0, 1e-300, -3.0, 19.999, 20.0, 20.001, -45.0, 700.0, 1e5, -np.inf, np.inf, np.nan])
    same(cvmap(jax.scipy.special.i1e, (x[0],), (0,))(x), jax.vmap(jax.scipy.special.i1e)(x))


def _chain(x, w, c):  # straight code (a long chain per entry), a loop, a sum over the samples
    y = x
    for i in range(12):
        y = jnp.tanh(y * w[i % 3] + jnp.roll(y, 1)) + (c if i == 11 else 0.1 * i)
    z = jax.lax.while_loop(lambda s: s[0] < 5, lambda s: (s[0] + 1, s[1] * 0.9 + jnp.sin(s[1])), (0, y[0]))[1]
    return y, z, jnp.sum(y)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_kernel_in_parts(tmp_path, monkeypatch, dtype):
    """A large kernel's block in parts, a C file each (emit._parts): the same results (float64 and float32 lanes,
    loops, sums over the samples), and a change near the end of the function compiles the parts after it only (the
    others' object files from the cache: their text does not depend on the rest)."""
    from kernelax import build, emit
    monkeypatch.setattr(emit, "SPLIT_MIN", 0)
    monkeypatch.setattr(emit, "PART", 3000)
    monkeypatch.setattr(emit, "FRAME_MAX", 1.0)
    monkeypatch.setattr(build, "CACHE", str(tmp_path))
    x = np.random.default_rng(5).standard_normal((517, 6)).astype(dtype)
    w = np.asarray([0.7, -1.1, 0.4], dtype)
    for c in (0.3, 0.6):
        g = cvmap(lambda x, w: _chain(x, w, c), (x[0], w), (0, None), K=1)
        objs = set(os.listdir(tmp_path / "obj"))
        for o, r in zip(g(x, w), jax.vmap(lambda x: _chain(x, w, c))(x)):
            np.testing.assert_allclose(o, r, rtol=2e-5 if dtype == np.float32 else 1e-12, atol=1e-6 if dtype == np.float32 else 1e-12)
        src = open(next(tmp_path.glob("*/kernel.c"))).read() if c == 0.3 else src
        if c == 0.3:
            first = objs
    assert src.count(build.PART_MARK) >= 3
    assert 0 < len(objs - first) < len(objs) / 2  # (most parts reused)


def test_k_chosen_over_long_calls(monkeypatch):
    """A first call of variant 0 longer than cpu.TUNE_LONG is not repeated for the choice of K (it ran variant 0 on all
    samples, all variants three times on 4096 or more, then the chosen one on all again: 2 to 14 times the call): the
    next calls run the other variants, one each, then variant 0 once more (its first time includes the warm-up of the
    first call), and the fastest is chosen after that, each call on its samples once. A short call chooses at once, as
    before."""
    from kernelax import cpu as ck
    x = np.random.default_rng(81).uniform(0.5, 1.5, (ck.TUNE_MIN, 3))
    for long, c in ((True, 0.613), (False, 0.627)):  # (c: kernels not built before)
        monkeypatch.setattr(ck, "TUNE_LONG", 0.0 if long else 1e9)
        f = lambda r, c=c: jnp.sin(r * c) * r + c
        g = cvmap(f, (x[0],), (0,))
        k, ref = g.kernel, jax.vmap(f)(x)
        assert len(k.Ks) > 1
        for i in range(len(k.Ks) + 1 if long else 1):
            close(g(x), ref, rtol=1e-13)
            if long:
                assert k.struct.runs == i + 1 and k.struct.work == len(x) * (i + 1), (i, k.struct.runs, k.struct.work)
                assert (k.K() is None) == (i < len(k.Ks)), (i, k.K())
        if not long:
            assert k.struct.runs >= 3 * len(k.Ks) + 2 and k.K() is not None


def test_k_chosen_at_the_cost_of_one_call(monkeypatch):
    """A first call between TUNE_SLICE and TUNE_LONG compares the variants on slices of its samples that each take
    about TUNE_SLICE (at least 64 per thread), not on TUNE_MIN or more (at n near TUNE_MIN: the whole batch, 3 rounds
    over 4 variants, 14 times the call): variant 0 on all samples, the rounds on the slices, the chosen one on all
    again, about 3 times the samples of the call."""
    from kernelax import cpu as ck
    monkeypatch.setattr(ck, "TUNE_SLICE", 1e-9)  # (every call longer than the slice: slices at their least)
    monkeypatch.setattr(ck, "TUNE_TIME", 0.0)  # (3 rounds)
    x = np.random.default_rng(82).uniform(0.5, 1.5, (ck.TUNE_MIN, 3))
    f = lambda r: jnp.sin(r * 0.641) * r + 0.641  # (a kernel not built before)
    g = cvmap(f, (x[0],), (0,), n_threads=4)
    k = g.kernel
    assert len(k.Ks) == 4
    close(g(x), jax.vmap(f)(x), rtol=1e-13)
    assert k.K() is not None and k.struct.runs == 3 * 4 + 2
    assert k.struct.work == 2 * len(x) + 3 * 4 * 64 * 4, k.struct.work


def test_parts_after_gcc_timeout(monkeypatch):
    """Many reads from the frame keep a block one file (emit.FRAME_MAX); if gcc runs out of time for that file, the
    block is built in parts all the same (a slower kernel instead of none)."""
    from kernelax import build, emit, cpu as ck
    for name, value in (("SPLIT_MIN", 0), ("PART", 3000), ("FRAME_MAX", 0.0)):
        monkeypatch.setattr(emit, name, value)
    x, w = np.random.default_rng(5).standard_normal((517, 6)), np.asarray([0.7, -1.1, 0.4])
    orig = ck.compile_c
    for timeout, c in ((False, 0.41), (True, 0.42)):  # (c: kernels not built before in this process)
        def gcc(src, *a, **k):
            if timeout and build.PART_MARK not in src:
                raise build.GccTimeout("kernelax: gcc took more than the limit (test)")
            return orig(src, *a, **k)
        monkeypatch.setattr(ck, "compile_c", gcc)
        built = len(ck._KERNELS)
        g = cvmap(lambda x, w: _chain(x, w, c), (x[0], w), (0, None), K=1)
        close(g(x, w), jax.vmap(lambda x: _chain(x, w, c))(x), rtol=1e-12)
        assert len(ck._KERNELS) == built + 1
        assert (build.PART_MARK in list(ck._KERNELS.values())[-1].sources[0]) == timeout


def test_batched_scatters(monkeypatch):
    """Scatters with batching dimensions at data-dependent indices (vmap over rows of x.at[i].set / add / max): one
    scatter per row (frontend._unbatched), each entry against its own row's positions (before: the candidates of all
    rows' indices together, exponential in the rows); scatter-adds into zeros by their entries (_scatter_entries:
    the batch position as a fixed coordinate; before: refused)."""
    def f(x):
        a = x[:16].reshape(8, 2)
        row = lambda i, v: (jnp.zeros(6).at[i].set(v), jnp.zeros(6).at[i].add(v), jnp.ones(6).at[i].add(v),
                            jnp.zeros(6).at[i].max(v))
        return jax.vmap(row)((jnp.abs(a) * 7).astype(jnp.int32) % 6, jnp.sin(a))
    x = np.random.default_rng(3).standard_normal((37, 16))
    monkeypatch.setattr(ck, "FIX_K", 1)
    close(jax.jit(jax.vmap(kernel(f)))(x), jax.vmap(f)(x), rtol=1e-14)
    g = lambda x: jnp.zeros(6).at[(jnp.abs(x[:2]) * 7).astype(jnp.int32) % 6].add(x[2:4])  # (an output from a sum
    close(jax.jit(jax.vmap(kernel(g)))(x), jax.vmap(g)(x), rtol=1e-14)  # of entries, made while jax.jit traces:
    # its binds concrete, as all of the frontend's; before: TracerArrayConversionError)


def test_complex_inputs_and_outputs():
    """Complex leaves in and out of cvmap and kernelax (as [re, im] on a trailing axis at the kernel), complex64 too, and
    derivatives through kernelax with complex values."""
    r = np.random.default_rng(43)
    z, x = r.standard_normal((300, 4)) + 1j * r.standard_normal((300, 4)), r.standard_normal((300, 4))
    f = lambda z, x: (jnp.exp(z) * x, jnp.abs(z) + x, jnp.fft.fft(z))
    close = lambda a, b, tol: [np.testing.assert_allclose(np.asarray(o), np.asarray(q), rtol=tol, atol=tol) for o, q in zip(a, b)]
    for dt, tol in ((np.complex128, 1e-12), (np.complex64, 2e-5)):
        zz, xx = z.astype(dt), x.astype(np.float32 if dt == np.complex64 else np.float64)
        out = cvmap(f, (zz[0], xx[0]), (0, 0))(zz, xx)
        assert [o.dtype for o in out] == [np.dtype(dt), xx.dtype, np.dtype(dt)]
        close(out, jax.vmap(f)(zz, xx), tol)
    close(jax.jit(jax.vmap(kernel(f)))(z, x), jax.vmap(f)(z, x), 1e-12)
    h = lambda z, x: jnp.sum(jnp.abs(jnp.exp(z) * x) ** 2)
    close([jax.vmap(jax.grad(kernel(h), argnums=1))(z, x)], [jax.vmap(jax.grad(h, argnums=1))(z, x)], 1e-12)
    jvp = lambda g: jax.vmap(lambda z, x: jax.jvp(g, (z, x), (z * 0.5, x))[1])(z, x)
    close(jvp(kernel(f)), jvp(f), 1e-12)


@pytest.mark.parametrize("R", [12, 4])
def test_input_read_first_in_a_loop_body(R):
    """An input entry read inside a while loop's body and after the loop: its load (transposed groups, or the contiguous
    rows for few entries) at block level, before the loop (emit.value; a reverse-mode FE residual's kernel, whose
    inner-axis sums load inside their loops, did not compile: 'G3_0_0' undeclared)."""
    def f(x):
        c, _ = jax.lax.while_loop(lambda c: c[1] < x[R - 1], lambda c: (c[0] + x[1] * 0.5, c[1] + 1.0), (x[0], 0.0))
        return c + x[1] * x[2] + x[R - 2]
    x = np.random.default_rng(44).uniform(0.0, 1.0, (300, R))
    x[:, R - 1] = np.random.default_rng(45).integers(1, 7, 300)
    np.testing.assert_allclose(cvmap(f, [x[0]], (0,))(x), jax.vmap(f)(x), rtol=1e-13)


def test_empty_loops_and_narrow_edges():
    """Loops of length 0 (scan, fori_loop), reduce_precision of 16-bit floats, argmin of booleans and of the least
    int64: as jax.vmap."""
    x = np.linspace(-1, 1, 12).reshape(4, 3)
    body = lambda c, _: (c * 2.0, c + 1.0)
    f = lambda x: (jax.lax.scan(body, x, None, length=0)[0], jax.lax.scan(body, x, None, length=0)[1].shape[0] + x,
                   jax.lax.fori_loop(0, 0, lambda i, c: c * 3.0, x),
                   jax.lax.reduce_precision(x.astype(jnp.float16), 5, 4).astype(jnp.float32),
                   jnp.argmin(x > 0), jnp.argmin(jnp.array([5, -(2 ** 63)], jnp.int64) + x[0].astype(jnp.int64)))
    close(cvmap(f, [x[0]], (0,))(x), jax.vmap(f)(x), rtol=0)


def test_user_function_named_like_jax_internals():
    """A user's custom_jvp function named exp_neg_sq (the name of JAX's own in erf's derivative) is evaluated as
    written, not replaced."""
    @jax.custom_jvp
    def exp_neg_sq(x):
        return jnp.exp(-x * x) * 7.0
    exp_neg_sq.defjvp(lambda p, t: (exp_neg_sq(p[0]), t[0]))
    x = np.linspace(-1, 1, 9)
    close(cvmap(exp_neg_sq, [x[0]], (0,))(x), jax.vmap(exp_neg_sq)(x), rtol=1e-14)


def test_keys_transposes_and_nested_kernels():
    """Typed PRNG keys in and out of a kernel (as their key data); jax.linear_transpose of a kernel linear in its
    input; a kernel calling another kernel (inlined); an empty inner batch beside an outer-mapped argument; a kernel
    without an array argument: ValueError."""
    keys, x = jax.random.split(jax.random.key(3), 50), np.linspace(0.0, 1.0, 50)
    f = lambda k, v: (jax.random.normal(k, (3,)) + v, jax.random.split(k)[1])
    out, ref = jax.vmap(kernel(f))(keys, x), jax.vmap(f)(keys, x)
    close(out[0], ref[0], rtol=1e-14)
    np.testing.assert_array_equal(jax.random.key_data(out[1]), jax.random.key_data(ref[1]))
    A = np.random.default_rng(46).standard_normal((3, 4))
    mv, xs, ct = (lambda v, A: A @ v), np.zeros((20, 4)), np.random.default_rng(47).standard_normal((20, 3))
    close(jax.linear_transpose(lambda v: jax.vmap(kernel(mv), (0, None))(v, A), xs)(ct),
          jax.linear_transpose(lambda v: jax.vmap(mv, (0, None))(v, A), xs)(ct), rtol=1e-13)
    inner = kernel(lambda v: jnp.sin(v) * 2.0)
    outer = lambda v: inner(v) + jax.vmap(inner)(jnp.stack([v, 2 * v]))[1]
    close(jax.vmap(kernel(outer))(x), jax.vmap(lambda v: jnp.sin(v) * 2.0 + jnp.sin(2 * v) * 2.0)(x), rtol=1e-14)
    c, e = np.ones((3, 2)), np.ones((3, 0, 2))
    assert jax.vmap(jax.vmap(kernel(lambda c, e: c * e), (None, 0)), (0, 0))(c, e).shape == (3, 0, 2)
    with pytest.raises(ValueError, match="array argument"):
        kernel(lambda: 1.0)()


def test_index_entries_cut_short(tmp_path, monkeypatch):
    """An index entry cut short (a crash while writing, a full disk) counts as none: the kernel is built again."""
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    f = lambda v: jnp.sin(v) * 3.25 + v
    ck.kernel(f, [np.zeros(3)], [True], cached=True)
    files = list(tmp_path.glob("*.json"))
    assert files
    for p in files:
        p.write_text('{"key": ')
    ck._KERNELS.clear()
    k = ck.kernel(f, [np.zeros(3)], [True], cached=True)
    x = np.random.default_rng(48).standard_normal((40, 3))
    close(cvmap(f, [x[0]], (0,))(x), jax.vmap(f)(x), rtol=1e-14)
    assert k is not None


def test_cache_directory_kept_private(tmp_path):
    """A cache directory others may write to (its libraries are run) is made private, with a warning."""
    import subprocess
    d = tmp_path / "cache"
    d.mkdir(mode=0o777)
    os.chmod(d, 0o777)
    r = subprocess.run([sys.executable, "-c", "import kernelax.build"], env=dict(os.environ, KERNELAX_CACHE_DIR=str(d)),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "made private" in r.stderr, r.stderr
    assert os.stat(d).st_mode & 0o022 == 0


def test_weak_types_of_outputs():
    """Outputs weakly typed in f (Python scalars, counters of loops) stay so through a kernel, as under jax.vmap (they
    came back strongly typed: a counter added to an int32 made it int64)."""
    x = np.random.default_rng(61).uniform(0.5, 1.5, (50, 3))
    f = lambda r: (jnp.sin(r), 2.0, jax.lax.while_loop(lambda c: c[0][0] < 4.0, lambda c: (c[0] * 1.1 + 0.1, c[1] + 1), (r, 0))[1])
    for o, r in zip(jax.vmap(kernel(f))(x), jax.vmap(f)(x)):
        assert jax.typeof(o) == jax.typeof(r), (jax.typeof(o), jax.typeof(r))
        np.testing.assert_allclose(o, r, rtol=1e-14)


def test_zero_size_outputs():
    """Per-sample outputs and scanned inputs without elements (a[:0], zeros((0, 3))): the kernel computes the others
    (the emitter failed on an empty group of stores)."""
    x = np.random.default_rng(62).uniform(size=(9, 3))
    for f in (lambda a: (jnp.sin(a), a[:0] * 2), lambda a: jax.lax.scan(lambda c, t: (jnp.sin(c) + t, c * 2), a, jnp.zeros((0, 3)))):
        for o, r in zip(jax.tree.leaves(jax.vmap(kernel(f))(x)), jax.tree.leaves(jax.vmap(f)(x))):
            assert o.shape == r.shape
            np.testing.assert_allclose(o, r, rtol=1e-14)


def test_negative_static_argnums_as_jax_jit():
    """Negative static_argnums as jax.jit resolves them: against f's signature (also for an argument passed by
    keyword), not the call's arguments (x was taken as static, or k where jax.jit takes s)."""
    x = np.random.default_rng(63).uniform(size=(30, 3))
    f = lambda x, k: jax.vmap(lambda r: r * k)(x) if k > 0 else x
    for g in (jax.jit, jit):
        np.testing.assert_allclose(g(f, static_argnums=-1)(x, k=2), x * 2)
        np.testing.assert_allclose(g(f, static_argnums=-1)(x, 3), x * 3)
    h = lambda x, k, s=1.0: x * s if k > 0 else x
    for g in (jax.jit, jit):
        with pytest.raises(jax.errors.TracerBoolConversionError):
            g(h, static_argnums=-1)(x, 2)


def test_argmin_of_many_entries():
    """argmin over 1500 entries per sample (a select tree that deep: its leaf count was computed recursively, beyond
    Python's limit)."""
    v = np.random.default_rng(64).uniform(size=(8, 1500))
    np.testing.assert_array_equal(jax.vmap(kernel(jnp.argmin))(v), jax.vmap(jnp.argmin)(v))


def test_scatter_plan_with_positions_from_a_loop():
    """A sum into a large array (>= emit.SCAT_BIG entries) at positions a while loop finds in a shared table (a binary
    search): the cached plan is redone when the table changes (the table, read only in the loop's body, was missing from
    the inputs the plan's hash covers: the second call summed at the first call's positions)."""
    from kernelax import emit
    M = emit.SCAT_BIG

    def f(x, table, edges):
        lo = jax.lax.while_loop(lambda c: c[1] - c[0] > 1, lambda c: (lambda mid, go: (jnp.where(go, mid, c[0]), jnp.where(go, c[1], mid)))(
            (c[0] + c[1]) // 2, edges[(c[0] + c[1]) // 2] <= x), (0, edges.shape[0] - 1))[0]
        return jnp.take(table, lo) * x
    loss = lambda g: lambda t, x, e: jax.vmap(g, (0, None, None))(x, t, e).sum()
    gk, gj = jax.jit(jax.grad(loss(kernel(f)))), jax.jit(jax.grad(loss(f)))
    x, table = np.random.default_rng(65).uniform(0.0, 1.0, 4096), np.ones(M)
    for e in (np.linspace(0.0, 1.0, M), np.linspace(0.0, 1.0, M) ** 2, np.linspace(0.0, 1.0, M)):
        np.testing.assert_allclose(gk(table, x, e), gj(table, x, e), rtol=1e-12, atol=1e-12)


def test_cases_that_failed_to_build():
    """Cases that failed to build or raised: a float32 counter of a tiled sum loop (vf and vd mixed), a dot of
    booleans (masks multiplied), an integer window maximum with padding (int(-inf)), a 64-bit constant shifted right by
    a per-sample amount (a scalar cast to a vector), jax.grad through a kernel of a custom_vjp function (custom_lin
    has no derivative: its JVP is transposed now), a thread count beyond the processors (ValueError, not libgomp); a
    primitive not covered names where it is."""
    x = np.ones((4, 32), np.float32)
    with jax.enable_x64(False):
        f = lambda x: jax.lax.while_loop(lambda c: c[0] < 3, lambda c: (c[0] + 1, c[1] + x * c[0]),
                                         (jnp.float32(0), jnp.zeros(32, jnp.float32)))[1]
        np.testing.assert_allclose(jax.vmap(kernel(f))(x), jax.vmap(f)(x))
    b = np.array([[True, False, True], [False, False, True]])
    np.testing.assert_array_equal(jax.vmap(kernel(lambda b: (jnp.dot(b, b), b @ jnp.eye(3, dtype=bool))))(b)[0],
                                  jax.vmap(lambda b: jnp.dot(b, b))(b))
    v = np.arange(12, dtype=np.int64).reshape(4, 3)
    g = lambda v: jax.lax.reduce_window(v, jnp.iinfo(jnp.int64).min, jax.lax.max, (2,), (1,), "SAME")
    np.testing.assert_array_equal(jax.vmap(kernel(g))(v), jax.vmap(g)(v))
    s = np.array([0, 1, 63], np.uint64)
    h = lambda s: (jax.lax.shift_right_logical(jnp.uint64(2 ** 64 - 1), s), jax.lax.shift_left(jnp.int64(3), s.astype(jnp.int64)))
    for o, r in zip(jax.vmap(kernel(h))(s), jax.vmap(h)(s)):
        np.testing.assert_array_equal(o, r)

    @jax.custom_vjp
    def c(x):
        return jnp.sin(x)
    c.defvjp(lambda x: (jnp.sin(x), x), lambda x, t: (2 * jnp.cos(x) * t,))
    y = np.array([0.1, 0.5, 1.0])
    np.testing.assert_allclose(jax.grad(lambda y: jax.vmap(kernel(c))(y).sum())(y), jax.grad(lambda y: jax.vmap(c)(y).sum())(y),
                               rtol=1e-14)
    with pytest.raises(ValueError, match="n_threads"):
        kernel(jnp.sin, n_threads=10 ** 6)
    with pytest.raises(NotImplementedError, match=r"tridiagonal_solve' is not supported on values computed per sample "
                                                  r"\(at .*test_kernel\.py"):
        jax.vmap(kernel(lambda d: jax.lax.linalg.tridiagonal_solve(d, d, d, d[:, None])))(np.ones((3, 4)))


def test_lowerings_that_grew_quadratically():
    """Lowerings that grew quadratically: the cotangent of T[i] for a shared table T (a candidate array per start of a
    dynamic_update_slice: O(M^2); now a sum at the sample's position; per sample, not summed, a select per position:
    O(M U)), a histogram per sample (scatter-add of V values into B bins
    at data-dependent bins: O(V B^2)), a scan rolled into a loop that stores its ys (O(n^2)); a sort too large to
    unroll stops before its network (MemoryError or minutes before); a fori_loop of 10^6 steps over 64 accumulators is
    no tiled loop (its buffer of shared values held K Q F vectors per thread)."""
    M, rng = 3000, np.random.default_rng(69)
    T, i = rng.standard_normal(M), rng.integers(0, M, 40)
    loss = lambda g: lambda T: jnp.sum(jax.vmap(g, (None, 0))(T, i))
    f = lambda T, i: T[i] * 2.0
    np.testing.assert_allclose(jax.grad(loss(kernel(f)))(T), jax.grad(loss(f))(T), rtol=1e-14)
    u = lambda i, x: (jax.lax.dynamic_update_slice(jnp.zeros((6, 2)), x, (i, 0)), jax.lax.dynamic_update_slice(x, x[:1] * 2, (i - 1, 1)))
    s, x = i[:8] % 9 - 2, rng.standard_normal((8, 3, 2))  # (starts beyond both ends: clamped)
    for o, r in zip(jax.vmap(kernel(u))(s, x), jax.vmap(u)(s, x)):
        np.testing.assert_array_equal(o, r)
    v = rng.uniform(0, 1, (20, 30))
    hist = lambda v: jnp.zeros(200).at[(v * 200).astype(jnp.int32)].add(1.0)
    np.testing.assert_array_equal(jax.vmap(kernel(hist))(v), jax.vmap(hist)(v))
    z = rng.standard_normal((8, 3))
    lorenz = lambda z: jax.lax.scan(lambda c, _: (c + 0.01 * jnp.array([10 * (c[1] - c[0]), c[0] * (28 - c[2]) - c[1],
                                                                          c[0] * c[1] - 8 / 3 * c[2]]), c), z, None, length=60)
    for o, r in zip(jax.vmap(kernel(lorenz))(z), jax.vmap(lorenz)(z)):
        np.testing.assert_allclose(o, r, rtol=1e-12)
    with pytest.raises(NotImplementedError, match="sort"):
        jax.vmap(kernel(jnp.sort))(rng.standard_normal((4, 3000)))
    a = rng.standard_normal((5, 64))
    w = lambda a: jax.lax.fori_loop(0, 10 ** 6, lambda k, c: c + a * jnp.sin(k * 1e-3), jnp.zeros(64))  # (sin(k): read by
    np.testing.assert_allclose(jax.vmap(kernel(w))(a), jax.vmap(w)(a), rtol=1e-8)  # all tiles, buffered when tiled)


def test_streamed_outputs(monkeypatch):
    """Outputs through the threads' buffers and non-temporal stores, here forced (taken only beyond half the threads'
    L3: no test reached them before): rows of several sizes, a reduced output, a remainder block."""
    from kernelax import emit
    monkeypatch.setattr(emit, "l3_per_core", lambda: 0)
    f = lambda x: (jnp.sin(x) * 2.0, jnp.cos(x[:2]), x[0] * x[1], x @ x)
    x = np.random.default_rng(73).standard_normal((1003, 5))
    g = cvmap(f, (x[0],), (0,), n_threads=4)
    assert "> (int64_t)nt * 0 / 2" in g.kernel.sources[0]  # (every output streamed)
    for o, r in zip(g(x), jax.vmap(f)(x)):
        np.testing.assert_allclose(o, r, rtol=1e-13)


def test_primitives_since_jax_0121():
    """Primitives and parameters of JAX 0.12.1, as jax.vmap: checkpoint_name (name), fma, fmuladd, exp2m1, log2p1
    (all refused), mul with out_dtype (float32 operands multiplied in double, as XLA converts them first; it rounded
    the product to float32), atan2pi of broadcast operands (a TypeError), tanh's derivative (one_minus_square as
    (1 + x)(1 - x), as XLA), sigmoid of float16 (XLA's stable form)."""
    from jax.ad_checkpoint import checkpoint_name
    x = np.random.default_rng(83).uniform(0.1, 0.9, (300, 4))
    fs = [lambda v: checkpoint_name(v * 2.0, "a") + 1.0,
          lambda v: (jax.lax.fma(v[0], v[1], v[2]), jax.lax.fmuladd(v[1], v[2], v[3])),
          lambda v: (jax.lax.exp2m1(v), jax.lax.log2p1(v)),
          lambda v: jax.lax.atan2pi(v[:2, None], v[None, 2:]),
          lambda v: jax.vmap(jax.grad(jnp.tanh))(v * 9.0)]
    for f in fs:
        for o, r in zip(jax.tree.leaves(cvmap(f, (x[0],), (0,))(x)), jax.tree.leaves(jax.vmap(f)(x))):
            close(o, r, rtol=1e-13)
    x32 = x.astype(np.float32)
    f = lambda v: jax.lax.mul(v[0], v[1], out_dtype=jnp.float64)
    np.testing.assert_array_equal(cvmap(f, (x32[0],), (0,))(x32), np.asarray(jax.vmap(f)(x32)))
    h = (x * 40 - 20).astype(np.float16)
    np.testing.assert_array_equal(np.asarray(cvmap(jax.nn.sigmoid, (h[0],), (0,))(h), np.float32),
                                  np.asarray(jax.vmap(jax.nn.sigmoid)(h), np.float32))


def test_index_keys_follow_jax_numerics():
    """JAX's settings that change what kernelax computes (jax_accurate_erf_inv: erf_inv's lowering) are part of the
    index key: a kernel built under one setting is not taken under the other (it was)."""
    from kernelax import cpu as ck
    f, ex = (lambda x: jax.scipy.special.erfinv(x) * 2.0), [np.zeros(3)]
    keys = []
    for flag in (True, False):
        jax.config.update("jax_accurate_erf_inv", flag)
        try:
            keys.append(ck._index_key(f, ex, [True], 8, None, None)[0])
        finally:
            jax.config.update("jax_accurate_erf_inv", True)
    assert keys[0] != keys[1]


def test_counted_loops_without_masks():
    """A loop whose condition is the same in all lanes (a counter from a constant, as a rolled scan's or fori_loop's)
    runs without masks: its carries are not blended with the old ones each step (sel), and reads at the counter
    (a scan's xs, v[i]) take the entry of all lanes at once (pick1), not a gather per lane."""
    f = lambda v: jax.lax.fori_loop(0, 1000, lambda i, c: (c[0] * 0.999 + v[i % 3], c[1] + jnp.sin(c[0])), (v[0], 0.0))
    x = np.random.default_rng(83).uniform(0.5, 1.5, (99, 3))
    g = cvmap(f, (x[0],), (0,))
    src = g.kernel.sources[0]
    assert "for (;;)" in src and "sel_d(act" not in src and "pick1_d" in src
    for o, r in zip(g(x), jax.vmap(f)(x)):
        close(o, r, rtol=1e-12)


def test_rolled_over_typed_keys(monkeypatch):
    """A long scan over typed keys of random.split (jax.random.key; gymnax' rollouts): the rolled loop reads its key
    by dynamic_slice of slice size (1,) on the key array, which takes the key at its step (frontend._Split,
    LAZY_KEYS), not all 1000 keys before the loop (as for the keys' uint32 data, slice size (1, 2))."""
    from kernelax import frontend
    monkeypatch.setattr(frontend, "LAZY_KEYS", True)
    monkeypatch.setattr(frontend, "ROLL_XS", 50000)
    f = lambda k, x: jax.lax.scan(lambda c, kk: (c * 0.99 + jax.random.normal(kk), None), x,
                                  jax.random.split(jax.random.wrap_key_data(k), 1000))[0]
    keys = jax.random.key_data(jax.random.split(jax.random.key(0), 99))
    x = np.random.default_rng(84).uniform(-1, 1, 99)
    g = cvmap(f, (keys[0], x[0]), (0, 0))
    src = g.kernel.sources[0]
    assert "for (;;)" in src and len(src) < 100_000, len(src)  # (all keys in advance: 2.6 MB)
    close(g(keys, x), jax.vmap(f)(keys, x), rtol=1e-12)


def test_kernel_variants_after_the_first_call():
    """kernelax.kernel builds K = 1 for its first call and its other variants (K = 2..4) in a thread at low priority
    (as kernelax.jit: Kernel.complete), which the first call does not wait for (gcc of 4 variants, the largest 4 times
    the code: connect_four's first call 7.2 -> 12.6 s); the next call with TUNE_MIN samples or more chooses among them."""
    from kernelax import auto, cpu as ck
    import time
    c = 0.5 + time.time_ns() % 10 ** 9 * 1e-10  # (a kernel not built before, also not by an earlier run: its libraries
    f = lambda r: jnp.sin(r * c) * r + c  # cached by their sources load at once, before the assertion below)
    x = np.random.default_rng(85).uniform(0.5, 1.5, (ck.TUNE_MIN, 3))
    g = jax.jit(jax.vmap(kernel(f)))
    close(g(x), jax.vmap(f)(x), rtol=1e-13)
    k = list(ck._KERNELS.values())[-1]  # (the one built last)
    assert k.Ks == [1]  # (the variants still building: gcc takes longer than the call)
    for t in auto._THREADS:
        t.join()
    assert k.Ks == [1, 2, 3, 4] and k.K() is None
    close(g(x), jax.vmap(f)(x), rtol=1e-13)
    assert k.K() is not None


def test_loop_kernels_at_o1(monkeypatch):
    """A source with a loop (a rolled scan, a while loop) of more than build.LOOP_BIG of C compiles at build.LOOP_OPT
    (-O1 with gcc's FMAs): RL rollouts per environment ran as fast or faster (Asterix 43.1 -> 34.5 ms, Snake 69.4 ->
    60.5) in less of gcc's time (8.2 -> 4.2 s, 29.2 -> 13.5 s); smaller kernels stay at -O3 (no gcc time saved, small
    loops 10 % slower), larger ones at BIG_OPT. The level is in the cache key of the library and its objects."""
    from kernelax import build
    body = "x = x * 0.5 + 1.0;\n" * (build.LOOP_BIG // 19 + 1)
    loop, straight = "for (;;) {\n" + body + "}\n", body
    assert {"-O1", "-fexpensive-optimizations"} <= set(build._flags(loop, ())) and "-O3" not in build._flags(loop, ())  # (FMAs)
    calls = "for (;;) {\n" + "x = threefry_v(a, b, c, d);\n" * (build.LOOP_BIG // build.THREEFRY_C + 1) + "}\n"
    assert "-O1" in build._flags(calls, ())  # (a call of threefry_v counts as its hash unrolled)
    assert "-ftree-slp-vectorize" in build._flags(loop, ()) and "-ftree-slp-vectorize" in build._flags(body * 9, ())  # (xland-
    # minigrid 88.6 -> 73.5 ms, Snake, othello 5 % faster, no gcc time; also for BIG)
    assert "-O3" in build._flags(straight, ()) and "-O3" in build._flags(loop[: build.LOOP_BIG // 2] + "}\n", ())
    key = build._key(loop, "gcc", ())
    monkeypatch.setattr(build, "LOOP_BIG", 10 * build.LOOP_BIG)  # (the same source at -O3: another library)
    assert "-O3" in build._flags(loop, ()) and build._key(loop, "gcc", ()) != key


def test_guarded_branches():
    """A select whose operand needs much work that nothing else needs (the reset of an environment where its episode
    ended, the untaken branch of lax.cond, a material law's plastic branch) computes it only where a lane of the block
    takes it: if (many(mask)) {...} (an RL rollout's auto-reset cost 10-73 % of its time, XLA's too: CartPole, Snake,
    TSP), in loops (outside them a lane of the block takes either side almost always: interp, expm3 slower). The
    results as before, NaN of the guarded work in lanes that do not take it included."""
    def heavy(x):
        for k in range(100):
            x = jnp.sin(x * 1.1 + k)
        return x

    def f(x):
        def body(c):
            i, y, z, w = c
            h = heavy(jnp.log(y - 0.95))  # (NaN below 0.95: lanes that do not take it)
            return (i + 1.0, jnp.where(y > 0.9, h, y * 0.99), jnp.where(y > 0.9, h * h, z),
                    jax.lax.cond(y < 0.05, lambda: heavy(y * 3.0), lambda: -w))
        return jax.lax.while_loop(lambda c: c[0] < 1.0 + 3.0 * x, body, (0.0, x, x, x))[1:]
    x = np.random.default_rng(86).uniform(0.0, 1.0, 999)
    g = cvmap(f, (x[0],), (0,))
    src = g.kernel.sources[0]
    import re
    assert len(re.findall(r"\n  +vd g\d+_0 = \{\};", src)) >= 3  # (the guarded values, declared before their blocks)
    for o, r in zip(g(x), jax.vmap(f)(x)):
        close(o, r, rtol=1e-12)


def test_guarded_branches_in_and_around_loops():
    """Selects of one condition (one the loop does not change) inside a loop's body and around the loop: the work of
    the body's select reads the loop's carries, so it is guarded in the body, the other's around the loop."""
    def heavy(x):
        for k in range(100):
            x = jnp.sin(x * 1.1 + k)
        return x

    def f(x):
        c = x > 0.9
        out = jnp.where(c, heavy(x * 2.0), x)  # (before the loop in the emission order)
        y = jax.lax.fori_loop(0, 3, lambda i, y: jnp.where(c, heavy(y + i), y * 0.5), x)
        w = jax.lax.while_loop(lambda v: v[0] < 2.0, lambda v: (v[0] + 0.5 + v[1], jnp.where(c, heavy(v[1] * v[0]), v[1])), (x, x))
        return out, y, w[1], jnp.where(c, heavy(y * 3.0), y)
    x = np.random.default_rng(87).uniform(0.0, 1.0, 999)
    g = cvmap(f, (x[0],), (0,))
    import re
    assert len(re.findall(r"\n  +vd g\d+_0 = \{\};", g.kernel.sources[0])) >= 1  # (the while loop's)
    for o, r in zip(g(x), jax.vmap(f)(x)):
        close(o, r, rtol=1e-12)


def test_no_guards_for_little_work():
    """No guarded block for work below dag.GUARD_MIN (2048's selects of 46 operations each: 13 % slower) or of fewer
    than dag.GUARD_PER_OUT operations per guarded value (xland-minigrid's grid after an action, 522 operations for 128
    cells: each value live from the block to its select, the GPU kernel's local memory 0.7 -> 6 KB per thread, 2.8 times
    slower); the results as before."""
    def f(x0, v0):
        def body(s):
            i, x, v = s
            c = x > 0.5
            small = x
            for k in range(15):
                small = jnp.sin(small * 1.1 + k)
            return i + 1.0, jnp.where(c, small, x * 0.9), jnp.where(c, jnp.sin(jnp.sin(v * 1.1 + 0.5) * 0.9 + x) + v, v * 0.9)  # (45; 64 x 6)
        return jax.lax.while_loop(lambda s: s[0] < 1.0 + 3.0 * x0, body, (0.0, x0, v0))[1:]
    rng = np.random.default_rng(89)
    x, v = rng.uniform(0.0, 1.0, 999), rng.uniform(0.0, 1.0, (999, 64))
    g = cvmap(f, (x[0], v[0]), (0, 0))
    import re
    assert not re.search(r"\n  +vd g\d+_0 = \{\};", g.kernel.sources[0])
    for o, r in zip(g(x, v), jax.vmap(f)(x, v)):
        close(o, r, rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_top_k_of_few_by_selection(dtype):
    """top_k of few (k * special.TOPK_SELECT <= n) by k passes over the entries, the largest not yet taken each (ties
    by lower index, floats in XLA's total order: NaN, -0.0), instead of sorting all of them (n^2 / 2 compare-swaps:
    the positions an environment's reset draws, jumanji's Maze 189 000 nodes of 331 000, xland-minigrid 38 500 of 67 000)."""
    rng = np.random.default_rng(90)
    x = rng.integers(-3, 4, (50, 100)).astype(dtype)  # (ties)
    if dtype != np.int32:
        x[:, ::7] = rng.normal(size=x[:, ::7].shape)
        x[::5, 3], x[::6, 11], x[::4, 20], x[::3, 40] = np.nan, -np.nan, -0.0, 0.0
    f = lambda v: (jax.lax.top_k(v, 1), jax.lax.top_k(v, 3), jax.lax.top_k(-v, 2))
    g = cvmap(f, (x[0],), (0,))
    assert len(g.kernel.sources[0]) < 1_000_000  # (sorted: 4.4-5.7 MB)
    for o, r in zip(jax.tree.leaves(g(x)), jax.tree.leaves(jax.vmap(f)(x))):
        np.testing.assert_array_equal(np.asarray(o).view(np.int32 if dtype != np.float64 else np.int64) if dtype != np.int32 and np.asarray(r).dtype.kind == "f" else o,
                                      np.asarray(r).view(np.int32 if dtype != np.float64 else np.int64) if dtype != np.int32 and np.asarray(r).dtype.kind == "f" else r)


def test_scatter_of_a_row_at_a_symbolic_index():
    """A scatter of a row at a symbolic row index (a stack's push, jumanji's maze generation: 46 500 of Cleaner's 93 000
    nodes) compares each entry with the positions of its column only (those the constant parts of its index allow),
    not with every position of the array; out of bounds dropped or clipped as before."""
    def f(x, i, j):
        s = jnp.zeros((64, 4)) + x[0]
        s = s.at[i].set(x[:4])  # (one row, symbolic row: dropped out of bounds)
        s = s.at[j, 2].set(x[4])  # (one entry)
        s = s.at[i + 1].add(x[1:5], mode="clip")
        return s, jnp.maximum(s.at[j].max(x[2:6]), 0.0)
    rng = np.random.default_rng(91)
    x, i, j = rng.normal(size=(300, 6)), rng.integers(-2, 67, 300), rng.integers(-2, 67, 300)
    g = cvmap(f, (x[0], i[0], j[0]), (0, 0, 0))
    assert len(g.kernel.sources[0]) < 250_000  # (every position: 0.6 MB)
    for o, r in zip(g(x, i, j), jax.vmap(f)(x, i, j)):
        close(o, r, rtol=1e-12)


@pytest.mark.parametrize("cplx", [False, True])
def test_equinox_errors_as_nan(cplx):
    """equinox's runtime errors (eqx.error_if, as in diffrax, optimistix, lineax, dynamiqs; a lax.cond whose branch calls
    a pure_callback that raises): in a kernel, where nothing can raise, the sample's values as with EQX_ON_ERROR=nan
    (NaN, an integer's largest value, True); before, the callback ran for every sample (both branches of a cond are
    computed per sample) and raised, and complex values (dynamiqs' mesolve) were refused."""
    eqx = pytest.importorskip("equinox")

    def f(x):
        y = (x + 1j * x) * 2.0 if cplx else x * 2.0
        y, n = eqx.error_if((y, jnp.int32(3)), x > 0.8, "too big")
        return (jnp.real(y * y) + jnp.imag(y) if cplx else y + 1.0), n
    x = np.linspace(0.0, 1.0, 50)
    out = jax.jit(jax.vmap(kernel(f)))(x)
    ok = x <= 0.8
    ref = jax.vmap(f)(x[ok])
    np.testing.assert_allclose(np.asarray(out[0])[ok], ref[0], rtol=1e-14, atol=1e-15)
    assert np.isnan(np.asarray(out[0])[~ok]).all() and (np.asarray(out[1])[ok] == 3).all()
    assert (np.asarray(out[1])[~ok] == np.iinfo(np.int32).max).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_threefry_in_32_bit_lanes(dtype):
    """jax.random's threefry2x32 per sample as one node (dag.threefry), computed by the kernel in 32-bit lanes
    (threefry_v: a rotation one instruction) instead of as 64-bit integer operations with a wraparound after each (an RL
    rollout's random numbers were 74-94 % of its operations); the bits as jax.vmap's."""
    def f(seed, x):
        k = jax.random.fold_in(jax.random.key(0), seed)
        k1, k2 = jax.random.split(k)
        return (jax.random.uniform(k1, (5,), dtype) * x, jax.random.randint(k2, (3,), 0, 1000),
                jax.random.bits(jax.random.fold_in(k2, 7), (2,)))
    seed, x = np.arange(301, dtype=np.int32), np.linspace(0.5, 1.5, 301).astype(dtype)
    g = cvmap(f, (seed[0], x[0]), (0, 0))
    src = g.kernel.sources[0]
    assert "threefry_v(" in src.split("threefry_v(vi a")[1] and len(src) < 60_000
    for o, r in zip(jax.tree.leaves(g(seed, x)), jax.tree.leaves(jax.vmap(f)(seed, x))):
        np.testing.assert_array_equal(o, r)


def test_guards_count_threefry_by_its_operations():
    """A reset drawing a few random numbers (gymnax' Acrobot: four uniforms, four threefry nodes) is guarded as when the
    hash was unrolled: a threefry node counts as dag.THREEFRY_WEIGHT operations (GUARD_MIN)."""
    def f(seed):
        def body(c):
            i, s, k = c
            k, k1 = jax.random.split(k)
            s = jnp.where(s[0] > 0.9, jax.random.uniform(k1, (4,)), s * 1.01 + 0.01)  # (the reset: four uniforms)
            return i + 1.0, s, k
        return jax.lax.while_loop(lambda c: c[0] < 20.0 + seed % 7, body, (0.0, jnp.zeros(4), jax.random.key(seed)))[1]
    seed = np.arange(200, dtype=np.int32)
    g = cvmap(f, (seed[0],), (0,))
    import re
    assert re.search(r"\n  +vd g\d+_0 = \{\};", g.kernel.sources[0])
    np.testing.assert_array_equal(g(seed), jax.vmap(f)(seed))


def test_counter_of_a_loop_with_a_data_dependent_end():
    """A counter (a carry from a constant by a constant step) of a while loop whose end depends on the sample is not
    the same in all lanes: lanes that ended keep theirs while the others go on, so a read at it is per lane (pickv),
    not lane 0's (pick1: the lanes still running read another step's entry)."""
    def f(a, t):
        def body(c):
            i, s = c
            return i + 1, s + a[i % 16]
        return jax.lax.while_loop(lambda c: c[1] < t, body, (0, 0.0))
    rng = np.random.default_rng(93)
    a, t = rng.uniform(0.5, 1.5, (400, 16)), rng.uniform(1.0, 30.0, 400)
    g = cvmap(f, (a[0], t[0]), (0, 0))
    for o, r in zip(g(a, t), jax.vmap(f)(a, t)):
        np.testing.assert_array_equal(o, r)



def test_reference_of_threefry():
    """The reference evaluation of the DAG (reference.evaluate: benchmarks/run.py's operation counts) knows the threefry
    node (dag.threefry): the bits as jax.vmap's (before: KeyError, benchmarks/run.py stopped at the random case)."""
    from kernelax import frontend
    from kernelax.reference import evaluate

    def f(seed):
        k1, k2 = jax.random.split(jax.random.fold_in(jax.random.key(0), seed))
        return jax.random.bits(k1, (3,)), jax.random.randint(k2, (2,), 0, 1000)
    seed = np.arange(64, dtype=np.int32)
    ins, outs, _ = frontend.trace(f, seed[0])
    got = evaluate([x for o in outs for x in np.asarray(o, dtype=object).ravel()], [seed[:, None]])
    want = [c for r in jax.vmap(f)(seed) for c in np.asarray(r).reshape(64, -1).T]
    bits = lambda g: np.asarray(g, np.float64).view(np.int64)  # (integer outputs: their bits as doubles, bitsf64)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(bits(g), w.astype(np.int64) if w.dtype != np.uint64 else w.view(np.int64))



def test_keys_stay_lazy_through_identity_transposes():
    """Keys of a split stay lazy (each computed at the step of the rolled scan that reads it) through a transpose or
    reshape that moves no data of the typed key array (vmap's per sample, as kernelax.jit's regions see the program):
    the permutation of a key<fry>[200] is (0,), its data's axis not counted (before: all 200 keys in advance)."""
    def step(c, k):
        for _ in range(20):
            c = jnp.sin(c) * 1.01
        return c + jax.random.uniform(k), None

    def f(data):  # (the primitives bound as kernelax.jit's regions bind them: JAX's functions skip identities)
        ks = jax.lax.reshape_p.bind(jax.lax.transpose_p.bind(jax.random.split(jax.random.wrap_key_data(data), 200),
                                                             permutation=(0,)), new_sizes=(200,), dimensions=None, sharding=None)
        return jax.lax.scan(step, 0.0, ks)[0]
    keys = np.asarray(jax.random.key_data(jax.random.split(jax.random.key(1), 300)))
    g = cvmap(f, (keys[0],), (0,))
    src = g.kernel.sources[0]
    assert "for (;;)" in src and src.count("threefry_v(") - 1 <= 4
    close(g(keys), jax.vmap(f)(keys), rtol=1e-12)


def test_stop_gradient_of_an_integer_keeps_its_node():
    """stop_gradient of an integer value is the value itself, not wrapped to its dtype again: under vmap, lax.switch
    with an index per sample selects each branch's operands as select_n(index == k, stop_gradient(x), x) (JAX's batching
    rule), which then folds to x. Before, the wrapped side kept the select and each branch computed its own copy of
    what it reads from x (xland-minigrid's six actions: three copies of a scatter at the agent's next cell, kernelax.jit
    88 ms against kernelax.kernel's 71 ms)."""
    from kernelax import dag

    def f(x, p):
        def body(i, c):
            return jax.lax.select(p, jax.lax.stop_gradient(c), c) * 3 + i
        return jax.lax.fori_loop(0, 100000, body, x)
    xs, ps = np.arange(-50, 250, dtype=np.int32), np.arange(300) % 3 == 0
    g = cvmap(f, (xs[0], ps[0]), (0, 0))
    roots = [x for o in g.kernel.outs for x in np.asarray(o, dtype=object).flat if isinstance(x, dag.Node)]

    def ops(nodes):
        return [x.op for x in dag.schedule(nodes)] + [o for x in dag.schedule(nodes) if x.op == "while" for o in ops(list(dag.loop_roots(x)))]
    assert "while" in ops(roots) and "iselect" not in ops(roots), ops(roots)
    np.testing.assert_array_equal(g(xs, ps), jax.vmap(f)(xs, ps))


def test_window_gather_picks_among_its_starts(monkeypatch):
    """A gather of a window (slice size > 1 along a dimension indexed per sample: vmap's form of dynamic_slice, as
    kernelax.jit's regions see it) picks among the window's clamped starts, dim - size + 1 of them, as dynamic_slice
    does. Before, among all dim positions (xland-minigrid's 7 x 7 field of view from the grid padded to 22: arrays of
    22 instead of 16 options, 379 of them filled per step; kernelax.jit 85 ms against kernelax.kernel's 71 ms)."""
    from kernelax import dag, frontend
    monkeypatch.setattr(frontend, "PICK_MIN", 4)  # (16 options: an array, as a pick among them, not a chain of selects)
    dn = jax.lax.GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(), start_index_map=(0,))

    def f(x, i):
        return jax.lax.gather(x, i[None], dn, slice_sizes=(7,), mode=jax.lax.GatherScatterMode.CLIP)
    xs = np.random.default_rng(3).normal(size=(300, 22))
    idx = (np.arange(300) % 30 - 4).astype(np.int32)  # (also out of range: clamped)
    g = cvmap(f, (xs[0], idx[0]), (0, 0))
    roots = [x for o in g.kernel.outs for x in np.asarray(o, dtype=object).flat if isinstance(x, dag.Node)]
    arrs = [x for x in dag.schedule(roots) if x.op == "arr"]
    assert arrs and all(len(a.args) == 22 - 7 + 1 for a in arrs), [len(a.args) for a in arrs]
    np.testing.assert_array_equal(g(xs, idx), jax.vmap(f)(xs, idx))
    fill = lambda x, i: jax.lax.gather(x, i[None], dn, slice_sizes=(7,), mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
                                       fill_value=-1.0)  # (a start out of range: all of the window filled)
    np.testing.assert_array_equal(cvmap(fill, (xs[0], idx[0]), (0, 0))(xs, idx), jax.vmap(fill)(xs, idx))


def test_window_update_at_two_symbolic_starts():
    """dynamic_update_slice at two symbolic starts (a Tetris piece placed on the grid), and the scatter vmap makes of it
    (a window over all axes at one index vector, mode CLIP, as kernelax.jit's regions see it): the update placed along
    one axis after the other (per position, a chain over the entries that can land there, at masks start == position -
    offset shared by all positions), then a select per position on whether it lies in the window. Before,
    dynamic_update_slice nested a candidate array per start of the first axis (jumanji's Tetris: 6450 integer
    selects, 169 picks), the scatter compared each entry's flat position with each position it can reach (1600
    comparisons more; kernelax.jit 79 ms against kernelax.kernel's 73 ms)."""
    from kernelax import dag
    dn = jax.lax.ScatterDimensionNumbers(update_window_dims=(0, 1), inserted_window_dims=(), scatter_dims_to_operand_dims=(0, 1))

    def dus(g, p, r, c):
        return jax.lax.dynamic_update_slice(g, p, (r, c))

    def sc(g, p, r, c):
        return jax.lax.scatter(g, jnp.stack([r, c]), p, dn, mode=jax.lax.GatherScatterMode.CLIP, unique_indices=True,
                               indices_are_sorted=True)
    rng = np.random.default_rng(4)
    gs, ps = rng.integers(0, 9, (300, 8, 8)).astype(np.int32), rng.integers(0, 9, (300, 3, 3)).astype(np.int32)
    rs, cs = (rng.integers(-3, 11, 300).astype(np.int32) for _ in range(2))  # (also out of range: clamped)
    for f in (dus, sc):
        g = cvmap(f, (gs[0], ps[0], rs[0], cs[0]), (0, 0, 0, 0), K=1)  # (straight selects: gcc -O3 slow for K = 4)
        roots = [x for o in g.kernel.outs for x in np.asarray(o, dtype=object).flat if isinstance(x, dag.Node)]
        ops = collections.Counter(x.op for x in dag.schedule(roots))
        assert ops["iselect"] + ops["select"] <= 3 * 6 * 3 * 6 + 2 and ops["eq"] + ops["icmp"] <= 12 and not ops["pick"], \
            (f.__name__, ops)
        np.testing.assert_array_equal(g(gs, ps, rs, cs), jax.vmap(f)(gs, ps, rs, cs))


def test_picks_as_chains_of_selects_by_their_runs():
    """A pick is a chain of selects where its options form up to 12 runs of equal neighbours (frontend.PICK_MIN: the
    selects the chain keeps), else an array of the options read per lane. The chain folds with what neighbours share
    and takes no stores: xland-minigrid's picks (16 starts of its field of view, 10 runs, one reader each) 70.6 -> 11.9
    ms, Snake's (12 cells, 3 readers) 41.5 -> 35.5; pgx 2048's 16 board cells, read by 3 picks each, stay an array
    (as chains 12.1 -> 13.4 ms)."""
    from kernelax import dag

    def ops(f, *args):
        g = cvmap(f, tuple(a[0] for a in args), (0,) * len(args), K=1)
        roots = [n for o in g.kernel.outs for n in np.asarray(o, dtype=object).flat if isinstance(n, dag.Node)]
        close(g(*args), jax.vmap(f)(*args), rtol=1e-12)
        return collections.Counter(n.op for n in dag.schedule(roots))
    i, x = np.arange(301, dtype=np.int32), np.linspace(-1, 1, 301)
    distinct = lambda i, x: jnp.stack([jnp.sin(x * (k + 1)) for k in range(16)])[i % 16]  # (16 runs)
    padded = lambda i, x: jnp.stack([jnp.sin(x * (k + 1)) for k in range(10)] + [jnp.zeros_like(x)] * 6)[i % 16]  # (11)
    assert ops(distinct, i, x)["pick"] == 1
    c = ops(padded, i, x)
    assert not c["pick"] and c["select"] >= 10, c
