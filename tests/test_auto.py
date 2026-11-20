# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from programs import make  # noqa: E402

from kernelax import auto, decouple, jaxprs, jit  # noqa: E402

jax.config.update("jax_enable_x64", True)
PROGRAMS = make(1003, np.random.default_rng(0))  # 1003 rows: not a multiple of the SIMD width


@pytest.fixture(autouse=True)
def _cpu():  # (C kernels: the programs on the CPU also where a GPU is the default device)
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def all_kernels(f, *args):
    """The program with every region as a kernel (whatever is faster) and the number of regions."""
    closed = jaxprs._flatten(f, args)
    _, steps, regions = auto._axis(closed.jaxpr)
    return jax.jit(auto._program(closed, steps, {r: r.spec(()) for r in regions})), len(regions)


def measured(path, nt=None):
    """The programs in the index at `path` as chosen with kernels by trials with nt threads (default NT): a choice made
    without them is measured again in a later process (P2)."""
    import glob
    from kernelax import cpu
    for p in glob.glob(os.path.join(str(path), "program-*.json")):
        cpu._update(os.path.basename(p)[:-5], lambda e: dict(e, use={str(nt or auto.NT): "c"}) if e.get("kernels") else e)


def check(f, args, regions=None):
    run, n = all_kernels(f, *args)
    ref = jax.tree.leaves(jax.jit(f)(*args))
    out = run(*jax.tree.leaves(args))
    assert regions is None or n == regions
    for a, b in zip(ref, out):
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_allclose(b, a, rtol=1e-11, atol=1e-11 * (1 + np.abs(np.where(np.isfinite(np.asarray(a, float)), np.asarray(a, float), 0)).max(initial=0)))


@pytest.mark.parametrize("name", list(PROGRAMS))
def test_programs(name):
    f, args = PROGRAMS[name]
    check(f, args, 2 if name == "centered_norms" else 1)


def test_integer_sums_over_the_samples():
    """Sums of integers over the samples (reductions, contractions, scatter-adds) are exact as XLA's: not accumulated
    in the kernels' doubles (64-bit integers are there by their bits)."""
    x = np.concatenate([np.random.default_rng(16).uniform(-1, 1, 517) * 50, [np.nan, np.inf]])
    i = np.random.default_rng(17).integers(0, 9, x.size)
    check(lambda x: jnp.sum(jnp.isfinite(x)) - 1, (x,))
    check(lambda x: (jnp.sum((x * 2).astype(jnp.int32)), jnp.sum(jnp.where(x > 0, 3, 1), dtype=jnp.int64)), (x,))
    check(lambda x, i: jnp.zeros(9, jnp.int64).at[i].add(jnp.isfinite(x) * (i + 1)), (x, i))
    check(lambda x, i: (x > 0).astype(jnp.int64) @ i, (x, i))


def test_nan_constants_at_run_time(tmp_path, monkeypatch):
    """A NaN among the constants passed at run time (here those of jnp.searchsorted's comparisons, whose index is a
    position of loads): the same in every variant of K, the kernel builds (an index of its own: built here)."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    ts, tm = np.linspace(0.25, 10.0, 100), np.random.default_rng(18).uniform(0, 11, 300)
    check(lambda ts, tm: ts[jnp.clip(jnp.searchsorted(ts, tm), 1, 99) - 1], (ts, tm), 1)


def test_same_rows_of_empty_constants():
    """A folded constant with empty rows (here (0, N) read with the rows on axis 1): its rows are all the same."""
    c = jaxprs._Const(np.zeros((0, 2), np.int32), None)
    assert auto._same_rows(c, "row1").shape == (0,)
    assert auto._same_rows(jaxprs._Const(np.arange(6).reshape(3, 2), None), "row1") is None


def test_64_bit_constants_without_x64():
    """A program with 64-bit integers traced with x64 (here a jaxpr run without it, as a jit traced in one context and
    called in another): its constants fold in the jaxpr's dtypes, as XLA runs it."""
    import jax.extend
    c = jnp.arange(16, dtype=jnp.int64).reshape(8, 2)
    closed = jax.make_jaxpr(lambda x: jnp.sin(x) * (c + 12).sum(1).astype(jnp.float32))(jax.ShapeDtypeStruct((300, 8), jnp.float32))
    f = jax.extend.core.jaxpr_as_fun(closed)
    x = np.random.default_rng(20).standard_normal((300, 8)).astype(np.float32)
    with jax.enable_x64(False):
        assert "add a 12:i64[]" in str(jax.make_jaxpr(f)(x))
        g = jit(f)
        g(x)
        g.wait()
        assert g.report()[0]["status"] in ("trial", "kernels")
        np.testing.assert_allclose(g(x)[0], jax.jit(f)(x)[0], rtol=1e-6)


def test_sample_axes_with_enough_rows_first():
    """The candidates for the sample axis: those with at least MIN_ROWS rows first. Derivatives lead their largest
    arrays with tangent axes (here 40 to 43 directions of an energy summed over 512 elements); the element axis
    must still be among the candidates, else the tangent axis becomes the sample axis and each sample unrolls all
    elements."""
    r = np.random.default_rng(22)
    u = r.standard_normal((512, 4))

    def f(u, a, b, c, d):
        g = lambda a, b, c, d: jnp.sum(jnp.sin(u[:, :1] * a[0] + u[:, 1:2] * b[1]) * jnp.exp(u[:, 2:3] * c[2] - u[:, 3:] * d[3])
                                       * (a[1] + b[2] + c[4] + d[5]), axis=0)
        return [jax.jacfwd(g, k)(a, b, c, d) for k in range(4)]
    args = (u,) + tuple(r.standard_normal(n) for n in (40, 41, 42, 43))
    closed = jaxprs._fold(decouple._decouple(jaxprs._inline(jaxprs._flatten(f, args))))  # (as _Plan plans)
    N = auto._axis(closed.jaxpr, sums=True)[0]
    assert N and np.prod(N[0]) >= auto.MIN_ROWS, N
    check(f, args)


def _small_mesh(E, Q, newton):
    """Element tangents (jacfwd of the residual: a sum over the points) of a neo-Hooke energy, with an internal variable
    from a local Newton solve per point (as plasticity) or without."""
    r = np.random.default_rng(26)
    u, dphi, w = 0.05 * r.standard_normal((E, 8, 3)), r.standard_normal((Q, 8, 3)), r.uniform(size=Q)

    def psi(F):
        C, J = F.T @ F, jnp.linalg.det(F)
        a = jax.lax.fori_loop(0, 8, lambda i, a: a - (a ** 3 + jnp.trace(C) * a - J) / (3 * a ** 2 + jnp.trace(C)), 0.5) \
            if newton else 0.0
        return 0.5 * (jnp.trace(C) - 3) - jnp.log(J) + 0.5 * jnp.log(J) ** 2 + a ** 2 * jnp.sum(C ** 2)
    residual = lambda ue: jnp.sum(jax.vmap(lambda d, wq: wq * jax.grad(lambda ue: psi(jnp.eye(3) + ue.T @ d))(ue))(dphi, w), 0)
    return lambda u: jax.vmap(jax.jacfwd(residual))(u), (u,)


def test_few_elements_with_much_work():
    """Fewer elements than MIN_ROWS: the element axis (loops over the points, linear rows) where it does far less work
    per value than the axes with more rows (elements x points: the points' bodies unrolled, a local solve per point and
    direction); else those (the work the same, more lanes). Worth a kernel by the work XLA would do."""
    plan = lambda f, args: auto._axis(jaxprs._fold(decouple._decouple(jaxprs._inline(jaxprs._flatten(f, args)))).jaxpr, sums=True)[0]
    f, args = _small_mesh(200, 8, newton=True)
    assert plan(f, args)[0] == 200
    g = jit(f)
    g(*args)
    g.wait()
    assert g.report()[0]["status"] in ("trial", "kernels")
    np.testing.assert_allclose(g(*args), jax.jit(f)(*args), rtol=1e-10, atol=1e-10)
    f, args = _small_mesh(200, 8, newton=False)
    assert isinstance(plan(f, args)[0], tuple)


def test_plan_replayed_from_the_index(tmp_path, monkeypatch):
    """A new program's planner choices (sample axes, inner and linear axes, map axes) go into the index with it; a
    later process (here a new kernelax.jit) replays them: the same plan without evaluating the other candidates. A choice
    that does not fit (here every axis replaced) gives another plan, not other values."""
    import glob
    import json
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    r = np.random.default_rng(37)  # (a loop per sample: kernels on CPUs and GPUs; next to it arrays of other lengths:
    c, y = r.uniform(0.5, 1.5, (2000, 3)), r.standard_normal((500, 7))  # more candidate axes)
    f = lambda x, y: (jax.vmap(lambda r: jax.lax.while_loop(lambda y: y < 4.0, lambda y: y * 1.1 + r[0], r[1]))(x),
                      jnp.cumsum(jnp.sin(y.T @ y), 0), jnp.exp(y).sum(0))
    args = (c, y)
    calls = []
    part = auto._partition
    monkeypatch.setattr(auto, "_partition", lambda *a, **k: calls.append(1) or part(*a, **k))
    monkeypatch.setattr(cpu, "FIX_K", 1)  # (K given: no variants to build)
    g = jit(f)
    g(*args)
    g.wait(variants=True)
    g(*args)
    searched, calls[:] = len(calls), []
    path = next(p for p in glob.glob(str(tmp_path / "program-*.json")) if json.load(open(p)).get("kernels"))
    entry = json.load(open(path))
    assert entry["hints"] and any(k.startswith("axis:") for k in entry["hints"])
    measured(tmp_path)
    h = jit(f)
    out = h(*args)
    rg, rh = g.report()[0], h.report()[0]
    assert rh["why"] == "index" and rh["regions"] == rg["regions"] and len(calls) < searched, (len(calls), searched)
    for a, b in zip(out, jax.jit(f)(*args)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
    entry["hints"] = {k: (7 if k.startswith("axis:") else v) for k, v in entry["hints"].items()}
    json.dump(entry, open(path, "w"))
    k = jit(f)
    for a, b in zip(k(*args), jax.jit(f)(*args)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def test_two_sample_axes_from_whole_arrays():
    """A region with two sample axes (elements x points) may read everything from whole arrays: an element's data as
    row s // Q, a point's as row s % Q. The kernel then has no input with rows: one more input (zeros per sample)
    gives it the number of samples."""
    r = np.random.default_rng(23)
    u, q = r.standard_normal((512, 3)), r.standard_normal(125)
    f = lambda u, q: jnp.sin(jnp.broadcast_to(q, (512, 125)) * u[:, :1] + u[:, 1:2]) + u[:, 2:]
    closed = jaxprs._flatten(f, (u, q))
    region = auto._Region({}, N=(512, 125))
    for e in closed.jaxpr.eqns:
        assert region.add(e, *auto._rule(e, region.N))
    region.outputs = list(closed.jaxpr.outvars)
    assert all(f == "whole" for _, f in region.inputs) and region.counted()
    run = jax.jit(auto._program(closed, [("region", region)], {region: region.spec(())}))
    np.testing.assert_allclose(run(u, q)[0], f(u, q), rtol=1e-12, atol=1e-14)


def test_partition_details():
    x = np.random.default_rng(1).uniform(-1, 1, (517, 3))
    check(lambda x: x + jnp.arange(3.0), (x,), 1)  # a broadcast of a uniform value, recomputed per sample
    check(lambda x: x * jnp.ones((1, 3)) + x[:, :1], (x,), 1)  # a uniform operand with a leading axis of 1
    check(lambda x: {"a": x[:, 0] > 0, "b": (x * 3).astype(jnp.int32)}, (x,), 1)  # pytree, bool and int outputs
    check(lambda x: jnp.sort(x, axis=1)[:, ::-1] @ np.eye(3), (x,), 1)
    check(lambda x: jnp.cumsum(x, 0)[::-1], (x,), 0)  # rows mixed: no region
    check(lambda x: (x @ x.T).sum(1), (x[:40],), 1)  # row i: from row i and the whole of x.T (uniform)
    check(lambda x: jax.lax.fori_loop(0, 3, lambda i, y: y * i, x), (x,), 1)
    check(lambda x: jnp.sin(x).at[:, 1].set(jnp.cos(x[:, 0])).at[:, 2].add(1.0), (x,), 1)  # scatters at uniform indices  # the counter, uniform, in the kernel
    scan = lambda x: jax.lax.scan(lambda c, a: (c + a, (c * a, a - c)), jnp.zeros(len(x)), jnp.sin(x).T)
    check(scan, (x,), 1)  # xs and ys with rows on axis 1, ys returned as they are
    check(lambda x, k: jax.lax.cond(k > 0, lambda y: (y * 2, k), lambda y: (y - 1, -k), x), (x, 1.5), 1)  # uniform output


def test_jit():
    f, (x,) = PROGRAMS["black_scholes"]
    g = jit(f)
    np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-11, atol=1e-13)  # XLA, or kernels from the index
    g.wait()
    np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-11, atol=1e-13)  # the kernels
    assert len(g.report()) == 1 and g.report()[0]["regions"][0]["chosen"] == "c"
    np.testing.assert_allclose(g(x[:500]), jax.jit(f)(x[:500]), rtol=1e-11, atol=1e-13)  # another signature: another plan
    assert len(g.report()) == 2
    h = jit(lambda x: jit(f)(x) * 2)  # nested: the inner one is part of the outer program
    np.testing.assert_allclose(h(x), 2 * jax.jit(f)(x), rtol=1e-11, atol=1e-13)
    grad = jax.grad(lambda x: g(x).sum())(x)  # under transformations: jax.jit
    np.testing.assert_allclose(grad, jax.grad(lambda x: f(x).sum())(x), rtol=1e-11, atol=1e-13)


def test_moves_and_axis():
    """Shape ops that move the rows between axis 0 and 1, one-step scans (lax.map in one batch), and the sample
    axis chosen by the data its regions cover (here the 300 rows, not the 4000 of the index array)."""
    x, idx = np.random.default_rng(3).uniform(-1, 1, (300, 4)), np.arange(4000) % 300
    split = lambda x: sum(jnp.split(jnp.sin(x), 2, axis=1))
    stack = lambda x: jnp.stack([jnp.sin(x), x]).transpose(1, 0, 2).sum(1)
    expand = lambda x: jnp.broadcast_to(jnp.sin(x)[None], (1,) + x.shape).reshape(x.shape) * 2
    mapped = lambda x, idx: (jax.lax.map(lambda r: jnp.sin(r) * r[0], x, batch_size=300), jnp.cumsum(idx)[-1])
    for f, args in ((split, (x,)), (stack, (x,)), (expand, (x,)), (mapped, (x, idx))):
        check(f, args, 1)
    closed = jaxprs._flatten(mapped, (x, idx))
    assert auto._axis(closed.jaxpr)[0] in ([300], [(300, 4)])  # (rows, or rows x entries: the same coverage)


def test_scatter_base_in_region():
    """A scatter-add into a value the region computes: the kernel adds into bases from outside, so not in it."""
    x, perm = np.random.default_rng(11).uniform(size=512), np.random.default_rng(12).permutation(512)
    check(lambda x, p: jnp.sin(x).at[p].add(jnp.exp(x)), (x, perm))


def test_order():
    """A broadcast of a value computed from a region's output (a cross-row sum) closes the region first."""
    x = np.random.default_rng(2).uniform(-1, 1, (301, 3))
    check(lambda x: (lambda y: y * jnp.sum(y, 0))(jnp.sin(x)), (x,), 2)
    check(lambda x: (lambda y: y + (y @ np.ones((3, 3)))[0])(jnp.sin(x)), (x,), 2)


def test_arguments_and_cache():
    f, (x,) = PROGRAMS["black_scholes"]
    h = lambda x, n, *, scale=1.0, shift: f(x[:, :n] * scale + shift)
    g = jit(h, static_argnums=1, static_argnames="shift")
    ref = lambda scale: jax.jit(h, static_argnums=1, static_argnames="shift")(x, 3, scale=scale, shift=0.1)
    np.testing.assert_allclose(g(x, 3, scale=2.0, shift=0.1), ref(2.0), rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(g(x, 3, scale=np.asarray(1.5), shift=0.1), ref(1.5), rtol=1e-11, atol=1e-13)
    assert len(g.report()) == 2  # a Python float (weakly typed) and an array: two signatures
    g(x, 3, shift=0.2, scale=2.0)
    assert len(g.report()) == 3  # another static value
    g.wait(variants=True)  # (the index entries of its kernels: written once their variants are built)
    rep = g.report()  # (per signature, in the order of their first calls; fixed values of status and why)
    assert [r["signature"] for r in rep] == [f"f64[{len(x)},3], f64[]{{weak}}; n=3, shift=0.1", f"f64[{len(x)},3], f64[]; n=3, shift=0.1",
                                             f"f64[{len(x)},3], f64[]{{weak}}; n=3, shift=0.2"], rep
    assert all(r["status"] in ("building", "trial", "kernels", "xla", "failed") and r.get("why") in
               (None, "index", "measured", "long calls", "cost model", "not built") for r in rep), rep
    rep[0]["errors"] = ["changed"]
    assert g.report()[0].get("errors") != ["changed"]  # (copies)
    again = jit(h, static_argnums=1, static_argnames="shift")  # a new function: its kernels from the index on disk
    np.testing.assert_allclose(again(x, 3, scale=2.0, shift=0.1), ref(2.0), rtol=1e-11, atol=1e-13)
    assert again.report()[0]["status"] in ("trial", "kernels")  # (not building: from the index; trials where not measured)
    k = lambda x, n: x * jnp.ones(n)  # a static argument by keyword, or by name in position (as jax.jit: either way)
    np.testing.assert_array_equal(jit(k, static_argnums=1)(x[0, :1], n=3), np.full(3, x[0, 0]))
    np.testing.assert_array_equal(jit(k, static_argnames="n")(x[0, :1], 3), np.full(3, x[0, 0]))


def test_gather_and_scatter_in_the_kernel():
    """A small finite-element-like assembly: nodal values gathered through a connectivity (an input in memory, read
    per sample), an element computation, and the scatter-add of the element vectors (a sum over the samples,
    accumulated in the kernel), also of concatenated updates with index arithmetic on the whole array."""
    r = np.random.default_rng(4)
    ne, nn = 1001, 700
    conn, nodes = r.integers(0, nn, (ne, 4)), r.standard_normal((nn, 2))

    def element(x):  # (ne, 4, 2) -> (ne, 4)
        return jnp.sin(x[..., 0]) * x[..., 1] + jnp.sum(x[..., 0], axis=1, keepdims=True)

    def residual(nodes, conn):
        return jnp.zeros(nn).at[conn.reshape(-1)].add(element(nodes[conn]).reshape(-1))

    def with_boundary(nodes, conn):  # concatenated element and boundary entries, negative indices wrapped
        idx = jnp.concatenate([conn.reshape(-1), jnp.arange(-5, 0)])
        upd = jnp.concatenate([element(nodes[conn]).reshape(-1), jnp.ones(5)])
        return jnp.zeros(nn).at[idx].add(upd)

    from kernelax import emit
    defaults = emit.SCAT_BIG, emit.WIDE_MIN
    for big, wide in (defaults, (1, defaults[1]), (1, 0)):  # (sums into large arrays: through buffers and a cached
        emit.SCAT_BIG, emit.WIDE_MIN = big, wide  # plan, positions from the input; its entry numbers 64-bit)
        try:
            for f in (residual, with_boundary):
                closed = jaxprs._flatten(f, (nodes, conn))
                _, steps, regions = auto._axis(closed.jaxpr)
                assert any("gather" in {e.primitive.name for e in r.orig} and "scatter-add" in {e.primitive.name for e in r.orig}
                           for r in regions)
                for _ in range(2):  # (the second call: the cached plan)
                    check(f, (nodes, conn), 1)
        finally:
            emit.SCAT_BIG, emit.WIDE_MIN = defaults


def test_inside_loops():
    """Stage 3: loops that are not row-wise as a whole (a global convergence check, lax.map in several batches)
    keep running in XLA with kernels for the row-wise regions of their bodies (enough work per call: MIN_WORK_LOOP)."""
    r = np.random.default_rng(5)
    x, B = r.uniform(0.5, 2.0, (1003, 8)), 0.1 * r.standard_normal((8, 8))

    def newton(x):  # per row c^3 + B c = x, until the largest residual of all rows is small
        F = lambda c: c ** 3 + c @ B.T - x
        step = lambda c: c - jax.vmap(lambda ci, fi: jnp.linalg.solve(jnp.diag(3 * ci ** 2) + B, fi))(c, F(c))
        cond = lambda s: (s[1] < 50) & (jnp.max(jnp.abs(F(s[0]))) > 1e-12)
        return jax.lax.while_loop(cond, lambda s: (step(s[0]), s[1] + 1), (x ** (1 / 3), 0))

    mapped = lambda x: jax.lax.map(lambda q: jnp.linalg.solve(B + jnp.diag(q ** 2 + 1), jnp.sin(q)), x, batch_size=300)
    for f in (newton, mapped):
        g = jit(f)
        g(x)
        g.wait()
        for a, b in zip(jax.tree.leaves(g(x)), jax.tree.leaves(jax.jit(f)(x))):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
        report = g.report()[0]
        assert report["loops"] and report["status"] in ("trial", "kernels"), report
        again = jit(f)  # the kernels of the loop bodies from the index
        for a, b in zip(jax.tree.leaves(again(x)), jax.tree.leaves(jax.jit(f)(x))):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_two_sample_axes():
    """Nested vmaps (rows x points): the sample axis is both leading axes; row data (N, ...) is read at row s // Q,
    point data (Q, ...) at s % Q, and sums over the points add into row s // Q (also as the cotangent of row data)."""
    r = np.random.default_rng(5)
    x, a, w = r.uniform(0.5, 2, (301, 27)), r.uniform(0.5, 1.5, (301, 2)), r.uniform(0.1, 1, 27)
    f, _ = PROGRAMS["quadrature"]
    _, steps, regions = auto._axis(jaxprs._flatten(f, (x, a, w)).jaxpr)
    assert len(regions) == 1 and [regions[0].form[v] for v in regions[0].outputs] == ["osum"]
    assert auto._axis(jaxprs._flatten(f, (x, a, w)).jaxpr)[0] == [(301, 27)]
    check(f, (x, a, w), 1)
    g = lambda x, a, w: jnp.sum(jnp.sin(x * a[:, :1]) * w + a[:, 1:] ** 2, 1)  # no loop per point: rows suffice
    check(g, (x, a, w), 1)
    check(jax.grad(lambda a, x, w: jnp.sum(g(x, a, w) ** 2)), (a, x, w))
    v = r.uniform(0.5, 2, (301, 27, 3))
    per_point = jax.vmap(jax.vmap(lambda v: jnp.sum(jax.lax.while_loop(lambda u: u[0] < 9, lambda u: u.at[0].set(2 * u[0] + u[1])
                                                                       .at[2].add(jnp.sin(u[0])), v))))  # x.at[i]: per sample
    check(lambda v: jnp.sum(per_point(v), 1), (v,), 1)
    y = r.uniform(0.5, 2, (301, 81))  # another pair of axes in the same program (elements x tangent directions)
    h = lambda v, y: jnp.sum(per_point(v), 1) + jnp.sum(jax.vmap(jax.vmap(lambda t: jax.lax.while_loop(lambda u: u < 5, jnp.exp, t)))(y), 1)
    assert auto._axis(jaxprs._flatten(h, (v, y)).jaxpr)[0] in ([301], [(301, 27), (301, 81)])  # (per row: loops over
    check(h, (v, y))  # each row's points with a batched predicate)


def test_size_one_rows_two_axes():
    """An elementwise operand of size 1 at both sample axes, these behind another axis (jax >= 0.12: mul broadcasts
    axes of size 1), here a folded constant: read whole, without both axes."""
    r = np.random.default_rng(8)
    x, C = r.uniform(0.5, 2, (301, 27)), r.uniform(0.5, 1.5, (3, 3))
    loop = jax.vmap(jax.vmap(lambda t: jax.lax.while_loop(lambda u: u < 5, jnp.exp, t)))
    f = lambda x: jnp.sum(loop(jax.lax.mul(jnp.stack([loop(x), x])[:, :, :, None, None], jnp.asarray(C)[None, None, None]).sum((0, 3, 4))), 1)
    assert auto._axis(jaxprs._flatten(f, (x,)).jaxpr)[0] == [(301, 27)]
    check(f, (x,), 1)


def test_random_keys():
    """Typed keys under vmap (wrap, fold_in, split, bits): row-wise, the per-key params as they are."""
    k = np.asarray(jax.random.key_data(jax.random.split(jax.random.PRNGKey(0), 300)))
    f = jax.vmap(lambda k: jax.random.normal(jax.random.split(jax.random.fold_in(jax.random.wrap_key_data(k), 3))[1], (4,)))
    check(f, (k,), 1)
    g = jax.vmap(lambda k: jax.random.split(jax.random.fold_in(jax.random.wrap_key_data(k), 1)))  # keys out of a kernel
    check(lambda k: jax.random.key_data(jax.vmap(lambda q: jax.random.fold_in(q[0], 2))(g(k))), (k,))


def test_batched_scatter():
    """x.at[i].set(v) / .add(v) under vmap with a per-sample index (scatter with batching dimensions): row-wise."""
    r = np.random.default_rng(6)
    x, i, v = r.standard_normal((301, 10, 10)), r.integers(0, 10, 301), r.standard_normal((301, 10))
    check(jax.vmap(lambda x, i, v: jnp.sin(x).at[i].set(v).at[(i + 3) % 10].add(jnp.cos(v))), (x, i, v), 1)


def test_row_constants_two_axes():
    """A constant (folded into the kernel) read by row s // Q under two sample axes (nested vmaps)."""
    r = np.random.default_rng(7)
    x, C = r.uniform(0.5, 2, (301, 27)), r.uniform(0.5, 1.5, (301, 2))
    f = lambda x: jnp.sum(jax.vmap(jax.vmap(lambda xq, c: jax.lax.while_loop(lambda y: y < 4.0, lambda y: y * (1 + c[0]) + c[1], xq),
                                             (0, None)), (0, 0))(x, C), 1)
    g = jit(f)
    g(x); g.wait()
    np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-12)


def test_coupled_loops_unvmap_any():
    """An any of equinox' unvmap_any in the body (shared control flow under vmap) becomes the sample's own value in
    one loop per sample: here it only picks recomputing over reusing an equal value (diffrax' first-same-as-last
    stage), so the results are jax.jit's."""
    eqxi = pytest.importorskip("equinox.internal")
    x = np.random.default_rng(8).uniform(0.5, 4.0, 301)

    def fsal(x):
        def body(c):
            i, y, m, s, fresh = c
            s0 = jnp.where(eqxi.unvmap_any(fresh), jnp.sin(y), s)  # (s: sin(y) unless fresh)
            new = y + 0.1 * s0 + 0.05
            return i + 1, jnp.where(m, new, y), m & (new < 4.0), jnp.where(m, jnp.sin(new), s), fresh & ~m
        out = jax.lax.while_loop(lambda c: jnp.any(c[2]) & (c[0] < 200), body, (0, x, x > 0, jnp.zeros_like(x), x > 0))
        return out[1], out[0]
    assert any(decouple._uses(e, "kernelax_lanes") for e in decouple._decouple(jaxprs._flatten(fsal, (x,))).jaxpr.eqns)
    g = jit(fsal)
    g(x); g.wait()
    for a, b in zip(jax.tree.leaves(g(x)), jax.tree.leaves(jax.jit(fsal)(x))):
        np.testing.assert_allclose(a, b, rtol=1e-12 * np.issubdtype(b.dtype, np.floating))


def test_coupled_loops():
    """While loops over all samples that run while any is active, with one step counter (equinox's under vmap:
    diffrax, optimistix), as one loop per sample where proved (_decouple), else as they are."""
    x = np.random.default_rng(8).uniform(0.5, 4.0, 301)
    decoupled = lambda f: any(decouple._uses(e, "kernelax_lanes") for e in decouple._decouple(jaxprs._flatten(f, (x,))).jaxpr.eqns)

    def newton(x):  # active samples step, the others stay; the counter's result: the steps of the slowest
        def body(c):
            i, y, m = c
            new = 0.5 * (y + x / y)
            return i + 1, jnp.where(m, new, y), m & (jnp.abs(new - y) > 1e-12)
        return jax.lax.while_loop(lambda c: jnp.any(c[2]) & (c[0] < 50), body, (0, x, x > 0))

    def masked(x, used=(0, 2)):  # a mask carry equal to the condition's (t < 1), kept so; the previous mask: a
        def body(c):  # carry that ends as if the sample had stopped later (read: not one loop per sample)
            i, t, y, m, old = c
            u = jnp.minimum(t + 0.1 * y, 1.0)
            lazy = jax.lax.cond(jnp.any(m & (y > 3)), lambda: jnp.where(m & (y > 3), y * 0.5, y), lambda: y)  # (an any
            return i + 1, jnp.where(m, u, t), jnp.where(m, lazy + 0.1, y), m & (u < 1.0), m  # that skips work)
        out = jax.lax.while_loop(lambda c: jnp.any(c[1] < 1.0) & (c[0] < 100), body, (0, 0.1 * x, x, 0.1 * x < 1.0, x > 0))
        return [out[k] for k in used]

    def coupled(x):  # all samples step while any is active: not one loop per sample
        return jax.lax.while_loop(lambda c: jnp.any(c[1] < 3.0), lambda c: (c[0] + 1, c[1] + 0.25 * x), (0, x))

    def any_in_body(x):  # an any over all samples that changes every active sample's step
        def body(c):
            i, y, m = c
            return i + 1, jnp.where(m, y + jnp.where(jnp.any(y > 3.0), 1.0, 0.1), y), m & (y < 5.0)
        return jax.lax.while_loop(lambda c: jnp.any(c[2]) & (c[0] < 200), body, (0, x, x > 0))

    def any_in_cond(x):  # a second any in the condition (over values the same in every step)
        body = lambda c: (c[0] + 1, jnp.where(c[2], c[1] * 1.5, c[1]), c[2] & (c[1] < 4.0))
        return jax.lax.while_loop(lambda c: jnp.any(c[2]) & (jnp.any(x > 1.0) | (c[0] > 1000)) & (c[0] < 200), body, (0, x, x > 0))

    for f, separate in ((newton, True), (masked, True), (lambda x: masked(x, (2, 4)), False), (coupled, False),
                        (any_in_body, False), (any_in_cond, False)):
        assert decoupled(f) == separate
        g = jit(f)
        g(x); g.wait()
        for a, b in zip(jax.tree.leaves(g(x)), jax.tree.leaves(jax.jit(f)(x))):  # (counters: equal)
            np.testing.assert_allclose(a, b, rtol=1e-12 * np.issubdtype(b.dtype, np.floating))
        if separate:
            assert any(r["heavy"] == ["while"] or "while" in r["heavy"] for r in g.report()[0]["regions"])


def test_work_of_nested_loops():
    """The work of a region counts each loop body once (as the kernel holds it), also with many carries around a
    nested loop; an unrolled scan per step."""
    x = np.random.default_rng(9).uniform(0.5, 1.5, (300, 40))

    def f(x):
        inner = lambda y: jax.lax.while_loop(lambda c: c[1] < 3, lambda c: (sum(jnp.sin(c[0] * k) for k in range(50)), c[1] + 1), (y, 0))[0]
        return jax.vmap(lambda r: jax.lax.while_loop(lambda c: c[0] < 2.0, lambda c: tuple(inner(v) + 1.0 for v in c), tuple(r)))(x)
    steps = auto._partition(jaxprs._flatten(f, (x,)).jaxpr, 300, sums=True)
    work = [auto._work(r)[0] for kind, r in steps if kind == "region" and any(e.primitive.name == "while" for e in r.orig)]
    assert work and work[0] < 20000  # (40 carries, each with its own inner loop of ~150 ops: ~6000, not 40 x that)


def test_linear_rows():
    """Tangent directions (jax.jacfwd over 24 degrees of freedom) through a narrow intermediate (F, 3 x 3) per point:
    from F on to the next narrow values (the stress) on the 9 unit vectors only; the same values with less work. Rows
    that are not linear in their inputs stay as they are."""
    r = np.random.default_rng(10)
    dN, u = r.standard_normal((8, 3)), r.standard_normal((8, 3)) * 0.1

    def W(F):
        C, A, out = F.T @ F, jnp.eye(3), 0.0
        for k in range(1, 5):
            A = A @ C / k
            out = out + jnp.trace(A) * 0.1 ** k + jnp.sin(jnp.trace(A @ F))
        return out + jnp.log(jnp.trace(C)) ** 2
    f = jax.jacfwd(lambda u, dN: dN @ jax.grad(W)(jnp.eye(3) + u.T @ dN).T)  # (8, 3, 8, 3)
    g, k = auto._linear_rows(f, (u, dN))
    assert k < 0.6
    np.testing.assert_allclose(jax.jit(g)(u, dN)[0], f(u, dN), rtol=1e-12, atol=1e-12)
    h = lambda X, A: jax.vmap(lambda x: jnp.sin(x @ A) @ A)(X)  # (rows of 3 entries, not linear)
    X, A = r.standard_normal((40, 3)), r.standard_normal((3, 3))
    assert auto._linear_rows(h, (X, A))[1] == 1.0


def test_fold_large_intermediates():
    """Values computed from constants alone fold into literals also through intermediate values larger than FOLD_MAX
    (an AD seed times shape function gradients, summed: the zeros of the result fold in the kernels); intermediates
    that nothing reads any more are dropped."""
    r = np.random.default_rng(11)
    A, B = r.standard_normal((80, 80)), r.standard_normal((80, 80))
    A[:, ::2] = 0.0
    f = lambda x: x + jnp.sum(jnp.asarray(A)[:, :, None] * jnp.asarray(B)[None, :, :], 1)  # (80, 80, 80) between
    folded = jaxprs._fold(jaxprs._flatten(f, (np.zeros((80, 80)),)))
    assert [e.primitive.name for e in folded.jaxpr.eqns] == ["add"] and not folded.consts
    c = folded.jaxpr.eqns[0].invars[1]
    assert isinstance(c, jaxprs._Const)
    np.testing.assert_allclose(c.val, A @ B, rtol=1e-12, atol=1e-12)


def test_single_sample():
    """A program without a sample axis whose time goes into a loop over small data (one local problem, time
    stepping) runs as one kernel on one sample; large data without a loop stays in XLA."""
    def f(y0, k):  # implicit Euler for a stiff system, a Newton solve per step
        def step(i, y):
            g = lambda z: z - y - 0.01 * jnp.stack([-k[0] * z[0] + k[1] * z[1] * z[2], k[0] * z[0] - k[1] * z[1] * z[2] - k[2] * z[1] ** 2,
                                                    k[2] * z[1] ** 2])
            return jax.lax.while_loop(lambda c: (jnp.linalg.norm(g(c[0])) > 1e-12) & (c[1] < 20),
                                      lambda c: (c[0] - jnp.linalg.solve(jax.jacfwd(g)(c[0]), g(c[0])), c[1] + 1), (y, 0))[0]
        return jax.lax.fori_loop(0, 200, step, y0)
    y0, k = np.array([1.0, 0.0, 0.0]), np.array([0.04, 1e4, 3e7])
    J = jit(f)
    J(y0, k)
    J.wait()
    rep = J.report()[0]
    assert rep["regions"] and rep["regions"][0].get("single") and rep["status"] in ("trial", "kernels")
    np.testing.assert_allclose(J(y0, k), jax.jit(f)(y0, k), rtol=1e-12, atol=1e-15)
    assert not auto._Single.suits(jaxprs._flatten(lambda x: jnp.sin(x) @ x, (np.ones((300, 300)),)))


def test_long_calls_skip_the_trials(tmp_path, monkeypatch):
    """A first call with kernels longer than TRIAL_MAX_CALL (here 0: every one) keeps the cost model's choice: no
    trials against XLA (a further XLA call would cost as much as the kernels might save), the choice in the index."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    monkeypatch.setattr(auto, "TRIAL_MAX_CALL", 0.0)
    c = np.random.default_rng(31).uniform(0.5, 1.5, (2000, 3))
    f = lambda x: jnp.sum(jax.vmap(lambda r: jax.lax.while_loop(lambda y: y < 4.0, lambda y: y * 1.1 + r[0], r[1]))(x))
    g = jit(f)
    g(c)
    g.wait(variants=True)
    for _ in range(3):
        np.testing.assert_allclose(g(c), f(c), rtol=1e-12)
    rep = g.report()[0]
    assert rep["why"] == "long calls", rep
    h = jit(f)  # (from the index, a choice not measured: the trials again, which a long call ends again)
    h(c)
    assert h.report()[0]["status"] == "trial"
    h(c)
    assert h.report()[0]["why"] == "long calls"


def test_wait_builds_at_the_first_call(tmp_path, monkeypatch):
    """wait: a new program's first call builds its kernels and runs with them (no XLA call before; the variants of K
    follow in a thread)."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    for t in list(auto._THREADS):  # (earlier tests' builds: their variants not counted below)
        t.join()
    before, complete = [], cpu.Kernel.complete
    monkeypatch.setattr(cpu.Kernel, "complete", lambda k: (before.append((list(k.Ks), list(k.rest[0]))), complete(k))[1])
    c = np.random.default_rng(32).uniform(0.5, 1.5, (2000, 3))
    # (its own kernel: not one that another test of this process built with its variants)
    f = lambda x: jnp.sum(jax.vmap(lambda r: jax.lax.while_loop(lambda y: y < 4.0, lambda y: y * 1.2 + r[0], r[1]))(x))
    g = jit(f, build_first=True)
    np.testing.assert_allclose(g(c), f(c), rtol=1e-12)
    assert g.report()[0]["status"] == "trial"  # (with its kernels: trials once the variants are built)
    g.wait(variants=True)
    assert before == [([1], [2, 3, 4])]  # (the first call waited for K = 1 only)
    np.testing.assert_allclose(g(c), f(c), rtol=1e-12)


def test_kernels_build_in_threads_at_once(monkeypatch):
    """Kernels build in threads at once: trace and emit one at a time (the frontend's lock), gcc outside the lock (the
    builds of two kernels overlap); first: K = 1 now, the other variants left for Kernel.complete."""
    import threading
    import time
    from kernelax import cpu as ck
    spans, compile_c = [], ck.compile_c

    def slow(src, *a):  # (a cached library too: the overlap shows)
        t0 = time.perf_counter()
        time.sleep(0.5)
        out = compile_c(src, *a)
        spans.append((t0, time.perf_counter()))
        return out
    monkeypatch.setattr(ck, "compile_c", slow)
    fs, out = [lambda x: jnp.sin(x) * 2.375 + x, lambda x: jnp.cos(x) * 3.625 - x], {}
    threads = [threading.Thread(target=lambda i=i: out.update({i: ck.kernel(fs[i], [np.zeros(3)], [True], first=True)}))
               for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    (a0, a1), (b0, b1) = sorted(spans)
    assert b0 < a1, spans
    for k in out.values():
        assert k.Ks == [1] and k.rest[0] == [2, 3, 4]
        k.complete()
        assert k.Ks == [1, 2, 3, 4]


def test_builds_finished_at_exit(tmp_path):
    """A process that calls a new program once (build_first=False) and exits finishes its kernel build at exit
    (KERNELAX_EXIT_WAIT): the
    next process runs the program with its kernels from the first call on, in trials (no choice was measured; it took
    the kernels for chosen, and kept them for good where they were slower than XLA)."""
    import subprocess
    script = tmp_path / "once.py"
    script.write_text("""import jax, jax.numpy as jnp, numpy as np
from kernelax import jit
jax.config.update("jax_enable_x64", True)
c = np.random.default_rng(33).uniform(0.5, 1.5, (2000, 3))
g = jit(lambda x: jnp.sum(jax.vmap(lambda r: jax.lax.while_loop(lambda y: y < 4.0, lambda y: y * 1.1 + r[0], r[1]))(x)), build_first=False)
g(c)
r = g.report()[0]
print(r["status"], r.get("why"), "first_ms" in r)
""")
    env = dict(os.environ, KERNELAX_CACHE_DIR=str(tmp_path / "cache"), JAX_PLATFORMS="cpu")
    run = lambda: subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    first = run()
    assert first.returncode == 0 and first.stdout.strip() == "building None False", first.stderr
    assert "finishing 1 kernel build" in first.stderr
    second = run()
    assert second.returncode == 0 and second.stdout.strip() == "trial None True", second.stderr


def test_solves_with_folded_right_hand_sides():
    """vmap of a triangular solve with an unbatched matrix folds the batch into the right-hand sides' columns
    (reshape, solve, reshape back): _unfold makes it a solve batched over that axis, the same values, rows along it
    for _partition."""
    r = np.random.default_rng(34)
    a = np.tril(r.standard_normal((4, 3, 3))) + 3 * np.eye(3)
    f = lambda b: jax.vmap(lambda bb: jax.lax.linalg.triangular_solve(a, bb, left_side=True, lower=True))(b)
    b = r.standard_normal((40, 4, 3, 2))
    closed = jaxprs._inline(jax.make_jaxpr(f)(b))
    assert any(e.primitive.name == "reshape" for e in closed.jaxpr.eqns)
    new = auto._unfold(closed, 40)
    solves = [e for e in new.jaxpr.eqns if e.primitive.name == "triangular_solve"]
    assert len(solves) == 1 and solves[0].invars[1].aval.shape[0] == 40 and not any(e.primitive.name == "reshape" for e in new.jaxpr.eqns)
    assert all(auto._rule(e, 40, True, True) is not None for e in new.jaxpr.eqns)
    np.testing.assert_allclose(jax.extend.core.jaxpr_as_fun(new)(b)[0], f(b), rtol=1e-12)


def test_platform_dependent_default_branch():
    """jax.lax.platform_dependent (a cond on platform_index): its default branch, as CPUs and GPUs run it, inlined for
    the planner (the cond's branches would need rules of their own; e.g. in the pivots of jnp.linalg.det's
    derivative)."""
    g = lambda x: jax.lax.platform_dependent(x, tpu=lambda x: x * 2, default=lambda x: x + 1)
    closed = jaxprs._inline(jax.make_jaxpr(g)(1.0))
    assert [e.primitive.name for e in closed.jaxpr.eqns] == ["add"]
    assert jax.extend.core.jaxpr_as_fun(closed)(1.0)[0] == 2.0


def test_concrete_arguments_under_a_trace(tmp_path, monkeypatch):
    """A kernelax.jit function called with concrete arguments inside a trace (jax.make_jaxpr, an enclosing jit; e.g. on
    closed-over constants) is jax.jit there: its program is part of the enclosing one, no kernel call in it (also when
    its kernels are in the index)."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    c = np.random.default_rng(14).uniform(0.5, 1.5, (2000, 3))
    f = lambda x: jnp.sum(jax.vmap(lambda r: jax.lax.while_loop(lambda y: y < 4.0, lambda y: y * 1.1 + r[0], r[1]))(x))
    g = jit(f)
    g(c)
    g.wait(variants=True)
    g(c)  # (one call with the kernels: they go into the index; not timed against XLA to the end: it may choose XLA)
    measured(tmp_path)
    h = jit(f)
    h(c)
    assert h.report()[0]["why"] == "index"
    closed = jax.make_jaxpr(lambda y: y + h(c))(1.0)
    names = lambda j: [e.primitive.name for e in j.eqns] + [n for e in j.eqns for v in e.params.values()
                                                           for b in (v if isinstance(v, tuple) else (v,)) if hasattr(b, "jaxpr")
                                                           for n in names(b.jaxpr if hasattr(b.jaxpr, "eqns") else b.jaxpr.jaxpr)]
    assert "kernelax_call" not in names(closed.jaxpr)
    np.testing.assert_allclose(jax.jit(lambda y: y + h(c))(1.0), 1.0 + f(c), rtol=1e-12)
    z = jit(lambda x, n: jnp.zeros(n) + x, static_argnames="n")  # (a static argument by name, given by position:
    np.testing.assert_allclose(jax.jvp(lambda y: z(1.0, 3) * y, (1.0,), (1.0,))[1], np.ones(3))  # static under a trace too)


def test_planning_under_compile_time_eval(tmp_path, monkeypatch):
    """Planning with the kernels in the index happens at the call: also inside jax.ensure_compile_time_eval (eager
    constant folding), where the kernel bodies are still traced (the sample's index stays symbolic)."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    r = np.random.default_rng(15)
    x, w, q = r.standard_normal((300, 4, 3)), r.standard_normal((300, 3)), r.standard_normal((4, 3))
    f = lambda x, w: jnp.sum(jnp.sin(x * q) * w[:, None, :], axis=1)  # (two sample axes; a sum over the inner one)
    g = jit(f)
    g(x, w)
    g.wait(variants=True)
    g(x, w)  # (into the index, as above)
    measured(tmp_path)
    h = jit(f)
    with jax.ensure_compile_time_eval():
        y = h(x, w)
    assert h.report()[0]["why"] == "index"
    np.testing.assert_allclose(y, f(x, w), rtol=1e-12)


def test_persist_before_the_index_entry(tmp_path, monkeypatch):
    """A kernel whose variants are still being built has no index entry yet (Kernel.complete enters it), also when
    another program (another index key) finds it by its source: storing the chosen K then waits for the entry."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    x = np.random.default_rng(19).standard_normal((4000, 3))
    g = jit(lambda x: jnp.sin(x) * jnp.cos(x) + x)
    g(x)
    g.wait(variants=True)
    k = next(k for k in cpu._KERNELS.values() if k.index and os.path.exists(os.path.join(str(tmp_path), k.index + ".json")))
    os.remove(os.path.join(str(tmp_path), k.index + ".json"))  # (as before complete)
    k.saved, k.struct.chosen = False, 0
    k.persist()
    assert not k.saved and cpu._read(k.index) is None
    k.complete()  # (enters it)
    assert cpu._read(k.index)["Ks"] == k.Ks


def test_kernels_across_sizes():
    """Arrays a kernel reads from in memory and sums into (an assembly over another mesh) may change size without a
    new compile: their sizes and the constants of the address arithmetic are passed at run time (the same source,
    the same library). The plan of the sums through buffers is rebuilt where the positions' inputs change."""
    r = np.random.default_rng(10)
    f = lambda u, conn, w: jnp.zeros(u.shape[0]).at[conn].add(jnp.sin(u[conn]) * w[:, None])  # per element: its nodes
    from kernelax import emit
    big = emit.SCAT_BIG
    try:
        for emit.SCAT_BIG in (big, 1):  # (sums into the threads' accumulators; through buffers and a plan)
            libs = []
            for nodes, elems in ((500, 300), (900, 700), (500, 300), (500, 300)):  # (the last: other positions, same sizes)
                args = (r.standard_normal(nodes), r.integers(0, nodes, (elems, 4)), r.standard_normal(elems))
                check(f, args, 1)
                plan = auto._Plan(jaxprs._flatten(f, args), ())
                libs.append([s.kernel.paths for s in plan.kernels.values()])
            assert libs[0] and libs[0] == libs[1] == libs[2]
    finally:
        emit.SCAT_BIG = big


def test_index_keys_and_context():
    """Kernels and programs that differ only in a constant inside a nested jaxpr (a jitted helper's closure) have
    their own index keys; the kernels build under the caller's (thread-local) x64."""
    from kernelax import kernel
    x = np.random.default_rng(13).uniform(size=(300, 3))
    for C in (np.array([1., 2., 3.]), np.array([10., 20., 30.])):
        inner = jax.jit(lambda y, C=C: jnp.sin(y) * C)
        np.testing.assert_allclose(jax.vmap(kernel(lambda v: inner(v)))(x), jax.vmap(inner)(x), rtol=1e-12)
        g = jit(lambda v: jax.vmap(lambda r: jax.lax.while_loop(lambda c: c[0] < 3, lambda c: (c[0] + 1, inner(c[1]) + r), (0, r))[1])(v))
        g(x); g.wait()
        np.testing.assert_allclose(g(x), jax.jit(g.__wrapped__)(x), rtol=1e-11, atol=1e-10)
    jax.config.update("jax_enable_x64", False)
    try:
        with jax.enable_x64(True):
            g = jit(lambda v: jnp.exp(jnp.sin(v) * 1.000000001) / (1 + v * v))
            g(x); g.wait()
            assert g.report()[0]["status"] in ("trial", "kernels")
    finally:
        jax.config.update("jax_enable_x64", True)


def test_staged_variants(tmp_path, monkeypatch):
    """Background builds compile K = 1 first (the program runs on it), the variants K > 1 after it (Kernel.complete):
    then the runtime chooses K anew, the results stay the same."""
    from kernelax import cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))  # (an index of its own: no variant chosen before)
    f = lambda x: jax.vmap(lambda v: jax.lax.fori_loop(0, 20, lambda i, c: c * 0.5 + 0.1 * jnp.sin(c) * v, v))(x)
    x = np.random.default_rng(11).standard_normal(50000)
    J = jit(f)
    ref = jax.jit(f)(x)
    J(x)
    J.wait()
    np.testing.assert_allclose(J(x), ref, rtol=1e-12)
    J.wait(variants=True)
    ck.stats(reset=True)  # (the kernels of this test: called from here on)
    for _ in range(3):
        np.testing.assert_allclose(J(x), ref, rtol=1e-12)
    k = [k for k in ck._KERNELS.values() if k.struct.calls and len(k.Ks) == 4]
    assert k and all(x.K() is not None for x in k)


def test_choices_per_thread_count(tmp_path, monkeypatch):
    """The index keeps the chosen K per thread count (of the call that chose it): a process with as many threads
    (cpu.NT) loads that variant only, one with others all of them, and chooses anew. kernelax.jit's choice between
    kernels and XLA likewise: measured with other threads only, it is measured again."""
    from kernelax import cpu as ck
    monkeypatch.setattr(ck, "_INDEX", str(tmp_path))
    f = lambda v: jax.lax.fori_loop(0, 20, lambda i, c: c * 0.5 + 0.1 * jnp.sin(c) * v, v)
    x, out = np.random.default_rng(12).standard_normal(50000), [np.empty(50000)]
    loaded = []

    def load(nt):
        ck._KERNELS.clear()
        monkeypatch.setattr(ck, "NT", nt)
        loaded.append(ck.kernel(f, [np.zeros(())], [True], cached=True))
        return loaded[-1]
    k = load(3)
    assert len(k.Ks) == 4 and ck._read(k.index)["K"] == {}
    k(3, x.size, [x], out)  # (K chosen with 3 threads)
    k.persist()
    assert ck._read(k.index)["K"] == {"3": k.K()}
    assert load(3).Ks == [k.K()]
    k5 = load(5)
    assert len(k5.Ks) == 4 and k5.K() is None
    k5(5, x.size, [x], out)
    k5.persist()
    assert set(ck._read(k.index)["K"]) == {"3", "5"}
    g = lambda x: jnp.sin(x) * jnp.cos(x) + x
    g1 = jit(g, build_first=True)
    g1(x)  # (kernels chosen without trials: measured by the next process)
    g1.wait(variants=True)  # (their index entries: written once the variants are built)
    measured(tmp_path)
    for nt, trials in ((auto.NT, False), (auto.NT + 1, True)):
        monkeypatch.setattr(auto, "NT", nt)
        h = jit(g)
        h(x)
        e = h.report()[0]
        assert (e["status"], e.get("why")) == (("trial", None) if trials else ("kernels", "index"))  # (from the index;
        assert ("first_ms" in e) == trials  # measured with other thread counts only: trials, whose first call is untimed)


def test_control_flow_with_large_uniform_outputs():
    """Control flow that passes large uniform state through (as a solver's cond around its state) stays in XLA: a
    kernel would compute those outputs per sample and keep the first sample's (auto.UNIFORM_MAX)."""
    x, u = np.random.default_rng(12).standard_normal(20000), np.arange(1000.0)
    f = lambda x, u: jax.lax.cond(u[0] >= 0, lambda: (jnp.sin(x) * 2, u + 1.0), lambda: (jnp.cos(x), u))
    small = lambda x, u: jax.lax.cond(u[0] >= 0, lambda: (jnp.sin(x) * 2, u[:3] + 1.0), lambda: (jnp.cos(x), u[:3]))
    for g, inside in ((f, False), (small, True)):
        plan = auto._Plan(jaxprs._flatten(g, (x, u)), ())
        assert any("cond" in {e.primitive.name for e in r.orig} for r in plan.kernels) == inside
        check(g, (x, u))


def test_forwarded_inputs():
    """Outputs that are inputs come back as new arrays of the inputs' values (as jax.jit's: not computed), with kernels
    and without; the other outputs as jax.jit's."""
    x = jnp.asarray(np.random.default_rng(0).uniform(1.0, 2.0, (4096, 3)))
    idx = jnp.arange(4096)
    for f in (lambda x, idx: (jnp.sqrt(x).sum(1) * jnp.exp(x[:, 0]), idx),  # (a kernel)
              lambda x, idx: (x + 1.0, idx)):  # (XLA)
        g = jit(f)
        g(x, idx)
        g.wait()
        for _ in range(4):
            out = g(x, idx)
        assert out[1] is not idx
        np.testing.assert_array_equal(out[1], idx)
        np.testing.assert_allclose(out[0], jax.jit(f)(x, idx)[0], rtol=1e-12)


def test_donation(tmp_path, monkeypatch):
    """donate_argnums / donate_argnames as in jax.jit: once settled, the donated arguments' buffers go to the program
    (deleted after the call: with kernels or XLA's), also for a function that asks kernelax.tracing(), with build_first
    or not; an input returned as an output stays valid (not donated there)."""
    import kernelax
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    monkeypatch.setattr(auto, "TRIAL_TIME", 0.0)  # (settled after TRIALS timed calls of each)
    f = lambda c: lambda x, y: jnp.sin(x) * jnp.exp(-c * x) + y  # (another program each: built anew)
    x = np.random.default_rng(40).standard_normal((4096, 3))

    def asks(x, y):
        kernelax.tracing()
        return f(2.0)(x, y)
    for h, donate, build_first in ((f(1.0), dict(donate_argnums=0), False), (asks, dict(donate_argnames="x"), False),
                                   (f(3.0), dict(donate_argnames="x"), True)):
        g, ref = jit(h, build_first=build_first, **donate), jax.jit(h)(x, 1.0)
        for _ in range(20):
            np.testing.assert_allclose(g(jnp.array(x), 1.0), ref, rtol=1e-12, atol=1e-12)
            g.wait(variants=True)  # (no trials while variants of K build)
            if g.report()[0]["status"] in ("kernels", "xla"):
                break
        a = jnp.array(x)
        np.testing.assert_allclose(g(a, 1.0), ref, rtol=1e-12, atol=1e-12)
        assert a.is_deleted() and g.report()[0]["status"] in ("kernels", "xla"), g.report()
    f = f(1.0)
    ref = jax.jit(f)(x, 1.0)
    g = jit(lambda x, y: (f(x, y), x), donate_argnums=0)
    for _ in range(4):
        a = jnp.array(x)
        out = g(a, 1.0)
        g.wait()
        np.testing.assert_allclose(out[0], ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(out[1], x)


def test_direct_call(tmp_path, monkeypatch):
    """Once its only signature has settled to kernels, kernelax.jit calls their program directly (jax.jit's dispatch, no
    signature in Python). A new signature (another shape, another static value) is traced there, refused
    (auto._Unplanned) and planned as such; the first keeps its program."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    f, (x,) = PROGRAMS["black_scholes"]
    h = lambda x, n, *, shift: f(x[:, :n] + shift)
    monkeypatch.setattr(cpu, "FIX_K", 1)  # (one variant: settled without a choice of K)
    g = jit(h, static_argnums=1, static_argnames="shift")
    g(x, 3, shift=0.1)
    g.wait(variants=True)
    g(x, 3, shift=0.1)  # (its kernels into the index: the next function uses them, no trials)
    measured(tmp_path)
    k = jit(h, static_argnums=1, static_argnames="shift")
    ref = jax.jit(h, static_argnums=1, static_argnames="shift")
    close = lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-13)
    for _ in range(3):
        close(k(x, 3, shift=0.1), ref(x, 3, shift=0.1))
    assert k.report()[0]["why"] == "index"
    direct = k.__closure__[k.__code__.co_freevars.index("direct")].cell_contents
    assert direct[0] is not None
    close(k(x[:700], 3, shift=0.1), ref(x[:700], 3, shift=0.1))
    assert direct[0] is None and len(k.report()) == 2
    close(k(x, 3, shift=0.2), ref(x, 3, shift=0.2))
    close(k(x, 3, shift=0.1), ref(x, 3, shift=0.1))
    assert len(k.report()) == 3


def test_complex_regions(tmp_path, monkeypatch):
    """Complex values in kernels (realify): as inputs, outputs, between regions (an fft over the samples in XLA) and
    summed over the samples, eig and schur of complex matrices (the Schur form up to its phases); they pass the
    kernels as [re, im] on a trailing axis (an index of its own: built here)."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    r = np.random.default_rng(41)
    x = r.standard_normal((3000, 8))
    z = x + 1j * r.standard_normal((3000, 8))
    progs = [(lambda z: jnp.abs(jnp.fft.fft(z * jnp.exp(1j * jnp.abs(z)), axis=-1)) ** 2, z),
             (lambda x: jnp.fft.rfft(jnp.sin(x) * x) * (1 + 2j), x),
             (lambda x: jnp.abs(jnp.exp(1j * (jnp.fft.fft(jnp.exp(1j * x) * x, axis=0) / 100) ** 2)) + jnp.cos(x), x),
             (lambda x: jnp.sum(jnp.exp(1j * x) * x[:, :1], axis=0), x),
             (lambda x: jnp.sort(jnp.abs(jnp.linalg.eigvals(x[:, :4, None] * x[:, None, 4:] + jnp.eye(4))), axis=-1), x),
             (lambda z: jnp.linalg.eig(jnp.sin(z[:, :4, None] * z[:, None, 4:])), z, 1e-9),  # (zgeev's vectors: unique;
             (lambda z: (lambda T, Z: (jnp.abs(T), Z @ T @ Z.conj().swapaxes(-1, -2)))(  # rounding x conditioning)
                 *jax.scipy.linalg.schur(jnp.sin(z[:, :4, None] * z[:, None, 4:]))), z, 1e-9)]
    for f, a, *tol in progs:
        g = jit(f, build_first=True)
        for o, x in zip(jax.tree.leaves(g(a)), jax.tree.leaves(jax.jit(f)(a)), strict=True):
            np.testing.assert_allclose(o, x, rtol=(tol or [1e-11])[0], atol=1e-11 * (1 + np.abs(np.where(np.isfinite(x), x, 0)).max(initial=0)))
        rep = g.report()[0]
        assert rep["status"] == "trial" and all(q["chosen"] == "c" for q in rep["regions"]), rep


def test_tracing_signal(tmp_path, monkeypatch):
    """kernelax.tracing(): true while kernelax traces a program for its kernels (kernelax.jit, cvmap), else false; a library
    function under jax.jit takes its path for kernels there (here other values, to see which one ran) and is cached
    apart (JAX's trace context), so that jax.jit afterwards still gets XLA's path."""
    import kernelax
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    seen = []

    @jax.jit
    def inner(x):
        seen.append(kernelax.tracing())
        return x * (2.0 if kernelax.tracing() else 3.0)
    x = np.random.default_rng(43).uniform(size=(3000, 3))
    np.testing.assert_allclose(jax.jit(lambda x: jnp.sin(inner(x)) + x)(x), np.sin(3 * x) + x, rtol=1e-12)
    np.testing.assert_allclose(jit(lambda x: jnp.sin(inner(x)) + x, build_first=True)(x), np.sin(2 * x) + x, rtol=1e-12)
    np.testing.assert_allclose(jax.jit(lambda x: jnp.cos(inner(x)))(x), np.cos(3 * x), rtol=1e-12)
    from kernelax.vmap import cvmap
    np.testing.assert_allclose(cvmap(lambda v: jnp.sin(inner(v)) * v, [x[0]], (0,))(x), np.sin(2 * x) * x, rtol=1e-12)
    assert True in seen and False in seen and not kernelax.tracing(), seen


def test_parts_first_then_the_whole_file(tmp_path, monkeypatch):
    """A large block kept whole only for its reads from the frame (emit.FRAME_MAX) builds in parts first (gcc
    compiles them in parallel); its whole file follows in the background (Kernel.complete), replaces the parts in the
    runtime and enters the index (here every block is large and cut into small parts)."""
    from kernelax import cpu, emit
    from kernelax.build import PART_MARK
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    monkeypatch.setattr(cpu, "BIG", 0)
    monkeypatch.setattr(emit, "SPLIT_MIN", 0)
    monkeypatch.setattr(emit, "PART", 1500)
    energy = lambda v: jnp.sum(jnp.sin(v * v[::-1]) * jnp.cos(v) + jnp.exp(-v ** 2) * v[0])
    f = lambda x: jax.vmap(jax.grad(energy))(x) + x  # (the forward pass's values read in the reverse pass)
    x = np.random.default_rng(47).uniform(size=(4000, 12))
    before = set(cpu._KERNELS)
    g = jit(f, build_first=False)
    np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-12)
    g.wait(variants=True)
    np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-12)
    new = [k for key, k in cpu._KERNELS.items() if key not in before]
    upgraded = [k for k in new if len(k.libs) > len(k.paths)]  # (the parts' libraries kept, the paths the whole file's)
    assert upgraded and all(PART_MARK not in open(os.path.join(os.path.dirname(p), "kernel.c")).read() for k in upgraded for p in k.paths)


def test_one_trace_without_the_signal(tmp_path, monkeypatch):
    """A program that does not ask kernelax.tracing() is traced once: its trace holds for XLA's program too (the first
    call's, while the kernels build)."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    traces = []

    def f(x):
        traces.append(1)
        return jnp.sin(x) * 2.0 + x
    x = np.random.default_rng(46).uniform(size=(3000, 3))
    g = jit(f, build_first=False)
    np.testing.assert_allclose(g(x), np.sin(x) * 2.0 + x, rtol=1e-13)
    g.wait()
    np.testing.assert_allclose(g(x), np.sin(x) * 2.0 + x, rtol=1e-13)
    assert len(traces) == 1, len(traces)


def test_cost_model_sees_xlas_own_program(tmp_path, monkeypatch):
    """Without kernels XLA's own program runs: for a program that asks kernelax.tracing(), the heavy operations only
    that one has count for every region (jnp.linalg.det's derivative: an LU solve in XLA's program, 2 operations per
    sample in the kernels, which the cost model left to XLA: 297 instead of 2.3 ms on 2^20 samples); noted in the
    index, so that a known program is planned without a trace of XLA's program."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    f = jax.vmap(jax.grad(jnp.linalg.det))
    x = np.eye(2) + 0.1 * np.random.default_rng(48).standard_normal((3000, 2, 2))
    for known in (False, True):  # (a new kernelax.jit: the program from the index)
        g = jit(f, build_first=True)
        np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-13)
        rep = g.report()[0]
        assert rep["status"] != "xla" and [r["chosen"] for r in rep["regions"]] == ["c"], rep
        assert {"lu", "triangular_solve"} <= set(rep["xla_heavy"]) and "lu" not in rep["regions"][0]["heavy"], rep
        monkeypatch.setattr(auto, "_prims", lambda j: pytest.fail("XLA's program traced for a known program's plan"))


def test_kernels_on_one_thread_in_the_callers_float_mode(tmp_path, monkeypatch):
    """Calls through XLA run on kernelax's kernel thread (one OpenMP pool: XLA calls from several threads, whose idle pools
    took the cores from the working one), in the calling thread's floating-point mode: XLA's threads flush
    subnormals, and so do the kernels. The pool's threads are bound, one CPU each, where the process's affinity is
    restricted (taskset, MPI: unless the environment binds); not where it may run anywhere (processes side by side)."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    f = lambda x: jnp.sin(x) * 1e-10 + x * 1e-12  # (subnormal results)
    x = np.full((5000, 3), 1e-300)
    g = jit(f, build_first=True)
    out, ref = g(x), jax.jit(f)(x)
    assert g.report()[0]["status"] == "trial"
    np.testing.assert_array_equal(out, ref)
    assert np.all(np.asarray(ref) == 0)
    pool = [t for t in os.listdir("/proc/self/task") if open(f"/proc/self/task/{t}/comm").read().strip() == "kernelax_pool"]
    assert pool
    if "OMP_PROC_BIND" not in os.environ and "GOMP_CPU_AFFINITY" not in os.environ:
        cpus, own = [os.sched_getaffinity(int(t)) for t in pool], os.sched_getaffinity(0)
        if len(own) < os.sysconf("SC_NPROCESSORS_ONLN"):
            assert all(len(c) == 1 for c in cpus) and len(set().union(*cpus)) == min(len(pool), len(own)), cpus
        else:
            assert all(c == own for c in cpus), cpus


def test_platform_dependent_in_kernels(tmp_path, monkeypatch):
    """jax.lax.platform_dependent with a branch for the CPU (e.g. a library's C kernel through FFI; here a host
    callback) and a default: kernels run the default branch, so its region is a kernel; XLA keeps the CPU's (an index
    of its own: built here)."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    calls = []

    def f(x):
        impl = lambda v: jnp.sin(v) * 2.0
        host = lambda v: jax.pure_callback(lambda a: calls.append(1) or np.sin(a) * 2.0, jax.ShapeDtypeStruct(v.shape, v.dtype),
                                           v, vmap_method="broadcast_all")
        return jnp.cos(jax.lax.platform_dependent(x, cpu=host, default=impl) * x) + x
    x = np.random.default_rng(42).standard_normal((4000, 3))
    g = jit(f, build_first=True)
    np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-12)
    rep = g.report()[0]
    assert rep["status"] == "trial" and [q["chosen"] for q in rep["regions"]] == ["c"], rep
    calls.clear()
    np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-12)  # (XLA: the callback once)
    assert len(calls) == 1


def test_scatter_plan_sees_sign_changes(monkeypatch):
    """The cached plan of a sum into a large array is reused only for the same positions: inputs that differ in the
    signs of two of their values (which an FNV hash of the words without mixing did not see) get a new plan."""
    from kernelax import emit
    monkeypatch.setattr(emit, "SCAT_BIG", 1)  # (sums into large arrays: through buffers and a cached plan)
    f = lambda x: jnp.zeros(64).at[jnp.floor(x).astype(jnp.int32)].add(x)
    x = np.linspace(1.5, 30.5, 16)
    y = x.copy()
    y[0], y[2] = -y[0], -y[2]  # (words 0 and 2: the same chain of the hash)
    run, _ = all_kernels(f, x)
    for a in (x, y, x, y):
        np.testing.assert_allclose(run(a)[0], jax.jit(f)(a), rtol=1e-12)


def test_wait_returns_after_a_donated_first_result(monkeypatch):
    """The build thread waits for the first call's results; donated to another program meanwhile (deleted), it
    builds anyway, and wait() returns (it hung: the thread died before setting ready)."""
    import threading
    import time
    orig = jax.block_until_ready
    monkeypatch.setattr(jax, "block_until_ready", lambda x: (time.sleep(1.0), orig(x))[1])
    g = jit(lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda y: y[0] < 4.0, lambda y: y * 1.1 + r[0], r))(x), donate_argnums=0)  # (worth kernels: built in a thread)
    x = np.random.default_rng(41).uniform(0.5, 1.5, (2000, 3))
    y = g(jnp.array(x))  # (the build thread waits for y)
    jax.jit(lambda v: v * 2.0, donate_argnums=0)(y)  # (another program donates y, e.g. an optimizer's step)
    assert y.is_deleted()
    t = threading.Thread(target=g.wait, daemon=True)
    t.start()
    t.join(300)
    assert not t.is_alive() and g.report()[0]["status"] != "building", g.report()


def test_a_failed_build_is_tried_again(tmp_path):
    """Kernels whose build failed for a passing reason (here gcc, as on a full disk) are built again by the next
    process (not noted as a program without kernels)."""
    import subprocess
    cc = tmp_path / "cc.sh"
    cc.write_text('#!/bin/sh\n[ -n "$KX_FAIL" ] && [ "$1" != "--version" ] && { echo "failing on purpose" >&2; exit 1; }\nexec gcc "$@"\n')
    cc.chmod(0o755)
    script = tmp_path / "once.py"
    script.write_text("""import warnings, jax, jax.numpy as jnp, numpy as np
from kernelax import jit
warnings.simplefilter("ignore")
jax.config.update("jax_enable_x64", True)
g = jit(lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda y: y[0] < 4.0, lambda y: y * 1.1 + r[0], r))(x), build_first=True)
g(np.random.default_rng(42).uniform(0.5, 1.5, (2000, 3)))
r = g.report()[0]
print(r["status"], r.get("why"))
""")
    env = dict(os.environ, KERNELAX_CACHE_DIR=str(tmp_path / "cache"), KERNELAX_CC=str(cc), JAX_PLATFORMS="cpu", KERNELAX_EXIT_WAIT="0")
    run = lambda **e: subprocess.run([sys.executable, str(script)], env=dict(env, **e), capture_output=True, text=True, timeout=600)
    first = run(KX_FAIL="1")
    assert first.returncode == 0 and first.stdout.strip() == "xla not built", first.stdout + first.stderr
    second = run()
    assert second.returncode == 0 and second.stdout.strip() == "trial None", second.stdout + second.stderr


def test_first_calls_from_threads():
    """A kernelax.jit function called from several threads at once: one entry, one build, the results of each."""
    import threading
    g = jit(lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda y: y[0] < 4.0, lambda y: y * 1.1 + r[0], r))(x))
    x = np.random.default_rng(43).uniform(0.5, 1.5, (2000, 3))
    out = {}
    ts = [threading.Thread(target=lambda i=i: out.__setitem__(i, [g(x) for _ in range(5)])) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    g.wait(variants=True)
    ref = jax.jit(g.__wrapped__)(x)
    assert len(out) == 8 and len(g.report()) == 1
    for o in out.values():
        for y in o + [g(x)]:
            np.testing.assert_allclose(y, ref, rtol=1e-12)


def test_forwarded_inputs_are_new_arrays():
    """Outputs that are inputs come back as new arrays (as jax.jit's): donating them later keeps the caller's input."""
    g = jit(lambda x, y: (x, jax.vmap(lambda r: jax.lax.while_loop(lambda y: y[0] < 4.0, lambda y: y * 1.1 + r[0], r))(x) * y))
    x = jnp.asarray(np.random.default_rng(44).uniform(0.5, 1.5, (2000, 3)))
    for _ in range(6):
        a, b = g(x, 2.0)
        g.wait(variants=True)
        assert a is not x
        jax.jit(lambda v: v + 1.0, donate_argnums=0)(a)
        assert not x.is_deleted()
    np.testing.assert_allclose(g(x, 2.0)[1], jax.jit(g.__wrapped__)(x, 2.0)[1], rtol=1e-12)


def test_side_effects_stay_in_xla(capfd):
    """jax.debug.print in a kernelax.jit program: printed at every call (XLA), not dropped nor printed while kernels
    are traced; the rest of the program may run with kernels."""
    def f(x):
        jax.debug.print("sum {s}", s=jnp.sum(x[:2]))
        return jnp.sin(x) * jnp.exp(-x)
    g = jit(f, build_first=True)
    x = np.random.default_rng(45).standard_normal((4096, 3))
    capfd.readouterr()
    for _ in range(3):
        np.testing.assert_allclose(g(x), jnp.sin(x) * jnp.exp(-x), rtol=1e-12)
    jax.effects_barrier()
    assert capfd.readouterr().out.count("sum ") == 3


def test_no_c_compiler(tmp_path):
    """Without a working C compiler (KERNELAX_CC): kernelax.jit runs in XLA and says why; kernelax.kernel stops with
    an error naming the compiler."""
    import subprocess
    script = tmp_path / "nocc.py"
    script.write_text("""import warnings, jax, jax.numpy as jnp, numpy as np
from kernelax import jit, kernel
warnings.simplefilter("ignore")
jax.config.update("jax_enable_x64", True)
x = np.random.default_rng(49).uniform(0.5, 1.5, (2000, 3))
g = jit(lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda y: y[0] < 4.0, lambda y: y * 1.1 + r[0], r))(x), build_first=True)
assert np.allclose(g(x), jax.jit(g.__wrapped__)(x))
r = g.report()[0]
print(r["status"], "KERNELAX_CC" in " ".join(r.get("errors", [])))
try:
    jax.vmap(kernel(jnp.sin))(x)
except Exception as e:
    print(type(e).__name__, "KERNELAX_CC" in str(e))
""")
    env = dict(os.environ, KERNELAX_CACHE_DIR=str(tmp_path / "cache"), KERNELAX_CC=str(tmp_path / "no-such-gcc"), JAX_PLATFORMS="cpu",
               KERNELAX_EXIT_WAIT="0")
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.split() == ["failed", "True", "RuntimeError", "True"], r.stdout + r.stderr


def test_gcc_version(tmp_path):
    """gcc >= 12 (build.GCC_MIN): an older one is refused with the reason and how to get a newer one (kernelax.jit:
    XLA); without KERNELAX_CC the newest of gcc and the gcc-NN on the PATH is taken (gcc 15 over a gcc 13: GCC PR
    target/117159 in gcc 12-14)."""
    import subprocess
    real = subprocess.run(["which", "gcc"], capture_output=True, text=True).stdout.strip()
    old = tmp_path / "gcc11"
    old.write_text(f'#!/bin/sh\n[ "$1" = -dumpversion ] && {{ echo 11.4.0; exit 0; }}\nexec {real} "$@"\n')
    old.chmod(0o755)
    script = tmp_path / "old.py"
    script.write_text("""import warnings, jax, jax.numpy as jnp, numpy as np
from kernelax import kernel
try:
    jax.vmap(kernel(jnp.sin))(np.ones((5, 3)))
except RuntimeError as e:
    print("refused", "gcc >= 12" in str(e), "conda-forge" in str(e))
""")
    env = dict(os.environ, KERNELAX_CACHE_DIR=str(tmp_path / "cache"), KERNELAX_CC=str(old), JAX_PLATFORMS="cpu", KERNELAX_EXIT_WAIT="0")
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    assert r.stdout.split() == ["refused", "True", "True"], r.stdout + r.stderr
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    g13 = tmp_path / "gcc13"
    g13.write_text(f'#!/bin/sh\n[ "$1" = -dumpversion ] && {{ echo 13.3.0; exit 0; }}\nexec {real} "$@"\n')
    g13.chmod(0o755)
    (bin_ / "gcc").symlink_to(g13)  # (new enough, but not the newest)
    (bin_ / "gcc-16").symlink_to(real)
    env = dict(os.environ, PATH=f"{bin_}{os.pathsep}{os.environ['PATH']}", KERNELAX_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("KERNELAX_CC", None)
    r = subprocess.run([sys.executable, "-c", "import kernelax.build as b; print(b.CC)"], env=env, capture_output=True, text=True, timeout=300)
    assert r.stdout.strip() == "gcc-16", r.stdout + r.stderr


def test_options_in_use(monkeypatch):
    """The decorators with options alone; a valid n_threads; kernelax.kernel's keep_nans; the report's N and, after
    the trials, ms; KERNELAX_BUILD_FIRST (auto.BUILD_FIRST: build_first by default)."""
    from kernelax import kernel
    monkeypatch.setattr(auto, "TRIAL_TIME", 0.0)

    @jit(n_threads=2)
    def g(x):
        return jax.vmap(lambda r: jax.lax.while_loop(lambda y: y[0] < 4.0, lambda y: y * 1.1 + r[0], r))(x)
    x = np.random.default_rng(50).uniform(0.5, 1.5, (2000, 3))
    ref = jax.jit(g.__wrapped__)(x)
    for _ in range(30):
        np.testing.assert_allclose(g(x), ref, rtol=1e-12)
        g.wait(variants=True)
        if g.report()[0]["status"] in ("kernels", "xla"):
            break
    r = g.report()[0]
    assert r["why"] == "measured" and set(r["ms"]) == {"c", "xla"} and r["N"], r

    @kernel(keep_nans=True)
    def nan_kept(v):
        return v * 0.0
    np.testing.assert_array_equal(jax.vmap(nan_kept)(np.array([1.0, np.inf, np.nan])), [0.0, np.nan, np.nan])
    monkeypatch.setattr(auto, "BUILD_FIRST", True)
    h = jit(lambda x: g.__wrapped__(x) * 2.0)
    np.testing.assert_allclose(h(x), 2.0 * ref, rtol=1e-12)
    assert h.report()[0]["status"] == "trial"


def test_trials_with_many_rows(tmp_path, monkeypatch):
    """The first call with kernels of at least cpu.TUNE_MIN rows chooses K (TUNE_TIME per variant): not the call that
    TRIAL_MAX_CALL judges (it took every new program of 4096 rows or more past it: no trials, 'long calls')."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    monkeypatch.setattr(auto, "TRIAL_TIME", 0.0)  # (TRIALS calls each, however short)
    f = lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda y: y[0] < 4.0, lambda y: y * 1.1 + r[0] * 0.5, r))(x)
    x = np.random.default_rng(65).uniform(0.5, 1.5, (8192, 3))
    g = jit(f)
    g(x)
    g.wait(variants=True)
    for _ in range(20):
        np.testing.assert_allclose(g(x), jax.jit(f)(x), rtol=1e-12)
        if g.report()[0]["status"] in ("kernels", "xla"):
            break
    rep = g.report()[0]
    assert rep["why"] == "measured" and "ms" in rep, rep


def test_tracing_seen_through_cached_inner_traces(tmp_path, monkeypatch):
    """A library function under jax.jit that asks kernelax.tracing(), traced for a first program's kernels: a second
    program finds its trace in jax.jit's cache (no call of tracing()), and still runs XLA's path in XLA (it took the
    kernels' path there)."""
    import kernelax
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))

    @jax.jit
    def assemble(x):
        if kernelax.tracing():
            return jax.vmap(lambda r: jnp.sin(r) * jnp.exp(-r @ r) + 1.0)(x)  # (+1: the kernels' path, to see it)
        return jax.lax.map(lambda r: jnp.sin(r) * jnp.exp(-r @ r), x, batch_size=256)
    x = np.random.default_rng(66).standard_normal((4000, 3))
    jit(lambda x: assemble(x) * 1.0, build_first=False)(x)
    g = jit(lambda x: assemble(x) * 2.0, build_first=False)
    np.testing.assert_allclose(g(x), jax.jit(lambda x: assemble(x) * 2.0)(x), rtol=1e-13)  # (the first call: XLA's)


def test_weak_types_through_kernels():
    """Weakly typed outputs (a loop's counter) stay so with the kernels, as under jax.jit (the type changed when the
    program went from XLA to its kernels)."""
    x = np.random.default_rng(67).uniform(0.5, 1.5, (3000, 3))
    f = lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda c: c[0][0] < 4.0, lambda c: (c[0] * 1.1 + 0.1, c[1] + 1), (r, 0)))(x)
    run, _ = all_kernels(f, x)
    for o, r in zip(run(x), jax.jit(f)(x)):
        assert jax.typeof(o) == jax.typeof(r), (jax.typeof(o), jax.typeof(r))


def test_report_tells_x64_apart(tmp_path, monkeypatch):
    """A signature called with and without x64: two entries of report() that read apart."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    g, x = jit(lambda x: jnp.sin(x) * 2.0), np.ones((2000, 3), np.float32)
    g(x)
    with jax.enable_x64(False):
        g(x)
    sigs = [r["signature"] for r in g.report()]
    assert len(sigs) == 2 and len(set(sigs)) == 2, sigs


_ONCE = """import warnings, jax, jax.numpy as jnp, numpy as np
from kernelax import jit
warnings.simplefilter("ignore")
jax.config.update("jax_enable_x64", True)
g = jit(lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda y: y[0] < 4.0, lambda y: y * 1.1 + r[0], r))(x), build_first=True)
x = np.random.default_rng(42).uniform(0.5, 1.5, (2000, 3))
np.testing.assert_allclose(g(x), jax.jit(g.__wrapped__)(x), rtol=1e-12)
r = g.report()[0]
print(r["status"], r.get("why"))
"""


def _run_once(tmp_path, **env):
    import subprocess
    script = tmp_path / "once.py"
    script.write_text(_ONCE)
    env = dict(os.environ, KERNELAX_CACHE_DIR=str(tmp_path / "cache"), JAX_PLATFORMS="cpu", KERNELAX_EXIT_WAIT="0", **env)
    return subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)


def test_node_limit_in_the_program_key(tmp_path):
    """A program whose kernels exceeded KERNELAX_MAX_NODES is built once the limit is raised (the limit is part of the
    program's key; it stayed 'not built')."""
    first = _run_once(tmp_path, KERNELAX_MAX_NODES="10")
    assert first.returncode == 0 and first.stdout.strip() == "xla not built", first.stdout + first.stderr
    second = _run_once(tmp_path)
    assert second.returncode == 0 and second.stdout.strip() == "trial None", second.stdout + second.stderr


def test_damaged_files_in_the_cache(tmp_path):
    """Libraries in the cache cut short (a crash, a full disk: dlopen crashed with SIGBUS) or empty (it failed in every
    process) are built anew; an index entry that is no dict counts as none."""
    import glob
    assert _run_once(tmp_path).stdout.strip() == "trial None"
    libs = glob.glob(str(tmp_path / "cache" / "**" / "*.so"), recursive=True)
    assert libs
    for damage in (lambda p: os.truncate(p, 1000), lambda p: os.truncate(p, 0)):
        for p in glob.glob(str(tmp_path / "cache" / "**" / "*.so"), recursive=True):
            damage(p)
        r = _run_once(tmp_path)
        assert r.returncode == 0 and r.stdout.strip() == "trial None", r.stdout + r.stderr  # (with kernels, built anew)
    for p in glob.glob(str(tmp_path / "cache" / "index" / "*.json")):
        open(p, "w").write("[]")
    r = _run_once(tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == "trial None", r.stdout + r.stderr


def test_settings_from_the_environment(tmp_path):
    """KERNELAX_* values that are not valid fail at import, naming the variable (KERNELAX_JOBS=-1 hung every build);
    empty ones count as unset (an empty KERNELAX_CACHE_DIR made the current directory the cache); KERNELAX_BUILD_FIRST
    takes true and false too."""
    import subprocess
    run = lambda code, **e: subprocess.run([sys.executable, "-c", code], env=dict(os.environ, **e), capture_output=True,
                                           text=True, timeout=300)
    for name, value in (("KERNELAX_JOBS", "-1"), ("KERNELAX_GCC_TIMEOUT", "nan"), ("KERNELAX_EXIT_WAIT", "x"),
                        ("KERNELAX_BUILD_FIRST", "maybe"), ("KERNELAX_MAX_NODES", "0"), ("KERNELAX_NVCC_JOBS", "-1")):
        r = run("import kernelax", **{name: value})
        assert r.returncode != 0 and name in r.stderr, (name, r.stderr[-500:])
    r = run("import kernelax.build as b, kernelax.auto as a; print(b.CACHE, a.BUILD_FIRST)", KERNELAX_CACHE_DIR="",
            KERNELAX_BUILD_FIRST="true", HOME=str(tmp_path))
    assert r.stdout.split() == [str(tmp_path / ".cache" / "kernelax"), "True"], r.stdout + r.stderr


def test_index_writes_that_fail(tmp_path, monkeypatch):
    """An index that cannot be written (a full disk, a quota, no rights) is warned about once; the calls go on with
    their choice (each call raised, after its results were computed: the writes were tried in the caller's thread, at
    every call)."""
    import glob
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    monkeypatch.setattr(auto, "TRIAL_TIME", 0.0)
    c = np.random.default_rng(70).uniform(0.5, 1.5, (2000, 3))
    f = lambda x: jnp.sum(jax.vmap(lambda r: jax.lax.while_loop(lambda y: y < 4.0, lambda y: y * 1.3 + r[0], r[1]))(x))
    g = jit(f, build_first=True)
    g(c)
    g.wait(variants=True)
    for p in glob.glob(os.path.join(str(tmp_path), "program-*.json")):  # (measured with other threads only: trials)
        cpu._update(os.path.basename(p)[:-5], lambda e: dict(e, use={str(cpu.NT + 1): "c"}))

    def full(*a):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(auto, "_update", full)
    monkeypatch.setattr(cpu, "_update", full)
    h = jit(f)  # (its kernels from the index: trials, then the choice written)
    with pytest.warns(UserWarning) as w:
        for _ in range(8):
            np.testing.assert_allclose(h(c), f(c), rtol=1e-12)
    assert [str(x.message) for x in w if "could not be written" in str(x.message)] == [
        "kernelax.jit: the index on disk could not be written, this program's choices are not kept: [Errno 28] No space left on device"]
    assert h.report()[0]["status"] in ("kernels", "xla")


def test_warnings_name_why_a_region_stays_in_xla(tmp_path, monkeypatch):
    """A region whose gcc runs out of time is warned as too large (it read "not covered", as for an operation kernelax
    lacks)."""
    from kernelax import build, cpu

    def slow(*a, **k):
        raise build.GccTimeout("kernelax: gcc took more than the limit (test)")
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    monkeypatch.setattr(cpu, "compile_c", slow)
    c = float(np.random.default_rng().uniform(1, 2))  # (a kernel not built before in this process)
    f, x = (lambda x: jnp.sin(x) * c + x), np.ones((5000, 3))
    g = jit(f, build_first=True)
    with pytest.warns(UserWarning) as w:
        np.testing.assert_allclose(g(x), f(x), rtol=1e-12)
        g.wait()
    msgs = [str(m.message) for m in w if "stays in XLA" in str(m.message)]
    assert msgs and all("too large: GccTimeout" in m and "not covered" not in m for m in msgs), msgs


def test_build_ending_during_the_first_xla_call(tmp_path, monkeypatch):
    """A build that outlasts BUILD_FIRST_WAIT and ends while the first call runs in XLA: the calls after it run the
    kernels in trials (it settled the process on XLA's program, the kernels unused); the gcc started after the wait ran
    out run in the background (their time limit counting CPU seconds, not a wall-clock one beside XLA's call)."""
    import time
    from kernelax import build, cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    monkeypatch.setattr(auto, "BUILD_FIRST_WAIT", 1e-3)
    monkeypatch.setattr(auto, "TRIAL_TIME", 0.0)
    full, gcc, seen = auto._Entry._full, build._gcc, []

    def late(self, outs, leaves):  # (this XLA call ends after the build)
        self.ready.wait()
        return full(self, outs, leaves)

    def spy(*a):
        time.sleep(0.05)  # (after the first call stopped waiting)
        seen.append(bool(getattr(build.BACKGROUND, "on", False)))
        return gcc(*a)
    monkeypatch.setattr(auto._Entry, "_full", late)
    monkeypatch.setattr(build, "_gcc", spy)
    c = float(np.random.default_rng().uniform(1, 2))  # (kernels not built before)
    f = lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda y: y[0] < 4.0, lambda y: y * 1.1 + r[0] * c, r))(x)
    x = np.random.default_rng(3).uniform(0.5, 1.5, (2000, 3))
    g, ref = jit(f, build_first=True), jax.jit(f)(x)
    for _ in range(12):
        np.testing.assert_allclose(g(x), ref, rtol=1e-12)
        g.wait(variants=True)
    r = g.report()[0]
    assert r["status"] in ("kernels", "xla") and r.get("why") in ("measured", "long calls"), r
    assert seen and all(seen), seen


def test_first_call_without_a_wait_limit(tmp_path):
    """KERNELAX_BUILD_FIRST_WAIT=inf: the first call waits for its kernels however long (it raised OverflowError)."""
    import subprocess
    code = """import jax, jax.numpy as jnp, numpy as np
import kernelax
jax.config.update("jax_enable_x64", True)
g = kernelax.jit(lambda x: jnp.sin(x) * 2.0 + x)
print(np.allclose(g(np.ones((5000, 3))), np.sin(1.0) * 2.0 + 1.0), g.report()[0]["status"])
"""
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, KERNELAX_BUILD_FIRST_WAIT="inf", KERNELAX_EXIT_WAIT="0",
                                                                KERNELAX_CACHE_DIR=str(tmp_path / "cache"), JAX_PLATFORMS="cpu"),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.split() == ["True", "trial"], r.stdout + r.stderr[-2000:]


def test_settled_below_tune_min(tmp_path, monkeypatch):
    """A program whose kernel calls have fewer samples than cpu.TUNE_MIN (K never chosen, nothing to store) settles after
    its trials: it is called directly (it stayed unsettled, every call through the signature and the lock, waited)."""
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    monkeypatch.setattr(auto, "TRIAL_TIME", 0.0)
    c = float(np.random.default_rng().uniform(1, 2))
    f = lambda x: jax.vmap(lambda r: jax.lax.while_loop(lambda y: y[0] < 4.0, lambda y: y * 1.1 + r[0] * c, r))(x)
    x = np.random.default_rng(5).uniform(0.5, 1.5, (1000, 3))
    g, ref = jit(f, build_first=True), jax.jit(f)(x)
    for _ in range(12):
        np.testing.assert_allclose(g(x), ref, rtol=1e-12)
        g.wait(variants=True)
    r = g.report()[0]
    direct = g.__closure__[g.__code__.co_freevars.index("direct")].cell_contents
    assert r["status"] == "xla" or direct[0] is not None, r


def test_entries_added_by_other_threads(monkeypatch):
    """Calls from several threads with new signatures: kernelax.jit's summary of its entries (all settled to XLA, the
    only one) is taken under its lock (it read the dict while another thread added to it: RuntimeError, the result of
    that call lost)."""
    import threading
    import time

    class Slow(auto._Entry):  # (reading settled yields to the other threads)
        settled = property(lambda e: (time.sleep(0.02), e.__dict__.get("_settled"))[1], lambda e, v: e.__dict__.update(_settled=v))
    monkeypatch.setattr(auto, "_Entry", Slow)
    g, errors = jit(lambda x: x * 2.0 + 1.0), []

    def call(n):
        try:
            for _ in range(3):
                np.testing.assert_allclose(g(np.ones(n)), 3.0)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=call, args=(n,)) for n in range(1, 13)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_k_by_the_threads_of_the_calls(tmp_path, monkeypatch):
    """The index keeps K per thread count of the calls that chose it and finds it by the threads of the calls (it
    looked under NT: kernels with n_threads loaded NT's choice); kernelax.kernel's choice reaches the index too, at
    exit (only kernelax.jit stored it: each process chose again)."""
    from kernelax import cpu, kernel
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    before = set(cpu._KERNELS)  # (kernels of earlier tests in this process: _persist enters them here too, by their threads)
    f = lambda x: jnp.sin(x) * 1.375 + jnp.cos(x)
    k = cpu.kernel(f, [np.zeros(3)], [True], cached=True)
    assert len(k.Ks) > 1
    cpu._update(k.index, lambda e: dict(e, K={str(cpu.NT + 1): k.Ks[-1], str(cpu.NT): k.Ks[0]}))
    del cpu._KERNELS[k.key]
    assert cpu.kernel(f, [np.zeros(3)], [True], cached=True, nt=cpu.NT + 1).Ks == [k.Ks[-1]]
    g = kernel(lambda x: jnp.tanh(x) * 0.625 + x)
    jax.vmap(g)(np.ones((cpu.TUNE_MIN, 3)))  # (K = 1; its variants build in a thread, chosen at the next call)
    for t in auto._THREADS:
        t.join()
    jax.vmap(g)(np.ones((cpu.TUNE_MIN, 3)))
    cpu._persist()
    mine = [q for q in cpu._KERNELS.values() if q.index and os.path.exists(os.path.join(str(tmp_path), q.index + ".json"))
            and q.key != k.key and q.key not in before]
    assert mine and all(cpu._read(q.index)["K"] == {str(cpu.NT): q.K()} for q in mine)


def test_single_kernel_limits():
    """A program with a loop over small data is a whole-program kernel (_Single), not with a side effect (jax.debug.print:
    it was planned, failed to build, and stood as "not built" in the index), nor with a loop over more than SINGLE_CARRY
    values (one lane of the vectors working: slower than XLA's loop, gcc's time quadratic in them: 75 s for 2048)."""
    loop = lambda p: lambda x: jax.lax.while_loop(lambda c: c[0] < 3, lambda c: (p(c[0]) or c[0] + 1, c[1] * 2), (0, x))
    assert auto._Single.suits(jax.make_jaxpr(loop(lambda i: None))(1.0))
    assert not auto._Single.suits(jax.make_jaxpr(loop(lambda i: jax.debug.print("{}", i)))(1.0))
    assert auto._Single.suits(jax.make_jaxpr(loop(lambda i: None))(np.ones(auto.SINGLE_CARRY - 1)))
    assert not auto._Single.suits(jax.make_jaxpr(loop(lambda i: None))(np.ones(1000)))


def test_xla_program_compiled_once(monkeypatch):
    """A program that stays in XLA (here: no region worth a kernel) compiles once: its program from the trace serves
    the further calls (they went to jax.jit's, compiled a second time), until their longer dispatch
    has cost as much as that compile (DISPATCH): then jax.jit's."""
    import logging
    seen = []
    handler = logging.Handler()
    handler.emit = lambda r: seen.append(r.getMessage()) if "Finished XLA compilation" in r.getMessage() else None
    logging.getLogger("jax").addHandler(handler)
    try:
        for dispatch, compiles in ((1e-5, 1), (1e3, 2)):
            monkeypatch.setattr(auto, "DISPATCH", dispatch, raising=False)
            seen.clear()
            g = jit(lambda x: x * 3.0 + dispatch)
            with jax.log_compiles(True):
                g(np.ones(5))
                g.wait()  # (planned in a thread: no kernel)
                for _ in range(6):
                    np.testing.assert_allclose(g(np.ones(5)), 3.0 + dispatch)
            assert g.report()[0]["status"] == "xla" and len(seen) == compiles, seen
    finally:
        logging.getLogger("jax").removeHandler(handler)


def test_cost_model_of_contractions():
    """Contractions count with their length (a 48 x 48 product per sample: 110592, not 2304), sums
    of outer products over the rows (X^T X) by their size (0 before); regions with more than DOT_MAX
    of them (gcc's time for a tie) or with a dense layer of CPU_DENSE_MIN weights or more (XLA's GEMM is faster) stay
    in XLA."""
    rng = np.random.default_rng(71)
    regions = lambda f, *a: [(r["work"], r["chosen"]) for r in auto._Plan(jaxprs._flatten(f, a), ()).report["regions"]]
    bmm = lambda a, b: jnp.einsum("sij,sjk->sik", a, b)
    assert regions(bmm, *rng.standard_normal((2, 4096, 32, 32))) == [(32768, "c")]
    assert regions(bmm, *rng.standard_normal((2, 4096, 48, 48))) == [(110592, "xla")]
    gram = lambda x: jnp.sin(x).T @ jnp.sin(x)
    assert regions(gram, rng.standard_normal((4000, 100))) == [(10200, "c")]
    assert regions(gram, rng.standard_normal((4000, 150))) == [(22800, "xla")]
    dense = lambda d: lambda x: jnp.tanh(x @ np.full((d, d), 0.1))
    assert regions(dense(48), rng.standard_normal((4000, 48))) == [(2352, "c")]
    assert regions(dense(64), rng.standard_normal((4000, 64))) == [(4160, "xla")]


def test_sums_the_same_for_every_k(tmp_path, monkeypatch):
    """Sums over the samples (a reduced output, a scatter-add into a small array) give the same bits with each variant
    of K: the threads' samples are whole groups of lcm(K) W samples (the first large call chooses K: the bits changed
    within a process)."""
    import ctypes
    from kernelax import cpu
    from kernelax.build import runtime
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    f = lambda x, i: (jnp.sin(x) * x[0], jnp.zeros(7).at[i[0] % 7].add(jnp.cos(x[1])))
    k = cpu.kernel(f, [np.zeros(3), np.zeros(1, np.int32)], [True, True], reduce=[True, True])
    assert k.Ks == [1, 2, 3, 4]
    rng, n = np.random.default_rng(72), 12 * cpu.simd_width() * 4 * 7 + 13  # (a remainder too)
    x, i = rng.standard_normal((n, 3)), rng.integers(0, 100, (n, 1)).astype(np.int32)
    outs = []
    for v in range(len(k.Ks)):
        runtime().kernelax_publish(ctypes.byref(k.struct), len(k.Ks), v)
        out = [np.zeros(3), np.zeros(7)]
        k(4, n, [x, i], out)
        outs.append(out)
    for o in outs[1:]:
        assert all(np.array_equal(a, b) for a, b in zip(o, outs[0])), (o, outs[0])
    np.testing.assert_allclose(outs[0][0], np.sum(np.sin(x) * x[:, :1], 0), rtol=1e-12)


def test_threads_bound_only_where_they_fill_the_mask(tmp_path):
    """Under a restricted mask (taskset; a container's or a scheduler's, shared by several processes) the pool's
    threads are bound only where they are as many as its CPUs: with fewer, every such process bound its threads to the
    same first CPUs (3 processes of 1 thread on 3 CPUs: 2.7x slower)."""
    import subprocess
    k = 4 if os.sysconf("SC_NPROCESSORS_ONLN") > 4 else 2  # (a mask of fewer CPUs than the machine's: restricted; on a
    mask = sorted(os.sched_getaffinity(0))[:k]  # runner of 4 CPUs, the first 4 were all of them, and nothing was bound)
    if len(mask) < k or k >= os.sysconf("SC_NPROCESSORS_ONLN") or "OMP_PROC_BIND" in os.environ or "GOMP_CPU_AFFINITY" in os.environ:
        pytest.skip("needs more than 2 CPUs and no binding by the environment")
    script = tmp_path / "pin.py"
    script.write_text("""import os, jax, jax.numpy as jnp, numpy as np
from kernelax import jit
jax.config.update("jax_enable_x64", True)
jit(lambda x: jnp.sin(x) * 2.0 + x, build_first=True)(np.ones((5000, 3)))
pool = [t for t in os.listdir("/proc/self/task") if open(f"/proc/self/task/{t}/comm").read().strip() == "kernelax_pool"]
print(sorted(len(os.sched_getaffinity(int(t))) for t in pool))
""")
    env = dict(os.environ, KERNELAX_CACHE_DIR=str(tmp_path / "cache"), JAX_PLATFORMS="cpu")
    for nt, cpus in ((k // 2, k), (k, 1)):  # (the pool's team: nt threads, each on its own CPU only where they fill the mask)
        r = subprocess.run(["taskset", "-c", ",".join(map(str, mask)), sys.executable, str(script)], env=dict(env, OMP_NUM_THREADS=str(nt)),
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and r.stdout.strip() == str([cpus] * nt), (nt, r.stdout, r.stderr[-2000:])


def test_import_without_a_cxx_compiler(tmp_path):
    """import kernelax builds no runtime (it did at import: a RuntimeError without g++, 3.4 s for the first import):
    it is built when a kernel is first lowered; without it kernelax.jit stays in XLA, warned."""
    import subprocess
    code = """import warnings, jax, jax.numpy as jnp, numpy as np
import kernelax
jax.config.update("jax_enable_x64", True)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    g = kernelax.jit(lambda x: jnp.sin(x) * 2.0 + x, build_first=True)
    print(np.allclose(g(np.ones((5000, 3))), np.sin(1.0) * 2.0 + 1.0), g.report()[0]["status"], any("C compiler" in str(x.message) for x in w))
"""
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, KERNELAX_CXX=str(tmp_path / "no-g++"),
                                                                KERNELAX_CACHE_DIR=str(tmp_path / "cache"), JAX_PLATFORMS="cpu"),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.split() in (["True", "xla", "True"], ["True", "failed", "True"]), r.stdout + r.stderr[-2000:]


def test_import_on_other_systems_and_in_odd_environments(tmp_path):
    """import kernelax: on a system other than Linux a clear ImportError (it failed on fcntl or os.sched_getaffinity);
    a cache directory it cannot create named with KERNELAX_CACHE_DIR; a PATH directory it cannot read skipped in the
    search for gcc-NN; the cache under XDG_CACHE_HOME where that is set."""
    import subprocess
    base = {k: v for k, v in os.environ.items() if k not in ("KERNELAX_CACHE_DIR", "XDG_CACHE_HOME")}
    run = lambda code, **env: subprocess.run([sys.executable, "-c", code], env=dict(base, JAX_PLATFORMS="cpu", **env),
                                             capture_output=True, text=True, timeout=300)
    r = run("import sys\nsys.platform = 'darwin'\nimport kernelax")
    assert r.returncode != 0 and "ImportError: kernelax needs Linux" in r.stderr, r.stderr[-2000:]
    r = run("import kernelax", KERNELAX_CACHE_DIR="/proc/kernelax-cache")
    assert r.returncode != 0 and "KERNELAX_CACHE_DIR" in r.stderr.splitlines()[-1], r.stderr[-2000:]
    locked = tmp_path / "locked"
    locked.mkdir(mode=0)
    if os.access(locked, os.R_OK):  # (root reads it anyway)
        pytest.skip("runs as root")
    path = f"{locked}{os.pathsep}{os.environ['PATH']}"
    r = run("import kernelax", KERNELAX_CACHE_DIR=str(tmp_path / "cache"), PATH=path)
    assert r.returncode == 0, r.stderr[-2000:]
    r = run("from kernelax import build\nprint(build.CACHE)", XDG_CACHE_HOME=str(tmp_path / "xdg"))
    assert r.returncode == 0 and r.stdout.split() == [str(tmp_path / "xdg" / "kernelax")], r.stdout + r.stderr[-2000:]


def test_cache_keys_of_compiler_wrappers(tmp_path, monkeypatch):
    """Compiler wrappers (KERNELAX_CC with flags) key the cache by their text: another wrapper's flags hit the old
    entry; a gcc killed at its time limit leaves no log in the cache."""
    import glob
    from kernelax import build
    wraps = []
    for flags in ("-O1", "-O2"):
        w = tmp_path / f"cc{flags}"
        w.write_text(f'#!/bin/sh\nexec {build.CC} {flags} "$@"\n')
        w.chmod(0o755)
        wraps.append(str(w))
    assert build._cc_version(wraps[0]) != build._cc_version(wraps[1])
    monkeypatch.setattr(build, "CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(build, "GCC_TIMEOUT", 0.001)
    src = "".join(f"double f{j}(double x) {{ return x * {j}.5 + {j}; }}\n" for j in range(3000))
    with pytest.raises(build.GccTimeout):
        build.compile_c(src)
    assert not glob.glob(str(tmp_path / "cache" / "**" / "*.log"), recursive=True)


def test_build_first_by_default(tmp_path, monkeypatch):
    """A new program's first call builds its kernels and runs with them (build_first, the default), then trials against
    XLA's program as after a build in the background (its choice was kept unmeasured before); a build
    longer than KERNELAX_BUILD_FIRST_WAIT leaves the first call in XLA, the kernels follow in the thread (numpyro's log
    density: 53 s of build for 3 s of XLA)."""
    import time
    from kernelax import cpu
    monkeypatch.setattr(cpu, "_INDEX", str(tmp_path))
    monkeypatch.setattr(auto, "TRIAL_TIME", 0.0)
    c = np.random.default_rng(75).uniform(0.5, 1.5, (2000, 3))
    loop = lambda k: lambda x: jnp.sum(jax.vmap(lambda r: jax.lax.while_loop(lambda y: y < 4.0, lambda y: y * k + r[0], r[1]))(x))
    build = auto._Entry._build_now  # (a build of at least a second: whether the first call waited shows)
    monkeypatch.setattr(auto._Entry, "_build_now", lambda e, *a: (time.sleep(1.0), build(e, *a))[1])
    g = jit(loop(1.15))
    t0 = time.perf_counter()
    np.testing.assert_allclose(g(c), loop(1.15)(c), rtol=1e-12)
    assert time.perf_counter() - t0 > 1.0 and g.report()[0]["status"] == "trial"  # (it waited: ran with its kernels)
    g.wait(variants=True)
    for _ in range(12):
        g(c)
    assert g.report()[0]["why"] in ("measured", "long calls"), g.report()[0]
    monkeypatch.setattr(auto, "BUILD_FIRST_WAIT", 0.2)
    h = jit(loop(1.25))
    t0 = time.perf_counter()
    np.testing.assert_allclose(h(c), loop(1.25)(c), rtol=1e-12)
    assert time.perf_counter() - t0 < 1.0 and h.report()[0]["status"] == "building"  # (in XLA: the build goes on)
    h.wait()
    assert h.report()[0]["status"] == "trial"


def test_rollout_with_keys_split_per_step():
    """A rollout as users write it (a scan over the steps, in it vmap over the N environments with their keys split
    from the step's key, jax.random.split(k, N)) is one region with the scan inside, a loop per sample in the kernel:
    key s of split(k, (N,)) is fold_in(k, s) (threefry_partitionable). Before, the split of a key the same for all
    samples into rows had no rule: the scan stayed in XLA, a kernel per step (gymnax CartPole 34.7 ms against 1.6 ms)."""
    N, T = 300, 200

    def step1(k, s):
        for _ in range(10):
            s = jnp.sin(s) * 0.9
        s = s + jax.random.uniform(k, (2,))
        return s, s.sum()

    def rollout(kd):
        k0, k1 = jax.random.split(jax.random.wrap_key_data(kd))
        s = jax.vmap(lambda k: jax.random.uniform(k, (2,)))(jax.random.split(k0, N))

        def step(c, k):
            s, ret = c
            s, r = jax.vmap(step1)(jax.random.split(k, N), s)
            return (s, ret + r), None
        return jax.lax.scan(step, (s, jnp.zeros(N)), jax.random.split(k1, T))[0]
    kd = np.asarray(jax.random.key_data(jax.random.key(4)))
    _, steps, regions = auto._axis(jaxprs._flatten(rollout, (kd,)).jaxpr)
    assert any(e.primitive.name == "scan" for r in regions for e in r.orig), steps
    g = jit(rollout)
    g(kd)
    g.wait()
    for a, b in zip(jax.tree.leaves(g(kd)), jax.tree.leaves(jax.jit(rollout)(kd))):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    assert not g.report()[0].get("errors"), g.report()  # (the rolled scan's body traced with the sample index symbolic:
    # frontend traces under ensure_compile_time_eval, which evaluated the index's equation, an equation without operands)


def test_rollout_reset_and_steps_in_one_kernel():
    """In a rollout as users write it, the reset from keys split per sample (rows of split(k, N): fold_in(k, s), each
    sample's own) and the scan over the steps are one region, one kernel: the split's rows are not rows equal for all
    samples (as broadcasts of uniform values are), so what is computed from them is not pulled into the regions that
    read it. Before, the scan counted as such a value: with the sample axis alone no region at all, the reset ran as a
    kernel of its own on the pair of axes (N, 2), the steps as a second one. The keys of the steps, split(k, T) of a key
    the same for all, are split in the kernel (lazily, per step), not read from an input."""
    N, T = 300, 50

    def rollout(kd):
        k0, k1 = jax.random.split(jax.random.wrap_key_data(kd))
        s = jax.vmap(lambda k: jax.random.uniform(k, (2,)))(jax.random.split(k0, N))

        def step(c, k):
            s, ret = c
            s = jax.vmap(lambda k, s: jnp.sin(s) * 0.9 + jax.random.uniform(k, (2,)))(jax.random.split(k, N), s)
            return (s, ret + s.sum(1)), None
        return jax.lax.scan(step, (s, jnp.zeros(N)), jax.random.split(k1, T))[0][1]
    kd = np.asarray(jax.random.key_data(jax.random.key(6)))
    _, steps, regions = auto._axis(jaxprs._flatten(rollout, (kd,)).jaxpr)
    names = [{e.primitive.name for e in r.orig} for r in regions]
    assert len(regions) == 1 and {"scan", "random_bits"} <= names[0], (names, steps)
    assert not any(v.aval.shape == (T,) for v, _ in regions[0].inputs), regions[0].inputs  # (the steps' keys split in the
    # kernel, lazily at each step: read as an input, the GPU kernel copied them to two arrays of T per thread, 0.2 ms)
    g = jit(rollout)
    g(kd)
    g.wait()
    np.testing.assert_allclose(g(kd), jax.jit(rollout)(kd), rtol=1e-6, atol=1e-6)


def test_reshape_with_dimensions_per_sample():
    """A reshape with dimensions (a transpose, then a reshape: pgx's connect_four observation, jnp.stack(...).reshape in
    a vmapped step) whose sample axis stays first is row-wise: per sample the same without that axis. Before, the
    scan of the rollout around it stayed in XLA (connect_four: 199 ms against kernelax.kernel's 6.2 ms)."""
    def step(c, _):
        b = jnp.stack([c > 0.5, c < 0.2]).reshape(2, 6, 7, 1)
        o = jax.lax.reshape(b, (6, 7, 2), dimensions=(1, 2, 0, 3))
        return jnp.sin(c) + o.sum(-1).astype(c.dtype) * 0.1, None

    def f(x):
        return jax.lax.scan(step, x, None, length=50)[0]
    x = np.random.default_rng(6).uniform(0.0, 1.0, (300, 6, 7))
    _, steps, regions = auto._axis(jaxprs._flatten(jax.vmap(f), (x,)).jaxpr)
    assert any(e.primitive.name == "scan" for r in regions for e in r.orig), steps
    g = jit(jax.vmap(f))
    np.testing.assert_allclose(g(x), jax.jit(jax.vmap(f))(x), rtol=1e-12)


def test_rollout_from_a_constant_state():
    """A rollout whose start is the same for all samples (a deterministic reset: gymnax' MinAtar Asterix), folded into
    a constant with rows, is one region with the scan inside: a loop's rows may be a constant's (before, an equation
    whose rows were constants alone stayed in XLA, the scan with a kernel per step: Asterix 139 ms against 27)."""
    N, T = 300, 200

    def step1(k, s):
        for _ in range(10):
            s = jnp.sin(s) * 0.9
        return s + jax.random.uniform(k, (2,)), s.sum()

    def rollout(kd):
        k0, k1 = jax.random.split(jax.random.wrap_key_data(kd))
        s = jax.vmap(lambda k: jnp.array([0.5, 1.5]))(jax.random.split(k0, N))  # (the same for all)

        def step(c, k):
            s, ret = c
            s, r = jax.vmap(step1)(jax.random.split(k, N), s)
            return (s, ret + r), None
        return jax.lax.scan(step, (s, jnp.zeros(N)), jax.random.split(k1, T))[0]
    kd = np.asarray(jax.random.key_data(jax.random.key(5)))
    j = auto._fold(auto._decouple(auto._inline(jaxprs._flatten(rollout, (kd,)).jaxpr)))
    _, steps, regions = auto._axis(j)
    assert any(e.primitive.name == "scan" for r in regions for e in r.orig), steps
    g = jit(rollout)
    for a, b in zip(jax.tree.leaves(g(kd)), jax.tree.leaves(jax.jit(rollout)(kd))):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_long_calls_checked_against_xla(monkeypatch):
    """A long first call with kernels (TRIAL_MAX_CALL) that took more than LONG_CHECK of XLA's first call (compile and
    run; known where it ran: build_first=False) is checked against one call of XLA's program, not kept untried (pgx'
    othello: kernels 2.8 s, XLA 2.3 s)."""
    monkeypatch.setattr(auto, "TRIAL_MAX_CALL", 0.0)
    monkeypatch.setattr(auto, "LONG_CHECK", 0.0)
    monkeypatch.setattr(auto, "TRIAL_TIME", 0.0)  # (long calls exceed it by themselves)
    x = np.random.default_rng(7).uniform(0.5, 1.5, (2000, 8))
    f = lambda x: jnp.sin(x) * jnp.exp(-x) + jnp.log1p(x * x)
    g = jit(f, build_first=False)
    g(x)
    g.wait(variants=True)
    for _ in range(6):
        out = g(x)
    np.testing.assert_allclose(out, jax.jit(f)(x), rtol=1e-12)
    assert g.report()[0].get("why") == "measured", g.report()


def test_reset_of_a_rollout_in_its_kernel():
    """The reset of a rollout as users write it (vmap of the reset over keys split from one) is recomputed per sample in
    the scan's kernel (pulled) on the one sample axis N: XLA's steps that only regions read, recomputing them, store
    nothing (_axis counted them, chose the pair of axes (N, k) for the reset first and left it to XLA: CartPole 0.6 of
    2.2 ms)."""
    N, T = 300, 200

    def step1(k, s):
        for _ in range(10):
            s = jnp.sin(s) * 0.9
        return s + jax.random.uniform(k, (2,)), s.sum()

    def rollout(kd):
        k0, k1 = jax.random.split(jax.random.wrap_key_data(kd))
        s = jax.vmap(lambda k: jax.random.uniform(k, (2,)))(jax.random.split(k0, N))

        def step(c, k):
            s, ret = c
            s, r = jax.vmap(step1)(jax.random.split(k, N), s)
            return (s, ret + r), None
        return jax.lax.scan(step, (s, jnp.zeros(N)), jax.random.split(k1, T))[0]
    kd = np.asarray(jax.random.key_data(jax.random.key(6)))
    j = auto._fold(auto._decouple(auto._inline(jaxprs._flatten(rollout, (kd,)).jaxpr)))
    axes, steps, regions = auto._axis(j)
    assert axes == [N], axes
    g = jit(rollout)
    for a, b in zip(jax.tree.leaves(g(kd)), jax.tree.leaves(jax.jit(rollout)(kd))):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_work_of_a_scan_rolled_within_roll_xs():
    """A long scan over xs (per step its element; an RL rollout's keys) counts its body once in the work of its region
    where the frontend rolls it (unrolled for its xs alone it would exceed frontend.ROLL_XS), not once per step: else
    the region exceeded MAX_WORK and stayed in XLA."""
    from kernelax import frontend
    if not frontend.ROLL_XS:
        pytest.skip("no ROLL_XS")

    def f(g):
        def step(v, x):
            for _ in range(100):  # (800 x 400 operations unrolled: beyond MAX_WORK)
                v = v + 0.01 * jnp.tanh(v * x)
            return v, None
        return jax.lax.scan(step, g, jnp.linspace(0.5, 1.5, 800) * g)[0]
    g = np.random.default_rng(10).uniform(0.5, 1.5, 512)
    j = auto._fold(auto._decouple(auto._inline(jaxprs._flatten(jax.vmap(f), (g,)).jaxpr)))
    _, steps, regions = auto._axis(j)
    assert any(e.primitive.name == "scan" for r in regions for e in r.orig) and all(
        auto._work(r)[0] <= auto.MAX_WORK for r in regions), steps
    np.testing.assert_allclose(jit(jax.vmap(f))(g), jax.jit(jax.vmap(f))(g), rtol=1e-12)



def test_rollout_with_a_uniform_loop_in_its_step():
    """In a rollout as users write it, a value computed the same for all samples in the step (from the step's key or a
    constant: pgx 2048's observation, a fori_loop writing into zeros broadcast over the samples) and then mixed with
    the samples' rows needs its row version: a loop whose outputs are its counter (uniform) and the rows. Before, the
    row version was kept only for equations whose outputs all have rows: the scan of the rollout stayed in XLA, a
    kernel per step (2048: 119 ms against kernelax.kernel's 12 ms)."""
    N, T = 300, 50

    def step1(k, s):
        b = jax.random.randint(k, (4,), 0, 3)  # (per sample: from its key; in the step, from the step's key for all)
        o = jax.lax.fori_loop(0, 4, lambda i, o: o.at[i, b[i]].set(i + 1.0), jnp.zeros((4, 3)))
        return jnp.where(s > 0.5, o, s * 0.9), s.sum()

    def rollout(kd):
        k0, k1 = jax.random.split(jax.random.wrap_key_data(kd))
        s = jax.vmap(lambda k: jax.random.uniform(k, (4, 3)))(jax.random.split(k0, N))

        def step(c, k):
            s, ret = c
            s, r = jax.vmap(step1)(jax.random.split(k, N), s)
            return (s, ret + r), None
        return jax.lax.scan(step, (s, jnp.zeros(N)), jax.random.split(k1, T))[0]
    kd = np.asarray(jax.random.key_data(jax.random.key(5)))
    _, steps, regions = auto._axis(jaxprs._flatten(rollout, (kd,)).jaxpr)
    assert any(e.primitive.name == "scan" for r in regions for e in r.orig), steps
    g = jit(rollout)
    g(kd)
    g.wait()
    for a, b in zip(jax.tree.leaves(g(kd)), jax.tree.leaves(jax.jit(rollout)(kd))):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_top_k_per_sample():
    """top_k along a sample's own axis is row-wise (xland-minigrid's reset draws positions by top_k): before, it split
    the reset from the rollout's scan into two kernels, the state through memory between them (88.7 ms against
    kernelax.kernel's 70.7)."""
    def f(x):
        v, i = jax.lax.top_k(x, 2)
        def step(c, _):
            return jnp.sin(c) * 0.9 + 0.1, None
        return jax.lax.scan(step, v * i.astype(v.dtype), None, length=40)[0]
    x = np.random.default_rng(12).uniform(0.0, 1.0, (300, 64))
    j = auto._fold(auto._decouple(auto._inline(jaxprs._flatten(jax.vmap(f), (x,)).jaxpr)))
    _, steps, regions = auto._axis(j)
    assert len(regions) == 1 and {"top_k", "scan"} <= {e.primitive.name for e in regions[0].orig}, steps
    np.testing.assert_allclose(jit(jax.vmap(f))(x), jax.jit(jax.vmap(f))(x), rtol=1e-12)


def test_loop_with_some_uniform_state_per_sample():
    """A loop carrying a few hundred uniform entries next to per-sample state (Brax' MJX frame loop: 348, its contact
    data) is per sample in a kernel (auto.UNIFORM_MAX 512; a solver's state of 1000 stays in XLA: the test above):
    before, the rollout around it stayed in XLA (Brax ant/mjx 1.0x against XLA, kernelax.kernel 354x)."""
    def f(x):
        def frame(c, _):
            v, u = c
            return (jnp.sin(v) * 0.9 + u[:3].sum() * 1e-3, u * 0.5 + 1.0), None
        def step(c, _):
            return jax.lax.scan(frame, c, None, length=5)[0], None
        return jax.lax.scan(step, (x, jnp.zeros(300)), None, length=40)[0][0]
    x = np.random.default_rng(13).uniform(0.0, 1.0, (300, 3))
    j = auto._fold(auto._decouple(auto._inline(jaxprs._flatten(jax.vmap(f), (x,)).jaxpr)))
    _, steps, regions = auto._axis(j)
    assert any(e.primitive.name == "scan" for r in regions for e in r.orig), steps
    np.testing.assert_allclose(jit(jax.vmap(f))(x), jax.jit(jax.vmap(f))(x), rtol=1e-12)


def test_region_of_a_long_scan_with_outputs_per_step():
    """A long scan with outputs per step (a neuron's voltage trace, unrolled by the frontend) whose region does more
    than MAX_WORK operations per sample (after _inner's loops, where they apply) still gets a kernel up to
    MAX_WORK_PLAIN (Jaxley's gradient per neuron: 371 708; kernelax.jit then as fast as kernelax.kernel, 13.8x against
    XLA instead of 1.0x); regions with inner loops keep MAX_WORK (test_few_elements_with_much_work)."""
    def f(g):
        def step(v, _):
            for _ in range(70):  # (800 x 280 operations, unrolled)
                v = v + 0.01 * jnp.tanh(v * g)
            return v, v
        return jax.lax.scan(step, g, None, length=800)
    g = np.random.default_rng(11).uniform(0.5, 1.5, 512)
    j = auto._fold(auto._decouple(auto._inline(jaxprs._flatten(jax.vmap(f), (g,)).jaxpr)))
    _, steps, regions = auto._axis(j)
    assert any(e.primitive.name == "scan" for r in regions for e in r.orig), steps
    assert max(auto._work(r)[0] for r in regions) > auto.MAX_WORK
    out = jit(jax.vmap(f))(g)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(jax.jit(jax.vmap(f))(g))):
        np.testing.assert_allclose(a, b, rtol=1e-12)
