# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kernelax.vmap import cvmap
from kernelax.build import simd_width

jax.config.update("jax_enable_x64", True)
u32 = jnp.uint32


def bits(x):  # integer semantics: wraparound, shifts, bitcasts, unsigned 64-bit values beyond 2^53
    a, b, c = x[0].astype(jnp.uint64).astype(u32), x[1].astype(jnp.uint64).astype(u32), x[2].astype(jnp.int32)
    rot = lambda v, r: (v << r) | (v >> (32 - r))
    s = a + b * u32(2654435761)
    t = rot(s ^ b, 13) + (a >> 7) - (b << 5)
    sh = (a << (c % 40).astype(u32)) | (b >> (c % 40).astype(u32))  # shifts >= 32 give 0
    lg = jax.lax.shift_right_logical(c - 1000, jnp.int32(3))
    ar = jax.lax.shift_right_arithmetic(c - 1000, jnp.int32(3))
    big = (a.astype(jnp.uint64) << 32) | b.astype(jnp.uint64)
    f64 = jax.lax.bitcast_convert_type((big >> 12) | jnp.uint64(0x3FF0000000000000), jnp.float64) - 1.0
    f32 = jax.lax.bitcast_convert_type((s >> 9) | u32(0x3F800000), jnp.float32) - 1.0
    back = jax.lax.bitcast_convert_type(x[0] * 0.5, jnp.int64) & 0xFFFF
    return s, t, sh, lg, ar, f64, f32, back, big > jnp.uint64(2 ** 63), (~a) & u32(0xFF00FF), a // 7 + b % 9


@pytest.mark.parametrize("W", [4, 8] if simd_width() == 8 else [simd_width()])
def test_integer_bits(W):
    """Integer nodes: bitwise equal to JAX."""
    rng = np.random.default_rng(0)
    x = np.stack([rng.integers(0, 2 ** 32, 999), rng.integers(0, 2 ** 32, 999), rng.integers(-50, 2000, 999)], 1)
    x = x.astype(np.float64)
    for o, r in zip(jax.tree.leaves(cvmap(bits, [x[0]], (0,), W=W)(x)), jax.tree.leaves(jax.vmap(bits)(x))):
        np.testing.assert_array_equal(o, np.asarray(r).astype(np.float64))


@pytest.mark.parametrize("W", [4, 8] if simd_width() == 8 else [simd_width()])
def test_uint64_divmod(W):
    """Division and remainder of uint64 values beyond 2^53: exact."""
    def f(x):
        big = (x[0].astype(jnp.uint64) << 32) | x[1].astype(jnp.uint64)
        d = x[2].astype(jnp.uint64) + jnp.uint64(1)
        return jax.lax.rem(big, d), jax.lax.div(big, d) & jnp.uint64(0xFFFFF), jax.lax.rem(big, jnp.uint64(10))
    rng = np.random.default_rng(1)
    x = np.stack([rng.integers(0, 2 ** 32, 999), rng.integers(0, 2 ** 32, 999), rng.integers(0, 2 ** 31, 999)], 1)
    x = x.astype(np.float64)
    for o, r in zip(cvmap(f, [x[0]], (0,), W=W)(x), jax.vmap(f)(x)):
        np.testing.assert_array_equal(o, np.asarray(r).astype(np.float64))


def test_integer_semantics():
    """As XLA: narrow integers wrap on overflow (products exact first), int64 division and remainder exact,
    x / 0 = -1 and x % 0 = x, float to integer saturated (NaN: 0), 64-bit constants beyond 2^53 exact, uint64
    loop carries by their bits."""
    rng = np.random.default_rng(1)
    i = rng.integers(-2 ** 31, 2 ** 31, 61).astype(np.int32)
    j = rng.integers(-2 ** 62, 2 ** 62, 61)
    f = np.concatenate([rng.uniform(-1e10, 1e10, 55), [np.nan, np.inf, -np.inf, 2.5e9, -2.5e9, 300.0]])
    u = rng.integers(0, 2 ** 63, 61, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    T = jnp.array([2 ** 60 + 5, 3, 2 ** 62 + 7, -(2 ** 61) - 1] * 4, jnp.int64)
    lcg = lambda s: jax.lax.while_loop(lambda c: c[0] < 9, lambda c: (c[0] + 1, c[1] * jnp.uint64(6364136223846793005)
                                                                     + jnp.uint64(1442695040888963407)), (0, s))[1]
    cases = [
        (lambda i: (i + jnp.int32(2 ** 30), i * jnp.int32(3), i * i, i.astype(jnp.uint32) * jnp.uint32(2654435761),
                    -i, i.astype(jnp.int8) + jnp.int8(100), i // jnp.int32(7), i % jnp.int32(7), i // (i % 3), i % (i % 3)), i),
        (lambda j: (j // 3, j % 3, j // (j % 3), j % (j % 3), jnp.where(j > 0, jnp.int64(2 ** 60 + 5), 7), T[j % 16]), j),
        (lambda f: (f.astype(jnp.int32), f.astype(jnp.uint8), f.astype(jnp.int64), f.astype(jnp.int16)), f),
        (lcg, u),
    ]
    for g, x in cases:
        for o, r in zip(jax.tree.leaves(cvmap(g, [x[0]], (0,))(x)), jax.tree.leaves(jax.vmap(g)(x))):
            np.testing.assert_array_equal(o, np.asarray(r).astype(o.dtype))


def test_no_rewrapping():
    """An integer value that an operation leaves as it is (x + 0, from a broadcast of zeros, as in an FE code's index
    expansion with one DOF per node) is not wrapped again, and a wrapped value read back as an integer is that
    integer node (no float round trip): no code for either, results as JAX."""
    from kernelax import dag
    from kernelax.frontend import trace
    rng = np.random.default_rng(3)
    b, w = rng.integers(-2 ** 31, 2 ** 31, (40, 6)).astype(np.int32), rng.integers(-9, 9, (40, 6)).astype(np.int32)
    z = jnp.zeros((1, 6), jnp.int32)
    f = lambda b, w: (b + z * w + z, (b + w) * jnp.int32(3) + jnp.int32(5))
    _, outs, _ = trace(f, b[0], w[0], uniform=[False, False])
    ops = [n.op for n in dag.schedule([x for o in outs for x in np.asarray(o, dtype=object).ravel() if isinstance(x, dag.Node)])]
    assert ops.count("iwrap") == 3 * 6 and ops.count("ftoi") == 6  # (per entry: the two sums and the product wrapped,
    # the first output not; an integer read from a float for b + w only: the product reads the wrapped sum)
    for o, r in zip(cvmap(f, [b[0], w[0]], (0, 0))(b, w), jax.vmap(f)(b, w)):
        np.testing.assert_array_equal(o, r)


def test_gather_fill():
    """Gathers in fill mode: out-of-bounds positions get the fill value (also from inputs in memory, also with one or
    two per-sample indices), never an entry read at it."""
    rng = np.random.default_rng(2)
    x, i, T = rng.normal(size=(37, 5)), (np.arange(37) % 13 - 6).astype(np.int32), rng.normal(size=100)
    for g, args, axes in [
        (lambda x: (x.at[7].get(mode="fill", fill_value=3.0), x.at[jnp.array([1, 7, -9])].get(mode="fill"),
                    x.at[7].get(mode="fill", fill_value=-1.0)), (x,), (0,)),
        (lambda x, i: x.at[i].get(mode="fill", fill_value=-7.0), (x, i), (0, 0)),
        (lambda x, i: x[:4].reshape(2, 2).at[i % 3, (i + 1) % 3].get(mode="fill", fill_value=0.5), (x, i), (0, 0)),
        (lambda x, T: x + jnp.asarray(T).at[jnp.array([1, 200])].get(mode="fill", fill_value=-1e6)[0]
         + jnp.asarray(T).at[jnp.array([1, 200])].get(mode="fill")[1], (x[:, 0], T), (0, None)),
    ]:
        ex = [a[0] if ax == 0 else a for a, ax in zip(args, axes)]
        for o, r in zip(jax.tree.leaves(cvmap(g, ex, axes)(*args)), jax.tree.leaves(jax.vmap(g, axes)(*args))):
            np.testing.assert_array_equal(o, r)


def _same(g, *args, axes=None):  # cvmap of g against jax.vmap, bitwise
    axes = axes or (0,) * len(args)
    ex = [a[0] if ax == 0 else a for a, ax in zip(args, axes)]
    for o, r in zip(jax.tree.leaves(cvmap(g, ex, axes)(*args)), jax.tree.leaves(jax.vmap(g, axes)(*args))):
        o = np.asarray(o)
        r = np.asarray(r).astype(o.dtype)
        if o.dtype in (jnp.bfloat16, jnp.float16):  # (numpy compares NaN as equal only in its own float types)
            o, r = o.astype(np.float32), r.astype(np.float32)
        np.testing.assert_array_equal(o, r)


def test_division_of_exact_multiples():
    """Quotient and remainder of narrow integers (computed in doubles): exact also at exact multiples, where
    x * (1/y) falls short of the integer (49 * (1/49) < 1); runtime and constant divisors, signs, INT_MIN / -1."""
    rng = np.random.default_rng(4)
    y = rng.integers(1, 3000, 400).astype(np.int32) * rng.choice([-1, 1], 400).astype(np.int32)
    k = rng.integers(-40000, 40000, 400).astype(np.int32)
    x = np.concatenate([k * y, k * y + 1, k * y - 1, [np.iinfo(np.int32).min]]).astype(np.int32)
    y = np.concatenate([y, y, y, [-1]]).astype(np.int32)
    _same(lambda x, y: (x // y, x % y, jax.lax.div(x, y), jax.lax.rem(x, y), jax.lax.div(x, y) < 0, jax.lax.div(x, y) // 2), x, y)
    _same(lambda x: (x // jnp.int32(49), x % jnp.int32(49), jax.lax.div(x, jnp.int32(-103)), x % jnp.int32(-103),
                     x // jnp.int32(64), x % jnp.int32(1000)), x)
    u = (np.abs(x.astype(np.int64)) * 3 % 2 ** 32).astype(np.uint32)
    _same(lambda u, y: (u // jnp.abs(y).astype(jnp.uint32), u % jnp.uint32(49), (u // jnp.uint32(98)).astype(jnp.int32)), u, y)
    s = (x % 30000).astype(np.int16)
    _same(lambda s: (s // jnp.int16(49), s % jnp.int16(49)), s)


def test_bitwise_reductions():
    """reduce_and / reduce_or / reduce_xor over integers: bitwise (not as masks)."""
    rng = np.random.default_rng(5)
    a = rng.integers(0, 2 ** 31, (50, 4)).astype(np.int32)
    _same(lambda a: (jax.lax.reduce_and(a, (0,)), jax.lax.reduce_or(a, (0,)), jax.lax.reduce_xor(a, (0,)),
                     jnp.bitwise_or.reduce(a.astype(jnp.uint32)), jax.lax.reduce_and(a.astype(jnp.int64) << 20, (0,)),
                     jax.lax.reduce_or(a > 2 ** 30, (0,)), jax.lax.reduce_and(a > 2 ** 20, (0,))), a)


def test_int64_products_and_abs():
    """int64 arithmetic beyond 2^53 (products of values held as doubles, abs): exact, wrapping as XLA."""
    rng = np.random.default_rng(6)
    v = np.concatenate([rng.integers(-2 ** 31, 2 ** 31, 60), [2 ** 31 - 1, -2 ** 31, 123456789]]).astype(np.int32)
    w = np.concatenate([rng.integers(-2 ** 62, 2 ** 62, 60), [-(2 ** 62) - 1, 2 ** 63 - 1, -(2 ** 63)]]).astype(np.int64)
    _same(lambda v, w: (v.astype(jnp.int64) ** 2 * 3, v.astype(jnp.int64) * v.astype(jnp.int64) * 5, jnp.abs(w),
                        w * 3 + v, jnp.prod(jnp.stack([v.astype(jnp.int64), v.astype(jnp.int64) + 7])),
                        jnp.cumsum(jnp.stack([w, w]))), v, w)


def test_uint64_ordering():
    """uint64 values from 2^63 on order above smaller ones everywhere: clamp, max/min, argmax/argmin, sort, cummax,
    reduce_max, top_k."""
    rng = np.random.default_rng(7)
    a = rng.integers(0, 2 ** 64, (30, 5), dtype=np.uint64)
    a[:, 1] = np.uint64(3)
    a[:, 2] = np.uint64(2 ** 63 + 5)
    b = jnp.uint64(2 ** 63 + 100)
    _same(lambda a: (jnp.clip(a[0], jnp.uint64(10), b), jnp.max(a), jnp.min(a), jnp.argmax(a), jnp.argmin(a),
                     jnp.sort(a), jax.lax.cummax(a), jax.lax.top_k(a, 2)[0], jnp.maximum(a[1], a[2])), a)


def test_float_to_narrow_int_saturation():
    """float32, float16 and bfloat16 to int32 / uint32: saturated at the integer limits (not at 2^31 or 2^32)."""
    f = np.array([3e9, -3e9, 4.5e9, 2.2e9, -1.0, np.nan, np.inf, -np.inf, 1.5, 2147483520.0], np.float32)
    _same(lambda f: (f.astype(jnp.int32), f.astype(jnp.int32) - 1, f.astype(jnp.uint32), f.astype(jnp.uint32) - 1,
                     f.astype(jnp.float16).astype(jnp.int32), f.astype(jnp.bfloat16).astype(jnp.uint32)), f)


def test_integer_scatter_and_window_sums_wrap():
    """Integer scatter-add / scatter-mul and window sums wrap as XLA's integer arithmetic: also where they are used
    further (a 4-byte kernel's int32 stores wrapped the outputs themselves) and in an 8-byte kernel."""
    v = np.array([[2 ** 31 - 10, 100, 7], [2 ** 30, 2 ** 30, 2 ** 30]], np.int32)
    i = jnp.array([0, 0, 1])
    f = lambda v: (jnp.zeros(2, jnp.int32).at[i].add(v), jnp.full(2, 3, jnp.int32).at[i].mul(v),
                   jax.lax.reduce_window(v, 0, jax.lax.add, (2,), (1,), "VALID"))
    _same(lambda v: f(v) + tuple(o < 0 for o in f(v)) + (v.astype(jnp.int64).sum(),), v)


def test_bfloat16_gather_fill():
    """A gather in fill mode of bfloat16 values fills NaN (jax's default for floats), as for float16, at constant and
    per-sample indices (jax writes the fill value into the gather: frontend._fill's defaults are not reached)."""
    x = np.arange(12.0).reshape(4, 3).astype(jnp.bfloat16)
    i = np.array([5, -1, 2, 1], np.int32)
    _same(lambda x, i: (x.at[5].get(mode="fill"), x.astype(jnp.float16).at[jnp.array([0, 9])].get(mode="fill"),
                        x.at[i].get(mode="fill"), x.astype(jnp.float16).at[i].get(mode="fill")), x, i)


def test_int64_to_float32_rounded_once():
    """int64 and uint64 to float32: rounded once to nearest even (not through a double: twice)."""
    v = np.array([2 ** 60 + 2 ** 36 + 1, 2 ** 60 + 2 ** 36, 2 ** 53 + 1, -(2 ** 60) - 2 ** 36 - 1, 2 ** 24 + 1, 7, -(2 ** 63),
                  2 ** 63 - 1, 2 ** 40 + 2 ** 16 + 2 ** 15 + 1, 0], np.int64)
    u = v.view(np.uint64)
    _same(lambda v, u: (v.astype(jnp.float32), u.astype(jnp.float32), v.astype(jnp.float64)), v, u)


def test_table_lookups_far_out_of_range():
    """A constant table read at indices far out of range (beyond 2^51, where rounding by the magic constant wraps):
    clamped to the first or last entry, as XLA's gather."""
    T = np.cos(np.linspace(0, 3, 40))
    i = np.array([-12_000_000_000_000_000, -7_000_000_000_000_000, 5_000_000_000_000_000, 3, -3, 39, 2 ** 62], np.int64)
    _same(lambda i: (jnp.asarray(T)[i], jnp.asarray(T)[jnp.clip(i, -40, 39)], jnp.asarray(T).at[i].get(mode="clip")), i)


def test_uint64_division_at_the_limits():
    """uint64 // and % with quotients from 2^63 on (divisor 1) and divisors from about 2^63 on, constant and per sample
    (exact; the signed corrections failed there), int64 at the same bits; int64 randint over a narrow range (a uint64
    remainder: it gave values out of range), uint64 gcd (it looped forever)."""
    x = np.array([2 ** 64 - 2, 12345678901234567890, 2 ** 63, 2 ** 63 + 5, 7, 0, 2 ** 64 - 1, 2 ** 63 - 1], np.uint64)
    y = np.array([1, 2 ** 64 - 59, 2 ** 63 + 1, 2 ** 63, 3, 5, 2 ** 63 - 8000, 2], np.uint64)
    _same(lambda x, y: (x // y, x % y, x // jnp.uint64(1), x % jnp.uint64(2 ** 64 - 59), x // jnp.uint64(2 ** 63 - 5000),
                        jnp.gcd(x, y), x.astype(jnp.int64) // y.astype(jnp.int64), x.astype(jnp.int64) % y.astype(jnp.int64)), x, y)
    k = np.asarray(jax.random.split(jax.random.PRNGKey(0), 8))
    _same(lambda k: (jax.random.randint(k, (), 5, 6, dtype=jnp.int64), jax.random.randint(k, (3,), -7, 1000, dtype=jnp.int64),
                     jax.random.randint(k, (), 0, 2 ** 40, dtype=jnp.uint64)), k)


def test_uint64_total_order_and_sign():
    """uint64 values from 2^63 on in total-order compares (searchsorted) and sign (1, not 2^64 - 1)."""
    T = np.array([1, 5, 2 ** 63 - 1, 2 ** 63, 2 ** 63 + 10, 2 ** 64 - 1], np.uint64)
    q = np.array([0, 3, 2 ** 63 + 1, 2 ** 64 - 2, 2 ** 62, 2 ** 63 + 10], np.uint64)
    _same(lambda q: (jnp.searchsorted(T, q), jnp.searchsorted(T, q, side="right"),
                     jnp.searchsorted(T, q, method="compare_all"), jnp.sign(q)), q)


def test_integers_beyond_doubles_and_orders():
    """Integer tables in total-order compares (searchsorted, digitize: the keys mixed doubles' bits with integers),
    int64 constants beyond 2^53 in additions, compares and loop carries (rounded as doubles: nanosecond timestamps),
    int64 constants whose bits are a NaN (-1, x // 0: the canonical NaN's bits), uint64 constants from 2^63 on as a
    loop's start (OverflowError), uint64 indices from 2^63 on (clamped to 0), a bitcast to the same dtype."""
    q = np.array([1, 2, 5, 6, 9, 10], np.int32)
    T = jnp.array([1, 5, 9], jnp.int32)
    _same(lambda q: (jnp.searchsorted(T, q), jnp.searchsorted(T, q, side="right"), jnp.digitize(q, jnp.array([2, 6]))), q)
    x = np.array([1, 2, 3, -7], np.int32)
    _same(lambda x: (x.astype(jnp.int64) + jnp.int64(2 ** 60), jnp.int64(1_700_000_000_123_456_789) + x.astype(jnp.int64),
                     (x.astype(jnp.int64) + 2 ** 60) - 2 ** 60 == x.astype(jnp.int64),
                     jax.lax.fori_loop(0, 3, lambda i, c: c + x.astype(jnp.int64), jnp.int64(2 ** 60 + 1)),
                     jax.lax.fori_loop(0, 3, lambda i, h: (h ^ x.astype(jnp.uint64)) * jnp.uint64(1099511628211),
                                       jnp.uint64(14695981039346656037))), x)
    n = np.array([3, -4, 0, 7], np.int64)
    _same(lambda n: (-1 + 0 * n, jax.lax.div(n, jnp.int64(0)), jnp.full(2, -5, jnp.int64) + 0 * n,
                     jnp.uint64(2 ** 64 - 3) + 0 * n.astype(jnp.uint64)), n)
    _same(lambda x: jax.lax.bitcast_convert_type(x, jnp.float64), np.array([1.5, -2.25]))
    u = np.array([2 ** 63, 3, 2 ** 64 - 1, 1], np.uint64)
    _same(lambda u: (jax.lax.dynamic_index_in_dim(jnp.arange(6) + 10, u, keepdims=False), (jnp.arange(6.0) * 2)[u]), u)


def test_int64_in_their_dtype():
    """64-bit integer inputs and outputs of a kernel: their bits read from and written into buffers of their dtype,
    not converted by XLA around the call (a copy in one thread: AutoPDEx 1.2.0's sparse indices, 576 per element, took
    longer than the kernel); exact beyond 2^53, the outputs' tangents float0."""
    import kernelax
    f = lambda x, k: (jnp.sin(x), (x * 1e3).astype(jnp.int64) * 2 ** 40 + k, (x * 1e3).astype(jnp.uint64) << 50)
    x, k = np.linspace(0.1, 3.0, 999), np.arange(999) * 2 ** 54 + 3
    g = jax.vmap(kernelax.kernel(f))
    (s, i, u), (rs, ri, ru) = g(x, k), jax.vmap(f)(x, k)
    np.testing.assert_allclose(s, rs, rtol=1e-15)
    assert i.dtype == ri.dtype and u.dtype == ru.dtype
    np.testing.assert_array_equal(i, ri)
    np.testing.assert_array_equal(u, ru)
    assert "bitcast_convert" not in jax.jit(g).lower(x, k).as_text()
    _, (ts, ti, tu) = jax.jvp(lambda x: g(x, k), (x,), (np.ones_like(x),))
    np.testing.assert_allclose(ts, np.cos(x), rtol=1e-14)
    assert ti.dtype == tu.dtype == jax.dtypes.float0
