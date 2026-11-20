# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from kernelax import kernel
from kernelax.vmap import cvmap
from kernelax.build import simd_width

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(0)
SPECIAL = [0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, 1e-310, 2.2250738585072014e-308, 1.0, -1.0, 1e300, -1e300]
logu = lambda n: np.exp(rng.uniform(np.log(1e-300), np.log(1e300), n))
CASES = {  # function: (jax, numpy, inputs)
    "exp": (jnp.exp, np.exp, [rng.uniform(-745, 710, 20000), rng.uniform(-1, 1, 10000)]),
    "expm1": (jnp.expm1, np.expm1, [rng.uniform(-50, 710, 20000), rng.uniform(-1e-3, 1e-3, 10000)]),
    "tanh": (jnp.tanh, np.tanh, [rng.uniform(-30, 30, 20000), rng.uniform(-1e-3, 1e-3, 10000)]),
}


def ulp(a, b):
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    ia, ib = (np.where(x.view(np.int64) < 0, np.int64(-2 ** 63) - x.view(np.int64), x.view(np.int64)) for x in (a, b))
    with np.errstate(over="ignore"):
        return np.where(same, 0, np.abs(ia - ib).astype(np.float64))


@pytest.mark.parametrize("W", [4, 8] if simd_width() == 8 else [simd_width()])
@pytest.mark.parametrize("name", list(CASES))
def test_inline_functions(name, W):
    """The inline elementary functions (emit._MATH) within 3 ulp of the C library, special values included."""
    fj, fn, parts = CASES[name]
    x = np.concatenate(parts + [SPECIAL])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = fn(x)
    assert ulp(cvmap(fj, [x[0]], (0,), W=W, K=1)(x), ref).max() <= 3


def test_error_free_products():
    """Products that error-free transformations need rounded are not contracted into their uses (-ffp-contract=fast
    fused them): two_prod as JAX writes it (Veltkamp's split) gives p and e bit-equal to XLA, and gamma, lgamma,
    digamma of float64 (JAX's double-word algorithms) are within an ulp of XLA's (they were up to 64, 7 and 1300 ulp
    off)."""
    def two_prod(a, b):
        p = a * b
        c, d = 134217729.0 * a, 134217729.0 * b
        ah, bh = c - (c - a), d - (d - b)
        al, bl = a - ah, b - bh
        return p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl
    r = np.random.default_rng(91)
    x = r.uniform(-1e3, 1e3, (4096, 2))
    f = lambda v: two_prod(v[0], v[1])
    for o, ref in zip(cvmap(f, (x[0],), (0,))(x), jax.vmap(f)(x)):
        np.testing.assert_array_equal(o, np.asarray(ref))
    z = r.uniform(-10, 30, 20000)
    for g in (jax.scipy.special.gamma, jax.lax.lgamma, jax.lax.digamma):
        a, b = np.asarray(cvmap(g, (z[0],), (0,))(z)), np.asarray(jax.vmap(g)(z))
        ok = np.isfinite(b)
        assert np.all(np.abs(a[ok] - b[ok]) <= np.spacing(np.abs(b[ok]))), g


def test_erf_inv():
    """erf_inv (XLA's algorithm, expanded): within 1e-15 of JAX, special values identical."""
    x = np.concatenate([rng.uniform(-1, 1, 20000), 1 - np.exp(rng.uniform(np.log(1e-16), 0, 5000)),
                        [0.0, -0.0, 1.0, -1.0, 1.5, np.nan, 1e-300]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_allclose(cvmap(jax.lax.erf_inv, [x[0]], (0,))(x), jax.vmap(jax.lax.erf_inv)(x), rtol=1e-15)


@pytest.mark.parametrize("name", ["lgamma", "digamma"])
def test_gamma_functions(name):
    """lgamma and digamma (JAX's implementations, as XLA lowers them): as close to the exact values as XLA's, poles
    and special values like XLA's."""
    f, exact = getattr(jax.lax, name), {"lgamma": special.gammaln, "digamma": special.digamma}[name]
    x = np.concatenate([rng.uniform(-20, 20, 20000), logu(2000), -np.exp(rng.uniform(np.log(1e-300), np.log(1e15), 2000))])
    s = np.array([0.0, -0.0, 1.0, 2.0, -1.0, -2.5, 0.5, -0.5, np.inf, -np.inf, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, ref = cvmap(f, [x[0]], (0,))(np.concatenate([x, s])), np.asarray(jax.vmap(f)(np.concatenate([x, s])))
        e, ok = exact(x), np.isfinite(exact(x))
    np.testing.assert_allclose(out[len(x):], ref[len(x):], rtol=3e-14)
    err = lambda a: np.abs(a[:len(x)][ok] - e[ok]) / np.maximum(np.abs(e[ok]), 1)
    assert err(out).max() <= 1.5 * err(ref).max()


def _same(g, *args, axes=None):  # cvmap of g against jax.vmap, bitwise (NaN equal to NaN, -0.0 apart from 0.0)
    axes = axes or (0,) * len(args)
    ex = [a[0] if ax == 0 else a for a, ax in zip(args, axes)]
    for o, r in zip(jax.tree.leaves(cvmap(g, ex, axes)(*args)), jax.tree.leaves(jax.vmap(g, axes)(*args))):
        o, r = np.asarray(o), np.asarray(r)
        np.testing.assert_array_equal(o, r.astype(o.dtype))
        if o.dtype.kind == "f":
            np.testing.assert_array_equal(np.signbit(o), np.signbit(r))


def test_total_order_and_logsumexp_at_infinities():
    """Total-order compares (searchsorted with NaN, compare_all and the default method) as XLA's TOTALORDER;
    cumlogsumexp of equal infinities (not NaN)."""
    T = np.array([0.0, 1.0, 2.0, np.nan, np.nan])
    a = np.stack([np.array([0.0, 1.0, np.nan, np.nan]), np.array([-np.inf, 0.0, 5.0, np.inf])] * 3)
    q = np.array([np.nan, 0.5, -0.0, 3.0, np.inf, -1.0])
    _same(lambda q: (jnp.searchsorted(T, q, method="compare_all"), jnp.searchsorted(T, q), jnp.searchsorted(T, q, side="right")), q)
    _same(lambda a: (jnp.searchsorted(a, 0.5), jnp.searchsorted(a, jnp.nan), jnp.searchsorted(a, 7.0, method="compare_all")), a)
    x = np.array([[-np.inf, -np.inf, 1.0], [np.inf, np.inf, 1.0], [np.inf, -np.inf, 2.0], [0.5, np.nan, 1.0]])
    _same(lambda x: (jax.lax.cumlogsumexp(x), jax.lax.cumlogsumexp(x, reverse=True)), x)


def test_remainder_and_rounding_edges():
    """Float remainder exact (fmod, sign of the dividend, -0.0), floor division, and rounding half away from zero
    keeping the sign of zero."""
    x = np.array([1.0, -6.0, 6.0, 7.5, -7.5, 1e15 + 3, 0.3, -0.1, 5.0, 1e-300])
    y = np.array([0.1, 3.0, -3.0, 2.0, 0.25, 3.0, 0.1, 0.1, -0.0, 7e-301])
    _same(lambda x, y: (jax.lax.rem(x, y), jnp.fmod(x, y), x // y, jnp.remainder(x, y)), x, y)
    r = np.array([-0.1, -0.5, 0.5, -2.5, 2.5, -0.0, 0.49999999999999994, -1e-300])
    _same(lambda r: (jax.lax.round(r, jax.lax.RoundingMethod.AWAY_FROM_ZERO), jax.lax.round(r, jax.lax.RoundingMethod.TO_NEAREST_EVEN)), r)


def test_complex_edges_and_approx_min_k():
    """Complex pow at a zero base (1, 0 or NaN as XLA), tanh and tan of large arguments (no inf / inf), approx_min_k
    with NaN (the k smallest ascending, NaN last)."""
    y = np.array([0.0, 2.5, -1.0, 1j, 2 + 1j, 0.5 - 0.5j, 3.0], complex)
    z = np.array([800 + 0.5j, -800 + 2j, 20 + 1j, 19 + 3j, 0.3 - 0.2j, -5 + 1e-3j, 1e-300 + 0j], complex)
    for f, a in ((lambda y: jnp.power(jnp.zeros((), complex), y), y), (jnp.tanh, z), (jnp.tan, z * 1j), (jnp.tan, z)):
        o, r = cvmap(f, [a[0]], (0,))(a), jax.vmap(f)(a)
        np.testing.assert_allclose(o, r, rtol=1e-14, atol=1e-300)
    x = np.array([[1.0, np.nan, 3.0, 3.0, -0.0, 0.0], [5.0, 4.0, np.nan, np.nan, 2.0, 1.0]])
    f = lambda x: jax.lax.approx_min_k(x, 3)
    for o, r in zip(cvmap(f, [x[0]], (0,))(x), jax.vmap(f)(x)):
        np.testing.assert_array_equal(o, r)


def test_complex_signed_zeros_and_infinities():
    """Complex functions at signed zeros and infinities as XLA: log of -0 (imaginary part pi), exp of a large real part
    with a zero imaginary part (0, not NaN), division by zero (inf, not NaN), sqrt of -4-0j (-2j: the branch cut)."""
    re = np.array([-0.0, 0.0, -4.0, 710.0, 1.0, -1.0, 0.0, 3.0])
    im = np.array([0.0, -0.0, -0.0, 0.0, 1.0, -0.0, 0.0, -2.0])
    f = lambda a, b: (jnp.log(jax.lax.complex(a, b)), jnp.exp(jax.lax.complex(a, b)), jnp.sqrt(jax.lax.complex(a, b)),
                      jax.lax.complex(a, b) / jax.lax.complex(0.0, 0.0), jnp.log(a.astype(complex)), jnp.sqrt(a.astype(complex)))
    for o, r in zip(cvmap(f, [re[0], im[0]], (0, 0))(re, im), jax.vmap(f)(re, im)):
        for p, q in ((np.asarray(o).real, np.asarray(r).real), (np.asarray(o).imag, np.asarray(r).imag)):
            np.testing.assert_allclose(p, q, rtol=1e-15, atol=0)  # (inf and NaN where XLA has them)
            np.testing.assert_array_equal(np.signbit(p), np.signbit(q))



def test_scatter_apply():
    """x.at[i].apply(f): f of the old values (the scatter's update_jaxpr; it was applied to position codes, reading
    other entries), at constant, repeated, per-sample and dropped indices, of floats and integers."""
    x = np.array([[1.0, 5.0, 7.0, 2.0, 3.0, 9.0], [0.5, -1.0, 2.0, 4.0, -3.0, 1.0], [8.0, 0.0, -2.0, 6.0, 1.5, -4.0]])
    k = np.array([2, 5, 0], np.int32)
    _same(lambda x, k: (x.at[2].apply(lambda v: v * 2), x.at[jnp.array([1, 1, 4])].apply(lambda v: v * 2 + 1),
                        x.at[k].apply(lambda v: jnp.floor(v * 0.5)), x.at[k + 4].apply(jnp.negative, mode="drop"),
                        x.astype(jnp.int32).at[k].apply(lambda v: v * 3 - 1)), x, k)


@pytest.mark.xfail(strict=True, reason="divisions by a shared reciprocal: beyond 2^1022 (float32: 2^126) it is subnormal, "
                   "in XLA's mode zero; scaling such denominators costs up to 25 %")
def test_division_by_huge_denominators():
    """x / y for |y| beyond 2^1022 (float32: 2^126), whose reciprocal is subnormal: in XLA's mode flushed to zero, so
    the quotient is 0 or NaN."""
    x = np.array([1e300, 1e308, np.inf, 1.0, -3e307, 5.0, 0.0, 1e-300, 7.0])
    y = np.array([1e308, 1e308, 1e308, 2.0 ** 1023, -1.7e308, 2.0, 1e308, 1e308, np.inf])
    xf, yf = np.float32([3e38, 1e38, 2.0, np.inf, 1.0, 6.0]), np.float32([2e38, -3e38, 1e38, 3e38, 2.0 ** 127, 3.0])
    f = lambda a, b: (a / b, (a * 0.5) / b, 1.0 / b, a / jnp.asarray(1.5e308, a.dtype) if a.dtype == jnp.float64 else a / b)
    for a, b, rtol in ((x, y, 1e-15), (xf, yf, 1e-6)):
        for o, r in zip(jax.jit(jax.vmap(kernel(f)))(a, b), jax.jit(jax.vmap(f))(a, b)):
            np.testing.assert_allclose(np.asarray(o), np.asarray(r), rtol=rtol, atol=0)


def test_remainder_of_large_quotients_and_infinities():
    """fmod beyond the quotients the vectors compute exactly (float32 from 2^23 on, doubles from 2^52 on: there libm's
    fmod; it was off by multiples of y) and at infinities (inf % inf: NaN, it was inf), as XLA, bitwise."""
    x = np.array([1e9, 1.34e8, 1e10, 1.0, 18048.0, np.inf, -np.inf, 5.0, -0.0, 1e300, 3.0, -7.5, 2.0 ** 60])
    y = np.array([3.0, 3.0, 7.0, 1e-20, -7.96e-5, np.inf, np.inf, -np.inf, 2.0, 1e-300, 0.0, 0.25, 3.0])
    for dt in (np.float64, np.float32):
        _same(lambda x, y: (jax.lax.rem(x, y), jnp.fmod(x, y), jnp.remainder(x, y)), x.astype(dt), y.astype(dt))
    f = lambda x, y: (jax.lax.rem(x, y), jnp.remainder(x, y))  # (through XLA: in its mode, subnormals flushed)
    for o, r in zip(jax.jit(jax.vmap(kernel(f)))(x, y), jax.jit(jax.vmap(f))(x, y)):
        np.testing.assert_array_equal(np.asarray(o), np.asarray(r))


def test_top_k_in_total_order():
    """top_k (and jnp.partition through it) in XLA's total order: -NaN lowest, NaN highest, 0.0 above -0.0 (it ordered
    as sort: every NaN highest, -0.0 equal to 0.0); approx_max_k as XLA's: as sort."""
    mn = np.copysign(np.nan, -1.0)
    x = np.array([[1.0, mn, 2.0, np.nan, 0.5, -0.0], [-0.0, 0.0, -0.0, 3.0, mn, 1.0], [0.0, -0.0, np.nan, mn, -np.inf, 2.0]])
    _same(lambda x: (jax.lax.top_k(x, 3), jnp.partition(x, 2), jax.lax.approx_max_k(x, 3), jax.lax.top_k(x.astype(jnp.float32), 4)), x)


def test_gamma_functions_of_float32():
    """float32 lgamma and digamma in double, rounded once: within 1 ulp of the exact values (XLA: lgamma 3 ulp; the
    Lanczos series in float32 was off by up to 2^30 ulp at lgamma's negative zeros)."""
    x = np.concatenate([rng.uniform(-20, 30, 20000), 1 + rng.uniform(-1e-3, 1e-3, 500), 2 + rng.uniform(-1e-3, 1e-3, 500),
                        -rng.uniform(2, 10, 5000), [1.0, 2.0, 0.5, -0.5, -2.5, 1e-30]]).astype(np.float32)
    bits = lambda a: np.where(a.view(np.int32) < 0, np.int32(-2 ** 31) - a.view(np.int32), a.view(np.int32)).astype(np.int64)
    for f, exact in ((jax.lax.lgamma, special.gammaln), (jax.lax.digamma, special.digamma)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, e = np.asarray(cvmap(f, [x[0]], (0,))(x), np.float32), exact(x.astype(np.float64)).astype(np.float32)
        ok = np.isfinite(e)
        assert np.abs(bits(out[ok]) - bits(e[ok])).max() <= 1, f
    s = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0, -3.0], np.float32)  # (poles, special values as XLA's)
    for f in (jax.lax.lgamma, jax.lax.digamma):
        np.testing.assert_array_equal(cvmap(f, [s[0]], (0,))(s), jax.vmap(f)(s))


def test_complex_sinh_cosh_of_real_arguments():
    """sinh and cosh of a + 0i: imaginary part +0 as XLA (cosh(a) sin(0) was inf * 0 = NaN for a large a; the sign of a
    zero b was kept)."""
    z = np.array([1e300 + 0j, complex(-1e300, -0.0), 800 + 0j, complex(3, -0.0), -3 + 0j, complex(0, -0.0), 800 + 2j, 1e300 + 1j])
    for f in (jnp.sinh, jnp.cosh):
        o, r = np.asarray(cvmap(f, [z[0]], (0,))(z)), np.asarray(jax.vmap(f)(z))
        for p, q in ((o.real, r.real), (o.imag, r.imag)):
            np.testing.assert_allclose(p, q, rtol=1e-15, atol=0)
            np.testing.assert_array_equal(np.signbit(p)[~np.isnan(q)], np.signbit(q)[~np.isnan(q)])


def test_preferred_element_type_and_folded_scales():
    """dot_general and conv_general_dilated of float32 operands with preferred_element_type=float64: products and sums
    in double, as XLA (they ran in float32); c1 (c2 x) folded only where c1 c2 is a normal number ((x 2^600) 2^600 was
    x inf; float32 (2^20 2^-100) 2^-30 flushed to zero)."""
    a = np.random.default_rng(66).uniform(-1, 1, (7, 3, 4)).astype(np.float32)
    b = np.random.default_rng(67).uniform(-1, 1, (7, 4)).astype(np.float32)
    f = lambda a, b: (jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float64),
                      jax.lax.conv_general_dilated(b[None, None], b[None, None, :2], (1,), "VALID", preferred_element_type=jnp.float64))
    for o, r in zip(cvmap(f, [a[0], b[0]], (0, 0))(a, b), jax.vmap(f)(a, b)):
        np.testing.assert_allclose(o, r, rtol=1e-15, atol=0)
    x = np.array([1e-300, 3e-300, 1.0])
    _same(lambda x: (x * 2.0 ** 600) * 2.0 ** 600, x)
    y = np.array([1.0, 3.0], np.float32)
    _same(lambda y: (y * np.float32(2.0 ** 20)) * np.float32(2.0 ** -100) * np.float32(2.0 ** -30), y)


def test_linear_algebra_of_degenerate_matrices():
    """eigh of clustered eigenvalues (8 x 8 {0, 0, 0, 1, 2, 2, 2, 2}: fixed sweeps left the clusters coupled), svd with
    vanishing singular values (ranks of 12 x 8 matrices with singular values in {0, 1, 2}; u orthonormal: its zero
    columns made Kabsch's rotation of planar point sets singular), eig of the zero matrix (NaN vectors), complex tan
    near its pole (cosh 2a + cos 2b cancelled)."""
    r, n, eps = np.random.default_rng(68), 64, np.finfo(np.float64).eps
    Q = np.linalg.qr(r.standard_normal((n, 8, 8)))[0]
    A = Q @ (np.array([0, 0, 0, 1, 2, 2, 2, 2.0])[:, None] * np.swapaxes(Q, -1, -2))
    w, v = cvmap(jnp.linalg.eigh, [A[0]], (0,))(A)
    assert np.abs(A @ v - v * w[:, None, :]).max() < 100 * eps
    U, Vt = np.linalg.qr(r.standard_normal((n, 12, 8)))[0], np.linalg.qr(r.standard_normal((n, 8, 8)))[0]
    M = (U * r.choice([0.0, 1.0, 2.0], (n, 1, 8))) @ Vt
    np.testing.assert_array_equal(cvmap(jnp.linalg.matrix_rank, [M[0]], (0,))(M), np.linalg.matrix_rank(M))
    P = r.standard_normal((n, 10, 3)) * np.array([1.0, 1.0, 0.0])  # (planar)
    H = np.swapaxes(P, -1, -2) @ P
    for a in (H, np.array([[[1.0, 2, 3], [1, 2, 3], [0, 0, 1]], [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]]])):
        u, s, vt = cvmap(jnp.linalg.svd, [a[0]], (0,))(a)
        np.testing.assert_allclose(np.swapaxes(u, -1, -2) @ u, np.broadcast_to(np.eye(3), u.shape), atol=1e-13)
        np.testing.assert_allclose((u * s[:, None, :]) @ vt, a, atol=1e-13 * (1 + np.abs(a).max()))
    z = np.zeros((2, 3, 3))
    for o, q in zip(cvmap(jnp.linalg.eig, [z[0]], (0,))(z), jax.vmap(jnp.linalg.eig)(z)):
        np.testing.assert_array_equal(o, q)
    t = np.array([1.5707 + 1e-5j, np.pi / 2 + 0j, 1.5707963 + 0j])
    np.testing.assert_allclose(cvmap(jnp.tan, [t[0]], (0,))(t), jax.vmap(jnp.tan)(t), rtol=1e-13)


def test_eig_and_schur_of_tiny_and_huge_matrices():
    """eig, eigvals and schur scaled as dgeev and dgees (the largest |entry| into [smlnum, bignum], then back): float64
    entries of 1e-160 lost 1 % and of 1e-300 gave diag(A), float32 entries of 1e-15 lost 2.6e-5 and of 1e-17 gave
    NaN."""
    B = np.random.default_rng(74).standard_normal((6, 4, 4))
    srt = lambda w: np.sort_complex(np.asarray(w[0]) + 1j * np.asarray(w[1]))
    for x64, cs, tol in ((True, (1e-160, 1e-300, 1e150), 1e-10), (False, (1e-15, 1e-17, 1e19), 2e-4)):
        dt = np.float64 if x64 else np.float32
        with jax.enable_x64(x64):
            ev = jax.vmap(kernel(lambda a: (jnp.real(jnp.linalg.eigvals(a)), jnp.imag(jnp.linalg.eigvals(a)))))
            sc = jax.vmap(kernel(lambda a: jax.lax.linalg.schur(a)))
            w1 = srt(ev(B.astype(dt)))
            for c in cs:
                A = (B * c).astype(dt)
                np.testing.assert_allclose(srt(ev(A)) / c, w1, rtol=tol, atol=tol * np.abs(w1).max(), err_msg=f"{dt.__name__} {c}")
                T, Z = (np.asarray(m, np.float64) for m in sc(A))  # (the order of T's eigenvalues: rounding's)
                assert np.all(np.tril(T, -2) == 0)
                np.testing.assert_allclose(Z @ T @ np.swapaxes(Z, 1, 2) / c, A / c, atol=tol * np.abs(B).max(), err_msg=f"{dt.__name__} {c}")
