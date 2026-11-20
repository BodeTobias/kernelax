# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""API coverage: jnp, linalg, scipy.special, jax.nn, control flow, AD and jax.random functions per sample, under
cvmap against jax.vmap. Gaps are strict xfails (they fail once covered)."""
import itertools

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg as jsl
import jax.scipy.special as jss
import numpy as np
import pytest
from jax.extend import core as jcore

from kernelax import frontend
from kernelax.vmap import cvmap

jax.config.update("jax_enable_x64", True)
r = np.random.default_rng(0)
n = 37
vec = lambda k=4, lo=-1.0, hi=1.0: r.uniform(lo, hi, (n, k))
pos = lambda k=4: r.uniform(0.2, 2.0, (n, k))
mat = lambda k=3: r.standard_normal((n, k, k)) + k * np.eye(k)
spd = lambda k=3: (lambda B: B @ B.transpose(0, 2, 1) / k + np.eye(k))(r.standard_normal((n, k, k)))
rect = lambda m=4, k=3: r.standard_normal((n, m, k))
ints = lambda k=4: r.integers(-20, 20, (n, k))
C = []  # (group, name, f, args)
def add(group, name, f, *args): C.append((group, name, f, args))
# --- jnp elementwise / math
for nm in ["abs", "sign", "floor", "ceil", "round", "trunc", "rint", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sqrt", "cbrt",
           "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arctanh", "square", "reciprocal",
           "negative", "positive", "sinc", "deg2rad", "rad2deg", "fabs", "signbit", "isnan", "isinf", "isfinite", "logical_not", "exp", "i0", "angle", "real", "conj"]:
    if hasattr(jnp, nm):
        add("jnp elementwise", nm, getattr(jnp, nm), pos() if nm in ("log", "log2", "log10", "sqrt") else vec(4, -0.9, 0.9))
add("jnp elementwise", "arccosh", jnp.arccosh, pos() + 1)
for nm in ["add", "subtract", "multiply", "divide", "true_divide", "floor_divide", "power", "float_power", "mod", "remainder", "fmod", "maximum",
           "minimum", "fmax", "fmin", "arctan2", "hypot", "copysign", "nextafter", "logaddexp", "logaddexp2", "heaviside", "ldexp",
           "greater", "less_equal", "equal", "not_equal", "logical_and", "logical_xor"]:
    if hasattr(jnp, nm):
        args = (pos(), vec()) if nm in ("power", "float_power") else (vec(), ints() % 5) if nm == "ldexp" else (vec(), vec() + 2.0)
        add("jnp binary", nm, getattr(jnp, nm), *args)
for nm in ["bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift", "gcd", "lcm"]:
    add("jnp integer", nm, getattr(jnp, nm), np.abs(ints()) + 1, np.abs(ints()) % 7 + 1)
add("jnp integer", "bitwise_not", jnp.bitwise_not, ints())
add("jnp integer", "population_count", jax.lax.population_count, np.abs(ints()))
add("jnp integer", "clz", jax.lax.clz, np.abs(ints()) + 1)
# --- reductions, sorting, scans
for nm in ["sum", "prod", "mean", "std", "var", "max", "min", "argmax", "argmin", "all", "any", "cumsum", "cumprod", "median",
           "ptp", "average", "nanmean", "nansum", "count_nonzero", "sort", "argsort", "diff", "cross", "trapezoid", "percentile"]:
    if not hasattr(jnp, nm): continue
    f = getattr(jnp, nm)
    f = (lambda x, f=f: f(x, 50.0)) if nm == "percentile" else (lambda x, f=f: f(x[:3], x[1:])) if nm == "cross" else f
    add("jnp reductions/sorting", nm, f, vec(4) if nm != "all" else vec(4) > 0)
add("jnp reductions/sorting", "searchsorted", lambda x: jnp.searchsorted(jnp.linspace(-1, 1, 9), x), vec())
add("jnp reductions/sorting", "lax.top_k", lambda x: jax.lax.top_k(x, 2), vec(6))
add("jnp reductions/sorting", "unique (size=)", lambda x: jnp.unique(jnp.round(x), size=4, fill_value=0), vec(6, -2, 2))
add("jnp reductions/sorting", "lax.cummax", lambda x: jax.lax.cummax(x), vec())
add("jnp reductions/sorting", "logsumexp", jax.nn.logsumexp, vec())
# --- array manipulation / indexing
add("jnp arrays", "reshape/transpose/concat", lambda x: jnp.concatenate([x.reshape(2, 2).T.ravel(), x[::-1]]), vec())
add("jnp arrays", "where/select", lambda x: jnp.where(x > 0, x, -2 * x), vec())
add("jnp arrays", "take (data index)", lambda x: x[(jnp.abs(x[0]) * 3).astype(int) % 4], vec())
two = lambda x: ((jnp.abs(x[0]) * 3).astype(int) % 3, (jnp.abs(x[1]) * 3).astype(int) % 3)  # two data indices
add("jnp arrays", "take (two data indices)", lambda a, x: a[two(x)] * a, mat(), vec())
add("jnp arrays", "gather (two data indices)", lambda a, x: a[jnp.stack(two(x))[None], jnp.stack(two(x))[::-1][None]], mat(), vec())
add("jnp arrays", "set (two data indices)", lambda a, x: a.at[two(x)].set(x[2]).at[two(x)[::-1]].add(1.0), mat(), vec())
add("jnp arrays", "dynamic_update_slice", lambda x: jax.lax.dynamic_update_slice(x, jnp.ones(2), ((x[0] > 0).astype(int),)), vec())
add("jnp arrays", ".at[].set (data index)", lambda x: x.at[(jnp.abs(x[0]) * 3).astype(int) % 4].set(5.0), vec())
add("jnp arrays", ".at[].add (data index)", lambda x: x.at[(jnp.abs(x[0]) * 3).astype(int) % 4].add(1.0), vec())
add("jnp arrays", "roll (data shift)", lambda x: jnp.roll(x, (jnp.abs(x[0]) * 3).astype(int)), vec())
add("jnp arrays", "outer/kron/einsum", lambda x: (jnp.outer(x, x), jnp.kron(x[:2], x[2:]), jnp.einsum("i,i->", x, x)), vec())
add("jnp arrays", "polyval/convolve", lambda x: (jnp.polyval(x, 0.3), jnp.convolve(x, x[:2])), vec())
add("jnp arrays", "interp", lambda x: jnp.interp(x, jnp.linspace(-1, 1, 20), jnp.linspace(-1, 1, 20) ** 3), vec())
add("jnp arrays", "fft.fft (complex)", lambda x: jnp.abs(jnp.fft.fft(x)), vec())
add("jnp arrays", "complex arithmetic", lambda x: jnp.abs((x[0] + 1j * x[1]) ** 2), vec())
# --- linalg
add("linalg", "det 2", jnp.linalg.det, mat(2)); add("linalg", "det 3", jnp.linalg.det, mat(3)); add("linalg", "det 5", jnp.linalg.det, mat(5))
add("linalg", "slogdet", lambda a: jnp.linalg.slogdet(a), mat(4))
add("linalg", "inv", jnp.linalg.inv, mat(4)); add("linalg", "solve", lambda a, b: jnp.linalg.solve(a, b), mat(4), vec(4))
add("linalg", "cholesky", jnp.linalg.cholesky, spd(4)); add("linalg", "qr", jnp.linalg.qr, rect(5, 3))
add("linalg", "qr complete", lambda a: jnp.linalg.qr(a, mode="complete"), rect(4, 3))
add("linalg", "eigh 3", jnp.linalg.eigvalsh, spd(3)); add("linalg", "eigh 6", jnp.linalg.eigvalsh, spd(6)); add("linalg", "eigh 8", jnp.linalg.eigvalsh, spd(8))
add("linalg", "eigh vectors (sign-free: v diag v^T)", lambda a: (lambda w, v: (v * w) @ v.T)(*jnp.linalg.eigh(a)), spd(4))
add("linalg", "svd values", lambda a: jnp.linalg.svd(a, compute_uv=False), rect(4, 3))
add("linalg", "svd (u s vt product)", lambda a: (lambda u, s, vt: (u * s) @ vt)(*jnp.linalg.svd(a, full_matrices=False)), rect(4, 3))
add("linalg", "eig (nonsymmetric)", lambda a: jnp.sort(jnp.abs(jnp.linalg.eigvals(a))), mat(3))
add("linalg", "pinv", jnp.linalg.pinv, rect(4, 3)); add("linalg", "lstsq", lambda a, b: jnp.linalg.lstsq(a, b)[0], rect(4, 3), vec(4))
add("linalg", "matrix_power 5", lambda a: jnp.linalg.matrix_power(a, 5), mat(3) / 3)
add("linalg", "norm 2 (matrix)", lambda a: jnp.linalg.norm(a, 2), mat(3)); add("linalg", "norm fro", lambda a: jnp.linalg.norm(a), mat(3))
add("linalg", "cond (condition number)", jnp.linalg.cond, mat(3)); add("linalg", "matrix_rank", jnp.linalg.matrix_rank, mat(3))
add("linalg", "scipy lu_factor/lu_solve", lambda a, b: jsl.lu_solve(jsl.lu_factor(a), b), mat(4), vec(4))
add("linalg", "scipy solve_triangular", lambda a, b: jsl.solve_triangular(jnp.tril(a), b, lower=True), mat(4), vec(4))
add("linalg", "scipy expm", jsl.expm, mat(3) / 5); add("linalg", "scipy sqrtm", lambda a: jnp.real(jsl.sqrtm(a)), spd(3))
add("linalg", "scipy logm (via eigh)", lambda a: (lambda w, v: (v * jnp.log(w)) @ v.T)(*jnp.linalg.eigh(a)), spd(3))
add("linalg", "scipy polar", lambda a: jsl.polar(a)[0], mat(3)); add("linalg", "scipy schur", lambda a: jnp.diag(jsl.schur(a)[0]), spd(3))
add("linalg", "scipy cho_factor/cho_solve", lambda a, b: jsl.cho_solve(jsl.cho_factor(a), b), spd(4), vec(4))
# --- special, nn, stats
for nm in ["erf", "erfc", "erfinv", "gammaln", "digamma", "gamma", "expit", "logit", "ndtr", "ndtri", "log_ndtr", "i0e", "i1e", "xlogy", "xlog1py",
           "entr", "betaln", "zeta", "polygamma", "betainc", "gammainc", "gammaincc", "expi", "exp1", "bessel_jn", "spence", "sph_harm"]:
    if not hasattr(jss, nm): continue
    f = getattr(jss, nm)
    if nm in ("xlogy", "xlog1py", "betaln", "zeta", "gammainc", "gammaincc"):
        add("scipy.special", nm, f, pos() + 1, pos() + 1)
    elif nm == "polygamma": add("scipy.special", nm, lambda x, f=f: f(1, x), pos())
    elif nm == "betainc": add("scipy.special", nm, lambda x, f=f: f(2.0, 3.0, x), vec(4, 0.1, 0.9))
    elif nm == "bessel_jn": add("scipy.special", nm, lambda x, f=f: f(x, v=2), pos())
    elif nm == "sph_harm": continue
    else: add("scipy.special", nm, f, vec(4, 0.1, 0.9) if nm in ("logit", "ndtri", "erfinv") else pos())
for nm in ["relu", "gelu", "silu", "softplus", "sigmoid", "softmax", "log_softmax", "elu", "selu", "leaky_relu", "hard_tanh", "one_hot"]:
    f = getattr(jax.nn, nm)
    add("jax.nn", nm, (lambda x, f=f: f(jnp.argmax(x), 4)) if nm == "one_hot" else f, vec())
add("jax.scipy.stats", "norm.logpdf/cdf/ppf", lambda x: (jsp.stats.norm.logpdf(x), jsp.stats.norm.cdf(x), jsp.stats.norm.ppf(jnp.abs(x) * 0.9 + 0.05)), vec())
add("jax.scipy.stats", "gamma.logpdf", lambda x: jsp.stats.gamma.logpdf(x, 2.0), pos())
# --- control flow, AD, random
add("control flow", "while_loop", lambda x: jax.lax.while_loop(lambda c: c[1] < 10, lambda c: (c[0] * 0.9 + 1, c[1] + 1), (x, 0))[0], vec())
add("control flow", "fori_loop (short)", lambda x: jax.lax.fori_loop(0, 5, lambda i, y: y * 0.5 + i, x), vec())
add("control flow", "fori_loop (long)", lambda x: jax.lax.fori_loop(0, 400, lambda i, y: jnp.sin(y) + 0.01 * i, x), vec())
add("control flow", "scan with xs", lambda x: jax.lax.scan(lambda c, a: (c * a, c), 1.0, x)[0], vec())
add("control flow", "scan long with xs", lambda x: jax.lax.scan(lambda c, a: (c + a, c), 0.0, jnp.tile(x, 100))[0], vec())
add("control flow", "cond", lambda x: jax.lax.cond(x[0] > 0, lambda y: jnp.sin(y), lambda y: y ** 2, x), vec())
add("control flow", "switch", lambda x: jax.lax.switch((jnp.abs(x[0]) * 2.9).astype(int), [jnp.sin, jnp.cos, jnp.tanh], x), vec())
add("control flow", "map (lax.map)", lambda x: jax.lax.map(jnp.sin, x), vec())
add("control flow", "inner vmap", lambda x: jax.vmap(lambda a: a * x[0])(x), vec())
add("control flow", "custom_jvp", lambda x: jax.nn.relu(x) + jax.numpy.logaddexp(x, 0.0), vec())
add("control flow", "checkpoint", lambda x: jax.checkpoint(lambda y: jnp.sin(y) ** 2)(x), vec())
add("AD", "grad", jax.grad(lambda x: jnp.sum(jnp.sin(x) * x)), vec())
add("AD", "hessian", jax.hessian(lambda x: jnp.sum(jnp.exp(x) * x[0])), vec())
add("AD", "jacrev of while (forward ok, reverse?)", jax.jacfwd(lambda x: jax.lax.while_loop(lambda c: c[1] < 5, lambda c: (c[0] * x, c[1] + 1), (x, 0))[0]), vec())
add("AD", "grad through fori_loop (scan)", jax.grad(lambda x: jax.lax.fori_loop(0, 5, lambda i, y: jnp.sin(y), x).sum()), vec())
add("random", "uniform/normal/randint", lambda s: (jax.random.uniform(jax.random.key(s[0].astype(int)), (2,)), jax.random.normal(jax.random.key(s[0].astype(int)), (2,))), r.integers(0, 1000, (n, 1)).astype(float))
add("random", "gamma", lambda s: jax.random.gamma(jax.random.key(s[0].astype(int)), 2.0, (2,)), r.integers(0, 1000, (n, 1)).astype(float))
add("random", "permutation/choice", lambda s: jax.random.permutation(jax.random.key(s[0].astype(int)), 5), r.integers(0, 1000, (n, 1)).astype(float))
add("dtypes", "float32 math", lambda x: jnp.sin(x.astype(jnp.float32)) * 2, vec())
add("dtypes", "float16", lambda x: (x.astype(jnp.float16) * 3).astype(jnp.float64), vec())
add("dtypes", "complex", lambda x: jnp.abs(jnp.exp(1j * x)), vec())


# --- more: cases found missing by a sweep over the API
add("more", "custom_root", lambda v: jax.lax.custom_root(lambda y: y ** 3 - v[0], 1.0, lambda g, y0: jax.lax.while_loop(
    lambda c: c[1] < 30, lambda c: (c[0] - g(c[0]) / (3 * c[0] ** 2), c[1] + 1), (y0, 0))[0], lambda g, y: y / 3), pos())
add("more", "random.multinomial", lambda v: jax.random.multinomial(jax.random.fold_in(jax.random.key(0), (v[0] * 1e3).astype(int)),
                                                                         5.0, jnp.array([0.2, 0.3, 0.5])), pos())
add("more", "device_put", lambda v: jax.device_put(v) * 2, vec())
add("more", "reduce_xor", lambda v: jax.lax.reduce_xor((v * 10).astype(jnp.int32), (0,)), vec())
add("more", "cumlogsumexp", lambda v: jax.lax.cumlogsumexp(v), vec())
add("more", "eigvalsh 2", jnp.linalg.eigvalsh, spd(2))
add("more", "searchsorted sort", lambda v: jnp.searchsorted(jnp.linspace(-1, 1, 5), v, method="sort"), vec())
add("more", "bincount length", lambda v: jnp.bincount((jnp.abs(v) * 4).astype(int), length=5), vec())
add("more", "grad of sort", jax.grad(lambda v: jnp.sum(jnp.sort(v) * jnp.arange(4.0))), vec())
add("more", "scatter max", lambda v: jnp.zeros(3).at[(jnp.abs(v) * 2).astype(int)].max(v), vec())
add("more", "reduce_window max", lambda v: jax.lax.reduce_window(v, -jnp.inf, jax.lax.max, (2,), (1,), "VALID"), vec())
add("more", "fft.rfft", lambda v: jnp.abs(jnp.fft.rfft(v)), vec())
add("more", "approx_max_k", lambda v: jax.lax.approx_max_k(v, 2)[0], vec())
add("more", "scatter min / mul (constant indices)", lambda v: (jnp.full(3, 0.5).at[jnp.array([0, 2, 0, 1])].min(v),
                                                               jnp.full(3, 2.0).at[jnp.array([0, 2, 0, 1])].mul(v)), vec())
add("more", "reduce_window 2d, padding, strides", lambda a: (jax.lax.reduce_window(a, 0.0, jax.lax.add, (2, 2), (1, 2), "SAME"),
                                                                jax.lax.reduce_window(a, jnp.inf, jax.lax.min, (2, 1), (1, 1), ((1, 0), (0, 1)),
                                                                                      window_dilation=(2, 1))), mat(3))
add("more", "approx_min_k", lambda v: jax.lax.approx_min_k(v, 2), vec())
add("more", "eigh 8 (vectors)", lambda a: (lambda w, v: (v * w) @ v.T)(*jnp.linalg.eigh(a)), spd(8))
# --- complex numbers inside (realify: pairs of real values), eig, schur
z = lambda x: x[0] + 1j * x[1]
add("complex", "div, sqrt, log, conj, sign, angle", lambda x: (jnp.abs(z(x) / (z(x).conj() + 2.0)), jnp.angle(jnp.sqrt(z(x))),
                                                                  jnp.real(jnp.log(z(x))), jnp.imag(jnp.sign(z(x)))), vec())
add("complex", "trig, powers", lambda x: (jnp.real(jnp.sin(z(x)) * jnp.cos(z(x))), jnp.imag(jnp.tan(z(x)) + jnp.tanh(z(x))),
                                            jnp.abs(z(x) ** -3), jnp.real(z(x) ** 2.5), jnp.imag(jnp.exp(z(x)) / (1 + z(x) ** 2))), vec())
add("complex", "matmul, sums, sort", lambda x: (lambda c: (jnp.abs(jnp.outer(c, c) @ c), jnp.real(jnp.cumsum(c)), jnp.imag(jnp.sum(c * c)),
                                                              jnp.imag(jnp.sort(c))))(x + 1j * x[::-1]), vec())
add("complex", "loops with complex carries", lambda x: (jnp.abs(jax.lax.fori_loop(0, 20, lambda i, c: c * (0.9 + 0.1j) + x[0], x[1] + 0j)),
                                                           jnp.real(jax.lax.scan(lambda c, a: (c * a, c), 1.0 + 0j, x * (1 + 1j))[0]),
                                                           jnp.imag(jax.lax.while_loop(lambda c: jnp.abs(c) < 5.0, lambda c: c * (1.2 + 0.3j) + 0.1, z(x) + 1.0)),
                                                           jnp.real(jax.lax.cond(x[2] > 0, lambda c: c * 1j, lambda c: c.conj(), z(x)))), vec())
add("complex", "ifft, irfft, fft2, odd lengths", lambda x: (jnp.real(jnp.fft.ifft(x)), jnp.fft.irfft(jnp.fft.rfft(x)), jnp.abs(jnp.fft.fft2(x.reshape(2, 3))),
                                                               jnp.abs(jnp.fft.rfft(x[:5])), jnp.fft.irfft(jnp.fft.rfft(x[:5]), 5)), vec(6))
add("complex", "eig 6 (nonsymmetric)", lambda a: jnp.sort(jnp.abs(jnp.linalg.eigvals(a))), mat(6))
add("complex", "eig 8 (nonsymmetric)", lambda a: jnp.sort(jnp.abs(jnp.linalg.eigvals(a))), mat(8))
add("complex", "schur 5 (Z T Z^T)", lambda a: (lambda T, Z: Z @ T @ Z.T)(*jsl.schur(a)), mat(5))
add("complex", "sqrtm (nonsymmetric)", lambda a: jnp.real(jsl.sqrtm(a)), mat(3))
add("complex", "eig vectors 4 (|v|, |A v - v w|)", lambda a: (lambda w, v: (jnp.abs(v), jnp.abs(a @ v - v * w)))(*jnp.linalg.eig(a)), mat(4))
add("complex", "eig vectors 6, badly scaled (balancing)", lambda a: (lambda w, v: (jnp.abs(w), jnp.abs(v)))(*jnp.linalg.eig(a)),
    mat(6) * np.exp(np.linspace(-6, 6, 6))[:, None] * np.exp(np.linspace(4, -4, 6))[None, :])
zm = lambda a: a + 1j * a[::-1].T  # (a complex matrix from a real one)
add("complex", "solve, inv, det, slogdet (5 x 5)", lambda a, v: (jnp.abs(jnp.linalg.solve(zm(a), v * (1 - 0.5j))), jnp.real(jnp.linalg.inv(zm(a))),
                                                                     jnp.angle(jnp.linalg.det(zm(a))), jnp.linalg.slogdet(zm(a))[1]), mat(5), vec(5))
add("complex", "lu (factors, pivots, permutation)", lambda a: (lambda lu, piv, perm: (jnp.real(lu), jnp.imag(lu), piv, perm))(*jax.lax.linalg.lu(zm(a))), mat(4))
add("complex", "lu_factor / lu_solve, solve_triangular (adjoint)", lambda a, v: (jnp.imag(jsl.lu_solve(jsl.lu_factor(zm(a)), v + 0j)),
                                                                                  jnp.real(jsl.solve_triangular(jnp.tril(zm(a)), v + 1j * v, lower=True, trans="C"))),
    mat(4), vec(4))
ri = lambda *xs: [p for x in xs for p in (jnp.real(x), jnp.imag(x))]
schur_inv = lambda T, Z: ri(jnp.diag(T)) + [jnp.abs(T), jnp.abs(Z), *ri(Z @ T @ Z.conj().T)]  # (LAPACK's phases: rounding)
add("complex", "eig of a complex matrix (values, vectors)", lambda a: ri(*jnp.linalg.eig(zm(a))), mat(5))
add("complex", "eig of a complex matrix, badly scaled (zgebal)", lambda a: ri(*jnp.linalg.eig(zm(a))),
    mat(5) * np.exp(np.linspace(-5, 5, 5))[:, None] * np.exp(np.linspace(4, -4, 5))[None, :])
add("complex", "eig of real values as complex (zgeev's order)", lambda a: ri(*jnp.linalg.eig(a + 0j)), mat(4))
add("complex", "eigvals of a complex matrix, float32", lambda a: ri(jnp.linalg.eigvals(zm(a))), mat(4).astype(np.float32))
add("complex", "schur of a complex matrix (up to phases)", lambda a: schur_inv(*jsl.schur(zm(a))), mat(5))
add("complex", "schur of real values, output complex (zgees' order)", lambda a: schur_inv(*jsl.schur(a, output="complex")), mat(5))
holes = lambda a: a.at[2, jnp.array([0, 1, 3, 4])].set(0).at[jnp.array([1, 2, 3, 4]), 0].set(0)  # (a row, a column: isolated)
blocks = lambda a: jnp.tril(a).at[1:3, 3:].set(a[1:3, 3:])[jnp.array([3, 0, 4, 1, 2])][:, jnp.array([3, 0, 4, 1, 2])]  # (rows
# and columns isolated)
add("eig", "eig of a lower triangular matrix (dgebal's permutations; |v|)", lambda a: (*ri(jnp.linalg.eigvals(jnp.tril(a))),
                                                                                       jnp.abs(jnp.linalg.eig(jnp.tril(a))[1])), mat(5))
add("eig", "eig with a row and a column isolated (real |v|; complex)", lambda a: (*ri(jnp.linalg.eigvals(holes(a))), jnp.abs(jnp.linalg.eig(holes(a))[1]),
                                                                                      *ri(*jnp.linalg.eig(holes(zm(a))))), mat(5))
add("eig", "eig of a permuted block triangular matrix, badly scaled", lambda a: ri(*jnp.linalg.eig(blocks(zm(a)))),
    mat(5) * np.exp(np.linspace(-5, 5, 5))[:, None] * np.exp(np.linspace(4, -4, 5))[None, :])
add("eig", "schur with isolated eigenvalues (real; complex up to phases)", lambda a: (
    (lambda T, Z: (jnp.diag(T), Z @ T @ Z.T))(*jsl.schur(holes(a))), schur_inv(*jsl.schur(blocks(zm(a))))), mat(5))
lr = lambda a, right=True: jax.lax.linalg.eig(a, compute_left_eigenvectors=True, compute_right_eigenvectors=right)
add("eig", "eig, left and right vectors (real: |v|)", lambda a: (lambda w, vl, vr: (*ri(w), jnp.abs(vl), jnp.abs(vr)))(*lr(a)), mat(5))
add("eig", "eig of a complex matrix, left and right vectors", lambda a: ri(*lr(zm(a))), mat(5))
add("eig", "eig, left vectors only, badly scaled, isolated rows (real |v|; complex)",
    lambda a: (jnp.abs(lr(holes(a), False)[1]), *ri(*lr(holes(zm(a)), False))),
    mat(5) * np.exp(np.linspace(-5, 5, 5))[:, None] * np.exp(np.linspace(4, -4, 5))[None, :])
GAPS = {}
TOL32 = {  # float32 eigenproblems: kernelax and XLA each off from the float64 solution by about the problem's condition
    # times eps (sample 17 of the badly scaled vectors: 5.8e-5 and 2.5e-5), and XLA's LAPACK rounds differently per CPU
    "eigvals of a complex matrix, float32": 1e-5, "eig vectors 6, badly scaled (balancing)": 1e-4}


@pytest.mark.parametrize("group, name, f, args", [
    pytest.param(*c, id=f"{c[0]}: {c[1]}", marks=[pytest.mark.xfail(strict=True, reason=GAPS[c[1]])] if c[1] in GAPS else [])
    for c in C])
def test_coverage(group, name, f, args):
    ref = jax.tree.leaves(jax.vmap(f)(*args))
    out = jax.tree.leaves(cvmap(f, [a[0] for a in args], (0,) * len(args), K=1)(*args))
    assert len(out) == len(ref)
    for o, x in zip(out, ref):
        rtol, atol = (1e-9, 1e-11) if np.asarray(x).dtype != np.float32 else (TOL32[name],) * 2 if name in TOL32 else (1e-6, 1e-7)  # (float32: ulps of the functions)
        x = np.asarray(x, np.float64)
        np.testing.assert_allclose(o, x, rtol=rtol, atol=atol * (1 + np.abs(np.where(np.isfinite(x), x, 0)).max(initial=0)), equal_nan=True)


def _prims(jaxpr):
    return {e.primitive.name for e in jaxpr.eqns}.union(*(_prims(getattr(j, "jaxpr", j)) for e in jaxpr.eqns
                                                          for j in jcore.jaxprs_in_params(e.params)))


@pytest.mark.parametrize("k", [2, 3])
def test_det_derivatives_through_the_adjugate_only_for_kernelax(k):
    """While kernelax traces, jnp.linalg.det of 2 x 2 and 3 x 3 matrices differentiates through its cofactors (JAX's
    rule solves with LU, _cofactor_solve); jax.jit keeps JAX's rule, also when traced before or after kernelax."""
    lu = {"lu", "triangular_solve"}
    for shape in [(k, k), (4, k, k)]:  # (plain first, then kernelax; kernelax first, then plain)
        jax.clear_caches()  # (a det derived under TRACING before, e.g. test_cases' neo_hooke, comes from JAX's caches
        # without a call of the rule: no count in ASKED; kernelax.jit then goes by the program's inner jaxprs)
        x = jax.ShapeDtypeStruct(shape, np.float64)
        f = jax.grad(lambda a: jnp.sum(jnp.linalg.det(a)))
        plain = lambda: _prims(jax.make_jaxpr(jax.jit(f))(x).jaxpr)
        asked = frontend.ASKED[0]
        if shape == (k, k):
            assert lu <= plain()
            assert frontend.ASKED[0] == asked  # (jax.jit's trace does not ask: kernelax.jit then traces once)
        with frontend.TRACING(True):
            assert not lu & _prims(jax.make_jaxpr(jax.hessian(lambda a: jnp.sum(jnp.linalg.det(a))))(x).jaxpr)
            assert not lu & _prims(jax.make_jaxpr(jax.jit(f))(x).jaxpr)
        assert frontend.ASKED[0] > asked  # (kernelax.jit traces XLA's program apart: JAX's rule)
        assert lu <= plain()


def _idet(m):  # exact determinant of an integer matrix
    return sum(np.prod([m[i][p[i]] for i in range(len(m))]) * (-1) ** sum(p[j] > p[i] for i in range(len(p)) for j in range(i))
               for p in itertools.permutations(range(len(m))))


@pytest.mark.parametrize("k", [2, 3])
def test_det_derivatives_exact_for_integer_and_singular_matrices(k):
    """Gradient (the cofactors) and Hessian of det under kernelax are exact for integer matrices, singular ones too
    (JAX's rule: 4e15 ulp off for some of rank k - 1, its Hessian NaN); float32, complex (JAX's rule needs complex
    cumprod) and a stack of matrices per sample agree with them; the value under value_and_grad is det's."""
    a = r.integers(-5, 6, (n, k, k)).astype(float)
    a[: n // 2, -1] = a[: n // 2, 0] * 2 if k == 2 else a[: n // 2, 0] + a[: n // 2, 1]  # rank k - 1
    a[n // 2 : n // 2 + 4] = r.integers(-3, 4, (4, k, 1)) * r.integers(-3, 4, (4, 1, k))  # rank 1
    E = np.eye(k * k).reshape(k * k, k, k)
    cof = np.array([[[_idet(m + E[i * k + j]) - _idet(m) for j in range(k)] for i in range(k)] for m in a.astype(int)])
    hess = np.array([[_idet(m + e + f) - _idet(m + e) - _idet(m + f) + _idet(m) if (e != f).any() else 0
                      for f in E] for m in a.astype(int) for e in E]).reshape(n, k, k, k, k)
    run = lambda f, *x: cvmap(f, [y[0] for y in x], (0,) * len(x), K=1)(*x)
    det = jnp.linalg.det
    np.testing.assert_array_equal(run(jax.grad(det), a), cof)
    np.testing.assert_array_equal(run(jax.hessian(det), a), hess)
    np.testing.assert_array_equal(run(jax.grad(det), a.astype(np.float32)), cof.astype(np.float32))
    v, g = run(jax.value_and_grad(det), a + 0.1)
    np.testing.assert_array_equal(v, run(det, a + 0.1))
    z = a + 1j * r.integers(-5, 6, a.shape)
    zc = np.array([[[_idet(m + E[i * k + j]) - _idet(m) for j in range(k)] for i in range(k)] for m in z.astype(complex)])
    np.testing.assert_array_equal(run(jax.grad(det, holomorphic=True), z), zc)
    s = np.stack([a, a[::-1]], 1)  # (n, 2, k, k): det of a stack per sample
    np.testing.assert_array_equal(run(jax.grad(lambda m: jnp.sum(det(m))), s), np.stack([cof, cof[::-1]], 1))
    nan = a[:1].copy()
    nan[0, 0, 0] = np.nan  # (a NaN reaches the cofactors that contain it; JAX's rule returns 0 for all)
    np.testing.assert_array_equal(np.isnan(run(jax.grad(det), nan)[0]), np.pad(np.ones((k - 1, k - 1), bool), ((1, 0), (1, 0))))
