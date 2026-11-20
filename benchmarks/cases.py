# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Per-sample test functions for coverage (tests/test_cases.py) and efficiency (benchmarks/run.py).

Each case: f(*args) for one sample, make(rng, n) -> args (batched ones with a leading axis n, the
others as they are), in_axes (0: batched, None: the same for all samples), and xfail: the reason if
the generator does not cover the case yet (the test is a strict xfail: it flags when it passes).
"""

import os
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, expm, schur, sqrtm
from jax.scipy.special import erf
from jax.scipy.stats import norm

jax.config.update("jax_enable_x64", os.environ.get("KERNELAX_X64", "1") == "1")  # KERNELAX_X64=0: float32

Case = namedtuple("Case", "name f make in_axes xfail", defaults=((0,), None))


def _spd(r, n, k):  # n symmetric positive definite k x k matrices
    B = r.standard_normal((n, k, k))
    return B @ B.transpose(0, 2, 1) / k + np.eye(k)


# ---------------------------------------------------------------- arithmetic and elementary functions

def horner(x):  # pure FMA work: degree 15 polynomial per value
    p = jnp.zeros_like(x)
    for c in np.linspace(1.0, 0.1, 16):
        p = p * x + c
    return p


def transcendental(x):
    return jnp.stack([jnp.exp(-x), jnp.log(x), jnp.sin(x), jnp.cos(x), jnp.tanh(x), jnp.sqrt(x), x ** 0.37,
                      jnp.arctan2(x, 1 + x), erf(x)])


def logsumexp(x):
    return jax.nn.logsumexp(x), jax.nn.softmax(x)


def reductions(x):
    return jnp.prod(x), jnp.linalg.norm(x), jnp.cumprod(x)


def int_ops(x):
    n = (x * 100).astype(jnp.int32)
    return n // 7, n % 7, n & 3


# ---------------------------------------------------------------- AD and mechanics

def neo_hooke(F, prm):  # energy, first Piola stress and its tangent by AD
    mu, lam = prm[0], prm[1]

    def W(F):
        J = jnp.linalg.det(F)
        return mu / 2 * (jnp.sum(F * F) - 3) - mu * jnp.log(J) + lam / 2 * jnp.log(J) ** 2
    return W(F), jax.grad(W)(F), jax.hessian(W)(F)


def _j2(eps, prm):
    E, nu, sy, H, m = prm[0], prm[1], prm[2], prm[3], prm[4]
    mu, K = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
    I, tr = jnp.eye(3), jnp.trace(eps)
    s_tr = 2 * mu * (eps - tr / 3 * I)
    q = jnp.sqrt(1.5 * jnp.sum(s_tr * s_tr))
    r = lambda dg: q - 3 * mu * dg - sy - H * (1 - jnp.exp(-m * dg))
    body = lambda c: (c[0] - r(c[0]) / (-3 * mu - H * m * jnp.exp(-m * c[0])), c[1] + 1)
    dg, _ = jax.lax.while_loop(lambda c: (q > sy) & (jnp.abs(r(c[0])) > 1e-10 * sy) & (c[1] < 50), body, (0.0, 0))
    return K * tr * I + s_tr * (1 - 3 * mu * dg / q), dg


def j2(eps, prm):  # J2 return mapping (local Newton) with its consistent tangent
    return (*_j2(eps, prm), jax.jacfwd(lambda e: _j2(e, prm)[0])(eps))


def _eps(r, n):
    e = r.standard_normal((n, 3, 3)) * 2e-3
    return 0.5 * (e + e.transpose(0, 2, 1))


def black_scholes(p):  # call price and all first derivatives (S, K, T, r, sigma)
    def price(p):
        S, K, T, r, s = p
        d1 = (jnp.log(S / K) + (r + s * s / 2) * T) / (s * jnp.sqrt(T))
        return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d1 - s * jnp.sqrt(T))
    return price(p), jax.grad(price)(p)


def lennard_jones(x):  # energy and forces of a cluster of 8 atoms
    i, j = np.triu_indices(x.shape[0], 1)

    def E(x):
        r2 = jnp.sum((x[i] - x[j]) ** 2, axis=-1)
        return jnp.sum(4 * (r2 ** -6 - r2 ** -3))
    return E(x), -jax.grad(E)(x)


def _cluster(r, n):
    g = np.stack(np.meshgrid(*[np.arange(2.0)] * 3, indexing="ij"), -1).reshape(-1, 3) * 1.2
    return g + 0.05 * r.standard_normal((n, 8, 3))


# ---------------------------------------------------------------- loops and control flow

def kepler(p):  # eccentric anomaly by Newton (while), true anomaly
    M, e = p
    f = lambda E: E - e * jnp.sin(E) - M
    E, _ = jax.lax.while_loop(lambda c: (jnp.abs(f(c[0])) > 1e-14) & (c[1] < 50),
                              lambda c: (c[0] - f(c[0]) / (1 - e * jnp.cos(c[0])), c[1] + 1), (M, 0))
    return E, 2 * jnp.arctan2(jnp.sqrt(1 + e) * jnp.sin(E / 2), jnp.sqrt(1 - e) * jnp.cos(E / 2))


def newton_linesearch(p):  # Newton on a 2 x 2 system, backtracking line search (nested while)
    F = lambda x: jnp.array([x[0] ** 3 + x[1] - p[0], x[1] ** 3 - x[0] - p[1]])
    nrm = lambda x: jnp.sum(F(x) ** 2)

    def step(c):
        x, it = c
        dx = -jnp.linalg.solve(jax.jacfwd(F)(x), F(x))
        a, _ = jax.lax.while_loop(lambda d: (nrm(x + d[0] * dx) > (1 - 1e-4 * d[0]) * nrm(x)) & (d[1] < 30),
                                  lambda d: (d[0] / 2, d[1] + 1), (1.0, 0))
        return x + a * dx, it + 1
    x, it = jax.lax.while_loop(lambda c: (nrm(c[0]) > 1e-24) & (c[1] < 50), step, (jnp.zeros(2), 0))
    return x, it


def rk4(y0, mu, steps=40):  # Van der Pol, fixed RK4 steps (fori_loop: a scan; unrolled at 40 steps)
    f = lambda y: jnp.array([y[1], mu[0] * (1 - y[0] ** 2) * y[1] - y[0]])

    def step(i, y):
        h = 0.01
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return jax.lax.fori_loop(0, steps, step, y0)


def rk4_long(y0, mu):  # 400 steps: a loop in the kernel
    return rk4(y0, mu, 400)


def switch(x):  # lax.switch on a data-dependent index
    i = jnp.clip(jnp.floor(x[0] * 3), 0, 2).astype(jnp.int32)
    return jax.lax.switch(i, [jnp.sin, jnp.cos, lambda v: v ** 2], x[1])


def interp(x):  # table lookup (searchsorted)
    xp = np.linspace(0.0, 1.0, 32)
    return jnp.interp(x, xp, np.sin(3 * xp))


_T = jnp.asarray(np.cos(np.linspace(0, 3, 64))[:, None] * np.sin(np.linspace(0, 2, 48))[None, :])
_TAB = jnp.asarray(np.linspace(-1, 2, 40))


def bilinear(p):  # bilinear interpolation in a 64 x 48 table (gather with two index components)
    x, y = p[0] * 63, p[1] * 47
    i, j = jnp.clip(jnp.floor(x).astype(int), 0, 62), jnp.clip(jnp.floor(y).astype(int), 0, 46)
    u, v = x - i, y - j
    return (1 - u) * (1 - v) * _T[i, j] + u * (1 - v) * _T[i + 1, j] + (1 - u) * v * _T[i, j + 1] + u * v * _T[i + 1, j + 1]


def take(k):  # integer indices, partly out of bounds: fill (NaN), clip, a comparison of table values
    return jnp.take(_TAB, k), jnp.take(_TAB, k, mode="clip"), _TAB[jnp.clip(k, 0, 39)] > 0.5


def random(s):  # random numbers per sample from its seed (threefry: integer nodes; normal: erf_inv)
    k1, k2, k3 = jax.random.split(jax.random.fold_in(jax.random.key(42), s[0].astype(jnp.int64)), 3)
    return jax.random.uniform(k1, (4,)), jax.random.randint(k2, (2,), 0, 100), jax.random.normal(k3, (2,))


def sort(x):
    return jnp.sort(x), jnp.argmax(x), jnp.max(x)


# ---------------------------------------------------------------- small dense linear algebra

def inv3(A):
    return jnp.linalg.inv(A), jnp.linalg.det(A)


def solve6(A, b):  # LU with partial pivoting
    return jnp.linalg.solve(A, b), jnp.linalg.det(A)


def cholesky4(A, b):
    L = jnp.linalg.cholesky(A)
    return L, cho_solve((L, True), b)


def qr43(A):
    return jnp.linalg.qr(A)


def log_strain(C):  # eigh 3 x 3: eigenvalues and the matrix logarithm (both sign-invariant)
    w, v = jnp.linalg.eigh(C)
    return w, (v * jnp.log(w)) @ v.T


def eigvals6(A):
    return jnp.linalg.eigvalsh(A)


def expm3(A):
    return expm(A)


def quaternion(q, v):  # rotate 8 vectors by a normalized quaternion
    w, x, y, z = q / jnp.linalg.norm(q)
    R = jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return v @ R.T


# ---------------------------------------------------------------- complex numbers, fft, nonsymmetric eigenproblems

def impedance(p, w):  # an RLC circuit (R, L in series, C parallel to them) at 16 frequencies: |Z| and phase
    R, L, C = p[0], p[1], p[2]
    z = R + 1j * w * L
    Z = z / (1 + 1j * w * C * z)
    return jnp.abs(Z), jnp.angle(Z)


def complex_ode(z0, prm):  # RK4 for dz/dt = (i w - g) z + f e^(i t), 100 steps: a complex loop carry
    w, g, f = prm[0], prm[1], prm[2]
    rhs = lambda t, z: (1j * w - g) * z + f * jnp.exp(1j * t)
    h = 0.05

    def step(k, z):
        t = k * h
        k1 = rhs(t, z)
        k2 = rhs(t + h / 2, z + h / 2 * k1)
        k3 = rhs(t + h / 2, z + h / 2 * k2)
        k4 = rhs(t + h, z + h * k3)
        return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    z = jax.lax.fori_loop(0, 100, step, z0[0] + 1j * z0[1])
    return jnp.real(z), jnp.imag(z)


def spectrum32(x):  # power spectrum of 32 real samples (rfft) and its mean frequency
    p = jnp.abs(jnp.fft.rfft(x)) ** 2
    return p, jnp.sum(p * jnp.arange(17)) / jnp.sum(p)


def fft_conv16(a, b):  # circular convolution of two 16-point signals through the fft
    return jnp.fft.irfft(jnp.fft.rfft(a) * jnp.fft.rfft(b), 16)


def spectrum256(x):  # rfft of 256 samples (as products with twiddle matrices: work growing with n^2)
    return jnp.abs(jnp.fft.rfft(x)) ** 2


def eig4(A):  # eigenvalues of a nonsymmetric 4 x 4 matrix: their sorted moduli, the spectral radius
    m = jnp.abs(jnp.linalg.eigvals(A))
    return jnp.sort(m), jnp.max(m)


def schur4(A):  # the real Schur form's diagonal, sorted (the real parts of the eigenvalues)
    return jnp.sort(jnp.diag(schur(A)[0]))


def sqrtm3(A):  # principal square root of a nonsymmetric 3 x 3 matrix (complex Schur form inside)
    return jnp.real(sqrtm(A))


# ---------------------------------------------------------------- size: dense layers

def mlp(x, W1, b1, W2, b2, w3):  # 8 -> 32 -> 32 -> 1, value and input gradient (weights uniform)
    def g(x):
        h = jnp.tanh(W1 @ x + b1)
        return w3 @ jnp.tanh(W2 @ h + b2)
    return g(x), jax.grad(g)(x)


def _mlp(r, n):
    return (r.standard_normal((n, 8)), r.standard_normal((32, 8)) / 3, r.standard_normal(32),
            r.standard_normal((32, 32)) / 6, r.standard_normal(32), r.standard_normal(32))


def matmul16(A, B):  # per-sample 16 x 16 product: 4096 multiplications
    return A @ B


CASES = [
    Case("horner", horner, lambda r, n: (r.uniform(-1, 1, (n, 16)),)),
    Case("transcendental", transcendental, lambda r, n: (r.uniform(0.1, 2, (n, 8)),)),
    Case("logsumexp", logsumexp, lambda r, n: (r.standard_normal((n, 16)),)),
    Case("reductions", reductions, lambda r, n: (r.uniform(0.5, 1.5, (n, 8)),)),
    Case("int_ops", int_ops, lambda r, n: (r.uniform(0, 10, (n, 4)),)),
    Case("neo_hooke", neo_hooke, lambda r, n: (np.eye(3) + 0.1 * r.standard_normal((n, 3, 3)), np.array([1.0, 10.0])),
         (0, None)),
    Case("j2", j2, lambda r, n: (_eps(r, n), np.array([210e3, 0.3, 250.0, 200.0, 50.0])), (0, None)),
    Case("black_scholes", black_scholes, lambda r, n: (np.stack([r.uniform(50, 150, n), r.uniform(80, 120, n),
                                                                r.uniform(0.1, 2, n), r.uniform(0, 0.1, n),
                                                                r.uniform(0.1, 0.5, n)], 1),)),
    Case("lennard_jones", lennard_jones, lambda r, n: (_cluster(r, n),)),
    Case("kepler", kepler, lambda r, n: (np.stack([r.uniform(0, 2 * np.pi, n), r.uniform(0, 0.9, n)], 1),)),
    Case("newton_linesearch", newton_linesearch, lambda r, n: (r.uniform(-2, 2, (n, 2)),)),
    Case("rk4", rk4, lambda r, n: (r.standard_normal((n, 2)), np.array([1.5])), (0, None)),
    Case("rk4_long", rk4_long, lambda r, n: (r.standard_normal((n, 2)), np.array([1.5])), (0, None)),
    Case("switch", switch, lambda r, n: (r.uniform(0, 1, (n, 2)),)),
    Case("interp", interp, lambda r, n: (r.uniform(0, 1, (n, 4)),)),
    Case("bilinear", bilinear, lambda r, n: (r.uniform(0, 1, (n, 2)),)),
    Case("take", take, lambda r, n: (r.integers(-5, 45, (n, 3)),)),
    Case("random", random, lambda r, n: (np.arange(n, dtype=np.float64)[:, None],)),
    Case("sort", sort, lambda r, n: (r.standard_normal((n, 8)),)),
    Case("inv3", inv3, lambda r, n: (_spd(r, n, 3),)),
    Case("solve6", solve6, lambda r, n: (r.standard_normal((n, 6, 6)) + 3 * np.eye(6), r.standard_normal((n, 6))), (0, 0)),
    Case("cholesky4", cholesky4, lambda r, n: (_spd(r, n, 4), r.standard_normal((n, 4))), (0, 0)),
    Case("qr43", qr43, lambda r, n: (r.standard_normal((n, 4, 3)),)),
    Case("log_strain", log_strain, lambda r, n: (_spd(r, n, 3),)),
    Case("eigvals6", eigvals6, lambda r, n: (_spd(r, n, 6),)),
    Case("expm3", expm3, lambda r, n: (0.3 * r.standard_normal((n, 3, 3)),)),
    Case("impedance", impedance, lambda r, n: (np.stack([r.uniform(1, 100, n), r.uniform(1e-3, 1e-1, n), r.uniform(1e-6, 1e-4, n)], 1),
                                               np.logspace(1, 5, 16)), (0, None)),
    Case("complex_ode", complex_ode, lambda r, n: (r.standard_normal((n, 2)), np.stack([r.uniform(1, 5, n), r.uniform(0.1, 1, n),
                                                                                         r.uniform(0, 1, n)], 1)), (0, 0)),
    Case("spectrum32", spectrum32, lambda r, n: (r.standard_normal((n, 32)),)),
    Case("fft_conv16", fft_conv16, lambda r, n: (r.standard_normal((n, 16)), r.standard_normal((n, 16))), (0, 0)),
    Case("spectrum256", spectrum256, lambda r, n: (r.standard_normal((n, 256)),)),
    Case("eig4", eig4, lambda r, n: (r.standard_normal((n, 4, 4)),)),
    Case("schur4", schur4, lambda r, n: (r.standard_normal((n, 4, 4)),)),
    Case("sqrtm3", sqrtm3, lambda r, n: (r.standard_normal((n, 3, 3)) + 3 * np.eye(3),)),
    Case("quaternion", quaternion, lambda r, n: (r.standard_normal((n, 4)), r.standard_normal((n, 8, 3))), (0, 0)),
    Case("mlp", mlp, _mlp, (0, None, None, None, None, None)),
    Case("matmul16", matmul16, lambda r, n: (r.standard_normal((n, 16, 16)), r.standard_normal((n, 16, 16))), (0, 0)),
]


def example(case, args):
    """Per-sample example arguments (cvmap's `example`)."""
    return [a[0] if ax == 0 else a for a, ax in zip(args, case.in_axes)]
