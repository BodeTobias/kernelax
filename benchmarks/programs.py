# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Whole programs for kernelax.jit (automatic partitioning): batched code as written with jnp, or via vmap."""
import jax
import jax.numpy as jnp
import numpy as np


def black_scholes(x):  # columns: spot, strike, time
    s, k, t = x[:, 0], x[:, 1], x[:, 2]
    d1 = (jnp.log(s / k) + 0.5 * 0.04 * t) / (0.2 * jnp.sqrt(t))
    d2 = d1 - 0.2 * jnp.sqrt(t)
    cdf = lambda z: 0.5 * (1 + jax.scipy.special.erf(z / np.sqrt(2.0)))
    return s * cdf(d1) - k * jnp.exp(-0.03 * t) * cdf(d2)


bs_grad = jax.grad(lambda x: black_scholes(x).sum())  # the sensitivities (spot, strike, time) per row


def mlp_mean(x, w1, w2):  # rows through a small network, then the mean over all rows (XLA)
    return jnp.tanh(jnp.tanh(x @ w1) @ w2).mean(0)


def lj_energy(x):  # pair distances squared per row, then the total
    r2 = jnp.sum(x ** 2, axis=1)
    return jnp.sum(4 * (r2 ** -6 - r2 ** -3))


def centered_norms(x):  # a cross-row mean between row-wise parts
    y = x - x.mean(0)
    return jnp.sqrt(jnp.sum(y * y, axis=1)) * jnp.cumsum(y[:, 0])[-1]


inv3 = jax.vmap(jnp.linalg.inv)


def cbrt_newton(x):  # data-dependent while loop under vmap
    def one(a):
        cond = lambda c: jnp.abs(c[0] ** 3 - a) > 1e-12 * a
        return jax.lax.while_loop(cond, lambda c: (c[0] - (c[0] ** 3 - a) / (3 * c[0] ** 2), c[1] + 1), (a, 0))
    return jax.vmap(one)(10 * x[:, 0])


def oscillator(x):  # fixed-length fori_loop (a scan over carries) under vmap
    step = lambda i, y: y + 0.01 * jnp.sin(y) * jnp.cos(y[::-1])
    return jax.vmap(lambda y: jax.lax.fori_loop(0, 20, step, y))(x)


def ema(x):  # scan with xs and ys under vmap: exponential moving average and its variance along each row
    def one(row):
        step = lambda c, a: ((0.9 * c[0] + 0.1 * a, 0.9 * c[1] + 0.1 * (a - c[0]) ** 2), c[0])
        return jax.lax.scan(step, (row[0], 0.0), row)
    return jax.vmap(one)(x)


def switched(x, k):  # cond on a uniform value inside vmap
    return jax.vmap(lambda y: jax.lax.cond(k > 0, lambda y: jnp.sin(y) * y, lambda y: jnp.exp(-y * y), y))(x)


def quadrature(x, a, w):  # nested vmaps: per row and point a local Newton solve, row data, a weighted sum over points
    def point(xq, ar):  # the root of y^3 + ar[0] y = xq (per point), then a little more work
        c = lambda s: jnp.abs(s[0] ** 3 + ar[0] * s[0] - xq) > 1e-13 * (1 + jnp.abs(xq))
        y = jax.lax.while_loop(c, lambda s: (s[0] - (s[0] ** 3 + ar[0] * s[0] - xq) / (3 * s[0] ** 2 + ar[0]), s[1] + 1),
                               (xq, 0))[0]
        return jnp.sin(y) * ar[1] + y ** 2
    per_point = jax.vmap(jax.vmap(point, (0, None)), (0, 0))(x, a)  # (rows, points)
    return jnp.sum(per_point * w, axis=1)  # (rows,)


def make(n, rng):
    x = rng.uniform(0.5, 1.5, (n, 3))
    a = rng.standard_normal((n // 50, 3, 3)) + 3 * np.eye(3)
    w1, w2 = rng.standard_normal((3, 16)), rng.standard_normal((16, 2))
    return {"black_scholes": (black_scholes, (x,)), "bs_grad": (bs_grad, (x,)), "mlp_mean": (mlp_mean, (x, w1, w2)), "lj_energy": (lj_energy, (x,)),
            "centered_norms": (centered_norms, (x,)), "inv3": (inv3, (a,)), "cbrt_newton": (cbrt_newton, (x,)),
            "oscillator": (oscillator, (x,)), "ema": (ema, (rng.standard_normal((n // 4, 16)),)),
            "switched": (switched, (x, 1.0)),
            "quadrature": (quadrature, (rng.uniform(0.5, 2.0, (n // 27, 27)), rng.uniform(0.5, 1.5, (n // 27, 2)),
                                        rng.uniform(0.1, 1.0, 27)))}
