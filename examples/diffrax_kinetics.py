# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Stiff chemical kinetics (Robertson's problem) for 16 384 sets of rate constants, each solved with diffrax's implicit
Kvaerno5 and its own adaptive steps under jax.vmap, as in a parameter study.

    pip install diffrax
    python examples/diffrax_kinetics.py         # kernelax.jit
    python examples/diffrax_kinetics.py jax     # jax.jit"""
import sys

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

import kernelax
from timing import timed

jax.config.update("jax_enable_x64", True)


def robertson(t, y, k):
    return jnp.stack([-k[0] * y[0] + k[2] * y[1] * y[2],
                      k[0] * y[0] - k[1] * y[1] ** 2 - k[2] * y[1] * y[2],
                      k[1] * y[1] ** 2])


def solve(k):  # the concentrations at t = 100 for the rate constants k
    sol = diffrax.diffeqsolve(diffrax.ODETerm(robertson), diffrax.Kvaerno5(), 0.0, 100.0, 1e-4, jnp.array([1.0, 0.0, 0.0]),
                              args=k, stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9), max_steps=4096)
    return sol.ys[-1]


k = np.array([0.04, 3e7, 1e4]) * np.random.default_rng(0).uniform(0.8, 1.2, (16_384, 3))
jit = jax.jit if "jax" in sys.argv[1:] else kernelax.jit  # the one change: kernelax.jit instead of jax.jit
timed(jit, jax.vmap(solve), k)
