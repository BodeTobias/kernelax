# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Rosenbrock's function minimised by optimistix's BFGS for two million parameter sets, each with its own iterations
under jax.vmap, as in a fit per sample.

    pip install optimistix
    python examples/optimistix_bfgs.py          # kernelax.jit
    python examples/optimistix_bfgs.py jax      # jax.jit"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

import kernelax
from timing import timed

jax.config.update("jax_enable_x64", True)


def rosenbrock(y, p):
    return (p[0] - y[0]) ** 2 + p[1] * (y[1] - y[0] ** 2) ** 2


def solve(p):  # the minimiser of rosenbrock(., p)
    return optx.minimise(rosenbrock, optx.BFGS(rtol=1e-10, atol=1e-10), jnp.array([-1.0, 1.0]), args=p, max_steps=512).value


r = np.random.default_rng(0)
p = np.stack([r.uniform(0.5, 1.5, 2_000_000), r.uniform(5, 50, 2_000_000)], 1)
jit = jax.jit if "jax" in sys.argv[1:] else kernelax.jit  # the one change: kernelax.jit instead of jax.jit
timed(jit, jax.vmap(solve), p)
