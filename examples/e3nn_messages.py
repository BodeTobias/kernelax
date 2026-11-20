# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Messages of an equivariant graph network (as in NequIP) with e3nn: spherical harmonics of each edge vector, their
tensor product with the neighbour's features and a radial weight, for four million edges in float32.

    pip install e3nn-jax
    python examples/e3nn_messages.py            # kernelax.jit
    python examples/e3nn_messages.py jax        # jax.jit"""
import sys

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np

import kernelax
from timing import timed

irreps = e3nn.Irreps("8x0e + 8x1o")


def message(vec, feat):
    sh = e3nn.spherical_harmonics("0e + 1o + 2e", vec, normalize=True)
    tp = e3nn.tensor_product(sh, e3nn.IrrepsArray(irreps, feat), filter_ir_out="0e + 1o + 2e")
    return tp.array * jnp.exp(-jnp.sum(vec ** 2, -1, keepdims=True))


r = np.random.default_rng(0)
n = 4_000_000
vec, feat = r.standard_normal((n, 3), np.float32), r.standard_normal((n, irreps.dim), np.float32)
jit = jax.jit if "jax" in sys.argv[1:] else kernelax.jit  # the one change: kernelax.jit instead of jax.jit
timed(jit, message, vec, feat)
