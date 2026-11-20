# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kernelax.vmap import cvmap
from kernelax.build import simd_width

jax.config.update("jax_enable_x64", True)


def draws(s):
    """Random numbers per sample from its seed: keys, fold_in, split, threefry, uint32/uint64 bits."""
    k = jax.random.fold_in(jax.random.key(42), s.astype(jnp.int64))
    k1, k2 = jax.random.split(k)
    return (jax.random.uniform(k1, (3,)), jax.random.bits(k2, (2,), jnp.uint32), jax.random.randint(k1, (2,), 0, 10),
            jax.random.uniform(jax.random.PRNGKey(s.astype(jnp.int64)), (2,), dtype=jnp.float32),
            jax.random.bernoulli(k1, 0.3, (3,)), jax.random.exponential(k2, (2,)))


@pytest.mark.parametrize("W", [4, 8] if simd_width() == 8 else [simd_width()])
def test_random(W):
    """Bitwise equal to JAX, except exponential (log: libmvec, within a few ulp)."""
    s = np.arange(1001, dtype=np.float64)
    out, ref = jax.tree.leaves(cvmap(draws, [s[0]], (0,), W=W)(s)), jax.tree.leaves(jax.vmap(draws)(s))
    for o, r in zip(out[:-1], ref[:-1]):
        np.testing.assert_array_equal(o, np.asarray(r).astype(np.float64))
    np.testing.assert_allclose(out[-1], ref[-1], rtol=1e-15)


def test_normal():
    """jax.random.normal (erf_inv as XLA computes it): within an ulp of JAX (log1p: libmvec)."""
    f = lambda s: jax.random.normal(jax.random.fold_in(jax.random.key(0), s.astype(jnp.int64)), (3,))
    s = np.arange(1001, dtype=np.float64)
    np.testing.assert_allclose(cvmap(f, [s[0]], (0,))(s), jax.vmap(f)(s), rtol=1e-15)


def test_random_since_jax_0121():
    """jax.random as JAX 0.12.1 writes it: bits in several streams (random_bits with num_streams: one stream was
    computed, the others missing or wrong), normal(mode="highest"), gamma of bfloat16 drawn in bfloat16 (it was drawn in
    float32: other samples than XLA's), clone (a KeyError), the derivative of gamma by its shape (random_gamma_grad:
    an AttributeError)."""
    s = np.arange(64.0)[:, None]
    key = lambda v: jax.random.fold_in(jax.random.key(7), v[0].astype(jnp.int64))
    cases = [lambda v: jax.random.bits(key(v), (3,), num_streams=2),
             lambda v: jax.random.normal(key(v), (2,), mode="highest"),
             lambda v: jax.random.gamma(key(v), jnp.asarray([1.5, 3.0], jnp.bfloat16), dtype=jnp.bfloat16).astype(jnp.float32),
             lambda v: jax.random.uniform(jax.random.clone(key(v)), (2,)),
             lambda v: jax.grad(lambda a: jax.random.gamma(key(v), a))(v[0] * 0.1 + 1.0)]
    for f in cases:
        for a, b in zip(jax.tree.leaves(cvmap(f, (s[0],), (0,))(s)), jax.tree.leaves(jax.vmap(f)(s))):
            np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=1e-12)
