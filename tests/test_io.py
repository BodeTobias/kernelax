# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kernelax.vmap import cvmap
from kernelax.build import simd_width

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("W", [4, 8] if simd_width() == 8 else [simd_width()])
@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("R", [1, 2, 3, 5, 6, 7, 8, 9, 17])
def test_row_lengths(R, K, W):
    """Rows of R entries in and out (contiguous and transposed paths), outputs of other lengths, a uniform
    argument, a pytree, a mask output, and n with a partial last block."""
    def f(d, c):
        x = d["x"]
        return {"y": x * c + jnp.arange(R), "s": jnp.sum(x) * c, "m": x > 0, "z": jnp.outer(x[:3], c * x[-2:])}
    n = 3 * K * W + 3
    x = np.random.default_rng(R).standard_normal((n, R))
    out = cvmap(f, [{"x": x[0]}, np.float64(1.5)], (0, None), W=W, K=K)({"x": x}, np.float64(1.5))
    ref = jax.vmap(f, (0, None))({"x": x}, 1.5)
    assert jax.tree.structure(out) == jax.tree.structure(ref)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(out)):
        np.testing.assert_allclose(b, np.asarray(a, np.float64), rtol=1e-14, atol=1e-14)


def test_chosen_K():
    """K by measurement at the first call with >= TUNE_MIN samples; results as with K = 1."""
    from kernelax.cpu import TUNE_MIN
    def f(x):  # a dependent chain: interleaving pays
        for _ in range(20):
            x = jnp.sin(x) * 0.9 + 0.1
        return x
    x = np.random.default_rng(0).standard_normal((TUNE_MIN + 5, 2))
    k = cvmap(f, [x[0]], (0,))
    small = k(x[:100])
    assert k.kernel.K() is None  # too few samples to choose
    out = k(x)
    assert k.kernel.K() in (1, 2, 3, 4)
    np.testing.assert_array_equal(out, cvmap(f, [x[0]], (0,), K=1)(x))
    np.testing.assert_array_equal(small, out[:100])
