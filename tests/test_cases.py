# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

import os
import sys

import jax
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from cases import CASES, example  # noqa: E402

from kernelax.vmap import cvmap  # noqa: E402
from kernelax.build import simd_width  # noqa: E402


@pytest.mark.parametrize("W, K", [(4, 1), (8, 1), (8, 2)] if simd_width() == 8 else [(simd_width(), 1), (simd_width(), 2)])
@pytest.mark.parametrize("case", [pytest.param(c, id=c.name, marks=[pytest.mark.xfail(strict=True, reason=c.xfail)]
                                               if c.xfail else []) for c in CASES])
def test_matches_jax_vmap(case, W, K):
    args = case.make(np.random.default_rng(0), 37)  # 37: not a multiple of the SIMD width
    ref = jax.tree.leaves(jax.vmap(case.f, in_axes=case.in_axes)(*args))
    out = jax.tree.leaves(cvmap(case.f, example(case, args), case.in_axes, W=W, K=K)(*args))
    assert len(out) == len(ref)
    for a, b in zip(ref, out):
        a = np.asarray(a, dtype=np.float64)
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12 * (1 + np.abs(np.where(np.isfinite(a), a, 0)).max(initial=0)))


@pytest.mark.skipif(os.uname().machine != "x86_64", reason="x86-64 only")
def test_avx2_builds(tmp_path):
    """The AVX2 kernels (W = 4) built for a CPU without AVX-512 (gcc -march=x86-64-v3 through a KERNELAX_CC wrapper:
    on AVX-512 machines W = 4 had EVEX encodings and 32 registers, an AVX2 CPU was never covered): cases with masks,
    loops, tables, integers, random bits, sorts and float32 against jax.vmap."""
    import subprocess
    wrap = tmp_path / "gcc-v3"
    from kernelax.build import CC
    wrap.write_text(f'#!/bin/sh\nexec {CC} "$@" -march=x86-64-v3 -mtune=generic\n')
    wrap.chmod(0o755)
    script = tmp_path / "avx2.py"
    script.write_text(f"""import sys
sys.path.insert(0, {os.path.join(os.path.dirname(__file__), "..", "benchmarks")!r})
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from cases import CASES, example
from kernelax.vmap import cvmap
names = "black_scholes int_ops interp j2 kepler random sort take transcendental reductions switch".split()
for case in [c for c in CASES if c.name in names]:
    args = case.make(np.random.default_rng(0), 37)
    ref = jax.tree.leaves(jax.vmap(case.f, in_axes=case.in_axes)(*args))
    out = jax.tree.leaves(cvmap(case.f, example(case, args), case.in_axes, W=4, K=1)(*args))
    for a, b in zip(ref, out):
        a = np.asarray(a, np.float64)
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12 * (1 + np.abs(np.where(np.isfinite(a), a, 0)).max(initial=0)))
f = lambda x: jax.lax.while_loop(lambda c: (c[1] < 20) & (c[0] > 1e-3), lambda c: (jnp.where(c[0] > 0.5, c[0] * 0.5, c[0] * 0.9), c[1] + 1), (x, 0))
x = np.linspace(0.01, 3.0, 37, dtype=np.float32)
with jax.enable_x64(False):
    for a, b in zip(jax.tree.leaves(jax.vmap(f)(x)), jax.tree.leaves(cvmap(f, (x[0],), (0,), W=4, K=1)(x))):
        np.testing.assert_allclose(b, a, rtol=1e-6)
print("ok", len(names))
""")
    env = dict(os.environ, KERNELAX_CC=str(wrap), KERNELAX_CACHE_DIR=str(tmp_path / "cache"), JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0 and r.stdout.split() == ["ok", "11"], r.stdout + r.stderr[-3000:]
