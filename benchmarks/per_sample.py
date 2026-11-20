# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""jax.jit(jax.vmap(f)) against jax.jit(jax.vmap(kernelax.kernel(f))) on JAX's default device (CPU or GPU), through
the public API, for per-sample functions of benchmarks/cases.py: time per call, best of repeated
calls after the first.

    python benchmarks/per_sample.py [N] [case ...]     # N samples (default 2^20); KERNELAX_X64=0: float32

On a GPU, JAX's eig takes ~0.26 ms per matrix (L40S, JAX 0.12): eig4 measured with N = 2^14 there."""
import sys
import time

import jax
import numpy as np

import kernelax
from cases import CASES

DEFAULT = "black_scholes neo_hooke j2 kepler log_strain eig4 rk4".split()


def best(f, args, budget=1.0):
    jax.block_until_ready(f(*args))
    times = []
    while len(times) < 3 or sum(times) < budget:
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - t0)
    return min(times)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 20
    names = sys.argv[2:] or DEFAULT
    print(f"{jax.devices()[0].device_kind}, N = {n}, {'float64' if jax.config.jax_enable_x64 else 'float32'}\n\n"
          "| case | jax.jit ms | kernelax ms | speedup |\n|---|---:|---:|---:|")
    for case in [c for c in CASES if c.name in names]:
        args = [jax.device_put(a) for a in case.make(np.random.default_rng(0), n)]
        t_x = best(jax.jit(jax.vmap(case.f, in_axes=case.in_axes)), args)
        t_k = best(jax.jit(jax.vmap(kernelax.kernel(case.f), in_axes=case.in_axes)), args)
        print(f"| {case.name} | {1e3 * t_x:.2f} | {1e3 * t_k:.2f} | {t_x / t_k:.0f} |", flush=True)
