# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""kernelax.jit against jax.jit on whole programs (benchmarks/programs.py): time per call (device arrays, best of
repeated runs, after the kernels are built) and the partitioning (benchmarks/first_call.py: the first call)."""
import sys
import time

import jax
import numpy as np

from kernelax import jit
from programs import make

jax.config.update("jax_enable_x64", True)


def best(f, args, budget=0.2):
    jax.block_until_ready(f(*args))
    times = []
    while len(times) < 3 or sum(times) < budget:
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - t0)
    return min(times)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    print(f"N = {n}\n\n| program | jax.jit ms | kernelax.jit ms | speedup | status | regions (eqns: chosen) |\n|---|---:|---:|---:|---|---|")
    for name, (f, args) in make(n, np.random.default_rng(0)).items():
        args = [jax.device_put(a) for a in args]
        g = jit(f)
        jax.block_until_ready(g(*args))
        g.wait(variants=True)  # the kernels built in the background (with their variants: timed after them)
        t_c, t_x = best(g, args), best(jax.jit(f), args)
        plan = g.report()[0]
        regions = ", ".join(f"{r['eqns']}: {r['chosen']}" for r in plan["regions"])
        print(f"| {name} | {1e3 * t_x:.2f} | {1e3 * t_c:.2f} | {t_x / t_c:.1f} | {plan['status']} {plan.get('why', '')} | {regions} |")
