# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""The examples' timing: the first call (tracing and compilation included) and the best of three later calls, the
inputs on the device beforehand (as in a program that made them)."""
import time

import jax


def timed(jit, f, *args):
    f, args = jit(f), [jax.device_put(a) for a in args]
    t0 = time.perf_counter()
    jax.block_until_ready(f(*args))
    first = time.perf_counter() - t0
    later = []
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        later.append(time.perf_counter() - t0)
    print(f"{'jax' if jit is jax.jit else 'kernelax'}.jit: first call {first:.2f} s, later calls {min(later):.3f} s", flush=True)
