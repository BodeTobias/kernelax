# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""First call of kernelax.jit against jax.jit, each in a fresh process (JAX's backend initialized first): the first
call, the time until the kernels are in use, the calls after; kernelax.jit as by default (build_first: the first call
builds the kernels) and with build_first=False (it runs in XLA, the kernels built in a thread meanwhile).

    python benchmarks/first_call.py [N]            # all programs of benchmarks/programs.py, cold and warm cache
    python benchmarks/first_call.py N name mode    # one measurement (mode: jax, kernelax, background), as JSON"""
import json
import os
import subprocess
import sys
import tempfile
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


def measure(n, name, mode):
    from kernelax import jit
    from programs import make
    f, args = make(n, np.random.default_rng(0))[name]
    args = [jax.device_put(a) for a in args]
    jax.block_until_ready(jax.numpy.zeros(1) + 1)  # the backend, as any program would have it already
    g = jax.jit(f) if mode == "jax" else jit(f, build_first=mode == "kernelax")
    t0 = time.perf_counter()
    jax.block_until_ready(g(*args))
    first = time.perf_counter() - t0
    if mode != "jax":
        g.wait()  # (kernels in use; their variants follow, timed after them)
    ready = time.perf_counter() - t0
    if mode != "jax":
        g.wait(variants=True)
    jax.block_until_ready(g(*args))
    ts = []
    for _ in range(5):
        t1 = time.perf_counter()
        jax.block_until_ready(g(*args))
        ts.append(time.perf_counter() - t1)
    status = " ".join(str(v) for k, v in g.report()[0].items() if k in ("status", "why")) if mode != "jax" else ""
    return dict(first=first, ready=ready, call=min(ts), status=status)


def run(n, name, mode, env):
    out = subprocess.run([sys.executable, __file__, str(n), name, mode], env=env, capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(os.path.abspath(__file__)))
    return json.loads(out.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    if len(sys.argv) == 4:
        print(json.dumps(measure(int(sys.argv[1]), sys.argv[2], sys.argv[3])))
        sys.exit()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from programs import make
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    print(f"N = {n}; seconds (first call), ms (later calls); bf: build_first (the default), bg: build_first=False\n")
    print("| program | jax.jit first s | bf cold: first s | bg cold: first s | bg: kernels in use after s | warm: first s | "
          "jax.jit call ms | kernelax call ms |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    caches = [dict(os.environ, KERNELAX_CACHE_DIR=tempfile.mkdtemp()) for _ in range(2)]  # (a cache per mode: the
    for env in caches:  # programs' kernels cold in each; the runtime built before, as once per cache)
        subprocess.run([sys.executable, "-c", "from kernelax.build import runtime; runtime()"], env=env, check=True)
    for name in make(1000, np.random.default_rng(0)):
        j, bf, bg = run(n, name, "jax", caches[0]), run(n, name, "kernelax", caches[0]), run(n, name, "background", caches[1])
        warm = run(n, name, "kernelax", caches[0])
        print(f"| {name} | {j['first']:.2f} | {bf['first']:.2f} | {bg['first']:.2f} | {bg['ready']:.2f} | {warm['first']:.2f} | "
              f"{1e3 * j['call']:.2f} | {1e3 * warm['call']:.2f} |", flush=True)
