# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""The README's table: every example in fresh processes, with jax.jit and with kernelax.jit alternately, R rounds;
kernelax.jit with an empty kernel cache (its first call builds the kernels, as on a first run). Medians in seconds.

    python examples/compare.py [R]"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

EXAMPLES = ["diffrax_kinetics", "optimistix_bfgs", "e3nn_messages", "jaxley_sweep"]
here = os.path.dirname(os.path.abspath(__file__))
R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
print("| example | first call: jax.jit | kernelax.jit | later calls: jax.jit | kernelax.jit |\n|---|---:|---:|---:|---:|")
for ex in EXAMPLES:
    t = {"jax": [], "kernelax": []}
    for _ in range(R):
        for mode in t:
            with tempfile.TemporaryDirectory() as cache:
                out = subprocess.run([sys.executable, os.path.join(here, ex + ".py"), mode], capture_output=True, text=True,
                                     env=dict(os.environ, KERNELAX_CACHE_DIR=cache), check=True).stdout
            m = re.search(r"first call ([\d.]+) s, later calls ([\d.]+) s", out)
            t[mode].append((float(m[1]), float(m[2])))
    med = {k: np.median(v, 0) for k, v in t.items()}
    print(f"| {ex} | {med['jax'][0]:.2f} | {med['kernelax'][0]:.2f} | {med['jax'][1]:.3f} | {med['kernelax'][1]:.3f} |", flush=True)
