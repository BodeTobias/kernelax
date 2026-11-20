# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""kernelax: per-sample JAX functions as C/SIMD kernels over a batch (CPU) or CUDA kernels (NVIDIA GPUs)."""
import sys

__version__ = "0.1.0"

if not sys.platform.startswith("linux"):  # (gcc-built shared libraries, Linux' CPU affinity and file locks)
    raise ImportError(f"kernelax needs Linux (x86-64 with AVX2 or AVX-512, or aarch64), not {sys.platform}")

from kernelax.auto import jit  # noqa: E402
from kernelax.frontend import tracing  # noqa: E402
from kernelax.primitive import kernel  # noqa: E402
from kernelax.standalone import export  # noqa: E402

__all__ = ["export", "jit", "kernel", "tracing"]
