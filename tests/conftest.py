# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""The test session's setup: a fresh kernel index (the libraries stay cached by their sources), so that choices noted
by earlier runs (K, kernels or XLA, with other thread counts) do not decide the tests; no tests on a CPU kernelax does
not run on."""

import shutil
import tempfile

import pytest

from kernelax.build import simd_width

try:
    simd_width()
    collect_ignore_glob = []
except RuntimeError as e:  # (no AVX2/FMA, AVX-512 or NEON: kernelax' kernels do not run here)
    collect_ignore_glob = ["test_*.py"]
    print(f"kernelax tests skipped: {e}")


@pytest.fixture(scope="session", autouse=True)
def _fresh_index():
    from kernelax import cpu
    index, cpu._INDEX = cpu._INDEX, tempfile.mkdtemp(prefix="kernelax-index-")
    try:
        yield
    finally:
        shutil.rmtree(cpu._INDEX, ignore_errors=True)
        cpu._INDEX = index
