# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""kernelax.export: the libraries it writes, built with their Makefile and called through ctypes (and from Fortran and
Java), against jax.vmap."""

import ctypes
import os
import re
import shutil
import subprocess
import sys
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from cases import CASES, example  # noqa: E402

import kernelax  # noqa: E402
from kernelax.build import simd_width  # noqa: E402

BY_NAME = {c.name: c for c in CASES}
CT = {"double": np.float64, "float": np.float32, "int32_t": np.int32, "uint32_t": np.uint32, "int64_t": np.int64,
      "uint64_t": np.uint64}
X86 = os.uname().machine == "x86_64"
VARIANTS = [i for i, w in (("avx2", 4), ("avx512", 8)) if X86 and simd_width() >= w]  # (those this CPU runs)


def build(d, defs="", *make):
    subprocess.run(["make", "-s", "-C", str(d), "-j4", *make] + ([f"DEFS={defs}"] if defs else []), check=True, capture_output=True)
    name = next(f[3:-3] for f in os.listdir(d) if f.startswith("lib") and f.endswith(".so"))
    lib = ctypes.CDLL(os.path.join(d, f"lib{name}.so"))
    if hasattr(lib, f"{name}_isa"):
        getattr(lib, f"{name}_isa").restype = ctypes.c_char_p
    return lib


def to_c(x, a):
    """An argument as the library takes it (NAME.h): complex as [re, im], 64-bit integers by their bits."""
    x = np.asarray(x)
    if a["complex"]:
        x = np.stack([x.real, x.imag], -1)
    t = CT[a["c_type"]]
    return np.ascontiguousarray(x.view(t) if np.dtype(t).itemsize == 8 and t != np.float64 else x.astype(t))


def outputs(desc, n):
    return [np.zeros((n,) + tuple(a["shape"]), CT[a["c_type"]]) for a in desc["outputs"]]


def from_c(o, a):
    o = o[..., 0] + 1j * o[..., 1] if a["complex"] else o
    return o.view(np.dtype(a["dtype"])) if a["c_type"] in ("int64_t", "uint64_t") else o


def close(res, ref, desc, rtol=1e-10):
    for o, r, a in zip(res, ref, desc["outputs"]):
        r = np.asarray(r)
        o = from_c(o, a)
        if np.dtype(a["dtype"]).kind in "iub":
            np.testing.assert_array_equal(o.astype(r.dtype), r, err_msg=a["name"])
        else:
            fin = np.abs(r[np.isfinite(r)])  # (NaN and inf compared as such, the tolerance from the finite values)
            np.testing.assert_allclose(o, r, rtol=rtol, atol=rtol * (1 + (fin.max() if fin.size else 0)), err_msg=a["name"])


P = lambda x: x.ctypes.data_as(ctypes.c_void_p)


def batch(lib, fn, xs, desc, n, nt=2):
    res = outputs(desc, n)
    assert getattr(lib, fn)(ctypes.c_int(nt), ctypes.c_int64(n), *map(P, xs), *map(P, res)) == 0
    return res


def point(lib, name, xs, desc, n):
    res = outputs(desc, n)
    for i in range(n):
        getattr(lib, f"{name}_point")(*[P(x[i:i + 1] if a["batched"] else x) for x, a in zip(xs, desc["inputs"])],
                                      *[P(o[i:i + 1]) for o in res])
    return res


@pytest.mark.parametrize("name", ["neo_hooke", "j2", "kepler", "newton_linesearch", "mlp", "int_ops", "interp", "sort",
                                  "complex_ode", "black_scholes"])
def test_cases(name, tmp_path):
    """Batch (the dispatcher, each variant this CPU runs, and the scalar path) and point function against jax.vmap:
    while loops, arguments shared by all samples, integers, tables, sorting, complex values."""
    case = BY_NAME[name]
    args = case.make(np.random.default_rng(0), 37)
    desc = kernelax.export(case.f, example(case, args), case.in_axes, name=name, out_dir=tmp_path / "lib")
    ref = jax.tree.leaves(jax.vmap(case.f, case.in_axes)(*args))
    xs = [to_c(x, a) for x, a in zip(jax.tree.leaves(args), desc["inputs"])]
    lib = build(tmp_path / "lib")
    assert getattr(lib, f"{name}_isa")().decode() == (VARIANTS[-1] if VARIANTS else "scalar")
    close(batch(lib, f"{name}_batch", xs, desc, 37), ref, desc)
    for v in VARIANTS:
        close(batch(lib, f"{name}_batch_{v}", xs, desc, 37), ref, desc)
    close(point(lib, name, xs, desc, 37), ref, desc)
    shutil.copytree(tmp_path / "lib", tmp_path / "scalar")
    os.remove(tmp_path / "scalar" / f"{name}_batch.o")
    lib = build(tmp_path / "scalar", "-DKERNELAX_SCALAR")
    assert getattr(lib, f"{name}_isa")().decode() == "scalar"
    close(batch(lib, f"{name}_batch", xs, desc, 37), ref, desc)


def test_float32_and_int64(tmp_path):
    """Without x64, float32 and int32 throughout (4-byte elements: float*, int32_t*); with x64, 64-bit integers in and
    out by their bits (exact beyond 2^53), next to doubles."""
    f = lambda x, p: (jnp.sin(x) * p[0], jnp.argmax(x).astype(jnp.int32), (x * 3).astype(jnp.int32))
    x, p = np.random.default_rng(1).uniform(-2, 2, (21, 5)).astype(np.float32), np.array([1.5], np.float32)
    with jax.enable_x64(False):
        desc = kernelax.export(f, (x[0], p), (0, None), name="f32", out_dir=tmp_path / "a")
        ref = jax.tree.leaves(jax.vmap(f, (0, None))(x, p))
    assert desc["element_bytes"] == 4 and [a["c_type"] for a in desc["outputs"]] == ["float", "int32_t", "int32_t"]
    lib, xs = build(tmp_path / "a"), [x, p]
    close(batch(lib, "f32_batch", xs, desc, 21), ref, desc, rtol=2e-6)
    close(point(lib, "f32", xs, desc, 21), ref, desc, rtol=2e-6)
    g = lambda k, y: (k * 3 + 1, jnp.sum(y) + k[0].astype(jnp.float64))
    k, y = np.random.default_rng(2).integers(-2 ** 60, 2 ** 60, (13, 2)), np.random.default_rng(3).standard_normal((13, 3))
    desc = kernelax.export(g, (k[0], y[0]), name="i64", out_dir=tmp_path / "b")
    assert [a["c_type"] for a in desc["inputs"]] == ["int64_t", "double"] and desc["outputs"][0]["c_type"] == "int64_t"
    lib, xs = build(tmp_path / "b"), [to_c(k, desc["inputs"][0]), y]
    ref = jax.tree.leaves(jax.vmap(g)(k, y))
    close(batch(lib, "i64_batch", xs, desc, 13), ref, desc)
    close(point(lib, "i64", xs, desc, 13), ref, desc)


def test_two_laws_in_one_library(tmp_path):
    """Two exported functions linked into one library: no symbol in common (the kernels, tables and run-time values
    are static; every visible name carries its prefix)."""
    objs, refs = [], {}
    for name in ("neo_hooke", "kepler"):
        case = BY_NAME[name]
        args = case.make(np.random.default_rng(4), 19)
        desc = kernelax.export(case.f, example(case, args), case.in_axes, name=name, out_dir=tmp_path / name)
        subprocess.run(["make", "-s", "-C", str(tmp_path / name), "-j4"], check=True, capture_output=True)
        objs += [str(p) for p in (tmp_path / name).glob("*.o")]
        refs[name] = (desc, [to_c(x, a) for x, a in zip(jax.tree.leaves(args), desc["inputs"])],
                      jax.tree.leaves(jax.vmap(case.f, case.in_axes)(*args)))
    subprocess.run(["g++", "-shared", "-fopenmp", "-o", str(tmp_path / "both.so"), *objs, "-lmvec", "-lm"], check=True)
    lib = ctypes.CDLL(str(tmp_path / "both.so"))
    for name, (desc, xs, ref) in refs.items():
        close(batch(lib, f"{name}_batch", xs, desc, 19), ref, desc)
        close(point(lib, name, xs, desc, 19), ref, desc)


def test_concurrent_calls(tmp_path):
    """Calls from several host threads at once (each with nt = 1, as a host that parallelizes itself): the same results
    as one after the other."""
    case = BY_NAME["j2"]
    args = case.make(np.random.default_rng(5), 64)
    desc = kernelax.export(case.f, example(case, args), case.in_axes, name="j2t", out_dir=tmp_path)
    lib, xs = build(tmp_path), [to_c(x, a) for x, a in zip(jax.tree.leaves(args), desc["inputs"])]
    want = (batch(lib, "j2t_batch", xs, desc, 64, nt=1), point(lib, "j2t", xs, desc, 64))
    got, errors = [], []

    def run():
        try:
            for _ in range(20):
                got.append((batch(lib, "j2t_batch", xs, desc, 64, nt=1), point(lib, "j2t", xs, desc, 64)))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(got) == 160
    for b, p in got:
        for x, y in zip(b + p, want[0] + want[1]):
            np.testing.assert_array_equal(x, y)


def test_modes_and_errors(tmp_path):
    """mode="point" alone (no batch kernel, no libmvec); mode="batch" alone (the scalar path then returns 1); NEON
    sources (not run here); names and dtypes refused; kernelax' version in the description and the header."""
    case = BY_NAME["black_scholes"]
    args = case.make(np.random.default_rng(6), 11)
    ref = jax.tree.leaves(jax.vmap(case.f, case.in_axes)(*args))
    desc = kernelax.export(case.f, example(case, args), case.in_axes, name="bs", out_dir=tmp_path / "p", mode="point")
    assert sorted(os.listdir(tmp_path / "p")) == ["Makefile", "bs.cpp", "bs.f90", "bs.h", "bs.java", "bs.json"]
    header = (tmp_path / "p" / "bs.h").read_text()
    assert desc["kernelax"] == kernelax.__version__ and f"generated by kernelax {kernelax.__version__}" in header
    xs = [to_c(x, a) for x, a in zip(jax.tree.leaves(args), desc["inputs"])]
    close(point(build(tmp_path / "p"), "bs", xs, desc, 11), ref, desc)
    kernelax.export(case.f, example(case, args), case.in_axes, name="bs", out_dir=tmp_path / "b", mode="batch")
    lib = build(tmp_path / "b", "-DKERNELAX_SCALAR")
    assert lib.bs_batch(ctypes.c_int(1), ctypes.c_int64(11), *map(P, xs), *map(P, outputs(desc, 11))) == 1
    kernelax.export(case.f, example(case, args), case.in_axes, name="bs", out_dir=tmp_path / "n", mode="batch", isa="neon")
    src = (tmp_path / "n" / "bs_batch_neon.c").read_text()
    assert "arm_neon.h" in src and "bs_batch_neon(" in src and not (tmp_path / "n" / "bs_batch_avx2.c").exists()
    for bad in ("1x", "int", "a-b", "x" * 41):
        with pytest.raises(ValueError, match="name"):
            kernelax.export(case.f, example(case, args), case.in_axes, name=bad, out_dir=tmp_path / "e")
    with pytest.raises(NotImplementedError, match="16-bit"):
        kernelax.export(lambda x: x * 2, (np.zeros(3, np.float16),), name="h", out_dir=tmp_path / "e")


def _inputs(tmp_path, desc, xs):  # (the inputs as binary files, for the Fortran and Java programs)
    for k, x in enumerate(xs):
        x.tofile(tmp_path / f"in{k}.bin")


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="no gfortran")
def test_fortran(tmp_path):
    """A Fortran program through NAME.f90's interfaces (column-major: one sample's row is a column): the batch function
    on all samples, the point function per sample (an array element as the start of its column)."""
    case = BY_NAME["neo_hooke"]
    n = 17
    args = case.make(np.random.default_rng(7), n)
    desc = kernelax.export(case.f, example(case, args), case.in_axes, name="nh", out_dir=tmp_path)
    assert all(a["c_type"] == "double" for a in desc["inputs"] + desc["outputs"])
    xs = [to_c(x, a) for x, a in zip(jax.tree.leaves(args), desc["inputs"])]
    _inputs(tmp_path, desc, xs)
    size = lambda a: int(np.prod(a["shape"], dtype=np.int64))
    arrays = [(f"in{k}", a, size(a) * (n if a["batched"] else 1)) for k, a in enumerate(desc["inputs"])] + \
             [(f"o{j}", a, size(a) * n) for j, a in enumerate(desc["outputs"])]
    decl = "\n".join(f"  real(c_double) :: {v}({m})" for v, _, m in arrays)
    reads = "\n".join(f"  open(10, file='in{k}.bin', access='stream', form='unformatted'); read(10) in{k}; close(10)"
                      for k in range(len(desc["inputs"])))
    names = ", ".join(v for v, _, _ in arrays)
    at = lambda v, a: f"{v}(i * {size(a)} + 1)" if a["batched"] else v
    pts = ", ".join(at(v, a) for v, a, _ in arrays)
    writes = lambda tag: "\n".join(f"  open(11, file='{tag}{j}.bin', access='stream', form='unformatted', status='replace');"
                                   f" write(11) o{j}; close(11)" for j in range(len(desc["outputs"])))
    (tmp_path / "main.f90").write_text(f"""program main
  use nh_mod
  use, intrinsic :: iso_c_binding
  implicit none
  integer :: i, rc
{decl}
{reads}
  rc = nh_batch(1_c_int, {n}_c_int64_t, {names})
  if (rc /= 0) stop 1
{writes('batch')}
  do i = 0, {n - 1}
    call nh_point({pts})
  end do
{writes('point')}
end program main
""")
    build(tmp_path)
    subprocess.run(["gfortran", "-o", "main", "nh.f90", "main.f90", "-L.", "-lnh", "-Wl,-rpath,."], cwd=tmp_path, check=True)
    subprocess.run(["./main"], cwd=tmp_path, check=True)
    ref = jax.tree.leaves(jax.vmap(case.f, case.in_axes)(*args))
    for tag in ("batch", "point"):
        res = [np.fromfile(tmp_path / f"{tag}{j}.bin").reshape((n,) + tuple(a["shape"])) for j, a in enumerate(desc["outputs"])]
        close(res, ref, desc)


def _jdk():
    if shutil.which("javac") is None:
        return 0
    m = re.search(r"(\d+)", subprocess.run(["javac", "-version"], capture_output=True, text=True).stdout)
    return int(m.group(1)) if m else 0


@pytest.mark.skipif(_jdk() < 22, reason="no JDK >= 22 (Foreign Function & Memory API)")
def test_java(tmp_path):
    """A Java program through NAME.java (FFM): batch on Java arrays (no copy) and on memory outside the heap, point per
    sample; an array too short for n samples refused (IllegalArgumentException)."""
    case = BY_NAME["j2"]
    n = 23
    args = case.make(np.random.default_rng(8), n)
    desc = kernelax.export(case.f, example(case, args), case.in_axes, name="j2j", out_dir=tmp_path)
    assert all(a["c_type"] == "double" for a in desc["inputs"] + desc["outputs"])
    xs = [to_c(x, a) for x, a in zip(jax.tree.leaves(args), desc["inputs"])]
    _inputs(tmp_path, desc, xs)
    size = lambda a: int(np.prod(a["shape"], dtype=np.int64))
    ins, outs = desc["inputs"], desc["outputs"]
    read = "\n".join(f"    double[] in{k} = read(\"in{k}.bin\");" for k in range(len(ins)))
    alloc = "\n".join(f"    double[] o{j} = new double[{size(a) * n}], q{j} = new double[{size(a) * n}];" for j, a in enumerate(outs))
    sl = lambda v, a: f"Arrays.copyOfRange({v}, i * {size(a)}, (i + 1) * {size(a)})" if a["batched"] else v
    tmp = "\n".join(f"      double[] t{j} = new double[{size(a)}];" for j, a in enumerate(outs))
    back = "\n".join(f"      System.arraycopy(t{j}, 0, q{j}, i * {size(a)}, {size(a)});" for j, a in enumerate(outs))
    off = "\n".join(f"    MemorySegment s{k} = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, in{k});" for k in range(len(ins))) + "\n" + \
        "\n".join(f"    MemorySegment r{j} = arena.allocate(ValueLayout.JAVA_DOUBLE, {size(a) * n});" for j, a in enumerate(outs))
    (tmp_path / "Main.java").write_text(f"""import java.lang.foreign.*;
import java.nio.*;
import java.nio.file.*;
import java.util.Arrays;

public class Main {{
  static double[] read(String f) throws Exception {{
    ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(Path.of(f))).order(ByteOrder.LITTLE_ENDIAN);
    double[] a = new double[b.remaining() / 8];
    b.asDoubleBuffer().get(a);
    return a;
  }}
  static void write(String f, double[] a) throws Exception {{
    ByteBuffer b = ByteBuffer.allocate(8 * a.length).order(ByteOrder.LITTLE_ENDIAN);
    b.asDoubleBuffer().put(a);
    Files.write(Path.of(f), b.array());
  }}
  public static void main(String[] args) throws Exception {{
{read}
{alloc}
    if (j2j.batch(2, {n}, {', '.join([f'in{k}' for k in range(len(ins))] + [f'o{j}' for j in range(len(outs))])}) != 0) System.exit(1);
    for (int i = 0; i < {n}; ++i) {{
{tmp}
      j2j.point({', '.join([sl(f'in{k}', a) for k, a in enumerate(ins)] + [f't{j}' for j in range(len(outs))])});
{back}
    }}
    try (Arena arena = Arena.ofConfined()) {{
{off}
      if (j2j.batch(1, {n}, {', '.join([f's{k}' for k in range(len(ins))] + [f'r{j}' for j in range(len(outs))])}) != 0) System.exit(1);
{chr(10).join(f'      write("seg{j}.bin", r{j}.toArray(ValueLayout.JAVA_DOUBLE));' for j in range(len(outs)))}
    }}
{chr(10).join(f'    write("batch{j}.bin", o{j}); write("point{j}.bin", q{j});' for j in range(len(outs)))}
    try {{  // (an array shorter than n samples: refused, not written past)
      j2j.batch(1, {n}, {', '.join([f'in{k}' for k in range(len(ins))] + ['new double[1]'] + [f'o{j}' for j in range(1, len(outs))])});
      System.exit(2);
    }} catch (IllegalArgumentException e) {{ System.out.println("refused"); }}
    System.out.println(j2j.isa());
  }}
}}
""")
    build(tmp_path)
    subprocess.run(["javac", "-d", ".", "j2j.java", "Main.java"], cwd=tmp_path, check=True)
    run = subprocess.run(["java", "--enable-native-access=ALL-UNNAMED", f"-Dj2j.lib={tmp_path / 'libj2j.so'}", "-cp", ".", "Main"],
                         cwd=tmp_path, check=True, capture_output=True, text=True)
    assert run.stdout.split() == ["refused", VARIANTS[-1] if VARIANTS else "scalar"]
    ref = jax.tree.leaves(jax.vmap(case.f, case.in_axes)(*args))
    for tag in ("batch", "point", "seg"):
        res = [np.fromfile(tmp_path / f"{tag}{j}.bin").reshape((n,) + tuple(a["shape"])) for j, a in enumerate(outs)]
        close(res, ref, desc)


def test_uint32_beyond_int32(tmp_path):
    """uint32 values from 2^31 on in a 4-byte program (the point function and the scalar fallback: the GPU emitter's
    code, whose integers are int32 otherwise): converted, compared and computed with exactly."""
    f = lambda x: (x.astype(jnp.float32), (x > jnp.uint32(3_000_000_000)).astype(jnp.float32), x // jnp.uint32(3),
                   jnp.maximum(x, jnp.uint32(2 ** 31 + 7)))
    x = np.array([4_000_000_000, 5, 2 ** 31, 2 ** 32 - 1, 3_000_000_001, 0, 2 ** 31 - 1], np.uint32)
    desc = kernelax.export(f, (x[0],), name="u32", out_dir=tmp_path)
    assert desc["element_bytes"] == 4
    lib, ref = build(tmp_path), jax.tree.leaves(jax.vmap(f)(x))
    close(point(lib, "u32", [x], desc, len(x)), ref, desc)
    close(batch(lib, "u32_batch", [x], desc, len(x)), ref, desc)


def _point_cases(x, i, s, a):  # (the point code's and the GPU's conversions, signed zeros and loops)
    R = jax.random
    return (x.astype(jnp.int32), x.astype(jnp.uint32), x.astype(jnp.float32).astype(jnp.int32),
            x.astype(jnp.float32).astype(jnp.uint32), i.astype(jnp.uint32).astype(x.dtype),
            jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32) // 7, jax.lax.rem(x, jnp.asarray(3.0, x.dtype)), jnp.fmod(x, 0.25),
            R.gamma(R.key(s), a), R.poisson(R.key(s), 3.0))


POINT_X = np.array([3e9, -3e9, 4.5e9, np.inf, -np.inf, np.nan, -0.0, -6.0, -7.5, 1.5, 2147483520.0, -0.1])
POINT_I = np.array([-1, 5, -2 ** 31, 2 ** 31 - 1, 7, 0, -7, 1, 2, 3, 4, -100], np.int32)


@pytest.mark.parametrize("x64", [False, True])
def test_point_code_conversions_zeros_and_loops(tmp_path, x64):
    """The point function (the GPU emitter's code) as XLA: floats to int32 / uint32 saturated (it converted before it
    clamped: INT_MIN), uint32 from conversions beyond int32 (int32 -1 as uint32, a float32's bits), remainders keeping
    the dividend's sign at zero, and two while loops side by side (they shared their carries' types: a key's bits next
    to a float32 were rounded)."""
    n = len(POINT_X)
    s, a = np.arange(n, dtype=np.uint32) * 7 + 3, np.random.default_rng(0).uniform(0.2, 5, n)
    with jax.enable_x64(x64):
        dt = np.float64 if x64 else np.float32
        args = (POINT_X.astype(dt), POINT_I, s, a.astype(dt))
        desc = kernelax.export(_point_cases, tuple(v[0] for v in args), name="pc", out_dir=tmp_path)
        ref = jax.tree.leaves(jax.vmap(_point_cases)(*args))
    res = point(build(tmp_path), "pc", [to_c(v, d) for v, d in zip(args, desc["inputs"])], desc, n)
    close(res, ref, desc, rtol=1e-5)
    for o, r, d in zip(res, ref, desc["outputs"]):  # (zeros with XLA's sign)
        if np.dtype(d["dtype"]).kind == "f":
            o, r = from_c(o, d), np.asarray(r)
            np.testing.assert_array_equal(np.signbit(o)[~np.isnan(r)], np.signbit(r)[~np.isnan(r)], err_msg=d["name"])


def test_argument_names_and_bad_sizes(tmp_path):
    """Arguments named like the kernel's macros (K, W, L) and the generated code's locals (kx_p) build and compute;
    n < 0, nt < 1 or nt beyond the processors: return code 2, nothing written (n = -1 crashed)."""
    f = lambda K, W, L, kx_p: (K @ W + L * kx_p[0], jnp.sum(K))
    r, n = np.random.default_rng(10), 17
    args = (r.standard_normal((n, 3, 3)), r.standard_normal((n, 3)), r.standard_normal((n, 3)), np.array([1.5]))
    desc = kernelax.export(f, (args[0][0], args[1][0], args[2][0], args[3]), (0, 0, 0, None), name="kwl", out_dir=tmp_path)
    assert [a["name"] for a in desc["inputs"]][:3] == ["K", "W", "L"]
    lib, xs, ref = build(tmp_path), list(args), jax.tree.leaves(jax.vmap(f, (0, 0, 0, None))(*args))
    close(batch(lib, "kwl_batch", xs, desc, n), ref, desc)
    close(point(lib, "kwl", xs, desc, n), ref, desc)
    res = outputs(desc, n)
    for fn in ["kwl_batch"] + [f"kwl_batch_{i}" for i in VARIANTS]:
        for nt, m in ((1, -1), (0, n), (-3, n), (10 ** 6, n)):  # (beyond the processors: libgomp would end the process)
            assert getattr(lib, fn)(ctypes.c_int(nt), ctypes.c_int64(m), *map(P, xs), *map(P, res)) == 2, (fn, nt, m)
    assert all(not o.any() for o in res)


def test_options_and_argument_errors(tmp_path):
    """One variant (isa="avx2"), keep_nans, an example of jax.ShapeDtypeStruct; a mode, an instruction set, in_axes or
    a reserved name that does not exist, or no batched argument: ValueError."""
    f = lambda x, c: (x * 0.0 + jnp.sin(x) * c[0], jnp.sum(x))  # (x * 0: NaN kept with keep_nans)
    x, c = np.array([[1.0, np.nan, np.inf], [0.5, 2.0, -1.0]]), np.array([3.0])
    ex = (jax.ShapeDtypeStruct((3,), np.float64), jax.ShapeDtypeStruct((1,), np.float64))
    desc = kernelax.export(f, ex, (0, None), name="opt", out_dir=tmp_path, isa="avx2", keep_nans=True)
    assert desc["isa"] == ["avx2"] and desc["keep_nans"] and not (tmp_path / "opt_batch_avx512.c").exists()
    lib, ref = build(tmp_path), jax.tree.leaves(jax.vmap(f, (0, None))(x, c))
    close(point(lib, "opt", [x, c], desc, 2), ref, desc)
    if "avx2" in VARIANTS:
        close(batch(lib, "opt_batch", [x, c], desc, 2), ref, desc)
    for kw, match in ((dict(mode="x"), "mode"), (dict(isa=("sse",)), "isa"), (dict(in_axes=(1, None)), "in_axes"),
                      (dict(in_axes=None), "batched"), (dict(name="nt"), "name"), (dict(name="kernelax_k"), "name")):
        with pytest.raises(ValueError, match=match):
            kernelax.export(f, ex, **({"in_axes": (0, None), "name": "e"} | kw), out_dir=tmp_path / "e")


def test_float_literals_beyond_float32():
    """A constant beyond float32 in float code (point mode, GPU kernels) is written as inf, as XLA rounds it (it was the
    invalid literal "inff")."""
    from kernelax.gpu import _lit
    assert _lit(1e39, "float") == "__builtin_huge_valf()" and _lit(-1e39, "float") == "-__builtin_huge_valf()"
    assert _lit(1e39, "double") == (1e39).hex() and _lit(1.5, "float") == "0x1.8000000000000p+0f"


@pytest.mark.skipif(not X86 or len(os.sched_getaffinity(0)) < 2, reason="MXCSR (x86-64), threads beside the caller's")
def test_host_threads_keep_their_floating_point_mode(tmp_path):
    """A host whose main thread flushes subnormals (FTZ, DAZ) calls the batch kernel on its OpenMP threads: they compute
    in the caller's mode, and afterwards have their own again (the host's later parallel regions)."""
    kernelax.export(lambda x: jnp.sin(x) * 2.0, (np.zeros(3),), name="fpm", out_dir=tmp_path, mode="batch")
    build(tmp_path)
    (tmp_path / "host.c").write_text(r'''#include <stdio.h>
#include <omp.h>
#include <xmmintrin.h>
#include "fpm.h"
#define NT (omp_get_num_procs() < 4 ? omp_get_num_procs() : 4)  /* (the batch function refuses more threads than CPUs) */
int main(void) {
  double x[3 * 64], y[3 * 64];
  for (int i = 0; i < 3 * 64; ++i) x[i] = i * 0.01;
  volatile int started = 0;
  #pragma omp parallel num_threads(NT)
  { started = 1; }  /* (the pool's threads made here, before: IEEE, as threads inherit their creator's mode) */
  _mm_setcsr(_mm_getcsr() | 0x8040);  /* (the main thread: FTZ, DAZ) */
  if (fpm_batch(NT, 64, x, y)) return 1;
  int kept = 0;
  #pragma omp parallel num_threads(NT) reduction(+:kept)
  { if (omp_get_thread_num() > 0 && (_mm_getcsr() & 0x8040)) kept += 1; }
  printf("%d\n", kept);
  return 0;
}
''')
    subprocess.run(["gcc", "-O1", "-fopenmp", f"-I{tmp_path}", "-o", str(tmp_path / "host"), str(tmp_path / "host.c"), f"-L{tmp_path}",
                    "-lfpm", f"-Wl,-rpath,{tmp_path}"], check=True)
    assert subprocess.run([str(tmp_path / "host")], capture_output=True, text=True, check=True).stdout.strip() == "0"


def test_16_bit_integers_and_empty_outputs(tmp_path):
    """int16 and uint16 arguments and outputs, as int32_t (they were refused as 16-bit floats); a per-sample
    output without elements (the emitter failed on its empty group of stores)."""
    f = lambda x, u: (x * 3 - 1, (u // 3).astype(jnp.uint16), x[:0].astype(jnp.float64), x.astype(jnp.float64) + u)
    x = np.array([[-30000, 5, 7], [32767, -1, 0]], np.int16)
    u = np.array([[65535, 3, 0], [1, 2, 60000]], np.uint16)
    desc = kernelax.export(f, (x[0], u[0]), name="i16", out_dir=tmp_path)
    assert [a["c_type"] for a in desc["inputs"]] == ["int32_t", "int32_t"]  # (int32 holds every uint16)
    lib, ref = build(tmp_path), jax.tree.leaves(jax.vmap(f)(x, u))
    xs = [to_c(v, a) for v, a in zip((x, u), desc["inputs"])]
    close(batch(lib, "i16_batch", xs, desc, 2), ref, desc)
    close(point(lib, "i16", xs, desc, 2), ref, desc)


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="no gfortran")
def test_names_for_every_language(tmp_path):
    """Argument names not ASCII (ε, σ0: gfortran refused them), two of 60 characters (Fortran lines beyond 132), the
    names of classes the Java binding uses (System: refused as a name, as an argument renamed), a function without a
    module (its header named None)."""
    long = "a" * 60
    ns = {}
    exec(f"def law(ε, σ0, {long}, {long[:-1]}b, String):\n    return ε * σ0 + {long} - {long[:-1]}b + String\n", ns)
    law = ns["law"]
    law.__module__ = None
    one = np.ones(2)
    desc = kernelax.export(law, (one, one, one, one, one), name="law", out_dir=tmp_path, mode="point")
    names = [a["name"] for a in desc["inputs"]]
    assert all(n.isascii() for n in names) and "String" not in names, names
    subprocess.run(["gfortran", "-c", "law.f90"], cwd=tmp_path, check=True, capture_output=True)
    assert max(len(line) for line in (tmp_path / "law.f90").read_text().splitlines()) <= 132
    with pytest.raises(ValueError, match="name"):
        kernelax.export(law, (one,) * 5, name="System", out_dir=tmp_path / "s")


def test_warning_without_x64(tmp_path):
    """Without x64, float64 examples become float32 (float in the library): said in a warning."""
    with jax.enable_x64(False), pytest.warns(UserWarning, match="x64"):
        kernelax.export(lambda x: x * 2, (np.zeros(3),), name="w", out_dir=tmp_path)


def range_cases():
    """(name, f, args, x64) of test_point_code_integer_ranges (test_gpu's too)."""
    t = np.array([10 ** 10, 3, -2 * 10 ** 10, 0], np.int64)
    a = np.array([3 * 10 ** 6, 2 ** 31 - 1, 7, -1], np.int32)
    f64 = lambda v: v.astype(jnp.float64)
    x = np.array([1.0, 3.0, 0.25, -7.5], np.float32)
    i = np.array([[1000, 1000, 1000, 1000], [2 ** 30, 2 ** 30, 2 ** 30, 2 ** 30], [5, 6, 7, 8], [-1, 2, -3, 4]], np.int32)
    k = np.array([2 ** 31 - 1, 2 ** 31 - 2, 2 ** 30, -2 ** 31, 7, -1], np.int32)
    y = np.array([np.nan, np.inf, -np.inf, 3e9, 2.5, -7.5], np.float32)
    return [("rng8", lambda t, a: (f64(t) * 1e9, f64(a) ** 3, f64(a) * 1e10 + f64(t),
                                   jax.lax.fori_loop(1, 22, lambda i, c: c * i, jnp.float64(1.0) + 0.0 * a)), (t, a), True),
            ("rng4", lambda x, i: (((x.astype(jnp.float64) + 1e8) - 1e8).astype(jnp.float32), jnp.prod(i).astype(jnp.float32),
                                   (jnp.sum(i.astype(jnp.int64)) // 4).astype(jnp.int32), jnp.int32(2 ** 31 - 1) + 0 * i[0],
                                   jnp.int32(16777217) + 0 * i[0], jnp.uint32(7) // jnp.uint32(0) + 0 * i[0].astype(jnp.uint32)),
             (x, i), True),
            ("pos", lambda k, v: (v.at[k].set(9.0, mode="clip"), jnp.take(v, k, mode="clip")),
             (k, np.tile(np.arange(8, dtype=np.float32), (len(k), 1))), False),
            ("clipnan", lambda y: jnp.clip(jnp.floor(y), -1e6, 1e6).astype(jnp.float64) + 1.0, (y,), True)]


def test_point_code_integer_ranges(tmp_path):
    """The GPU emitter's integers (its point code: the same scalar code) by value ranges, as the C kernels' doubles: an
    integral float64 beyond int64 stays a double (t * 1e9 wrapped as long long; 21! in a loop); float64 intermediates
    of a 4-byte program (x64) are doubles, int64 ones long long (cut to float32, int; int + int summed in int); constant
    int32 outputs as integer literals (2^31 - 1 became INT_MIN); positions near INT32_MAX (an offset wrapped); a user's
    clip of a NaN stays NaN (not an integer)."""
    for name, f, args, x64 in range_cases():
        with jax.enable_x64(x64):
            desc = kernelax.export(f, tuple(a[0] for a in args), name=name, out_dir=tmp_path / name, mode="point")
            ref = jax.tree.leaves(jax.vmap(f)(*args))
        res = point(build(tmp_path / name), name, [to_c(a, d) for a, d in zip(args, desc["inputs"])], desc, len(args[0]))
        for o, r, d in zip(res, ref, desc["outputs"]):
            np.testing.assert_array_equal(np.asarray(from_c(o, d)).astype(np.asarray(r).dtype), np.asarray(r), err_msg=f"{name} {d['name']}")


CHUNKS = 12


@pytest.mark.parametrize("f32", [False, True], ids=["float64", "float32"])
@pytest.mark.parametrize("chunk", range(CHUNKS))
def test_point_code_coverage(tmp_path, chunk, f32):
    """The coverage cases (tests/test_coverage.py) through the point function, the GPU emitter's scalar code compiled
    for the CPU, against jax.vmap: the GPU emitter's test where there is no GPU (it would have found its integer and
    type faults). Cases in bundles, one function each (a compile per bundle); a case the
    point mode does not export (16-bit floats, sums over the samples) is left out."""
    sys.path.insert(0, os.path.dirname(__file__))
    from test_coverage import C, GAPS
    from test_f32 import _check, _f32
    cases = []
    for _, name, f, args in [c for c in C if c[1] not in GAPS][chunk::CHUNKS]:
        if f32 and ("eig" in name and "eigh" not in name or "schur" in name):  # (float32 without FMA: the order of eigenvalues
            continue  # depends on rounding, as LAPACK's)
        args = [_f32(a) for a in args] if f32 else list(args)
        try:
            ref = jax.tree.leaves(jax.vmap(f)(*args))
        except (TypeError, ValueError):  # (float64 constants in a float32 case: not for JAX either)
            continue
        cases.append((name, f, args, ref))
    n = min(len(a) for _, _, args, _ in cases for a in args)

    def export(group, d):
        sizes = [len(args) for _, _, args, _ in group]
        g = lambda *flat: tuple(f(*flat[sum(sizes[:k]):sum(sizes[:k + 1])]) for k, (_, f, _, _) in enumerate(group))
        flat = [a[:n] for _, _, args, _ in group for a in args]
        desc = kernelax.export(g, tuple(a[0] for a in flat), name="cov", out_dir=d, mode="point")
        lib = build(d, "", "FLAGS=-O1 -fno-math-errno -ffp-contract=fast -fwrapv -fPIC -Wno-psabi")  # (-O1: a third of
        res = point(lib, "cov", [to_c(a, x) for a, x in zip(flat, desc["inputs"])], desc, n)  # -O3's time, the same code)
        return [np.asarray(from_c(o, x)) for o, x in zip(res, desc["outputs"])]
    try:
        groups = [(cases, export(cases, tmp_path / "all"))]
    except NotImplementedError:  # (one case the point mode lacks: each on its own)
        groups = []
        for k, c in enumerate(cases):
            try:
                groups.append(([c], export([c], tmp_path / str(k))))
            except NotImplementedError:
                pass
    checked = 0
    for group, outs in groups:
        refs = [r[:n] for _, _, _, ref in group for r in ref]
        for (name, *_), o, r in zip([c for c in group for _ in c[3]], outs, refs):
            if f32 or np.asarray(r).dtype in (np.float32, np.complex64):
                r = np.asarray(r)
                _check([o.real, o.imag] if np.iscomplexobj(r) else [o], [r.real, r.imag] if np.iscomplexobj(r) else [r])
            else:
                fin = np.abs(np.asarray(r, complex)[np.isfinite(np.asarray(r, complex))])
                np.testing.assert_allclose(o, r, rtol=1e-9, atol=1e-11 * (1 + (fin.max() if fin.size else 0)), equal_nan=True, err_msg=name)
            checked += 1
    assert checked


@pytest.mark.skipif(not X86 or len(os.sched_getaffinity(0)) < 2, reason="MXCSR (x86-64), threads beside the caller's")
def test_reserved_names_and_scalar_floating_point_mode(tmp_path):
    """Arguments named typeof, asm (GNU C keywords: the build failed) or I (a macro of <complex.h> in a host) are renamed;
    the scalar path of the dispatcher (no SIMD variant) computes in the caller's floating-point mode in all its threads
    (FTZ, DAZ: before, only the calling thread's samples were flushed)."""
    f = lambda typeof, asm, I: typeof * 3.0 + asm * I
    desc = kernelax.export(f, (np.zeros(2), np.zeros(2), np.zeros(2)), name="rsv", out_dir=tmp_path)
    assert not {"typeof", "asm", "I"} & {a["name"] for a in desc["inputs"]}
    build(tmp_path, "-DKERNELAX_SCALAR")
    (tmp_path / "host.c").write_text(r'''#include <stdio.h>
#include <omp.h>
#include <xmmintrin.h>
#include "rsv.h"
#define NT (omp_get_num_procs() < 4 ? omp_get_num_procs() : 4)  /* (the batch function refuses more threads than CPUs) */
int main(void) {
  double x[2 * 64], z[2 * 64], y[2 * 64];
  for (int i = 0; i < 2 * 64; ++i) { x[i] = 1e-310; z[i] = 0.0; }
  #pragma omp parallel num_threads(NT)
  { (void)0; }  /* (the pool's threads made before: IEEE) */
  _mm_setcsr(_mm_getcsr() | 0x8040);  /* (FTZ, DAZ) */
  if (rsv_batch(NT, 64, x, z, z, y)) return 1;
  int nonzero = 0;
  for (int i = 0; i < 2 * 64; ++i) nonzero += y[i] != 0.0;
  printf("%s %d\n", rsv_isa(), nonzero);
  return 0;
}
''')
    subprocess.run(["gcc", "-O1", "-fopenmp", f"-I{tmp_path}", "-o", str(tmp_path / "host"), str(tmp_path / "host.c"), f"-L{tmp_path}",
                    "-lrsv", f"-Wl,-rpath,{tmp_path}"], check=True)
    assert subprocess.run([str(tmp_path / "host")], capture_output=True, text=True, check=True).stdout.split() == ["scalar", "0"]
