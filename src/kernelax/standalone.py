# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""export: a per-sample JAX function as a standalone library, without Python, XLA or the kernelax runtime.

Two modes over one trace:
- batch: the C kernel over n samples (emit.emit_map: SIMD lanes, OpenMP threads), one file per instruction set
  (AVX2, AVX-512, NEON), the variant for the CPU chosen at every call (NAME_batch.c);
- point: one sample per call, the GPU emitter's scalar code compiled for the CPU (gpu.emit(cpu=True)): no SIMD, OpenMP
  or libmvec, the cheaper call for hosts that integrate point by point (MFront, UMAT).
Written with a C header, a Fortran module (ISO_C_BINDING), a Java class (Foreign Function & Memory API), a Makefile
and a JSON description of the arguments.
"""

import inspect
import json
import os
import re
import warnings

import jax
import numpy as np

from kernelax import __version__, cpu, gpu
from kernelax.build import BIG
from kernelax.dag import Node
from kernelax.emit import emit_map
from kernelax.frontend import _flat
from kernelax.realify import complex_leaves
from kernelax.vec import lit_d

ISAS = {"avx2": (4, "x86-64-v3", "__x86_64__"), "avx512": (8, "x86-64-v4", "__x86_64__"), "neon": (2, "armv8-a", "__aarch64__")}
FLAGS = "-O3 -fno-math-errno -ffp-contract=fast -fwrapv -fPIC -Wno-psabi"  # (-fwrapv: integers wrap as in XLA; build.FLAGS)
_JAVA = {"double": ("double", "JAVA_DOUBLE"), "float": ("float", "JAVA_FLOAT"), "int32_t": ("int", "JAVA_INT"),
         "uint32_t": ("int", "JAVA_INT"), "int64_t": ("long", "JAVA_LONG"), "uint64_t": ("long", "JAVA_LONG")}
_FORTRAN = {"double": "real(c_double)", "float": "real(c_float)", "int32_t": "integer(c_int32_t)",
            "uint32_t": "integer(c_int32_t)", "int64_t": "integer(c_int64_t)", "uint64_t": "integer(c_int64_t)"}
_KEYWORDS = set("""auto break case char const continue default do double else enum extern float for goto if inline int long
register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while bool true
false class namespace new delete template this throw try catch public private protected virtual friend operator using
typename abstract assert boolean byte extends final finally implements import instanceof interface native package super
synchronized throws transient var record module function subroutine end program real integer logical constexpr noexcept
explicit mutable static_assert thread_local alignas alignof decltype nullptr char8_t char16_t char32_t wchar_t concept
requires co_await co_return co_yield consteval constinit export typeid dynamic_cast static_cast reinterpret_cast
const_cast and and_eq bitand bitor compl not not_eq or or_eq xor xor_eq strictfp yield sealed permits typeof asm I
complex imaginary
System String Object Math Arena Linker MemorySegment MethodHandle FunctionDescriptor SymbolLookup ValueLayout Throwable
RuntimeException UnsatisfiedLinkError IllegalArgumentException""".split())  # (the last two rows: classes the Java binding
# uses, which a class or parameter of that name would hide)
_OWN = {"nt", "n", "kernel", "kernelax_k"}  # (names of the generated code next to the arguments; compared without case;
# also everything beginning with kx_; the definitions name their arguments by position, kx_a0, ...: the kernel's macros)


def export(f, example, in_axes=0, *, name, out_dir=".", mode=("batch", "point"), isa=("avx2", "avx512", "neon"),
           keep_nans=False):
    """Writes f, a function of one sample, as a standalone library NAME into out_dir; returns its description (the
    dict of NAME.json).

    example: the arguments of one sample (pytrees of arrays, or jax.ShapeDtypeStruct); in_axes: 0 (batched: one row
    per sample) or None (the same for all samples, e.g. material parameters), one for all arguments or one per argument
    (static arguments: functools.partial). name: a C identifier, the prefix of every symbol and file. mode: "batch"
    and/or "point". isa: the batch kernel's variants ("avx2", "avx512": x86-64, the CPU's chosen at run time; "neon":
    aarch64, experimental). Files: NAME.h, NAME_batch_<isa>.c, NAME_batch.c (the dispatcher), NAME.cpp (the point
    function), NAME.f90, NAME.java, Makefile (libNAME.so for the machine it runs on), NAME.json. The C types are the
    kernel's (NAME.h): float64 as double, 64-bit integers as int64_t (their bits), float32 as float, other integers as
    int32_t / uint32_t; outputs of kernels with 8-byte elements (any 8-byte dtype in or out) as double (integers: exact
    values; 64-bit ones: int64_t, their bits); complex values as [re, im] on a trailing axis."""
    mode, isa = (mode,) if isinstance(mode, str) else tuple(mode), (isa,) if isinstance(isa, str) else tuple(isa)
    _check_name(name)
    if not mode or set(mode) - {"batch", "point"}:
        raise ValueError(f"kernelax.export: mode {mode!r}: 'batch' and/or 'point'")
    if "batch" in mode and (not isa or set(isa) - set(ISAS)):
        raise ValueError(f"kernelax.export: isa {isa!r}: choose from {', '.join(ISAS)}")
    leaf = lambda x: np.zeros(x.shape, x.dtype) if isinstance(x, jax.ShapeDtypeStruct) else np.asarray(x)
    raw = [jax.tree.map(leaf, a) for a in example]
    example = [jax.tree.map(lambda a: a.astype(jax.dtypes.canonicalize_dtype(a.dtype)), a) for a in raw]
    if any(x.dtype != y.dtype for a, b in zip(raw, example) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))):
        warnings.warn("kernelax.export: jax_enable_x64 is off: 64-bit examples become float32 and int32 (float, int32_t "
                      "in the library); jax.config.update('jax_enable_x64', True) for double and int64_t")
    in_axes = (in_axes,) * len(example) if in_axes in (0, None) else tuple(in_axes)
    if len(in_axes) != len(example) or any(a not in (0, None) for a in in_axes):
        raise ValueError("kernelax.export: in_axes needs 0 or None per argument")
    if 0 not in in_axes:
        raise ValueError("kernelax.export: no batched argument (in_axes: 0 for the arguments per sample)")
    ins, outs, g, flat = _arguments(f, example, in_axes)
    batched = [a["batched"] for a in ins]
    with cpu._LOCK:  # (the frontend's DAG: one trace, its emits, then the next trace may clear it)
        inputs, nodes_out, _, mem = cpu._trace(g, flat, batched, keep_nans=keep_nans)
        E, kin, kout = mem
        if set(kin + kout) & set("hb") or any(jax.dtypes.issubdtype(a["dtype"], np.floating) and np.dtype(a["dtype"]).itemsize == 2
                                              for a in ins + outs):  # (16-bit integers: as int32_t, uint32_t)
            raise NotImplementedError("kernelax.export: 16-bit floats (pass them as float32)")
        sources = {i: emit_map(inputs, nodes_out, batched, ISAS[i][0], cpu.FIX_K or 1, None, mem, split=False, static=True)
                   for i in isa} if "batch" in mode else {}
        point = None
        if "point" in mode:
            nodes = gpu._all_nodes([x for o in nodes_out for x in _flat(o) if isinstance(x, Node)])
            point = gpu.emit(inputs, nodes_out, batched, None, mem, nodes=nodes, smem=0, live=10 ** 9, cpu=True)[0]
    for a, k in zip(ins, kin):
        a["c_type"] = ("int64_t" if np.dtype(a["dtype"]).kind == "i" else "uint64_t") if k == "d" and np.dtype(a["dtype"]).kind in "iu" \
            else {"d": "double", "f": "float", "i": "int32_t", "u": "uint32_t"}[k]
    for a, k in zip(outs, kout):
        wide = np.dtype(a["dtype"]).itemsize == 8 and np.dtype(a["dtype"]).kind in "iu"
        a["c_type"] = ("int64_t" if np.dtype(a["dtype"]).kind == "i" else "uint64_t") if wide else \
            "double" if E == 8 else {"f": "float", "i": "int32_t", "u": "uint32_t"}[k]
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for i, (src, params) in sources.items():
        files[f"{name}_batch_{i}.c"] = _variant(name, i, src, params, ins, outs)
    if sources:
        files[f"{name}_batch.c"] = _dispatcher(name, isa, ins, outs, point is not None)
    if point is not None:
        files[f"{name}.cpp"] = _point(name, point, ins, outs, kin, E)
    desc = dict(name=name, kernelax=__version__, kernelax_sources=cpu._VERSION[:16], jax=jax.__version__,
                keep_nans=bool(keep_nans), element_bytes=E, mode=list(mode), isa=list(isa) if sources else [], inputs=ins, outputs=outs,
                functions=_signatures(name, isa if sources else (), point is not None, ins, outs),
                compile=_commands(name, {i: len(s) for i, (s, _) in sources.items()}, point is not None),
                requires=(["gcc >= 12 (vector extensions)", "glibc >= 2.35 (libmvec; aarch64: >= 2.39)", "OpenMP (libgomp)"]
                          if sources else []) + (["a C++ compiler"] if point is not None else []),
                notes=["Arrays are row-major (C): a batched argument is (n, *shape), one row per sample; a shared one "
                       "(*shape). Fortran sees the transposed shapes: (n, 6) in C is (6, n) in Fortran.",
                       "NAME_batch's nt: OpenMP threads; 1 where the host parallelizes itself.",
                       "The kernels compute in the caller's floating-point mode; results are not bitwise equal to XLA's."])
    desc["notes"] = [s.replace("NAME", name) for s in desc["notes"]]
    files[f"{name}.h"] = _header(name, desc, f)
    files[f"{name}.f90"] = _fortran(name, desc)
    files[f"{name}.java"] = _java(name, desc)
    files["Makefile"] = _makefile(name, {i: len(s) for i, (s, _) in sources.items()}, point is not None)
    files[f"{name}.json"] = json.dumps(desc, indent=1) + "\n"
    for fn, text in files.items():
        with open(os.path.join(out_dir, fn), "w") as h:
            h.write(text)
    return desc


def _check_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", name) or name in _KEYWORDS | _OWN:
        raise ValueError(f"kernelax.export: name {name!r}: a C identifier of at most 40 characters, starting with a "
                         "letter, not a keyword of C, C++, Java or Fortran nor a class the Java binding uses")


def _ident(s, taken):
    """s as an identifier valid in C, C++, Java and Fortran, not in `taken` (compared without case: Fortran)."""
    s = re.sub(r"\W+", "_", s, flags=re.ASCII).strip("_")[:60] or "x"  # (Fortran: at most 63 characters, ASCII)
    s = ("x_" + s) if s[0].isdigit() else ("a" + s) if s.lower().startswith("kx_") else s
    s = s + "_" if s in _KEYWORDS else s
    base, k = s, 1
    while s.lower() in taken:
        s, k = f"{base}_{k}", k + 1
    taken.add(s.lower())
    return s


def _key(k):
    return str(getattr(k, "key", getattr(k, "idx", getattr(k, "name", k))))


def _arguments(f, example, in_axes):
    """The input and output leaves (name, path, dtype, shape per sample, batched), f on the flat real leaves (complex ones
    as [re, im] on a trailing axis: realify.complex_leaves) and those leaves."""
    try:
        pos = [q.name for q in inspect.signature(f).parameters.values()]
    except (TypeError, ValueError):
        pos = []
    taken, ins = set(_OWN), []
    for i, (x, ax) in enumerate(zip(example, in_axes)):
        base = pos[i] if i < len(pos) else f"in{i}"
        for path, leaf in jax.tree_util.tree_flatten_with_path(x)[0]:
            c = np.iscomplexobj(leaf)
            ins.append(dict(name=_ident(base + "".join("_" + _key(k) for k in path), taken), path=base + jax.tree_util.keystr(path),
                            dtype=str(np.asarray(leaf).dtype), shape=list(np.shape(leaf)) + [2] * c, batched=ax == 0, complex=c))
    outs = []
    for path, o in jax.tree_util.tree_flatten_with_path(jax.eval_shape(f, *example))[0]:
        c = jax.numpy.iscomplexobj(o)
        nm = "out" if not path else ("out" if hasattr(path[0], "idx") else "") + "_".join(_key(k) for k in path)
        outs.append(dict(name=_ident(nm, taken), path="out" + jax.tree_util.keystr(path), dtype=str(o.dtype),
                         shape=list(o.shape) + [2] * c, batched=True, complex=c))
    leaves, tree = jax.tree.flatten(list(example))
    cin = [np.iscomplexobj(x) for x in leaves]
    flat = [np.stack([np.real(x), np.imag(x)], -1) if c else x for x, c in zip(leaves, cin)]
    g = complex_leaves(f, tree, cin) if any(cin) or any(a["complex"] for a in outs) else lambda *l: f(*jax.tree.unflatten(tree, l))
    return ins, outs, g, flat


def _size(a):
    return int(np.prod(a["shape"], dtype=np.int64))


def _params(ins, outs, positional=False):  # (positional: kx_a0, ... for the definitions, next to the kernel's macros)
    nm = (lambda k, a: f"kx_a{k}") if positional else (lambda k, a: a["name"])
    return ", ".join([f"const {a['c_type']}* {nm(k, a)}" for k, a in enumerate(ins)] +
                     [f"{a['c_type']}* {nm(k, a)}" for k, a in enumerate(outs, len(ins))])


def _pos(ins, outs):  # the definitions' names of the arguments, in order
    return [f"kx_a{k}" for k in range(len(ins) + len(outs))]


def _signatures(name, isa, point, ins, outs):
    sig = {}
    if isa:
        sig[f"{name}_batch"] = f"int {name}_batch(int nt, int64_t n, {_params(ins, outs)})"
        sig.update({f"{name}_batch_{i}": f"int {name}_batch_{i}(int nt, int64_t n, {_params(ins, outs)})" for i in isa})
        sig[f"{name}_isa"] = f"const char* {name}_isa(void)"
    if point:
        sig[f"{name}_point"] = f"void {name}_point({_params(ins, outs)})"
    return sig


def _variant(name, isa, src, params, ins, outs):
    prm = ", ".join(lit_d(v) for v in params)
    pos = _pos(ins, outs)
    args = ", ".join([f"(void*){a}" for a in pos[:len(ins)]] + ["(void*)kx_prm"] + [f"(void*){a}" for a in pos[len(ins):]])
    return (f"/* {name}_batch_{isa}.c: generated by kernelax; the batch kernel for {isa} (compile with -march={ISAS[isa][1]} "
            f"-fopenmp, see {name}.h) */\n" + src
            + f"\nint {name}_batch_{isa}(int nt, int64_t n, {_params(ins, outs, True)}) {{\n"
            + "  if (n < 0 || nt < 1 || nt > omp_get_num_procs()) return 2;\n"
            + f"  static const double kx_prm[] = {{{prm}}};  /* (run-time values: constants of the address arithmetic, sizes) */\n"
            + f"  void* kx_p[] = {{{args}}};\n  kernel(nt, n, kx_p);\n  return 0;\n}}\n")


def _dispatcher(name, isa, ins, outs, point):
    pos = _pos(ins, outs)
    call = ", ".join(pos)
    picks = []
    for arch in ("__x86_64__", "__aarch64__"):
        here = [i for i in ("avx512", "avx2", "neon") if i in isa and ISAS[i][2] == arch]
        if here:
            picks.append(f"#if defined({arch})")
            if arch == "__x86_64__":
                picks.append("  __builtin_cpu_init();")
            for i in here:
                cond = f"__builtin_cpu_supports(\"{ISAS[i][1]}\")" if arch == "__x86_64__" else "1"
                picks.append(f"  if ({cond}) {{ *isa = \"{i}\"; return {name}_batch_{i}; }}")
            picks.append("#endif")
    rows = [f"{p}{f' + kx_i * {_size(a)}' if a['batched'] else ''}" for p, a in zip(pos, ins)] + \
        [f"{p} + kx_i * {_size(a)}" for p, a in zip(pos[len(ins):], outs)]
    scalar = (f"const uint64_t kx_mode = KX_FPGET();  /* the caller's floating-point mode in all threads (as the kernels) */\n"
              f"  #pragma omp parallel num_threads(nt)\n  {{\n    const uint64_t kx_own = KX_FPGET();\n    KX_FPSET(kx_mode);\n"
              f"    #pragma omp for schedule(static)\n    for (int64_t kx_i = 0; kx_i < n; ++kx_i)\n"
              f"      {name}_point({', '.join(rows)});\n    KX_FPSET(kx_own);\n  }}\n  return 0;") if point else "(void)nt; (void)n;\n  return 1;"
    return f"""/* {name}_batch.c: generated by kernelax; {name}_batch runs the variant of the batch kernel for this CPU, chosen
   at every call (x86-64: AVX-512 where the CPU has x86-64-v4, else AVX2 with x86-64-v3; aarch64: NEON), else
   {'the point function per sample' if point else 'nothing: it returns 1'}. -DKERNELAX_SCALAR: no variant (tests). */
#include <stdint.h>
#include <omp.h>
#include "{name}.h"
#if defined(__x86_64__)
#include <xmmintrin.h>
#define KX_FPGET() ((uint64_t)_mm_getcsr())
#define KX_FPSET(c) _mm_setcsr((unsigned)(c))
#elif defined(__aarch64__)
#define KX_FPGET() __builtin_aarch64_get_fpcr64()
#define KX_FPSET(c) __builtin_aarch64_set_fpcr64(c)
#else
#define KX_FPGET() ((uint64_t)0)
#define KX_FPSET(c) (void)(c)
#endif

typedef int (*{name}_fn)(int, int64_t, {', '.join([f"const {a['c_type']}*" for a in ins] + [f"{a['c_type']}*" for a in outs])});

static {name}_fn {name}_pick(const char** isa) {{
#if !defined(KERNELAX_SCALAR)
{chr(10).join(picks)}
#endif
  *isa = "{'scalar' if point else 'none'}";
  return 0;
}}

const char* {name}_isa(void) {{
  const char* isa;
  {name}_pick(&isa);
  return isa;
}}

int {name}_batch(int nt, int64_t n, {_params(ins, outs, True)}) {{
  if (n < 0 || nt < 1 || nt > omp_get_num_procs()) return 2;
  const char* kx_isa;
  const {name}_fn kx_f = {name}_pick(&kx_isa);
  if (kx_f)
    return kx_f(nt, n, {call});
  {scalar}
}}
"""


def _point(name, src, ins, outs, kin, E):
    ct = {"d": "double", "f": "float", "i": "int", "u": "unsigned"}
    pos = _pos(ins, outs)
    args = [f"(const {ct[k]}*)(const void*){p}" for p, k in zip(pos, kin)] + \
           [f"({'float' if E == 4 else 'double'}*)(void*){p}" for p in pos[len(ins):]]
    return (f"// {name}.cpp: generated by kernelax; {name}_point: one sample per call (see {name}.h)\n" + src
            + f"\nextern \"C\" void {name}_point({_params(ins, outs, True)}) {{\n  kernelax_k(1, {', '.join(args)});\n}}\n")


def _header(name, desc, f):
    rows = lambda xs: "\n".join(f"     {a['name']:<16} {a['c_type'] + '*':<10} {a['dtype']:<10} {str(tuple(a['shape'])):<14} "
                                f"{'batched: (n, *shape)' if a['batched'] else 'shared by all samples'}  ({a['path']})" for a in xs)
    protos = "\n".join(s + ";" for s in desc["functions"].values())
    src = f"{getattr(f, '__module__', None) or '?'}.{getattr(f, '__qualname__', None) or getattr(f, '__name__', None) or '?'}"
    cmds = "\n".join("     " + c for c in desc["compile"])
    return f"""/* {name}.h: generated by kernelax {__version__} from {src}.

   Inputs (row-major; batched: one row per sample):
{rows(desc['inputs'])}
   Outputs (one row per sample):
{rows(desc['outputs'])}

   {name}_batch(nt, n, ...): n samples on nt OpenMP threads (nt = 1 where the host parallelizes); 0, 1 if no variant
     runs on this CPU (and no point function was exported), 2 if n < 0, nt < 1 or nt beyond the processors. {name}_isa(): the variant it runs.
     {name}_batch_<isa>: a variant directly (the CPU must have its instruction set).
   {name}_point(...): one sample. Both are reentrant (concurrent calls from several host threads). Outputs must not
     overlap inputs (an output is written as soon as it is computed, inputs are read when used).
   Build: make (libNAME.so for this machine), or:
{cmds}
*/
#ifndef {name.upper()}_H
#define {name.upper()}_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {{
#endif
{protos}
#ifdef __cplusplus
}}
#endif
#endif
""".replace("libNAME", f"lib{name}")


def _flags(size):
    return FLAGS.replace("-O3", "-O1") if size > BIG else FLAGS  # (large sources: about as fast at -O1, in half gcc's time)


def _commands(name, sizes, point):
    out = [f"$CC {_flags(s)} -march={ISAS[i][1]} -fopenmp -c {name}_batch_{i}.c  # ({'x86-64' if ISAS[i][2] == '__x86_64__' else 'aarch64'})"
           for i, s in sizes.items()]
    if sizes:
        out.append(f"$CC {FLAGS} -fopenmp -c {name}_batch.c")
    if point:
        out.append(f"$CXX {FLAGS} -c {name}.cpp")
    objs = " ".join(([f"{name}_batch.o", f"{name}_batch_<isa>.o ..."] if sizes else []) + ([f"{name}.o"] if point else []))
    out.append(f"$CXX -shared {'-fopenmp ' if sizes else ''}-o lib{name}.so {objs}{' -lmvec -lm' if sizes else ' -lm'}")
    return out


def _makefile(name, sizes, point):
    x86 = " ".join(f"{name}_batch_{i}.o" for i in sizes if ISAS[i][2] == "__x86_64__")
    arm = " ".join(f"{name}_batch_{i}.o" for i in sizes if ISAS[i][2] == "__aarch64__")
    rules = "".join(f"{name}_batch_{i}.o: {name}_batch_{i}.c\n\t$(CC) {_flags(s)} -march={ISAS[i][1]} -fopenmp -c $< -o $@\n\n"
                    for i, s in sizes.items())
    objs = ([f"{name}_batch.o"] if sizes else []) + ([f"{name}.o"] if point else [])
    return f"""# generated by kernelax: lib{name}.so for this machine (x86-64: the AVX2 and AVX-512 variants of the batch kernel,
# chosen at run time; aarch64: NEON). make DEFS=-DKERNELAX_SCALAR: the batch function without them (tests).
ifeq ($(origin CC),default)
CC = gcc
endif
ifeq ($(origin CXX),default)
CXX = g++
endif
FLAGS = {FLAGS}
ARCH := $(shell uname -m)
OBJS = {' '.join(objs)}
ifeq ($(ARCH),x86_64)
OBJS += {x86}
endif
ifeq ($(ARCH),aarch64)
OBJS += {arm}
endif

lib{name}.so: $(OBJS)
\t$(CXX) -shared {'-fopenmp ' if sizes else ''}-o $@ $^ {'-lmvec -lm' if sizes else '-lm'}

{rules}{f'''{name}_batch.o: {name}_batch.c {name}.h
\t$(CC) $(FLAGS) $(DEFS) -fopenmp -c $< -o $@

''' if sizes else ''}{f'''{name}.o: {name}.cpp
\t$(CXX) $(FLAGS) -c $< -o $@

''' if point else ''}clean:
\trm -f *.o lib{name}.so
"""


def _fortran(name, desc):
    ins, outs = desc["inputs"], desc["outputs"]
    akinds = {_FORTRAN[a["c_type"]].split("(")[1][:-1] for a in ins + outs}
    decl = lambda a, intent: f"      {_FORTRAN[a['c_type']]}, intent({intent}) :: {a['name']}(*)"
    args = ", ".join(a["name"] for a in ins + outs)
    body = "\n".join([decl(a, "in") for a in ins] + [decl(a, "out") for a in outs])
    def wrap(s):  # (free form: lines up to 132 characters; broken between arguments, continued with &)
        lines = [""]
        for part in s.split(", "):
            if lines[-1].strip() and len(lines[-1]) + len(part) > 100:
                lines[-1] += ", &"
                lines.append("        " + part)
            else:
                lines[-1] += (", " if lines[-1].strip() else "") + part
        return "\n".join(lines)
    parts = []
    if "batch" in desc["mode"]:
        parts.append(wrap(f"    integer(c_int) function {name}_batch(nt, n, {args}) bind(C, name=\"{name}_batch\")") + f"""
      import :: {', '.join(sorted(akinds | {'c_int', 'c_int64_t'}))}
      integer(c_int), value :: nt
      integer(c_int64_t), value :: n
{body}
    end function {name}_batch""")
    if "point" in desc["mode"]:
        parts.append(wrap(f"    subroutine {name}_point({args}) bind(C, name=\"{name}_point\")") + f"""
      import :: {', '.join(sorted(akinds))}
{body}
    end subroutine {name}_point""")
    return f"""! {name}.f90: generated by kernelax; Fortran interfaces of lib{name} (ISO_C_BINDING, see {name}.h).
! Arrays are column-major: a C argument of shape (n, 6) is a Fortran array (6, n); pass contiguous arrays, outputs
! not overlapping inputs. {name}_batch: 0, 1 if nothing runs on this CPU, 2 if n < 0, nt < 1 or nt beyond the processors.
module {name}_mod
  use, intrinsic :: iso_c_binding
  implicit none
  interface
{chr(10).join(parts)}
  end interface
end module {name}_mod
"""


def _java(name, desc):
    ins, outs = desc["inputs"], desc["outputs"]
    jt = lambda a: _JAVA[a["c_type"]][0] + "[]"
    arrs = ", ".join(f"{jt(a)} {a['name']}" for a in ins + outs)
    segs = ", ".join(f"MemorySegment {a['name']}" for a in ins + outs)
    heap = ", ".join(f"MemorySegment.ofArray({a['name']})" for a in ins + outs)
    names = ", ".join(a["name"] for a in ins + outs)
    addr = ", ".join(["ADDRESS"] * len(ins + outs))
    nb = {"double": 8, "float": 4, "int32_t": 4, "uint32_t": 4, "int64_t": 8, "uint64_t": 8}
    need = lambda have, per: " ".join(  # (lengths checked: a shorter array would be written past its end)
        f'kx_need({have(a)}, {"n * " if per and a["batched"] else ""}{_size(a)}L, "{a["name"]}");' for a in ins + outs)
    arr_b, seg_b = need(lambda a: f"{a['name']}.length", True), need(lambda a: f"{a['name']}.byteSize() / {nb[a['c_type']]}", True)
    arr_p, seg_p = need(lambda a: f"{a['name']}.length", False), need(lambda a: f"{a['name']}.byteSize() / {nb[a['c_type']]}", False)
    methods = []
    if "batch" in desc["mode"]:
        methods.append(f"""  private static final MethodHandle KX_BATCH = kx_handle("{name}_batch", FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_LONG, {addr}), false);
  private static final MethodHandle KX_BATCH_HEAP = kx_handle("{name}_batch", FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_LONG, {addr}), true);
  private static final MethodHandle KX_ISA = kx_handle("{name}_isa", FunctionDescriptor.of(ADDRESS), false);

  /** n samples on nt OpenMP threads (1 where the caller parallelizes), arrays as in {name}.h (outputs not overlapping
   * inputs); 0, 1 if nothing runs on this CPU, 2 if nt &lt; 1. IllegalArgumentException: n &lt; 0 or an array too short. */
  public static int batch(int nt, long n, {arrs}) {{
    kx_need(n, 0, "n"); {arr_b}
    try {{ return (int) KX_BATCH_HEAP.invokeExact(nt, n, {heap}); }} catch (Throwable kx_e) {{ throw new RuntimeException(kx_e); }}
  }}

  /** As batch, on memory outside the heap. */
  public static int batch(int nt, long n, {segs}) {{
    kx_need(n, 0, "n"); {seg_b}
    try {{ return (int) KX_BATCH.invokeExact(nt, n, {names}); }} catch (Throwable kx_e) {{ throw new RuntimeException(kx_e); }}
  }}

  /** The variant batch runs on this CPU ("avx512", "avx2", "neon", "scalar"). */
  public static String isa() {{
    try {{ return ((MemorySegment) KX_ISA.invokeExact()).reinterpret(16).getString(0); }} catch (Throwable kx_e) {{ throw new RuntimeException(kx_e); }}
  }}""")
    if "point" in desc["mode"]:
        methods.append(f"""  private static final MethodHandle KX_POINT = kx_handle("{name}_point", FunctionDescriptor.ofVoid({addr}), false);
  private static final MethodHandle KX_POINT_HEAP = kx_handle("{name}_point", FunctionDescriptor.ofVoid({addr}), true);

  /** One sample, arrays as in {name}.h (outputs not overlapping inputs). IllegalArgumentException: an array too short. */
  public static void point({arrs}) {{
    {arr_p}
    try {{ KX_POINT_HEAP.invokeExact({heap}); }} catch (Throwable kx_e) {{ throw new RuntimeException(kx_e); }}
  }}

  /** As point, on memory outside the heap. */
  public static void point({segs}) {{
    {seg_p}
    try {{ KX_POINT.invokeExact({names}); }} catch (Throwable kx_e) {{ throw new RuntimeException(kx_e); }}
  }}""")
    return f"""// {name}.java: generated by kernelax; lib{name} through the Foreign Function & Memory API (JDK >= 22), arrays as in
// {name}.h (row-major). The library: -D{name}.lib=/path/lib{name}.so, else lib{name}.so on the library path; run with
// --enable-native-access=ALL-UNNAMED. Java arrays are passed without a copy (Linker.Option.critical).
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import static java.lang.foreign.ValueLayout.*;

public final class {name} {{
  private {name}() {{}}

  private static final Linker KX_LINKER = Linker.nativeLinker();
  private static final SymbolLookup KX_LIB = SymbolLookup.libraryLookup(
      System.getProperty("{name}.lib", System.mapLibraryName("{name}")), Arena.global());

  private static MethodHandle kx_handle(String symbol, FunctionDescriptor d, boolean heap) {{
    MemorySegment s = KX_LIB.find(symbol).orElseThrow(() -> new UnsatisfiedLinkError(symbol));
    return heap ? KX_LINKER.downcallHandle(s, d, Linker.Option.critical(true)) : KX_LINKER.downcallHandle(s, d);
  }}

  private static void kx_need(long have, long need, String what) {{
    if (have < need) throw new IllegalArgumentException("{name}: " + what + " has " + have + " elements, needs " + need);
  }}

{chr(10).join(methods)}
}}
"""
