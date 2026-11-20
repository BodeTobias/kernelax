# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""gcc build of generated C sources, cached on disk by source, flags, compiler and CPU flags."""

import atexit
import contextlib
import math
import struct
import shutil
import stat
import warnings
import ctypes
import fcntl
import functools
import gc
import hashlib
import os
import pathlib
import re
import subprocess
import threading
import time


_GC = [0, None, threading.Lock()]  # (threads in no_gc, whether the collector was on before)


@contextlib.contextmanager
def no_gc():
    """The cyclic garbage collector off (in all threads) while a program is planned or a kernel traced and emitted (not
    while gcc runs: the program's garbage would grow meanwhile): its full collections walk the planner's and tracer's
    millions of objects again and again (a large program's plan: 3.3 of 10.1 s); on again after (as it was)."""
    with _GC[2]:
        if not _GC[0]:
            _GC[1] = gc.isenabled()
            gc.disable()
        _GC[0] += 1
    try:
        yield
    finally:
        with _GC[2]:
            _GC[0] -= 1
            if not _GC[0] and _GC[1]:
                gc.enable()


def thread_map(f, items, jobs):
    """[f(x) for x in items] on up to `jobs` threads, the first error raised: plain threads, which unlike
    concurrent.futures' pools also start at interpreter exit (kernelax.jit finishes kernel builds for the index there)."""
    items, out, errors, lock = list(items), [None] * len(items), [], threading.Lock()
    todo = iter(range(len(items)))

    def work():
        while True:
            with lock:
                i = next(todo, None)
            if i is None:
                return
            try:
                out[i] = f(items[i])
            except BaseException as e:  # noqa: BLE001 (raised in the caller)
                errors.append((i, e))
    threads = [threading.Thread(target=work, daemon=True) for _ in range(max(1, min(jobs, len(items))))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise min(errors, key=lambda x: x[0])[1]
    return out


def env(name, default, kind=int, low=0, inf=False):
    """A KERNELAX_* setting: its environment variable as `kind` (int, float; bool: 0/1, false/true, no/yes, off/on),
    `default` where it is unset or empty; a value that is not one, below `low`, NaN or (unless `inf`) infinite raises
    ValueError naming the variable (e.g. a negative KERNELAX_JOBS would stall every build)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        if kind is bool:
            return {"0": False, "false": False, "no": False, "off": False, "1": True, "true": True, "yes": True, "on": True}[raw.lower()]
        v = kind(raw)
        if v >= low and (math.isfinite(v) or inf and v > 0):
            return v
    except (KeyError, ValueError):
        pass
    want = "0 or 1 (false or true)" if kind is bool else f"{'an integer' if kind is int else 'a number'} >= {low:g}"
    raise ValueError(f"kernelax: {name}={raw!r}: expected {want}")


CACHE = os.path.abspath(os.path.expanduser(os.environ.get("KERNELAX_CACHE_DIR") or os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache"), "kernelax")))
GCC_MIN = 12  # kernelax' kernels and runtime: gcc >= 12 (the newest on the PATH preferred: gcc 12-14 spill AVX-512
# masks a byte wide and reload them two bytes wide, GCC PR target/117159, whose known pattern the kernels avoid, vec._kjoin,
# and test_f32 checks in their assembly; gcc 15 also builds large kernels faster)


def _major(cc):  # a compiler's major version (-dumpversion), None if it does not run
    try:
        return int(subprocess.run([cc, "-dumpversion"], capture_output=True, text=True, check=True).stdout.split(".")[0])
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _names(d):  # the files of a PATH directory; none where it cannot be read (as the shell's lookup)
    try:
        return os.listdir(d)
    except OSError:
        return []


def _compiler(var, name):
    """The compiler of KERNELAX_CC / KERNELAX_CXX; else the newest of `name` (gcc, g++) and the name-NN on the PATH
    (distributions' gcc-15, ...; `name` where as new), if gcc >= GCC_MIN, else `name` (refused at the first build with
    the reason, _cc_version)."""
    if os.environ.get(var):
        return os.environ[var]
    found = {int(m.group(1)) for d in os.environ.get("PATH", "").split(os.pathsep) for f in _names(d)
             if (m := re.fullmatch(re.escape(name) + r"-(\d+)", f))}
    v, cc = max([(_major(name) or 0, name)] + [(v, f"{name}-{v}") for v in found], key=lambda c: (c[0], c[1] == name))
    return cc if v >= GCC_MIN else name


CC = _compiler("KERNELAX_CC", "gcc")
CXX = _compiler("KERNELAX_CXX", "g++")  # the runtime (runtime.cpp: XLA FFI handler)
BIG, BIG_OPT = 2 << 20, ["-O1", "-ftree-slp-vectorize"]  # kernels of more C than this compile at BIG_OPT: half gcc's
# time, about as fast
LOOP_BIG, LOOP_OPT = 256 << 10, ["-O1", "-fexpensive-optimizations", "-ftree-slp-vectorize"]  # kernels with a loop
# (rolled scans, while loops) of more C than LOOP_BIG (up to BIG) compile at LOOP_OPT: RL rollouts per environment as
# fast or faster (Asterix 43.1 -> 34.5 ms, Snake 69.4 -> 57.4) in less of gcc's time (8.2 -> 4.2 s, 29.2 -> 13.5 s);
# -fexpensive-optimizations: gcc forms the FMAs of -ffp-contract=fast only with it (at plain -O1 a float32 eigenproblem
# rounded otherwise and swapped two eigenvalues, test_f32), but on a large straight kernel it took gcc over 9 minutes
# instead of 34 s (mlp's gradient, 2.8 MB): not for BIG; -ftree-slp-vectorize, the one flag of -O2 that mattered
# (xland-minigrid 88.6 -> 73.5 ms, at -O3 66.3 in 23.5 s of gcc instead of 14.1; Snake, othello, connect_four 3-5 %
# faster; no gcc time); -O2 itself was slower on the others (Asterix 34.4 -> 43.1 ms); smaller kernels stay at -O3
# (benchmarks/run.py at LOOP_OPT: no gcc time saved, kepler, newton_linesearch 10 % slower)
THREEFRY_C = 10_000  # bytes a call of threefry_v counts for (as its hash unrolled: dag.THREEFRY_WEIGHT): the levels by the
# code's size as before (gymnax' Asterix rollout 454 KB unrolled, 199 KB with the calls: -O3 instead of LOOP_OPT, 25 % slower)
FLAGS = ["-O3", "-march=native", "-fno-math-errno", "-ffp-contract=fast", "-fwrapv", "-fopenmp", "-g0", "-shared", "-fPIC",
         "-Wno-psabi", "-fno-tree-pta", "--param=sccvn-max-alias-queries-per-access=100", "--param=dse-max-alias-queries-per-store=32"]
# (-fwrapv: integers wrap as in XLA, int32 / int64 arithmetic of the kernels in signed C types)
# (the last three: gcc's alias analysis grows faster than linearly in a kernel's loads and stores, a third of the time
# of large kernels of index arithmetic; the kernels' memory is restrict pointers and local vectors: no slower code)


def _own_dir(path):
    """The cache directory, created private (0o700). Its libraries are loaded and run: one that another user owns is
    refused, one of this user's that others may write to is made private (a library planted there would run as that
    user)."""
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"kernelax: cannot create the cache directory {path} ({e.strerror}); set "
                           "KERNELAX_CACHE_DIR to a directory of your own") from None
    st = os.stat(path)
    if st.st_uid != os.getuid():
        raise RuntimeError(f"kernelax: the cache directory {path} belongs to another user; set KERNELAX_CACHE_DIR to a "
                           "directory of your own")
    if st.st_mode & 0o022:
        os.chmod(path, stat.S_IMODE(st.st_mode) & ~0o022)
        warnings.warn(f"kernelax: the cache directory {path} was writable by others; made private (kernelax loads "
                      "and runs the libraries in it)")


_own_dir(CACHE)


@functools.cache
def _cpuinfo():
    return pathlib.Path("/proc/cpuinfo").read_text() if os.path.exists("/proc/cpuinfo") else ""


@functools.lru_cache(maxsize=None)
def _cc_version(cc):  # and the CPU flags: -march=native code must not be shared across CPU types
    flags = next((l for l in _cpuinfo().splitlines(True) if l.startswith(("flags", "Features"))), "")  # (x86, aarch64)
    try:
        version = subprocess.run([cc, "--version"], capture_output=True, text=True, check=True).stdout + flags
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"kernelax: the C compiler {cc!r} does not run (KERNELAX_CC, KERNELAX_CXX: gcc and g++ >= {GCC_MIN}): {e}") from None
    path = os.path.realpath(shutil.which(cc) or cc)  # (a wrapper script, e.g. KERNELAX_CC with flags: its text too)
    with contextlib.suppress(OSError):
        head = pathlib.Path(path).read_bytes()[:2048]
        version += f"{path}\0{head.decode(errors='replace') if head.startswith(b'#!') else ''}\0{os.environ.get('CPATH', '')}"
    if (major := _major(cc)) is not None and major < GCC_MIN:
        raise RuntimeError(f"kernelax: {cc!r} is gcc {major}; kernelax builds its kernels with gcc >= {GCC_MIN}: e.g. "
                           f"`conda install -c conda-forge gcc gxx` (no root needed), a module, or the distribution's "
                           f"gcc-NN; KERNELAX_CC and KERNELAX_CXX name it if it is not gcc/g++ or gcc-NN")
    return version


GCC_TIMEOUT = env("KERNELAX_GCC_TIMEOUT", 120.0, float, 0.001)  # seconds; beyond, a kernel is too large (frontend.MAX_NODES
# bounds its operations, not gcc's time: long loop bodies, e.g. local Newton solves unrolled per quadrature point)
BACKGROUND = threading.local()  # .on: builds in a background thread run gcc at low priority, half of JOBS at a time


class Flag:
    """A BACKGROUND.on its owner sets later: a build that becomes a background one while it runs (kernelax.jit's
    first call that stopped waiting for its kernels); each gcc started after reads it."""

    def __init__(self):
        self.on = False

    def __bool__(self):
        return self.on


BACKGROUND_WALL = 10  # background gcc: GCC_TIMEOUT counts its CPU seconds (at idle priority a busy program may hold
# all cores for minutes), and its wall-clock time is limited to this many times GCC_TIMEOUT


class CompileError(RuntimeError):
    """gcc failed (its messages in a log file next to the output, the end of them in the message)."""


_TIMEOUT, _CHRT = shutil.which("timeout"), shutil.which("chrt")  # (coreutils, util-linux; without them: gcc ends with
# the process only through its own kill, background gcc at nice 19 instead of SCHED_IDLE)


class GccTimeout(NotImplementedError):
    """gcc ran out of time (not noted in the index: another limit, or less load, may let it finish). starved: a
    background gcc that did not get the CPU (the program kept the cores busy), not one too slow."""

    def __init__(self, message, starved=False):
        super().__init__(message)
        self.starved = starved


PART_MARK = "\n/*@kernelax-part@*/\n"  # (emit._parts) a kernel's source: its driver, then its parts, an object file each
JOBS = env("KERNELAX_JOBS", 0) or len(os.sched_getaffinity(0))  # gcc processes at a time (_gcc; in the
# background, at low priority: half of them, so that the running program keeps its cores)


_COMPILING = {}  # builds running: their compiler processes (a link's driver source: its path) -> their temporary
# files (stopped and removed at exit: _stop)


@atexit.register  # (registered before auto's wait for the builds at exit, KERNELAX_EXIT_WAIT: runs after it)
def _stop():
    """Compilers still running at exit (KERNELAX_EXIT_WAIT 0 or passed) stopped and their temporary files removed: in
    their own process groups they would run on, and leave them in the cache."""
    for proc, files in list(_COMPILING.items()):
        if isinstance(proc, subprocess.Popen):
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, 9)
        for f in files:
            with contextlib.suppress(OSError):
                os.remove(f)


_SLOTS, _RUNNING = threading.Condition(), [0]  # gcc processes running in this process (kernels build in threads at once)


def _cpu_seconds(pid):
    """CPU seconds of process pid and its descendants (those it reaped included); 0 for a process gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            ticks = sum(map(int, f.read().rsplit(")", 1)[1].split()[11:15]))  # utime, stime, cutime, cstime
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            kids = f.read().split()
    except OSError:
        return 0.0
    return ticks / os.sysconf("SC_CLK_TCK") + sum(_cpu_seconds(int(k)) for k in kids)


def _gcc(cc, args, size):
    """Runs gcc (at low priority in the background builds); a GccTimeout after GCC_TIMEOUT seconds (in the background:
    of its CPU time). At most JOBS gcc processes at a time (in the background: half, so that the running program keeps
    its cores)."""
    background = bool(getattr(BACKGROUND, "on", False))  # (as when this gcc starts)
    with _SLOTS:
        _SLOTS.wait_for(lambda: _RUNNING[0] < (max(1, JOBS // 2) if background else JOBS))
        _RUNNING[0] += 1
    proc = None
    try:
        wall = GCC_TIMEOUT * (BACKGROUND_WALL if background else 1)
        log = (args[args.index("-o") + 1] if "-o" in args else os.path.join(CACHE, f"gcc.{os.getpid()}.{threading.get_ident()}")) + ".log"
        # (its messages: the end of them in a CompileError)
        with open(log, "w") as err:
            try:
                proc = subprocess.Popen([*([_CHRT, "-i", "0"] if background and _CHRT else []),
                                         *([_TIMEOUT, "-k", "5", f"{wall + 10:g}"] if _TIMEOUT else []), cc, *args],
                                        process_group=0, stderr=err)
            except FileNotFoundError:
                raise CompileError(f"kernelax: the C compiler {cc!r} (KERNELAX_CC) was not found") from None
        # (timeout: also a gcc whose process ended without waiting
        # for it, e.g. killed, ends; in its own process group it would run on. Background: SCHED_IDLE, which a waking
        # thread of the program preempts at once; at nice 19 the gcc kept the core for its slice (3 ms), and the
        # kernel's threads waited at their barriers: calls 6-9 instead of 0.3 ms, 20-100 % of them. In the program's
        # session: in its own, its own autogroup, against which neither counts)
        _COMPILING[proc] = [a for a in args if f".{os.getpid()}." in a] + [log]  # (its temporary files: _stop)
        t0 = time.monotonic()
        while True:
            try:
                code = proc.wait(timeout=1.0 if background else GCC_TIMEOUT)
                break
            except subprocess.TimeoutExpired:
                starved = background and time.monotonic() - t0 > wall
                if not background or starved or _cpu_seconds(proc.pid) > GCC_TIMEOUT:
                    os.killpg(proc.pid, 9)  # the whole group: gcc's cc1 too
                    proc.wait()
                    pathlib.Path(log).unlink(missing_ok=True)  # (its messages so far: none of use)
                    raise GccTimeout(f"kernelax: gcc got less than {GCC_TIMEOUT} s of CPU in {wall:g} s for a kernel of "
                                     f"{size // 1024} KB (the program kept the cores busy)" if starved else
                                     f"kernelax: gcc took more than {GCC_TIMEOUT} s for a kernel of {size // 1024} KB: "
                                     "too large (KERNELAX_GCC_TIMEOUT raises the limit)", starved) from None
        if code:
            tail = pathlib.Path(log).read_text(errors="replace")[-3000:]
            missing = code == 127 and not tail.strip() or "No such file or directory" in tail and cc in tail
            raise CompileError(f"kernelax: the C compiler {cc!r} (KERNELAX_CC) was not found" if missing else
                               f"kernelax: {os.path.basename(cc)} failed (exit {code}; all of its messages in {log}):\n{tail}")
        os.remove(log)
    finally:
        _COMPILING.pop(proc, None)
        with _SLOTS:
            _RUNNING[0] -= 1
            _SLOTS.notify_all()


def intact(path):
    """Whether a library or object file (ELF; also a cubin) is whole: its section headers, at its end, inside the file.
    One cut short (a crash, a full disk) would make dlopen crash (SIGBUS) or fail in every process; it is built anew."""
    try:
        with open(path, "rb") as f:
            h, size = f.read(64), os.fstat(f.fileno()).st_size
    except OSError:
        return False
    if len(h) < 64 or h[:4] != b"\x7fELF" or h[4] != 2:  # (64-bit ELF)
        return False
    o = "<" if h[5] == 1 else ">"
    return struct.unpack_from(o + "Q", h, 0x28)[0] + struct.unpack_from(o + "H", h, 0x3A)[0] * struct.unpack_from(o + "H", h, 0x3C)[0] <= size


def _flags(src, extra):
    size = len(src) + THREEFRY_C * (src.count("threefry_v(") - src.count("vi threefry_v("))  # (calls, not the definition)
    opt = BIG_OPT if size > BIG else LOOP_OPT if size > LOOP_BIG and "for (;;)" in src else None
    return [g for f in FLAGS for g in (opt if opt and f.startswith("-O") else [f])] + list(extra)


def _key(src, cc, extra):  # the cache key of src's library or object: also by its flags (the -O level by its size)
    return hashlib.sha256("\0".join([src, cc, *_flags(src, extra), _cc_version(cc)]).encode()).hexdigest()[:20]


def _object(part, cc, extra):
    """The object file of a part of a kernel (emit._parts), cached on disk by its source: a kernel built again after a
    change of its function compiles the parts that changed only."""
    obj = os.path.join(CACHE, "obj", _key(part, cc, extra) + ".o")
    if not intact(obj):
        os.makedirs(os.path.dirname(obj), exist_ok=True)
        tmp_c, tmp = f"{obj}.{os.getpid()}.{threading.get_ident()}.c", f"{obj}.{os.getpid()}.{threading.get_ident()}"
        pathlib.Path(tmp_c).write_text(part)
        try:
            _gcc(cc, [*[f for f in _flags(part, extra) if f != "-shared"], "-c", "-o", tmp, tmp_c], len(part))
        finally:
            os.remove(tmp_c)
        os.replace(tmp, obj)
    return obj


@contextlib.contextmanager
def locked(path):
    """An exclusive lock on the file `path` (fcntl.flock: between processes, e.g. MPI ranks building the same kernel at
    once, and between threads): the first builds, the others wait and then find its result."""
    with open(path, "a") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


@functools.lru_cache(maxsize=None)
def compile_c(src, cc=CC, extra=()):
    """Shared library of `src`, cached on disk by source, compiler, flags and compiler version (and in
    process). A source in parts (PART_MARK: the driver, then the parts) compiles its parts in parallel, each to an
    object file cached by its own source, and links them. Returns (ctypes library, compile seconds, cache hit, path
    of the library). Built once at a time (locked): another process or thread building it is waited for."""
    path = os.path.join(CACHE, _key(src, cc, extra))
    so, c = os.path.join(path, "kernel.so"), os.path.join(path, "kernel.c")
    t0, hit = time.perf_counter(), intact(so)
    if not hit:
        os.makedirs(path, exist_ok=True)
        with locked(os.path.join(path, "lock")):
            hit = intact(so)
            if not hit:  # (none, or one cut short: built anew)
                _link(src, cc, extra, so, c)
    return ctypes.CDLL(so), time.perf_counter() - t0, hit, so


def _link(src, cc, extra, so, c):
    """compile_c's build (temporary files per process and thread)."""
    driver, *parts = src.split(PART_MARK)
    tmp_c, tmp = f"{c}.{os.getpid()}.{threading.get_ident()}.c", f"{so}.{os.getpid()}.{threading.get_ident()}"
    pathlib.Path(tmp_c).write_text(driver)
    _COMPILING[tmp_c] = [tmp_c]  # (while its parts compile; then in gcc's arguments)
    try:
        if parts:
            background = getattr(BACKGROUND, "on", False)

            def build(part):
                BACKGROUND.on = background  # (the pool's threads: as the caller's)
                return _object(part, cc, extra)
            objs = thread_map(build, parts, JOBS)  # (gcc processes at a time: _gcc)
        else:
            objs = []
        _gcc(cc, [*_flags(driver, extra), "-o", tmp, tmp_c, *objs, "-lmvec", "-lm"], len(src))
    except BaseException:
        os.remove(tmp_c)
        raise
    finally:
        _COMPILING.pop(tmp_c, None)
    pathlib.Path(tmp_c).write_text(src)  # (the whole source next to the library: for reading)
    os.replace(tmp_c, c)
    os.replace(tmp, so)


@functools.cache
def simd_width():
    """Lanes of doubles in a register: 8 with AVX-512, 4 with AVX2, 2 with NEON (aarch64), else an error."""
    flags = _cpuinfo()
    if " avx512f" in flags:
        return 8
    if " avx2" in flags and " fma" in flags:
        return 4
    if os.uname().machine == "aarch64" and " asimd" in flags:
        return 2
    raise RuntimeError("kernelax: kernels need a CPU with AVX2/FMA, AVX-512 or (aarch64) NEON")


@functools.cache
def runtime():
    """The runtime library (runtime.cpp): kernelax_run, kernelax_register and the XLA FFI handler kernelax_call."""
    import jax
    import jaxlib
    src = pathlib.Path(__file__).with_name("runtime.cpp").read_text()
    lib = compile_c(src, CXX, ("-std=c++17", "-Wno-return-type", "-static-libstdc++", f"-I{jax.ffi.include_dir()}",
                               f"-DKERNELAX_JAX={jax.__version__}", f"-DKERNELAX_JAXLIB={jaxlib.__version__}"))[0]  # (static libstdc++: a newer g++'s, e.g. from
    # conda-forge, need not be the one the process loads)
    lib.kernelax_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p]
    lib.kernelax_register.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
    lib.kernelax_publish.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    return lib
