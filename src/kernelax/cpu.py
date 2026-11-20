# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""A per-sample function as a compiled kernel over a batch: trace, emit its variants, build, register."""

import atexit
import ctypes
import hashlib
import json
import math
import os
import pathlib
import threading
import time
import weakref

import jax
import jaxlib
import numpy as np
from jax.extend import core as jcore

from kernelax import build as _build_mod
from kernelax.build import BACKGROUND, BIG, CACHE, CC, FLAGS, PART_MARK, GccTimeout, _cc_version, compile_c, intact, locked, no_gc, runtime, simd_width, thread_map
from kernelax.dag import Node, loop_roots, schedule
from kernelax import emit, frontend
from kernelax.emit import emit_map
from kernelax.frontend import TRACING, _flat, trace

TUNE_NODES = 20000  # interleaving candidates K with nodes x K above this are not built (gcc time grows with K)
TUNE_MIN, TUNE_TIME = 4096, 0.06  # K is chosen at the first call with >= TUNE_MIN samples (TUNE_TIME: seconds per
# variant: the choice is kept in the index, see Kernel.persist, and some variants are only sometimes fast)
TUNE_LONG = 0.25  # seconds: a first call longer than this is not repeated for the choice of K, the next calls run the
# other variants, one each (runtime.cpp: kernelax_run; repeated, a call of 60 s on 4096 samples took 14 minutes)
TUNE_SLICE = 0.02  # seconds of a run of the comparison on part of a shorter call's samples (kernelax_run)
FIX_K = None  # interleaved sub-blocks of the kernels of functions made from now on (tests, benchmarks; None: the
# fastest of K = 1..4, measured: Kernel)
FIX_W = None  # lanes of doubles of those kernels' vectors: 4 (AVX2, also on AVX-512 CPUs), 8 (AVX-512), 2 (NEON); None:
# the CPU's (build.simd_width)


def _threads():
    """Threads of a call by default: OMP_NUM_THREADS (its first entry, as OpenMP's), at most the CPUs of the process's
    affinity (the runtime binds one thread to each: runtime.cpp), else those CPUs."""
    cpus, env = len(os.sched_getaffinity(0)), os.environ.get("OMP_NUM_THREADS", "").strip()
    if not env:
        return cpus
    n = env.split(",")[0].strip()
    if not n.isdigit() or int(n) < 1:
        raise ValueError(f"kernelax: OMP_NUM_THREADS={env!r} is not a positive number of threads")
    return min(int(n), cpus)


NT = _threads()  # threads of a call by default (primitive, vmap); the index keeps choices per thread count (K, and
# kernelax.jit's kernels or XLA): the kepler case on 48 threads with K chosen on 8: 1.26 times slower


def fixed():
    """FIX_K and FIX_W, checked: K a positive number, W a width this CPU runs."""
    if FIX_K is not None and (not isinstance(FIX_K, int) or FIX_K < 1):
        raise ValueError(f"kernelax: FIX_K={FIX_K!r} is not a positive number of sub-blocks")
    if FIX_W is not None and FIX_W not in {8: (4, 8), 4: (4,), 2: (2,)}[simd_width()]:
        raise ValueError(f"kernelax: FIX_W={FIX_W!r}: this CPU runs vectors of {simd_width()} doubles"
                         + (" or 4 (AVX2)" if simd_width() == 8 else ""))
    return FIX_K, FIX_W


class _Struct(ctypes.Structure):  # kernelax_kernel in runtime.cpp
    _fields_ = [("nvar", ctypes.c_int), ("chosen", ctypes.c_int), ("tune_min", ctypes.c_int64),
                ("tune_time", ctypes.c_double), ("tune_long", ctypes.c_double), ("tune_slice", ctypes.c_double),
                ("fn", ctypes.c_void_p * 4), ("calls", ctypes.c_int64), ("ns", ctypes.c_int64), ("last_n", ctypes.c_int64),
                ("chosen_nt", ctypes.c_int64), ("per", ctypes.c_double * 4), ("runs", ctypes.c_int64), ("work", ctypes.c_int64),
                ("next", ctypes.c_int)]


_KERNELS = {}  # by key: kept alive while the runtime's registry points to them


class Kernel:
    """The kernel of f(*example) over a batch (emit.emit_map): batched[k] tells whether input leaf k has a
    sample axis. Variants K = 1..4 (those with nodes x K <= TUNE_NODES), or the given K; the runtime picks
    one at the first call with >= TUNE_MIN samples (kernelax_run). Built once per source (kernel()). `inputs`,
    `outs` and `sources` are None for a kernel loaded from the index (kernel(cached=True))."""

    def __init__(self, key, libs, paths, Ks, out_tree, params, inputs=None, outs=None, sources=None, compile_time=0.0, index=None):
        self.key, self.libs, self.paths, self.Ks, self.out_tree = key, libs, paths, Ks, out_tree
        self.params = np.asarray(params, np.float64)  # run-time values (emit.emit_map): passed after the inputs
        self.inputs, self.outs, self.sources, self.compile_time, self.index = inputs, outs, sources, compile_time, index
        fns = [ctypes.cast(lib.kernel, ctypes.c_void_p).value for lib in self.libs]
        self.struct = _Struct(len(Ks), 0 if len(Ks) == 1 else -1, TUNE_MIN, TUNE_TIME, TUNE_LONG, TUNE_SLICE,
                              (ctypes.c_void_p * 4)(*fns))
        self.saved, self.rest = len(Ks) == 1, ((), ())  # rest: (Ks, sources) of variants to build later (complete)
        self.whole = None  # (sources by K, run-time values) of the whole file of a kernel built in parts first (complete)
        runtime().kernelax_register(key, ctypes.byref(self.struct))

    def __call__(self, nt, n, ins, outs):
        """Runs the kernel on n samples: inputs and outputs, C-contiguous arrays of its elements (float64 or float32,
        elements)."""
        arrays = list(ins) + [self.params] + list(outs)
        table = (ctypes.c_void_p * len(arrays))(*[a.ctypes.data for a in arrays])
        runtime().kernelax_run(ctypes.byref(self.struct), nt, n, table)

    def stats(self, reset=False):
        """(calls through XLA, seconds in them, samples of the last call) since the start or the last reset."""
        out = (self.struct.calls, self.struct.ns * 1e-9, self.struct.last_n)
        if reset:
            self.struct.calls = self.struct.ns = 0
        return out

    def complete(self):
        """Builds the variants a background build left for later (it builds K = 1 first: the program runs on the kernel
        sooner), adds them (the runtime chooses K anew at its next call with >= TUNE_MIN samples) and enters the
        kernel in the index. A gcc timeout keeps the variants built so far. A kernel built in parts first gets its
        whole file instead (_upgrade). A gcc starved by the program (build.GccTimeout.starved) leaves the kernel out
        of the index: a later process builds the rest again (gcc's results come from the disk cache)."""
        starved = False
        try:
            if self.whole is not None and self._upgrade():
                return
        except GccTimeout:
            starved = True
        with _LOCK:  # (taken once: another thread completing the same kernel finds nothing left)
            (Ks, sources), self.rest = self.rest, ((), ())
        built = []
        for src in sources if not starved else ():
            try:
                built.append(compile_c(src))
            except GccTimeout as e:
                starved = e.starved
                break
        if built:
            for i, b in enumerate(built, len(self.Ks)):
                self.struct.fn[i] = ctypes.cast(b[0].kernel, ctypes.c_void_p).value
            self.libs, self.paths, self.Ks = self.libs + [b[0] for b in built], self.paths + [b[3] for b in built], self.Ks + Ks[:len(built)]
            runtime().kernelax_publish(ctypes.byref(self.struct), len(self.Ks), -1)  # (after fn: the runtime reads them)
            self.saved = False
        if self.index and not starved:
            _update(self.index, lambda e: {"key": self.key, "so": self.paths, "Ks": self.Ks, "K": self._ks(e), "params": self.params.tolist()})

    def _upgrade(self):
        """The whole file, all variants, of a kernel built in parts first (a large block kept whole only for its reads
        from the frame, emit.FRAME_MAX: in parts gcc compiles it in parallel, ready much sooner; whole it can be
        faster), in place of the parts (the runtime takes them at its next call; results may differ from the parts' in
        the last bits: products and sums fused across part boundaries); entered in the index. False if gcc ran out of
        time: the parts stay (a starved gcc's GccTimeout passes on)."""
        (sources, params), self.whole = self.whole, None
        try:
            built = [compile_c(src) for src in sources.values()]
        except GccTimeout as e:
            if e.starved:
                raise
            return False
        n = len(self.Ks)
        for i, b in enumerate(built):
            self.struct.fn[i] = ctypes.cast(b[0].kernel, ctypes.c_void_p).value
        self.libs, self.paths, self.Ks, self.rest = self.libs + [b[0] for b in built], [b[3] for b in built], list(sources), ((), ())
        runtime().kernelax_publish(ctypes.byref(self.struct), len(self.Ks), -1 if len(self.Ks) != n else self.struct.chosen)
        self.saved = False  # (the parts' libraries stay loaded: a call may still run them)
        if self.index:
            _update(self.index, lambda e: {"key": self.key, "so": self.paths, "Ks": self.Ks, "K": self._ks(e), "params": params.tolist()})
        return True

    def never_chooses(self):
        """Whether its calls have too few samples for a choice of K (the runtime chooses at TUNE_MIN): no K to store."""
        return self.struct.chosen < 0 and 0 < self.struct.last_n < TUNE_MIN

    def K(self):
        """The chosen K (None before it is chosen)."""
        return self.Ks[self.struct.chosen] if self.struct.chosen >= 0 else None

    def chosen(self, entry=None):
        """{threads: K} for the index: the choices of entry, with this process's (by the threads of the call that
        chose it)."""
        return (entry or {}).get("K", {}) | ({str(self.struct.chosen_nt): self.K()} if self.struct.chosen_nt and self.K() else {})

    def _ks(self, entry):  # the choices of K for an entry of this kernel's variants: those of entry that it has, and this
        return {t: k for t, k in self.chosen(entry).items() if k in self.Ks}  # process's

    def persist(self):
        """Stores the chosen K in the index entry, so that later processes with as many threads load that variant only
        (no choice)."""
        if not self.saved and self.index and self.K() is not None and _read(self.index) is not None:
            _update(self.index, lambda e: dict(e, K=self._ks(e)) if e is not None and "error" not in e else e)  # (no entry
            self.saved = True  # yet: its variants are still being built, complete enters it, then with K)


@atexit.register
def _persist():  # the K chosen in this process into the index (kernelax.jit stores it after its calls; kernelax.kernel's
    for k in list(_KERNELS.values()):  # kernels: here)
        try:
            k.persist()
        except OSError:  # (an index that cannot be written: kernelax.jit warned)
            pass


_LOCK = threading.RLock()  # the frontend's DAG is global: one trace at a time (kernels may build in a thread)
_INDEX = os.path.join(CACHE, "index")
_VERSION = hashlib.sha256(b"".join(pathlib.Path(__file__).with_name(n).read_bytes() for n in
                                   sorted(os.listdir(os.path.dirname(__file__))) if n.endswith((".py", ".cpp", ".cu")))
                           + f"jax {jax.__version__} {jaxlib.__version__}".encode()).hexdigest()  # (kernelax' sources and
# jax': expansions of primitives come from jax (internals.jax_function))


def _read(key):
    """An index entry, None if there is none (or it cannot be read: e.g. cut short by a crash; written anew)."""
    path = os.path.join(_INDEX, key + ".json")
    try:
        entry = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return None
    return entry if isinstance(entry, dict) else None  # (another shape: damaged, as none)


def _update(key, change):
    """The entry change(old entry or None), written under a lock on it (processes and threads changing it at once)."""
    os.makedirs(_INDEX, exist_ok=True)
    with locked(os.path.join(_INDEX, key + ".lock")):
        _write(key, change(_read(key)))


def _put(key, entry):
    """A new entry, under its lock as _update (not between another process's read and write of it)."""
    _update(key, lambda _: entry)


def _write(key, entry):
    os.makedirs(_INDEX, exist_ok=True)
    path = os.path.join(_INDEX, key + ".json")
    with open(f"{path}.{os.getpid()}.{threading.get_ident()}", "w") as f:
        json.dump(entry, f)
    os.replace(f.name, path)


_TRACED, _KEYS = weakref.WeakKeyDictionary(), weakref.WeakKeyDictionary()  # f -> {its signature: its jaxpr} and
# {_index_key's arguments: the key}: a kernel's function is traced for its output shapes (Spec), its index key (asked
# for twice: whether it is indexed, then loading it) and the frontend; once


_MEMO = threading.RLock()  # (the plan and kernel builds run in threads at once)


def _memo(table, f, key, make):
    with _MEMO:
        try:
            per = table.setdefault(f, {})
        except TypeError:  # (not weakly referenceable: not kept)
            per = None
        if per is not None and key in per:
            return per[key]
    value = make()  # (outside the lock: it traces the user's function, which may run a kernel and take _LOCK)
    if per is None:
        return value
    with _MEMO:
        return per.setdefault(key, value)


KERNEL_TRACE = threading.local()  # .on: a function traced for a kernel (kernelax.kernel calls in it: f inlined)


def traced(f, example):
    """f's jaxpr on example (arrays or avals), traced once per function and signature."""
    sig = (repr([(tuple(x.shape), str(x.dtype), getattr(x, "weak_type", False)) for x in example]), jax.config.jax_enable_x64)

    def make():
        was = getattr(KERNEL_TRACE, "on", False)
        KERNEL_TRACE.on = True
        try:
            with TRACING(True):  # (kernelax.tracing())
                return jax.make_jaxpr(f)(*example)
        finally:
            KERNEL_TRACE.on = was
    return _memo(_TRACED, f, sig, make)


def _index_key(f, example, batched, W, K, reduce, keep_nans=False):
    """The index key of a kernel: f's jaxpr (with its constants), the example's shapes and dtypes, the options,
    kernelax's sources, the compiler, its flags and the CPU's."""
    key = (repr([(np.shape(x), str(np.asarray(x).dtype)) for x in example]), repr(list(batched)), str(W), str(K), repr(reduce),
           keep_nans, jax.config.jax_enable_x64, emit.SCAT_BIG, emit.WIDE_MIN, frontend.MAX_NODES, frontend.ROLL_XS, frontend.LAZY_KEYS,
           _VERSION, frontend.numerics())
    return _memo(_KEYS, f, key, lambda: _index_key_now(f, example, batched, W, K, reduce, keep_nans))


def _index_key_now(f, example, batched, W, K, reduce, keep_nans=False):
    closed = traced(f, [np.asarray(x) for x in example])
    h = hashlib.sha256("\0".join([str(closed), repr([(np.shape(x), str(np.asarray(x).dtype)) for x in example]),
                                  repr(list(batched)), repr(reduce), str(W), str(K), str(keep_nans), _VERSION, _cc_version(CC), *FLAGS,
                                  str(emit.SCAT_BIG), str(emit.WIDE_MIN), str(frontend.MAX_NODES), str(frontend.ROLL_XS),
                                  str(frontend.LAZY_KEYS), frontend.numerics()]).encode())
    _hash_consts(h, closed)
    return h.hexdigest()[:24], closed


def _hash_consts(h, closed):
    """The constants of closed and of the jaxprs nested in it (a jitted helper's closure, a loop body's) into the
    hash h: the printed jaxpr does not show them."""
    for c in closed.consts:
        if isinstance(c, jax.core.Tracer):  # (a value of the trace itself, e.g. closed over by equinox's closure
            h.update(str(c.aval).encode())  # conversion: computed by the program, which the key holds)
            continue
        a = np.asarray(jax.random.key_data(c) if jax.dtypes.issubdtype(getattr(c, "dtype", np.float64), jax.dtypes.prng_key) else c)
        h.update(f"{a.dtype}{a.shape}".encode() + a.tobytes())
    for e in closed.eqns:
        for c in jcore.jaxprs_in_params(e.params):
            _hash_consts(h, c)


def indexed(f, example, batched, W=None, K=None, reduce=None, keep_nans=False):
    """Whether kernel(..., cached=True) finds the kernel in the index (and loads it without building, or raises at once
    if it could not be built)."""
    entry = _read(_index_key(f, example, batched, W or simd_width(), K, reduce, keep_nans)[0])
    return bool(entry) and (_failed(entry) if "error" in entry else all(os.path.exists(p) and _in_cache(p) for p in entry["so"]))


def _in_cache(path):  # (libraries are loaded from the cache only: an index entry naming another file is not used)
    return os.path.realpath(path).startswith(os.path.realpath(CACHE) + os.sep)


def _noted(entry, key):  # an index entry's error, as raised again
    path = os.path.join(_INDEX, key)
    return f"{entry['error']} (noted in the index by an earlier call; delete {path}.json to try again)"


def _failed(entry):
    """Whether an index entry's error stands: a gcc timeout only after two (one may come from a busy machine) and while
    the limit is not larger (build.GCC_TIMEOUT)."""
    return "timeout" not in entry or entry["tries"] >= 2 and entry["timeout"] >= _build_mod.GCC_TIMEOUT


def kernel(f, example, batched, W=None, K=None, reduce=None, cached=False, keep_nans=False, first=False, nt=None):
    """Kernel of f over a batch, for per-sample arguments shaped like `example` (see Kernel); outputs j with
    reduce[j] are summed over the samples (emit.emit_map). cached: look the kernel up in the index on disk by
    f's jaxpr first (no trace, emit or compile; with K chosen for the calls' threads nt (default NT), no choice either), and
    enter it there.
    first: K = 1 built now, the other variants left for Kernel.complete (as in a background build). Kernels build in
    threads at once: trace and emit one at a time (the frontend's DAG), gcc in parallel."""
    W = W or simd_width()
    with _LOCK:
        index = closed = entry = None
        if cached:
            index, closed = _index_key(f, example, batched, W, K, reduce, keep_nans)
            entry = _read(index)
            if entry and "error" in entry and _failed(entry):
                raise NotImplementedError(_noted(entry, index))
            if entry and "error" not in entry and all(_in_cache(p) and intact(p) for p in entry["so"]):
                if entry["key"] not in _KERNELS:
                    k = entry.get("K", {}).get(str(nt or NT))
                    pick = [entry["Ks"].index(k)] if k in entry["Ks"] else range(len(entry["Ks"]))
                    try:
                        libs = [ctypes.CDLL(entry["so"][i]) for i in pick]
                    except OSError:  # (a library that does not load: built anew below, compile_c)
                        libs = None
                    if libs is not None:
                        _KERNELS[entry["key"]] = Kernel(entry["key"], libs, [entry["so"][i] for i in pick], [entry["Ks"][i] for i in pick],
                                                        jax.tree.structure(jax.eval_shape(f, *example)), entry["params"], index=index)
                if entry["key"] in _KERNELS:
                    return _KERNELS[entry["key"]]
    try:
        return _build(f, example, batched, W, K, reduce, closed, index, keep_nans, first)
    except NotImplementedError as e:  # (not a kernel: e.g. too large) noted, so that later processes know at once
        if index and not (isinstance(e, GccTimeout) and e.starved):  # (a starved gcc: no try, a later process builds it)
            tries = (entry or {}).get("tries", 0) + 1 if isinstance(e, GccTimeout) else None
            _put(index, {"error": str(e)} | ({"timeout": _build_mod.GCC_TIMEOUT, "tries": tries} if tries else {}))
        raise


def _bytes(dtype):  # (keys: their uint32 key data)
    return 4 if jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key) else np.dtype(dtype).itemsize


def elements(avals, outs):
    """The bytes of a kernel's array elements: 4 (float32; integers as the bits of int32 or uint32: twice the lanes per
    vector, 16 with AVX-512, 8 with AVX2) where all inputs and outputs fit in 4 bytes, else 8 (doubles)."""
    return 4 if all(_bytes(a.dtype) <= 4 for a in list(avals) + list(outs)) else 8


def _kind(dtype, inputs=False):
    """The elements of an array of `dtype` in memory (primitive.to_kernel): d (double; 64-bit integers by their bits;
    inputs only), f (float32), i / u (the bits of int32 / uint32)."""
    if inputs and _bytes(dtype) == 8:
        return "d"
    return "f" if jax.dtypes.issubdtype(dtype, np.floating) else "u" if _bytes(dtype) == 4 and not \
        jax.dtypes.issubdtype(dtype, np.signedinteger) and dtype != bool else "i"


def stats(reset=False):
    """Per kernel of this process (by key): Kernel.stats, the time of its calls through XLA (profiling)."""
    with _LOCK:
        kernels = list(_KERNELS.items())
    return {key: k.stats(reset) for key, k in kernels}


def _nodes(roots):  # nodes of the code, loop bodies included (once)
    order = schedule(roots)
    return len(order) + sum(_nodes(list(loop_roots(n))) for n in order if n.op == "while")


def _trace(f, example, batched, reduce=None, keep_nans=False, closed=None):
    """f traced on `example` for a kernel (under _LOCK: the frontend's DAG, which the next trace clears): the input
    nodes, the outputs, the output tree, and the elements of its arrays in memory, mem = (E, input kinds, output kinds)
    (emit.emit_map, gpu.emit)."""
    inputs, outs, out_tree = trace(f, *example, closed=closed, uniform=[not b for b in batched], reduce=reduce, keep_nans=keep_nans)
    avals, oavals = [jax.typeof(x) for x in jax.tree.leaves(example)], jax.tree.leaves(jax.eval_shape(f, *example))
    return inputs, outs, out_tree, (elements(avals, oavals), [_kind(a.dtype, True) for a in avals], [_kind(a.dtype) for a in oavals])


def _build(f, example, batched, W, K, reduce, closed, index, keep_nans=False, first=False):
    """Traces and emits the kernel (under _LOCK), compiles it (outside: kernels of other threads trace meanwhile;
    kernel(): the index around it)."""
    with _LOCK, no_gc():
        inputs, outs, out_tree, mem = _trace(f, example, batched, reduce, keep_nans, closed)
        nodes = _nodes([x for o in outs for x in _flat(o) if isinstance(x, Node)])
        Ks = [K] if K else [k for k in (1, 2, 3, 4) if k == 1 or nodes * k <= TUNE_NODES]

        def emitted(frame_max):
            with _LOCK, no_gc():
                sources, params = zip(*[emit_map(inputs, outs, batched, W, k, reduce, mem, frame_max, group=math.lcm(*Ks))
                                        for k in Ks])  # (the threads' samples in groups of lcm(Ks) W: sums the same for each)
            assert all(np.array_equal(p, params[0], equal_nan=True) for p in params), "run-time values differ between the variants"
            # (the same sources for another size of the arrays read and summed into: no gcc, see build)
            return sources, params[0], int(hashlib.sha256("\0".join(sources).encode() + params[0].tobytes()).hexdigest()[:15], 16)
        sources, params, key = emitted(True)
        later = 1 if first or getattr(BACKGROUND, "on", False) else len(Ks)  # (first, or in the background: K = 1 now,
        # the others later, one after the other, Kernel.complete; in the background at low priority, see build.BACKGROUND)
        whole = None  # (a large block kept whole only for its reads from the frame: in parts now, gcc compiles them in
        if later == 1 and PART_MARK not in sources[0] and len(sources[0]) > BIG:  # parallel; the whole file follows in
            again = emitted(False)  # Kernel.complete: an 8.4 MB element tangent ready after 13.5 instead of 29.5 s of
            if again[0] != sources:  # gcc)
                assert np.array_equal(again[1][:len(params)], params, equal_nan=True), "the parts' run-time values extend the file's"
                whole, (sources, params, key) = (dict(zip(Ks, sources)), params), again

    def compiled(sources):
        if later == 1:
            return [compile_c(sources[0])]
        return thread_map(compile_c, sources, len(sources))  # (gcc processes in parallel)
    if key not in _KERNELS:  # (a gcc timeout raises NotImplementedError: noted in the index as well)
        t0 = time.perf_counter()
        try:
            built = compiled(sources)
        except GccTimeout:  # one file too large for gcc's time, kept whole for its many reads from the frame
            again = emitted(False)  # (emit.FRAME_MAX): in parts, a slower kernel instead of none
            if again[0] == sources:
                raise
            sources, params, key = again
            built = compiled(sources)
        with _LOCK:
            if key not in _KERNELS:  # (else built by another thread meanwhile, from the same source)
                _KERNELS[key] = Kernel(key, [b[0] for b in built], [b[3] for b in built], Ks[:later], out_tree, params, inputs,
                                       outs, sources, time.perf_counter() - t0, index)
                _KERNELS[key].rest, _KERNELS[key].whole = (Ks[later:], sources[later:]), whole
    with _LOCK:
        if _KERNELS[key].inputs is None:  # (loaded from the index before: now with its trace)
            _KERNELS[key].inputs, _KERNELS[key].outs, _KERNELS[key].sources = inputs, outs, sources
        if index:
            _KERNELS[key].index = index
            if not _KERNELS[key].rest[0]:  # (else entered by complete)
                _put(index, {"key": key, "so": _KERNELS[key].paths, "Ks": _KERNELS[key].Ks, "K": _KERNELS[key].chosen(), "params": params.tolist()})
    return _KERNELS[key]
