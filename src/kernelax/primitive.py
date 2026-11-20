# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""kernelax: a per-sample JAX function as a JAX primitive whose calls run the C kernel through XLA's FFI.

Every call is one primitive (``kernelax_call``) over a batch of samples (the kernel's sample axis):
- batching folds vmap's batch into the sample axis (one call over all samples, nested vmaps included);
- the JVP binds the kernel of ``jax.jvp`` of the per-sample function (linear in the tangents), the
  transpose binds the kernel of ``jax.vjp`` of a linear one (the output cotangents as inputs), so that
  derivatives of any order, forward and reverse, run in C as well;
- the lowering is an FFI custom call (runtime.cpp: kernelax_call) that runs the kernel on XLA's buffers.
"""

import functools
import inspect
import os
import threading

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

from kernelax import gpu, internals
from kernelax.build import runtime
from kernelax.frontend import _is_key, _wide
from kernelax import cpu
from kernelax.cpu import NT, _kind, elements, fixed, indexed, traced
from kernelax.realify import complex_leaves


@functools.cache
def _registered():  # the runtime's FFI handler, registered when a kernel is first lowered for the CPU (import kernelax
    jax.ffi.register_ffi_target("kernelax_call", jax.ffi.pycapsule(runtime().kernelax_call), platform="cpu")  # without
    # g++: exports, GPU kernels and kernelax.jit, which then stays in XLA, work; the runtime's build: 3.4 s, once)


def to_kernel(x):
    """A leaf as kernels read it (cpu._kind): 8-byte dtypes as float64 (64-bit integers as they are: the kernels read
    their bits, exact also beyond 2^53: random bits, keys; XLA's conversion would copy them), float32 as it is, 16-bit
    floats as they are (the C kernels: float32, _impl), other integers as int32 (uint32 as it is)."""
    x = jnp.asarray(x)
    if np.dtype(x.dtype).itemsize == 8:
        return x if _wide(x.dtype) else x.astype(jnp.float64)
    if jax.dtypes.issubdtype(x.dtype, np.floating):
        return x if np.dtype(x.dtype).itemsize == 2 else x.astype(jnp.float32)  # (16-bit floats: as they are, read by
    # the GPU kernels as such; for the C kernels converted in the lowering, _impl)
    return x if x.dtype == jnp.uint32 else x.astype(jnp.int32)


_KIND16 = {np.dtype(jnp.float16): "h", np.dtype(jnp.bfloat16): "b"}  # 16-bit float inputs, read by the GPU kernels as such


def from_kernel(o, dtype, E=8):
    """A kernel's output (see to_kernel) as `dtype`."""
    if E == 4 and not jax.dtypes.issubdtype(dtype, np.floating):
        return jax.lax.bitcast_convert_type(o, jnp.uint32 if dtype == jnp.uint32 else jnp.int32).astype(dtype)
    return o if o.dtype == dtype else jax.lax.bitcast_convert_type(o, dtype) if _wide(dtype) else o.astype(dtype)


def _weak(x, aval):
    """x weakly typed where aval is (a Python scalar or a loop's counter in f: as under jax.vmap and jax.jit)."""
    return internals.weakly_typed(x) if getattr(aval, "weak_type", False) else x


class Spec:
    """A per-sample function f(*leaves) -> tuple of leaves, the per-sample shapes and dtypes of its input
    leaves (avals), per input whether it has a sample axis (batched) or is the same for all samples, the
    inputs f is linear in, the kernel options, and per output whether it is summed over the samples
    (reduce: the cotangents of inputs that are the same for all samples). Specs are cached, so that equal
    ones are identical."""

    def __init__(self, f, avals, batched, linear, opts, reduce=None):
        self.f, self.avals, self.batched, self.linear, self.opts = f, avals, batched, linear, opts
        self.outs = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype, weak_type=a.weak_type) for a in traced(f, avals).out_avals)  # (traced once:
        # also for its index key and the frontend, cpu.traced)
        self.reduce = reduce or (False,) * len(self.outs)

    @functools.cached_property
    def E(self):
        """The bytes of the kernel's elements in memory (cpu.elements), its arrays' dtype."""
        return elements(self.avals, self.outs)

    @property
    def dtype(self):
        return jnp.float32 if self.E == 4 else jnp.float64

    @functools.cached_property
    def out_dtypes(self):
        """The dtypes of the primitive's outputs: the kernel's (dtype), 16-bit floats as they are (written as such by
        the GPU kernels; the C kernels' converted in the lowering) and 64-bit integers (their bits: XLA's conversion
        from doubles would copy them), except sums over the samples."""
        return tuple(o.dtype if (np.dtype(o.dtype) in _KIND16 or _wide(o.dtype)) and not r else self.dtype
                     for o, r in zip(self.outs, self.reduce))

    def _float(self, dtype):
        """The dtype of tangents and cotangents of `dtype` (those of integers: zero): a float's own, else the kernel's
        (so that derived kernels have this kernel's elements)."""
        return dtype if jax.dtypes.issubdtype(dtype, np.floating) else self.dtype

    def shapes(self, n):
        """Output shapes for n samples."""
        return [o.shape if r else (n,) + o.shape for o, r in zip(self.outs, self.reduce)]

    @functools.cached_property
    def kernel(self):  # built when first lowered or run (or loaded from the index on disk): K = 1 for the first call, the
        # other variants in a thread at low priority (Kernel.complete; finished at exit for the index, auto.EXIT_WAIT)
        k = cpu.kernel(*self._args(), cached=True, first=True, nt=self.nt, **self._kw())
        if k.rest[0] or k.whole is not None:
            from kernelax import auto
            t = threading.Thread(target=_complete, args=(k,), daemon=True)
            auto._THREADS.append(t)
            t.start()
        return k

    @functools.cached_property
    def gpu(self):  # the GPU kernel (gpu.py), built when first lowered for a GPU (or loaded from the index on disk)
        return self._gpu(False)

    def _gpu(self, defer):
        mem = (self.E, [_KIND16.get(np.dtype(a.dtype), None) or _kind(a.dtype, True) for a in self.avals],
               [_KIND16[np.dtype(d)] if np.dtype(d) in _KIND16 else _kind(o.dtype) for o, d in zip(self.outs, self.out_dtypes)])
        return gpu.Kernel(*self._args(), self.reduce, mem, dict(self.opts).get("keep_nans", False), defer=defer)

    def start(self, platform="cpu"):
        """Builds the kernel for `platform` (built(platform)); a GPU kernel's nvcc runs on after the emit (its key known:
        the program calling it compiles meanwhile; built() waits for it); a C kernel's K = 1 first (its variants later:
        Kernel.complete)."""
        if platform == "gpu" and "gpu" not in self.__dict__:
            self.__dict__["gpu"] = self._gpu(True)
        elif platform != "gpu" and "kernel" not in self.__dict__:
            self.__dict__["kernel"] = cpu.kernel(*self._args(), cached=True, first=True, nt=self.nt, **self._kw())
        return self.built(platform, wait=False)

    def built(self, platform="cpu", wait=True):
        """The kernel for `platform` (cpu: the C kernel, gpu: the CUDA one), built or loaded."""
        if platform != "gpu":
            return self.kernel
        if wait:
            self.gpu.wait()
        return self.gpu

    def indexed(self, platform="cpu"):
        """Whether the kernel for `platform` is in the index on disk (loads without trace, emit or compile)."""
        if platform == "gpu":
            return "gpu" in self.__dict__ or gpu.indexed(*self._args(), self.reduce, self.E, dict(self.opts).get("keep_nans", False))
        return "kernel" in self.__dict__ or indexed(*self._args(), **self._kw())

    def _args(self):
        return self.f, [np.zeros(a.shape, a.dtype) for a in self.avals], self.batched

    @property
    def nt(self):  # the threads of the kernel's calls (CPU)
        return dict(self.opts).get("n_threads") or NT

    def _kw(self):
        return dict({k: v for k, v in self.opts if k in ("K", "W", "keep_nans")}, reduce=self.reduce)

    @functools.cache
    def variant(self, batched, reduce):
        return Spec(self.f, self.avals, batched, self.linear, self.opts, reduce)

    @functools.cache
    def linear_in(self, wrt):
        """This kernel taken as linear in the inputs wrt (jax.linear_transpose, the matvec of custom_linear_solve)."""
        return Spec(self.f, self.avals, self.batched, tuple(sorted(set(self.linear) | set(wrt))), self.opts, self.reduce)

    @functools.cache
    def jvp(self, mask):
        """The tangents of the inputs in `mask` appended; outputs: the output tangents (linear in them)."""
        idx, n = [i for i, m in enumerate(mask) if m], len(self.avals)

        def g(*a):
            f = lambda *x: self.f(*[x[idx.index(i)] if i in idx else v for i, v in enumerate(a[:n])])
            ts = jax.jvp(f, tuple(a[i] for i in idx), tuple(a[n:]))[1]  # (integer outputs: zero tangents)
            return tuple(jnp.zeros(t.shape, self._float(o.dtype)) if t.dtype == jax.dtypes.float0 else t for t, o in zip(ts, self.outs))
        avals = self.avals + tuple(jax.ShapeDtypeStruct(self.avals[i].shape, self._float(self.avals[i].dtype)) for i in idx)
        return Spec(g, avals, self.batched + tuple(self.batched[i] for i in idx), tuple(range(n, n + len(idx))), self.opts,
                    self.reduce)  # sums commute with the (linear) tangent map

    @functools.cache
    def vjp(self, wrt):
        """The output cotangents appended (per sample; the same for all samples for a summed output); outputs:
        the cotangents of the inputs `wrt` (linear ones of f): per sample, summed over the samples for inputs
        that are the same for all samples (map-reduce in the kernel instead of per-sample arrays)."""
        n = len(self.avals)

        def g(*a):
            p = [jnp.zeros_like(x) if i in self.linear else x for i, x in enumerate(a[:n])]  # f is linear in these
            f = lambda *x: self.f(*[x[wrt.index(i)] if i in wrt else p[i] for i in range(n)])
            cts = tuple(c if jax.dtypes.issubdtype(o.dtype, np.inexact) else np.zeros(o.shape, jax.dtypes.float0)
                        for c, o in zip(a[n:], self.outs))  # (f is linear in wrt: transposed, as jax.vjp would through
            return tuple(jax.linear_transpose(f, *[p[i] for i in wrt])(cts))  # custom_vjp's custom_lin, which it cannot)
        avals = self.avals + tuple(jax.ShapeDtypeStruct(o.shape, self._float(o.dtype)) for o in self.outs)
        return Spec(g, avals, self.batched + tuple(not r for r in self.reduce), tuple(range(n, len(avals))), self.opts,
                    tuple(not self.batched[i] for i in wrt))


def _complete(k):  # the variants of K > 1 of a kernel built with K = 1 first, as kernelax.jit's (build.BACKGROUND)
    os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    cpu.BACKGROUND.on = True
    k.complete()


def _samples(args, spec):
    return next(a.shape[0] for a, b in zip(args, spec.batched) if b)


sample_p = Primitive("kernelax_sample")  # the index of the sample in the kernel's batch: only inside kernels (frontend)
sample_p.def_abstract_eval(lambda: jax.core.ShapedArray((), jax.dtypes.canonicalize_dtype(jnp.int64)))  # (int32 without x64)

call_p = Primitive("kernelax_call")
call_p.multiple_results = True
call_p.def_abstract_eval(lambda *args, spec: [jax.core.ShapedArray(sh, d) for sh, d in zip(spec.shapes(_samples(args, spec)), spec.out_dtypes)])


def _impl(*args, spec):
    args = [a.astype(jnp.float32) if jax.dtypes.issubdtype(a.dtype, np.floating) and np.dtype(a.dtype).itemsize == 2 else a
            for a in args]  # (16-bit floats: float32 for the C kernels)
    n, k = _samples(args, spec), spec.kernel
    _registered()
    out = [jax.ShapeDtypeStruct(sh, d if _wide(d) else spec.dtype) for sh, d in zip(spec.shapes(n), spec.out_dtypes)]
    # (the run-time values: doubles by their bits, which JAX passes as they are also without x64)
    outs = jax.ffi.ffi_call("kernelax_call", out)(*args, k.params.view(np.uint32), key=np.int64(k.key), nt=np.int64(spec.nt),
                                              n=np.int64(n))
    return [o.astype(d) for o, d in zip(outs, spec.out_dtypes)]  # (16-bit float outputs: from float32)


def _impl_gpu(*args, spec):
    n, k = _samples(args, spec), spec.gpu
    out = [jax.ShapeDtypeStruct(sh, d) for sh, d in zip(spec.shapes(n), spec.out_dtypes)]
    return jax.ffi.ffi_call("kernelax_gpu_call", out)(*args, key=np.int64(k.key), n=np.int64(n))


@functools.cache
def _eager(spec):  # a call outside jit, compiled for the device (the lowering by platform), as JAX's dispatch calls a
    return jax.jit(lambda *args: call_p.bind(*args, spec=spec))  # primitive (Specs are cached: equal ones identical)


call_p.def_impl(lambda *args, spec: _eager(spec)(*args))
mlir.register_lowering(call_p, mlir.lower_fun(_impl, multiple_results=True), platform="cpu")
mlir.register_lowering(call_p, mlir.lower_fun(_impl_gpu, multiple_results=True), platform="cuda")


_zero = lambda o: ad.Zero(jax.typeof(o).to_tangent_aval())  # (integers': float0)


def _jvp(primals, tangents, *, spec):
    out = call_p.bind(*primals, spec=spec)
    mask = tuple(type(t) is not ad.Zero and jnp.issubdtype(a.dtype, jnp.inexact) for t, a in zip(tangents, spec.avals))
    if not any(mask):
        return out, [_zero(o) for o in out]
    ts = call_p.bind(*primals, *[t for t, m in zip(tangents, mask) if m], spec=spec.jvp(mask))
    return out, [t if jnp.issubdtype(o.dtype, jnp.inexact) else _zero(o) for o, t in zip(out, ts)]
    # (64-bit integer outputs: float0)


def _transpose(cts, *args, spec):
    wrt = tuple(i for i, a in enumerate(args) if ad.is_undefined_primal(a))
    if not set(wrt) <= set(spec.linear):  # (jax.linear_transpose of the kernel: f linear in these, as jax takes it)
        spec = spec.linear_in(wrt)
    known = [jnp.zeros(a.aval.shape, a.aval.dtype) if ad.is_undefined_primal(a) else a for a in args]
    n = _samples(known, spec)
    vjp = spec.vjp(wrt)  # (the cotangents as the vjp kernel reads them: their declared dtypes, e.g. float32 beside
    cts = [jnp.zeros(sh, a.dtype) if type(c) is ad.Zero else c.astype(a.dtype)  # 8-byte values, whose outputs are doubles)
           for c, sh, a in zip(cts, spec.shapes(n), vjp.avals[len(args):])]
    res = dict(zip(wrt, call_p.bind(*known, *cts, spec=vjp)))
    return [res[i].astype(args[i].aval.dtype) if i in res else None for i in range(len(args))]  # (the inputs' dtypes)


def _batch(args, dims, *, spec):
    """vmap's batch B folded into the sample axis n: (B, n, ...) -> (B n, ...). An unmapped batched input
    of one sample becomes the same for all samples; a mapped one that was the same for all samples
    becomes batched (repeated n times). Summed outputs are summed per batch entry, outside the kernel."""
    B = next(a.shape[d] for a, d in zip(args, dims) if d is not None)
    args = [a if d is None else jnp.moveaxis(a, d, 0) for a, d in zip(args, dims)]
    n = next(a.shape[d is not None] for a, d, b in zip(args, dims, spec.batched) if b)
    new, flags = [], []
    for a, d, b in zip(args, dims, spec.batched):
        mapped = d is not None
        if b and not mapped and n == 1:
            new.append(a[0])
        elif b and not mapped:
            new.append(jnp.broadcast_to(a, (B,) + a.shape).reshape((B * n,) + a.shape[1:]))
        elif b:
            new.append(a.reshape((B * n,) + a.shape[2:]))
        elif mapped:
            new.append(jnp.repeat(a, n, axis=0) if n != 1 else a)
        else:
            new.append(a)
        flags.append(mapped or (b and n > 1))
    outs = call_p.bind(*new, spec=spec.variant(tuple(flags), (False,) * len(spec.outs)))
    outs = [o.reshape((B, n) + o.shape[1:]) for o in outs]
    return [o.astype(d).sum(1) if r else o for o, r, d in zip(outs, spec.reduce, spec.out_dtypes)], [0] * len(outs)


ad.primitive_jvps[call_p] = _jvp
ad.primitive_transposes[call_p] = _transpose
batching.primitive_batchers[call_p] = _batch


def statics(f, static_argnums, static_argnames):
    """Static arguments as jax.jit takes them (hashable values, part of the signature): positions and names completed
    through f's signature, either way. Returns (static_argnums, static_argnames, f's positional parameter names,
    parts), parts(args, kwargs) -> (static positional {position: value}, static named {name: value}, the dynamic
    arguments (args, kwargs), their leaves, the signature: (tree, avals, static items, named items, the static values'
    types: 2 and 2.0 are equal, but trace differently, as in jax.jit))."""
    nums = (static_argnums,) if isinstance(static_argnums, int) else tuple(static_argnums)
    names = (static_argnames,) if isinstance(static_argnames, str) else tuple(static_argnames)
    try:
        params = list(inspect.signature(f).parameters.values())
    except (TypeError, ValueError):
        params = []
    pos = [q.name for q in params if q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
    if nums and not names:  # (as jax.jit: one completed from the other if not both are given; negative positions
        names = tuple(q.name for i, q in enumerate(params) if q.kind == q.POSITIONAL_OR_KEYWORD  # from the end of the
                      and (i in nums or i - len(params) in nums))  # signature, then taken modulo its positional ones)
    elif names and not nums:
        nums = tuple(i for i, q in enumerate(params) if q.kind == q.POSITIONAL_OR_KEYWORD and q.name in names)
    if pos and not any(q.kind == q.VAR_POSITIONAL for q in params):
        nums = tuple(i % len(pos) if i < 0 else i for i in nums)

    def parts(args, kwargs):
        if nums or names:
            static = {i: a for i, a in enumerate(args) if i in nums or i - len(args) in nums}
            named = {k: v for k, v in kwargs.items() if k in names}
            dyn = ([a for i, a in enumerate(args) if i not in static], {k: v for k, v in kwargs.items() if k not in named})
        else:
            static, named, dyn = {}, {}, (list(args), kwargs)
        leaves, tree = jax.tree.flatten(dyn)
        named_items = tuple(sorted(named.items()))
        return static, named, dyn, leaves, (tree, tuple(map(jax.typeof, leaves)), tuple(static.items()), named_items,
                                            tuple(type(v) for v in static.values()) + tuple(type(v) for _, v in named_items),
                                            jax.config.jax_enable_x64)
    return nums, names, pos, parts


def bind(f, n, static, named):
    """f of the dynamic arguments (args, kwargs) of a call with n positional arguments and these static ones."""
    def g(args, kwargs):
        it = iter(args)
        return f(*[static[i] if i in static else next(it) for i in range(n)], **kwargs, **named)
    return g


@functools.cache
def _spec(f, in_tree, avals, opts, fixed=(0, (), ())):
    """The Spec of f on leaves of avals (in_tree: of the dynamic arguments (args, kwargs); fixed: the number of
    positional arguments and the static ones, items of parts); complex leaves in and out as [re, im] on a trailing
    axis (realify.complex_leaves: spec.cin, spec.cout)."""
    h = bind(f, fixed[0], dict(fixed[1]), dict(fixed[2]))
    weak = tuple(getattr(a, "weak_type", False) for a in avals)  # (Python scalars: weakly typed in f, as under jax.jit)
    kin = [internals.key_impl(a.dtype) if _is_key(a) else None for a in avals]  # (typed PRNG keys: as their key data)
    g = lambda *l: h(*jax.tree.unflatten(in_tree, [internals.weakly_typed(x) if w else jax.random.wrap_key_data(x, impl=k)
                                                   if k is not None else x for x, w, k in zip(l, weak, kin)]))
    avals = tuple(jax.ShapeDtypeStruct(a.shape + k.key_shape, np.uint32) if k is not None else a for a, k in zip(avals, kin))
    kout = [internals.key_impl(o.dtype) if _is_key(o) else None for o in jax.tree.leaves(jax.eval_shape(g, *avals))]
    cin = tuple(jnp.iscomplexobj(a) for a in avals)
    cout = tuple(jnp.iscomplexobj(o) for o in jax.tree.leaves(jax.eval_shape(g, *avals)))
    if any(cin) or any(cout):
        g = complex_leaves(h, in_tree, cin)
        avals = tuple(jax.ShapeDtypeStruct(a.shape + (2,), jnp.finfo(a.dtype).dtype) if c else a for a, c in zip(avals, cin))
    spec = Spec(lambda *leaves: tuple(jax.random.key_data(o) if k is not None else o for o, k in zip(jax.tree.leaves(g(*leaves)), kout)),
                avals, (True,) * len(avals), (), opts)
    spec.out_tree, spec.cin, spec.cout, spec.kin, spec.kout = jax.tree.structure(jax.eval_shape(g, *avals)), cin, cout, kin, kout
    return spec


def _opts(n_threads, keep_nans):
    """The options of a kernelax.kernel or kernelax.jit function, as its kernels take them: n_threads, keep_nans, and cpu.FIX_K and
    FIX_W as set when the function is made (tests, benchmarks)."""
    if n_threads is not None and (not isinstance(n_threads, int) or isinstance(n_threads, bool) or not 1 <= n_threads <= os.cpu_count()):
        raise ValueError(f"kernelax: n_threads={n_threads!r}: a number of threads from 1 to the {os.cpu_count()} processors")
    K, W = fixed()
    return tuple((k, v) for k, v in (("K", K), ("W", W), ("n_threads", n_threads), ("keep_nans", bool(keep_nans) or None))
                 if v is not None)


def kernel(f=None, *, static_argnums=(), static_argnames=(), n_threads=None, keep_nans=False):
    """Decorator: f(*args) for one sample (pytrees of arrays) runs as a C kernel. Under ``jax.vmap`` the batch
    becomes the kernel's sample axis (one call, SIMD lanes and threads over the samples); ``jax.jit``,
    ``jax.grad``, ``jax.jvp``, ``jax.jacfwd``/``jacrev`` and higher derivatives run in C too (their
    kernels are built when first needed). float32 values are computed in float32, others in float64; the outputs
    are returned in f's dtypes. Keyword arguments as in jax.vmap (mapped along their leading axis); static_argnums,
    static_argnames: as in jax.jit (hashable values, part of the signature; under an enclosing jax.jit static there
    too). n_threads: threads of a kernel call (CPU), by default OMP_NUM_THREADS, at most the CPUs of the process's affinity,
    else those (cpu.NT). keep_nans: keep x * 0, x - x and 0 / x as operations, so that NaN and inf propagate as in
    IEEE (and XLA); by default they fold to 0, which keeps the code of derivatives small (the zeros of jacfwd's basis
    vectors)."""
    if f is None:
        return functools.partial(kernel, static_argnums=static_argnums, static_argnames=static_argnames, n_threads=n_threads,
                                 keep_nans=keep_nans)
    opts, parts = _opts(n_threads, keep_nans), statics(f, static_argnums, static_argnames)[3]

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if getattr(cpu.KERNEL_TRACE, "on", False):  # (traced for another kernel: f's equations in it)
            return f(*args, **kwargs)
        static, named, dyn, leaves, sig = parts(args, kwargs)
        if not leaves:
            raise ValueError("kernelax: a kernel needs an array argument (its samples)")
        if any(isinstance(v, jax.core.Tracer) for v in [*static.values(), *named.values()]):
            raise ValueError("kernelax: a static argument is traced (an enclosing jax.jit must take it as static too)")
        try:
            hash(sig[2:])
        except TypeError:
            raise ValueError("kernelax: static arguments must be hashable (as in jax.jit)") from None
        avals = tuple(jax.ShapeDtypeStruct(np.shape(x), jnp.result_type(x), weak_type=jax.typeof(x).weak_type) for x in leaves)
        spec = _spec(f, sig[0], avals, opts, (len(args),) + sig[2:])
        if spec.E == 8 and not jax.config.jax_enable_x64:
            raise RuntimeError("kernelax: this kernel's arrays are float64 (64-bit dtypes): enable jax_enable_x64")
        leaves = [jnp.stack([jnp.real(x), jnp.imag(x)], -1) if c else jax.random.key_data(x) if k is not None else x
                  for x, c, k in zip(leaves, spec.cin, spec.kin)]
        outs = call_p.bind(*[to_kernel(x)[None] for x in leaves], spec=spec)
        outs = [_weak(from_kernel(o[0], s.dtype, spec.E), s) for o, s in zip(outs, spec.outs)]
        return jax.tree.unflatten(spec.out_tree, [jax.lax.complex(o[..., 0], o[..., 1]) if c else jax.random.wrap_key_data(o, impl=k)
                                                  if k is not None else o for o, c, k in zip(outs, spec.cout, spec.kout)])
    return wrapped
