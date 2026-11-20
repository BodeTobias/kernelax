# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Jaxpr helpers of kernelax.jit's planner (auto): calls and one-step scans inlined (_inline), library calls expanded
(_library), dead code removed (_dce), values of constants computed at planning time (_fold), equations evaluated on
values (_bind, _run), and the primitive classes the planner's rules know (_EW, _REDUCE, _CUM, _LAYOUT, _NESTED)."""

import threading

import jax
import numpy as np
from jax import lax
from jax.extend import core as jcore

from kernelax.npeval import numpy_bind as _numpy
from kernelax.frontend import _CALLS, _hi_expand, _is_key, _sub_jaxpr


_EW = frozenset("""abs acos acosh add add_any and asin asinh atan atan2 atanh bessel_i0e bessel_i1e bitcast_convert_type cbrt ceil clamp clz
convert_element_type copy cos cosh digamma div eq erf erf_inv erfc exp exp2 expm1 floor ge gt igamma igamma_grad_a
igammac integer_pow is_finite le lgamma log log1p logistic lt max min mul ne neg nextafter not or polygamma polynomial
population_count pow reduce_precision regularized_incomplete_beta rem round rsqrt select_n
shift_left shift_right_arithmetic shift_right_logical sign sin sinh sqrt square stop_gradient sub tan tanh threefry2x32
xor zeta one_minus_square log2 eq_to lt_to le_to real imag complex conj sinpi cospi tanpi asinpi acospi atanpi
atan2pi exp2m1 log2p1 fma fmuladd name random_gamma_grad""".split())  # (bitcast_convert_type between
# equal widths: the rank stays)


_REDUCE = frozenset("reduce_sum reduce_max reduce_min reduce_prod reduce_and reduce_or reduce_xor argmax argmin".split())


_CUM = frozenset("cumsum cumprod cummax cummin cumlogsumexp".split())


_LAYOUT = frozenset("reshape broadcast_in_dim transpose squeeze slice rev pad concatenate copy split stack".split())


class _Const:
    """A value folded at planning time (_fold), in place of a variable: read like a literal (val), hashable."""
    __slots__ = ("val", "aval")

    def __init__(self, val, aval):
        self.val, self.aval = val, aval


_var = lambda v: not isinstance(v, (jcore.Literal, _Const))


def _note_platforms(jaxpr):
    """Notes the values of jaxpr's platform_index equations with a default branch (index of that branch): conds on
    them are jax.lax.platform_dependent's (_control)."""
    reg = _INLINED.__dict__.setdefault("platforms", {})
    for e in jaxpr.eqns:
        if e.primitive.name == "platform_index" and None in e.params["platforms"]:
            reg[e.outvars[0]] = e.params["platforms"].index(None)


def _platform_default(eqn):
    """The branch kernels take of a jax.lax.platform_dependent cond (eqn): its default one, else None (the cond as it
    is). XLA would run the platform's (e.g. a library's C / CUDA kernel through FFI); in a kernel the default branch
    fuses with its neighbours (e.g. a library's JAX branch, fused, was as fast as its CUDA kernel and faster than its C
    kernel; for short calls the trials against XLA decide)."""
    return _INLINED.__dict__.get("platforms", {}).get(eqn.invars[0]) if _var(eqn.invars[0]) else None


_FN = "kernelax per sample"  # a rule's params with this key: the per-sample function itself (where it is not the primitive
# on the per-sample operands, e.g. an axis of length 1 kept)


def _bind(eqn, params, args):  # as jax.core.eval_jaxpr binds (call primitives: their functions from params)
    if _FN in params:
        return list(params[_FN](*args))
    out = eqn.primitive.bind(*args, **eqn.primitive.get_bind_params(params))
    return out if eqn.primitive.multiple_results else [out]


def _run(eqns, env):
    """Binds eqns on env (var -> value), in place."""
    for eqn in eqns:
        env.update(zip(eqn.outvars, _bind(eqn, eqn.params, [v.val if not _var(v) else env[v] for v in eqn.invars])))
    return env


_INLINED = threading.local()  # .cache: id of a closed jaxpr -> (it, inlined), and _per_sample's, while a program is
# planned (_Entry): bodies are planned per candidate sample axis and per level


def _inline(closed):
    """closed (a jaxpr) with calls (jit, custom_jvp/vjp, checkpoint, custom_linear_solve) and scans of one step
    inlined, without dead code."""
    cache = _INLINED.__dict__.setdefault("cache", {})
    if id(closed) not in cache:
        cache[id(closed)] = (closed, _inline_now(closed))
        cache.setdefault(id(cache[id(closed)][1]), (cache[id(closed)][1],) * 2)  # (inlined: stays as it is)
    return cache[id(closed)][1]


def _inline_now(closed):
    if not any(_inlines(e) for e in closed.eqns):  # (nothing to inline: not traced again)
        return _dce(closed)
    spliced = _splice(closed)
    if spliced is not None:
        return _dce(spliced)
    made = {}  # id of a traced value -> (the equation that made it, its operand values, the value): across calls

    pieces = {}

    def piece(x, axis, lo, hi):  # x[lo:hi] along axis, from the pieces of the operands of the elementwise ops and
        key = (id(x), axis, lo, hi)  # broadcasts that made it (e.g. index arithmetic on the piece), each made once
        if key not in pieces:
            pieces[key] = (cut(x, axis, lo, hi), x)
        return pieces[key][0]

    def cut(x, axis, lo, hi):
        e, args, _, j = made.get(id(x), (None, None, None, 0))
        if e is not None and e.primitive.name in _EW and all(np.ndim(y) == 0 or np.shape(y) == np.shape(x) for y in args):
            return _bind(e, e.params, [piece(y, axis, lo, hi) if np.ndim(y) else y for y in args])[j]  # (output j: threefry)
        if e is not None and e.primitive.name == "concatenate" and e.params["dimension"] == axis:  # exactly one of its
            starts = np.cumsum([0] + [np.shape(y)[axis] for y in args])  # operands (or a piece of one): that one
            for y, a, b in zip(args, starts[:-1], starts[1:]):
                if a <= lo and hi <= b:
                    return piece(y, axis, lo - a, hi - a) if (lo, hi) != (a, b) else y
        if e is not None and e.primitive.name == "broadcast_in_dim" and axis in e.params["broadcast_dimensions"]:
            a = e.params["broadcast_dimensions"].index(axis)
            if np.shape(args[0])[a] == np.shape(x)[axis]:
                shape = tuple(hi - lo if d == axis else n for d, n in enumerate(e.params["shape"]))
                return _bind(e, dict(e.params, shape=shape), [piece(args[0], a, lo, hi)])[0]
        return lax.slice_in_dim(x, lo, hi, axis=axis)

    def ev(jaxpr, consts, vals):
        env = dict(zip(jaxpr.constvars, consts)) | dict(zip(jaxpr.invars, vals))
        for eqn in jaxpr.eqns:
            vals = [v.val if not _var(v) else env[v] for v in eqn.invars]
            cat = made.get(id(vals[2]), (None,))[0] if eqn.primitive.name == "scatter-add" else None
            if cat is not None and cat.primitive.name == "concatenate" and \
                    cat.params["dimension"] not in eqn.params["dimension_numbers"].update_window_dims:
                # a scatter-add of concatenated updates: one per piece (an assembly of several sets of entries)
                d, uwd, parts = cat.params["dimension"], eqn.params["dimension_numbers"].update_window_dims, made[id(vals[2])][1]
                axis, out, lo = [x for x in range(np.ndim(vals[2])) if x not in uwd].index(d), vals[0], 0
                for x in parts:
                    out = _bind(eqn, eqn.params, [out, piece(vals[1], axis, lo, lo + np.shape(x)[d]), x])[0]
                    lo += np.shape(x)[d]
                env[eqn.outvars[0]] = out
                continue
            if eqn.primitive.name in _CALLS:
                env.update(zip(eqn.outvars, ev(*_sub_jaxpr(eqn), vals)))
            elif eqn.primitive.name == "call_hi_primitive":  # (its expansion into primitives)
                hi = _hi_expand(eqn)
                env.update(zip(eqn.outvars, ev(hi, hi.consts, vals)))
            elif eqn.primitive.name == "cond" and (src := made.get(id(vals[0]), (None,))[0]) is not None and \
                    src.primitive.name == "platform_index" and None in src.params["platforms"] and \
                    not {"cpu", "cuda", "rocm", "gpu"} & {x for ps in src.params["platforms"] if ps for x in ps}:  # (jax.lax.
                b = eqn.params["branches"][src.params["platforms"].index(None)]  # platform_dependent: on CPUs and GPUs,
                env.update(zip(eqn.outvars, ev(b, b.consts, vals[1:])))  # its default branch)
            elif eqn.primitive.name in _REDUCE and not eqn.params.get("axes", True):  # (over no axis)
                env[eqn.outvars[0]] = vals[0]
            elif (impl := _library(eqn)) is not None:
                env.update(zip(eqn.outvars, ev(impl, impl.consts, vals)))
            elif eqn.primitive.name == "custom_linear_solve":  # its solve
                n, solve = eqn.params["const_lengths"], eqn.params["jaxprs"].solve
                env.update(zip(eqn.outvars, ev(solve, solve.consts, vals[n.matvec + n.vecmat:][:n.solve] + vals[sum(n):])))
            elif eqn.primitive.name == "scan" and eqn.params["length"] == 1:  # one step (lax.map in a single batch)
                consts, carry, xs = map(list, eqn.params["ft_in"].update(vals).unpack())
                out = ev(eqn.params["jaxpr"], eqn.params["jaxpr"].consts, consts + carry + [x[0] for x in xs])
                carry, ys = map(list, eqn.params["ft_out"].update(out).unpack())
                env.update(zip(eqn.outvars, carry + [y[None] for y in ys]))
            else:
                _run([eqn], env)
                made.update((id(env[o]), (eqn, vals, env[o], j)) for j, o in enumerate(eqn.outvars))
        return [v.val if not _var(v) else env[v] for v in jaxpr.outvars]
    return _dce(jax.make_jaxpr(lambda *a: ev(closed, closed.consts, a))(*closed.in_avals))


class _Traced(Exception):
    """An equation _splice cannot inline by substitution: the jaxpr is traced again (_inline_now)."""


def _splice(closed):
    """closed with what _inline_now inlines by substitution, without tracing it again (~30 us per equation): calls,
    custom_linear_solve's solve, jax.lax.platform_dependent's default branch, reductions over no axis, hi primitives'
    and libraries' implementations, their variables renamed; None where an equation needs new ones (a scan of one
    step, a scatter-add of concatenated updates)."""
    constvars, consts, eqns, made = list(closed.constvars), list(closed.consts), [], {}

    def sub(jaxpr, jconsts, args):  # the new atoms of jaxpr's outputs, from those of its inputs (jconsts None: its
        env = dict(zip(jaxpr.invars, args))  # constants kept, the outermost jaxpr's)
        for v, c in zip(jaxpr.constvars, jaxpr.constvars if jconsts is None else jconsts):
            env[v] = v if jconsts is None else jcore.Var(v.aval)
            if jconsts is not None:
                constvars.append(env[v])
                consts.append(c)
        read = lambda v: env[v] if _var(v) else v
        for e in jaxpr.eqns:
            ins, name, p = [read(v) for v in e.invars], e.primitive.name, e.params
            src = made.get(ins[0]) if ins and _var(ins[0]) else None
            if name == "scatter-add" and _var(ins[2]) and (cat := made.get(ins[2])) is not None and \
                    cat.primitive.name == "concatenate" and cat.params["dimension"] not in p["dimension_numbers"].update_window_dims:
                raise _Traced  # (an assembly of several sets of entries: one scatter per piece)
            if name == "scan" and p["length"] == 1:
                raise _Traced
            if name in _CALLS:
                outs = sub(*_sub_jaxpr(e), ins)
            elif name == "call_hi_primitive":
                hi = _hi_expand(e)
                outs = sub(hi, hi.consts, ins)
            elif name == "cond" and src is not None and src.primitive.name == "platform_index" and None in src.params["platforms"] \
                    and not {"cpu", "cuda", "rocm", "gpu"} & {x for ps in src.params["platforms"] if ps for x in ps}:
                b = p["branches"][src.params["platforms"].index(None)]
                outs = sub(b, b.consts, ins[1:])
            elif name in _REDUCE and not p.get("axes", True):
                outs = [ins[0]]
            elif (impl := _library(e)) is not None:
                outs = sub(impl, impl.consts, ins)
            elif name == "custom_linear_solve":
                n, solve = p["const_lengths"], p["jaxprs"].solve
                outs = sub(solve, solve.consts, ins[n.matvec + n.vecmat:][:n.solve] + ins[sum(n):])
            else:
                outs = [v if isinstance(v, jcore.DropVar) else jcore.Var(v.aval) for v in e.outvars]
                eqns.append(e.replace(invars=ins, outvars=outs))
                made.update((o, eqns[-1]) for o in outs)
            env.update(zip(e.outvars, outs))
        return [read(v) for v in jaxpr.outvars]
    j = closed
    try:
        outvars = sub(j, None, list(j.invars))
    except _Traced:
        return None
    return jcore.Jaxpr(constvars, j.invars, outvars, eqns, frozenset().union(*(e.effects for e in eqns)), j.debug_info,
                       consts=consts)


def _inlines(eqn):  # whether _inline_now changes eqn (a call, a hi primitive, a scatter-add of pieces, ...)
    name = eqn.primitive.name
    return name in _CALLS or name in ("call_hi_primitive", "custom_linear_solve", "scatter-add", "platform_index") or \
        name == "scan" and eqn.params["length"] == 1 or name in _REDUCE and not eqn.params.get("axes", True) or \
        _library(eqn) is not None


def _dce(closed):
    """closed without dead code (e.g. the primal value of a gradient, which would close regions)."""
    live, eqns = {v for v in closed.outvars if _var(v)}, []
    for eqn in reversed(closed.eqns):
        if eqn.effects or any(v in live for v in eqn.outvars):
            eqns.append(eqn)
            live.update(v for v in eqn.invars if _var(v))
    return closed.replace(eqns=eqns[::-1])


def _library(eqn):
    """The implementation (a closed jaxpr) of a primitive defined outside jax and kernelax, written in jax (equinox's
    unvmap_any, nonbatchable, while-loop buffers, ...), or None (jax's own: applied by XLA; impl not traceable)."""
    prim, f = eqn.primitive, getattr(eqn.primitive.impl, "func", eqn.primitive.impl)
    if eqn.effects or (getattr(f, "__module__", None) or "jax").split(".")[0] in ("jax", "jaxlib", "kernelax", "functools"):
        return None
    key = (prim, tuple(v.aval for v in eqn.invars), _pkey(eqn.params))
    if key not in _LIBRARY:  # (with the params: kept alive, the ids in the key stay theirs)
        try:
            _LIBRARY[key] = jax.make_jaxpr(lambda *a: prim.impl(*a, **eqn.params))(*(v.aval for v in eqn.invars)), eqn.params
        except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError,
                jax.errors.TracerBoolConversionError, jax.errors.TracerIntegerConversionError):
            _LIBRARY[key] = None, eqn.params  # (an implementation for concrete values only)
    return _LIBRARY[key][0]


_LIBRARY = {}


def _pkey(x):  # a hashable key of params (jaxprs by identity)
    if isinstance(x, dict):
        return tuple(sorted((k, _pkey(v)) for k, v in x.items()))
    if isinstance(x, (tuple, list)):
        return tuple(map(_pkey, x))
    if isinstance(x, jcore.Jaxpr):
        return ("jaxpr", id(x))
    if isinstance(x, (np.ndarray, jax.Array)) and not jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
        x = np.asarray(x)
        return (x.dtype.str, x.shape, x.tobytes())
    return repr(x)


def _flatten(f, args):
    """The jaxpr of f(*args) over the leaves of args, calls inlined, without dead code."""
    return _inline(jax.make_jaxpr(lambda *l: f(*jax.tree.unflatten(jax.tree.structure(args), l)))(*jax.tree.leaves(args)))


_NESTED = ("while", "scan", "cond")


def _needs(eqns, targets, known=()):
    """The equations (in order) that compute `targets` from `known` values and the rest."""
    need, out = {v for v in targets if _var(v)} - set(known), []
    for e in reversed(eqns):
        if any(o in need for o in e.outvars):
            out.append(e)
            need.update(v for v in e.invars if _var(v) and v not in known)
    return out[::-1]


FOLD_MAX = 1 << 18  # constants up to this size, and values computed from them alone, are folded into the kernels


FOLD_WORK = 1 << 22  # (intermediate values of such computations up to this size: computed while planning)


def _fold(closed):
    """closed with its constants, and the values computed from constants alone, as literals (each up to FOLD_MAX
    entries, intermediate values up to FOLD_WORK): kernels hold them as constants (folded into the code: shape
    functions at quadrature points, AD seeds and their products, whose zeros fold) instead of reading them as inputs."""
    j, val = closed, {}
    val.update((v, np.asarray(c, v.aval.dtype)) for v, c in zip(j.constvars, closed.consts)  # (in their avals' dtypes:
               if np.size(c) <= FOLD_WORK and not _is_key(v.aval))  # without x64, NumPy's int64 constants are int32)
    consts = {}
    lit = lambda v: consts.setdefault(v, _Const(val[v], v.aval)) if _var(v) and v in val and val[v].size <= FOLD_MAX else v
    eqns = []  # (equation, computed here: its results too large for literals, kept where read)
    for e in j.eqns:
        ins = [lit(v) for v in e.invars]
        known = (all(not _var(v) or v in val for v in ins) and not e.effects and e.primitive.name not in _NESTED +
                 ("kernelax_call", "kernelax_sample", "platform_index")  # (platform_dependent: kernels take the default branch)
                 and not e.primitive.name.startswith("random") and all(o.aval.size <= FOLD_WORK and not _is_key(o.aval) for o in e.outvars))
        if known:
            args = [val[v] if _var(v) else np.asarray(v.val, v.aval.dtype) for v in ins]
            out = _numpy(e, args)
            if out is None:
                with jax.ensure_compile_time_eval(), jax.enable_x64(True):  # (the jaxpr's dtypes: also 64-bit values of a
                    out = _bind(e, e.params, args)  # program traced without x64, as XLA runs it)
            val.update((o, np.asarray(x, o.aval.dtype)) for o, x in zip(e.outvars, out))
            if all(o.aval.size <= FOLD_MAX for o in e.outvars):
                continue
        eqns.append((e.replace(invars=ins), known))
    if not val:
        return closed
    outvars = [lit(v) for v in j.outvars]
    need, kept = {v for v in outvars if _var(v)}, []
    for e, known in reversed(eqns):  # (computed results no longer read: dropped)
        if not known or any(o in need for o in e.outvars):
            kept.append(e)
            need.update(v for v in e.invars if _var(v))
    eqns = kept[::-1]
    keep = [(v, c) for v, c in zip(j.constvars, closed.consts) if v in need]
    return jcore.Jaxpr([v for v, _ in keep], j.invars, outvars, eqns, j.effects, j.debug_info, consts=[c for _, c in keep])
