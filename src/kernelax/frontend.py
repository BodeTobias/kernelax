# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""jaxpr -> scalar DAG: the per-sample function is traced with jax.make_jaxpr (AD, custom_jvp and
control flow are resolved by JAX) and its jaxpr interpreted over numpy object arrays of DAG nodes."""

import functools
import itertools
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jcore, source_info_util
from jax.extend.random import threefry2x32_p

from kernelax import dag, internals
from kernelax.build import env
from kernelax.npeval import concrete, numpy_bind
from kernelax.dag import (_NP_CMP, _isi, tof32, tof64, iadd, ibit, icmp, imaxmin, ishift, iselect, iwrap, load, table, threefry, to_float, to_int, _NP_UNARY, Node, _first_max, _isb, _leafy, _push, add, b2f, band, bnot, bor,
                      cmp, div, hidden, ipow, is_c, maxmin, mk, mul, neg, pick, powr, schedule, select, sub, unary)
from kernelax.linalg import _cholesky, _eigh, _lu, _qr, _svd, _triangular_solve
from kernelax.realify import _CALLS, _sub_jaxpr, eqx_error, eqx_nan, realify


TRACING = jax.make_user_context(False)  # kernelax traces the function for its kernels (kernelax.tracing()); part of
# JAX's trace context: inner jax.jit functions are traced and cached apart under it


ASKED = [0]  # calls of tracing(): a trace that did not ask holds for XLA as well (auto._Entry: one trace, not two)


def tracing():
    """Whether kernelax traces the running code for its kernels (kernelax.jit, kernelax.kernel): a library may take a
    path that suits kernels there, where XLA wants another, e.g. one ``jax.vmap`` over all elements of a mesh
    instead of ``lax.map`` in batches (a kernel computes one sample at a time anyway; the batches would only cost
    copies of their results). Parts of such a program that stay in XLA run that path too."""
    ASKED[0] += 1
    return TRACING.value


def _adjugate_rule(det, jax_rule):
    """The rule of JAX's closed-form det of 2 x 2 and 3 x 3 matrices: while kernelax traces, d det = sum(cof(a) * da)
    with the cofactors as 2 x 2 minors, the value det's own. JAX's rule solves with an LU factorization
    (_cofactor_solve): 432 instead of 27 nodes for the gradient of a 3 x 3 det, 1575 instead of 9 for its Hessian, which
    is NaN for singular matrices, and 0 for all entries if one is NaN. jax.jit keeps JAX's rule, and so does
    kernelax.jit's XLA program (its cost model counts the LU solve there: auto._Entry._xla_heavy)."""
    def rule(primals, tangents):
        if not (TRACING.value and tracing()):  # (asks only while kernelax traces: jax.jit's traces count as none)
            return jax_rule(primals, tangents)
        (a,), (da,) = primals, tangents
        if a.shape[-1] == 2:
            cof = jnp.stack([jnp.stack([a[..., 1, 1], -a[..., 1, 0]], -1), jnp.stack([-a[..., 0, 1], a[..., 0, 0]], -1)], -2)
        else:
            m = lambda i, j: a[..., i % 3, j % 3]
            cof = jnp.stack([jnp.stack([m(i + 1, j + 1) * m(i + 2, j + 2) - m(i + 1, j + 2) * m(i + 2, j + 1)
                                        for j in range(3)], -1) for i in range(3)], -2)
        return det(a), jnp.sum(cof * da, axis=(-2, -1))
    return rule


for _det in internals.closed_form_dets():
    _det.jvp = _adjugate_rule(_det, _det.jvp)


_MOVE = {"reshape", "broadcast_in_dim", "transpose", "squeeze", "slice",
         "dynamic_slice", "dynamic_update_slice", "pad", "concatenate", "gather", "scatter", "rev",
         "copy", "stack", "unstack", "split", "select_n", "tile"}
# operands that select data (must be constant for _move); all others are moved by position code
_INDEX = {"gather": slice(1, 2), "scatter": slice(1, 2), "scatter-add": slice(1, 2), "select_n": slice(0, 1),
          "dynamic_slice": slice(1, None), "dynamic_update_slice": slice(2, None)}


def _ew(fn, *xs):
    if len(xs) < 64:  # np.frompyfunc's limit (gathers from long tables have more operands)
        return np.frompyfunc(fn, len(xs), 1)(*xs)
    b = np.broadcast_arrays(*[np.asarray(x, dtype=object) for x in xs])
    out = np.empty(b[0].shape, dtype=object)
    for i in np.ndindex(out.shape):
        out[i] = fn(*(a[i] for a in b))
    return out


def _dot_general(x, y, dimension_numbers, **_):
    (lc, rc), (lb, rb) = dimension_numbers
    lf = [i for i in range(x.ndim) if i not in lc and i not in lb]
    rf = [i for i in range(y.ndim) if i not in rc and i not in rb]
    xs, ys = np.shape(x), np.shape(y)
    bs = tuple(xs[i] for i in lb)
    k = int(np.prod([xs[i] for i in lc]))
    xt = np.transpose(x, list(lb) + lf + list(lc)).reshape(bs + (-1, 1, k))
    yt = np.transpose(y, list(rb) + rf + list(rc)).reshape(bs + (1, -1, k))
    out = (xt.astype(object) * yt.astype(object)).sum(-1)
    return out.reshape(bs + tuple(xs[i] for i in lf) + tuple(ys[i] for i in rf))


def _mask(v):  # bool value of v (mask, b2f of a mask, or a number != 0)
    return v if _isb(v) else v.args[0] if isinstance(v, Node) and v.op == "b2f" else cmp("ne", v, 0.0)


def _pick(i, opts):
    """opts[i] for a symbolic index or predicate i, clamped to the options as in gather and
    dynamic_slice: index trees with constant leaves are mirrored, a mask selects between two
    options, otherwise a chain over i <= v."""
    if _leafy(i):
        return _push(lambda j: opts[min(max(int(j), 0), len(opts) - 1)], i)
    if _isb(i) or (i.op == "b2f" and len(opts) == 2):
        return select(_mask(i), *opts)
    runs = 1 + sum(not (a is b or is_c(a) and is_c(b) and dag._bits(a) == dag._bits(b)) for a, b in zip(opts, opts[1:]))
    if runs > PICK_MIN and not any(_isb(o) or _isi(o) and not PICK_INTS for o in opts):  # many float options (with
        return pick(i, list(opts))  # PICK_INTS also integer ones): an array read per lane (runs: the selects a chain keeps)
    return functools.reduce(lambda out, v: select(cmp("le", i, v), out, opts[v]), range(len(opts) - 2, -1, -1), opts[-1])


ROUND16 = False  # float16 and bfloat16 values as one node each (tof16, tobf16: the GPU kernels' conversions, gpu.py)
# instead of special.round_float's bit operations
PICK_MIN = 12  # options in more runs of equal neighbours than this: an array (pick), else a chain of selects (folded
# where neighbours are equal; no stores). Was 4 options: xland-minigrid 70.7 -> 11.9 ms (16 options in 10 runs, one
# pick each), Snake 41.5 -> 35.5 (12 runs, 3 picks); pgx 2048's 16 runs read by 3 picks each stay an array (as chains
# 12.1 -> 13.4 ms). GPU kernels: gpu.PICK_MIN
PICK_INTS = False  # integer options too (dag.pick: ipick; the GPU kernels, gpu.py: an array per thread; in the C kernels
# a gather per pick was slower than the selects: jumanji's Tetris 73.6 -> 94.6 ms, Asterix 3 %)


def _select_n(p, *cases):
    return _ew(lambda q, *c: _pick(q, c), p, *cases)


def _ibits(dtype):
    return np.iinfo(dtype).bits, bool(np.issubdtype(dtype, np.signedinteger))


def _convert(x, src, dst):
    """Integers are carried as exact doubles (integer nodes where they need their bits), booleans as masks."""
    if dst == bool:
        return x if src == bool else _ew(_mask, x)
    if src == bool:
        return _ew(b2f, x)
    if np.issubdtype(src, np.integer) and np.issubdtype(dst, np.integer):  # wrapped where the value could change
        (b, sg), (b0, sg0) = _ibits(dst), _ibits(src)
        narrow = b < b0 or (sg0 and not sg) or (b == b0 and not sg0 and sg)
        return _ew(lambda v: iwrap(v, b, sg) if _isi(v) or narrow and not _leafy(v) else
                   _push(lambda c: iwrap(c, b, sg) if narrow else c, v), x)
    if np.issubdtype(src, np.integer):  # to a float
        if np.dtype(src).itemsize == 8 and np.dtype(dst).itemsize < 8:  # (64-bit integers to float32 and narrower: rounded
            return _ew(lambda v: _odd_double(v, src == np.uint64), x)  # once, not through a double)
        return _ew(lambda v: to_float(v, src == np.uint64), x)
    if np.issubdtype(dst, np.integer) and jax.dtypes.issubdtype(src, np.inexact):  # truncation, saturated, NaN: 0 (as XLA;
        # also bfloat16, which numpy does not count as inexact)
        return _ew(lambda v: _push(lambda c: _ftoi(c, dst), v), x)
    return x


def _odd_double(v, unsigned):
    """A 64-bit integer as a double rounded to odd at bit 11 (bits 0-10 kept as one sticky bit): exact, and rounded to
    float32 (or narrower) it gives the integer rounded once to nearest even."""
    if not _isi(v):
        return to_float(v, unsigned)  # (a double already: exact below 2^53)
    below = None if unsigned else icmp("lt", v, 0)
    m = v if unsigned else iselect(below, v, neg(v))  # (|v|; -2^63 as 2^63, unsigned)
    m = iselect(icmp("ge", m, 1 << 53, True), m, ibit("ior", ibit("iand", m, -2048), iselect(icmp("ne", ibit("iand", m, 2047), 0), 0, 2048)))
    d = to_float(m, True)
    return d if unsigned else select(below, d, neg(d))


def _ftoi(c, dst):
    """A float as an integer of dtype dst: truncated, out of range the nearest end, NaN 0 (64 bits: an integer node)."""
    info = np.iinfo(dst)
    if is_c(c):
        return 0 if c != c else int(min(max(np.trunc(c), info.min), info.max)) if abs(c) < 2.0 ** 64 else (info.max if c > 0 else info.min)
    nan, t = cmp("ne", c, c), unary("trunc", c)
    if info.bits < 64 and isinstance(t, Node) and t.t == "s":  # float32: clamped to the last float32 up to the limits,
        hi = np.float32(info.max)  # (int8, int16: the limit itself)
        hi = float(hi if int(hi) == info.max else np.nextafter(hi, np.float32(0)))
        v = tof64(maxmin("min", maxmin("max", select(nan, t, 0.0), float(info.min)), hi))  # then converted (in range,
        return select(cmp("gt", t, hi), v, float(info.max))  # NaN as 0 before: the point code converts it to an integer,
    # gpu._bounds); beyond, the limit (2^31 - 1, 2^32 - 1 are no float32)
    if info.bits < 64:
        return select(nan, maxmin("min", maxmin("max", t, float(info.min)), float(info.max)), 0.0)
    t = tof64(t)  # (64 bits: the limits below compared in double; as float32 they round to 2^63, 2^64)
    tc = select(nan, maxmin("min", maxmin("max", t, -2.0 ** 63), 2.0 ** 64 - 2048), 0.0)  # (converted: no inf, NaN or
    below = lambda v: maxmin("min", v, 2.0 ** 63 - 1024)  # value beyond int64, whose conversion C leaves undefined)
    if info.min < 0:  # int64: the ends as integers (2^63 - 1 is no double)
        i = iselect(cmp("ge", t, 2.0 ** 63), iselect(cmp("lt", t, -2.0 ** 63), to_int(below(tc)), info.min), info.max)
    else:  # uint64: from 2^63 on through the upper half
        hi = cmp("ge", t, 2.0 ** 63)
        i = iselect(hi, to_int(below(maxmin("max", tc, 0.0))), iadd(to_int(below(sub(maxmin("max", tc, 2.0 ** 63), 2.0 ** 63))), -(1 << 63)))
        i = iselect(cmp("ge", t, 2.0 ** 64), i, -1)
    return iselect(nan, i, 0)


_BITS = {"and": (band, np.bitwise_and), "or": (bor, np.bitwise_or), "not": (bnot, np.invert),
         "xor": (lambda a, b: bor(band(a, bnot(b)), band(bnot(a), b)), np.bitwise_xor)}


def _bits(name, dtype, *xs):
    """Logical ops on masks; bitwise ops on integers: folded for constant leaves (failure flags), else on
    integer nodes (not: xor with all ones; the result is wrapped to the dtype in _eval)."""
    fb, fi = _BITS[name]
    if dtype == bool:
        return _ew(fb, *xs)
    op = lambda *c: ibit("i" + name, *c) if name != "not" else ibit("ixor", c[0], -1)
    return _ew(lambda *v: _push(lambda *c: int(fi(*map(int, c))) if all(map(is_c, c)) else op(*c), *v), *xs)


def _shift(name, dtype, a, s):
    """Shifts of integers of `dtype` on their bits (b of them): 0 (logical) or the sign (arithmetic) for s >= b."""
    b, signed = _ibits(dtype)

    def one(a, s):
        a, s = to_int(a), to_int(s)
        big = icmp("ge", s, b, True)  # also negative shifts (as unsigned: huge), as XLA
        s = imaxmin("min", s, 63, True)
        if name == "shift_left":
            return iselect(big, ishift("ishl", a, s), 0)
        if name == "shift_right_arithmetic":  # on the value as signed b-bit integer
            return ishift("ishr", iwrap(a, b, True), s)
        bits = a if b == 64 else iwrap(a, b, False)  # logical: on the unsigned b-bit pattern
        return iselect(big, ishift("ishru" if b == 64 else "ishr", bits, s), 0)
    return _ew(one, a, s)


def _bitcast(x, src, dst):
    """bitcast_convert_type between float64 and (u)int64, float32 and (u)int32, float16 / bfloat16 and (u)int16,
    integers of one width (wrapped in _eval)."""
    if np.dtype(src) == np.dtype(dst):  # (the bits as they are)
        return x
    if np.issubdtype(src, np.integer) and np.issubdtype(dst, np.integer) and np.dtype(src).itemsize == np.dtype(dst).itemsize:
        return _convert(x, src, dst)  # two's complement: the conversion wraps alike
    fl = lambda d: jax.dtypes.issubdtype(d, np.floating)
    if np.dtype(src).itemsize == np.dtype(dst).itemsize == 2 and fl(src) != fl(dst) and np.dtype(src if fl(src) else dst) in _FLOATS:
        from kernelax import special  # (16-bit floats, carried as float32 values: their encodings by integer operations)
        half = np.dtype(src if fl(src) else dst) == np.float16
        fn = (special.f16_bits if half else special.bf16_bits) if fl(src) else (special.f16_from_bits if half else special.bf16_from_bits)
        closed = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(np.shape(x), np.float32 if fl(src) else np.int32))
        out = _eval(closed, closed.consts, [np.asarray(x, dtype=object)])[0]
        return _convert(out, np.dtype(np.int32), np.dtype(dst)) if fl(src) else out  # (wrapped to int16 / uint16)
    one = {(8, True): lambda v: mk("bitsf64", to_int(v)), (8, False): lambda v: mk("fbits", to_float(v)),
           (4, True): lambda v: mk("bitsf32", to_int(v)), (4, False): lambda v: mk("f32bits", tof32(v))}
    if np.dtype(src).itemsize != np.dtype(dst).itemsize or np.dtype(src).itemsize not in (4, 8):
        raise NotImplementedError(f"kernelax: bitcast_convert_type from {src} to {dst} is not supported")
    f = one[(np.dtype(src).itemsize, bool(np.issubdtype(dst, np.inexact)))]
    const = lambda c: np.asarray(c, src).view(dst).item()
    return _ew(lambda v: const(v) if is_c(v) else f(v), x)


def _reduce(fn, a, axes, **_):
    """fn over the last axes after moving `axes` there (one result per remaining index)."""
    a = np.moveaxis(np.asarray(a, dtype=object), list(axes), list(range(-len(axes), 0)))
    rows = a.reshape(a.shape[:a.ndim - len(axes)] + (-1,))
    out = np.empty(rows.shape[:-1], dtype=object)
    for i in np.ndindex(out.shape):
        out[i] = fn(list(rows[i]))
    return out


def _round_away(x):  # round half away from zero
    t = unary("trunc", x)  # (t itself where nothing is added: -0.1 rounds to -0.0)
    return select(cmp("ge", unary("abs", sub(x, t)), 0.5), t, add(t, unary("sign", x)))


def _cond(eqn, vals):
    """lax.cond/switch: a constant index evaluates its branch; otherwise all branches are evaluated
    and selected per output (_pick; NaN/inf of untaken branches do not leak, loops in them run)."""
    branches, i = eqn.params["branches"], np.asarray(vals[0], dtype=object).item()
    if not isinstance(i, Node):
        b = branches[min(max(int(i), 0), len(branches) - 1)]
        return _eval(b, b.consts, vals[1:], keep=True)  # (as a call: sums over the samples stay sums)
    res = [[o.nodes() if isinstance(o, _LAZY) else o for o in _eval(b, b.consts, vals[1:])] for b in branches]
    return [_ew(lambda *r: _pick(i, r), *(np.asarray(r[k], dtype=object) for r in res)) for k in range(len(res[0]))]


def _lu_perm(pivots, *, permutation_size, **_):
    """lu_pivots_to_permutation with symbolic pivots: the row swaps as selects (as in _lu)."""
    piv = np.asarray(pivots, dtype=object)

    def one(pv):
        perm = [float(i) for i in range(permutation_size)]
        for i, p in enumerate(pv):
            pi = perm[i]
            for j in range(i + 1, permutation_size):
                h = _push(lambda x, j=j: cmp("eq", x, float(j)), p)
                perm[i], perm[j] = select(h, perm[i], perm[j]), select(h, perm[j], pi)
        return perm
    out = [one(pv) for pv in piv.reshape(-1, piv.shape[-1])]
    return np.asarray(out, dtype=object).reshape(piv.shape[:-1] + (permutation_size,))


def _tgt(a, b):  # a > b in sort's order: NaN after all numbers
    return bor(cmp("gt", a, b), band(cmp("ne", a, a), cmp("eq", b, b))) if not (_isb(a) or _isi(a) or _isb(b) or _isi(b)) else cmp("gt", a, b)


def _teq(a, b):  # a == b in sort's order: NaNs equal
    return bor(cmp("eq", a, b), band(cmp("ne", a, a), cmp("ne", b, b))) if not (_isb(a) or _isi(a) or _isb(b) or _isi(b)) else cmp("eq", a, b)


def _sort(*ops, dimension, num_keys, **_):
    """sort with symbolic operands (e.g. the inverse pivot permutation of a reverse-mode LU solve): an
    odd-even transposition network that swaps only strictly greater neighbours (stable), the keys
    compared lexicographically."""
    xs = [np.moveaxis(np.asarray(o, dtype=object), dimension, 0).copy() for o in ops]
    n = xs[0].shape[0]
    if n * (n - 1) // 2 * np.size(xs[0][0]) * (3 * num_keys + 2 * len(xs)) > MAX_NODES:  # (n^2 / 2 compare-swaps)
        raise NotImplementedError(f"kernelax: sort of {n} entries ({np.size(xs[0][0])} rows): too large to unroll (an odd-even "
                                  f"network; KERNELAX_MAX_NODES)")
    for r in range(n):
        for i in range(r % 2, n - 1, 2):
            gt = False
            for k in reversed(range(num_keys)):
                gt = _ew(lambda a, b, g: bor(_tgt(a, b), band(_teq(a, b), g)), xs[k][i], xs[k][i + 1], gt)
            for x in xs:
                lo, hi = np.array(x[i], dtype=object), np.array(x[i + 1], dtype=object)
                x[i], x[i + 1] = _ew(select, gt, lo, hi), _ew(select, gt, hi, lo)
    return [np.moveaxis(x, 0, dimension) for x in xs]


def _cum(f):  # cumsum, cumprod, cummax, cummin
    def rule(a, *, axis, reverse, **_):
        a = np.asarray(a, dtype=object)
        a = np.flip(a, axis) if reverse else a
        out = np.frompyfunc(f, 2, 1).accumulate(a, axis=axis, dtype=object)
        return np.flip(out, axis) if reverse else out
    return rule


def _batched(rule, eqn, vals):
    """A matrix rule on the trailing two axes, per index of the leading (batch) axes."""
    vals = [np.asarray(v, dtype=object) for v in vals]
    batch = vals[0].shape[:-2]
    outs = [rule(*(v[i] for v in vals), **eqn.params) for i in np.ndindex(batch)]
    return [np.stack([np.asarray(o[k], dtype=object) for o in outs]).reshape(batch + np.shape(outs[0][k]))
            for k in range(len(outs[0]))]


_LINALG = {"qr": _qr, "lu": _lu, "cholesky": _cholesky, "triangular_solve": _triangular_solve, "eigh": _eigh, "svd": _svd}

@functools.lru_cache(maxsize=None)
def _rounding(shape, dtype, e, m, flush):
    from kernelax import special
    return jax.make_jaxpr(lambda x: special.round_float(x, e, m, flush))(jax.ShapeDtypeStruct(shape, dtype))


def _round(a, dtype, e, m, flush=False):
    """The values of a (nodes of dtype float32 or float64) rounded to a float of e exponent and m mantissa bits: as a
    conversion to it (float16, bfloat16: values of such equations, computed in float32), or as reduce_precision
    (flush: without subnormals)."""
    if (e, m) == (11, 52) or (e, m) == (8, 23) and dtype == np.float32:
        return a
    if ROUND16 and not flush and dtype == np.float32 and (e, m) in ((5, 10), (8, 7)):  # (one node: the GPU's conversion)
        op, np16 = ("tof16", np.float16) if (e, m) == (5, 10) else ("tobf16", jax.dtypes.bfloat16)
        return _ew(lambda v: float(np.asarray(v, np.float32).astype(np16).astype(np.float32)) if is_c(v) else mk(op, v), a)
    if e > 8 and dtype == np.float32:
        raise NotImplementedError(f"kernelax: reduce_precision of float32 to {e} exponent bits is not supported")
    closed = _rounding(np.shape(a), np.dtype(dtype), e, m, flush)
    return _eval(closed, closed.consts, [np.asarray(a, dtype=object)])[0]


def _reduce_window(f, init):
    """reduce_window_max / _min / _sum: the windows (padding and dilations of the base with `init`) folded with f."""
    def rule(a, *, window_dimensions, window_strides, padding, base_dilation, window_dilation, **_):
        a = np.asarray(a, dtype=object)
        pad = init if not any(_isi(x) for x in a.flat) or not np.isinf(init) else -2 ** 63 if init < 0 else 2 ** 63 - 1  # (int64
        b = np.full([(n - 1) * d + 1 if n else 0 for n, d in zip(a.shape, base_dilation)], pad, dtype=object)  # nodes)
        b[tuple(slice(None, None, d) for d in base_dilation)] = a
        b = np.pad(b, [(max(lo, 0), max(hi, 0)) for lo, hi in padding], constant_values=pad)
        b = b[tuple(slice(max(-lo, 0), n - max(-hi, 0)) for (lo, hi), n in zip(padding, b.shape))]
        out = [(n - (w - 1) * d - 1) // s + 1 for n, w, d, s in zip(b.shape, window_dimensions, window_dilation, window_strides)]
        res = None
        for o in np.ndindex(*window_dimensions):
            part = b[tuple(slice(k * d, k * d + (m - 1) * s + 1, s) for k, d, m, s in zip(o, window_dilation, out, window_strides))]
            res = part if res is None else _ew(f, res, part)
        return res
    return rule


_fold = lambda f: lambda a, axes, **_: _reduce(lambda r: functools.reduce(f, r), a, axes)
_pushed = lambda f: lambda *xs, **_: _ew(lambda *v: _push(f, *v), *xs)
_UORDER = frozenset("clamp sort argmax argmin reduce_max reduce_min cummax cummin reduce_window_max reduce_window_min "
                    "scatter-max scatter-min lt_to le_to eq_to".split())  # rules that order their operands (uint64: _uflip)


def _uflip(v, back=False):  # uint64 values as int64 of the same order, u - 2^63 (the top bit flipped), and back
    c = (lambda x: (int(x) + 2 ** 63) % 2 ** 64) if back else (lambda x: int(x) % 2 ** 64 - 2 ** 63)
    return _ew(lambda x: c(x) if is_c(x) and not _isi(x) else ibit("ixor", x, -(1 << 63)), v)


def _jax_own(info, name, src):  # whether a function (its debug info) is JAX's own `name` in jax/_src/`src`
    where = (getattr(info, "func_src_info", None) or "").replace(os.sep, "/")
    return getattr(info, "func_name", None) == name and f"jax/_src/{src}:" in where


def _total_key(x):  # an integer of a value's place in XLA's total order: floats by their bits, the lower 63 flipped
    if _isi(x) or _isb(x) or is_c(x) and not isinstance(x, float):  # below 0 (-NaN < -inf < ... < -0.0 < 0.0 < ... < NaN)
        return x
    if is_c(x):
        b = int(np.float64(x).view(np.int64))
        return b ^ ((b >> 63) & (2 ** 63 - 1))
    b = mk("fbits", tof64(x))  # (float32 values: as doubles, in the same order)
    return ibit("ixor", b, ibit("iand", ishift("ishr", b, 63), 2 ** 63 - 1))


def _logaddexp(x, y):  # as jnp.logaddexp: equal infinities (x - y NaN) give x + y
    d = sub(x, y)
    return select(cmp("ne", d, d), add(maxmin("max", x, y), unary("log1p", unary("exp", neg(unary("abs", d))))), add(x, y))


def _bitreduce(mask, op):  # reduce_and / or / xor: masks (bool) as masks, integers by their bits
    return lambda a, axes, **_: _reduce(lambda r: functools.reduce(
        lambda x, y: mask(x, y) if _isb(x) or isinstance(x, (bool, np.bool_)) else ibit(op, x, y), r), a, axes)


_RULES = {
    "add": _pushed(add), "add_any": _pushed(add), "sub": _pushed(sub), "mul": _pushed(mul),
    "div": lambda a, b, **_: _ew(div, a, b), "neg": lambda a, **_: _ew(neg, a),
    "max": _pushed(lambda x, y: maxmin("max", x, y)), "min": _pushed(lambda x, y: maxmin("min", x, y)),
    "integer_pow": lambda a, y, **_: _ew(lambda x: ipow(x, y), a),
    "square": lambda a, **_: _ew(lambda x: mul(x, x), a),
    "pow": lambda a, b, **_: _ew(powr, a, b),
    "select_n": _select_n, "stop_gradient": lambda a, **_: a, "name": lambda a, **_: a,
    "fma": _pushed(lambda x, y, z: add(mul(x, y), z)), "fmuladd": _pushed(lambda x, y, z: add(mul(x, y), z)),
    "is_finite": lambda a, **_: _ew(lambda x: cmp("lt", unary("abs", x), np.inf), a),
    "reduce_sum": lambda a, axes, **_: np.asarray(a, dtype=object).sum(axis=tuple(axes)),
    "reduce_max": _fold(lambda x, y: maxmin("max", x, y)), "reduce_min": _fold(lambda x, y: maxmin("min", x, y)),
    "reduce_or": _bitreduce(bor, "ior"), "reduce_and": _bitreduce(band, "iand"),
    "argmax": lambda a, axes, **_: _reduce(lambda r: _first_max(r, nan=True), a, axes),
    "argmin": lambda a, axes, **_: _reduce(lambda r: _first_max(r, nan=True, least=True), a, axes),
    "cumsum": _cum(add), "cumprod": _cum(mul), "cummax": _cum(lambda x, y: maxmin("max", x, y)),
    "cummin": _cum(lambda x, y: maxmin("min", x, y)), "reduce_prod": _fold(mul),
    "lu_pivots_to_permutation": _lu_perm, "sort": _sort,
    "one_minus_square": lambda a, **_: _ew(lambda x: mul(add(1.0, x), sub(1.0, x)), a),  # derivative of tanh, as XLA
    "clamp": _pushed(lambda lo, x, hi: maxmin("min", maxmin("max", x, lo), hi)),
    "round": lambda a, rounding_method, **_: _ew(lambda x: unary("rint", x) if int(rounding_method) else _round_away(x), a),
    "rem": _pushed(lambda x, y: float(np.fmod(x, y)) if is_c(x) and is_c(y) else mk("fmod", to_float(x), to_float(y))),
    "atan2": _pushed(lambda x, y: float(np.arctan2(x, y)) if is_c(x) and is_c(y) else mk("atan2", x, y)),
    "dot_general": _dot_general,
    "polynomial": lambda x, *c, **_: _ew(lambda x, *c: functools.reduce(lambda p, a: add(mul(p, x), a), c[-2::-1], c[-1]), x, *c),
    "optimization_barrier": lambda *xs, **_: list(xs), "device_put": lambda *xs, **_: list(xs),
    "reduce_xor": _bitreduce(lambda x, y: bor(band(x, bnot(y)), band(bnot(x), y)), "ixor"),
    "cumlogsumexp": _cum(_logaddexp),
    "reduce_window_max": _reduce_window(lambda x, y: maxmin("max", x, y), -np.inf),
    "reduce_window_min": _reduce_window(lambda x, y: maxmin("min", x, y), np.inf),
    "reduce_window_sum": _reduce_window(add, 0.0),

}
_RULES.update({op: (lambda op: lambda a, **_: _ew(lambda x: unary(op, x), a))(op) for op in _NP_UNARY})
_RULES["abs"] = lambda a, **_: _ew(lambda x: iselect(icmp("lt", x, 0), x, neg(x)) if _isi(x) else unary("abs", x), a)
_RULES.update({op: _pushed((lambda op: lambda x, y: cmp(op, x, y))(op)) for op in _NP_CMP})
_RULES.update({op + "_to": (lambda op: _pushed(lambda x, y: icmp(op, _total_key(x), _total_key(y))))(op)
               for op in _NP_CMP})  # total-order compares (sort, searchsorted; floats: their dispatch in _eval)


def _symbolic(x):
    return isinstance(x, Node) or isinstance(x, np.ndarray) and x.dtype == object


def _library(eqn):  # (jaxprs._library; imported here at the call: jaxprs imports this module)
    from kernelax.jaxprs import _library
    return _library(eqn)


def _norm(value, aval):
    """Object array of the aval's shape, or a plain numeric array if nothing symbolic is left."""
    value = np.asarray(value, dtype=object).reshape(aval.shape)
    if any(isinstance(v, Node) for v in value.flat):
        return value
    return np.asarray(value.tolist(), dtype=aval.dtype).reshape(aval.shape)


def _fill(eqn):
    """The value of a gather in fill mode at out-of-bounds positions (jax's defaults: NaN, the least signed, the
    largest unsigned integer, True)."""
    p, dt = eqn.params, eqn.outvars[0].aval.dtype
    fill = p["fill_value"] if p["fill_value"] is not None else np.nan if jax.dtypes.issubdtype(dt, np.inexact) else \
        True if dt == bool else np.iinfo(dt).min if np.issubdtype(dt, np.signedinteger) else np.iinfo(dt).max
    return bool(fill) if dt == bool else float(fill)


def _np_move(eqn, args):
    """The data movement of eqn on concrete arrays by numpy (no compiled program, launch and copy per call: the trace
    moves codes thousands of times), or None where not covered."""
    p, x = eqn.params, args[0]
    name = eqn.primitive.name
    if name == "reshape" and p.get("dimensions") is None:
        out = np.reshape(x, p["new_sizes"])
    elif name == "broadcast_in_dim":
        bd = p["broadcast_dimensions"]
        out = np.broadcast_to(np.reshape(x, [x.shape[bd.index(d)] if d in bd else 1 for d in range(len(p["shape"]))]), p["shape"])
    elif name == "transpose":
        out = np.transpose(x, p["permutation"])
    elif name == "squeeze":
        out = np.squeeze(x, axis=tuple(p["dimensions"]))
    elif name == "slice":
        out = x[tuple(slice(a, b, c) for a, b, c in zip(p["start_indices"], p["limit_indices"], p["strides"] or (1,) * x.ndim))]
    elif name == "rev":
        out = np.flip(x, axis=tuple(p["dimensions"]))
    elif name == "concatenate":
        out = np.concatenate(args, axis=p["dimension"])
    elif name == "copy":
        out = x
    else:
        return None
    assert out.shape == tuple(eqn.outvars[0].aval.shape), name
    return [out]


def _move(eqn, vals):
    """Data movement on symbolic operands: bind once on position codes, read the sources back (a key's
    position holds its key data)."""
    pool, args = [], []
    index = range(len(vals))[_INDEX.get(eqn.primitive.name, slice(0, 0))]
    filled = eqn.primitive.name == "gather" and eqn.params["mode"] == jax.lax.GatherScatterMode.FILL_OR_DROP
    for i, val in enumerate(vals):
        if i in index:
            args.append(val)
        else:
            a = eqn.invars[i].aval
            val = np.asarray(val, dtype=object)
            items = list(val.reshape((-1,) + val.shape[len(a.shape):])) if _is_key(a) else val.ravel().tolist()
            codes = np.arange(len(pool) + 1, len(pool) + 1 + len(items), dtype=np.float64)
            pool.extend(items)
            args.append(codes.reshape(a.shape))
    outs = _np_move(eqn, args)  # (numpy where it is the same; else the primitive, code -1: out of bounds, the fill value)
    if outs is None:
        outs = eqn.primitive.bind(*args, **(dict(eqn.params, fill_value=-1.0) if filled else eqn.params))
        outs = outs if eqn.primitive.multiple_results else [outs]
    lut = np.empty(len(pool) + 2, dtype=object)
    lut[0] = 0.0
    for j, x in enumerate(pool, 1):
        lut[j] = x
    lut[-1] = _fill(eqn) if filled else None
    result = []
    for out, var in zip(outs, eqn.outvars):
        codes = np.asarray(out)
        assert np.all(codes == np.round(codes)), f"{eqn.primitive.name} is not pure data movement"
        out = lut[codes.astype(np.int64).ravel()]
        result.append(np.stack(list(out)).reshape(_data(var.aval).shape) if _is_key(var.aval) and out.size else out.reshape(codes.shape))
    return result


def _linear(eqn, vals):
    """A primitive that is linear in its data operands, with constant index operands (scatter-add,
    from transposed gathers): each output is the sum of the inputs its 0/1 matrix selects (the
    concrete bind, vmapped over the unit vectors of each data operand; integer data too)."""
    index = range(len(vals))[_INDEX[eqn.primitive.name]]
    data = [i for i in range(len(vals)) if i not in index]
    out = np.full(eqn.outvars[0].aval.size, 0.0, dtype=object)
    for i in data:
        a = eqn.invars[i].aval
        if i == 0 and eqn.primitive.name == "scatter-add":  # (its operand is added as it is: no matrix)
            out = _ew(add, out, np.asarray(vals[0], dtype=object).ravel())
            continue
        if a.size * out.size > 10 ** 8:
            raise NotImplementedError(f"kernelax: {eqn.primitive.name} of {a.shape} into {eqn.outvars[0].aval.shape}: too large to unroll")

        def bind(e):
            args = [np.zeros(eqn.invars[j].aval.shape, eqn.invars[j].aval.dtype) if j in data else v for j, v in enumerate(vals)]
            args[i] = e.reshape(a.shape)
            return eqn.primitive.bind(*args, **eqn.params)
        m = np.asarray(jax.vmap(bind)(np.eye(a.size, dtype=a.dtype))).reshape(a.size, out.size).T
        x = np.asarray(vals[i], dtype=object).ravel()
        for r, c in zip(*np.nonzero(m)):
            out[r] = add(out[r], mul(float(m[r, c]), x[c]))
    return out


def _bilinear(eqn, vals):
    """A primitive bilinear in its two operands (convolution): out[o] = sum of T[i, j, o] x[i] y[j], T bound on all
    pairs of unit vectors (vmapped); a constant operand folds into T's sums."""
    a, b = eqn.invars[0].aval, eqn.invars[1].aval
    if a.size * b.size * eqn.outvars[0].aval.size > 10 ** 7:
        raise NotImplementedError(f"kernelax: {eqn.primitive.name} of {a.shape} and {b.shape}: too large to unroll")
    bind = lambda u, v: eqn.primitive.bind(u.reshape(a.shape), v.reshape(b.shape), **eqn.params)
    T = np.asarray(jax.vmap(lambda u: jax.vmap(lambda v: bind(u, v))(np.eye(b.size, dtype=b.dtype)))(np.eye(a.size, dtype=a.dtype)))
    x, y = (np.asarray(v, dtype=object).ravel() for v in vals)
    out = np.full(eqn.outvars[0].aval.size, 0.0, dtype=object)
    for i, j, o in zip(*np.nonzero(T.reshape(a.size, b.size, -1))):
        out[o] = add(out[o], mul(mul(float(T.flat[(i * b.size + j) * out.size + o]), x[i]), y[j]))
    return out



TABLE_MIN = 9  # constant operands of at least this size indexed by sample data are tables in memory


class _Mem:
    """An input that is the same for all samples, kept in memory instead of one node per entry: the entries of
    input k at `codes` (its flat positions). Data movement changes the codes, a gather with sample indices
    reads it per sample (load), any other use materializes the entries as input nodes. kind: "bool" (an entry is a
    mask), "bits" (a 64-bit integer by its bits), else None. offset: a node, the flat position the codes count from
    (a slice at a symbolic index, e.g. a sample's row of a whole array: read where its entries are, so that a
    further index composes into one load), else None; fills: (out of bounds, the fill value) of such indices in fill
    mode (jnp.take's default), innermost first."""

    def __init__(self, k, codes, kind=None, offset=None, fills=()):
        self.k, self.codes, self.kind, self.offset, self.fills = k, np.asarray(codes, dtype=np.int64), kind, offset, fills
        self.shape, self.size, self.ndim = self.codes.shape, self.codes.size, self.codes.ndim

    def entry(self, node):  # an entry's value from its node (input or load)
        v = _mask(node) if self.kind == "bool" else mk("fbits", node) if self.kind == "bits" else node
        for oob, fill in self.fills:
            v = select(oob, v, fill)
        return v

    def at(self, f):  # the input's flat position of code f
        return f if self.offset is None else add(self.offset, f)

    def nodes(self):
        node = (lambda c: mk("in", self.k, int(c))) if self.offset is None else (lambda c: load(self.k, self.at(float(c))))
        return np.vectorize(lambda c: self.entry(node(c)), otypes=[object])(self.codes) if self.size else \
            np.empty(self.shape, object)


LAZY_KEYS = True  # jax.random.split of a per-sample key into a 1-D array of keys as _Split: each key computed where it
# is read (e.g. per step of a rolled scan over them) instead of all of them in advance


class _Split:
    """The keys of random_split (eqn) of one per-sample threefry key (data: its 2 uint32 nodes) into n keys, not yet
    computed. With threefry_partitionable, key i is threefry2x32(k1, k2, 0, i) (jax' _threefry_split_foldlike). A
    dynamic_slice of one key computes that key alone (at), as does a step of an unrolled scan over them; random_wrap,
    random_unwrap and calls pass it on, loops and scans take it as constant or xs; any other use materializes all
    keys (nodes: the split as it is)."""

    def __init__(self, eqn, vals):
        self.eqn, self.vals, self.data = eqn, vals, np.asarray(vals[0], dtype=object)
        self.shape = tuple(eqn.params["shape"]) + self.data.shape
        self.size, self.ndim, self._nodes = int(np.prod(self.shape)), len(self.shape), None

    @staticmethod
    def applies(eqn, vals):
        impl = internals.key_impl(eqn.invars[0].aval.dtype)
        return LAZY_KEYS and jax.config.jax_threefry_partitionable and impl.name == "threefry2x32" and \
            eqn.invars[0].aval.shape == () and len(eqn.params["shape"]) == 1 and _symbolic(vals[0])

    def same(self, eqn):  # whether eqn (on it) moves no data: e.g. vmap's transposes, per sample the identity (of its
        p, name = eqn.params, eqn.primitive.name  # operand's own shape: a typed key array's has no axis of the data)
        return name == "copy" or name == "transpose" and tuple(p["permutation"]) == tuple(range(len(p["permutation"]))) or \
            name == "reshape" and p.get("dimensions") is None and tuple(p["new_sizes"]) == tuple(eqn.invars[0].aval.shape)

    def nodes(self):
        if self._nodes is None:
            self._nodes = _random(self.eqn, self.vals, lazy=False)[0]
        return self._nodes

    def at(self, i, dtype):  # the key at index i (of dtype; clamped as dynamic_slice), shape (2,)
        return _eval(*_split_at(self.shape[0], np.dtype(dtype)), [self.data, i if _symbolic(i) else np.asarray(i, dtype)])[0]


@functools.lru_cache(maxsize=None)
def _split_at(n, dtype):
    def one(k, i):
        c = jax.numpy.clip(i, 0, n - 1).astype(np.uint32)
        return jax.numpy.stack(threefry2x32_p.bind(k[0], k[1], np.uint32(0), c))
    closed = jax.make_jaxpr(one)(jax.ShapeDtypeStruct((2,), np.uint32), jax.ShapeDtypeStruct((), dtype))
    return closed, closed.consts


_LAZY = (_Mem, _Split)  # (materialized by nodes() where a value is needed as nodes)


DENSE_MIN = 0  # contractions of a matrix in memory (_Mem) of at least this many entries with per-sample values: one node
# "dense" (_dense; a loop over the matrix rows in the kernel) instead of the products unrolled; 0: off (the C kernels),
# the GPU kernels: 64 (gpu.py)


def _dense(eqn, vals):
    """dot_general of a matrix W in memory (DENSE_MIN entries or more, its positions affine in row and column) with
    per-sample nodes X (one contracted axis, no batch axes): `dense` (((input, offset, row step, column step, rows M,
    columns K, columns T of X), the K x T entries of X)) and its results `denseout` (dense, j * T + t), in dot_general's
    output order; else None."""
    (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
    w = next((i for i in (0, 1) if isinstance(vals[i], _Mem)), None)
    if lb or rb or w is None or isinstance(vals[1 - w], _Mem) or vals[w].ndim != 2 or vals[w].size < DENSE_MIN or vals[w].kind or \
            vals[w].offset is not None or vals[w].fills or \
            not np.issubdtype(eqn.outvars[0].aval.dtype, np.floating):
        return None
    (wc,), (xc,) = ((lc, rc)[w], (lc, rc)[1 - w]) if len(lc) == 1 else ((None,), (None,))
    if wc is None:
        return None
    codes = np.moveaxis(vals[w].codes, wc, 1)  # (M, K)
    x = np.moveaxis(np.asarray(vals[1 - w], dtype=object), xc, 0)  # (K, free axes of X...)
    if not all(isinstance(v, Node) for v in x.flat):  # (constants, e.g. the seeds of a jvp: unrolled, they fold)
        return None
    M, K = codes.shape
    c0, sj, sk = int(codes[0, 0]), int(codes[min(1, M - 1), 0] - codes[0, 0]), int(codes[0, min(1, K - 1)] - codes[0, 0])
    if not np.array_equal(codes, c0 + sj * np.arange(M)[:, None] + sk * np.arange(K)):
        return None
    T = int(np.prod(x.shape[1:], dtype=int))
    node = mk("dense", ((vals[w].k, c0, sj, sk, M, K, T), tuple(x.reshape(K * T))))  # (X in the tuple: not traversed
    # by schedule, the kernels compute only the columns they use; dag.hidden)
    out = np.array([mk("denseout", node, i) for i in range(M * T)], dtype=object).reshape((M,) + x.shape[1:])
    return out if w == 0 else np.moveaxis(out, 0, -1)  # (W the right operand: X's free axes first)


_MEM_MOVE = {"reshape", "broadcast_in_dim", "transpose", "squeeze", "slice", "rev", "copy",
             "gather", "dynamic_slice"}  # with constant indices
MEM_MIN = 64  # inputs shared by all samples from this size on are kept in memory (_Mem)


def _tabled(operand, index):
    """A lookup in memory: a constant operand of TABLE_MIN or more entries, indices that are sample data
    (not trees with constant leaves, as LU pivots: those fold into selects)."""
    return not _symbolic(operand) and np.size(operand) >= TABLE_MIN and \
        not any(_leafy(x) for x in np.asarray(index, dtype=object).flat if isinstance(x, Node))


def _table(operand, comps, base, oob=None, fill=None):
    """Lookups in the constant `operand`: `base` holds per output position its flat position with the symbolic
    index components at 0 (the primitive bound on position codes), comps the symbolic components as (index per
    output position, operand dim, largest start); out-of-bounds positions (oob) get `fill`."""
    steps = None  # (the codes' strides where they are affine in the operand's index: a view of an input, read without
    if isinstance(operand, _Mem):  # a table)  an input in memory: its entries at the codes
        codes, op = operand.codes.ravel(), np.empty(operand.shape)
        same = np.array_equal(codes, np.arange(codes.size))
        steps = None if same else _affine(operand.codes)
        if steps is not None:  # (code = codes[base] + steps . index: computed, as for the input itself)
            base = np.where(np.isnan(np.asarray(base, float)), np.nan, codes[np.nan_to_num(np.asarray(base, float)).astype(np.int64)])
            same = True
        codes_t = None if same else tuple(map(float, codes))  # (a permutation: a table of the codes, made once)
        fetch = lambda f: operand.entry(load(operand.k, operand.at(f if same else table(codes_t, f))))
    else:
        op = np.asarray(operand)
        if op.dtype.kind in "iu" and op.dtype.itemsize == 8 and op.size and (np.abs(op.astype(object)) >= 2 ** 53).any():
            values = tuple(np.ascontiguousarray(op).view(np.float64).ravel())  # (64-bit integers beyond doubles:
            fetch = lambda f: mk("fbits", table(values, f))  # their bits)
        else:
            values = tuple(float(v) for v in op.ravel())
            fetch = lambda f: table(values, f)
    strides = np.cumprod((1,) + op.shape[:0:-1])[::-1] if steps is None else np.asarray(steps)
    lim = lambda i, hi: maxmin("min", maxmin("max", i, 0.0), float(hi))
    flat, fold = np.asarray(base, dtype=object), np.full(np.shape(base), None, dtype=object)
    if len(comps) == 1 and not isinstance(operand, _Mem):  # the same value at every index of the range: that constant
        _, d, hi = comps[0]  # (e.g. the structural zeros of AD seeds read at a quadrature point)
        for p in np.ndindex(flat.shape):
            vals = {op.flat[int(flat[p]) + int(strides[d]) * k].item() for k in range(int(hi) + 1)} if flat[p] == flat[p] else ()
            fold[p] = vals.pop() if len(vals) == 1 else None
    one = lambda a: x[0] if len(x := list({id(y): y for y in np.asarray(a, dtype=object).flat}.values())) == 1 else None
    uniform, out_of = [one(i) for i, _, _ in comps], None if oob is None else one(oob)  # (one node at all positions)
    if isinstance(operand, _Mem) and same and comps and None not in uniform and (oob is None or out_of is not None) and \
            not np.isnan(np.asarray(base, float)).any():  # (one symbolic offset for all: lazy)
        offset = functools.reduce(add, [mul(float(strides[d]), lim(x, hi)) for x, (_, d, hi) in zip(uniform, comps)])
        return _Mem(operand.k, np.asarray(base, np.int64), operand.kind, operand.at(offset),
                    operand.fills + (() if oob is None else ((out_of, fill),)))
    for i, d, hi in comps:
        flat = _ew(lambda f, c, x, s=float(strides[d]), hi=hi: f if f != f or c is not None else add(f, mul(s, lim(x, hi))), flat, fold, i)
    out = _ew(lambda f, c: c if c is not None else fill if f != f else fetch(f), flat, fold)  # f != f: constant out of bounds
    if op.dtype == bool:
        out = _ew(lambda x: _mask(x) if isinstance(x, Node) else bool(x), out)
    if oob is not None:
        out = _ew(lambda o, x: select(o, x, fill), oob, out)
    return out


def _affine(codes):
    """The strides of codes (an array of an input's flat positions) per axis if they are affine in the array's index
    (a transpose, broadcast or slice of the input: codes[0] + strides . index), else None."""
    if codes.size == 0:
        return None
    steps = [int(np.take(codes, 1, axis=d).flat[0] - codes.flat[0]) if n > 1 else 0 for d, n in enumerate(codes.shape)]
    at = codes.flat[0] + sum(s * np.arange(n).reshape([-1 if k == d else 1 for k in range(codes.ndim)])
                             for d, (s, n) in enumerate(zip(steps, codes.shape)) if s)
    return steps if np.array_equal(np.broadcast_to(at, codes.shape), codes) else None


def _codes(operand):
    return np.arange(np.size(operand), dtype=np.float64).reshape(np.shape(operand))


def _gather(eqn, vals):
    """Gather with symbolic index components: from a constant table in memory (_tabled), else per candidate
    value of each symbolic component, selected per output position by index == value (index trees with
    constant leaves, as pivot rows of small LU solves, fold, so each position keeps only its few candidates)."""
    operand, idx = vals[0], np.asarray(vals[1], dtype=object)
    dn = eqn.params["dimension_numbers"]
    if isinstance(operand, _Mem) or _tabled(operand, idx):
        p = eqn.params
        sym = [c for c in range(idx.shape[-1]) if any(isinstance(x, Node) for x in idx[..., c].flat)]
        zero = np.where(np.vectorize(lambda x: isinstance(x, Node))(idx), 0, idx).astype(eqn.invars[1].aval.dtype)
        base = eqn.primitive.bind(_codes(operand), zero, **{**p, "fill_value": np.nan})
        out_shape = np.shape(base)
        comps = []
        for c in sym:
            d = dn.start_index_map[c]
            ic = np.broadcast_to(np.expand_dims(idx[..., c], tuple(dn.offset_dims)), out_shape)
            comps.append((ic, d, np.shape(operand)[d] - p["slice_sizes"][d]))
        oob, fill = None, None
        if p["mode"] == jax.lax.GatherScatterMode.FILL_OR_DROP:
            fill = _fill(eqn)
            oob = functools.reduce(lambda a, b: _ew(bor, a, b), [_ew(lambda x, hi=hi: bor(cmp("lt", x, 0.0), cmp("gt", x, float(hi))), i)
                                                                for i, _, hi in comps])
        return _table(operand, comps, np.where(np.isnan(base), np.nan, base) if fill is None else base, oob, fill)
    sym = [c for c in range(idx.shape[-1]) if any(isinstance(x, Node) for x in idx[..., c].flat)]
    c, res = sym[0], []
    d = dn.start_index_map[c]  # (the starts a window can have, clamped, as in _dynamic_slice: _pick clamps the index)
    for v in range(np.shape(operand)[d] - eqn.params["slice_sizes"][d] + 1):  # (several symbolic components: one after the other)
        const = idx.copy()
        const[..., c] = v
        res.append(np.asarray(_gather(eqn, [operand, const]) if len(sym) > 1 else
                              _move(eqn, [operand, const.astype(eqn.invars[1].aval.dtype)])[0], dtype=object))
    ic = np.broadcast_to(np.expand_dims(idx[..., c], tuple(dn.offset_dims)), res[0].shape)
    out = _ew(lambda i, *rv: _pick(i, rv), ic, *res)  # (clamped)
    if eqn.params["mode"] == jax.lax.GatherScatterMode.FILL_OR_DROP:  # (out of bounds: the fill value)
        d = dn.start_index_map[c]
        hi, fill = float(np.shape(operand)[d] - eqn.params["slice_sizes"][d]), _fill(eqn)
        out = _ew(lambda i, o: select(bor(cmp("lt", i, 0.0), cmp("gt", i, hi)), o, fill), ic, out)
    return out


def _dynamic_slice(eqn, vals):
    """dynamic_slice (dynamic_update_slice: e.g. a transposed dynamic_slice) with symbolic starts (e.g.
    a constant table indexed by sample data): one bind per candidate start, selected per output
    position (_pick, clamped like dynamic_slice)."""
    k = 2 if eqn.primitive.name == "dynamic_update_slice" else 1
    starts = vals[k:]
    if k == 1 and (isinstance(vals[0], _Mem) or _tabled(vals[0], [np.asarray(x, dtype=object).item() for x in starts])):  # a
        # constant table, or an input in memory
        zero = [np.asarray(0 if _symbolic(x) else x, eqn.invars[1 + j].aval.dtype) for j, x in enumerate(starts)]
        base = eqn.primitive.bind(_codes(vals[0]), *zero, **eqn.params)
        comps = [(np.broadcast_to(np.asarray(x, dtype=object).item(), np.shape(base)), d,
                  np.shape(vals[0])[d] - eqn.params["slice_sizes"][d]) for d, x in enumerate(starts) if _symbolic(x)]
        return _table(vals[0], comps, base)
    sym = [d for d, x in enumerate(starts) if _symbolic(x)]
    if k == 2:
        return _update_at(vals[0], vals[1], [np.asarray(x, dtype=object).item() for x in starts], sym)
    d, sizes = sym[0], np.shape(vals[1]) if k == 2 else eqn.params["slice_sizes"]
    fix = lambda v: list(vals[:k]) + [np.asarray(v, eqn.invars[k + j].aval.dtype) if j == d else x for j, x in enumerate(starts)]
    res = [_dynamic_slice(eqn, fix(v)) if len(sym) > 1 else _move(eqn, fix(v))[0]  # (several symbolic starts:
           for v in range(np.shape(vals[0])[d] - sizes[d] + 1)]  # one after the other)
    i = np.asarray(starts[d], dtype=object).item()
    return _ew(lambda *rv: _pick(i, rv), *res)


def _update_entries(vals):
    """The entries (flat position, value) of dynamic_update_slice into zeros (_Scatter): the update at its start,
    clamped as XLA's."""
    shape, up = np.shape(vals[0]), np.asarray(vals[1], dtype=object)
    strides = np.cumprod((1,) + shape[:0:-1])[::-1]
    base = 0.0
    for x, m, u, st in zip(vals[2:], shape, up.shape, strides):
        x = np.asarray(x, dtype=object).item()
        base = add(base, mul(float(st), maxmin("min", maxmin("max", to_float(x), 0.0), float(m - u)) if _symbolic(x) else
                             float(min(max(int(x), 0), m - u))))
    return [(add(base, float(np.dot(strides, u))), up[u]) for u in np.ndindex(up.shape)]


def _update_at(operand, update, starts, dims):
    """dynamic_update_slice with the starts along axes `dims` symbolic (the others constant, clamped as XLA's). One
    axis: position by position, the update's entry that the start puts there (a select per entry that can land there,
    at start == position - offset), or its value. Several: the update placed along each of them in turn (per position
    of the axis, a chain over the entries that can land there, at the axis' masks; any value where none does), then a
    select per position on whether it lies in the window (the masks shared by all positions: no nested candidates)."""
    op, up = np.asarray(operand, dtype=object), np.asarray(update, dtype=object)
    lim = [min(max(int(s), 0), m - u) if e not in dims else None for e, (s, m, u) in enumerate(zip(starts, op.shape, up.shape))]
    s = {d: maxmin("min", maxmin("max", to_float(starts[d]), 0.0), float(op.shape[d] - up.shape[d])) for d in dims}
    cands = lambda d, p: range(max(0, p - (op.shape[d] - up.shape[d])), min(up.shape[d], p + 1))  # (entries landing at
    inside = lambda p: all(e in dims or lim[e] <= p[e] < lim[e] + up.shape[e] for e in range(op.ndim))  # p along d)
    out = op.copy()
    if len(dims) == 1:
        d = dims[0]
        for p in filter(inside, np.ndindex(op.shape)):
            q, v = [p[e] - lim[e] if e != d else 0 for e in range(op.ndim)], op[p]
            for j in cands(d, p[d]):
                q[d] = j
                v = select(cmp("eq", s[d], float(p[d] - j)), v, up[tuple(q)])
            out[p] = v
        return out
    w, hit = up, {}
    for d in dims:
        placed = np.empty(w.shape[:d] + (op.shape[d],) + w.shape[d + 1:], dtype=object)
        for p in range(op.shape[d]):
            js = list(cands(d, p))
            masks = [cmp("eq", s[d], float(p - j)) for j in js]
            hit[d, p] = functools.reduce(bor, masks)
            for r in np.ndindex(w.shape[:d] + w.shape[d + 1:]):
                at = lambda j: r[:d] + (j,) + r[d:]
                placed[at(p)] = functools.reduce(lambda v, jm: select(jm[1], v, w[at(jm[0])]), zip(js[1:], masks[1:]), w[at(js[0])])
        w = placed
    for p in filter(inside, np.ndindex(op.shape)):
        q = tuple(p[e] - lim[e] if e not in dims else p[e] for e in range(op.ndim))
        out[p] = select(functools.reduce(band, [hit[d, p[d]] for d in dims]), op[p], w[q])
    return out


def _unbatched(eqn, vals, f):
    """A scatter with batching dimensions (vmap of one at per-sample indices: e.g. the pivots of LU solves per
    quadrature point) as one scatter per batch position (f on its slices): each entry then meets the candidate
    positions of its own slice only, not those of the whole operand."""
    p = eqn.params
    dn = p["dimension_numbers"]
    ob, ib, uw = (list(map(int, d)) for d in (dn.operand_batching_dims, dn.scatter_indices_batching_dims, dn.update_window_dims))
    ub = [[d for d in range(np.ndim(vals[2])) if d not in uw][i] for i in ib]  # (the updates' dims of the index dims)
    less = lambda ds, gone: tuple(int(d) - sum(g < d for g in gone) for d in ds)
    new = dict(p, dimension_numbers=dn._replace(
        update_window_dims=less(uw, ub), inserted_window_dims=less(dn.inserted_window_dims, ob),
        scatter_dims_to_operand_dims=less(dn.scatter_dims_to_operand_dims, ob), operand_batching_dims=(),
        scatter_indices_batching_dims=()))
    at = lambda dims, b, n: tuple(b[dims.index(d)] if d in dims else slice(None) for d in range(n))
    take = lambda x, dims, b: np.asarray(x, dtype=object)[at(dims, b, np.ndim(x))]
    first = [take(v, d, (0,) * len(ob)) for v, d in zip(vals, (ob, ib, ub))]
    one = jax.make_jaxpr(lambda *a: eqn.primitive.bind(*a, **new))(
        *[jax.ShapeDtypeStruct(np.shape(x), v.aval.dtype) for x, v in zip(first, eqn.invars)]).eqns[0]
    out = np.array(vals[0], dtype=object)
    for b in np.ndindex(*[np.shape(vals[0])[d] for d in ob]):
        out[at(ob, b, out.ndim)] = f(one, [take(v, d, b) for v, d in zip(vals, (ob, ib, ub))])
    return out


def _scatter(eqn, vals):
    """scatter (scatter-add: e.g. a transposed gather) with one symbolic index (e.g. a stage index): one
    bind per candidate position and per out-of-range position (-1, n: dropped or clamped as the scatter's
    mode says), picked."""
    if eqn.params["dimension_numbers"].operand_batching_dims:
        return _unbatched(eqn, vals, _scatter)
    idx, dn = np.asarray(vals[1], dtype=object), eqn.params["dimension_numbers"]
    if eqn.primitive.name == "scatter" and eqn.params["mode"] != jax.lax.GatherScatterMode.FILL_OR_DROP and \
            not dn.inserted_window_dims and np.ndim(vals[2]) == np.ndim(vals[0]) == len(dn.update_window_dims) and \
            idx.shape == (len(dn.scatter_dims_to_operand_dims),):  # a window over all axes at one index vector, clamped:
        starts = [0] * np.ndim(vals[0])  # dynamic_update_slice (vmap's form of it)
        for k, d in enumerate(dn.scatter_dims_to_operand_dims):
            starts[d] = idx[k]
        return _update_at(vals[0], vals[2], starts, [d for d, x in enumerate(starts) if isinstance(x, Node)])
    sym = [p for p in np.ndindex(idx.shape) if isinstance(idx[p], Node)]
    rows = sorted({p[:-1] for p in sym})  # index rows (the index vector is the last axis)
    n = np.shape(vals[0])[eqn.params["dimension_numbers"].scatter_dims_to_operand_dims[sym[0][-1] if sym[0] else 0]]
    if np.size(vals[2]) <= len(rows) * (n + 2):  # (by entries: fewer nodes than a candidate array per position)
        return _scatter_by_entries(eqn, vals)
    if len(sym) > 1 and eqn.primitive.name == "scatter-add" and len(rows) == len(sym):
        # linear in the updates: the operand plus one scatter per symbolic row (its updates only, the other
        # rows' indices constant), each with a single symbolic index
        upd, dn = np.asarray(vals[2], dtype=object), eqn.params["dimension_numbers"]
        row = np.empty(upd.shape, dtype=object)
        for u in np.ndindex(upd.shape):
            row[u] = tuple(u[d] for d in range(upd.ndim) if d not in dn.update_window_dims)
        part = lambda keep: np.where(np.vectorize(keep, otypes=[bool])(row), upd, 0.0)
        const = np.where(np.vectorize(lambda x: isinstance(x, Node))(idx), 0, idx)
        out = _linear(eqn, [vals[0], const.astype(eqn.invars[1].aval.dtype), part(lambda q: q not in rows)])
        out = out.reshape(eqn.outvars[0].aval.shape)
        for r in rows:
            ir = const.astype(object)
            ir[r] = idx[r]
            out = _ew(add, out, _scatter(eqn, [np.zeros(np.shape(vals[0])), ir, part(lambda q, r=r: q == r)]))
        return out
    if len(sym) > 1:  # several symbolic indices: per entry (not (n + 2)^k candidates)
        return _scatter_by_entries(eqn, vals)
    p = sym[0]
    if (n + 2) * np.size(vals[0]) > 10 ** 7:  # (a candidate per position, each the whole array)
        raise NotImplementedError(f"kernelax: {eqn.primitive.name} into {np.shape(vals[0])} at a symbolic index: too large to unroll")
    place = ((lambda v: _scatter(eqn, v)) if len(sym) > 1 else  # (several symbolic indices: one after the other)
             (lambda v: _linear(eqn, v).reshape(eqn.outvars[0].aval.shape)) if eqn.primitive.name == "scatter-add"
             else (lambda v: _move(eqn, v)[0]))
    const = ((lambda v: (lambda c: c.__setitem__(p, v) or c)(idx.copy())) if len(sym) > 1 else
             (lambda v: np.where(np.vectorize(lambda x: isinstance(x, Node))(idx), v, idx).astype(eqn.invars[1].aval.dtype)))
    lo = 0 if np.issubdtype(eqn.invars[1].aval.dtype, np.unsignedinteger) else -1  # (unsigned: none below 0)
    res = [place([vals[0], const(v), vals[2]]) for v in range(lo, n + 1)]
    i = _push(lambda c: add(c, float(-lo)), idx[p])
    return _ew(lambda *rv: _pick(i, rv), *res)


def _scatter_by_entries(eqn, vals):
    """scatter / scatter-add at symbolic positions: each entry in order (the last one set wins), compared with each
    position (entries x size nodes; beyond SCATTER_MAX refused: the region stays in XLA, which writes each entry once)."""
    flat = np.asarray(vals[0], dtype=object).ravel().copy()
    add_ = eqn.primitive.name == "scatter-add"
    entries = _scatter_entries(eqn, vals, cands=True)
    if sum(0 if is_c(f) else len(at) for f, _, at in entries) > SCATTER_MAX:
        raise NotImplementedError(f"kernelax: {eqn.primitive.name} into {np.shape(vals[0])} at a symbolic index: too large to unroll")
    for f, x, at in entries:
        for q in at if not is_c(f) else [int(f)] if f >= 0 else []:  # (the positions its constant index parts allow)
            hit = cmp("eq", f, float(q)) if not is_c(f) else True
            flat[q] = add(flat[q], select(hit, 0.0, x)) if add_ else select(hit, flat[q], x)
    return flat.reshape(np.shape(vals[0]))


class _Scatter:
    """Updates added at sample-dependent positions of an array of `shape` (zeros else): entries (flat position, value),
    positions -1 dropped. An output summed over the samples is accumulated in the kernel per thread (emit_map);
    any other use materializes it as an array of nodes (`dense`: the scatter-adds evaluated as before)."""

    def __init__(self, shape, entries, dense):
        self.shape, self.entries, self.dense = tuple(shape), entries, dense
        self.size = int(np.prod(self.shape, dtype=int))


def _scatter_entries(eqn, vals, cands=False):
    """The entries (flat position, value) that scatter-add puts into its operand: per update element, its index
    vector and window offset per operand dimension, out-of-bounds windows dropped (clamped in clip mode). cands: as
    (flat position, value, the positions it can have: those its constant index parts allow)."""
    p, (idx, upd) = eqn.params, (np.asarray(v, dtype=object) for v in vals[1:3])
    dn, shape = p["dimension_numbers"], eqn.outvars[0].aval.shape
    batch = list(zip(dn.operand_batching_dims, dn.scatter_indices_batching_dims))  # (an operand dim at the entry's
    # position in an index dim: e.g. vmap of a scatter at per-sample indices)
    win = [d for d in range(len(shape)) if d not in dn.inserted_window_dims and d not in dn.operand_batching_dims]
    # (operand dims of the window, in order)
    size = {d: upd.shape[u] for d, u in zip(win, dn.update_window_dims)}
    strides = np.cumprod((1,) + shape[:0:-1])[::-1]
    clip = p["mode"] == jax.lax.GatherScatterMode.CLIP
    entries = []
    for u in np.ndindex(upd.shape):
        s = tuple(u[d] for d in range(upd.ndim) if d not in dn.update_window_dims)
        w = dict(zip(win, (u[d] for d in dn.update_window_dims)))
        start = dict(zip(dn.scatter_dims_to_operand_dims, (to_float(idx[s + (c,)]) for c in range(idx.shape[-1]))))
        start.update({d: float(s[i]) for d, i in batch})
        flat, ok, at = 0.0, True, [0]  # (at: the flat positions the entry can have)
        for d in range(len(shape)):
            hi = float(shape[d] - size.get(d, 1))
            st = start.get(d, 0.0)
            vs = [min(max(int(st), 0), int(hi))] if is_c(st) else range(int(hi) + 1)
            at = [a + int(strides[d]) * (v + w.get(d, 0)) for a in at for v in vs]
            if clip:
                st = maxmin("min", maxmin("max", st, 0.0), hi)
            elif is_c(st):
                ok = ok if 0 <= st <= hi else False
            elif len(shape) == 1 and hi == shape[0] - 1:  # (one entry of a vector: the kernel drops positions outside)
                pass
            else:
                ok = band(ok, band(cmp("ge", st, 0.0), cmp("le", st, hi))) if ok is not True else band(cmp("ge", st, 0.0), cmp("le", st, hi))
            flat = add(flat, mul(float(strides[d]), add(st, float(w.get(d, 0)))))
        if ok is not False:
            entries.append((flat if ok is True else select(ok, -1.0, flat), upd[u]) + ((at,) if cands else ()))
    return entries


def _scatter_op(eqn, vals, f):
    """scatter-max / scatter-min / scatter-mul: each entry combined (f) into the position it hits (compared with each
    position for a symbolic one)."""
    if eqn.params["dimension_numbers"].operand_batching_dims:
        return _unbatched(eqn, vals, lambda e, v: _scatter_op(e, v, f))
    flat = np.asarray(vals[0], dtype=object).ravel().copy()
    entries = _scatter_entries(eqn, vals, cands=True)
    if sum(1 if is_c(q) else len(at) for q, _, at in entries) > 10 ** 6:
        raise NotImplementedError(f"kernelax: {eqn.primitive.name} into {np.shape(vals[0])} at symbolic indices: too large to unroll")
    for q, x, at in entries:
        for r in ([int(q)] if q >= 0 else []) if is_c(q) else at:
            flat[r] = f(flat[r], x) if is_c(q) else select(cmp("eq", q, float(r)), flat[r], f(flat[r], x))
    return flat.reshape(np.shape(vals[0]))


def _zeros(x):
    return not _symbolic(x) and not isinstance(x, (_Mem, _Split, _Scatter)) and not np.any(np.asarray(x))


SCATTER_MAX = 1 << 15  # compare-selects per sample of a scatter at symbolic positions (_scatter_by_entries)
MAX_NODES = env("KERNELAX_MAX_NODES", 300000, int, 1)  # beyond this, tracing stops (C size and gcc time grow with it)
SCAN_UNROLL = 8000  # scans with length x body equations above this run as loops (_rolled; measured)
SCAN_SHORT = 0  # scans of at most this many steps unrolled whatever their size, and while loops peeled as long as their
# condition folds to True, at most this many steps (_while); the GPU kernels: 16 (gpu.py)
ROLL_XS = 50000  # scans unrolled for their xs only (long, but xs large next to the body) rolled where unrolled they
# would add more nodes than this (xs read at the step: picks, PICK_INTS; 0: never; on the CPU since loops over a counter
# run without masks (emit) and typed keys stay lazy: RL rollouts 1.1-2.3x faster, 2-10x smaller, review.md §53)
SHORT_NODES = 0  # with SCAN_SHORT: unrolled only within this many new nodes per loop (0: no limit; the GPU kernels:
# 20 000, gpu.py; e.g. blackjax' scan over 10 NUTS steps, 11 000 nodes each, stays a loop)


def _size(jaxpr):  # equations of a jaxpr, those of nested jaxprs included
    return sum(1 + sum(_size(j) for j in jcore.jaxprs_in_params(e.params)) for e in jaxpr.eqns)


def _rolled(p, short=True, xs_too=True):
    """Whether a scan (params p) runs as a loop: long (length x body size above SCAN_UNROLL) and its xs and ys
    small next to its body (a loop reads xs and writes ys at the step: their elements per step, over the steps);
    short: not if at most SCAN_SHORT steps; xs_too: xs counted."""
    j, n = p["jaxpr"], p["length"]
    consts, carry, xs = p["ft_in"].update(list(range(len(j.invars)))).unpack()
    io = sum(int(np.prod(v.aval.shape)) for v in [j.invars[i] for i in xs if xs_too] + j.outvars[len(carry):])
    size = _size(j)
    return n * size > SCAN_UNROLL and (n > SCAN_SHORT or not short) and n * io <= size


def _scan(eqn, vals):
    """lax.scan with symbolic operands: unrolled (static length; e.g. the binary search of jnp.searchsorted
    in jnp.interp), or as a while loop (_rolled; traced by JAX, with the step number as a further carry; xs read
    and ys written at it), so that the kernel holds its body once."""
    p = eqn.params
    consts, carry, xs = map(list, p["ft_in"].update(list(vals)).unpack())
    body, n = p["jaxpr"], p["length"]

    def rolled():
        nc, ncarry = len(consts), len(carry)
        yavals = [v.aval for v in body.outvars[ncarry:]]

        def loop(*a):
            def step(c):
                i, cs, ys = c
                k = n - 1 - i if p["reverse"] else i
                out = jcore.jaxpr_as_fun(body)(*a[:nc], *cs, *[jax.lax.dynamic_index_in_dim(x, k, keepdims=False) for x in a[nc + ncarry:]])
                return i + 1, tuple(out[:ncarry]), tuple(jax.lax.dynamic_update_index_in_dim(y, o, k, 0) for y, o in zip(ys, out[ncarry:]))
            ys = tuple(jax.numpy.zeros((n,) + y.shape, y.dtype) for y in yavals)
            _, cs, ys = jax.lax.while_loop(lambda c: c[0] < n, step, (0, tuple(a[nc:nc + ncarry]), ys))
            return (*cs, *ys)
        with internals.no_constant_folding():  # (trace() binds under ensure_compile_time_eval: an equation without
            # operands, the sample index of kernelax.jit's per-sample bodies, would be evaluated, not traced)
            closed = jax.make_jaxpr(loop)(*[jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in eqn.invars])
        _ROLLED.update((id(e.params["body_jaxpr"]), e.params["body_jaxpr"]) for e in closed.eqns if e.primitive.name == "while")  # (not
        return _eval(closed, closed.consts, vals)  # peeled again by _while: a counter to a constant bound)
    if _rolled(p):
        return rolled()
    budget = SHORT_NODES if SHORT_NODES and _rolled(p, short=False) else \
        ROLL_XS if ROLL_XS and _rolled(p, xs_too=False) else 0  # (unrolled for SCAN_SHORT or its xs only: within a budget)
    steps = range(p["length"] - 1, -1, -1) if p["reverse"] else range(p["length"])
    xavals = [v.aval for v in body.invars[len(consts) + len(carry):]]
    ys, start = [], dag.size()
    for i in steps:  # the step's slices of xs as arrays of the body's avals (an element alone would be a bare node)
        x_i = [x.at(i, np.int32) if isinstance(x, _Split) else np.asarray(x, dtype=object)[i] if _is_key(a) else
               _norm(np.asarray(x, dtype=object)[i], a) for x, a in zip(xs, xavals)]
        out = _eval(body, body.consts, consts + carry + x_i)
        if budget and not ys and n * (dag.size() - start) > budget:  # (all steps like the first one: too
            return rolled()  # many nodes; the first one's stay unused)
        carry, y = map(list, p["ft_out"].update(out).unpack())
        ys.append(y)
    if not ys:  # (length 0: the carry as it is, empty ys)
        return carry + [np.empty(v.aval.shape, dtype=object) for v in eqn.outvars[len(carry):]]
    ys = ys[::-1] if p["reverse"] else ys
    return carry + [np.stack([np.asarray(y[k], dtype=object) for y in ys]) for k in range(len(ys[0]))]


_DEPTH = [0]  # nesting depth of the loop body being traced: tags the carry placeholders


def _is_key(aval):
    return jax.dtypes.issubdtype(aval.dtype, jax.dtypes.prng_key)


def _data(aval):  # the aval of the values that carry it (keys: their uint32 key data)
    return jax.core.ShapedArray(aval.shape + internals.key_impl(aval.dtype).key_shape, np.uint32) if _is_key(aval) else aval


_IPROD = frozenset("mul integer_pow square reduce_prod cumprod dot_general scatter-mul".split())  # integer products
_IARITH = _IPROD | frozenset("add add_any sub neg abs reduce_sum cumsum scatter-add reduce_window_sum".split())  # may
# overflow


def _idivrem(x, y, rem, dtype):
    """Integer quotient (truncated) or remainder (the dividend's sign) as XLA: x / 0 = -1 (all ones), x % 0 = x;
    exact on integer nodes (64 bits), on doubles below 2^53."""
    b, signed = _ibits(dtype)
    ones = -1 if signed else (1 << b) - 1
    if is_c(x) and is_c(y):
        x, y = int(x), int(y)
        if y == 0:
            return x if rem else ones
        q = abs(x) // abs(y) * (1 if (x < 0) == (y < 0) else -1)
        return x - q * y if rem else q
    zero = cmp("eq", y, 0)
    if _isi(x) or _isi(y) or b == 64:
        x, y = to_int(x), to_int(y)
        if signed:
            ax, ay = (iselect(icmp("lt", v, 0), v, neg(v)) for v in (x, y))
            q, r = _udivmod(ax, iselect(zero, ay, 1))
            q = iselect(icmp("lt", ibit("ixor", x, y), 0), q, neg(q))
            r = iselect(icmp("lt", x, 0), r, neg(r))
        else:
            q, r = _udivmod(x, iselect(zero, y, 1))
        return iselect(zero, r, x) if rem else iselect(zero, q, ones)
    ys = select(zero, y, 1.0)  # x times 1/y rounded away from zero (by one or two ulp): x * fl(1/y) can fall short of
    if is_c(ys):  # an exact multiple (49 * fl(1/49) < 1), this never; nor does it reach the next integer below 2^50
        m = 1.0 / ys if np.frexp(abs(ys))[0] == 0.5 else float(np.nextafter(1.0 / ys, np.copysign(np.inf, ys)))
    else:  # (relative error ~2^-50 against a distance of at least 1/|y|)
        m = mul(mk("recip", ys), 1.0 + 2.0 ** -52)
    q = unary("trunc", mul(x, m))
    r = sub(x, mul(ys, q))  # (exact: |q y| <= |x| < 2^53)
    if signed and not (is_c(y) and y != -1):  # (INT_MIN / -1 wraps to INT_MIN, as XLA)
        q = select(cmp("eq", q, 2.0 ** (b - 1)), q, -2.0 ** (b - 1))
    return select(zero, r, x) if rem else select(zero, q, float(ones))


_UBIG = (1 << 63) - (1 << 16)  # (_udivmod: divisors from here on by compares)


def _udivmod(x, y):
    """Quotient and remainder of uint64 integer nodes, exact. Divisors below _UBIG: a quotient from doubles (off by
    up to ~2^12 for 64-bit values; from 2^63 on, for the divisor 1, converted through the upper half), corrected by a
    second double estimate of the remainder (between -2^13 and y + 2^13: signed, as int64) and a final step of one.
    From _UBIG on the quotient is 0, 1 or 2: by unsigned compares (a constant divisor: only the path it takes)."""
    x, y = to_int(x), to_int(y)

    def large():
        q1 = icmp("ge", x, y, True)
        r = iselect(q1, x, sub(x, y))
        q2 = icmp("ge", r, y, True)
        return add(iselect(q1, 0, 1), iselect(q2, 0, 1)), iselect(q2, r, sub(r, y))

    def small():
        yf = to_float(y, True)
        qf = maxmin("min", unary("trunc", div(to_float(x, True), yf)), 2.0 ** 64 - 2048)  # (converted: no value
        q = iselect(cmp("ge", qf, 2.0 ** 63), to_int(maxmin("min", qf, 2.0 ** 63 - 1024)),  # beyond the int64 it
                    add(to_int(maxmin("max", sub(qf, 2.0 ** 63), 0.0)), -(1 << 63)))  # becomes, C leaves that undefined)
        r = sub(x, mul(q, y))
        c = to_int(unary("trunc", div(to_float(r), yf)))  # (r is small and may be negative: signed)
        q, r = add(q, c), sub(r, mul(c, y))
        neg = icmp("lt", r, 0)
        q, r = iselect(neg, q, sub(q, 1)), iselect(neg, r, add(r, y))
        big = icmp("ge", r, y, True)
        return iselect(big, q, add(q, 1)), iselect(big, r, sub(r, y))

    if is_c(y):
        return large() if int(y) % 2 ** 64 >= _UBIG else small()
    (ql, rl), (qs, rs), at = large(), small(), icmp("ge", y, _UBIG, True)
    return iselect(at, qs, ql), iselect(at, rs, rl)


def _random(eqn, vals, lazy=True):
    """jax.random's key primitives: keys are carried as their uint32 data (shape + key shape); seed, fold_in,
    bits and split evaluate the key implementation's functions per key (threefry2x32: see _eval); split also as
    _Split (lazy, LAZY_KEYS)."""
    name, p = eqn.primitive.name, eqn.params
    if name in ("random_wrap", "random_unwrap", "random_clone"):
        return [vals[0]]
    if name == "random_split" and lazy and _Split.applies(eqn, vals):
        return [_Split(eqn, vals)]
    impl = p["impl"] if name == "random_seed" else internals.key_impl(eqn.invars[0].aval.dtype)
    fn = {"random_seed": impl.seed, "random_fold_in": impl.fold_in,
          "random_bits": lambda k: internals.jax_function("random.prng", "_call_impl_random_bits")(  # (JAX 0.12.1:
              impl, k, p["bit_width"], p["shape"], p.get("num_streams", 1)),  # num_streams outputs)
          "random_split": lambda k: impl.split(k, p.get("shape"))}[name]
    avals = [jax.ShapeDtypeStruct(v.aval.shape + (impl.key_shape if _is_key(v.aval) else ()),
                                  np.uint32 if _is_key(v.aval) else v.aval.dtype) for v in eqn.invars]
    batch = eqn.invars[0].aval.shape  # one evaluation per key (operands broadcast like the keys)
    per = [jax.ShapeDtypeStruct(a.shape[len(batch):], a.dtype) for a in avals]
    closed = jax.make_jaxpr(fn)(*per)
    vals = [np.broadcast_to(np.asarray(v, dtype=object), a.shape) for v, a in zip(vals, avals)]
    res = [_eval(closed, closed.consts, [_norm(v[i], a) for v, a in zip(vals, per)]) for i in np.ndindex(batch)]
    return [np.stack([np.asarray(r[j], dtype=object) for r in res]).reshape(batch + np.shape(res[0][j])) if res else
            np.zeros((0,), object) for j in range(len(closed.out_avals))]


def _threefry(eqn, vals):
    """The threefry2x32 hash: per entry one node (dag.threefry), its two words by a mask and a shift; of constant
    entries their value (special.threefry2x32)."""
    from kernelax.special import threefry2x32
    shape = np.broadcast_shapes(*(np.shape(v) for v in vals))
    vs = [np.broadcast_to(np.asarray(v, dtype=object), shape) for v in vals]
    lo, hi = np.empty(shape, object), np.empty(shape, object)
    for i in np.ndindex(shape):
        a = [v[i] for v in vs]
        if any(isinstance(x, Node) for x in a):
            t = threefry(*a)
            lo[i], hi[i] = ibit("iand", t, 0xFFFFFFFF), ishift("ishru", t, 32)
        else:
            with np.errstate(over="ignore"):
                lo[i], hi[i] = map(int, threefry2x32(*(np.asarray(int(x) % (1 << 32), np.uint32) for x in a)))
    return [lo, hi]


NUMERICS = ("jax_accurate_erf_inv", "jax_accurate_logistic", "jax_threefry_partitionable")  # (erfc: glibc's libmvec
# whatever jax_accurate_default_erfc says; within 6 ulp of XLA's, float32 closer to the exact value)


def numerics():
    """JAX's settings that change what the expansions compute (erf_inv's lowering, logistic's, the random bits): part
    of the index keys, so that a kernel built under one setting is not taken under another."""
    return " ".join(f"{k}={getattr(jax.config, k, None)}" for k in NUMERICS)


XLA_GPU = False  # numerics as XLA lowers them on the GPU (gpu.py: True): float32 lgamma and digamma in float32 with
# subnormal results, logistic keeping subnormals; else as on the CPU (float32 lgamma and digamma in double)


def _gamma(name, x):  # lgamma, digamma (JAX >= 0.12.1: XLA lowers them through these jnp functions; on the CPU
    # through C++ of the same algorithms, float64 lgamma within an ulp of it)
    impl = internals.jax_function(f"lax.{name}", f"{name}_impl")
    with jax.enable_x64(True):  # (doubles in the expansion also where x64 is off)
        return impl(x, use_f64_for_f32=not XLA_GPU, subnormals=XLA_GPU)


def _expansions():
    """Primitives XLA lowers through a JAX function or in C++, as functions of other primitives: name -> f(*operands,
    **params) -> output. Those of special.py (XLA's C++ lowerings as jnp: erf_inv with JAX's flag off, zeta,
    polygamma, ...), else JAX's own, imported at their first use (internals.jax_function)."""
    from kernelax import special
    from kernelax.schur import schur
    def jx(name, up=True):  # (up: broadcast, 16-bit floats in float32, as JAX lowers it)
        def f(*a, **_):
            fn = internals.jax_function("lax.special", name)
            return (internals.jax_function("lax.special", "_up_and_broadcast")(fn) if up else fn)(*a)
        return f

    def jf(module, name):  # JAX's jnp function `name` through which XLA lowers a primitive
        return lambda *a, **_: internals.jax_function(module, name)(*a)

    def trig(name):  # sinpi, ..., atan2pi (JAX >= 0.12.1): XLA lowers them through these jnp functions (operands
        impl = lambda: internals.jax_function("lax.trig", f"_{name}_impl")  # broadcast)
        return lambda *a, **_: impl()(*jnp.broadcast_arrays(*a))
    erf_inv = lambda x, **_: (internals.jax_function("lax.special", "erf_inv_impl") if jax.config.jax_accurate_erf_inv
                              else special.erf_inv)(x)  # (JAX's flag: its jnp lowering, else XLA's in C++)
    return {"erf_inv": erf_inv, "igamma": jx("igamma_impl"), "igammac": jx("igammac_impl"),
            "igamma_grad_a": jx("igamma_grad_a_impl"),
            "regularized_incomplete_beta": jx("regularized_incomplete_beta_impl"),
            "bessel_i0e": jx("bessel_i0e_impl", False),
            "lgamma": lambda x, **_: _gamma("lgamma", x), "digamma": lambda x, **_: _gamma("digamma", x),
            "population_count": lambda x, **_: special.population_count(x), "clz": lambda x, **_: special.clz(x),
            "nextafter": lambda x, y, **_: special.nextafter(x, y),
            "top_k": lambda x, k, axis=-1, **_: special.top_k(x, k, axis),
            "zeta": lambda x, q, **_: special.zeta(x, q), "polygamma": lambda m, x, **_: special.polygamma(m, x),
            "bessel_i1e": jx("bessel_i1e_impl", False), "exp2m1": jf("lax.exp_log", "_exp2m1_impl"),
            "log2p1": jf("lax.exp_log", "_log2p1_impl"),
            "logistic": lambda x, accuracy=None, **_: internals.jax_function("lax.lax", "logistic_impl")(
                x, accuracy, preserve_subnormals=XLA_GPU),
            "approx_top_k": lambda x, *, k, reduction_dimension, is_max_k, **_: special.approx_top_k(x, k, reduction_dimension, is_max_k),
            "random_gamma": lambda key, a, log_space, **_: internals.jax_function(  # (JAX 0.12.1: out_sharding)
                "random.core", "_gamma_impl_local")(key, a, log_space=log_space, use_vmap=True),
            "random_poisson": lambda key, lam, dtype, **_: internals.jax_function(  # (a primitive since JAX 0.12.1)
                "random.core", "_poisson_impl_local")(key, lam, dtype=dtype),
            "random_gamma_grad": lambda a, x, **_: internals.jax_function("lax.special", "random_gamma_grad")(a, x),
            "schur": schur, **{n: trig(n) for n in ("sinpi", "cospi", "tanpi", "asinpi", "acospi", "atanpi", "atan2pi")}}


def _expand(eqn, vals):
    """A primitive by its expansion (_EXPANSIONS); of 16-bit floats computed in float32 and rounded once at the end,
    as XLA lowers them (the primitive's result rounded to the format, not each step of its expansion: lgamma's
    series rounded per step was off by 12 ulps)."""
    fn, narrow = functools.partial(_EXPANSIONS[eqn.primitive.name], **eqn.params), lambda d: jax.dtypes.issubdtype(d, np.floating) and np.dtype(d) in _FLOATS16
    if any(narrow(v.aval.dtype) for v in eqn.invars) and eqn.primitive.name not in _OWN16:
        f = fn
        fn = lambda *a: [o.astype(v.aval.dtype) for o, v in zip(jax.tree.leaves(f(*[x.astype(np.float32) if narrow(x.dtype) else x
                                                                                    for x in a])), eqn.outvars)]
    closed = jax.make_jaxpr(fn)(*[jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in eqn.invars])
    return _eval(closed, closed.consts, vals)


_EXPANSIONS = _expansions()
_OWN16 = {"nextafter", "random_gamma", "random_poisson", "random_gamma_grad", "logistic"}  # expansions of 16-bit floats
# in their format, as XLA (nextafter: the format's neighbour, by its bits; samples drawn in the dtype of a; logistic:
# rounded per operation)


def bitsf64(x):  # an integer's 64 bits as a double (a loop carry)
    return mk("bitsf64", x) if isinstance(x, Node) else float(np.uint64(int(x) % 2 ** 64).view(np.float64))


def _same_bits(x, ph):  # whether x is the bits of placeholder ph as they came in (a carry passed through)
    return isinstance(x, Node) and x.op == "bitsf64" and isinstance(x.args[0], Node) and x.args[0].op == "fbits" and x.args[0].args[0] is ph


_ROLLED = {}  # id -> the body of a while loop that is a rolled scan (_scan; kept: an id is reused once freed)


def _while(eqn, vals):
    """A while loop: unrolled if its condition folds to True for at most SCAN_SHORT iterations, then to False, within
    SHORT_NODES new nodes (a counter to a constant bound: constant indices, e.g. the stages of an RK step as diffrax
    writes them); its first iterations unrolled if the condition folds for two or more and then depends on the
    carries (diffrax' stages inside its step loop), the rest symbolic (_loop); else symbolic as it is (one known
    iteration alone: the body twice, no index made constant, e.g. NUTS' loops)."""
    p = eqn.params
    nc, nb = p["cond_nconsts"], p["body_nconsts"]
    if id(p["body_jaxpr"]) in _ROLLED:  # (a scan rolled on purpose)
        return _loop(eqn, vals)
    start, peeled, k, known = dag.size(), list(vals), 0, True
    for k in range(SCAN_SHORT + 1):
        carry = list(peeled[nc + nb:])
        cond = np.asarray(_eval(p["cond_jaxpr"], p["cond_jaxpr"].consts, list(peeled[:nc]) + carry)[0], dtype=object)
        known = cond.size == 1 and is_c(cond.item())
        if not known or k == SCAN_SHORT or SHORT_NODES and dag.size() - start > SHORT_NODES:
            break
        if not cond.item():
            return carry
        out = _eval(p["body_jaxpr"], p["body_jaxpr"].consts, list(peeled[nc:nc + nb]) + carry)
        peeled = list(peeled[:nc + nb]) + [o.nodes() if isinstance(o, _LAZY) else o for o in out]
    return _loop(eqn, peeled if not known and k > 1 else vals)  # (too long or too large to unroll: the loop as it is)


def _loop(eqn, vals):
    """Symbolic while loop. Body and cond are traced on positional carry placeholders; the loop
    node keeps them, the body outputs and the condition in its first argument (not traversed by
    `schedule`), its other arguments are
    the initial carries and the carry-independent nodes the body reads (`free`), so binding-time
    analysis, emission order and tabulation treat the loop like any other node. Identical loops
    are one node (hash-consing). Carries the body passes through unchanged are replaced by their
    initial value, which also hoists all work that depends only on them out of the loop.
    Placeholders are tagged with the nesting depth: a nested loop reads the carries of the
    enclosing loops as free nodes, and sibling loops of the same depth may share placeholders."""
    p = eqn.params
    nc, nb = p["cond_nconsts"], p["body_nconsts"]
    avals = [_data(v.aval) for v in eqn.outvars]
    sizes = [int(np.prod(a.shape, dtype=int)) for a in avals]
    ints = [np.issubdtype(a.dtype, np.integer) for a, n in zip(avals, sizes) for _ in range(n)]  # integer carries: as
    # doubles (exact below 2^53), so that the loop's placeholders keep one type; those the body computes as integer
    # nodes (bit operations, wraparound: e.g. 64-bit hashes) by their bits (bits: their indices; a double's 64 bits)
    raw = [x for v in vals[nc + nb:] for x in np.asarray(v, dtype=object).ravel()]
    kinds = [a.dtype == bool for a, n in zip(avals, sizes) for _ in range(n)]
    depth = _DEPTH[0]
    ph = [mk("carrymask", depth, j) if b else mk("carry", depth, j, _vtype(a.dtype)) for j, (b, a) in
          enumerate(zip(kinds, (a for a, n in zip(avals, sizes) for _ in range(n))))]
    unflat = lambda xs: [_norm(xs[o:o + n], a) for a, n, o in zip(avals, sizes, np.cumsum([0] + sizes))]
    carry_index = [i for a in avals for i in np.ndindex(a.shape)]  # flat carry -> index in its array
    bits = {j for j, (x, i) in enumerate(zip(raw, ints)) if i and (_isi(x) or dag._big(x))}  # (64-bit inputs, integer
    # results, constants beyond 2^53)
    while True:
        init = [bitsf64(to_int(x)) if j in bits else to_float(x) if i else x for j, (x, i) in enumerate(zip(raw, ints))]
        fixed, _DEPTH[0] = {}, depth + 1
        while True:
            carry = unflat([to_int(raw[j]) if j in bits and j in fixed else mk("fbits", ph[j]) if j in bits else
                            fixed.get(j, ph[j]) for j in range(len(ph))])
            out = [o.nodes() if isinstance(o, _LAZY) else o for o in _eval(p["body_jaxpr"], p["body_jaxpr"].consts, vals[nc:nc + nb] + carry)]
            out = [x for v in out for x in np.asarray(v, dtype=object).ravel()]
            more = {j for j, x in enumerate(out) if ints[j] and j not in bits and _isi(x)}
            if more:
                break
            new = [bitsf64(to_int(x)) if j in bits else to_float(x) if i else x for j, (x, i) in enumerate(zip(out, ints))]
            cond = np.asarray(_eval(p["cond_jaxpr"], p["cond_jaxpr"].consts, vals[:nc] + carry)[0], dtype=object)
            if cond.size > 1:  # a batched predicate (while under vmap): each element keeps its carries once its own
                new = [select(cond[idx[:cond.ndim]], ph[j], x) if not is_c(cond[idx[:cond.ndim]]) or not cond[idx[:cond.ndim]]
                       else x for j, (x, idx) in enumerate(zip(new, carry_index))]  # predicate is false; the loop runs
                cond = functools.reduce(bor, cond.ravel())  # while any is true
            else:
                cond = cond.item()
            same = [j for j in range(len(ph)) if j not in fixed and (new[j] is ph[j] or j in bits and _same_bits(new[j], ph[j]))]
            if not same:
                break
            fixed.update((j, init[j]) for j in same)
        _DEPTH[0] = depth
        if not more:
            break
        bits |= more
    _DEPTH[0] = depth
    J = [j for j in range(len(ph)) if j not in fixed]
    assert not (is_c(cond) and cond), "while loop whose condition is always true"
    back = lambda j, x: to_int(raw[j]) if j in bits and j in fixed else mk("fbits", x) if j in bits else x  # (bits: as integers)
    if is_c(cond) or not J:
        return unflat([to_int(raw[j]) if j in bits else x for j, x in enumerate(init)])
    roots = [new[j] for j in J] + [cond]
    dep, free = {ph[j].id for j in J}, {}
    for n in schedule(roots):
        if n.id not in dep and any(isinstance(a, Node) and a.id in dep for a in n.args):
            dep.add(n.id)
            free.update((a.id, a) for a in n.args if isinstance(a, Node) and a.id not in dep)
    free.update((r.id, r) for r in roots if isinstance(r, Node) and r.id not in dep)
    loop = mk("while", (tuple(ph[j] for j in J), tuple(roots[:-1]), cond), *[init[j] for j in J], *free.values())
    out = list(init)
    for i, j in enumerate(J):
        out[j] = mk("loopmask" if kinds[j] else "loopout", loop, i)
    return unflat([back(j, x) for j, x in enumerate(out)])


def _hi_expand(eqn):
    """The expansion of a call_hi_primitive (jax >= 0.12: searchsorted, custom_root, ...) into primitives, its
    operands as the pytree it takes."""
    hp = eqn.params["_prim"]
    expand = (lambda *a: jax.tree.leaves(hp.expand(*jax.tree.unflatten(hp.in_tree, a)))) if getattr(hp, "in_tree", None) else hp.expand
    return jax.make_jaxpr(expand)(*[jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in eqn.invars])


def _eval(jaxpr, consts, args, keep=False):
    """The values of jaxpr's outputs; sums over the samples (_Scatter) as arrays of nodes unless `keep` (a call's,
    the traced function's: still sums, for the kernel's reduced outputs)."""
    env = {}
    read = lambda v: np.asarray(v.val) if isinstance(v, jcore.Literal) else env[v]
    env.update(zip(jaxpr.constvars, [np.asarray(c) for c in consts]))
    env.update(zip(jaxpr.invars, args))
    for eqn in jaxpr.eqns:
        vals = [read(v) for v in eqn.invars]
        name = eqn.primitive.name
        if any(jax.dtypes.issubdtype(getattr(v.aval, "dtype", np.float64), np.complexfloating) for v in eqn.outvars):
            raise NotImplementedError(f"kernelax: complex numbers ('{name}' at {source_info_util.summarize(eqn.source_info)})")
        data = [i for i in range(len(vals)) if i not in range(len(vals))[_INDEX.get(name, slice(0, 0))]]
        if name not in _CALLS and any(isinstance(v, _Scatter) for v in vals):  # sample sums: extended, else dense
            if name == "scatter-add" and isinstance(vals[0], _Scatter) and _symbolic(vals[1]):
                env[eqn.outvars[0]] = _Scatter(vals[0].shape, vals[0].entries + _scatter_entries(eqn, vals),
                                               lambda e=eqn, v=vals: _scatter(e, [v[0].dense()] + v[1:]))
                continue
            if name in ("add", "add_any") and all(isinstance(v, _Scatter) for v in vals):
                env[eqn.outvars[0]] = _Scatter(vals[0].shape, vals[0].entries + vals[1].entries,
                                               lambda a=vals[0], b=vals[1]: _ew(add, a.dense(), b.dense()))
                continue
            vals = [v.dense() if isinstance(v, _Scatter) else v for v in vals]
        if name == "scatter-add" and _zeros(vals[0]) and _symbolic(vals[1]):  # e.g. an assembly
            env[eqn.outvars[0]] = _Scatter(np.shape(vals[0]), _scatter_entries(eqn, vals), lambda e=eqn, v=vals: _scatter(e, v))
            continue
        if name == "dynamic_update_slice" and _zeros(vals[0]) and sum(map(_symbolic, vals[2:])) == 1:  # into zeros at a
            env[eqn.outvars[0]] = _Scatter(np.shape(vals[0]), _update_entries(vals), lambda e=eqn, v=vals: _dynamic_slice(e, v))
            continue  # sample's start, e.g. the cotangent of T[i] for a shared T: a sum at positions where summed
        if name not in _CALLS and any(isinstance(v, _Split) for v in vals):  # keys not yet computed (_Split): passed
            first = isinstance(vals[0], _Split)  # on, read one at a time, taken by loops, else materialized
            kept = (set(range(len(vals))) if name in ("random_wrap", "random_unwrap") or first and vals[0].same(eqn) else
                    {0} if first and name == "dynamic_slice" and tuple(eqn.params["slice_sizes"]) in ((1, 2), (1,)) else  # (one
                    # key, of its uint32 data or of the key array itself: typed keys, e.g. a rolled scan's xs)
                    set(range(eqn.params["cond_nconsts"] + eqn.params["body_nconsts"])) if name == "while" else
                    set(range(len(vals))) - set(eqn.params["ft_in"].update(list(range(len(vals)))).unpack()[1]) if name == "scan" else
                    set())
            vals = [v.nodes() if isinstance(v, _Split) and i not in kept else v for i, v in enumerate(vals)]
        vals = [v.nodes() if isinstance(v, _Mem) and i not in data else v for i, v in enumerate(vals)]  # (indices: nodes)
        dense = _dense(eqn, vals) if name == "dot_general" and DENSE_MIN else None
        if dense is None and name not in _CALLS and any(isinstance(v, _Mem) for v in vals):  # inputs in memory (passed into calls):
            # moved, gathered per sample, or materialized
            filled = name == "gather" and eqn.params["mode"] == jax.lax.GatherScatterMode.FILL_OR_DROP
            if name in _MEM_MOVE and data == [0] and isinstance(vals[0], _Mem) and not any(_symbolic(v) for v in vals[1:]):
                outs = np.asarray(eqn.primitive.bind(vals[0].codes, *vals[1:], **(dict(eqn.params, fill_value=-1) if filled else eqn.params)))
                if not (outs < 0).any():  # (out of bounds in fill mode, code -1: the entries as nodes, the fill value)
                    env[eqn.outvars[0]] = _Mem(vals[0].k, outs, vals[0].kind, vals[0].offset, vals[0].fills)
                    continue
            kept = (1 if name in ("gather", "dynamic_slice") and isinstance(vals[0], _Mem) and any(_symbolic(v) for v in vals[1:])
                    else  # (read per sample: load) loop constants:
                    eqn.params["cond_nconsts"] + eqn.params["body_nconsts"] if name == "while" else  # read in the loop
                    len(vals) if name == "cond" else  # (the branches: as calls)
                    len(eqn.params["ft_in"].update(list(vals)).unpack()[0]) if name == "scan" else 0)
            vals = [v.nodes() if isinstance(v, _Mem) and i >= kept else v for i, v in enumerate(vals)]
        if name in _INDEX or name.startswith("scatter"):  # uint64 indices (integer nodes): their unsigned values as
            for k in range(len(vals))[_INDEX.get(name, slice(1, 2))]:  # doubles, so that 2^63 and beyond clamp to the end
                if eqn.invars[k].aval.dtype == np.uint64 and _symbolic(vals[k]):  # (not, as int64, to 0)
                    vals[k] = _ew(lambda x: to_float(x, True) if _isi(x) else x, vals[k])
        if name == "mul" and eqn.params.get("out_dtype") is not None:  # (operands converted first, as XLA's lowering)
            out = np.dtype(eqn.params["out_dtype"])
            if out not in (np.float64, np.float32):
                raise NotImplementedError(f"kernelax: mul with out_dtype {out} is not supported")
            to = tof64 if out == np.float64 else tof32
            vals = [np.asarray(_ew(to, v), dtype=object) if _symbolic(v) else v for v in vals]
        if name in ("dot_general", "conv_general_dilated") and eqn.params.get("preferred_element_type") == np.float64:
            vals = [np.asarray(_ew(tof64, v), dtype=object) if _symbolic(v) and eqn.invars[k].aval.dtype != np.float64 else v
                    for k, v in enumerate(vals)]  # (float32 operands: products and sums in double)
        if eqn.effects:  # (jax.debug.print, io_callback: run per call by XLA, never by a kernel nor while it is traced)
            raise NotImplementedError(f"kernelax: '{name}' has side effects (jax.debug.print, callbacks), which a kernel "
                                      f"cannot have ({source_info_util.summarize(eqn.source_info)})")
        if name in _IPROD and eqn.outvars and np.issubdtype(eqn.outvars[0].aval.dtype, np.integer) and \
                not _is_key(eqn.outvars[0].aval):  # (products of integers: exact on 64-bit integer nodes, then wrapped;
            vals = [_ew(lambda x: to_int(x) if isinstance(x, Node) and not _isb(x) and not _leafy(x) else x, v)  # beyond
                    if _symbolic(v) else v for v in vals]  # 2^53 as doubles they would round)
        if dense is not None:
            outs = [dense]
        elif eqx_error(eqn):  # equinox's runtime error (eqx.error_if): in a kernel nothing raises, the sample's values as
            outs = [eqx_nan(v.aval) for v in eqn.outvars]  # with EQX_ON_ERROR=nan (the callback's cond selects them)
        elif name == "custom_jvp_call" and vals and _symbolic(vals[0]) and \
                _jax_own(eqn.params["call_jaxpr"].debug_info, "exp_neg_sq", "lax/erf.py"):
            outs = [_ew(lambda x: unary("exp", neg(mul(x, x))), vals[0])]  # jax >= 0.12 (erf'): bit tricks for accuracy
        elif name in _CALLS:
            outs = _eval(*_sub_jaxpr(eqn), vals, keep=True)
        elif name in _EXPANSIONS and any(_symbolic(v) for v in vals):
            outs = _expand(eqn, vals)
        elif name.startswith("random_") and name not in _EXPANSIONS:
            outs = _random(eqn, vals)
        elif vals and isinstance(vals[0], _Split) and vals[0].same(eqn):
            outs = [vals[0]]
        elif vals and name == "dynamic_slice" and isinstance(vals[0], _Split):  # (one key: at its index)
            outs = [np.reshape(vals[0].at(vals[1], eqn.invars[1].aval.dtype), (1, 2))]
        elif name == "threefry2x32" and any(_symbolic(v) for v in vals):
            outs = _threefry(eqn, vals)
        elif name == "kernelax_sample":  # (primitive.sample_p) the sample's index in the batch
            outs = [mk("sample")]
        elif name == "platform_index" and None in eqn.params["platforms"]:  # jax.lax.platform_dependent: the
            outs = [np.asarray(eqn.params["platforms"].index(None))]  # kernels are their own platform: default branch
        elif not any(_symbolic(v) or isinstance(v, _LAZY) for v in vals):  # (lazy values: of inputs, symbolic)
            args = concrete(eqn, vals) if name not in _CALLS and not eqn.primitive.multiple_results else None
            outs = numpy_bind(eqn, args) if args is not None else None  # (by NumPy where XLA's to the bit)
            if outs is None:
                outs = eqn.primitive.bind(*vals, **eqn.params)
                outs = [np.asarray(o) for o in (outs if eqn.primitive.multiple_results else [outs])]
        elif name == "call_hi_primitive":  # jax >= 0.12 (e.g. searchsorted): its expansion into primitives
            closed = _hi_expand(eqn)
            outs = _eval(closed, closed.consts, vals)
        elif name == "while":
            outs = _while(eqn, vals)
        elif name == "scan":
            outs = _scan(eqn, vals)
        elif name in ("shift_left", "shift_right_logical", "shift_right_arithmetic"):
            outs = [_shift(name, eqn.invars[0].aval.dtype, *vals)]
        elif name == "bitcast_convert_type":
            outs = [_bitcast(vals[0], eqn.invars[0].aval.dtype, eqn.params["new_dtype"])]
        elif name in tuple(_NP_CMP) + ("max", "min") and eqn.invars[0].aval.dtype == np.uint64:  # unsigned compares
            f = (lambda x, y: icmp(name, x, y, True)) if name in _NP_CMP else (lambda x, y: imaxmin(name, x, y, True))
            outs = [_ew(f, *vals)]
        elif name in ("lt_to", "le_to", "eq_to") and not jax.dtypes.issubdtype(eqn.invars[0].aval.dtype, np.floating) and \
                eqn.invars[0].aval.dtype != np.uint64:  # integers: their order is the total one (keys would mix the bits
            outs = [_RULES[name[:-3]](*vals)]  # of integers carried as doubles with integer constants: searchsorted)
        elif name in _UORDER and any(v.aval.dtype == np.uint64 for v in eqn.invars) and \
                not any(isinstance(v, (_Mem, _Split, _Scatter)) for v in vals):  # uint64 ordered as int64 of the top
            flipped = [_uflip(v) if u.aval.dtype == np.uint64 else v for v, u in zip(vals, eqn.invars)]  # bit flipped
            res = _scatter_op(eqn, flipped, lambda x, y: maxmin(name[8:], x, y)) if name.startswith("scatter") else \
                _RULES[name](*flipped, **eqn.params)
            outs = [_uflip(o, True) if u.aval.dtype == np.uint64 else o
                    for o, u in zip(res if eqn.primitive.multiple_results else [res], eqn.outvars)]
        elif name == "sign" and eqn.invars[0].aval.dtype == np.uint64:  # (uint64 nodes hold signed bits)
            outs = [_ew(lambda x: int(int(x) != 0) if is_c(x) else iselect(icmp("eq", x, 0), 1, 0), vals[0])]
        elif name == "scatter" and eqn.params["update_jaxpr"] is not None:  # x.at[i].apply(f): f of the old value
            up = eqn.params["update_jaxpr"]  # (each entry in turn, as XLA's scatter with an update computation)
            outs = [_scatter_op(eqn, vals, lambda x, y: _eval(up, eqn.params["update_consts"], [np.asarray(x, dtype=object),
                                                                                             np.asarray(y, dtype=object)])[0][()])]
        elif name == "convert_element_type":
            outs = [_convert(vals[0], eqn.invars[0].aval.dtype, eqn.params["new_dtype"])]
        elif name in _BITS:
            outs = [_bits(name, eqn.invars[0].aval.dtype, *vals)]
        elif name in _MOVE and not any(_symbolic(v) for v in vals[_INDEX.get(name, slice(0, 0))]):
            outs = _move(eqn, vals)
        elif name == "gather":  # (also from an input in memory)
            outs = [_gather(eqn, vals)]
        elif name in ("dynamic_slice", "dynamic_update_slice"):
            outs = [_dynamic_slice(eqn, vals)]
        elif name == "reduce_precision":
            dt = eqn.invars[0].aval.dtype  # (16-bit floats are held as float32)
            outs = [_round(vals[0], np.float32 if np.dtype(dt) in _FLOATS16 else dt, eqn.params["exponent_bits"],
                           eqn.params["mantissa_bits"], flush=True)]
        elif name in ("scatter-max", "scatter-min", "scatter-mul"):
            outs = [_scatter_op(eqn, vals, (lambda x, y: maxmin("max", x, y)) if name == "scatter-max" else
                                (lambda x, y: maxmin("min", x, y)) if name == "scatter-min" else mul)]
        elif name == "scatter":
            outs = [_scatter(eqn, vals)]
        elif name == "cond":
            outs = _cond(eqn, vals)
        elif name == "conv_general_dilated":
            outs = [_bilinear(eqn, vals)]
        elif name in _LINALG:
            outs = _batched(_LINALG[name], eqn, vals)
        elif name == "custom_linear_solve":  # its solve (derivatives are resolved in the jaxpr)
            n, solve = eqn.params["const_lengths"], eqn.params["jaxprs"].solve
            outs = _eval(solve, solve.consts, vals[n.matvec + n.vecmat:][:n.solve] + vals[sum(n):])
        elif name == "scatter-add" and not _symbolic(vals[1]):
            outs = [_linear(eqn, vals)]
        elif name == "scatter-add":
            outs = [_scatter(eqn, vals)]
        elif name in ("div", "rem") and np.issubdtype(eqn.outvars[0].aval.dtype, np.integer):  # (as XLA, also / 0)
            outs = [_ew(lambda a, b: _push(lambda x, y: _idivrem(x, y, name == "rem", eqn.outvars[0].aval.dtype), a, b), *vals)]
        elif name == "dot_general" and eqn.outvars[0].aval.dtype == bool:  # (booleans: an or of ands; masks as 0/1, the
            num = [_ew(lambda m: b2f(m) if _isb(m) else float(m) if isinstance(m, (bool, np.bool_)) else m, v) for v in vals]
            outs = [_ew(lambda s: cmp("ne", s, 0.0), np.asarray(_dot_general(*num, **eqn.params), dtype=object))]  # sum != 0)
        elif name in _RULES:
            outs = _RULES[name](*vals, **eqn.params)
            outs = outs if eqn.primitive.multiple_results else [outs]
        elif (impl := _library(eqn)) is not None:  # a library's primitive written in jax (equinox's unvmap_any, ...)
            outs = _eval(impl, impl.consts, vals)
        else:  # poison: an error only if an output depends on it (see trace)
            outs = [np.full(v.aval.shape, mk("unsupported", name, source_info_util.summarize(eqn.source_info)), dtype=object)
                    for v in eqn.outvars]
        for var, out in zip(eqn.outvars, outs):
            if isinstance(out, (_Mem, _Split, _Scatter)):  # (from a call; keys passed on)
                env[var] = out
                continue
            if not _is_key(var.aval) and not (isinstance(out, np.ndarray) and out.dtype != object):  # the values of the
                if jax.dtypes.issubdtype(var.aval.dtype, np.floating) and np.dtype(var.aval.dtype) not in _FLOATS and \
                        name not in _MOVE:  # (float8 and narrower: no rounding to them; read, they are exact as float32)
                    raise NotImplementedError(f"kernelax: '{name}' with {np.dtype(var.aval.dtype).name} results (floats: float16, "
                                              "bfloat16, float32, float64)")
                if _vtype(var.aval.dtype) == "s":  # dtype: rounded to float32, or as doubles (exact; only if there are
                    out = _ew(tof32, out)  # float32 values at all)
                    if var.aval.dtype in (np.float16, jax.dtypes.bfloat16) and name not in _MOVE and _symbolic(out):  # (then to
                        out = _round(out, np.float32, *((5, 10) if var.aval.dtype == np.float16 else (8, 7)))  # the format)
                elif dag._F32:
                    out = _ew(tof64, out)
                if var.aval.dtype == np.float64 and not dag._F64 and _symbolic(out):
                    dag._F64.append(True)
            if not _is_key(var.aval) and np.issubdtype(var.aval.dtype, np.integer):  # integers wrapped to their dtype:
                b, sg = _ibits(var.aval.dtype)  # nodes, constant leaves, and results of arithmetic that may overflow
                wrap = name in _IARITH and b <= 32  # (doubles: exact below 2^53, so wrapped after; 64 bits: as they are)
                given = {x.id for v in eqn.invars if not isinstance(v, jcore.Literal) and not isinstance(env.get(v), (_Mem, _Split, _Scatter))
                         for x in np.asarray(env[v], dtype=object).ravel() if isinstance(x, Node)} if wrap else set()  # (an operand's value as it
                kept = {x.id for v in eqn.invars if not isinstance(v, jcore.Literal) and v.aval.dtype == var.aval.dtype and
                        not isinstance(env.get(v), (_Mem, _Split, _Scatter)) for x in np.asarray(env[v], dtype=object).ravel()
                        if isinstance(x, Node)}  # (an operand's node of the same dtype, passed on: wrapped already;
                # stop_gradient in vmap's select_n(index == k, stop_gradient(x), x), which then folds)
                out = _ew(lambda v: _push(lambda c: c if isinstance(c, Node) and c.id in kept else
                                         iwrap(c, b, sg) if _isi(c) or is_c(c) and not isinstance(c, (bool, np.bool_)) else
                                         to_float(iwrap(c, b, sg)) if wrap and isinstance(c, Node) and not _isb(c) and
                                         c.id not in given else c, v),  # is, e.g. x + 0: of the dtype already)
                          out) if _symbolic(np.asarray(out, dtype=object)) else out
            if not isinstance(var, jcore.DropVar):
                env[var] = np.asarray(out, dtype=object) if _is_key(var.aval) else _norm(out, var.aval)
        if len(dag._table) > MAX_NODES:
            raise NotImplementedError(
                f"kernelax: the function unrolls into more than {MAX_NODES} scalar operations, here at '{name}' "
                f"({source_info_util.summarize(eqn.source_info)}). Large arrays, long scans with xs or ys, and table "
                "lookups do not suit a scalar kernel: keep them in JAX (KERNELAX_MAX_NODES, for GPU kernels KERNELAX_GPU_MAX_NODES, raises "
                "the limit).")
    return [read(v) if keep or not isinstance(read(v), _Scatter) else read(v).dense() for v in jaxpr.outvars]


def _flat(output):  # the nodes and constants of an output (of a sample sum: its positions and values)
    return [x for e in output.entries for x in e] if isinstance(output, _Scatter) else list(np.asarray(output, dtype=object).flat)


_FLOATS16 = {np.dtype(np.float16), np.dtype(jax.dtypes.bfloat16)}
_FLOATS = _FLOATS16 | {np.dtype(np.float32), np.dtype(np.float64)}  # floats computed in (_eval)


def _vtype(dtype):
    """The type of the DAG values of a dtype (Node.t): float32 for floats of up to 32 bits, else double."""
    return "s" if jax.dtypes.issubdtype(dtype, np.floating) and np.dtype(dtype).itemsize <= 4 else "d"


_wide = lambda dtype: jax.dtypes.issubdtype(dtype, np.integer) and np.dtype(dtype).itemsize == 8  # (exact as doubles only
# to 2^53; keys: not integers)


def trace(fun, *example_args, closed=None, uniform=None, reduce=None, keep_nans=False):
    """Trace ``fun(*example_args)`` (or its jaxpr `closed`, if given) into the scalar DAG.

    uniform[k]: input leaf k is the same for all samples; from MEM_MIN entries on it stays in memory (_Mem).
    reduce[j]: output leaf j is summed over the samples (a sum at sample positions stays one: _Scatter).
    keep_nans: x * 0, x - x and 0 / x are not folded to 0 (NaN and inf propagate as in IEEE; dag._KEEP_NANS).
    Returns (inputs, outputs, out_tree): one object array of input nodes per input leaf (empty entries for
    inputs in memory), one object/numeric array per output leaf (or a _Scatter: a sum over the samples).
    """
    dag._table.clear()
    dag._WEIGHT[0] = 0
    _ROLLED.clear()
    dag._LEAVES.clear()
    dag._IN.clear()
    dag._F32.clear()
    dag._F64.clear()
    dag._KEEP_NANS[:] = [True] if keep_nans else []
    _DEPTH[0] = 0
    if closed is None:
        with TRACING(True):
            closed = jax.make_jaxpr(fun)(*example_args)
    closed = realify(closed)  # (complex values inside: pairs of real ones)
    leaves = jax.tree.leaves(example_args)
    wide = [_wide(np.asarray(leaf).dtype) for leaf in leaves]  # 64-bit integers: passed by their bits (primitive.to_kernel)
    dag._IN.update((k, "s") for k, leaf in enumerate(leaves) if _vtype(np.asarray(leaf).dtype) == "s")
    mem = [bool(uniform and uniform[k]) and np.size(leaf) >= MEM_MIN for k, leaf in enumerate(leaves)]
    inputs = [np.empty(np.shape(leaf), object) if m else np.array([mk("in", k, i) for i in range(np.size(leaf))], dtype=object)
              .reshape(np.shape(leaf)) for k, (leaf, m) in enumerate(zip(leaves, mem))]
    values = [_Mem(k, np.arange(np.size(leaf)).reshape(np.shape(leaf)), "bool" if np.asarray(leaf).dtype == bool else "bits" if w else None)
              if m else _ew(_mask, x) if np.asarray(leaf).dtype == bool
              else _ew(lambda n: mk("fbits", n), x) if w else x
              for k, (x, leaf, m, w) in enumerate(zip(inputs, leaves, mem, wide))]  # bools: masks
    with jax.ensure_compile_time_eval(), jax.default_device(jax.devices("cpu")[0]):  # concrete binds (also when called
        # while tracing: lazy kernel builds) on the CPU: on a GPU each would be a compiled program, a launch and a copy
        # back (diffrax_stiff: 13 000 of them, 2.7 s of a 5 s trace)
        outputs = [o.nodes() if isinstance(o, _LAZY) else o for o in _eval(closed, closed.consts, values, keep=True)]
        outputs = [o.dense() if isinstance(o, _Scatter) and not (reduce and reduce[j]) else o for j, o in enumerate(outputs)]
        # (a sum at sample positions that is not summed over the samples: per sample)
    outputs = [_ew(lambda x, d=v.aval.dtype: np.asarray(x, d).view(np.float64).item() if is_c(x) else mk("bitsf64", to_int(x)), o)
               if _wide(v.aval.dtype) and not isinstance(o, _Scatter) else o for o, v in zip(outputs, closed.outvars)]
    stack, seen = [x for o in outputs for x in _flat(o) if isinstance(x, Node)], set()
    while stack:  # unsupported primitives the outputs depend on, also through loop bodies
        order = schedule(stack, seen)
        seen.update(n.id for n in order)
        bad = [n.args for n in order if n.op == "unsupported"]
        if bad:
            raise NotImplementedError(f"kernelax: primitive '{bad[0][0]}' is not supported on values computed per "
                                      f"sample (at {bad[0][1]})")
        stack = [r for n in order for r in hidden(n)]
    out_tree = jax.tree.structure(jax.eval_shape(fun, *example_args))
    return inputs, outputs, out_tree
