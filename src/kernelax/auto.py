# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""jit: ``jax.jit`` that runs the sample-parallel parts of a program as C (CPU) or CUDA (GPU) kernels.

The program's jaxpr (calls and one-step scans inlined, dead code removed) is split along the sample axis: axis 0 of
size N, N the leading length of large arrays whose regions cover the most data (`_axis`). An equation is row-wise if
row i of its outputs depends only on row i of its row operands and on uniform operands (`_rule`): elementwise ops,
shape ops that keep the rows, reductions and contractions over other axes, batched linear algebra, gathers batched
over axis 0, and ``while``, ``scan`` and ``cond`` whose bodies are row-wise per sample (`_control`; under vmap, a
while has a batched predicate and a scan's xs and ys have their rows on axis 1). Row-wise equations form regions; a
region ends where the rest of the program reads one of its outputs as a whole. A region is a per-sample function
(its equations with the row axis removed from their parameters) that runs as a kernelax kernel or stays in XLA, as a
cost model expects (_worth). A new program's first call builds its kernels and runs with them (build_first; beyond
BUILD_FIRST_WAIT it runs in XLA while they are built in a thread); trials against XLA's program then decide
(_Entry). Kernels in the index on disk load without trace, emit or compile. While loops coupled over the samples
(any sample active, one step counter: equinox's under vmap) become one loop per sample where proved (decouple.py).
Jaxpr helpers (inlining, folding, evaluation): jaxprs.py."""

import atexit
import collections
import functools
import hashlib
import math
import os
import sys
import threading
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core as jcore

from kernelax.build import BACKGROUND, CC, JOBS, Flag, GccTimeout, _cc_version, env, no_gc, thread_map
from kernelax.cpu import NT, _VERSION, _hash_consts, _read, _update
from kernelax import frontend, internals, gpu as gpu_mod
from kernelax.decouple import _decouple, lanes_p
from kernelax.frontend import _data, _is_key
from kernelax.jaxprs import (_CUM, _EW, _FN, _INLINED, _LAYOUT, _NESTED, _REDUCE, _bind, _dce, _fold, _inline, _needs,
                         _note_platforms, _platform_default, _run, _var)
from kernelax.primitive import Spec, _opts, _weak, bind, call_p, from_kernel, sample_p, statics, to_kernel

_BATCHED = {"lu": 3, "cholesky": 3, "eigh": 3, "qr": 3, "geqrf": 3, "householder_product": 2, "triangular_solve": 3,
            "lu_pivots_to_permutation": 2, "eig": 3, "schur": 3}  # linear algebra batched over leading axes: least operand rank


def _less(dims, drop=0):  # axis numbers without `drop`, shifted to the per-sample array
    return tuple(int(d) - (d > drop) for d in dims if d != drop)


def _rule(eqn, N, sums=False, inner=False):
    """_row_rule, without sums over the samples of integers: the kernels accumulate in doubles (not exact beyond 2^53,
    no wrap-around as XLA's integer sums); those sums stay in XLA."""
    rule = _row_rule(eqn, N, sums, inner)
    if rule is not None and any(f in ("sum", "osum", "rsum") and not jax.dtypes.issubdtype(v.aval.dtype, np.inexact)
                                for v, f in zip(eqn.outvars, rule[2])):
        return None
    return rule


def _row_rule(eqn, N, sums=False, inner=False):
    """(per-sample params, role per operand, form per output) if eqn is row-wise, else None. Roles: "row" (the
    sample's row of the operand: rows on axis 0), "row1" (rows on axis 1), "uni" (the whole operand, the same
    for all samples), "uni0" (the operand without its leading axis of size 1). Forms of values: "row", "row1"
    or "whole" (the same for all samples); "row@k" / "row2@k": the sample axes from axis k on (_behind)."""
    if sums and not isinstance(N, tuple) and eqn.primitive.name == "reduce_sum":  # a sum over the rows (inner loops)
        a, axes = eqn.invars[0].aval, tuple(eqn.params["axes"])
        for k in (k for k in range(a.ndim) if a.shape[k] == N and k in axes):
            return dict(eqn.params, axes=_less(tuple(d for d in axes if d != k), k)), [_rows_at(N, k)], ["rsum"]
    if sums and not isinstance(N, tuple) and eqn.primitive.name == "dot_general":  # a contraction over the rows
        ins = [v.aval for v in eqn.invars]
        at = lambda a: [k for k in range(a.ndim) if a.shape[k] == N] if hasattr(a, "shape") else []
        pair = _pair(eqn.params, ins, at, 1)
        if pair is not None and pair[2] is None:
            (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
            kl, kr, _ = pair
            dn = ((_less(lc, kl), _less(rc, kr)), (_less(lb, kl), _less(rb, kr)))
            return dict(eqn.params, dimension_numbers=dn), [_rows_at(N, kl), _rows_at(N, kr)], ["rsum"]
    rule = _rule2(eqn, N) if isinstance(N, tuple) else _rule1(eqn, N, groups=not inner)  # (inner axes: rows k at a
    return rule if rule is not None else _behind(eqn, N)  # time are rows behind other axes, not k rows per sample)


def _rule1(eqn, N, groups=True):  # _rule for one sample axis
    name, p = eqn.primitive.name, eqn.params
    ins, outs = [v.aval for v in eqn.invars], [v.aval for v in eqn.outvars]
    rows = lambda a: hasattr(a, "shape") and a.ndim > 0 and a.shape[0] == N
    if name in ("while", "scan", "cond"):
        return _control(eqn, N, rows, behind=not groups)
    if name in ("transpose", "stack", "broadcast_in_dim", "reshape", "squeeze"):
        rule = _moves(name, p, ins, outs, N, rows)
        if rule is not None:
            return rule
    if name in ("scatter", "scatter-add") and (params := _uscatter(p, ins, 1, rows)) is not None:
        return params, ["row", "uni", "row"], ["row"]
    if name in ("scatter", "scatter-add") and (params := _bscatter(p, ins, 1, rows)) is not None:
        return params, ["row", "row", "row"], ["row"]
    group = lambda a: groups and a.ndim and a.shape[0] > N and a.shape[0] % N == 0
    if groups and name == "scatter-add" or outs and all(group(a) for a in outs) or name == "reshape" and group(ins[0]) and rows(outs[0]):
        return _groups(name, p, ins, outs, N, rows)
    if not outs or not all(rows(a) for a in outs):
        return None
    if name == "random_split" and ins[0].ndim == 0 and tuple(p["shape"]) == (N,) and _foldlike(ins[0]):
        return {_FN: _split_row}, ["uni"], ["row"]  # (a key the same for all samples split into one per sample: a
        # rollout's per step; one sample axis: its index is the row)
    rule = _rows(name, p, ins, outs[0], N, rows)
    return None if rule is None else rule + (["row"] * len(outs),)


def _rule2(eqn, sig):
    """_rule for a sample axis of two leading axes sig = (N, Q), as nested vmaps make them (elements x quadrature
    points, particles x neighbours, paths x steps, ...); sample s is (s // Q, s % Q). Forms: "row2" (the sample's
    entry of an array with the leading axes sig), "orow" (row s // Q of an array (N, ...): the same for the Q
    samples of a row; computed per sample, or read from the whole array), "osum" (a sum over Q: an array (N, ...),
    each sample adding into row s // Q). Roles: those forms, "orow1" (an operand (N, 1, ...)), "inner" / "inner1"
    (row s % Q of an operand (Q, ...) / (1, Q, ...): the same for all rows), "uni", "uni0", "uni00" (the operand
    without its leading axes of size 1)."""
    name, p = eqn.primitive.name, eqn.params
    ins, outs = [v.aval for v in eqn.invars], [v.aval for v in eqn.outvars]
    N, Q = sig
    rows = lambda a: hasattr(a, "shape") and tuple(a.shape[:2]) == sig
    cut = lambda dims: tuple(int(d) - 2 for d in dims)  # axis numbers >= 2, in the per-sample array
    if name in ("while", "scan", "cond"):
        return _control(eqn, sig, rows)
    if name == "reduce_sum" and rows(ins[0]) and 1 in p["axes"] and 0 not in p["axes"]:  # a sum over Q
        return dict(p, axes=cut(a for a in p["axes"] if a > 1)), ["row2"], ["osum"]
    if outs and all(a.ndim and a.shape[0] == N and not rows(a) for a in outs) and not any(rows(a) for a in ins):
        rows1 = lambda a: a.ndim > 0 and a.shape[0] == N  # per row: "orow"
        rule = _moves(name, p, ins, outs, N, rows1) if name in ("transpose", "squeeze", "reshape") else None
        rule = rule if rule and set(rule[1] + rule[2]) == {"row"} else _rows(name, p, ins, outs[0], N, rows1)
        return None if rule is None else (rule[0], ["orow" if r == "row" else r for r in rule[1]], ["orow"] * len(outs))
    if not outs or not all(rows(a) for a in outs):
        return None
    out, row2 = outs[0], lambda r: None if r is None else (r[0], r[1], ["row2"] * len(outs))
    lead = lambda a: "row2" if rows(a) else "orow1" if tuple(a.shape[:2]) == (N, 1) else \
        "inner1" if tuple(a.shape[:2]) == (1, Q) else "uni00" if tuple(a.shape[:2]) == (1, 1) else None
    if name in _EW:
        roles = ["uni" if a.ndim == 0 else lead(a) if a.ndim == out.ndim else None for a in ins]
        return None if None in roles else (p, roles, ["row2"])
    if name in _RANDOM and ins and rows(ins[0]) and all(rows(a) or a.ndim == 0 for a in ins):
        return p, ["row2" if rows(a) else "uni" for a in ins], ["row2"] * len(outs)
    if name == "broadcast_in_dim" and len(ins) == 1:
        bd, a = tuple(p["broadcast_dimensions"]), ins[0]
        new = dict(p, shape=tuple(p["shape"][2:]))
        if 0 not in bd and 1 not in bd:
            return dict(new, broadcast_dimensions=cut(bd)), ["uni"], ["row2"]
        if bd[:2] == (0, 1) and lead(a):
            return dict(new, broadcast_dimensions=cut(bd[2:])), [lead(a)], ["row2"]
        if bd[0] == 0 and 1 not in bd and a.shape[0] in (N, 1):  # (N, ...) or (1, ...): the same for all q
            return dict(new, broadcast_dimensions=cut(bd[1:])), ["orow" if a.shape[0] == N else "uni0"], ["row2"]
        if bd[0] == 1 and 0 not in bd and a.shape[0] == Q:  # (Q, ...): the same for all rows
            return dict(new, broadcast_dimensions=cut(bd[1:])), ["inner"], ["row2"]
        return None
    if name == "iota":
        return (dict(p, shape=tuple(p["shape"][2:]), dimension=p["dimension"] - 2), [], ["row2"]) if p["dimension"] > 1 else None
    if name == "gather":
        dn, (op, idx) = p["dimension_numbers"], ins
        ob, ib = tuple(dn.operand_batching_dims), tuple(dn.start_indices_batching_dims)
        if rows(idx) and not ib and 0 not in dn.offset_dims and 1 not in dn.offset_dims:  # into a uniform operand
            return row2((dict(p, dimension_numbers=dn._replace(offset_dims=cut(dn.offset_dims))), ["uni", "row2"]))
        if rows(op) and rows(idx) and ob[:2] == (0, 1) and ib[:2] == (0, 1) and not {0, 1} & set(dn.offset_dims):  # vmapped
            new = dn._replace(offset_dims=cut(dn.offset_dims), collapsed_slice_dims=cut(dn.collapsed_slice_dims),
                              start_index_map=cut(dn.start_index_map), operand_batching_dims=cut(ob[2:]),
                              start_indices_batching_dims=cut(ib[2:]))
            return row2((dict(p, dimension_numbers=new, slice_sizes=tuple(p["slice_sizes"][2:])), ["row2", "row2"]))
        params = _ugather(p, ins, 2, rows)
        return None if params is None else row2((params, ["row2", "uni"]))
    if name in ("scatter", "scatter-add") and (params := _uscatter(p, ins, 2, rows)) is not None:
        return params, ["row2", "uni", "row2"], ["row2"]
    if name in ("scatter", "scatter-add") and (params := _bscatter(p, ins, 2, rows)) is not None:
        return params, ["row2", "row2", "row2"], ["row2"]
    if not rows(ins[0]):
        return None
    above = lambda dims: all(d > 1 for d in dims)
    if name == "reshape" and p["dimensions"] is None and tuple(p["new_sizes"][:2]) == sig:
        return row2((dict(p, new_sizes=tuple(p["new_sizes"][2:])), ["row2"]))
    if name == "transpose" and tuple(p["permutation"][:2]) == (0, 1):
        return row2((dict(p, permutation=cut(p["permutation"][2:])), ["row2"]))
    if name in _REDUCE and above(p["axes"]):
        return row2((dict(p, axes=cut(p["axes"])), ["row2"]))
    if name in _CUM | {"split", "unstack"} and p["axis"] > 1:
        return row2((dict(p, axis=p["axis"] - 2), ["row2"]))
    if name in ("rev", "squeeze") and above(p["dimensions"]):
        return row2((dict(p, dimensions=cut(p["dimensions"])), ["row2"]))
    if name == "slice" and tuple(p["start_indices"][:2]) == (0, 0) and tuple(p["limit_indices"][:2]) == sig and \
            tuple((p["strides"] or (1, 1))[:2]) == (1, 1):
        return row2((dict(p, start_indices=p["start_indices"][2:], limit_indices=p["limit_indices"][2:],
                          strides=p["strides"] and p["strides"][2:]), ["row2"]))
    if name in ("concatenate", "sort") and p["dimension"] > 1 and all(rows(a) for a in ins):
        return row2((dict(p, dimension=p["dimension"] - 2), ["row2"] * len(ins)))
    if name == "pad" and all(tuple(c) == (0, 0, 0) for c in p["padding_config"][:2]):
        return row2((dict(p, padding_config=p["padding_config"][2:]), ["row2", "uni"]))
    if name in _BATCHED and all(rows(a) and a.ndim >= _BATCHED[name] + 1 for a in ins):
        return row2((p, ["row2"] * len(ins)))
    if name == "fft" and rows(ins[0]) and ins[0].ndim > len(p["fft_lengths"]) + 1:
        return row2((p, ["row2"]))
    if name == "dot_general":
        (lc, rc), (lb, rb) = p["dimension_numbers"]
        if tuple(lb[:2]) == (0, 1) and tuple(rb[:2]) == (0, 1) and rows(ins[1]):
            return row2((dict(p, dimension_numbers=((cut(lc), cut(rc)), (cut(lb[2:]), cut(rb[2:])))), ["row2", "row2"]))
        if not lb and above(lc):
            return row2((dict(p, dimension_numbers=((cut(lc), tuple(rc)), ((), ()))), ["row2", "uni"]))
    return None


def _moves(name, p, ins, outs, N, rows):
    """Shape ops that move the rows between axis 0 and 1 (as vmap does around scans and stacks)."""
    form, rows1 = ("row", "row1"), lambda a: a.ndim > 1 and a.shape[1] == N
    ax = 0 if rows(ins[0]) else 1 if rows1(ins[0]) else None
    if ax is None:
        return None
    if name == "transpose" and p["permutation"].index(ax) < 2:
        return dict(p, permutation=_less(p["permutation"], ax)), [form[ax]], [form[p["permutation"].index(ax)]]
    if name == "stack" and ax == 0 and all(rows(a) for a in ins) and p["axis"] <= 1:  # rows to axis 1 if stacked on 0
        return dict(p, axis=p["axis"] - (p["axis"] > 0)), ["row"] * len(ins), [form[p["axis"] == 0]]
    if name == "broadcast_in_dim" and len(ins) == 1 and ax == 0 and p["broadcast_dimensions"][0] == 1 and rows1(outs[0]):
        bd = p["broadcast_dimensions"]
        return dict(p, shape=_sample(p["shape"], "row1"), broadcast_dimensions=_less(bd[1:], 1)), ["row"], ["row1"]
    if name == "squeeze" and ax not in p["dimensions"]:
        return dict(p, dimensions=_less(p["dimensions"], ax)), [form[ax]], [form[ax - sum(d < ax for d in p["dimensions"])]]
    if name == "reshape" and p["dimensions"] is None:  # (1, N, ...) -> (N, ...) and back
        new = p["new_sizes"]
        if ax == 1 and ins[0].shape[0] == 1 and new[0] == N:
            return dict(p, new_sizes=tuple(new[1:])), ["row1"], ["row"]
        if ax == 0 and len(new) > 1 and new[:2] == (1, N):
            return dict(p, new_sizes=(1,) + tuple(new[2:])), ["row"], ["row1"]
    return None


def _groups(name, p, ins, outs, N, rows):
    """Values with k rows per sample (form "row*k": N k rows, e.g. an element's entries flattened for an assembly):
    elementwise ops, broadcasts and reshapes between them and rows, and scatter-add of rows into a uniform array
    (form "sum": the base plus the sum of all samples' scatters, accumulated in the kernel)."""
    form = lambda a: "row" if rows(a) else f"row*{a.shape[0] // N}" if a.ndim and a.shape[0] > N and a.shape[0] % N == 0 else None
    if name == "scatter-add":
        dn, (_, idx, upd) = p["dimension_numbers"], ins
        if dn.operand_batching_dims or 0 in dn.update_window_dims or form(idx) is None or form(idx) != form(upd):
            return None
        if form(idx) == "row":
            dn = dn._replace(update_window_dims=_less(dn.update_window_dims, -1))
        return dict(p, dimension_numbers=dn), ["zero", form(idx), form(idx)], ["sum"]
    out, f = outs[0], form(outs[0])
    if name in _EW:
        roles = ["uni" if a.ndim == 0 or a.ndim == out.ndim and a.shape[0] == 1 else f if a.shape == out.shape else None
                 for a in ins]
        return None if None in roles else (p, roles, [f])
    if name == "broadcast_in_dim" and len(ins) == 1:
        bd, a = p["broadcast_dimensions"], ins[0]
        role = "uni" if 0 not in bd else f if form(a) == f and bd[0] == 0 else None
        return None if role is None else (dict(p, shape=_sample(p["shape"], f)), [role], [f])
    if name == "reshape" and p["dimensions"] is None and form(ins[0]) is not None:
        a = ins[0]
        if np.prod(_sample(a.shape, form(a))) == np.prod(_sample(out.shape, f)):  # samples are contiguous blocks
            return dict(p, new_sizes=_sample(p["new_sizes"], f)), [form(a)], [f]
    return None


_RANDOM = frozenset("""random_wrap random_unwrap random_bits random_split random_fold_in random_seed random_clone
random_gamma random_poisson""".split())  # (params per
# key: under vmap the keys' shape leads; per sample the same params)


def _foldlike(a):
    """Whether key s of random_split(k, (N,)) is fold_in(k, s) for keys of aval a: threefry keys with
    threefry_partitionable (both threefry2x32(k, 0, s))."""
    impl = getattr(getattr(a, "dtype", None), "_impl", None)
    return bool(jax.config.jax_threefry_partitionable) and impl is not None and impl.name == "threefry2x32"


def _split_row(k):  # the sample's row of random_split(k, (N,)) (_foldlike)
    return [jax.random.fold_in(k, sample_p.bind())]


def _rows(name, p, ins, out, N, rows):  # (params, roles) of equations whose outputs all have rows on axis 0
    if name in _EW:
        roles = ["uni" if a.ndim == 0 else None if a.ndim != out.ndim else "row" if rows(a) else
                 "uni0" if a.shape[0] == 1 else None for a in ins]
        return None if None in roles else (p, roles)
    if name == "broadcast_in_dim" and len(ins) == 1:
        bd, a = p["broadcast_dimensions"], ins[0]
        new = dict(p, shape=p["shape"][1:])
        if 0 not in bd:
            return dict(new, broadcast_dimensions=_less(bd, -1)), ["uni"]
        if bd[0] == 0:
            return dict(new, broadcast_dimensions=_less(bd[1:], -1)), ["row" if rows(a) else "uni0"]
        return None
    if name in _RANDOM and ins and rows(ins[0]) and all(rows(a) or a.ndim == 0 for a in ins):
        return p, ["row" if rows(a) else "uni" for a in ins]
    if name == "iota":
        return (dict(p, shape=p["shape"][1:], dimension=p["dimension"] - 1), []) if p["dimension"] != 0 else None
    if name == "gather":  # the operand may be uniform
        return _gather(p, ins, rows)
    if not all(rows(a) for a in ins[:1]):
        return None
    if name == "reshape" and (p["dimensions"] is None or p["dimensions"][0] == 0) and p["new_sizes"][0] == N:
        dims = None if p["dimensions"] is None else tuple(d - 1 for d in p["dimensions"][1:])  # (a transpose before: the
        return dict(p, new_sizes=tuple(p["new_sizes"][1:]), dimensions=dims), ["row"]  # sample axis kept first)
    if name in _REDUCE and 0 not in p["axes"]:
        return dict(p, axes=_less(p["axes"])), ["row"]
    if name in _CUM and p["axis"] != 0:
        return dict(p, axis=p["axis"] - 1), ["row"]
    if name == "tile" and p["reps"][0] == 1:
        return dict(p, reps=tuple(p["reps"][1:])), ["row"]
    if name in ("split", "unstack") and p["axis"] != 0:
        return dict(p, axis=p["axis"] - 1), ["row"]
    if name == "rev" and 0 not in p["dimensions"]:
        return dict(p, dimensions=_less(p["dimensions"])), ["row"]
    if name == "slice" and p["start_indices"][0] == 0 and p["limit_indices"][0] == N and (p["strides"] or (1,))[0] == 1:
        return dict(p, start_indices=p["start_indices"][1:], limit_indices=p["limit_indices"][1:],
                    strides=p["strides"] and p["strides"][1:]), ["row"]
    if name == "top_k" and p["axis"] % ins[0].ndim != 0:  # (along a sample's own axis: xland-minigrid's reset)
        return dict(p, axis=p["axis"] % ins[0].ndim - 1), ["row"]
    if name in ("concatenate", "sort") and p["dimension"] != 0 and all(rows(a) for a in ins):
        return dict(p, dimension=p["dimension"] - 1), ["row"] * len(ins)
    if name == "pad" and tuple(p["padding_config"][0]) == (0, 0, 0):
        return dict(p, padding_config=p["padding_config"][1:]), ["row", "uni"]
    if name in _BATCHED and all(rows(a) and a.ndim >= _BATCHED[name] for a in ins):
        return p, ["row"] * len(ins)
    if name == "fft" and rows(ins[0]) and ins[0].ndim > len(p["fft_lengths"]):  # (over the trailing axes)
        return p, ["row"]
    if name == "dot_general":
        return _dot_general(p, ins, rows)
    return None


UNIFORM_MAX = 512  # uniform outputs (entries) of control flow in a kernel: computed per sample, the first sample's kept
# (Brax' MJX frame loop: 348 entries of contact data, its rollout per sample 354x XLA; a solver's state of 1000: XLA)


def _control(eqn, N, rows, behind=False):
    """_control_forms, where the uniform outputs are few: a kernel computes them per sample and keeps the first
    sample's (control flow that passes large uniform state through, e.g. a solver's, stays in XLA)."""
    r = _control_forms(eqn, N, rows, behind)
    if r is not None and sum(v.aval.size for v, f in zip(eqn.outvars, r[2]) if f == "whole") > UNIFORM_MAX:
        return None
    return r


def _control_forms(eqn, N, rows, behind=False):
    """while (also with a batched predicate: each sample loops until its own is false), scan (xs and ys with
    rows on axis 1, as under vmap) and cond (a uniform index): operands with rows on axis 0 are rows, the others
    uniform; bodies, conditions and branches per sample (_per_sample), with uniform carries kept uniform. behind
    (the rows of an inner axis, _inner): also rows behind other axes, at the one axis of N entries (e.g. a local
    Newton solve per quadrature point under jacfwd: carries (directions, points))."""
    p, name, row = eqn.params, eqn.primitive.name, _row(N)

    def role(a, start=0):  # rows on axis `start` (xs: 1), or (behind) on the one later axis of N entries
        if start == 0 and rows(a):  # (two sample axes: rows tells)
            return row
        if start == 1 and row == "row" and getattr(a, "ndim", 0) > 1 and a.shape[1] == N:
            return "row1"
        ks = [k for k in range(start + 1, getattr(a, "ndim", 0)) if a.shape[k] == N] if behind and row == "row" else []
        return _rows_at(N, ks[0]) if len(ks) == 1 else "uni"
    roles = [role(v.aval) for v in eqn.invars]
    if all(r == "uni" for r in roles) and not (row == "row" and name == "scan" and any(
            v.aval.ndim > 1 and (v.aval.shape[1] == N or behind and N in v.aval.shape[2:]) for v in eqn.invars)):
        return None  # nothing with rows goes in (a cheap check: the per-sample bodies cost traces)
    carry_forms = lambda rs: ["whole" if r == "uni" else r for r in rs]
    if name == "while":
        nc, nb = p["cond_nconsts"], p["body_nconsts"]
        carry = roles[nc + nb:]
        body = _per_sample(p["body_jaxpr"], roles[nc:nc + nb] + carry, carry, N)
        cond = _per_sample(p["cond_jaxpr"], roles[:nc] + carry, [None], N)
        pred = cond and cond[0].out_avals[0].shape  # a per-sample batched predicate (nested vmaps): the leading axes
        if body is None or cond is None or any(tuple(a.shape[:len(pred)]) != pred for a in body[0].out_avals):  # of the carries
            return None
        return dict(p, body_jaxpr=body[0], cond_jaxpr=cond[0]), roles, carry_forms(carry)
    if name == "cond":
        if eqn.invars[0].aval.ndim:
            return None
        outs = [role(v.aval) for v in eqn.outvars]
        d = _platform_default(eqn)  # (platform_dependent: kernels run its default branch; the region's own equations,
        branches = [_per_sample(b, roles[1:], outs, N) for b in (p["branches"] if d is None else [p["branches"][d]] * len(p["branches"]))]
        # in XLA, the platform's)
        if any(b is None for b in branches):
            return None
        return dict(p, branches=tuple(b[0] for b in branches)), ["uni"] + roles[1:], carry_forms(outs)
    consts, carry, xs = map(list, p["ft_in"].update(list(range(len(roles)))).unpack())
    if row != "row" and (xs or len(eqn.outvars) != len(carry)):
        return None  # (xs and ys with two sample axes: not covered)
    x_roles = [role(v.aval, 1) if row == "row" else "uni" for v in (eqn.invars[i] for i in xs)]
    roles = [roles[i] for i in consts + carry] + x_roles
    c_roles = roles[len(consts):len(consts) + len(carry)]
    step = lambda r: "uni" if r == "uni" else _rows_at(N, _at(r)[1] - 1)  # (an element of xs: without the scan's axis)
    body = _per_sample(p["jaxpr"], roles[:len(consts) + len(carry)] + [step(r) for r in x_roles],
                       c_roles + [None] * (len(eqn.outvars) - len(carry)), N)
    if body is None:
        return None
    ys = ["whole" if f == "whole" else _rows_at(N, _at(f)[1] + 1) for f in body[1][len(carry):]]  # (behind the scan's axis)
    return dict(p, jaxpr=body[0]), roles, carry_forms(c_roles) + ys


def _dot_general(p, ins, rows):
    (lc, rc), (lb, rb) = p["dimension_numbers"]
    a, b = ins
    if lb and lb[0] == 0 and rb[0] == 0 and rows(b):  # batched over axis 0 on both sides
        dn = ((_less(lc), _less(rc)), (_less(lb), _less(rb)))
        return dict(p, dimension_numbers=dn), ["row", "row"]
    if not lb and 0 not in lc:  # axis 0 of lhs is the first free axis: output axis 0
        return dict(p, dimension_numbers=((_less(lc), tuple(rc)), ((), ()))), ["row", "uni"]
    return None


def _ugather(p, ins, P, rows):
    """gather from rows (P leading axes, all sliced and kept as the leading output axes) at a uniform index (x[:, i]
    under vmap): per sample, the same gather without the leading axes."""
    dn, (op, idx) = p["dimension_numbers"], ins
    lead = set(range(P))
    if not rows(op) or rows(idx) or tuple(dn.offset_dims[:P]) != tuple(range(P)) or tuple(p["slice_sizes"][:P]) != tuple(op.shape[:P]) \
            or lead & (set(dn.collapsed_slice_dims) | set(dn.start_index_map) | set(dn.operand_batching_dims)):
        return None
    less = lambda d: tuple(int(x) - P for x in d)
    return dict(p, slice_sizes=tuple(p["slice_sizes"][P:]), dimension_numbers=dn._replace(
        offset_dims=less(dn.offset_dims[P:]), collapsed_slice_dims=less(dn.collapsed_slice_dims),
        start_index_map=less(dn.start_index_map), operand_batching_dims=less(dn.operand_batching_dims)))


def _at(form):  # (number of sample axes, their first axis) of a form with rows, else None
    base, _, k = form.partition("@")
    return {"row": (1, 0), "row1": (1, 1), "row2": (2, 0)}.get(base) if not k else (1 + (base == "row2"), int(k))


def _rows_at(N, k):  # the form of rows from axis k on
    return ("row2" if isinstance(N, tuple) else "row1" if k == 1 else "row") + (f"@{k}" if k > 1 or k and isinstance(N, tuple) else "")


def _pair(p, ins, at, P):
    """(kl, kr, i) for a dot_general whose operands both have rows (from axes kl and kr on) paired as batch
    dimensions (from batch position i on), or as contracting dimensions (i None); else None."""
    (lc, rc), (lb, rb) = p["dimension_numbers"]
    for kl in at(ins[0]):
        for kr in at(ins[1]):
            for dl, dr, batch in ((lb, rb, True), (lc, rc, False)):
                dl, dr = [int(d) for d in dl], [int(d) for d in dr]
                if kl in dl and all(kl + t in dl and dl.index(kl + t) == dl.index(kl) + t and dr[dl.index(kl) + t] == kr + t
                                    for t in range(P)):
                    return kl, kr, dl.index(kl) if batch else None
    return None


def _behind_all(eqn, N):
    """Rows behind other axes, as vmap leaves them after stacks and contractions ((3, N, ...), then transposed back):
    elementwise ops, stack, transpose, dot_general and shape ops: the rules for each placement of the rows (a
    length can occur at several axes: 27 nodes and 27 points of an element)."""
    sig = N if isinstance(N, tuple) else (N,)
    P, name, p = len(sig), eqn.primitive.name, eqn.params
    ins, outs = [v.aval for v in eqn.invars], [v.aval for v in eqn.outvars]
    at = lambda a: [k for k in range(a.ndim - P + 1) if tuple(a.shape[k:k + P]) == sig] if hasattr(a, "shape") else []
    form = lambda k: _rows_at(N, k)
    less = lambda dims, k: tuple(int(d) - P * (d > k) for d in dims)
    one = lambda a, k: tuple(a.shape[k:k + P]) == (1,) * P  # size 1 at the rows' axes: the same for all rows
    if name in _EW and outs:
        for k in at(outs[0]):
            if all(a.ndim == 0 or a.ndim == outs[0].ndim and (k in at(a) or one(a, k)) for a in ins):
                yield p, ["uni" if a.ndim == 0 else form(k) if k in at(a) else f"uni@{form(k)}" for a in ins], [form(k)] * len(outs)
    rows = lambda k: set(range(k, k + P))
    for k in at(ins[0]) if ins and name in _REDUCE | _CUM | {"squeeze", "rev", "slice", "pad", "concatenate", "reshape", "split",
                                                         "unstack"} else []:
        if name in _CUM | {"split", "unstack"} and p["axis"] not in rows(k):
            yield dict(p, axis=p["axis"] - P * (p["axis"] > k)), [form(k)], [form(k - (name == "unstack" and p["axis"] < k))] * len(outs)
        if name == "split" and len(p["sizes"]) == 1:  # (one piece: the operand)
            yield {_FN: lambda x: [x]}, [form(k)], [form(k)]
        if name in _REDUCE and not rows(k) & set(p["axes"]):
            yield dict(p, axes=less(p["axes"], k)), [form(k)], [form(k - sum(d < k for d in p["axes"]))]
        if name in ("squeeze", "rev") and not rows(k) & set(p["dimensions"]):
            yield dict(p, dimensions=less(p["dimensions"], k)), [form(k)], [form(k - (name == "squeeze") * sum(d < k for d in p["dimensions"]))]
        if name == "slice" and all(p["start_indices"][d] == 0 and p["limit_indices"][d] == ins[0].shape[d] and
                                   (p["strides"] or [1] * ins[0].ndim)[d] == 1 for d in rows(k)):
            drop = lambda xs: xs and tuple(x for d, x in enumerate(xs) if d not in rows(k))
            yield dict(p, start_indices=drop(p["start_indices"]), limit_indices=drop(p["limit_indices"]), strides=drop(p["strides"])), \
                [form(k)], [form(k)]
        if name == "pad" and all(tuple(p["padding_config"][d]) == (0, 0, 0) for d in rows(k)):
            yield dict(p, padding_config=tuple(c for d, c in enumerate(p["padding_config"]) if d not in rows(k))), [form(k), "uni"], [form(k)]
        if name == "concatenate" and p["dimension"] not in rows(k) and all(k in at(a) for a in ins):
            yield dict(p, dimension=p["dimension"] - P * (p["dimension"] > k)), [form(k)] * len(ins), [form(k)]
        if name == "reshape" and p["dimensions"] is None:  # the rows stay between the same numbers of entries
            new, a = tuple(p["new_sizes"]), ins[0]
            for j in at(outs[0]):
                if np.prod(a.shape[:k]) == np.prod(new[:j]):
                    yield dict(p, new_sizes=_sample(new, form(j))), [form(k)], [form(j)]
    if name == "triangular_solve" and P == 1 and not at(ins[0]):  # a uniform matrix, the rows in the right-hand sides
        kb = ins[1].ndim - 1 if p["left_side"] else ins[1].ndim - 2  # (vmap of a solve: batching folds them into its
        if kb in at(ins[1]):  # columns, its rows with left_side False): per sample one of them
            yield {_FN: lambda a, b: [jnp.squeeze(eqn.primitive.bind(a, jnp.expand_dims(b, kb), **p), kb)]}, ["uni", form(kb)], [form(kb)]
    if name == "broadcast_in_dim" and len(ins) == 1:  # the rows' axes stay together: from axis bd[k] on
        bd = tuple(int(d) for d in p["broadcast_dimensions"])
        for k in at(ins[0]):
            j = bd[k]
            if bd[k:k + P] == tuple(range(j, j + P)):
                rest = [d for i, d in enumerate(bd) if not k <= i < k + P]
                yield dict(p, shape=_sample(p["shape"], form(j)), broadcast_dimensions=less(rest, j)), [form(k)], [form(j)]
    if name == "stack" and ins:
        for k in sorted(set.intersection(*(set(at(a)) for a in ins))):
            a = p["axis"]
            if a <= k or a >= k + P:
                yield dict(p, axis=a - P * (a > k)), [form(k)] * len(ins), [form(k + (a <= k))]
    if name == "transpose":
        perm = tuple(int(d) for d in p["permutation"])
        for k in at(ins[0]):
            rows = tuple(range(k, k + P))
            for j in range(len(perm) - P + 1):
                if perm[j:j + P] == rows:
                    yield dict(p, permutation=less([d for d in perm if d not in rows], k)), [form(k)], [form(j)]
    if name == "gather" and P == 1:  # batched (vmap): the rows a pair of batching axes of operand and indices (the
        dn, (op, idx) = p["dimension_numbers"], ins  # indices' last axis: the index vector), behind other axes too
        ob, ib = [int(d) for d in dn.operand_batching_dims], [int(d) for d in dn.start_indices_batching_dims]
        bout = [d for d in range(outs[0].ndim) if d not in dn.offset_dims]  # (the output axes of the indices' axes)
        for kb, ki in zip(ob, ib):
            if op.shape[kb] == N and idx.shape[ki] == N and ki < idx.ndim - 1:
                ko = bout[ki]
                new = dn._replace(offset_dims=less(dn.offset_dims, ko), collapsed_slice_dims=less(dn.collapsed_slice_dims, kb),
                                  start_index_map=less(dn.start_index_map, kb), operand_batching_dims=less([d for d in ob if d != kb], kb),
                                  start_indices_batching_dims=less([d for d in ib if d != ki], ki))
                yield dict(p, dimension_numbers=new, slice_sizes=tuple(s for d, s in enumerate(p["slice_sizes"]) if d != kb)), \
                    [form(kb), form(ki)], [form(ko)]
        window = [d for d in range(op.ndim) if d not in dn.collapsed_slice_dims and d not in ob]  # at a uniform index: the
        for j, kb in enumerate(window):  # rows a window axis taken whole (x[..., i] under vmap: a diagonal, a pivot row)
            if op.shape[kb] == N and p["slice_sizes"][kb] == N and kb not in dn.start_index_map:
                ko = int(dn.offset_dims[j])
                new = dn._replace(offset_dims=less([d for d in dn.offset_dims if d != ko], ko),
                                  collapsed_slice_dims=less(dn.collapsed_slice_dims, kb), start_index_map=less(dn.start_index_map, kb),
                                  operand_batching_dims=less(ob, kb))
                yield dict(p, dimension_numbers=new, slice_sizes=tuple(s for d, s in enumerate(p["slice_sizes"]) if d != kb)), \
                    [form(kb), "uni"], [form(ko)]
    if name in ("scatter", "scatter-add") and P == 1:  # batched as gather; or at a uniform index into rows on a
        dn, (op, idx, upd) = p["dimension_numbers"], ins  # window axis of operand and updates
        ob, ib, uw = ([int(d) for d in x] for x in (dn.operand_batching_dims, dn.scatter_indices_batching_dims, dn.update_window_dims))
        uscat = [d for d in range(upd.ndim) if d not in uw]  # (the updates' axes of the indices' axes)
        for kb, ki in zip(ob, ib):
            if op.shape[kb] == N and idx.shape[ki] == N and ki < idx.ndim - 1 and upd.shape[uscat[ki]] == N:
                ku = uscat[ki]
                new = dn._replace(update_window_dims=less(uw, ku), inserted_window_dims=less(dn.inserted_window_dims, kb),
                                  scatter_dims_to_operand_dims=less(dn.scatter_dims_to_operand_dims, kb),
                                  operand_batching_dims=less([d for d in ob if d != kb], kb),
                                  scatter_indices_batching_dims=less([d for d in ib if d != ki], ki))
                yield dict(p, dimension_numbers=new), [form(kb), form(ki), form(ku)], [form(kb)]
        window = [d for d in range(op.ndim) if d not in dn.inserted_window_dims and d not in ob]  # (operand window axes,
        for j, kb in enumerate(window):  # in the order of update_window_dims)
            ku = uw[j] if j < len(uw) else None
            if ku is not None and op.shape[kb] == N and upd.shape[ku] == N and idx.shape and N not in idx.shape \
                    and kb not in dn.scatter_dims_to_operand_dims:
                new = dn._replace(update_window_dims=less(uw[:j] + uw[j + 1:], ku), inserted_window_dims=less(dn.inserted_window_dims, kb),
                                  scatter_dims_to_operand_dims=less(dn.scatter_dims_to_operand_dims, kb),
                                  operand_batching_dims=less(ob, kb))
                yield dict(p, dimension_numbers=new), [form(kb), "uni", form(ku)], [form(kb)]
    if name == "dot_general" and (pair := _pair(p, ins, at, P)) is not None:
        (lc, rc), (lb, rb) = p["dimension_numbers"]
        kl, kr, i = pair
        if i is not None:  # rows on both sides as batch dimensions: in the output from batch position i on
            dn = ((less(lc, kl), less(rc, kr)), (less([d for d in lb if not kl <= d < kl + P], kl),
                                                   less([d for d in rb if not kr <= d < kr + P], kr)))
            yield dict(p, dimension_numbers=dn), [form(kl), form(kr)], [form(i)]
    if name == "dot_general":
        (lc, rc), (lb, rb) = p["dimension_numbers"]
        for side, (a, c, b) in enumerate(((ins[0], lc, lb), (ins[1], rc, rb))):
            for k in at(a):
                if set(range(k, k + P)) & (set(c) | set(b)):
                    continue
                free = [d for d in range(a.ndim) if d not in c and d not in b]
                pos = len(lb) + (side and ins[0].ndim - len(lc) - len(lb)) + free.index(k)
                dn = ((less(lc, k), tuple(rc)), (less(lb, k), tuple(rb))) if side == 0 else \
                    ((tuple(lc), less(rc, k)), (tuple(lb), less(rb, k)))
                yield dict(p, dimension_numbers=dn), [form(k), "uni"] if side == 0 else ["uni", form(k)], [form(pos)]


def _behind(eqn, N):
    return next(_behind_all(eqn, N), None)


def _rules(eqn, N, sums=False, inner=False):
    """_rule and, after it, the other placements of the rows (_behind_all) for a region to try in turn."""
    first = _rule(eqn, N, sums, inner)
    return [] if first is None else [first] + [r for r in _behind_all(eqn, N) if r != first]


def _bscatter(p, ins, P, rows):
    """scatter / scatter-add batched over the P leading axes (vmap of x.at[i].set(y) with a per-sample index): per
    sample, the same scatter without the batching dimensions."""
    dn, (op, idx, upd) = p["dimension_numbers"], ins
    ob, ib = [int(d) for d in dn.operand_batching_dims], [int(d) for d in dn.scatter_indices_batching_dims]
    if not (rows(op) and rows(idx) and rows(upd)) or ob[:P] != list(range(P)) or ib[:P] != list(range(P)):
        return None
    uw = [int(d) for d in dn.update_window_dims]
    if set(range(P)) & set(uw):  # (the leading axes of the updates are the batch, not a window)
        return None
    less = lambda d: tuple(int(x) - P for x in d)
    return dict(p, dimension_numbers=dn._replace(
        update_window_dims=less(uw), inserted_window_dims=less(dn.inserted_window_dims),
        scatter_dims_to_operand_dims=less(dn.scatter_dims_to_operand_dims), operand_batching_dims=less(ob[P:]),
        scatter_indices_batching_dims=less(ib[P:])))


def _uscatter(p, ins, P, rows):
    """scatter / scatter-add into rows (P leading axes) of updates with rows at a uniform index (x.at[i].set(y)
    under vmap): per sample, the same scatter without the leading axes."""
    dn, (op, idx, upd) = p["dimension_numbers"], ins
    lead = set(range(P))
    if not (rows(op) and rows(upd)) or rows(idx) or tuple(dn.update_window_dims[:P]) != tuple(range(P)) or \
            lead & set(dn.inserted_window_dims) | lead & set(dn.scatter_dims_to_operand_dims) | lead & set(dn.operand_batching_dims) \
            or dn.scatter_indices_batching_dims:
        return None
    less = lambda d: tuple(int(x) - P for x in d)
    return dict(p, dimension_numbers=dn._replace(
        update_window_dims=less(dn.update_window_dims[P:]), inserted_window_dims=less(dn.inserted_window_dims),
        scatter_dims_to_operand_dims=less(dn.scatter_dims_to_operand_dims), operand_batching_dims=less(dn.operand_batching_dims)))


def _gather(p, ins, rows):
    dn, (_, idx) = p["dimension_numbers"], ins
    ob, ib = tuple(dn.operand_batching_dims), tuple(dn.start_indices_batching_dims)
    common = dict(offset_dims=_less(dn.offset_dims, -1), collapsed_slice_dims=_less(dn.collapsed_slice_dims),
                  start_index_map=_less(dn.start_index_map))
    if 0 in ob and ib[ob.index(0)] == 0 and rows(idx) and 0 not in dn.offset_dims:  # batched over axis 0 (vmap)
        k = ob.index(0)
        new = lax.GatherDimensionNumbers(**common, operand_batching_dims=_less(ob[:k] + ob[k + 1:], -1),
                                         start_indices_batching_dims=_less(ib[:k] + ib[k + 1:], -1))
        return dict(p, dimension_numbers=new, slice_sizes=p["slice_sizes"][1:]), ["row", "row"]
    if (params := _ugather(p, ins, 1, rows)) is not None:  # all rows sliced, axis 0 as output axis 0
        return params, ["row", "uni"]
    if rows(idx) and 0 not in ib and 0 not in dn.offset_dims:  # an index per row into a uniform operand (a lookup
        new = lax.GatherDimensionNumbers(offset_dims=_less(dn.offset_dims, -1), collapsed_slice_dims=dn.collapsed_slice_dims,
                                         start_index_map=dn.start_index_map, operand_batching_dims=ob,  # in memory
                                         start_indices_batching_dims=_less(ib, -1))  # in the kernel)
        return dict(p, dimension_numbers=new), ["uni", "row"]
    return None


def _kdata(aval):
    """The aval of the values kernels pass a value of aval as: keys by their data, complex numbers as [re, im] on a
    trailing axis (the kernels see real values: realify)."""
    if jax.dtypes.issubdtype(aval.dtype, np.complexfloating):
        return jax.core.ShapedArray(aval.shape + (2,), np.float32 if aval.dtype == np.complex64 else np.float64)
    return _data(aval)


def _to_data(x, aval):
    return jax.random.key_data(x) if _is_key(aval) else jnp.stack([jnp.real(x), jnp.imag(x)], -1) if _cplx(aval) else x


def _from_data(x, aval):
    return jax.random.wrap_key_data(x, impl=internals.key_impl(aval.dtype)) if _is_key(aval) else lax.complex(x[..., 0], x[..., 1]) \
        if _cplx(aval) else x


_cplx = lambda aval: jax.dtypes.issubdtype(aval.dtype, np.complexfloating)


def _tail(aval):  # the trailing shape of the values kernels pass (_kdata), else ()
    return tuple(_kdata(aval).shape[len(aval.shape):])


def _same_rows(c, form):
    """The row of a constant (_Const) read in a form with rows whose rows are all the same (a broadcast over the
    samples, folded), else None."""
    P, k = _at(form) or (0, 0)
    if not P or np.ndim(c.val) < k + P:
        return None
    rows = int(np.prod(c.val.shape[k:k + P]))  # (not -1: also for empty rows)
    x = np.moveaxis(np.asarray(c.val), tuple(range(k, k + P)), tuple(range(P))).reshape((rows,) + _sample(c.val.shape, form))
    return x[0] if rows and (x == x[:1]).all() else None


def _form(role):  # the form of the value a role reads ("zero": none, zeros of its shape)
    return "whole" if role in ("uni", "uni0", "uni00", "inner", "inner1") or role.startswith("uni@") else "orow" if role == "orow1" else role


def _row(N):  # the form of rows for a sample axis N (two leading axes: a tuple)
    return "row2" if isinstance(N, tuple) else "row"


def _sample(shape, form):  # the per-sample shape of a value of this form
    if form.startswith("row*"):  # k rows per sample
        return (int(form[4:]),) + tuple(shape[1:])
    if form == "orow":
        return tuple(shape[1:])
    P, k = _at(form) or (0, 0)
    return tuple(d for i, d in enumerate(shape) if not k <= i < k + P)


class _PerSample:
    """Equations evaluated per sample, on values (var, form). Steps: (eqn, params, roles, output forms); `rep`
    holds row-wise versions of values computed elsewhere (broadcasts of uniform values, transposes), computed
    per sample where a row is read. `open`: values not computed here are inputs (a region), else only the declared ones (a body)."""

    def __init__(self, rep, inputs=(), open=True, N=None):
        self.rep, self.open, self.N = rep, open, N  # N: the sample axis (a tuple (N, Q): two leading axes)
        self.steps, self.inputs, self.orig, self.form, self.base = [], list(inputs), [], {}, {}
        self.have, self.defined = set(inputs), set()

    def _need(self, v, role, todo):  # the steps and inputs that make v available for `role`, or None
        if role == "zero":
            return []
        form = _form(role)
        if not _var(v) and (form in ("whole", "orow") or not np.ndim(v.val) or _same_rows(v, form) is not None) or \
                (v, form) in self.have or (v, form) in todo:
            return []  # (constants used whole, by row s // Q or the same in all rows: in the code; else inputs)
        if not _var(v):
            todo.add((v, form))
            return [("input", (v, form))]
        if form != "whole" and v in self.rep and self.rep[v][3][self.rep[v][0].outvars.index(v)] == form:
            return self._add(*self.rep[v], todo)
        if form == "whole" and v in self.rep and self.rep[v][3] == ["whole"] and (v, form) not in self.have:  # (keys
            again = set(todo)  # split from one the same for all: in the kernel, lazily, _partition; else read)
            if (plan := self._add(*self.rep[v], again)) is not None:
                todo.update(again)
                return plan
        if form == "orow":  # row s // Q of the whole value (an input, or a uniform value of a body)
            if (v, "whole") in self.have or (v, "whole") in todo:
                return []
            form = "whole"
        if not self.open or v in self.defined:
            return None
        todo.add((v, form))
        return [("input", (v, form))]

    def _add(self, eqn, params, roles, forms, todo):
        roles = ["sum" if r == "zero" and _var(v) and ((v, "sum") in self.have or (v, "sum") in todo) else r
                 for v, r in zip(eqn.invars, roles)]  # a scatter into a sum of this region: its sum per sample
        if any(r == "zero" and _var(v) and v in self.defined for v, r in zip(eqn.invars, roles)):
            return None  # (a scatter's base computed here: the kernel adds into bases from outside, read before it runs)
        plan = []
        for v, r in zip(eqn.invars, roles):
            more = self._need(v, r, todo)
            if more is None:
                return None
            plan += more
        todo.update(zip(eqn.outvars, forms))
        return plan + [("step", (eqn, params, roles, forms))]

    def _apply(self, plan):
        if plan is None:
            return False
        for kind, x in plan:
            self.inputs.append(x) if kind == "input" else self.steps.append(x)
            self.have.update([x] if kind == "input" else zip(x[0].outvars, x[3]))
        return True

    def add(self, eqn, params, roles, forms, orig=True):
        """Adds eqn (and what it needs) if its operands are available in their roles; returns whether it did."""
        if not self._apply(self._add(eqn, params, roles, forms, set())):
            return False
        if orig:
            self.orig.append(eqn)
            self.defined.update(eqn.outvars)
            self.form.update(zip(eqn.outvars, forms))
            self.base.update((o, self.base.get(eqn.invars[0], eqn.invars[0])) for o, f in zip(eqn.outvars, forms) if f == "sum")
        return True

    def provide(self, v, role):
        """Makes v available for `role` (an output); returns whether it could."""
        return self._apply(self._need(v, role, set()))

    def __call__(self, outputs, *values):
        env, rq = dict(zip(self.inputs, values)), []

        def row_q():  # (s // Q, s % Q) of the sample s (two sample axes (N, Q))
            if not rq:
                s = sample_p.bind()
                Q = np.asarray(self.N[1], s.dtype)
                rq.extend([lax.div(s, Q), lax.rem(s, Q)])
            return rq

        def read(v, r):
            if r in ("orow", "orow1") and (_var(v) or np.ndim(v.val)):  # the row s // Q: computed per sample, or from the whole
                x = env[(v, "orow")] if (v, "orow") in env else jnp.take(v.val if not _var(v) else env[(v, "whole")], row_q()[0], axis=0, mode="clip")
                return x.reshape(v.aval.shape[2:]) if r == "orow1" else x
            if r in ("inner", "inner1"):  # row s % Q of a value the same for all rows
                x = jnp.take(v.val if not _var(v) else env[(v, "whole")], row_q()[1], axis=int(r == "inner1"), mode="clip")
                return x.reshape(v.aval.shape[2:]) if r == "inner1" else x
            if not _var(v) and _form(r) not in ("whole", "orow") and (row := _same_rows(v, _form(r))) is not None:
                return row  # (a constant the same in all rows: its row)
            whole = lambda: v.val if not _var(v) else env[(v, "whole")]
            return (jnp.zeros(v.aval.shape, v.aval.dtype) if r == "zero" else
                    whole().reshape(v.aval.shape[{"uni0": 1, "uni00": 2}[r]:]) if r in ("uni0", "uni00") else
                    whole().reshape(_sample(v.aval.shape, r[4:])) if r.startswith("uni@") else  # (size 1 at the rows of form r[4:])
                    whole() if _form(r) == "whole" else env[(v, _form(r))])
        for eqn, params, roles, forms in self.steps:
            env.update(zip(zip(eqn.outvars, forms), _bind(eqn, params, [read(v, r) for v, r in zip(eqn.invars, roles)])))

        def out(v, f):
            if f != "osum":
                return v.val if not _var(v) else env[(v, f)]
            c = env[(v, f)]  # a sample's part of a sum over Q: into row s // Q
            return jnp.zeros((self.N[0],) + c.shape, c.dtype).at[row_q()[0]].add(c)
        return tuple(out(v, f) for v, f in outputs)

    def avals(self):
        return tuple(jax.ShapeDtypeStruct(_sample(v.aval.shape, f), v.aval.dtype) for v, f in self.inputs)


def _per_sample(closed, roles, outs, N):
    """The per-sample version of a body, condition or branch (closed jaxpr) whose operands have `roles` ("row" or
    "uni") and whose outputs must have `outs` ("row", "uni" or None: either), as (closed jaxpr, output forms), or
    None: each equation row-wise or uniform. Cached while a program is planned (_INLINED: the partition asks for the
    same ones again, e.g. per deferral)."""
    cache = _INLINED.__dict__.setdefault("cache", {})
    key = ("per sample", id(closed), tuple(roles), None if outs is None else tuple(outs), repr(N))
    if key not in cache:
        cache[key] = (closed, _per_sample_now(closed, roles, outs, N))
    return cache[key][1]


def _per_sample_now(closed, roles, outs, N):
    closed = _inline(closed)
    j = closed
    _note_platforms(j)
    row = _row(N)
    ps = _PerSample({}, [(v, _form(r)) for v, r in zip(j.invars, roles)] + [(v, "whole") for v in j.constvars], open=False, N=N)
    behind = any(_form(r) not in ("whole", row) for r in list(roles) + [o for o in outs if o is not None])  # (rows
    # behind other axes, in or out: _control_forms' behind)
    for eqn in j.eqns:
        if eqn.effects:  # (side effects: no kernel's, frontend)
            return None
        if eqn.primitive is lanes_p:  # (the sample's own value: a loop of _decouple)
            a = eqn.invars[0].aval
            if not ps.add(eqn.replace(primitive=lax.convert_element_type_p), dict(new_dtype=a.dtype, weak_type=False,
                          sharding=None), [row], ["whole"]):
                return None
            continue
        rule = _rule(eqn, N)
        if all(not _var(v) or (v, "whole") in ps.have for v in eqn.invars):  # uniform
            ps.add(eqn, eqn.params, ["uni"] * len(eqn.invars), ["whole"] * len(eqn.outvars))
            if rule is not None:  # (unused uniform steps cost nothing: the kernel keeps what the outputs need; behind:
                ps.rep.update((v, (eqn,) + rule) for v, f in zip(eqn.outvars, rule[2])  # also with rows behind
                              if f in (row, "orow") or behind and _at(f))  # other axes, e.g. a broadcast over tangent
                # directions; per output: a loop's counter stays uniform next to its rows)
        elif rule is None or not any(ps.add(eqn, *r) for r in [rule] + _rules(eqn, N)[1:]):
            return None
    have = lambda v: row if (v, row) in ps.have else next((f for x, f in ps.have if x is v and behind and _at(f)), "whole")
    forms = [have(v) if r is None else _form(r) for v, r in zip(j.outvars, outs)]
    if not all(ps.provide(v, f if f == row or behind and f != "whole" else "uni") for v, f in zip(j.outvars, forms)):
        return None
    avals = [jax.ShapeDtypeStruct(_sample(v.aval.shape, f), v.aval.dtype) for v, f in ps.inputs[:len(j.invars)]]
    return _dce(jax.make_jaxpr(lambda *a: ps(list(zip(j.outvars, forms)), *a, *closed.consts))(*avals)), forms


class _Region(_PerSample):
    """Row-wise equations of the program, run as a kernel or in XLA (orig: its own equations)."""
    gpu = False  # (planned for the GPU: _axis sets it; its inner loops from GPU_INNER_WORK on)

    def reads(self):  # the values it reads from outside: as a kernel (also those of values computed per sample) and
        inside = {v for e in self.orig for v in e.outvars}  # as its equations in XLA (if its kernel does not build)
        return list({v: None for v in [v for v, _ in self.inputs] + list(self.base.values()) +
                     [v for e in self.orig for v in e.invars] if _var(v) and v not in inside})

    def counted(self):  # no input with rows (two sample axes: all read from whole arrays): one more input, zeros
        return not any(f != "whole" for _, f in self.inputs)  # per sample, that gives the kernel its number of samples

    def batched(self):
        return tuple(f != "whole" for _, f in self.inputs) + (True,) * self.counted()

    def avals(self):  # (keys, complex numbers: as kernels pass them, _kdata)
        return tuple(jax.ShapeDtypeStruct(_sample(v.aval.shape, f) + _tail(v.aval), _kdata(v.aval).dtype) for v, f in self.inputs) + \
            (jax.ShapeDtypeStruct((), np.float32),) * self.counted()  # (4 bytes: never widens the kernel's elements)

    def keyed(self, *values):  # the per-sample function on the kernel's values (keys, complex numbers: _kdata, in and out)
        values = [_from_data(x, v.aval) for x, (v, _) in zip(values, self.inputs)]
        return tuple(_to_data(o, v.aval) for o, v in zip(self([(v, self.form[v]) for v in self.outputs], *values), self.outputs))

    @functools.cached_property
    def inner(self):  # (the per-sample function of the outputs, its work): sums over an inner axis as loops (_inner)
        if isinstance(self.N, tuple) or _work(self, False)[0] <= (GPU_INNER_WORK if self.gpu else MAX_WORK):
            return self.keyed, None
        return _inner(self.keyed, self.avals(), list(self.batched()))

    def spec(self, opts):  # sums over the samples: reduced outputs of the kernel
        return Spec(self.inner[0], self.avals(), self.batched(), (), opts, tuple(self.form[v] in ("sum", "osum", "rsum") for v in self.outputs))

    def bases(self):  # per sum output, the value its scatters add into (outside the region)
        return [self.base[v] for v in self.outputs if self.form[v] == "sum"]

    def call(self, spec, vals, bases, run=None):
        """The kernel on the values of self.inputs (the rows on the kernel's sample axis, and back) and the bases; or
        run(*inputs) in its place, on the kernel's inputs (sample axis first, as they are) and with its outputs (rows,
        sums over the samples; _Mapped)."""
        def to(x, f):  # the sample axes to the front, as one
            P, k = _at(f) or (0, 0)
            x = jnp.moveaxis(x, range(k, k + P), range(P)) if k else x
            return x.reshape((-1,) + _sample(x.shape, f.partition("@")[0])) if P == 2 or "*" in f else x

        def back(o, v, f):
            P, k = _at(f) or (0, 0)
            if not k:
                return o.reshape(v.aval.shape + _tail(v.aval))
            o = o.reshape(tuple(v.aval.shape[k:k + P]) + o.shape[1:])
            return jnp.moveaxis(o, range(P), range(k, k + P))
        vals = [_to_data(x, v.aval) for x, (v, _) in zip(vals, self.inputs)]
        count = [jnp.zeros(int(np.prod(self.N)), np.float32)] * self.counted()
        if run is None:
            outs = call_p.bind(*[to_kernel(to(x, f)) for x, (_, f) in zip(vals, self.inputs)], *count, spec=spec)
        else:
            outs = run(*[to(x, f) for x, (_, f) in zip(vals, self.inputs)], *count)
        bases, res = iter(bases), []
        for v, o in zip(self.outputs, outs):
            f = self.form[v]
            o = o[0] if f == "whole" else o if f in ("sum", "osum", "rsum") else back(o, v, f)
            o = _from_data(o if run else from_kernel(o, _kdata(v.aval).dtype, spec.E), v.aval)
            res.append(_weak((lambda b: o if b is None else b + o)(next(bases)) if f == "sum" else o, v.aval))
        return res


# Loops coupled over the samples. Under vmap, a while loop of equinox (diffrax, optimistix, lineax) runs while any
# sample is active (its unvmap_any: an any over the samples) with one step counter for all; a sample that is done is
# left as it is. That is one loop per sample (each with its own counter) where _decouple proves it on the jaxpr;
# lanes_p marks the values that are then the sample's own (_per_sample), as the batched program computes them over
# all samples.


def _partition(jaxpr, N, items=None, sums=False, inner=False, consts=False):
    """Steps of the program: ("xla", eqn) or ("region", _Region), in an order that respects dependencies:
    a row-wise equation joins the open region unless it reads a row of the region as a whole (uniform)
    operand; equations that read the region's values otherwise (and what reads theirs) follow it, and the region
    closes when a row-wise equation needs one of theirs (the jaxpr's order is topological: independent equations
    may move). `items`:
    the equations, or the steps of another sample axis (its regions kept as they are). consts: equations whose
    rows are a constant's join regions too (_Mapped: their other operands are a sample's values, not uniform)."""
    steps, pulled, equal, region, new = [], {}, set(), None, []
    later, blocked = [], set()  # equations outside the open region that read its values (or theirs): after it
    _note_platforms(jaxpr)
    for eqn in jaxpr.eqns if items is None else items:
        old = isinstance(eqn, _Region)
        ins = [v for v in (eqn.reads() if old else eqn.invars) if _var(v)]
        rule = None if old else _rule(eqn, N, sums, inner)
        if not old and eqn.effects and any(not e[0] == "region" and e[1].effects for e in later):  # (side effects in
            later.append(("xla", eqn))  # their order: after one moved behind the region)
            blocked.update(eqn.outvars)
            continue
        if any(v in blocked for v in ins):
            if rule is None or not any(r.startswith("row") for r in rule[1]):  # not for the region: after it too
                later.append(("region" if old else "xla", eqn))
                blocked.update(eqn.outputs if old else eqn.outvars)
                continue
            steps += [("region", region)] + later  # for the region: it closes, they follow
            region, later, blocked = None, [], set()
        reads = region is not None and any(v in region.defined for v in ins)
        if rule is not None:
            params, roles, forms = rule
            rows = [v for v, r in zip(eqn.invars, roles) if r.startswith("row") and (
                consts or _var(v) or eqn.primitive.name in ("scan", "while"))]  # (a loop from a constant start, e.g.
            # a deterministic reset folded: its body per sample)
            if not reads and (eqn.primitive.name == "transpose" or all(f == "orow" for f in forms) or
                              all(v in equal for v in rows) and all(f == _row(N) for f in forms)):
                pulled.update((v, (eqn, params, roles, forms)) for v in eqn.outvars)  # equal rows (from uniform values)
                equal.update(eqn.outvars if eqn.primitive.name != "transpose" and not _samples(params) else [])  # (a
                # split's rows per sample, fold_in(k, s): not equal) and transposes of outside
                steps.append(("xla", eqn))  # values (not what is computed from them): computed per sample where read
                continue
            if rows:
                if region is None:
                    region = _Region(pulled, N=N)
                    new.append(region)
                if any(region.add(eqn, *r) for r in [rule] + _rules(eqn, N, sums, inner)[1:]):  # (other placements of the rows)
                    continue
        if reads:  # (after the region; the region goes on with what does not need it)
            later.append(("region" if old else "xla", eqn))
            blocked.update(eqn.outputs if old else eqn.outvars)
            continue
        if not old and eqn.primitive.name == "random_split" and rule is None:  # keys of a key the same for all (a scan's
            pulled[eqn.outvars[0]] = (eqn, eqn.params, ["uni"], ["whole"])  # per step): split where a region reads
            # them, lazily at the step that reads one (read as an input, the GPU kernels copied them to arrays per thread)
        steps.append(("region" if old else "xla", eqn))
    if region is not None:
        steps.append(("region", region))
    steps += later
    cache = _INLINED.__dict__.setdefault("cache", {})  # var -> the equations that read it (None: an output of the
    if ("readers", id(jaxpr)) not in cache:  # jaxpr), per jaxpr: the same for each candidate axis
        readers = {}
        for e in jaxpr.eqns:
            for v in e.invars:
                if _var(v):
                    readers.setdefault(v, set()).add(id(e))
        for v in jaxpr.outvars:
            if _var(v):
                readers.setdefault(v, set()).add(None)
        cache[("readers", id(jaxpr))] = (jaxpr, readers)
    readers, more = cache[("readers", id(jaxpr))][1], {}
    for kind, r in steps:  # (regions read also what they recompute per sample from: rep)
        if kind == "region":
            for v in r.reads():
                more.setdefault(v, set()).add(id(r))
    for r in new:  # region outputs: the values read outside the region
        if r.orig:
            orig = set(map(id, r.orig))
            r.outputs = [v for e in r.orig for v in e.outvars if readers.get(v, set()) - orig or more.get(v, set()) - {id(r)}]
    return steps


def _reads_rows(region):
    """Whether a region reads anything per sample: inputs with rows, or (two sample axes) rows s // Q and s % Q of
    whole arrays (an element's data and a point's, read in the kernel), or the sample's index (keys split per sample
    from one the same for all: a rollout from a constant start, _split_row)."""
    return any(f != "whole" for _, f in region.inputs) or any(r in ("orow", "orow1", "inner", "inner1")
                                                              for _, _, roles, _ in region.steps for r in roles) or \
        any(_samples(params) for _, params, _, _ in region.steps)


def _samples(params):  # whether per-sample params read the sample's index (_split_row; in bodies: kernelax_sample)
    return params.get(_FN) is _split_row or any(e.primitive is sample_p or _samples(e.params)
                                               for j in jcore.jaxprs_in_params(params) for e in j.eqns)


_HINTS = threading.local()  # .notes: the planner's choices while a new program is planned (into the index with it);
# .given: those of an earlier process, replayed (only the chosen candidates evaluated: the plan without the search)


def _shape_key(jaxpr, *extra):
    """A key of a jaxpr's structure (its primitives and their results' shapes) and of extra, the same in every
    process: a choice of the planner is noted and replayed under it (_HINTS)."""
    items = [(e.primitive.name, tuple(tuple(getattr(v.aval, "shape", ())) for v in e.outvars)) for e in jaxpr.eqns]
    return hashlib.sha1(repr((items, extra)).encode()).hexdigest()[:20]


def _choices(site, key, candidates):
    """The candidates the planner evaluates at a choice: the one chosen before (none: nothing was), else all."""
    given = getattr(_HINTS, "given", None)
    if given is None or f"{site}:{key}" not in given:
        return candidates
    c = given[f"{site}:{key}"]
    return [] if c is None else [tuple(c) if isinstance(c, list) else c]


def _hinting():  # whether choices are noted or replayed (a key is worth computing)
    return getattr(_HINTS, "notes", None) is not None or getattr(_HINTS, "given", None) is not None


def _chosen(site, key, choice):
    notes = getattr(_HINTS, "notes", None)
    if notes is not None:
        notes[f"{site}:{key}"] = None if choice is None else list(map(int, choice)) if isinstance(choice, tuple) else int(choice)


def _axis(jaxpr, candidates=4, axes=3, sums=False, gpu=False):
    """(sample axes, steps, regions with rows): the sample axis whose kernels leave the least data to memory, among the leading
    lengths of the program's largest arrays (an element loop's axis, not that of a long index array), then on the
    rest the next one (elements x quadrature points next to elements x tangent directions), up to `axes` of them."""
    size = {}
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:  # candidates: leading lengths, and pairs of them (nested vmaps)
            shape = tuple(v.aval.shape)
            for N in shape[:1] + ((shape[:2],) if len(shape) > 1 else ()):
                if np.prod(N) > 1 and (not isinstance(N, tuple) or N[1] > 1):
                    size[N] = size.get(N, 0) + v.aval.size
    chosen, steps, regions = [], [("xla", e) for e in jaxpr.eqns], []
    key = _shape_key(jaxpr, candidates, axes, sums, gpu) if _hinting() else None
    for p in range(axes):
        items = [x for kind, x in steps for x in ([x] if kind == "xla" or x in regions else x.orig)]
        evals = []  # (N, steps, regions, (-stored, covered), total work in kernels, worth a kernel)
        bysize = [N for N in sorted(size, key=size.get, reverse=True) if N not in chosen]  # (the largest arrays' leading
        rows = [N for N in bysize if np.prod(N) >= MIN_ROWS]  # lengths, and those with enough rows: tangent axes of
        for N in _choices("axis", f"{key}:{p}", list(dict.fromkeys(bysize[:candidates] + rows[:candidates]))):  # derivatives
            # lead large arrays too)
            new = _partition(jaxpr, N, items, sums)
            for kind, x in new:
                if kind == "region":
                    x.gpu = gpu
            rs = [x for kind, x in new if kind == "region" and x not in regions and x.outputs and  # (data movement
                  _reads_rows(x) and any(e.primitive.name not in _LAYOUT for e in x.orig)]  # alone: XLA)
            rs = [r for r in rs if not chosen or _worth(r, N, gpu=gpu)[0]]  # (further axes: for kernels only)
            rs = [r for r in rs if _work(r, False)[0] <= MAX_WORK or _work(r)[0] <= MAX_WORK_PLAIN]  # (inner loops: only
            if rs:  # (the least data through memory: the outputs of the kernels and of the operations outside;  # if needed)
                kernels = set(rs + regions)  # layout: free either way)
                read = {v for kind, x in new if kind == "region" for e in ([] if x in kernels else x.orig) for v in e.invars
                        if _var(v)} | {v for kind, x in new if kind == "region" and x in kernels for v, _ in x.inputs} | \
                    {v for v in jaxpr.outvars if _var(v)}
                for kind, x in reversed(new):  # (what is read: backwards through XLA's steps; those that only regions
                    if kind == "xla" and any(o in read for o in x.outvars):  # read, recomputing them per sample, are
                        read.update(v for v in x.invars if _var(v))  # dropped and store nothing: a rollout's reset, pulled)
                stored = sum(v.aval.size for kind, x in new for v in (x.outputs if kind == "region" and x in kernels else [
                    v for e in ([x] if kind == "xla" else x.orig) if e.primitive.name not in _LAYOUT and (
                        kind != "xla" or any(o in read for o in e.outvars)) for v in e.outvars]))
                covered = sum(v.aval.size for r in rs for e in r.orig if e.primitive.name not in _LAYOUT for v in e.outvars)
                evals.append((N, new, rs, (-stored, covered), lambda N=N, rs=rs: int(np.prod(N)) * sum(_work(r)[0] for r in rs),
                              lambda N=N, rs=rs: any(_worth(r, N, gpu=gpu)[0] for r in rs)))
        few = any(np.prod(e[0]) < MIN_ROWS for e in evals)  # (the work of kernels and their worth: only to weigh an
        evals = [e[:4] + ((e[4](), e[5]()) if few else (0, False)) for e in evals]  # axis with fewer rows, _inner's)
        cost = lambda e: e[4] / max(e[3][1], 1)  # (work per value computed in kernels)
        least = min((cost(e) for e in evals if np.prod(e[0]) >= MIN_ROWS), default=np.inf)
        # axes with enough rows first; one with fewer only where its kernels are worth it and do at most half the work
        # per value they compute (an element axis with loops over points: bodies once, linear rows (_inner); next to
        # the points unrolled)
        best = max(evals, key=lambda e: (np.prod(e[0]) >= MIN_ROWS or e[5] and 2 * cost(e) <= least, e[3], -len(e[2]),
                                         isinstance(e[0], tuple)), default=(None, None, []))  # (ties: the most data
        best = (None, best[0], best[1], best[2])  # computed in kernels, fewer regions, two axes: more lanes)
        _chosen("axis", f"{key}:{p}", best[1])
        if best[1] is None:
            break
        chosen.append(best[1])
        steps, regions = best[2], regions + best[3]
    return chosen, steps, regions


def _program(closed, steps, kernels, need=None):
    """The program with the regions in `kernels` (region -> Spec) as kernel calls, the others as their equations
    (returns the values of `need` instead of the outputs, if given)."""
    zeros = {o for e in closed.eqns if e.primitive.name == "broadcast_in_dim" and not _var(e.invars[0])
             and not np.any(np.asarray(e.invars[0].val)) for o in e.outvars}  # (bases of sums that are zeros: none)

    def run(*leaves):
        env = dict(zip(closed.constvars, closed.consts)) | dict(zip(closed.invars, leaves))
        for kind, x in steps:
            if kind == "xla" or x not in kernels:
                _run([x] if kind == "xla" else x.orig, env)
            else:
                read = lambda v: v.val if not _var(v) else env[v]
                env.update(zip(x.outputs, x.call(kernels[x], [read(v) for v, _ in x.inputs],
                                                 [None if _var(b) and b in zeros else read(b) for b in x.bases()])))
        return [v.val if not _var(v) else env[v] for v in need or closed.outvars]
    return run


_HEAVY = frozenset("""while scan cond lu cholesky triangular_solve eigh qr householder_product geqrf gather scatter-add
dot_general exp exp2 expm1 log log1p sin cos tan tanh asin acos atan atan2 sinh cosh asinh acosh atanh erf erfc erf_inv
lgamma digamma igamma igammac logistic pow integer_pow sqrt rsqrt cbrt div rem sort cumsum cumprod fft eig schur""".split())
TRIALS = 2  # calls of each program (with kernels, XLA's) timed before the faster one is chosen (_Entry; each after an
# untimed first call), alternately, for TRIAL_TIME seconds at least (a GPU idle while the kernels were built clocks up
# over its first calls: lineax' 0.26 ms program measured at 0.33 in the first five)
TRIAL_TIME = 0.05
TRIAL_MARGIN = 0.05  # XLA's program is chosen only if this much faster than the one with kernels: the choice is kept (the
# index, per thread count), and a tie within the trials' noise should not undo kernels the cost model chose
TRIAL_MAX_CALL = 0.05  # a first call with kernels longer than this (seconds): no trials, the cost model's choice stays
# (a further XLA call would cost as much, e.g. a whole simulation in one call; the cost model's misses were programs
# of 0.1-0.2 ms, which the trials still check)
LONG_CHECK = 0.3  # a long first call with kernels that took more than this fraction of XLA's first call (its compile
# and run) is checked against one call of XLA's program (the cost model chose kernels slower than XLA: pgx' othello
# rollout, 2.8 s against 2.3 s); faster ones stay without (an RL rollout 27 ms against XLA's 2 s: one call of XLA's
# would cost 70 of them)
DISPATCH = 5e-6  # seconds a call through kernelax.jit's dispatch to its own XLA program takes beyond jax.jit's (_xla_settled;
# measured 4.9 us for two arrays, jax.jit's 4.0)
BUILD_FIRST = env("KERNELAX_BUILD_FIRST", True, bool)  # a new program's first call builds its kernels and runs with them
# (0: it runs in XLA, the kernels build in a thread for the further calls; programs called once or a few
# times won with it, e.g. lineax' solve 2.06 -> 1.10 s, optimistix 3.8 -> 1.4 s, diffrax 4.6 -> 2.9 s for ~3 s of XLA)
BUILD_FIRST_WAIT = env("KERNELAX_BUILD_FIRST_WAIT", 10.0, float, inf=True)  # seconds the first call waits for its
# kernels (planning, trace, gcc of K = 1); beyond, it runs in XLA and they follow as without build_first (numpyro's
# log density: 53 s of build for 3 s of XLA; evosax: a trace of 10 s to the node limit)
EXIT_WAIT = env("KERNELAX_EXIT_WAIT", 600.0, float, inf=True)  # seconds at exit to finish kernel builds still running, so
# that they reach the index (a program called once per process, e.g. a simulation, else never gets its kernels; 0: off)
_THREADS = []  # the build threads of this process (_Entry)
@atexit.register
def _finish_builds():
    alive = [t for t in _THREADS if t.is_alive()]
    if not alive or EXIT_WAIT <= 0:
        return
    print(f"kernelax: finishing {len(alive)} kernel build(s) for the index before exit (KERNELAX_EXIT_WAIT={EXIT_WAIT:g} s; 0 skips)",
          file=sys.stderr, flush=True)
    end = time.monotonic() + EXIT_WAIT
    for t in alive:
        t.join(None if end == math.inf else max(0.0, end - time.monotonic()))
MIN_WORK_LOOP = 64  # in a loop body (a kernel call per iteration): at least this work per sample, or a loop itself
MIN_ROWS, MIN_WORK = 256, 16  # kernels for regions of >= MIN_ROWS rows with heavy operations or >= MIN_WORK per row
MIN_TOTAL = 1 << 20  # (or fewer rows with this much work in all)
MAX_WORK = 200000  # per sample, beyond: no kernel (the scalar code grows too large for gcc; frontend.MAX_NODES), unless
# sums over an inner axis run as loops (_inner)
MAX_WORK_PLAIN = 400000  # regions up to this with those loops where they apply (a long scan unrolled for its outputs per
# step: Jaxley's gradient per neuron, 371 708, runs as kernelax.kernel's); raising MAX_WORK itself would drop _inner's
# loops below it (the small Newton mesh's element axis unrolled, its work per value worse than the pair's)
GPU_INNER_WORK = 20000  # per sample, beyond: sums over an inner axis as loops (_inner) in GPU kernels (the C kernels:
# from MAX_WORK on; unrolled, nvcc's time and the registers grow with the code: an element tangent with a local Newton
# solve per quadrature point, 90 000-180 000 operations, took nvcc minutes)


def _body_work(params, name=None):  # the elements of the results in the jaxprs of a loop's or conditional's params
    js = list(jcore.jaxprs_in_params(params))
    body = sum(sum(math.prod(o.aval.shape) for o in e.outvars) * (e.primitive.name not in _LAYOUT) +
               _body_work(e.params, e.primitive.name) for j in js for e in j.eqns)
    if name != "scan" or frontend._rolled(params):  # (as the kernel holds it: a body once, an unrolled scan's per step;
        return body  # one unrolled for its xs and ys rolled where that would exceed ROLL_XS, as frontend._scan does)
    n = params["length"]
    return body if frontend.ROLL_XS and frontend._rolled(params, xs_too=False) and n * body > frontend.ROLL_XS else n * body


def _work(region, inner=True):
    """(elements of the per-sample results of a region's operations, loop bodies included; its heavy operations):
    sums over the samples (scatters) count by their updates, which are results of other operations. inner: for a
    region with sums over an inner axis run as loops (_inner), the work of that. (Kept with the region: asked for
    again and again while the axes are chosen.)"""
    done = region.__dict__.setdefault("_works", {})
    if inner not in done:
        done[inner] = _work_now(region, inner)
    return done[inner]


def _work_now(region, inner):
    ops = [e for e in region.orig if e.primitive.name not in _LAYOUT]
    if inner and isinstance(region, _Region) and region.inner[1] is not None:
        return region.inner[1], sorted({e.primitive.name for e in ops} & _HEAVY)
    steps = {id(e): (p, r) for e, p, r, _ in region.steps}
    work = sum(math.prod(_sample(v.aval.shape, region.form[v])) * _length(e, steps) for e in ops for v in e.outvars
               if region.form[v] not in ("sum", "osum"))  # (a contraction over the rows, rsum: an outer product per sample)
    work += sum(_body_work(params, e.primitive.name) for e, params, _, _ in region.steps if e.primitive.name in _NESTED)
    return work, sorted({e.primitive.name for e in ops} & _HEAVY)


def _length(e, steps):  # a contraction's length per sample (a 48 x 48 product per sample: 110592 multiply-adds,
    if e.primitive.name != "dot_general" or id(e) not in steps:  # not 2304), else 1
        return 1
    params, roles = steps[id(e)]
    shape = e.invars[0].aval.shape[1:] if roles[0] == "uni0" else _sample(e.invars[0].aval.shape, roles[0])
    return math.prod(shape[d] for d in params["dimension_numbers"][0][0])


_LIN_EACH = frozenset("add sub add_any concatenate stack".split())  # linear if all operands are (or are zero)
_LIN_ONE = frozenset("mul dot_general".split())  # linear in one operand, the other fixed
_LIN_FIRST = frozenset("""neg reduce_sum reshape squeeze broadcast_in_dim transpose slice rev copy
convert_element_type reduce_precision cumsum split unstack dynamic_slice gather div""".split())  # in operand 0 (others fixed)


def _linear(jaxpr, lin):
    """The values of jaxpr that are linear in its inputs `lin` (a set of variables), if its outputs are (else None):
    its equations add, scale (by values that do not depend on them), move, select and contract them (as jvp and vmap
    leave tangents); zeros (constants) count as linear."""
    state = dict.fromkeys(lin, "lin")
    kind = lambda v: state.get(v) if _var(v) else "zero" if not np.any(v.val) else None
    for e in jaxpr.eqns:
        ks, name = [kind(v) for v in e.invars], e.primitive.name
        if "lin" not in ks:  # (zeros stay zeros through moves and products)
            if ks and (name in _LIN_FIRST | {"pad"} and ks[0] == "zero" or name in _LIN_EACH and all(k == "zero" for k in ks)
                       or name in _LIN_ONE and "zero" in ks or name == "select_n" and all(k == "zero" for k in ks[1:])):
                state.update(dict.fromkeys(e.outvars, "zero"))
            continue
        fixed = lambda ks: "lin" not in ks
        if not (name in _LIN_EACH and all(ks) or name in _LIN_ONE and fixed(ks[:1]) != fixed(ks[1:]) or
                name in _LIN_FIRST and ks[0] == "lin" and fixed(ks[1:]) or name == "select_n" and fixed(ks[:1]) and all(ks[1:])
                or name == "pad" and ks == ["lin", "zero"] or name == "dynamic_update_slice" and all(ks[:2]) and fixed(ks[2:])
                or name == "scatter-add" and ks[0] and ks[2] and fixed(ks[1:2])) or \
                not all(jax.dtypes.issubdtype(o.aval.dtype, np.floating) for o in e.outvars):
            return None
        state.update(dict.fromkeys(e.outvars, "lin"))
    return {v for v, k in state.items() if k == "lin"} if all(kind(v) for v in jaxpr.outvars) else None


def _narrowest(jaxpr, lin):
    """The fewest entries of linear values (`lin`) that every path from the inputs to the outputs passes (minimal
    vertex cuts, the capacities sizes: max flow, Edmonds-Karp): (the cut nearest the inputs, the one nearest the outputs)."""
    cap, INF = collections.defaultdict(dict), float("inf")

    def edge(a, b, c):
        cap[a][b] = cap[a].get(b, 0) + c
        cap[b].setdefault(a, 0)
    for v in lin:
        edge(("i", v), ("o", v), int(np.prod(v.aval.shape)))
    for v in jaxpr.invars:
        if v in lin:
            edge("S", ("i", v), INF)
    for e in jaxpr.eqns:
        for u in (u for u in e.invars if _var(u) and u in lin):
            for w in e.outvars:
                edge(("o", u), ("i", w), INF)
    for v in jaxpr.outvars:
        if _var(v) and v in lin:
            edge(("o", v), "T", INF)

    def reach(start, fwd=True):  # BFS in the residual graph (backwards: the nodes that reach `start`): parents
        parent, todo = {start: None}, collections.deque([start])
        while todo:
            a = todo.popleft()
            for b in cap[a]:
                if (cap[a][b] if fwd else cap[b][a]) > 0 and b not in parent:
                    parent[b] = a
                    todo.append(b)
        return parent
    while "T" in (parent := reach("S")):
        path, b = [], "T"
        while parent[b] is not None:
            path.append((parent[b], b))
            b = parent[b]
        flow = min(cap[a][b] for a, b in path)
        for a, b in path:
            cap[a][b] -= flow
            cap[b][a] += flow
    to = reach("T", False)
    order = list(dict.fromkeys(list(jaxpr.invars) + [v for e in jaxpr.eqns for v in e.outvars]))
    return ([v for v in order if v in lin and ("i", v) in parent and ("o", v) not in parent],
            [v for v in order if v in lin and ("o", v) in to and ("i", v) not in to])


def _linear_rows(f, avals):
    """(f, the factor of its work): f with its regions that are row-wise along an axis of B rows and linear in their rows
    (tangent directions of jax.jacfwd: an element's B degrees of freedom) evaluated in three parts: up to their
    narrowest linear values A of d entries per row (a deformation gradient's 9) per row; from A on to the next narrow
    values Z (a stress's 9, or the outputs) on the d unit vectors only, giving the matrix M, Z[j] = A[j] @ M; the rest
    per row. Regions whose work does not shrink stay as they are."""
    closed = _fold(_inline(jax.make_jaxpr(f)(*avals)))
    lengths = {}
    for e in closed.eqns:
        for v in e.outvars:
            for d in set(v.aval.shape) - {0, 1}:
                lengths[d] = lengths.get(d, 0) + v.aval.size
    size = lambda v: int(np.prod(v.aval.shape))
    best = (0, None, {}, None)
    key = _shape_key(closed, [tuple(a.shape) for a in avals]) if _hinting() else None
    for B in _choices("linear", key, sorted(lengths, key=lengths.get, reverse=True)[:4]):
        steps, found, gain = _partition(closed, B), {}, 0
        for kind, x in steps:
            rows = [(v, f) for v, f in x.inputs if f != "whole"] if kind == "region" else []
            if not rows or not all(_at(f) and jax.dtypes.issubdtype(v.aval.dtype, np.floating) for v, f in rows) or \
                    not all(_at(x.form[v]) for v in x.outputs) or any(_is_key(v.aval) for v, _ in x.inputs):
                continue
            pj = _fold(_inline(jax.make_jaxpr(lambda *a, x=x: x([(v, x.form[v]) for v in x.outputs], *a))(  # (per row;
                *[jax.ShapeDtypeStruct(v.aval.shape if f == "whole" else _sample(v.aval.shape, f), v.aval.dtype)  # zero
                  for v, f in x.inputs])))  # constants: linear)
            lin = _linear(pj, {u for u, (_, f) in zip(pj.invars, x.inputs) if f != "whole"})
            outs = list(dict.fromkeys(v for v in pj.outvars if _var(v) and v in (lin or ())))
            if not outs:
                continue
            A, Z = _narrowest(pj, lin)
            d = sum(map(size, A))
            for Z in (Z, outs):  # (the matrix between the narrow values, or onto the outputs)
                mid = _needs(pj.eqns, Z, A)
                w = sum(size(o) for e in mid if e.primitive.name not in _LAYOUT for o in e.outvars if o in lin)
                saved = (B - d) * w - B * d * sum(map(size, Z))  # (the work between: on d unit vectors instead of B
                if saved > found.get(x, (0,))[0]:  # rows; Z from A and M)
                    found[x] = (saved, pj, A, Z, _needs(pj.eqns, A), mid, _needs(pj.eqns, pj.outvars, Z))
            gain += found.get(x, (0,))[0]
        best = max(best, (gain, steps, found, B), key=lambda b: b[0])
    gain, steps, found, B = best
    _chosen("linear", key, B if found else None)
    if not found:
        return f, 1.0

    def combined(x, vals):  # the region's outputs, from its three parts
        _, pj, A, Z, up, mid, rest = found[x]
        consts = dict(zip(pj.constvars, pj.consts)) | {u: a for u, a, (_, f) in zip(pj.invars, vals, x.inputs) if f == "whole"}
        rows = [(u, jnp.moveaxis(jnp.asarray(a), _at(f)[1], 0)) for u, a, (_, f) in zip(pj.invars, vals, x.inputs) if f != "whole"]

        def part(eqns, env, outs):
            _run(eqns, env)
            return [jnp.broadcast_to(v.val if not _var(v) else env[v], v.aval.shape) for v in outs]
        X = jax.vmap(lambda *r: part(up, consts | dict(zip([u for u, _ in rows], r)), A))(*[a for _, a in rows])
        X = jnp.concatenate([c.reshape(x.N, -1) for c in X], 1)
        at = np.cumsum([0] + [size(v) for v in A])
        M = jax.vmap(lambda e: part(mid, consts | {v: e[a:b].reshape(v.aval.shape).astype(v.aval.dtype)
                                                   for v, a, b in zip(A, at[:-1], at[1:])}, Z))(jnp.eye(at[-1], dtype=X.dtype))
        Y = [jnp.tensordot(X.astype(m.dtype), m, 1) for m in M]
        out = jax.vmap(lambda *z: part(rest, consts | dict(zip(Z, z)), pj.outvars))(*Y)
        return [jnp.moveaxis(o, 0, _at(x.form[v])[1]) for o, v in zip(out, x.outputs)]

    def g(*leaves):
        env = dict(zip(closed.constvars, closed.consts)) | dict(zip(closed.invars, leaves))
        read = lambda v: v.val if not _var(v) else env[v]
        for kind, x in steps:
            if kind == "region" and x in found:
                env.update(zip(x.outputs, combined(x, [read(v) for v, _ in x.inputs])))
            else:
                _run([x] if kind == "xla" else x.orig, env)
        return tuple(read(v) for v in closed.outvars)
    return g, min(_ops(jax.make_jaxpr(g)(*avals)) / max(_ops(closed), 1), 1.0)


def _ops(jaxpr):  # the work of a jaxpr (as _work: elements of the results; contractions times their length)
    return sum(int(np.prod(o.aval.shape)) * (int(np.prod([e.invars[0].aval.shape[k] for k in e.params["dimension_numbers"][0][0]]))
                                             if e.primitive.name == "dot_general" else 1)
               for e in jaxpr.eqns if e.primitive.name not in _LAYOUT for o in e.outvars) + \
        sum(_body_work(e.params, e.primitive.name) for e in jaxpr.eqns if e.primitive.name in _NESTED)


def _inner(f, avals, sample=None):
    """(f, its work) with the sums over an inner axis of a sample's computation (quadrature points of an element,
    neighbours of a particle: an axis the computation is row-wise in, then summed over) as loops over that axis
    (lax.fori_loop: the kernel holds the body once, the sums in accumulators), instead of unrolled; work: of the
    rest plus the loop bodies once. (f, None) if there are none."""
    closed = _fold(_inline(jax.make_jaxpr(f)(*avals)))  # (constants the same in all rows: not inputs of the bodies)
    lengths = {}
    for e in closed.eqns:
        for v in e.outvars:
            for d in set(v.aval.shape) - {0, 1}:
                lengths[d] = lengths.get(d, 0) + v.aval.size
    best = (0, None, [], None)
    key = _shape_key(closed, [tuple(a.shape) for a in avals], sample) if _hinting() else None
    for Q in _choices("inner", key, sorted(lengths, key=lengths.get, reverse=True)[:4]):  # the axis whose loops hold the
        steps = _partition(closed, Q, sums=True, inner=True)  # most work
        loops = [r for kind, r in steps if kind == "region" and r.outputs and all(r.form[v] == "rsum" for v in r.outputs)
                 and all(f == "whole" or _at(f) for _, f in r.inputs)]
        best = max(best, (sum(_work(r, False)[0] for r in loops), steps, loops, Q), key=lambda b: b[0])
    _, steps, loops, Q = best
    _chosen("inner", key, Q if loops else None)
    if not loops:
        return f, None

    plain = lambda r: lambda *a: r([(v, "rsum") for v in r.outputs], *a)  # a step's sums (bodies: with their linear
    bodies = {r: (plain(r), 1.0) if any(_is_key(v.aval) or _cplx(v.aval) for v, _ in r.inputs) else _linear_rows(plain(r), [  # regions
        jax.ShapeDtypeStruct(v.aval.shape if f == "whole" else _sample(v.aval.shape, f), v.aval.dtype) for v, f in r.inputs])
        for r in loops}  # on unit vectors, _linear_rows)

    def loop(r, vals):
        def body(q, accs):
            ins = [x if f == "whole" else lax.dynamic_index_in_dim(x, q, _at(f)[1], keepdims=False) for x, (_, f) in zip(vals, r.inputs)]
            return tuple(a + o for a, o in zip(accs, bodies[r][0](*ins)))
        return lax.fori_loop(0, r.N, body, tuple(jnp.zeros(v.aval.shape, v.aval.dtype) for v in r.outputs))

    def g(*leaves):
        env = dict(zip(closed.constvars, closed.consts)) | dict(zip(closed.invars, leaves))
        read = lambda v: v.val if not _var(v) else env[v]
        for kind, x in steps:
            if kind == "region" and x in loops:
                env.update(zip(x.outputs, loop(x, [read(v) for v, _ in x.inputs])))
            else:
                _run([x] if kind == "xla" else x.orig, env)
        return tuple(read(v) for v in closed.outvars)
    dep = {v for v, a in zip(closed.invars, sample) if a}  # the rest: what depends on the sample's own data (uniform
    rest = []  # values fold or stay in memory)
    for kind, x in steps:
        for e in [] if kind == "region" and x in loops else [x] if kind == "xla" else x.orig:
            if any(_var(v) and v in dep for v in e.invars):
                dep.update(e.outvars)
                rest.append(e)
    size = lambda e: e.invars[2].aval.size if e.primitive.name in ("scatter", "scatter-add") else \
        e.invars[1].aval.size if e.primitive.name == "dynamic_update_slice" else sum(v.aval.size for v in e.outvars)
    work = sum(size(e) for e in rest if e.primitive.name not in _LAYOUT)  # (scatters: by their updates)
    g.carried = sum(v.aval.size for r in loops for v in r.outputs)  # (the loops' accumulators)
    return g, work + sum(_body_work(e.params, e.primitive.name) for e in rest) + sum(int(_work(r)[0] * bodies[r][1]) for r in loops)


MAP_CARRIED = 1024  # a GPU kernel whose inner loops carry more accumulators than this per sample (an element tangent
# of 81 x 81 summed over the quadrature points: far beyond the registers, spilled), or doing more work per sample than
# MAP_WORK (its inner loops counted once): its samples' map axes looked for (_mapped; below, the kernel per sample
# builds faster and runs about as fast: a plasticity element with 171 accumulators)
MAP_WORK = 100000
MAP_MIN = 8  # a map axis: at least this long
MAP_SUMS = 64  # sums over a map axis: at most this many entries per sample (each entry's part goes through memory)
MAP_DEPTH = 2  # map axes within map axes: at most this deep
MAP_INNER_WORK = 1000  # a map kernel doing more work per sample than this: its sums over an inner axis as loops


def _unfold(closed, M):
    """closed with the triangular solves whose right-hand sides batching folded an axis of length M into (vmap of a
    solve with an unbatched matrix: reshape (..., c, M) to (..., c * M), solve, reshape back) as solves batched over
    M (the matrix broadcast): rows along M for _partition, one solve per row."""
    j, uses, made = closed, {}, {}
    for e in j.eqns:
        for v in e.invars:
            if _var(v):
                uses[v] = uses.get(v, 0) + 1
        made.update((v, e) for v in e.outvars)
    for v in j.outvars:
        if _var(v):
            uses[v] = uses.get(v, 0) + 1
    readers = {v: e for e in j.eqns for v in e.invars if _var(v)}
    plan = {}  # the reshape back -> (the solve, the reshape before it, the axis of M in its operand)
    for e in j.eqns:
        if e.primitive.name != "triangular_solve" or not _var(e.invars[1]) or uses.get(e.invars[1]) != 1 or uses.get(e.outvars[0]) != 1:
            continue
        r1, r2, b = made.get(e.invars[1]), readers.get(e.outvars[0]), e.invars[1].aval
        if r1 is None or r2 is None or r1.primitive.name != "reshape" or r2.primitive.name != "reshape" or \
                r1.params["dimensions"] is not None or r2.params["dimensions"] is not None:
            continue
        x, m = r1.invars[0].aval, b.ndim - 1 if e.params["left_side"] else b.ndim - 2  # (the axis of the columns / rows)
        if tuple(r2.outvars[0].aval.shape) == tuple(x.shape) and x.ndim == b.ndim + 1 and x.shape[m + 1] == M and \
                tuple(x.shape[:m]) == tuple(b.shape[:m]) and tuple(x.shape[m + 2:]) == tuple(b.shape[m + 1:]) and \
                M not in e.invars[0].aval.shape:
            plan[id(r2)] = (e, r1, m + 1)
    if not plan:
        return closed
    skip = {id(x) for e, r1, _ in plan.values() for x in (e, r1)}

    def run(*a):
        env = dict(zip(j.constvars, closed.consts)) | dict(zip(j.invars, a))
        read = lambda v: v.val if not _var(v) else env[v]
        for e in j.eqns:
            if id(e) in skip:
                continue
            if id(e) in plan:
                t, r1, k = plan[id(e)]
                mat = read(t.invars[0])
                y = t.primitive.bind(jnp.broadcast_to(mat, (M,) + jnp.shape(mat)), jnp.moveaxis(read(r1.invars[0]), k, 0), **t.params)
                env[e.outvars[0]] = jnp.moveaxis(y, 0, k)
                continue
            env.update(zip(e.outvars, _bind(e, e.params, [read(v) for v in e.invars])))
        return [read(v) for v in j.outvars]
    return _inline(jax.make_jaxpr(run)(*closed.in_avals))


def _mapped(region, opts, gpu, depth=0):
    """A _Mapped for a region whose samples each compute along a map axis (the tangent directions of an element's
    derivative, the outputs of a layer, ...), or None: the axis among the per-sample function's lengths whose
    regions (_partition along it, rows at any axis; constant rows too: the AD seeds) cover the most work, at least a
    quarter of it (much of the rest: values equal along the axis, which the map kernels compute per row where read)."""
    closed = _fold(_inline(jax.make_jaxpr(region.keyed)(*region.avals())))
    total, lengths = _work(region, False)[0], {}
    for e in closed.eqns:
        for v in e.outvars:
            for d in set(v.aval.shape) - {0, 1}:
                lengths[d] = lengths.get(d, 0) + v.aval.size
    best = None
    key = _shape_key(closed, repr(region.N), gpu, depth) if _hinting() else None
    for M in _choices("map", key, [M for M in sorted(lengths, key=lengths.get, reverse=True)[:4] if M >= MAP_MIN]):
        unfolded = _unfold(closed, M)
        steps = _partition(unfolded, M, sums=True, inner=True, consts=True)
        maps = [r for kind, r in steps if kind == "region" and r.outputs and _reads_rows(r) and
                any(e.primitive.name not in _LAYOUT for e in r.orig) and
                sum(int(np.prod(_sample(v.aval.shape, r.form[v]))) for v in r.outputs if r.form[v] in ("sum", "rsum")) <= MAP_SUMS]
        for r in maps:
            r.gpu = gpu
        maps = [r for r in maps if _worth(r, int(np.prod(region.N)) * M, gpu=gpu)[0]]
        covered = M * sum(_work(r, False)[0] for r in maps)
        if 4 * covered >= total and (best is None or covered > best[0]):
            best = (covered, unfolded, M, steps, maps)
    _chosen("map", key, None if best is None else best[2])
    return None if best is None else _Mapped(region, *best[1:], opts, gpu, depth)


class _Mapped:
    """A region whose samples each compute along a map axis M: its per-sample function (closed) partitioned along
    M (steps), the regions along M (maps) as kernels over N x M samples (sample s: row s // M's sample, entry s % M;
    the rows at whatever axis they are), the rest batched over the N samples as a program (sub: planned on its own,
    its kernels too), instead of one kernel per sample that unrolls M (nvcc's time and the registers grow with it;
    one thread per sample: few threads)."""

    def __init__(self, region, closed, M, steps, maps, opts, gpu, depth):
        self.region, self.closed, self.M, self.steps, self.N = region, closed, M, steps, int(np.prod(region.N))
        self.inputs, self.outputs, self.orig, self.form = region.inputs, region.outputs, region.orig, region.form
        j = closed
        made, uses = {v: e for e in j.eqns for v in e.outvars}, collections.Counter(v for e in j.eqns for v in e.invars if _var(v))
        self.sums = set()  # the scatter-adds of the sums over the samples into a base (an assembly: one scatter for all
        for v, w in zip(region.outputs, j.outvars):  # samples, not a dense array per sample)
            e = made.get(w) if region.form[v] == "sum" and _var(w) else None
            while e is not None and e.primitive.name == "scatter-add" and uses[e.outvars[0]] <= 1:
                self.sums.add(id(e))
                e = made.get(e.invars[0]) if _var(e.invars[0]) else None
        self.batched = {v for v, (_, f) in zip(j.invars, region.inputs) if f != "whole"} | set(j.invars[len(region.inputs):])
        for kind, x in steps:  # (values per sample of the region: batched over N; a kernel's outputs all)
            for e in [x] if kind == "xla" else x.orig:
                if any(_var(v) and v in self.batched for v in e.invars[id(e) in self.sums:]):
                    self.batched.update(e.outvars)
                elif id(e) in self.sums:
                    self.batched.difference_update(e.outvars)
            if kind == "region" and x in maps:
                self.batched.update(x.outputs)
        self.maps = {r: self._spec(r, opts) for r in maps}
        avals = [v.aval for v, _ in region.inputs] + [b.aval for b in region.bases()]
        self.sub = _Plan(_dce(jax.make_jaxpr(self._lifted(set(self.maps.values())))(*avals)), opts, gpu=gpu, depth=depth + 1)
        # (without the steps of the partition whose values only the map kernels read: they compute them per sample)
        self.report = dict(M=M, kernels=len(maps), work=[int(_work(r)[0]) for r in maps], sub=self.sub.report)

    def bases(self):
        return self.region.bases()

    def specs(self):
        return list(self.maps.values()) + self.sub.specs()

    def _counted(self, r):  # no input with N x M rows: one more, zeros per sample (their number)
        return "rows" not in self._modes(r)

    def _modes(self, r):  # per input of a map region: how its kernel reads it
        return ["perN" if f == "whole" and v in self.batched else "whole" if f == "whole" else
                "rows" if v in self.batched else "rowsM" for v, f in r.inputs]

    def _spec(self, r, opts):
        """The kernel of map region r over N x M samples: rows ("rows": an input (N x M, ...)), a sample's own value
        ("perN": row s // M of an input (N, ...)), rows the same for all N ("rowsM": row s % M of (M, ...))."""
        M, modes = self.M, self._modes(r)
        aval = lambda v, f: jax.ShapeDtypeStruct(_sample(v.aval.shape, f) + _tail(v.aval), _kdata(v.aval).dtype)
        avals = [jax.ShapeDtypeStruct((self.N,) + a.shape, a.dtype) if m == "perN" else
                 jax.ShapeDtypeStruct((M,) + a.shape, a.dtype) if m == "rowsM" else a
                 for (v, f), m in zip(r.inputs, modes) for a in [aval(v, f)]] + [jax.ShapeDtypeStruct((), np.float32)] * self._counted(r)
        batched = [m == "rows" for m in modes] + [True] * self._counted(r)

        def f(*vals):
            s = sample_p.bind()
            n, m = lax.div(s, np.asarray(M, s.dtype)), lax.rem(s, np.asarray(M, s.dtype))
            return r.keyed(*[jnp.take(x, n, axis=0, mode="clip") if md == "perN" else jnp.take(x, m, axis=0, mode="clip")
                             if md == "rowsM" else x for x, md in zip(vals, modes)])
        if _work(r, False)[0] > MAP_INNER_WORK:  # (sums over an inner axis as loops)
            f = _inner(f, avals, batched)[0]
        return Spec(f, tuple(avals), tuple(batched), (), opts)

    def _call(self, r, spec, args):
        """Map region r as its kernel over N x M samples: args, outputs as values batched over N (their axis first)."""
        N, M, ins = self.N, self.M, []
        for (v, f), x, md in zip(r.inputs, args, self._modes(r)):
            x = _to_data(x, v.aval)
            if md in ("rows", "rowsM"):
                k = _at(f)[1] + (md == "rows")  # (the axis of the rows in x)
                x = jnp.moveaxis(x, k, int(md == "rows"))
                x = x.reshape((N * M,) + x.shape[2:]) if md == "rows" else x
            ins.append(x)
        outs = call_p.bind(*[to_kernel(x) for x in ins], *[jnp.zeros(N * M, np.float32)] * self._counted(r), spec=spec)
        res = []
        for v, o in zip(r.outputs, outs):
            f = r.form[v]
            o = from_kernel(o, _kdata(v.aval).dtype, spec.E).reshape((N, M) + o.shape[1:])
            o = o[:, 0] if f == "whole" else o.sum(1) if f in ("sum", "rsum") else jnp.moveaxis(o, 1, _at(f)[1] + 1)
            res.append(_from_data(o, v.aval))
        return res

    def _batched(self, ok, ins):
        """The per-sample function on values batched over the N samples (the kernel inputs of the region): the map
        regions whose specs are in `ok` as kernels, the rest vmapped over N."""
        j, N = self.closed, self.N
        env = dict(zip(j.constvars, self.closed.consts)) | dict(zip(j.invars, ins))
        once = {v for e in j.eqns if id(e) in self.sums for v in e.outvars}  # (sums over the samples, by one scatter)
        read = lambda v: v.val if not _var(v) else env[v]
        lift = lambda v, x: jnp.broadcast_to(x, (N,) + jnp.shape(x)) if v in self.batched and jnp.ndim(x) == v.aval.ndim else x

        def bind(e):
            b = [_var(v) and v in self.batched for v in e.invars]
            if id(e) in self.sums and any(b[1:]):  # (the samples a further leading axis of indices and updates)
                op, idx, upd = map(read, e.invars)
                idx, upd = [x if bx else jnp.broadcast_to(x, (N,) + jnp.shape(x)) for x, bx in zip((idx, upd), b[1:])]
                dn = e.params["dimension_numbers"]
                dn = dn._replace(update_window_dims=tuple(d + 1 for d in dn.update_window_dims),
                                 scatter_indices_batching_dims=tuple(d + 1 for d in dn.scatter_indices_batching_dims))
                env[e.outvars[0]] = e.primitive.bind(op, idx, upd, **dict(e.params, dimension_numbers=dn))
                return
            run([e])

        def run(chunk):  # equations vmapped over N at once (one trace for all of them; unbatched values stay so)
            if not chunk:
                return
            made = {v for e in chunk for v in e.outvars}
            ins = list(dict.fromkeys(v for e in chunk for v in e.invars if _var(v) and v not in made))
            outs = [v for e in chunk for v in e.outvars]

            def f(*a):
                local = dict(zip(ins, a))
                for e in chunk:
                    local.update(zip(e.outvars, _bind(e, e.params, [v.val if not _var(v) else local[v] for v in e.invars])))
                return [local[v] for v in outs]
            if any(v in self.batched for v in ins):
                res = jax.vmap(f, in_axes=[0 if v in self.batched else None for v in ins],
                               out_axes=[0 if v in self.batched else None for v in outs])(*map(read, ins))
            else:
                res = f(*map(read, ins))
            env.update((v, lift(v, o)) for v, o in zip(outs, res))
        chunk = []
        for kind, x in self.steps:
            if kind == "region" and x in self.maps and self.maps[x] in ok:
                run(chunk)
                chunk = []
                outs = self._call(x, self.maps[x], [read(v) for v, _ in x.inputs])
                for v, o in zip(x.outputs, outs):  # (sums into a base outside the region: added)
                    env[v] = o + read(x.base[v]) if x.form[v] == "sum" else o
            else:
                for e in [x] if kind == "xla" else x.orig:
                    if id(e) in self.sums:
                        run(chunk)
                        chunk = []
                        bind(e)
                    else:
                        chunk.append(e)
        run(chunk)
        summed = [self.form[v] in ("sum", "osum", "rsum") for v in self.outputs]  # (summed over the samples: unbatched,
        outs = [o if _var(w) and (w in self.batched or s and w in once) else  # from the scatters of self.sums)
                jnp.broadcast_to(o, (N,) + jnp.shape(o)) for w, o, s in zip(j.outvars, map(read, j.outvars), summed)]
        return [o.sum(0) if s and not (_var(w) and w in once) else o for w, o, s in zip(j.outvars, outs, summed)]

    def _lifted(self, ok):  # the region's outputs from its inputs and bases, computed as _batched
        n = len(self.inputs)
        return lambda *a: self.region.call(None, list(a[:n]), list(a[n:]), run=lambda *ins: self._batched(ok, ins))

    def call(self, ok, vals, bases):
        bases = [jnp.zeros(b.aval.shape, b.aval.dtype) if x is None else x for b, x in zip(self.bases(), bases)]
        if all(s in ok for s in self.maps.values()):
            return self.sub.program(ok)(*vals, *bases)
        return self._lifted(ok)(*vals, *bases)


def _prims(jaxpr):  # the primitives of a jaxpr, those of its inner jaxprs too
    return {e.primitive.name for e in jaxpr.eqns}.union(*(_prims(getattr(j, "jaxpr", j)) for e in jaxpr.eqns
                                                          for j in jcore.jaxprs_in_params(e.params)))


def _worth(region, N, loop=False, gpu=False):
    """Whether a region runs as a kernel (a cost model instead of timing, so that the first call stays short): enough
    rows, per row operations XLA does not fuse cheaply (transcendental functions, divisions, loops, linear algebra,
    gathers and scatters, contractions) or enough of the cheap ones (the elements of their results), and not more
    than MAX_WORK_PLAIN of them (a kernel that large would take gcc too long). Heavy operations of XLA's own program that the
    program for the kernels lacks count for every region (_Entry._xla_heavy)."""
    work, own = _work(region)  # (the kernel's: sums over inner axes as loops, _inner)
    heavy, info = sorted(set(own) | set(getattr(_HINTS, "xla_heavy", None) or ())), {"work": work, "heavy": own}
    xla = _work(region, False)[0]  # (XLA's: the work a kernel saves is measured by it)
    out = sum(int(np.prod(_sample(v.aval.shape, region.form[v]))) for v in region.outputs if region.form[v] not in ("sum", "osum", "rsum"))
    if not (set(heavy) & GPU_HEAVY if gpu else heavy) and xla < 2 * out:  # data expansion (about an operation per
        return False, info  # output entry): XLA fuses it into its readers
    if gpu:
        return _worth_gpu(region, N, loop, work, heavy, xla), info
    if loop and xla < MIN_WORK_LOOP and not set(heavy) & set(_NESTED):  # (a kernel call per iteration of the loop:
        return False, info  # its cost per call must be paid by enough work)
    if _dots(region) > DOT_MAX or 2 * _dense_work(region, CPU_DENSE_MIN) >= xla and not set(heavy) & set(_NESTED):
        return False, info  # (XLA's matrix products)
    rows = int(np.prod(N))  # (enough rows, or few with much work each: e.g. a loop per run of an optimizer)
    return (rows >= MIN_ROWS or rows * xla >= MIN_TOTAL) and bool(heavy or xla >= MIN_WORK) and work <= MAX_WORK_PLAIN, info


CPU_DENSE_MIN = 4096  # regions whose work is half or more in contractions with a uniform operand of this size (a weight
# matrix: dense layers) stay in XLA (its GEMM), unless they have loops; measured on 8 threads, x @ W of d x d, tanh, @ W: kernels 1.6x faster at d = 48, 0.83x at 64,
# 0.62x at 128 (flax and jraph programs were slower with kernels)
DOT_MAX = 1 << 15  # multiply-adds per sample of a region's largest contraction (_dots), beyond: XLA (gcc's time grows with them,
# XLA's batched products catch up: a 48 x 48 product per sample took 33 s of gcc for a tie, X^T X of 150 columns 55 s
# for 1.5x, of 200 87 s for a tie; 32 x 32 products: 1.4x, 9 s)


GPU_HEAVY = frozenset("while scan cond lu cholesky triangular_solve eigh qr householder_product geqrf sort cumsum cumprod fft eig "
                      "schur".split())
GPU_FFT_MAX = {False: 128, True: 32}  # (float32, float64) longer ffts: XLA's (cuFFT) on the GPU; a thread per sample
# is faster up to these lengths (measured on an L40S: f32 rfft 2.1x at 32, 1.7x at 128, 0.5x at 256; f64 1.1x at 32,
# 0.7x at 64)
GPU_MIN_WORK = 64  # without those: at least this work per row (XLA fuses elementwise work and gathers on the GPU well;
# measured: ties up to ~80, gains from ~90)
DENSE_MIN = 64  # contractions with a uniform operand of this size (a weight matrix): XLA's (cuBLAS) on the GPU, unless
# its kernels hold them as dense nodes (gpu.DENSE_MIN: loops where unrolled they would spill)
GPU_FILL_SM = 231  # rows per multiprocessor from which one thread per row fills the GPU (_gpu_fill; on an L40S, 142 SMs:
# 2^15 rows); fewer: only regions with GPU_HEAVY operations (XLA spreads the other work over all elements; measured:
# numpyro_logp, 4096 rows)


@functools.cache
def _gpu_fill():  # GPU_FILL_SM times the multiprocessors of the first GPU
    return GPU_FILL_SM * jax.devices("gpu")[0].core_count


def _dense(region, size=DENSE_MIN):  # whether a region contracts per-sample values with a uniform matrix (a dense layer)
    return any(e.primitive.name == "dot_general" and any(r is not None and r.startswith("uni") and np.prod(v.aval.shape) >= size
                                                         for v, r in zip(e.invars, roles)) for e, _, roles, _ in region.steps)


def _dense_work(region, size):  # multiply-adds per sample of a region's contractions with a uniform operand of `size`
    steps = {id(e): (p, r) for e, p, r, _ in region.steps}  # entries or more (dense layers)
    return sum(math.prod(_sample(v.aval.shape, region.form[v])) * _length(e, steps) for e, _, roles, _ in region.steps
               if e.primitive.name == "dot_general" and any(r is not None and r.startswith("uni") and np.prod(x.aval.shape) >= size
                                                            for x, r in zip(e.invars, roles)) for v in e.outvars)


def _dots(region):  # multiply-adds per sample of a region's largest contraction (over the rows, rsum: twice, their
    steps = {id(e): (p, r) for e, p, r, _ in region.steps}  # accumulators)
    return max((math.prod(_sample(v.aval.shape, region.form[v])) * _length(e, steps) * (1 + (region.form[v] == "rsum"))
                for e in region.orig if e.primitive.name == "dot_general" for v in e.outvars), default=0)


def _worth_gpu(region, N, loop, work, heavy, xla):
    """_worth on a GPU (one thread per row): loops, linear algebra and sorts (XLA runs them operation by operation, its
    loops with a host round trip per iteration), or GPU_MIN_WORK per row of floating-point work on _gpu_fill() rows or
    more (integer work: index arithmetic, which XLA fuses into its gathers and scatters; e.g. an assembly at a
    template's positions); dense layers without loops stay in XLA (its matrix products) if the kernels unroll them;
    the rows and loop bodies as on the CPU."""
    big = set(heavy) & GPU_HEAVY
    if loop and xla < MIN_WORK_LOOP and not big & set(_NESTED):
        return False
    if any(e.primitive.name == "fft" and math.prod(e.params["fft_lengths"]) >
           GPU_FFT_MAX[np.dtype(e.invars[0].aval.dtype) in (np.float64, np.complex128)] for e in region.orig):
        return False  # (XLA's cuFFT)
    if _dense(region) and not big & set(_NESTED) and not gpu_mod.DENSE_MIN:
        return False
    rows = int(np.prod(N))
    floats = big or sum(int(np.prod(_sample(v.aval.shape, region.form[v]))) for e in region.orig if e.primitive.name not in _LAYOUT
                        for v in e.outvars if region.form[v] not in ("sum", "osum", "rsum") and jnp.issubdtype(v.aval.dtype, jnp.inexact))
    return (rows >= MIN_ROWS or rows * xla >= MIN_TOTAL) and bool(big or floats >= GPU_MIN_WORK and rows >= _gpu_fill()) \
        and work <= MAX_WORK


SINGLE_MAX = 1 << 14  # a program without a sample axis whose inputs and outputs have at most this many entries, with a
# loop, runs as one kernel on one sample (_Single)
SINGLE_CARRY = 256  # ... if its loops carry at most this many values (one lane of the vectors does the work: XLA's loops
# over larger arrays are as fast, measured a tie from 128 values and 2.7x slower at 1000; gcc's time grows quadratically:
# 75 s for 2048 values)


class _Single:
    """The whole program as one kernel on one sample (as kernelax.kernel): for a program without a sample axis whose time
    goes into loops over small data (one local problem, time stepping); XLA runs such loops operation by operation.
    Its inputs are the same for all (the one) samples; a dummy input of one sample sets the batch."""

    def __init__(self, closed):
        self.closed, self.orig = closed, closed.eqns
        self.inputs = [(v, "whole") for v in closed.invars]
        self.outputs = list(dict.fromkeys(v for v in closed.outvars if _var(v)))

    @staticmethod
    def suits(closed):
        j = closed
        avals = [v.aval for v in j.invars] + [v.aval for v in j.outvars]
        loops = [e for e in j.eqns if e.primitive.name == "while" or e.primitive.name == "scan" and e.params["length"] > 1]
        carry = lambda e: sum(v.aval.size for v in (e.params["ft_out"].update(e.outvars).unpack()[0] if e.primitive.name == "scan" else e.outvars))
        return (bool(loops) and sum(a.size for a in avals) <= SINGLE_MAX and not any(_is_key(a) for a in avals) and not j.effects
                and all(carry(e) <= SINGLE_CARRY for e in loops) and _ops(j) <= MAX_WORK)

    def bases(self):
        return []

    def spec(self, opts):
        j = self.closed

        def f(_, *leaves):
            env = _run(j.eqns, dict(zip(j.constvars, self.closed.consts)) | dict(zip(j.invars, leaves)))
            return tuple(env[v] for v in self.outputs)
        return Spec(f, (jax.ShapeDtypeStruct((), np.float64 if jax.config.jax_enable_x64 else np.float32),) +
                    tuple(jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in self.closed.invars),
                    (True,) + (False,) * len(self.inputs), (), opts)

    def call(self, spec, vals, bases):
        outs = call_p.bind(jnp.zeros(1, spec.dtype), *[to_kernel(x) for x in vals], spec=spec)
        return [_weak(from_kernel(o[0], v.aval.dtype, spec.E), v.aval) for o, v in zip(outs, self.outputs)]


class _Plan:
    """Which regions of a jaxpr run as kernels (_worth), also in the bodies of loops and conditionals that stay in
    XLA (sub-plans per body). `program(ok)` assembles the program with the kernels whose specs are in `ok`."""

    def __init__(self, closed, opts, loop=False, gpu=False, depth=0):
        closed = _fold(_decouple(closed))
        self.closed, self.kernels, self.loops = closed, {}, {}
        N, steps, regions = _axis(closed, sums=True, gpu=gpu)  # (sums over the samples: reduced in the kernels)
        self.report, self.steps = {"N": [np.asarray(n).tolist() for n in N], "regions": [], "loops": []}, []
        for kind, r in steps:  # regions without a kernel: their equations (loops in them get sub-plans)
            if kind == "region" and r in regions:
                c, info = _worth(r, r.N, loop, gpu)
                self.report["regions"].append(dict(info, N=np.asarray(r.N).tolist(), eqns=len(r.orig), chosen="c" if c else "xla"))
                if c:  # (on a GPU, samples that each compute along a map axis: kernels over both, _Mapped)
                    m = _mapped(r, opts, gpu, depth) if gpu and depth < MAP_DEPTH and not isinstance(r.N, tuple) and \
                        (getattr(r.inner[0], "carried", 0) > MAP_CARRIED or _work(r)[0] > MAP_WORK) else None  # (two
                    # sample axes: rows read at the sample's index, not batched)
                    if m is not None:
                        self.report["regions"][-1]["mapped"] = m.report
                    self.kernels[m or r] = m or r.spec(opts)
                    self.steps.append((kind, m or r))
                    continue
            self.steps += [("xla", e) for e in r.orig] if kind == "region" else [(kind, r)]
        if not loop and not gpu and not self.kernels and _Single.suits(closed):  # (no sample axis: the whole program,
            # one sample; on a GPU one thread: not measured)
            single = _Single(closed)
            self.kernels, self.steps = {single: single.spec(opts)}, [("region", single)]
            self.report["regions"].append(dict(work=_ops(closed), N=1, eqns=len(closed.eqns), chosen="c", single=True))
        for i, (kind, eqn) in enumerate(self.steps):
            if kind == "xla" and eqn.primitive.name in _NESTED:
                p = eqn.params
                bodies = (list(enumerate(p["branches"])) if eqn.primitive.name == "cond" else
                          [("body_jaxpr", p["body_jaxpr"])] if eqn.primitive.name == "while" else
                          [("jaxpr", p["jaxpr"])] if p["length"] else [])
                subs = [(k, _Plan(_inline(b), opts, loop=True, gpu=gpu, depth=depth)) for k, b in bodies]
                subs = [(k, sub) for k, sub in subs if sub.specs()]
                if subs:
                    self.loops[i] = subs
                    self.report["loops"] += [dict(sub.report, loop=eqn.primitive.name, body=k) for k, sub in subs]

    def specs(self):
        return [s for r, k in self.kernels.items() for s in (r.specs() if isinstance(r, _Mapped) else [k])] + \
            [s for subs in self.loops.values() for _, sub in subs for s in sub.specs()]

    def program(self, ok, skip=()):
        """The program over the flat inputs, with the kernels of the specs in `ok` (the others' regions in XLA); without
        the outputs at the positions in `skip`."""
        steps = list(self.steps)
        for i, subs in self.loops.items():
            eqn = steps[i][1]
            params = dict(eqn.params)
            for k, sub in subs:
                if any(s in ok for s in sub.specs()):
                    new = jax.make_jaxpr(sub.program(ok))(*sub.closed.in_avals)
                    if eqn.primitive.name == "cond":
                        params["branches"] = tuple(new if j == k else b for j, b in enumerate(params["branches"]))
                    else:
                        params[k] = new
            steps[i] = ("xla", eqn.replace(params=params))
        need = [v for k, v in enumerate(self.closed.outvars) if k not in skip] if skip else None
        return _program(self.closed, steps, {r: ok if isinstance(r, _Mapped) else s for r, s in self.kernels.items()
                                             if isinstance(r, _Mapped) or s in ok}, need)  # (_Mapped: its kernels in ok)


def _program_key(closed, opts, platform="cpu"):
    """The index key of a program: its jaxpr with its constants, the options, the platform it runs on, kernelax's sources,
    the compiler and CPU."""
    h = hashlib.sha256("\0".join(["program", str(closed), repr(opts), platform, _VERSION, _cc_version(CC),
                                  f"{frontend.MAX_NODES} {gpu_mod.MAX_NODES}", frontend.numerics()]).encode())  # (a region too large under
    # a limit is built under a higher one)
    _hash_consts(h, closed)
    return "program-" + h.hexdigest()[:24]


def _platform(leaves):
    """The platform a program runs on (as jax.jit places it): that of its committed arrays, else of the default device
    (jax.default_device: a device or a platform's name), else the default backend's."""
    d = jax.config.jax_default_device
    return next((d.platform for x in leaves if isinstance(x, jax.Array) and x.committed for d in x.devices()),
                jax.default_backend() if d is None else (jax.devices(d)[0] if isinstance(d, str) else d).platform)


def _abstract(x):  # an argument as the lowering takes it (_build_now: compiled for the calls' arguments)
    if isinstance(x, jax.Array):
        return jax.ShapeDtypeStruct(x.shape, x.dtype, weak_type=jax.typeof(x).weak_type, sharding=x.sharding if x.committed else None)
    return jax.ShapeDtypeStruct(x.shape, x.dtype) if isinstance(x, np.ndarray) else x


class _Unplanned(Exception):
    """The program with kernels traced for a signature other than its own (kernelax.jit's direct call of it: a new
    signature, to be planned)."""


class _Entry:
    """f for one input signature, from its trace (the one jax.jit makes too). A program seen before (by its jaxpr,
    in the index on disk) is planned at once (_Plan, a cost model, no timing) and runs with its kernels from the
    first call on (loaded from the index: no trace, emit or compile; the XLA part compiles faster than the whole
    program), or as jax.jit if it had no kernels. A new program's first call waits while a thread plans it, builds
    its kernels and compiles the program with them (build_first; beyond BUILD_FIRST_WAIT, and with build_first=False,
    it runs in XLA, as jax.jit, and the build goes on in the background), which then takes over."""

    def __init__(self, f, args, opts, plain, caller, split, sig, build_first=False, donates=False):
        asked = frontend.ASKED[0]
        with frontend.TRACING(True):  # (kernelax.tracing(): the program for the kernels; plain, XLA's, stays without)
            traced = jax.jit(f).trace(*args)
        asked = frontend.ASKED[0] != asked or bool(frontend.ASKED[0]) and any(
            next(jcore.jaxprs_in_params(e.params), None) is not None for e in traced.jaxpr.eqns)  # (a library asked
        # before: its inner traces, jax.jit's and loop bodies', may come from their caches, without a call of tracing())
        self.build_first = build_first  # (the first call builds the kernels and runs with them)
        self.plain = plain  # jax.jit of the caller's function: XLA's program, called on the caller's arguments
        self.caller, self.split, self.sig = caller, split, sig  # (jax.jit with the caller's static arguments, the
        # dynamic leaves and the signature of the caller's arguments, this one's signature: _mixed)
        self.out_tree, self.closed, self.opts = jax.tree.structure(traced.out_info), traced.jaxpr, opts
        self.platform = _platform(jax.tree.leaves(args))
        try:
            key, missing = _program_key(traced.jaxpr, opts, self.platform), None
        except RuntimeError as e:  # (no C compiler, KERNELAX_CC: the program stays in XLA, reported)
            key, missing = None, e
        self.key, self.specs, self.unsaved, self.transient = key, [], [], False
        self.nt = str(dict(opts).get("n_threads") or NT)  # (the threads of the calls: the index's choices are by them)
        self.report = {"status": "xla", "why": "cost model"}  # (as .report() documents it; set by _status)
        self.plan = self.mixed = self.thread = self.hints = None
        j = traced.jaxpr  # outputs that are inputs: not computed (jax.jit copies them into new buffers), the input
        self.forward = {k: j.invars.index(v) for k, v in enumerate(j.outvars) if v in j.invars}  # arrays returned instead
        self.own_xla = bool(self.forward or not asked)  # XLA's program from a trace (xla_fn): this one if it did not
        self._xla_from = (f, jax.tree.map(lambda x: (lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, weak_type=a.weak_type))(
            jax.typeof(x)), args)) if asked else traced.jaxpr  # ask kernelax.tracing() (no second trace), else one without,
        # made when first called (with the kernels from the index, never: 3.8 of a large program's 4.3 s warm start)
        self.ready = threading.Event()  # set once the program with kernels is in use (or failed); variants of K follow
        self.background = Flag()  # (build_first's build: a background one once the first call stopped waiting)
        self.times, self.chosen, self.warm = {"c": [], "xla": []}, None, set()  # (warm: programs called once, untimed)
        self.settled = None  # the program of all further calls ("xla": plain), once compiled and nothing is left to
        # time, build or store
        self.donates, self.xla_s, self.spent = donates, None, False  # (xla_s: xla_fn's first call, its compile; _xla_settled)
        self.first_xla = None  # (seconds of XLA's first call, its compile and run: LONG_CHECK)
        known = _read(self.key) if self.key else {"kernels": 0}
        if missing is not None:
            self._status("failed")
            self.report["errors"] = [str(missing)]
            warnings.warn(f"kernelax.jit: no kernels, the program stays in XLA: {missing}")
        elif known is not None and not known["kernels"] and "why" in known:
            self._status("xla", known["why"])
        use = (known or {}).get("use", {})  # (kernels or XLA, as measured with as many threads)
        if use.get(self.nt) == "xla":  # measured slower with kernels before: as jax.jit
            known = {"kernels": 0}
            self._status("xla", "measured")
        if known is not None and known["kernels"]:
            self._plan(known.get("hints"))
            if all(s.indexed(self.platform) for s in self.specs):  # (measured with other thread counts only: again)
                self.chosen = "c" if use.get(self.nt) == "c" else None  # (measured; else trials: also a choice made
                # without them, by build_first or long calls, or a process that ended before them)
                self._status(*(("kernels", "index") if self.chosen else ("trial",)))
                self.mixed = self._mixed(self._kernels())
                return
        if known is None or known["kernels"]:
            self._status("building")

    def _status(self, status, why=None):  # the report's status and why (jit's report)
        self.report["status"] = status
        if why is None:
            self.report.pop("why", None)
        else:
            self.report["why"] = why

    def _note(self, kernels, use=None):  # the program's index entry: its kernels, the planner's choices (hints), and
        # kernels ("c") or XLA as measured, by the threads of the calls
        if self.key is None:
            return
        why = {"why": "not built"} if not kernels and self.report.get("errors") else {}
        self._store(lambda: _update(self.key, lambda e: {"kernels": kernels, "use": (e or {}).get("use", {}) |
                                                         ({self.nt: use} if use else {}), "hints": self.hints} | why))

    def _store(self, write):  # an index write; one that fails (a full disk, a quota, no rights) warned once, not tried
        try:  # again (the program runs on as chosen, kept in this process only)
            write()
        except OSError as e:
            self.key, self.unsaved = None, []
            warnings.warn(f"kernelax.jit: the index on disk could not be written, this program's choices are not kept: {e}")

    def _plan(self, given=None):  # (given: the choices of an earlier process's plan, replayed)
        _HINTS.notes, _HINTS.given = ({} if given is None else None), given
        try:
            _HINTS.xla_heavy = self._xla_heavy(given)
            with jax.default_device(jax.devices("cpu")[0]), no_gc():  # (the planner's concrete values, folded
                self.plan = _Plan(_inline(self.closed), self.opts, gpu=self.platform == "gpu")  # constants: as the trace's)
            self.hints = _HINTS.notes if given is None else given
        finally:
            _HINTS.notes = _HINTS.given = _HINTS.xla_heavy = None
            _INLINED.cache, _INLINED.platforms = {}, {}
        self.report.update(self.plan.report)
        if self.hints and self.hints.get("xla:heavy"):
            self.report["xla_heavy"] = self.hints["xla:heavy"]
        self.specs = self.plan.specs()
        self.unsaved = list(self.specs)

    def _mixed(self, ok):
        """The program with the kernels of ok, under jax.jit: on the caller's arguments (kernelax.jit then calls it directly
        while it is the only signature, through jax.jit's dispatch; traced for another signature: _Unplanned), with
        forwarded inputs on the leaves (returned outside jax.jit, which would copy them)."""
        program = self.plan.program(ok, self.forward)
        if self.forward:
            return jax.jit(program)

        def run(*args, **kwargs):
            leaves, sig = self.split(args, kwargs)
            if sig != self.sig:
                raise _Unplanned
            return jax.tree.unflatten(self.out_tree, program(*leaves))
        return self.caller(run)

    def _run(self, call, leaves):  # the program with kernels on the caller's arguments (call: args, kwargs)
        if self.forward:
            return jax.tree.unflatten(self.out_tree, self._full(self.mixed(*leaves), leaves))
        return self.mixed(*call[0], **call[1])

    def _kernels(self, defer=False):  # the specs whose kernels build (or load); the others' regions stay in XLA
        # (defer: GPU kernels emitted, their nvcc still running: _waited)
        x64, background = jax.config.jax_enable_x64, getattr(BACKGROUND, "on", False)  # (this thread's, for the others)

        def one(s):  # None if built (or loaded), else the error
            try:
                with jax.enable_x64(x64):
                    BACKGROUND.on = background
                    if s.E == 8 and not x64:
                        raise NotImplementedError("kernelax: a kernel of float64 arrays (64-bit dtypes) needs jax_enable_x64")
                    s.start(self.platform)  # (GPU: emitted here, one at a time, nvcc in parallel after; CPU: in
                    if not defer:  # threads, traced and emitted one at a time, gcc in parallel)
                        s.built(self.platform)
                return None
            except Exception as e:  # noqa: BLE001 (not traceable into a kernel, e.g. too large: this region stays in XLA)
                return e
        return self._keep(thread_map(one, self.specs, 1 if self.platform == "gpu" else JOBS))

    def _waited(self):  # the specs (emitted, _kernels) whose kernels built, once their nvcc ended
        def one(s):
            try:
                s.built(self.platform)
                return None
            except Exception as e:  # noqa: BLE001 (reported: _keep)
                return e
        return self._keep([one(s) for s in self.specs])

    def _keep(self, errors):  # the specs (self.specs, in order) without an error; the errors reported and warned
        for e in errors:  # (a region the cost model chose for a kernel stays in XLA: not covered, too large for gcc, ...)
            if e is not None:
                self.transient |= isinstance(e, GccTimeout) or not isinstance(e, NotImplementedError)
                kind = ("not covered" if type(e) is NotImplementedError else f"too large: {type(e).__name__}"
                        if isinstance(e, NotImplementedError) else f"failed: {type(e).__name__}")
                warnings.warn(f"kernelax.jit: a region stays in XLA ({kind}): {str(e).splitlines()[0] if str(e) else ''}")
                self.report.setdefault("errors", []).append(str(e))
        ok = {s for s, e in zip(self.specs, errors) if e is None}
        self.specs, self.unsaved = [s for s in self.specs if s in ok], [s for s in self.unsaved if s in ok]
        return ok

    def _build(self, first, call, leaves):  # (the first call's arguments; the program compiled for them)
        try:
            self._build_now(first, call, leaves)
        except Exception as e:  # noqa: BLE001 (beyond the build itself, e.g. the index on disk: XLA, reported)
            self._status("failed")
            self.report.setdefault("errors", []).append(str(e))
            warnings.warn(f"kernelax.jit: building the kernels failed, the program stays in XLA: {e}")
        finally:
            self.ready.set()  # (also where the build died: wait() returns)

    def _build_now(self, first, call, leaves):
        if first:  # (after the first call's results, its XLA program had the CPU to itself; []: build_first, before
            try:  # any call, at the caller's priority: it waits)
                jax.block_until_ready(first.pop())
            except RuntimeError:  # (its buffers donated to a later call meanwhile: computed already)
                pass
            self.warm.add("xla")  # (its first call: compiled, untimed)
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)  # this thread, and gcc (one at a time), at
            BACKGROUND.on = True  # low priority: the program keeps running meanwhile
        else:  # (build_first: at the caller's priority while it waits; beyond BUILD_FIRST_WAIT the gcc started later run
            BACKGROUND.on = self.background  # in the background, their time limit counting CPU seconds)
        try:
            if self.plan is None:
                self._plan()
            if self.platform == "gpu" and self.specs:  # (no kernels: no runtime, which a new cache builds with nvcc)
                gpu_mod.runtime()  # (its FFI target registered: the program compiles while nvcc runs)
            compile = lambda m: (m.lower(*leaves) if self.forward else m.lower(*call[0], **call[1])).compile()  # (here,
            emitted = self._kernels(defer=True)  # for arguments as the calls' so that they find it; called through
            mixed = self._mixed(emitted) if emitted else None  # jax.jit's dispatch: a Compiled object's call costs
            if mixed is not None:  # ~60 us more)
                compile(mixed)  # (GPU kernels: while their nvcc runs)
            ok = self._waited()
            if ok != emitted:  # (a kernel failed in nvcc: the program without it)
                mixed = self._mixed(ok) if ok else None
                if mixed is not None:
                    compile(mixed)
        except Exception as e:  # noqa: BLE001 (reported, and the program stays in XLA)
            self._status("failed")
            self.report.setdefault("errors", []).append(str(e))
            warnings.warn(f"kernelax.jit: building the kernels failed, the program stays in XLA: {e}")
            self.ready.set()
            return
        if ok or not self.transient:  # (all failed for a passing reason, e.g. gcc's: a later process builds again)
            self._note(len(ok))
        self.mixed = mixed
        self._status(*(("trial",) if ok else ("xla", "not built" if self.report.get("errors") else "cost model")))
        self.ready.set()
        self.unsaved = list(ok)  # (their chosen K into the index, once chosen)
        if self.platform != "cpu" or not ok:  # (GPU kernels: their variants built with them, gpu.build)
            return

        def variants():  # the variants of K > 1 (Kernel.complete), at low priority while the program runs on K = 1
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)

            def one(s):
                BACKGROUND.on = True
                s.kernel.complete()
            thread_map(one, ok, JOBS)  # (gcc processes at a time: build._gcc)
        variants()  # (in this thread: the program runs on K = 1 meanwhile)

    def _start(self, first, call, leaves):  # _build in a thread (first: [the first call's results], or [] for build_first)
        x64, device = jax.config.jax_enable_x64, jax.config.jax_default_device  # (the caller's: also a thread-local
        # jax.enable_x64 or jax.default_device, as the program's platform: _platform)

        def build(*a):
            with jax.enable_x64(x64), jax.default_device(device):
                self._build(*a)
        self.thread = threading.Thread(target=build, args=(first, *jax.tree.map(_abstract, (call, leaves))), daemon=True)
        # (the arguments as shapes, the results until computed: their buffers not held while the kernels build)
        _THREADS.append(self.thread)
        self.thread.start()

    def _xla_heavy(self, given):
        """The heavy operations of XLA's own program that the program for the kernels lacks, if it asked
        kernelax.tracing() (jnp.linalg.det's derivative without JAX's LU solve, frontend._adjugate_rule): without kernels,
        XLA's program runs, so the cost model counts them for every region (_worth; loops and conditionals not: the rules
        on them concern the kernels' program, a kernel call per iteration); noted with the planner's choices (a known
        program: no trace of XLA's program)."""
        if given is not None:
            return given.get("xla:heavy", [])
        heavy = sorted((_prims(self.xla_jaxpr) - _prims(self.closed)) & (GPU_HEAVY if self.platform == "gpu" else _HEAVY)
                       - set(_NESTED) if isinstance(self._xla_from, tuple) else [])
        _HINTS.notes["xla:heavy"] = heavy
        return heavy

    @functools.cached_property
    def xla_jaxpr(self):  # XLA's own program (see own_xla)
        return jax.jit(self._xla_from[0]).trace(*self._xla_from[1]).jaxpr if isinstance(self._xla_from, tuple) else self._xla_from

    @functools.cached_property
    def xla_fn(self):
        u = self.xla_jaxpr
        kept = [v for k, v in enumerate(u.outvars) if k not in self.forward]
        return jax.jit(jcore.jaxpr_as_fun(u.replace(outvars=kept)))

    def _full(self, outs, leaves):  # the flat outputs: a program's, the forwarded inputs at their positions
        it = iter(outs)
        return [(lambda x: jnp.array(x, copy=True) if isinstance(x, jax.Array) else jnp.asarray(x))(leaves[self.forward[k]])
                if k in self.forward else next(it) for k in range(self.out_tree.num_leaves)]  # (new arrays, as jax.jit's)

    def _xla_settled(self):
        """The program of further calls settled on XLA: plain's ("xla"), or xla_fn where inputs are returned (those as new
        arrays, _full) or where it ran already (own_xla, nothing donated): plain's compile saved, until the calls'
        longer dispatch through kernelax.jit (DISPATCH each) has cost as much as that compile (then plain's: at most
        twice the better of both)."""
        if self.forward:
            return lambda c, l: jax.tree.unflatten(self.out_tree, self._full(self.xla_fn(*l), l))
        if not self.own_xla or self.donates or self.xla_s is None or self.spent:
            return "xla"
        left = [self.xla_s / DISPATCH]

        def run(c, l):
            left[0] -= 1
            if left[0] < 0:  # (the next call through the entry: plain's from then on, also kernelax.jit's shortcut
                self.spent, self.settled = True, None  # to it, xla_only)
            return jax.tree.unflatten(self.out_tree, self.xla_fn(*l))
        return run

    def __call__(self, call, leaves, dispatch):  # (call: the caller's arguments, (args, kwargs); dispatch: the work
        if self.own_xla:  # kernelax.jit does per call to find the program with kernels)
            def xla():
                t0 = time.perf_counter()
                out = jax.tree.unflatten(self.out_tree, self._full(self.xla_fn(*leaves), leaves))
                self.xla_s = self.xla_s or time.perf_counter() - t0  # (its first call: its compile)
                return out
        else:
            xla = lambda: self.plain(*call[0], **call[1])
        run = lambda: self._run(call, leaves)
        if self.mixed is None and self.build_first and self.thread is None and self.report["status"] == "building":
            self._start([], call, leaves)  # (built now, at the caller's priority; this call waits up to BUILD_FIRST_WAIT,
            if self.ready.wait(None if BUILD_FIRST_WAIT == math.inf else BUILD_FIRST_WAIT) and self.mixed is not None:
                t0 = time.perf_counter()  # (else in XLA; trials once the variants of K are built, as after a build in
                out = jax.block_until_ready(run())  # the background)
                self.report["first_ms"] = round(1e3 * (time.perf_counter() - t0), 3)
                return out
            self.background.on = True  # (the build goes on in the background while this call runs in XLA)
        if self.mixed is None:
            t0 = time.perf_counter()
            out = xla()
            if self.first_xla is None:  # (LONG_CHECK: its compile and run)
                jax.block_until_ready(out)
                self.first_xla = self.xla_s or time.perf_counter() - t0
            self.warm.add("xla")  # (its first call: compiled, untimed)
            if self.thread is None and self.report["status"] == "building":  # after this call (_build)
                self._start([out], call, leaves)
            elif self.mixed is None and self.report["status"] != "building":  # (none to build, or XLA chosen; not a
                self.settled = self._xla_settled()  # build that finished during this call: trials from the next)
            return out
        if self.chosen is None and self.thread is not None and self.thread.is_alive():  # (variants still building: gcc and the choice of K would
            return run()  # bias the trials below; no trial yet)
        if self.chosen is None:  # the program with kernels against XLA's, on the calls themselves (no extra runs): the
            c = len(self.times["c"]) <= len(self.times["xla"])  # faster one stays (so kernelax.jit is not slower than jax.jit)
            t0 = time.perf_counter()  # (each as called once chosen: with kernels after kernelax.jit's dispatch, XLA's
            out = jax.block_until_ready((dispatch(), run())[1] if c else xla())
            dt = time.perf_counter() - t0  # (through jax.jit's; a program's first call untimed: it loads its kernels, the
            if c and "c" in self.warm and not self.times["c"] and dt > TRIAL_MAX_CALL and not (  # allocator grows for
                    self.first_xla is not None and dt > LONG_CHECK * self.first_xla):  # its buffers, the runtime chooses K) a
                self.chosen = "c"  # long call: the cost model's choice, no trials (unless near XLA's: one of each)
                self._status("kernels", "long calls")
                self._note(len(self.specs))
            elif ("c" if c else "xla") in self.warm:
                self.times["c" if c else "xla"].append(dt)
            elif c and "first_ms" not in self.report:  # (the first call with kernels, untimed: build_first's first)
                self.report["first_ms"] = round(1e3 * dt, 3)
            self.warm.add("c" if c else "xla")
            need = 1 if self.times["c"] and min(self.times["c"]) > TRIAL_MAX_CALL else TRIALS  # (long calls: one each)
            if min(map(len, self.times.values())) >= need and sum(map(sum, self.times.values())) >= TRIAL_TIME:
                self.chosen = "c" if min(self.times["c"]) <= min(self.times["xla"]) * (1 + TRIAL_MARGIN) else "xla"
                self._status("kernels" if self.chosen == "c" else "xla", "measured")
                self.report["ms"] = {k: round(1e3 * min(v), 3) for k, v in self.times.items()}
                self._note(len(self.specs), self.chosen)
                if self.chosen == "xla":  # (settled at once: the next call is plain's, which donates as jax.jit)
                    self.mixed = None
                    self.settled = self._xla_settled()
                    return out
            if not c:
                return out
        else:
            out = run()
        if self.unsaved:  # the chosen K into the index, once known
            jax.block_until_ready(out)
            self._store(lambda: [s.built(self.platform).persist() for s in self.unsaved])
            self.unsaved = [s for s in self.unsaved if not s.built(self.platform).saved and not (
                self.platform == "cpu" and s.built(self.platform).never_chooses())]  # (calls too small to choose K)
        if not self.unsaved and self.chosen is not None:
            self.settled = self._run
        return out


def jit(f=None, *, static_argnums=(), static_argnames=(), donate_argnums=(), donate_argnames=(), n_threads=None,
        keep_nans=False, build_first=None):
    """Drop-in for ``jax.jit``: the sample-parallel regions of f run as C kernels where a cost model expects them to be
    faster (_worth), and stay so where the calls measure faster than XLA's (trials). The first call of a signature
    builds its kernels and runs with them (build_first), unless that takes longer than KERNELAX_BUILD_FIRST_WAIT
    seconds: then it runs in XLA and the kernels follow in a thread (``.wait()`` waits for them); with the kernels on
    disk it runs them at once. ``.report()`` tells per signature what runs. Inside other transformations (an enclosing jit, grad, vmap) f is jax.jit(f), so
    that an enclosing kernelax.jit sees its equations. static_argnums, static_argnames: as in jax.jit (hashable values,
    part of the signature). donate_argnums, donate_argnames: as in jax.jit (the arguments' buffers may be reused for
    the outputs, by XLA's program and the one with kernels; not by the first calls' and trials' XLA program of a
    function that does not ask kernelax.tracing(), nor where inputs are returned as outputs). n_threads, keep_nans: as in
    kernelax.kernel. build_first: as above (default: the environment's KERNELAX_BUILD_FIRST, 1 unless set); False: the
    first calls run in XLA while the kernels build in a thread (for programs called many times, whose first calls
    should cost what jax.jit's do). Kernels still building at exit are finished for the index (KERNELAX_EXIT_WAIT)."""
    if f is None:
        return functools.partial(jit, static_argnums=static_argnums, static_argnames=static_argnames,
                                 donate_argnums=donate_argnums, donate_argnames=donate_argnames, n_threads=n_threads,
                                 keep_nans=keep_nans, build_first=build_first)
    build_first = BUILD_FIRST if build_first is None else bool(build_first)
    opts = _opts(n_threads, keep_nans)
    entries = {}
    static_argnums, static_argnames, pos, parts = statics(f, static_argnums, static_argnames)
    donate = dict(zip(("donate_argnums", "donate_argnames"), statics(f, donate_argnums, donate_argnames)[:2]))
    plain = jax.jit(f, static_argnums=static_argnums, static_argnames=static_argnames, **donate)  # (both given: jax.jit
    # infers neither from the other)
    plain_size = internals.jit_cache_size(plain)
    lock = threading.RLock()  # (calls from several threads: an entry made once, its trials and build started once)
    xla_only = [False]  # every signature seen settled to XLA: calls go to plain (jax.jit's dispatch, ~10 us less)
    direct = [None]  # the only signature seen, settled to kernels: its program, called directly (likewise)
    caller = lambda fn: jax.jit(fn, static_argnums=static_argnums, static_argnames=static_argnames, **donate)

    def signature(args, kwargs):  # parts; key None: tracers, or concrete arguments under a trace (part of the
        static, names, dyn, leaves, key = parts(args, kwargs)  # enclosing program)
        if any(isinstance(x, jax.core.Tracer) for x in leaves) or not internals.top_level():
            key = None
        else:
            try:
                hash(key)
            except TypeError:
                raise ValueError("kernelax.jit: static arguments must be hashable (as in jax.jit)") from None
        return static, names, dyn, leaves, key

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if direct[0] is not None and internals.top_level():
            try:
                return direct[0](*args, **kwargs)
            except _Unplanned:  # (a signature not seen: planned below)
                direct[0] = None
        if xla_only[0]:
            n = plain_size()
            out = plain(*args, **kwargs)
            if plain_size() != n:  # (a signature not seen: planned from its next call on)
                xla_only[0] = False
            return out
        static, names, dyn, leaves, key = signature(args, kwargs)
        if key is None:
            return plain(*args, **kwargs)
        if (e := entries.get(key)) is not None and e.settled is not None:  # (compiled: outside the context below, which
            return plain(*args, **kwargs) if e.settled == "xla" else e.settled((args, kwargs), leaves)
        with internals.no_constant_folding(), lock:  # (as jax's dispatch: also inside jax.ensure_compile_time_eval, the
            if key not in entries:  # program and the kernel bodies are traced, not evaluated)
                entries[key] = _Entry(bind(f, len(args), static, names), dyn, opts, plain, caller, lambda a, kw: parts(a, kw)[3:], key,
                                      build_first, any(donate.values()))
            e = entries[key]
            out = e((args, kwargs), leaves, lambda: None if len(entries) == 1 and not e.forward else  # (called directly)
                    entries.get(signature(args, kwargs)[-1]))
            xla_only[0] = all(e.settled == "xla" for e in entries.values())  # (under the lock: another thread's new
            direct[0] = e.mixed if len(entries) == 1 and e.settled is not None and e.settled != "xla" and not e.forward else None
        return out  # signature changed the dict while it was read)

    def wait(variants=False):
        """Waits until the kernels being built are in use (or failed: see .report()); with `variants`, also until
        their variants of K > 1 are built (Kernel.complete: they follow in the background)."""
        with lock:
            pending = [e for e in entries.values() if e.thread is not None]
        for e in pending:
            e.thread.join() if variants else e.ready.wait()

    def report():
        """Per signature called (in the order of their first calls), a dict (a copy):
        - signature: the dtypes and shapes of the dynamic arguments' leaves, then the static values;
        - status: building (the kernels are being built, the calls run in XLA), trial (the calls alternate between the
          program with kernels and XLA's, timed), kernels, xla, or failed (the build failed: XLA);
        - why (kernels and xla): index (as measured or chosen before), measured, long calls (the cost model's choice,
          no trials), cost model (no region worth a kernel), not built (no kernel could be built);
        - the plan: N (the sample axes' lengths), regions (per region: work, eqns, chosen "c" or "xla", ...), loops,
          xla_heavy (heavy operations only XLA's own program has, if f asked kernelax.tracing(): counted for every region);
        - errors (regions left in XLA, also warned), ms (the trials' best, ms), first_ms (the first call with kernels,
          ms)."""
        out = []
        with lock:
            items = list(entries.items())
        for (_, avals, static, names, *_), e in items:
            fixed = [f"{pos[i] if i < len(pos) else f'arg{i}'}={v!r}" for i, v in static] + [f"{k}={v!r}" for k, v in names]
            sig = ", ".join(a.str_short(short_dtypes=True) + "{weak}" * a.weak_type for a in avals) + ("; " + ", ".join(fixed) if fixed else "")
            rep = {"signature": sig, **e.report}
            out.append(rep | ({"errors": list(rep["errors"])} if "errors" in rep else {}))
        for r, (key, _) in zip(out, items):  # (a signature called with x64 and without: told apart)
            if [q["signature"] for q in out].count(r["signature"]) > 1:
                r["signature"] += f" (x64 {'on' if key[-1] else 'off'})"
        return out
    wrapped.wait, wrapped.report = wait, report
    return wrapped
