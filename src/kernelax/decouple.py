# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""While loops coupled over the samples (any sample active, one step counter: equinox's under vmap) as one loop per
sample, where proved on the jaxpr (_decouple): the loops' conditions and carries as terms (_Terms),
the lanes they couple (_lanes), the loop rebuilt per lane (_per_lane, lanes_p)."""

import functools
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core as jcore
from jax.interpreters import mlir

from kernelax.frontend import _CALLS, _sub_jaxpr
from kernelax.jaxprs import _EW, _INLINED, _NESTED, _bind, _inline, _pkey, _var


def _lanes_impl(x, *, op):
    return jnp.any(x) if op == "any" else jnp.max(x)


lanes_p = jcore.Primitive("kernelax_lanes")  # any / max over all axes; per sample (in a loop of _decouple): its own


lanes_p.def_impl(_lanes_impl)


lanes_p.def_abstract_eval(lambda x, *, op: jax.core.ShapedArray((), x.dtype))


mlir.register_lowering(lanes_p, mlir.lower_fun(_lanes_impl, multiple_results=False))


class _Terms:
    """Hash-consed terms of the values of jaxprs (leaves: inputs, carries, constants), simplified on known booleans,
    selects, conds and copies; under assumptions `sub` (term -> term: a predicate is true, a carry false), two
    values are equal if their terms are (sound, not complete)."""

    def __init__(self):
        self.ids, self.key, self.aval, self._free, self._lits = {}, [], [], {}, {}

    def make(self, key, aval):
        if key not in self.ids:
            self.ids[key] = len(self.key)
            self.key.append(key)
            self.aval.append((tuple(aval.shape), aval.dtype))
        return self.ids[key]

    def leaf(self, tag, aval):
        return self.make(("leaf", tag), aval)

    def lit(self, x, aval):
        if id(x) not in self._lits:  # (keys: by identity)
            a = np.asarray(x) if isinstance(aval.dtype, np.dtype) else None
            self._lits[id(x)] = (x, ("id", id(x)) if a is None else a.flat[0].item() if a.size and (a == a.flat[0]).all()
                                 else (a.shape, a.tobytes()))
        return self.make(("lit", str(aval.dtype), tuple(aval.shape), self._lits[id(x)][1]), aval)

    def const(self, t):  # the value of a constant term with all elements equal, else None
        k = self.key[t]
        return k[3] if k[0] == "lit" and not isinstance(k[3], tuple) else None

    def free(self, t):  # the leaves a term depends on
        if t not in self._free:
            k = self.key[t]
            self._free[t] = {t} if k[0] == "leaf" else set() if k[0] == "lit" else set().union(*map(self.free, k[2]))
        return self._free[t]

    def conj(self, t):  # the operands of a conjunction of booleans
        k = self.key[t]
        return [x for y in k[2] for x in self.conj(y)] if k[0] == "and" and self.aval[t][1] == bool else [t]

    def scalar(self, t):  # the scalar whose broadcast t is (a scalar, a broadcast of one, a constant), else None
        k, (shape, dtype) = self.key[t], self.aval[t]
        if not shape or k[0] == "bcast":
            return t if not shape else k[2][0]
        return None if self.const(t) is None else self.lit(np.asarray(self.const(t), dtype), jax.core.ShapedArray((), dtype))

    def bcast(self, t, shape):  # the scalar t broadcast to shape
        dtype, c = self.aval[t][1], self.const(t)
        if c is not None and all(shape):  # (lit's term of the constant array, without making it)
            return self.make(("lit", str(dtype), tuple(shape), np.asarray(c, dtype).item()), jax.core.ShapedArray(shape, dtype))
        return self.lit(np.full(shape, c, dtype), jax.core.ShapedArray(shape, dtype)) if c is not None \
            else self.make(("bcast", (), (t,), 0), jax.core.ShapedArray(shape, dtype))

    def node(self, eqn, kids, j, sub):
        return self.op(eqn.primitive, eqn.params, kids, j, eqn.outvars[j].aval, sub)

    def op(self, prim, params, kids, j, a, sub):
        name, same = prim.name, lambda t: self.aval[t] == (tuple(a.shape), a.dtype)
        if name in _EW and a.shape:  # (scalar operands: broadcast)
            kids = [k if self.aval[k][0] else self.bcast(k, tuple(a.shape)) for k in kids]
        c = [self.const(k) for k in kids]
        t = None
        if name in ("copy", "stop_gradient") or name == "convert_element_type" and same(kids[0]):
            t = kids[0]
        elif name in ("and", "or") and a.dtype == bool and any(x is not None for x in c):
            unit, rest = name == "and", [k for k, x in zip(kids, c) if x is None]  # (and: True leaves the other)
            t = (self.lit(np.full(a.shape, not unit), a) if any(x is not None and bool(x) != unit for x in c) else
                 self.lit(np.full(a.shape, unit), a) if not rest else rest[0] if len(rest) == 1 and same(rest[0]) else None)
        elif name == "select_n" and (c[0] is not None or all(k == kids[1] for k in kids[2:])):
            pick = kids[1 + int(c[0])] if c[0] is not None else kids[1]
            t = pick if same(pick) else None
        elif name == "broadcast_in_dim" and (c[0] is not None or not self.aval[kids[0]][0]):  # (of a scalar: bcast)
            t = self.bcast(self.scalar(kids[0]), tuple(a.shape))
        elif name in _EW and a.shape and all(self.scalar(k) is not None for k in kids):  # (on scalars, then broadcast)
            t = self.bcast(self.op(prim, params, [self.scalar(k) for k in kids], j, jax.core.ShapedArray((), a.dtype), sub), tuple(a.shape))
        elif name in _EW and not a.shape and all(x is not None for x in c) and not prim.multiple_results:  # (constants)
            t = self.lit(np.asarray(prim.bind(*[np.asarray(x, self.aval[k][1]) for x, k in zip(c, kids)], **params), a.dtype), a)
        if t is None:
            t = self.make((name, _pkey(params), tuple(kids), j), a)
        return sub.get(t, t)

    def any(self, t):  # x if t is an any over the samples of x (a reduce_or of a vector of booleans), else None
        k = self.key[t]
        return k[2][0] if k[0] == "reduce_or" and len(self.aval[k[2][0]][0]) == 1 and self.aval[k[2][0]][1] == bool else None

    def run(self, closed, ins, sub, seen=None, over=None):
        """Terms of the outputs of closed on terms `ins`; seen: (index, operand, result) of its anys over the samples;
        over: equation index -> the term of its (one) result, in place of the equation's."""
        j = closed
        env = {v: sub.get(t, t) for v, t in zip(j.invars, ins)} | {v: self.lit(x, v.aval) for v, x in zip(j.constvars, closed.consts)}
        for i, eqn in enumerate(j.eqns):
            kids = [env[v] if _var(v) else self.lit(v.val, v.aval) for v in eqn.invars]
            env.update(zip(eqn.outvars, [over[i]] if over and i in over else self.eqn(eqn, kids, sub)))
            if seen is not None and eqn.primitive.name == "reduce_or" and self.any(env[eqn.outvars[0]]) is not None:
                seen.append((i, kids[0], env[eqn.outvars[0]]))
        return [env[v] if _var(v) else self.lit(v.val, v.aval) for v in j.outvars]

    def eqn(self, eqn, kids, sub):
        if eqn.primitive.name in _CALLS:
            return self.run(_sub_jaxpr(eqn)[0], kids, sub)
        if eqn.primitive.name == "cond" and self.const(kids[0]) is not None:  # (a known branch)
            return self.run(eqn.params["branches"][int(self.const(kids[0]))], kids[1:], sub)
        return [self.node(eqn, kids, j, sub) for j in range(len(eqn.outvars))]


def _nested(eqn):  # the jaxprs in eqn's params
    return list(jcore.jaxprs_in_params(eqn.params))


def _uses(eqn, name):  # whether eqn is or holds an equation of primitive `name`
    return eqn.primitive.name == name or any(_uses(e, name) for c in _nested(eqn) for e in c.eqns)


def _unvmapped(closed):
    """The number of equinox' unvmap_any in closed and the calls in it, not in its loops and conditionals (before
    _inline expands them; a body whose nested loops _decouple replaced: the original's)."""
    if id(closed) in _UNVMAPPED and _UNVMAPPED[id(closed)][0]() is closed:
        return _UNVMAPPED[id(closed)][1]
    return sum(e.primitive.name == "unvmap_any" for e in closed.eqns) + \
        sum(_unvmapped(c) for e in closed.eqns if e.primitive.name not in _NESTED for c in _nested(e))


_UNVMAPPED = {}  # id of a body _decouple_in made -> (weak reference to it, _unvmapped of the original)


def _anys(closed):  # the anys over vectors (reduce_or to a scalar) in closed, also in nested jaxprs
    return [e for e in closed.eqns if e.primitive.name == "reduce_or" and e.invars[0].aval.ndim == 1] + \
        [x for e in closed.eqns for c in _nested(e) for x in _anys(c)]


def _lanes(T, eqn, ins, used):
    """(N, counters, anys over the samples of the condition and of the body) if the while eqn (operand terms `ins`)
    is the same as one loop per sample with its own counters and, for each any over the samples, the sample's own
    value; else None. Proved on terms: the condition is any(x) and bounds on uniform carries that count steps (+1
    each); the body leaves a sample with x false as it is (given a carry m equal to x: kept so by the body and at
    the start); an any in the body changes nothing for a sample whose own value is false, unless all of them are
    equinox' unvmap_any (shared control flow under vmap: per sample its own value, as the program without vmap
    computes it; e.g. diffrax' first-same-as-last stage, an efficiency only); no effects. All of it for
    the carries whose results are `used` (indices) and what they and the condition depend on (the others may differ:
    a carry set to another's previous value, as equinox's, ends as it would if the sample had stopped later)."""
    p = eqn.params
    nc, nb = p["cond_nconsts"], p["body_nconsts"]
    cj, bj = _inline(p["cond_jaxpr"]), _inline(p["body_jaxpr"])
    if any(e.effects for e in cj.eqns + bj.eqns):
        return None
    avals = [v.aval for v in eqn.invars[nc + nb:]]
    s = [T.leaf(("carry", id(eqn), k), a) for k, a in enumerate(avals)]
    cin, bin, start = ins[:nc], ins[nc:nc + nb], ins[nc + nb:]

    def pred(carry, sub):  # (x, the other operands of the condition's conjunction), or None
        seen = []
        out = T.conj(T.run(cj, cin + carry, sub, seen)[0])
        anys = [t for t in out if T.any(t) is not None]  # (one any over the samples in all: x's; others couple)
        return (T.any(anys[0]), [t for t in out if t != anys[0]], [i for i, _, _ in seen]) if len(anys) == len(seen) == 1 else None
    top = lambda closed: sum(e.primitive.name == "reduce_or" and e.invars[0].aval.ndim == 1 for e in closed.eqns)
    first = pred(s, {})
    if first is None or len(_anys(cj)) != top(cj) or len(_anys(bj)) != top(bj):
        return None  # (anys in nested jaxprs: not covered)
    x, rest, in_cond = first
    N = T.aval[x][0][0]
    rows = [k for k, a in enumerate(avals) if a.shape[:1] == (N,)]
    outs = T.run(bj, bin + s, {})
    counters = [k for k, a in enumerate(avals) if k not in rows and outs[k] != s[k]]
    for k in counters:  # (the others: the same in every step)
        a = avals[k]
        if a.shape or not np.issubdtype(a.dtype, np.integer) or T.key[outs[k]][0] != "add" or \
                sorted(T.key[outs[k]][2]) != sorted([s[k], T.lit(np.ones((), a.dtype), a)]):
            return None
    lanes = {s[k] for k in rows}
    if any(T.free(t) & lanes for t in rest) or T.free(x) & {s[k] for k in counters}:
        return None
    live, more = set(), set(used) | {k for k in range(len(s)) if s[k] in T.free(x) or any(s[k] in T.free(t) for t in rest)}
    while more:
        live |= more
        more = {j for k in more for j in range(len(s)) if s[j] in T.free(outs[k])} - live
    rows = [k for k in rows if k in live]
    known = lambda v, t: T.lit(np.full(T.aval[t][0], v), jax.core.ShapedArray(*T.aval[t]))
    for m in [None] + [k for k in rows if avals[k].shape == (N,) and avals[k].dtype == bool]:
        hyp = lambda v: {x: known(v, x)} | ({s[m]: known(v, s[m])} if m is not None else {})
        done = T.run(bj, bin + s, hyp(False))
        if any(done[k] != hyp(False).get(s[k], s[k]) for k in rows):
            continue  # (a sample with x false: not left as it is)
        step, then, init = T.run(bj, bin + s, hyp(True)), None, None
        if m is not None:
            then, init = pred(step, hyp(True)), pred(start, {})
            if then is None or init is None or step[m] != then[0] or start[m] != init[0]:
                continue
        break
    else:
        return None
    seen = []
    T.run(bj, bin + s, hyp(True), seen)
    shared = _unvmapped(p["body_jaxpr"]) >= len(seen)  # (all of them equinox' unvmap_any: shared control flow under
    for i, z, u in seen:  # vmap, the sample's own value as without vmap: e.g. diffrax' first-same-as-last stage)
        if T.aval[z] != ((N,), bool):
            return None
        if shared:
            continue
        a, b = (T.run(bj, bin + s, hyp(True) | {z: known(False, z)}, over={i: known(v, u)}) for v in (True, False))
        if any(a[k] != b[k] for k in live):  # (z true: any is true, as the sample's own value; z false: any either
            return None  # way, the same)
    return N, counters, set(in_cond), {i for i, _, _ in seen}


def _decouple(closed):
    """closed with the while loops that are one loop per sample (_lanes; nested ones first) as such: counters per
    sample (the program's: their maximum), the anys over the samples as lanes_p. The same values, also run batched
    in XLA. Cached while a program is planned (_INLINED; a result stays as it is: the bodies of its loops, planned
    on their own, are decoupled already)."""
    cache = _INLINED.__dict__.setdefault("cache", {})
    if ("decoupled", id(closed)) not in cache:
        out = _decouple_now(closed)
        cache[("decoupled", id(closed))] = (closed, out)
        cache.setdefault(("decoupled", id(out)), (out, out))
    return cache[("decoupled", id(closed))][1]


def _decouple_now(closed):
    j = closed
    if not any(_uses(e, "while") for e in j.eqns):
        return closed
    eqns = [_decouple_in(e) for e in j.eqns]
    T, plans = _Terms(), {}
    env = {v: T.leaf(("in", k), v.aval) for k, v in enumerate(j.invars)} | {v: T.lit(x, v.aval) for v, x in zip(j.constvars, closed.consts)}
    read = {v for e in eqns for v in e.invars if _var(v)} | {v for v in j.outvars if _var(v)}
    for i, e in enumerate(eqns):
        ins = [env[v] if _var(v) else T.lit(v.val, v.aval) for v in e.invars]
        if e.primitive.name == "while" and (plan := _lanes(T, e, ins, {k for k, v in enumerate(e.outvars) if v in read})) is not None:
            plans[i] = plan
        env.update(zip(e.outvars, T.eqn(e, ins, {})))
    if not plans and all(a is b for a, b in zip(eqns, j.eqns)):
        return closed
    return _inline(_rebuild(closed, eqns, {i: functools.partial(_per_lane, e, plans[i]) for i, e in enumerate(eqns) if i in plans}))


def _rebuild(closed, eqns, replace):
    """closed with the equations eqns (the same results), those at the indices in replace (index -> f(list of operand
    values) -> results) by f's jaxpr, traced on their operands, spliced in; the others as they are (not traced again)."""
    j, env, new, constvars, consts = closed, {}, [], list(closed.constvars), list(closed.consts)
    read = lambda v: env.get(v, v) if _var(v) else v
    for i, e in enumerate(eqns):
        ins = [read(v) for v in e.invars]
        if i not in replace:
            new.append(e.replace(invars=ins))
            continue
        at = [k for k, v in enumerate(ins) if _var(v)]

        def f(*a, ins=ins, at=at, g=replace[i]):
            vals = [None if _var(v) else v.val for v in ins]  # (literals' values; the variables' below)
            for k, x in zip(at, a):
                vals[k] = x
            return g(vals)
        t = jax.make_jaxpr(f)(*[ins[k].aval for k in at])
        sub = dict(zip(t.invars, [ins[k] for k in at]))
        for v, c in zip(t.constvars, t.consts):
            sub[v] = jcore.Var(v.aval)
            constvars.append(sub[v])
            consts.append(c)
        new += [te.replace(invars=[sub.get(v, v) if _var(v) else v for v in te.invars]) for te in t.eqns]
        env.update((o, sub.get(to, to) if _var(to) else to) for o, to in zip(e.outvars, t.outvars))
    return jcore.Jaxpr(constvars, j.invars, [read(v) for v in j.outvars], new, frozenset().union(*(e.effects for e in new)),
                       j.debug_info, consts=consts)


def _decouple_in(eqn):  # eqn with the loops in its bodies and branches decoupled
    p, name = eqn.params, eqn.primitive.name
    if name not in _NESTED or not _uses(eqn, "while"):
        return eqn
    keys = ["cond_jaxpr", "body_jaxpr"] if name == "while" else ["jaxpr"] if name == "scan" else []
    new = {k: [p[k]] for k in keys} | ({"branches": list(p["branches"])} if name == "cond" else {})
    new = {k: [(c, _inline(c), _decouple(_inline(c))) for c in v] for k, v in new.items()}
    for c, a, b in (x for v in new.values() for x in v):
        if a is not b:  # (a body with its loops decoupled: the count of the original's unvmap_any, for _lanes)
            _UNVMAPPED[id(b)] = (weakref.ref(b), _unvmapped(c))
    new = {k: [(a, b) for _, a, b in v] for k, v in new.items()}
    if all(a is b for v in new.values() for a, b in v):
        return eqn
    return eqn.replace(params=dict(p, **{k: tuple(b for _, b in v) if k == "branches" else v[0][1] for k, v in new.items()}))


def _per_lane(eqn, plan, vals):  # the while eqn as _decouple runs it
    N, counters, in_cond, in_body = plan
    p = eqn.params
    nc, nb = p["cond_nconsts"], p["body_nconsts"]
    vec = lambda c: jnp.broadcast_to(jnp.asarray(c, c.dtype), (N,))

    def ev(closed, consts, carry, lanes):
        carry = [lanes_p.bind(c, op="max") if k in counters else c for k, c in enumerate(carry)]
        env = dict(zip(closed.constvars, closed.consts)) | dict(zip(closed.invars, list(consts) + carry))
        for i, e in enumerate(closed.eqns):
            ins = [v.val if not _var(v) else env[v] for v in e.invars]
            env.update(zip(e.outvars, [lanes_p.bind(ins[0], op="any")] if i in lanes else _bind(e, e.params, ins)))
        return [v.val if not _var(v) else env[v] for v in closed.outvars]
    cond = lambda c: ev(_inline(p["cond_jaxpr"]), vals[:nc], c, in_cond)[0]
    body = lambda c: [vec(o) if k in counters else o for k, o in enumerate(ev(_inline(p["body_jaxpr"]), vals[nc:nc + nb], c, in_body))]
    out = lax.while_loop(cond, body, [vec(c) if k in counters else c for k, c in enumerate(vals[nc + nb:])])
    return [jnp.max(o) if k in counters else o for k, o in enumerate(out)]
