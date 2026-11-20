# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Reference evaluation of the DAG with numpy, vectorized over samples (benchmarks/run.py: loop iteration counts)."""

from collections import ChainMap

import numpy as np

from kernelax.dag import _NP_CMP, _NP_UNARY, Node, is_c, schedule


_NP_OPS = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "neg": np.negative,
           "recip": lambda a: 1.0 / a, "pow": np.power, "atan2": np.arctan2, "fmod": np.fmod, "max": np.maximum, "min": np.minimum,
           "select": lambda p, f, t: np.where(p, t, f),
           "table": lambda t, i: np.asarray(t)[np.clip(np.nan_to_num(i), 0, len(t) - 1).astype(np.int64)],
           "b2i": lambda p: np.asarray(p).astype(np.int64),
           "ftoi": lambda x: np.trunc(np.nan_to_num(x)).astype(np.int64), "fbits": lambda x: np.asarray(x, np.float64).view(np.int64),
           "itof": lambda a, u: (_i(a).view(np.uint64) if u else _i(a)).astype(np.float64),
           "bitsf64": lambda a: _i(a).view(np.float64),
           "bitsf32": lambda a: (_i(a) & 0xFFFFFFFF).astype(np.uint32).view(np.float32),
           "f32bits": lambda x: np.asarray(x, np.float32).view(np.int32).astype(np.int64),
           "tof32": lambda x: np.asarray(x, np.float32), "tof64": lambda x: np.asarray(x, np.float64),
           "iadd": lambda a, b: _i(a) + _i(b), "isub": lambda a, b: _i(a) - _i(b), "imul": lambda a, b: _i(a) * _i(b),
           "iand": lambda a, b: _i(a) & _i(b), "ior": lambda a, b: _i(a) | _i(b), "ixor": lambda a, b: _i(a) ^ _i(b),
           "ishl": lambda a, s: _i(a) << _i(s), "ishr": lambda a, s: _i(a) >> _i(s),
           "ishru": lambda a, s: (_i(a).view(np.uint64) >> _i(s).astype(np.uint64)).view(np.int64),
           "iwrap": lambda a, b, s: (_i(a) << (64 - b)) >> (64 - b) if s else _i(a) & ((1 << b) - 1),
           "threefry": lambda a, b, c, d: _threefry(a, b, c, d),
           "icmp": lambda op, a, b, u: _NP_CMP[op](*((_i(a).view(np.uint64), _i(b).view(np.uint64)) if u else (_i(a), _i(b)))),
           "iselect": lambda p, f, t: np.where(p, _i(t), _i(f)), "b2f": lambda p: p.astype(np.float64),
           "and": np.logical_and, "or": np.logical_or, "not": np.logical_not,
           **_NP_UNARY, **_NP_CMP}


def _threefry(a, b, c, d):
    """dag.threefry: threefry2x32's two words packed into an int64 (special.threefry2x32 on uint32 arrays)."""
    from kernelax.special import threefry2x32
    u = lambda x: (_i(x) & 0xFFFFFFFF).astype(np.uint32)
    with np.errstate(over="ignore"):
        lo, hi = threefry2x32(u(a), u(b), u(c), u(d))
    return ((np.asarray(hi).astype(np.uint64) << np.uint64(32)) | np.asarray(lo).astype(np.uint64)).view(np.int64)


def _i(x):
    """int64 array of an integer value (Python ints as two's complement)."""
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        x = (int(x) + 2 ** 63) % 2 ** 64 - 2 ** 63
    return np.asarray(x, dtype=np.int64)


def evaluate(roots, input_values, iters=None):
    """Evaluate the DAG with numpy, vectorized over samples (reference / debugging).

    input_values[k] has shape (n, *leaf_shape); returns one (n,) array per root.
    Loops run masked over the samples; ``iters`` (dict) receives the iterations per sample
    of each loop node (of a nested loop: summed over the iterations of the enclosing loops).
    A nested loop runs only on the samples still active in the enclosing loop.
    """
    flat = [np.asarray(v).reshape(v.shape[0], -1) for v in input_values]
    n = flat[0].shape[0]
    val = {}  # loop bodies: a ChainMap per iteration over the values of the enclosing scopes

    def run(nodes, env, live):
        get = lambda a: env[a.id] if isinstance(a, Node) else a
        for node in nodes:
            if node.op == "in":
                env[node.id] = flat[node.args[0]][:, node.args[1]]
            elif node.op == "sample":
                env[node.id] = np.arange(n, dtype=np.float64)
            elif node.op == "load":
                f = flat[node.args[0]]
                i = np.clip(np.nan_to_num(np.broadcast_to(get(node.args[1]), (n,))), 0, f.shape[1] - 1).astype(np.int64)
                env[node.id] = f[np.arange(n), i]
            elif node.op == "arr":
                env[node.id] = np.stack([np.broadcast_to(get(a), (n,)) for a in node.args])
            elif node.op == "pick":
                a = env[node.args[0].id]
                i = np.clip(np.nan_to_num(np.broadcast_to(get(node.args[1]), (n,))), 0, len(a) - 1).astype(np.int64)
                env[node.id] = a[i, np.arange(n)]
            elif node.op == "while":
                env[node.id] = loop(node, env, live)
            elif node.op in ("loopout", "loopmask"):
                env[node.id] = env[node.args[0].id][node.args[1]]
            else:
                env[node.id] = _NP_OPS[node.op](*map(get, node.args))
            if node.t == "s":  # (float32 values: as float32 arrays, so that numpy computes in float32)
                env[node.id] = np.asarray(env[node.id], np.float32)

    def loop(L, env, act):
        (phs, outs, cond), inits = L.args[0], L.args[1:1 + len(L.args[0][0])]
        cur = [np.broadcast_to(env[x.id] if isinstance(x, Node) else x, (n,)) for x in inits]
        scope = ChainMap(dict.fromkeys(p.id for p in phs), env)
        head = schedule([cond], scope)  # as in emit_map: the condition first, the body on the active samples
        body = schedule(outs, ChainMap(dict.fromkeys(x.id for x in head), scope))
        it = np.zeros(n, dtype=np.int64)
        while True:
            loc = ChainMap({p.id: c for p, c in zip(phs, cur)}, env)
            run(head, loc, act)
            act = act & np.broadcast_to(loc[cond.id], (n,))
            if not act.any():
                break
            run(body, loc, act)
            cur = [np.where(act, loc[o.id] if isinstance(o, Node) else o, c) for o, c in zip(outs, cur)]
            it += act
        if iters is not None:
            iters[L.id] = iters.get(L.id, 0) + it
        return cur

    run(schedule(roots), val, np.ones(n, dtype=bool))
    out = [np.full(n, r, dtype=np.float64) if is_c(r) else val[r.id] for r in roots]
    val.clear()  # run and loop reference each other: free the values without waiting for the cycle GC
    return out
