# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Scalar DAG -> C: one kernel over a batch of samples, every scalar a SIMD vector of W samples."""

import functools
import hashlib
import os
import re
import zlib

import numpy as np

from kernelax import vec
from kernelax.build import PART_MARK
from kernelax.dag import _INT_OPS, Node, _isi, every, exact_products, is_c, schedule
from kernelax.dag import GUARD_DEPTH, guards as dag_guards
from kernelax.vec import _ROUND, ICMP, IPRED, MVEC, PRED, lit_d, lit_i, lit_s

_VT = {"d": "vd", "s": "vf", "i": "vi", "m": "vm"}  # C vector type of a node's type (Node.t)
_CT = {"d": "double", "f": "float", "i": "int32_t", "u": "uint32_t"}  # C type of an input's elements in memory (kinds)


def _lit(c, t="d"):
    return lit_d(c) if t == "d" else lit_s(c)


def _mcmp(op, a, b, R, kind):
    """C of a comparison of vectors a and b as a mask (vm); kind: of the operands, d / s (floats), i / u (int64,
    signed / unsigned). R: register bytes (vec)."""
    if kind in ("d", "s"):
        return f"cmp_{kind}({a}, {b}, {PRED[op]})"
    if R == 64:
        return f"cmp_{kind}({a}, {b}, {IPRED[op]})"
    cast = "(vu)" if kind == "u" else ""  # (AVX2: the vector comparison, -1/0 lanes)
    return f"({cast}{a} {ICMP[op]} {cast}{b})"
_IBIN = {"iadd": "+", "isub": "-", "imul": "*", "iand": "&", "ior": "|", "ixor": "^", "ishl": "<<", "ishr": ">>"}
_CONV = {"ftoi": "__builtin_convertvector({}, vi)", "fbits": "(vi){}", "bitsf64": "(vd){}", "tof32": "__builtin_convertvector({}, vf)",
         "bitsf32": "(vf)__builtin_convertvector({}, vi32)", "f32bits": "__builtin_convertvector((vi32){}, vi)",
         "tof64": "__builtin_convertvector({}, vd)"}  # conversions between the value types


def _c_int(node, name, R, ilit=lit_i):
    """C of the integer nodes (vi: int64 lanes; vu: their unsigned view) and the conversions; ilit: C of a constant."""
    op, args = node.op, node.args
    i = lambda x: name(x) if isinstance(x, Node) else ilit(x)
    v = lambda x: name(x) if isinstance(x, Node) else f"bm({ilit(x)})"  # as a vector
    if op in _IBIN:  # (a shift's first operand a vector: a constant one broadcast)
        return f"{v(args[0]) if op in ('ishl', 'ishr') else i(args[0])} {_IBIN[op]} {i(args[1])}"
    if op == "ishru":
        return f"(vi)((vu){v(args[0])} >> {'(vu)' if isinstance(args[1], Node) else ''}{i(args[1])})"
    if op == "iwrap":
        a, bits, signed = args
        return f"(({i(a)} << {64 - bits}) >> {64 - bits})" if signed else f"({i(a)} & {lit_i((1 << bits) - 1)})"
    if op == "threefry":  # (vec: threefry_v, in 32-bit lanes)
        return f"threefry_v({', '.join(v(x) for x in args)})"
    if op == "icmp":
        cop, a, b, u = args
        return _mcmp(cop, v(a), v(b), R, "u" if u else "i")
    if op == "iselect":
        return f"misel({name(args[0])}, {v(args[1])}, {v(args[2])})"
    if op == "b2i":
        return f"mb2i({name(args[0])})"
    if op == "itof":
        return f"__builtin_convertvector({'(vu)' if args[1] else ''}{name(args[0])}, vd)"
    if op in _CONV:
        return _CONV[op].format(i(args[0]))
    raise NotImplementedError(f"kernelax: '{op}' is not supported in C kernels")


def _c_expr(node, name, R, lit=_lit, ilit=lit_i):
    """C of a node in the helpers of its type (vec); lit(c, t) / ilit(c): C of its constant operands (float of type
    t / int64). R: register bytes."""
    op = node.op
    if op in _INT_OPS or op in _CONV or op in ("itof", "icmp"):
        return _c_int(node, name, R, ilit)
    t = node.t if op not in ICMP else next(x.t for x in node.args if isinstance(x, Node))  # (compares: the operands')
    a = [name(x) if isinstance(x, Node) else lit(x, t) for x in node.args]
    b = [name(x) if isinstance(x, Node) else f"bc_{t}({lit(x, t)})" for x in node.args]  # broadcast for calls
    if op in ("add", "sub", "mul"):
        return f"{a[0]} {dict(add='+', sub='-', mul='*')[op]} {a[1]}"
    if op == "neg":
        return f"-{a[0]}"
    if op == "recip":
        return f"{lit(1.0, t)} / {a[0]}"
    if op in ICMP:
        return _mcmp(op, b[0], b[1], R, t)
    if op in ("max", "min"):  # (NaN as XLA: the intrinsic gives its second operand where one is NaN)
        x, y = node.args
        if not isinstance(x, Node) or not isinstance(y, Node):  # a constant first: free
            c, v = (x, y) if not isinstance(x, Node) else (y, x)
            return f"{op}_{t}(bc_{t}({lit(c, t)}), {name(v)})"
        return f"sel_{t}(cmp_{t}({a[0]}, {a[0]}, _CMP_UNORD_Q), {op}_{t}({a[0]}, {a[1]}), {a[0]})"
    if op in _ROUND:
        return f"round_{t}({b[0]}, {_ROUND[op]})"
    if op in ("sqrt", "abs", "sign") or op in MVEC:
        return f"{op}_{t}({b[0]})"
    if op in ("pow", "atan2", "fmod"):
        return f"{op}_{t}({b[0]}, {b[1]})"
    if op == "select":
        return f"sel_{t}({a[0]}, {b[1]}, {b[2]})"
    if op == "b2f":
        return f"b2f_d({a[0]})"
    if op == "and":
        return f"(vm)({a[0]} & {a[1]})"
    if op == "or":
        return f"(vm)({a[0]} | {a[1]})"
    if op == "not":
        return f"MNOT({a[0]})"
    raise NotImplementedError(f"kernelax: '{op}' is not supported in C kernels")


_SCATTER = """\
/* the values v[] and positions i[] of e entries streamed into the buffers of a sum through buffers (driver) */
static __attribute__((noinline)) void bput(double* restrict bv, double* restrict bi, const vd* v, const vd* i, int e) {
  for (int j = 0; j < e; ++j) { STREAM_d(bv + j * W, v[j]); if (bi) STREAM_d(bi + j * W, i[j]); }
}
/* v into s[i] lane by lane, for the lanes l0 + l < m (not padding) and 0 <= i < n (else dropped); r: the range of
   positions touched (lowest, highest) */
static inline void scat(double* restrict s, int64_t n, vd i, vd v, int64_t l0, int64_t m, int64_t* restrict r) {
  const vi k = __builtin_convertvector(i, vi);
  const int c = m - l0 < W ? (int)(m - l0) : W;
  int64_t lo = r[0], hi = r[1];
  for (int l = 0; l < c; ++l) {
    const int64_t q = k[l];
    if ((uint64_t)q < (uint64_t)n) {
      s[q] += v[l];
      lo = q < lo ? q : lo;
      hi = q > hi ? q : hi;
    }
  }
  r[0] = lo;
  r[1] = hi;
}
/* scat for the e entries (positions i[], values v[]) of a chunk: one call instead of e inlined loops */
static __attribute__((noinline)) void scatn(double* restrict s, int64_t n, int64_t l0, int64_t m, int64_t* restrict r,
                                            const vd* i, const vd* v, int e) {
  for (int j = 0; j < e; ++j) scat(s, n, i[j], v[j], l0, m, r);
}
"""


def _transpose(W, size):
    """tr<size>(r): the W x W block r[row][lane] of memory vectors of `size`-byte elements transposed in registers.
    log2(W) stages; the stage of bit h swaps the entries whose row and lane differ in bit h (two two-input shuffles
    per pair of rows)."""
    v, ive = ("vd", "vi") if size == 8 else ("vf", "vi32")
    out, h = [f"static inline __attribute__((always_inline)) void tr{size}({v}* r) {{"], W // 2
    while h:
        for i in (i for i in range(W) if not i & h):
            lo = ", ".join(str(p + W - h if p & h else p) for p in range(W))
            hi = ", ".join(str(p + W if p & h else p + h) for p in range(W))
            out.append(f"  {{ const {v} a = r[{i}], b = r[{i + h}]; r[{i}] = __builtin_shuffle(a, b, ({ive}){{{lo}}}); "
                       f"r[{i + h}] = __builtin_shuffle(a, b, ({ive}){{{hi}}}); }}")
        h //= 2
    return "\n".join(out + ["}"]) + "\n"


def _lanes(lanes, src, W, ive="ive"):
    """C expression of the memory vector whose lane s is lane l of the vector src(v), for lanes[s] = (v, l): one
    two-input shuffle per source vector after the first (none for a whole source vector); ive: the lane indices' type."""
    order = list(dict.fromkeys(v for v, _ in lanes))
    mask = lambda f: f"({ive}){{" + ", ".join(str(f(s, v, l)) for s, (v, l) in enumerate(lanes)) + "}"
    if len(order) == 1:
        same = all(l == s for s, (_, l) in enumerate(lanes))
        return src(order[0]) if same else f"__builtin_shuffle({src(order[0])}, {mask(lambda s, v, l: l)})"
    a, b = order[:2]
    e = f"__builtin_shuffle({src(a)}, {src(b)}, {mask(lambda s, v, l: l if v == a else W + l if v == b else 0)})"
    for c in order[2:]:
        e = f"__builtin_shuffle({e}, {src(c)}, {mask(lambda s, v, l: W + l if v == c else s)})"
    return e


@functools.cache
def l3_per_core():
    """Bytes of L3 cache per core of the first CPU this process may use (sysfs); 0 where unknown (then the outputs
    always go through the non-temporal stores)."""
    d = f"/sys/devices/system/cpu/cpu{min(os.sched_getaffinity(0))}/cache/index3/"
    try:
        with open(d + "size") as f:
            size = f.read().strip()
        with open(d + "shared_cpu_list") as f:
            cpus = sum(int(b) - int(a) + 1 for a, _, b in (r.partition("-") if "-" in r else (r, "-", r)
                                                           for r in f.read().strip().split(",")))
    except OSError:
        return 0
    return int(size[:-1]) * {"K": 1 << 10, "M": 1 << 20}[size[-1]] // cpus if size[-1] in "KM" else int(size) // cpus


def _contiguous(R, W):
    """Rows of R entries moved as the block's R contiguous vectors and R (R - 1) shuffles at most,
    instead of W x W transpositions (W log2 W shuffles per W entries)."""
    return R * (R - 1) <= W * (W.bit_length() - 1)


def prelude(W, R, E):
    """The value types and their helpers (vec), the scatter helpers, tr."""
    return vec.prelude(W, R, E) + _SCATTER + "".join(_transpose(W, size) for size in (8, 4) if size * W <= R)


SCAT_CHUNK = 256  # entries of a scatter output per call of scatn
SCAT_BIG = 1 << 16  # scatter outputs of this size and more: through buffers and a cached plan (see the driver)
WIDE_MIN = 1 << 32  # entries of such a sum per call from which its plan holds 64-bit entry numbers (else 32-bit: less
# to read in the sums; a test sets 0)
TILE_BUFFER = 1 << 22  # bytes of a tiled sum loop's buffer of shared values per thread at most (_tiles)
TILES = (16, 32, 64, 81, 128)  # tile sizes tried for loops of accumulators (_tiles)


def _tiles(L, outside):
    """(counters, Q, tiles, shared) for a loop over counters (from 0 by 1 to Q, the same in all lanes) whose other
    carries only accumulate (carry + a value that reads none of them), in tiles of contiguous accumulators (the size
    of the least code: shared values once, each tile's own); shared: the values of several tiles the tiles read.
    None for other loops (or one tile)."""
    (phs, outs, cond), inits = L.args[0], L.args[1:1 + len(L.args[0][0])]
    step = lambda j: not is_c(outs[j]) and outs[j].op == "add" and phs[j] in outs[j].args and 1.0 in outs[j].args
    counters = [j for j in range(len(phs)) if step(j) and is_c(inits[j]) and inits[j] == 0]
    if (not counters or is_c(cond) or cond.op != "lt" or cond.args[0] not in [phs[j] for j in counters]
            or not is_c(cond.args[1]) or any(p.op != "carry" for p in phs)):
        return None
    acc = [j for j in range(len(phs)) if j not in counters]
    if len(acc) < 2 * TILES[0] or not all(not is_c(outs[j]) and outs[j].op == "add" and phs[j] in outs[j].args for j in acc):
        return None
    contrib = {j: next(a for a in outs[j].args if a is not phs[j]) for j in acc}
    done = set(outside) | {phs[j].id for j in counters}
    union = schedule([c for c in contrib.values() if isinstance(c, Node)], done)  # (args before their users)
    ids = {n.id for n in union}
    if any(phs[j].id in ids for j in acc):
        return None  # (an accumulator read by a value: not a sum)
    size = TILES[0]  # (the code is the union of the tiles' values whatever their size: the smallest tiles)
    tiles = [acc[i:i + size] for i in range(0, len(acc), size)]
    lo, hi = {}, {}  # per value the first and the last tile that reads it: shared if they differ
    for t, tile in enumerate(tiles):
        for j in tile:
            if isinstance(contrib[j], Node):
                lo[contrib[j].id], hi[contrib[j].id] = min(lo.get(contrib[j].id, t), t), max(hi.get(contrib[j].id, t), t)
    for n in reversed(union):  # (users before their args)
        for a in n.args:
            if isinstance(a, Node) and a.id in ids and n.id in lo:
                lo[a.id], hi[a.id] = min(lo.get(a.id, lo[n.id]), lo[n.id]), max(hi.get(a.id, hi[n.id]), hi[n.id])
    shared = {i for i in lo if lo[i] != hi[i]}
    front = {}  # the shared values that tile code reads (computed once per step, into the buffer)
    for n in union:
        if n.id not in shared:
            front.update((a.id, a) for a in n.args if isinstance(a, Node) and a.id in shared)
    for tile in tiles:
        front.update((contrib[j].id, contrib[j]) for j in tile if isinstance(contrib[j], Node) and contrib[j].id in shared)
    best = (None, tiles, list(front.values()))
    Q = float(cond.args[1])  # (counter < Q from 0 by 1: ceil(Q) steps)
    return (counters, int(np.ceil(Q)), best[1], best[2]) if np.isfinite(Q) and Q > 0 else None


def _sources(roots):
    """The inputs (indices) the values of `roots` depend on, through loops too (dag.every: an input read in a loop's
    body only, e.g. a table a binary search reads, is no argument of the loop node; missing it would keep a scatter
    plan made for other values of it)."""
    return sorted({n.args[0] for n in every(roots) if n.op in ("in", "load")})


_DATA = frozenset("in load table pick arr while loopout loopmask carry carrymask sample".split())
_CMP = frozenset("lt le gt ge eq ne icmp".split())


def _address(outputs):
    """Ids of the nodes that compute positions (of the loads from inputs in memory and of the scatter entries), back
    to data: their constants are passed at run time (emit_map)."""
    scat = [a for a in outputs if hasattr(a, "entries")]
    nodes = every([x for a in outputs for x in ([y for e in a.entries for y in e] if hasattr(a, "entries") else
                                                np.asarray(a, dtype=object).ravel())])
    stack = [n.args[1] for n in nodes if n.op == "load"] + [f for a in scat for f, _ in a.entries if isinstance(f, Node)]
    addr = set()
    while stack:
        n = stack.pop()
        if n.id not in addr and n.op not in _DATA:
            addr.add(n.id)
            stack += [a for a in n.args if isinstance(a, Node)]
    return addr | {n.id for n in nodes if n.op in _CMP and any(isinstance(a, Node) and a.id in addr for a in n.args)}  # (and
    # the comparisons of positions: bounds checks)


SPLIT_MIN = 1 << 20  # bytes of a block's C (K = 1) from which it is split into
PART = 1 << 19  # parts of about this many bytes (_parts)
FRAME_MAX = 0.035  # at most this many reads from the frame per statement
# (unless gcc ran out of time for the block as one file: kernel._build)
_SIG = ("const void* const* restrict in, elem* const* restrict out, vd* restrict acc, double* restrict sacc, "
        "int64_t* restrict srng, void* const* restrict big, int64_t s0, int64_t m")
_VOID = "  (void)in; (void)out; (void)acc; (void)sacc; (void)srng; (void)big; (void)s0; (void)m;"
_SLOT = "typedef union { vd d; vf f; vi i; vm m; ve e; } slot;  /* a value between the parts of a kernel */\n"
_MEM = {"vd": "d", "vf": "f", "vi": "i", "vm": "m", "ve": "e"}
_NAME = r"\b(?:(?:t|A|act|q)\d+|(?:c|u|G|V|D|S)\d+_\d+|keep)_#|\b(?:SB|q)\d+\b"  # the names emit_map gives values
_DECL = re.compile(rf"^  (const )?(static _Thread_local )?(vd|vf|vi|vm|ve)(\*?) ({_NAME})(?:\[([^\]]+)\])? *[=;]")
_WRITE = re.compile(rf"(?<![\w.])({_NAME})\s*(?:\[[^\]]*\])?\s*[-+*/|&^]?=(?!=)|\btr\d*\(({_NAME})\)")


def _parts(lines, K, W, cheap, tables, nprm, frame_max=True):
    """The block's statements (before their copies per sub-block, so that every K gets the same parts and run-time
    values) as functions in parts of about PART bytes, each its own C file (PART_MARK): (the block's code calling
    them, the parts' sources, their run-time values (appended to PRM), the driver's declarations), or None if there
    would be one part. A part's text depends on its own statements only: values from other parts come through a
    per-thread frame of slots at positions it reads from a table (OI), its names, run-time values and tables are
    numbered in the order of their first use (a table by its content); cuts fall where the statements' content says
    (not at fixed sizes), so that a change of the function leaves the other parts' objects in the cache. Values that
    only read memory (loads) are recomputed in the parts that need them instead of passed. A statement that writes
    a value declared before (carries of loops, transposed groups, buffers) stays in the part of its declaration.
    frame_max: None also where the parts would read the frame often (FRAME_MAX)."""
    items, i = [], 0  # top-level statements (a compound one with the _Pragma before it)
    while i < len(lines):
        j, depth = i, 0
        while True:
            depth += lines[j].count("{") - lines[j].count("}")
            if depth <= 0 and not lines[j].lstrip().startswith("_Pragma"):
                break
            j += 1
        items.append(lines[i:j + 1])
        i = j + 1
    if sum(len(l) + 1 for l in lines) < SPLIT_MIN:
        return None
    name = re.compile(_NAME)
    decl, refs, last = {}, [], {}  # name -> (item, type, length, const); per item its names; name -> its last writer
    for k, it in enumerate(items):
        text = "\n".join(it)
        inner = {m.group(5) for l in it[1:] for m in [_DECL.match(l.lstrip(" ").rjust(len(l.lstrip(" ")) + 2))] if m}
        inner |= set(re.findall(r"for \(int (q\d+) =", text))
        refs.append(set(name.findall(text)) - inner)  # (values and counters of a loop's body: its own)
        m = _DECL.match(it[0])
        if m:
            decl[m.group(5)] = (k, m.group(3) + m.group(4), m.group(6), bool(m.group(1)) and not m.group(2))
        for w in _WRITE.finditer(text):
            last[w.group(1) or w.group(2)] = k
    used = {}  # name -> the last item that reads it
    for k, r in enumerate(refs):
        used.update(dict.fromkeys(r, k))
    reach = list(range(len(items)))  # the last item of the unit an item starts
    for x, (k, t, n, const) in decl.items():
        if not const:  # (a static buffer or a group of stores: with all that reads it; else up to its last write)
            reach[k] = max(reach[k], used.get(x, k) if t.endswith("*") or x[0] == "S" else last.get(x, k))
    units, k = [], 0
    while k < len(items):
        end, j = reach[k], k
        while j <= end:
            end, j = max(end, reach[j]), j + 1
        units.append(list(range(k, end + 1)))
        k = end + 1
    at = {}
    for x, d in decl.items():
        at.setdefault(d[0], set()).add(x)
    defs = [set().union(*(at.get(k, ()) for k in u)) for u in units]
    uses = [set().union(*(refs[k] for k in u)) for u in units]
    texts = ["\n".join(l for k in u for l in items[k]) for u in units]
    size = [len(t) + 1 for t in texts]
    if sum(size) < SPLIT_MIN:
        return None
    owner = {x: u for u, d in enumerate(defs) for x in d}
    remat = [bool(d) and all(x in cheap for x in d) and "for (" not in texts[u] for u, d in enumerate(defs)]
    width = lambda x: 1 if decl[x][2] is None else W if decl[x][2] == "W" else int(decl[x][2])  # (array entries)
    live = np.zeros(len(units) + 1)  # live[u]: the values a cut after unit u passes through the frame (loads: none)
    lastuse = {}
    for u, us in enumerate(uses):
        lastuse.update(dict.fromkeys(us, u))
    for x, u in owner.items():
        if not remat[u] and lastuse.get(x, u) > u:
            live[u] += width(x)
            live[lastuse[x]] -= width(x)
    live = np.cumsum(live)
    norm = lambda t: re.sub(r"\d+", "", re.sub(_NAME, "x", t))  # (a statement's shape: the same after renumbering)
    D = max(1, round(PART / 8 / (sum(size) / len(units))))  # (anchors: about 8 per part)
    anchor = [zlib.crc32(norm(t).encode()) % D == 0 for t in texts]
    cum = np.cumsum(size)
    parts, s0 = [], 0
    while cum[-1] - (cum[s0 - 1] if s0 else 0) > 3 * PART // 2:  # cuts: the anchor with the fewest values through the
        base = cum[s0 - 1] if s0 else 0  # frame between PART / 2 and 2 PART after the last cut (else any boundary there)
        window = [u for u in range(s0, len(units) - 1) if PART // 2 <= cum[u] - base <= 2 * PART] or \
            [u for u in range(s0, len(units) - 1) if cum[u] - base >= PART // 2][:1]  # (after a unit larger than that)
        if not window:
            break
        cands = [u for u in window if anchor[u]] or window
        u = min(cands, key=lambda u: (live[u], u))
        parts.append(list(range(s0, u + 1)))
        s0 = u + 1
    parts.append(list(range(s0, len(units))))
    parts = [[u for u in us if not remat[u]] for us in parts]  # (loads: in the parts that read them, _again_)
    parts = [us for us in parts if us]
    if len(parts) < 2:
        return None
    where = {u: p for p, us in enumerate(parts) for u in us}
    need = []  # per part: the loads it recomputes, the values it reads from the frame
    for p, us in enumerate(parts):
        mine = set().union(*(defs[u] for u in us))
        again, frame, stack = set(), set(), list({x for u in us for x in uses[u]} - mine)
        while stack:
            x = stack.pop()
            u = owner[x]
            if remat[u]:
                if u not in again:
                    again.add(u)
                    stack += list(uses[u] - defs[u])
            else:
                assert where[u] < p, f"kernelax: {x} used before its part"
                frame.add(x)
        need.append((sorted(again), frame))
    outs = [[] for _ in parts]
    for p, (_, frame) in enumerate(need):
        for x in frame:
            if x not in outs[where[owner[x]]]:
                outs[where[owner[x]]].append(x)
    if frame_max and sum(width(x) for _, frame in need for x in frame) > FRAME_MAX * len(items):  # (many reads from the
        return None  # frame, as from a forward pass to its reverse pass: more time in the kernel than the parts save in
    # gcc)
    slot, nslot = {}, 0  # name -> its first slot: the parts' outputs one after the other, K copies each
    for o in outs:
        o.sort(key=lambda x: decl[x][0])
        for x in o:
            slot[x], nslot = nslot, nslot + K * width(x)
    copies = lambda line: [line.replace("#", str(b)) for b in range(K)] if "#" in line else [line]
    tab = {t: f"TAB_{hashlib.sha256(np.asarray(v, float).tobytes()).hexdigest()[:16]}" for v, t in tables.items()}
    srcs, calls, extra, tabdefs, protos, ois = {}, [], [], {}, [], []
    for p, us in enumerate(parts):
        again, frame = need[p]
        order, placed = [], set()  # the loads just before their first use (as emit_map placed them)

        def place(u):
            for x in sorted(uses[u] - defs[u], key=lambda x: (owner[x], x)):
                v = owner[x]
                if remat[v] and v in again and v not in placed:
                    placed.add(v)
                    place(v)
                    order.append(v)
        for u in us:
            place(u)
            order.append(u)
        # values from the frame: read just before the first unit that uses them, written right after the unit that
        # defines them (not all at the head / the end of a part: thousands of them live at once, spilled again)
        mine_out = {x: K * sum(width(y) for y in outs[p][:i]) for i, x in enumerate(outs[p])}
        body, ins, got = [], [], set()
        for u in order:
            for x in sorted((uses[u] - defs[u]) & frame - got, key=lambda x: (owner[x], x)):
                got.add(x)
                i, t, n = len(ins), decl[x][1], width(x)
                ins.append(x)
                if t not in _MEM:
                    raise NotImplementedError(f"kernelax: a value of type {t} between the parts of a large kernel is "
                                              "not supported")
                body.append(f"  const {t} {x} = FR[OI[{i}] + #].{_MEM[t]};" if decl[x][2] is None else
                            f"  {t} {x}[{n}]; for (int e_ = 0; e_ < {n}; ++e_) {x}[e_] = FR[OI[{i}] + # * {n} + e_].{_MEM[t]};")
            body += [l for k in units[u] for l in items[k]]
            for x in sorted(defs[u] & mine_out.keys(), key=lambda x: mine_out[x]):
                t, n, off = decl[x][1], width(x), mine_out[x]
                body.append(f"  FO[{off} + #].{_MEM[t]} = {x};" if decl[x][2] is None else
                            f"  for (int e_ = 0; e_ < {n}; ++e_) FO[{off} + # * {n} + e_].{_MEM[t]} = {x}[e_];")
        pre, post = [], []
        code = "\n".join(c for line in pre + body + post for c in copies(line))
        names, prms, tabs = {}, {}, {}
        code = re.sub(_NAME.replace("_#", r"_\d+"), lambda m: names.setdefault(m.group(0), f"z{len(names)}"), code)
        code = re.sub(r"PRM\[(\d+)\]", lambda m: f"PRM[{prms.setdefault(int(m.group(1)), len(prms))}]", code)
        code = re.sub(r"\bTAB(\d+)\b", lambda m: tabs.setdefault(tab[int(m.group(1))], tab[int(m.group(1))]), code)
        local = []  # the pointers to the arrays as local copies (as in the driver, whose block is inlined: the stores
        for arr, t in (("in", "const void*"), ("out", "elem*"), ("big", "void*")):  # into the arrays do not change them)
            n = max((int(i) + 1 for i in re.findall(rf"\b{arr}\[(\d+)\]", code)), default=0)
            if n:
                local.append(f"  {t} const {arr}[{n}] = {{{', '.join(f'{arr}_[{i}]' for i in range(n))}}};")
        code = "\n".join(local + [code])
        fn = "p_" + hashlib.sha256(code.encode()).hexdigest()[:16]
        sig = (f"void {fn}({_SIG.replace(' in,', ' in_,').replace(' out,', ' out_,').replace(' big,', ' big_,')}, "
               "const double* restrict PRM, slot* restrict FR, const int* restrict OI, slot* restrict FO)")
        lens = {tab[t]: len(vec.table_pad(v)) for v, t in tables.items()}
        srcs[fn] = ("".join(f"extern const double {t}[{lens[t]}];\n" for t in tabs) + sig + " {\n"
                    + _VOID.replace("(void)in;", "(void)in_;").replace("(void)out;", "(void)out_;").replace("(void)big;", "(void)big_;")
                    + f" (void)PRM; (void)FR; (void)OI; (void)FO;\n{code}\n}}\n")
        tabdefs.update((tab[t], v) for v, t in tables.items() if tab[t] in tabs)
        protos.append(f"void {fn}({_SIG}, const double* restrict PRM, slot* restrict FR, const int* restrict OI, slot* restrict FO);")
        ois.append(f"static const int OI_{p}[] = {{{', '.join(str(slot[x]) for x in ins) or '0'}}};")
        calls.append(f"  {fn}(in, out, acc, sacc, srng, big, s0, m, PRM + {nprm + len(extra)}, FR_, OI_{p}, FR_ + {slot[outs[p][0]] if outs[p] else 0});")
        extra += [g for g in sorted(prms, key=prms.get)]
    block = ["  static _Thread_local slot* FR_ = NULL;  /* (per thread: the values between the parts) */",
             f"  if (!FR_) FR_ = aligned_alloc(sizeof(slot), (size_t){max(nslot, 1)} * sizeof(slot));"] + calls
    decls = (_SLOT + "".join(f"const double {t}[{len(vec.table_pad(v))}] = {{{', '.join(map(_lit, vec.table_pad(v)))}}};\n"
                             for t, v in tabdefs.items())
             + "\n".join(dict.fromkeys(protos)) + "\n" + "\n".join(ois) + "\n")
    return block, list(srcs.values()), extra, decls


def emit_map(inputs, outputs, batched, W, K=1, reduce=None, mem=None, frame_max=True, split=True, static=False, group=1):
    """(C source of ``void kernel(int nt, int64_t n, void** p)``, its run-time values): the outputs of a traced
    function for n samples.

    inputs:  per input leaf, the object array of its input nodes (frontend.trace); batched[k]: leaf k has a
             leading sample axis (array (n, size), row-major), else it is the same for all samples (size,).
    outputs: per output leaf, an object array; written as arrays (n, size), or summed over the samples into
             (size,) if reduce[j]: every thread adds the block's values lane by lane into its own accumulators
             (the padding lanes of the last block masked out), the driver sums lanes and threads in a fixed
             order (deterministic for a given thread count and K).
    p:       the input arrays, the run-time values (PRM: constants of the address arithmetic, sizes of the inputs in
             memory and of the scatter outputs; the source depends on their number, not their values), the outputs.
    W:       SIMD width in doubles (8: AVX-512, 4: AVX2): registers of R = 8 W bytes.
    mem:     (E, kinds of the inputs, kinds of the outputs): the elements of the arrays in memory
             (primitive.to_kernel). Inputs: d (double; 64-bit integers by their bits), f (float32), i / u (int32 /
             uint32). Outputs: doubles (E = 8: integers as exact doubles, 64-bit ones by their bits) or 4 bytes
             (E = 4; kind f: float32, i / u: the bits of int32 / uint32). A kernel runs R / E samples per vector (W in
             the code below and in C). Default: doubles.
    split: a large block may be split into parts compiled apart (_parts; False: one file, export); static: the driver
    `static` (export: one wrapper per file is visible), and the caller's output buffers left as they are (no madvise).
    Samples run in blocks of K W: K interleaved sub-blocks of W samples, one SIMD lane per sample. Every
    statement is emitted once per sub-block, the copies next to each other (names carry the sub-block
    in place of '#'), so that K independent dependency chains hide the latency of each other. A sub-block
    reads and writes its rows as vectors: short rows (_contiguous) as its contiguous vectors, deinterleaved
    by shuffles, longer ones W entries at a time, transposed in registers (tr). The blocks are spread over
    nt OpenMP threads; the last n % (K W) samples run in a block padded with copies of the last sample.
    While loops run as masked SIMD loops: lanes whose condition fails keep their carries, the loop ends
    when no lane of the block is active.
    """
    E, kin, kout = mem or (8, ["d"] * len(inputs), ["f"] * len(outputs))
    RB = 8 * W  # register bytes
    W = RB // E  # lanes
    names, acts, tables, uni, top, whiles = {}, [], {}, set(), [], []  # uni: values the same in all lanes; top: the block
    # level's lines and done (value); whiles: the data-dependent loops
    cheap = {"keep_#"}  # values that only read memory (loads; the samples' indices): recomputed where a part needs them
    sizes = [int(np.size(a)) for a in inputs]
    roots = [x for a in outputs for x in ([y for e in a.entries for y in e] if hasattr(a, "entries") else
                                          np.asarray(a, dtype=object).ravel()) if isinstance(x, Node)]
    rounded = exact_products(every(roots))  # (products error-free transformations need rounded: not contracted, dag)
    guards, opened, refused, inside = dag_guards(roots), set(), set(), []  # (work only one side of selects needs: in a
    # block where a lane takes it; refused: (work, scope) where not; inside: the work of the open blocks)
    prm, addr = {}, _address(outputs)  # values passed at run time (PRM, an input after the others): the constants of
    # the address arithmetic and the sizes of the inputs in memory and of the scatter outputs, so that the code does not
    # depend on the sizes of the arrays it reads from and sums into (another mesh, the same elements: the same source,
    # its compiled library from the cache); slots by where the value is used (not by value: equal by chance elsewhere)
    P = lambda v, at: f"PRM[{prm.setdefault(at + (float(v),), len(prm))}]"
    pos = lambda x, at, lines, done: f"bc_d({P(x, at)})" if is_c(x) else value(x, lines, done)  # (a scatter entry's position)
    vtype = lambda n: _VT[n.t]
    name = lambda n: names[n.id]

    msz = lambda k: 8 if kin[k] == "d" else 4  # bytes of input k's elements in memory
    mptr = lambda k: f"((const {_CT[kin[k]]}*)in[{k}])"  # its pointer (elements)
    vptr = lambda k: f"((const {'double' if kin[k] == 'd' else 'float'}*)in[{k}])"  # (vectors: 4-byte ones as floats)

    def fromem(x, k, t, scalar=None):
        """C of the value (type t) of input k's entries x (a memory vector; or element `scalar` of it)."""
        if scalar is not None:
            return f"bc_{t if kin[k] in 'df' else 'd'}({mptr(k)}[{scalar}])"
        if kin[k] in "iu":
            x = f"(v{kin[k]}32)({x})"  # (the bits of int32 / uint32)
        elif t == ("d" if kin[k] == "d" else "s"):
            return x
        return f"__builtin_convertvector({x}, {_VT[t]})"

    def todbl(x, v):
        """C of the value v of x as doubles: masks as 0/1, integers as doubles."""
        if not isinstance(x, Node) or x.t == "d":
            return v
        return f"b2f_d({v})" if x.t == "m" else f"__builtin_convertvector({v}, vd)"

    def tomem(x, v, j):
        """C of the value v of x (an entry of output j) as a memory vector (ve)."""
        if E == 8:
            return todbl(x, v)
        if kout[j] == "f":
            return f"__builtin_convertvector({todbl(x, v)}, vf)" if not isinstance(x, Node) or x.t != "s" else v
        i = v if _isi(x) else f"mb2i({v})" if isinstance(x, Node) and x.t == "m" else f"__builtin_convertvector({todbl(x, v)}, vi)"
        return f"(vf)__builtin_convertvector({i}, vi32)"  # (the int32 / uint32 bits)

    def value(x, lines, done, t="d"):
        """Name of x after emitting what it needs and is not `done` into `lines` (a constant: as a vector of type t).
        Inputs are never carry-dependent, so their loads land at block level (before the loops that read them)."""
        if is_c(x):
            return f"bc_{t}({_lit(x, t)})"
        for n in schedule([x], done):
            if n.id in done:  # (emitted meanwhile, in a guarded block)
                continue
            if len(inside) < GUARD_DEPTH and any(guard(g, lines, done) for g in sorted(  # (the work of a select's side
                    (g for g in guards.get(n.id, ()) if g[3] not in opened and (g[3], id(done)) not in refused),
                    key=lambda g: -len(g[3]))):  # alone: the outermost group first)
                continue
            done.add(n.id)
            same = lambda y: is_c(y) or y.id in uni  # (the same in all lanes: constants, parameters, counters, and
            if n.op in ("load", "table", "pick") and same(n.args[1]) and (n.op != "pick" or same(n.args[0])) or \
                    n.op not in ("in", "sample", "while", "loopout", "loopmask", "carry", "carrymask", "load", "table", "pick") \
                    and all(same(y) for y in n.args if isinstance(y, Node)):  # what is computed from them alone)
                uni.add(n.id)
            if n.op == "in":  # (at block level, before the loops that read it, also when a loop's body needs it first)
                if ("in", n.id) not in top[1]:  # (its own key: at block level, done holds n.id already)
                    top[1].update((("in", n.id), n.id))
                    load(n, *top)
            elif n.op == "while":
                loop(n, lines, done)
            elif n.op in ("loopout", "loopmask"):
                names[n.id] = f"c{n.args[0].id}_{n.args[1]}_#"
            elif n.op == "sample":  # the samples' indices in the batch, per lane
                names[n.id] = f"t{n.id}_#"
                cheap.add(names[n.id])
                lines.append(f"  const vd t{n.id}_# = __builtin_convertvector(bm(s0) + {lanes}, vd);")
            elif n.op == "load":  # an entry of an input in memory, per lane
                names[n.id] = f"t{n.id}_#"
                g = f"lookup{'1' if same(n.args[1]) else ''}_{'d' if kin[n.args[0]] == 'd' else 's'}({vptr(n.args[0])}, {name(n.args[1])}, (long long){P(sizes[n.args[0]], ('in', n.args[0]))})"
                lines.append(f"  const {vtype(n)} t{n.id}_# = {fromem(g, n.args[0], n.t)};")
            elif n.op == "arr":  # options of a pick, as an array of vectors
                names[n.id] = f"A{n.id}_#"
                lines.append(f"  const {vtype(n)} A{n.id}_#[{len(n.args)}] = {{{', '.join(value(x, lines, done, n.t) for x in n.args)}}};")
            elif n.op == "pick":
                names[n.id] = f"t{n.id}_#"
                lines.append(f"  const {vtype(n)} t{n.id}_# = {'pick1' if same(n.args[1]) else 'pickv'}_{n.t}({name(n.args[0])}, {name(n.args[1])}, {len(n.args[0].args)});")
            elif n.op == "table":
                t = tables.setdefault(n.args[0], len(tables))
                names[n.id] = f"t{n.id}_#"
                f = "lookup1_d" if same(n.args[1]) else "tperm_d" if RB == 64 and len(n.args[0]) <= vec.TABLE_PERM else "lookup_d"
                lines.append(f"  const vd t{n.id}_# = {f}(TAB{t}, {name(n.args[1])}, {len(n.args[0])});")
            else:
                names[n.id] = f"t{n.id}_#"
                lit = lambda c, t, n=n: P(c, (n.id,)) if t == "d" else f"((float){P(c, (n.id,))})"
                ilit = lambda c, n=n: f"((long long){P(c, (n.id,))})" if abs(int(c)) < 2 ** 53 else lit_i(c)
                e = _c_expr(n, name, RB, lit, ilit) if n.id in addr else _c_expr(n, name, RB)
                e = f"__builtin_assoc_barrier({e})" if n.id in rounded else e
                lines.append(f"  const {vtype(n)} t{n.id}_# = {e};")
        return None if x.op == "while" else names[x.id]  # (a loop: named are its outputs, emitted before a guarded block)

    def guard(grp, lines, done):
        """The work of a group of selects' side (dag.guards) in a block run only where a lane of the block (of the
        enclosing loop's active ones) takes that side; its results into values declared before (zero where none did:
        the selects take the other side there)."""
        p, side, outs, work = grp
        pre = [n for n in schedule([p, *outs], done) if n.id not in work]  # (what the work reads from outside it: before
        if any(n.op in ("carry", "carrymask") for n in pre):  # the block, where later users see it; a carry not defined
            refused.add((work, id(done)))  # here: selects of another loop's body that shares these placeholders, as
            return False  # a loop and its copy with fewer carries do)
        opened.add(work)
        m = value(p, lines, done)
        for n in pre:
            if n.id not in done:
                value(n, lines, done)
        terms = [(m if side == 2 else f"MNOT({m})").replace("#", str(j)) for j in range(K)]
        terms = [f"({t} & {acts[-1].replace('#', str(j))})" for j, t in enumerate(terms)] if acts else terms
        lines += [f"  {vtype(o)} g{o.id}_# = {{}};" for o in outs]
        inner = []
        inside.append(work)
        for o in outs:
            inner.append(f"  g{o.id}_# = {value(o, inner, done)};")
        inside.pop()
        lines += [f"  if (many((vm)({' | '.join(terms)}))) {{"] + ["  " + line for line in inner] + ["  }"]
        for o in outs:
            names[o.id] = f"g{o.id}_#"
        return True

    def load(n, lines, done):  # (lines, done: the block level's)
        k, i = n.args
        names[n.id] = f"t{n.id}_#"
        cheap.add(names[n.id])
        if not batched[k]:
            lines.append(f"  const {vtype(n)} t{n.id}_# = {fromem(None, k, n.t, i)};")
            return
        g, R, z = i // W, sizes[k], msz(k)
        mv, ive = ("vd", "vi") if z == 8 else ("vf", "vi32")  # (memory vectors of input k)
        row = lambda off: f"{vptr(k)} + {off} + # * {W * R}"  # rows of sub-block #
        if _contiguous(R, W):  # the sub-block's rows as R vectors, entry i deinterleaved from them
            src = f"D{k}_{i}_#"
            if ("V", k) not in done:
                done.add(("V", k))
                lines += [f"  const {mv} V{k}_{v}_# = ld{z}({row(v * W)}, W);" for v in range(R)]
                cheap.update([f"V{k}_{v}_#" for v in range(R)] + [f"D{k}_{e}_#" for e in range(R)])
                lines += [f"  const {mv} D{k}_{e}_# = {_lanes([divmod(s * R + e, W) for s in range(W)], lambda v: f'V{k}_{v}_#', W, ive)};"
                          for e in range(R)]
        else:
            src = f"G{k}_{g}_#[{i % W}]"
            if ("G", k, g) not in done:  # entries gW .. gW + W - 1 of the sub-block's W rows, transposed
                done.add(("G", k, g))
                rows = ", ".join(f"ld{z}({row(s * R + g * W)}, {min(W, R - g * W)})" for s in range(W))
                lines += [f"  {mv} G{k}_{g}_#[W] = {{{rows}}};", f"  tr{z}(G{k}_{g}_#);"]
                cheap.add(f"G{k}_{g}_#")
        lines.append(f"  const {vtype(n)} t{n.id}_# = {fromem(src, k, n.t)};")

    def uniform(x, U, memo):
        """Whether x is the same in all lanes once the carries in U are (as value's uni: constants, values the same
        for all samples, what is computed from them alone; loads, tables and picks at such an index)."""
        if is_c(x) or x.id in uni or x.id in U:
            return True
        if x.id not in memo:
            memo[x.id] = False  # (cycles: not uniform)
            memo[x.id] = x.op not in ("in", "sample", "while", "loopout", "loopmask", "carry", "carrymask") and \
                all(uniform(y, U, memo) for y in (x.args[1:] if x.op in ("load", "table") else x.args) if isinstance(y, Node))
        return memo[x.id]

    def loop(L, lines, done):
        (phs, outs, cond), inits = L.args[0], L.args[1:1 + len(L.args[0][0])]
        tiled = _tiles(L, done)
        if tiled and K * tiled[1] * len(tiled[3]) * W * 8 <= TILE_BUFFER:  # (the shared values' buffer: else a plain loop)
            return sums(L, *tiled, lines, done)
        whiles.append(L.id)
        const = lambda x, p: ("MALL" if x else "MNONE") if p.op == "carrymask" else value(x, lines, done, p.t)
        scope, body = set(done), []
        for i, (p, x) in enumerate(zip(phs, inits)):
            names[p.id] = f"c{L.id}_{i}_#"
            scope.add(p.id)
            lines.append(f"  {vtype(p)} c{L.id}_{i}_# = {const(x, p) if is_c(x) else value(x, lines, done)};")
        U = {p.id for p, x in zip(phs, inits) if uniform(x, (), {})}  # carries the same in all lanes: from such a start,
        while U != (V := {p.id for p, o in zip(phs, outs) if p.id in U and uniform(o, U, {})}):  # by such steps (a
            U = V  # counter), the greatest such set
        free = uniform(cond, U, {})  # (all lanes run the same steps: no masks, the carries not blended)
        if free:  # (else lanes that ended keep their carries while the others go on: the same only while all run;
            uni.update(U)  # a read at such a counter per lane, not lane 0's)
        else:
            U = set()
        act = value(cond, body, scope)  # an inner loop in the condition: masked by the enclosing loop's lanes
        body += [f"  const vm act{L.id}_# = (vm)({act}{f' & {acts[-1]}' if acts else ''});",
                 f"  if (!many((vm)({' | '.join(f'act{L.id}_{j}' for j in range(K))}))) break;"]
        acts.append(f"act{L.id}_#")
        new = []
        for i, (p, o) in enumerate(zip(phs, outs)):
            v = const(o, p) if is_c(o) else value(o, body, scope)
            if not is_c(o) and o.op in ("carry", "carrymask"):  # another carry: copy before the blends
                body.append(f"  const {vtype(p)} u{L.id}_{i}_# = {v};")
                v = f"u{L.id}_{i}_#"
            new.append(v)
        body += [f"  c{L.id}_{i}_# = {v};" if free else f"  c{L.id}_{i}_# = (vm)((act{L.id}_# & {v}) | (MNOT(act{L.id}_#) & c{L.id}_{i}_#));"
                 if p.op == "carrymask" else f"  c{L.id}_{i}_# = sel_{p.t}(act{L.id}_#, c{L.id}_{i}_#, {v});"
                 for i, (p, v) in enumerate(zip(phs, new))]
        acts.pop()
        lines += ["  for (;;) {"] + ["  " + s for s in body] + ["  }"]

    def sums(L, counters, Q, tiles, shared, lines, done):
        """A loop over a counter (Q steps, the same in all lanes) whose other carries only accumulate, in tiles: the
        values several tiles read are computed once per step in a first loop and
        kept in a buffer, then each tile of accumulators runs its own loop over the steps, its accumulators in
        registers. Small loops instead of one large body: gcc compiles them much faster."""
        (phs, outs, _), inits = L.args[0], L.args[1:1 + len(L.args[0][0])]
        F = len(shared)
        lines.append(f"  static _Thread_local vd* SB{L.id} = NULL;")
        lines.append(f"  if (!SB{L.id}) SB{L.id} = aligned_alloc(sizeof(vd), (size_t){K * max(1, Q * F)} * sizeof(vd));")
        lines.append(f"  if (!SB{L.id}) abort();  /* (out of memory: TILE_BUFFER bounds it) */")
        for j in (j for t in tiles for j in t):
            names[phs[j].id] = f"c{L.id}_{j}_#"
            lines.append(f"  {vtype(phs[j])} c{L.id}_{j}_# = {value(inits[j], lines, done, phs[j].t)};")
        step = lambda: [f"  const {vtype(phs[j])} q{L.id}_{j}_# = bc_{phs[j].t}(({'float' if phs[j].t == 's' else 'double'})q{L.id});" for j in counters]
        # (a counter of its carry's type: float32 too)
        sb = lambda x, k: f"(*({_VT[x.t] if isinstance(x, Node) else 'vd'}*)&SB{L.id}[(# * {Q} + q{L.id}) * {F} + {k}])"  # (a
        # buffer of vd: the largest type)
        for j in counters:
            names[phs[j].id] = f"q{L.id}_{j}_#"
            uni.add(phs[j].id)
        scope = set(done) | {phs[j].id for j in counters}
        if F:
            body = step()
            for k, x in enumerate(shared):
                body.append(f"  {sb(x, k)} = {value(x, body, scope)};")
            lines += ["  _Pragma(\"GCC unroll 1\")", f"  for (long long q{L.id} = 0; q{L.id} < {Q}; ++q{L.id}) {{"] + ["  " + b for b in body] + ["  }"]
        for t in tiles:
            for k, x in enumerate(shared):
                names[x.id] = sb(x, k)
            tscope = scope | {x.id for x in shared}
            body = step()
            for j in t:
                c = next(a for a in outs[j].args if a is not phs[j])
                body.append(f"  c{L.id}_{j}_# += {value(c, body, tscope, phs[j].t)};")
            lines += ["  _Pragma(\"GCC unroll 1\")", f"  for (long long q{L.id} = 0; q{L.id} < {Q}; ++q{L.id}) {{"] + ["  " + b for b in body] + ["  }"]
        lines += [f"  const {vtype(phs[j])} c{L.id}_{j}_# = bc_{phs[j].t}({float(Q)});" for j in counters]  # (the counters' final values)

    reduce = reduce or [False] * len(outputs)
    allscat = [hasattr(a, "entries") for a in outputs]  # sums over the samples at sample-dependent positions
    assert all(r for r, c in zip(reduce, allscat) if c), "a scatter output must be reduced"
    big = [c and a.size >= SCAT_BIG for a, c in zip(outputs, allscat)]  # into large arrays: through buffers (_BIG)
    hsrc = [_sources([f for f, _ in a.entries]) for a, b in zip(outputs, big) if b]  # (inputs the positions depend on)
    scat = [c and not b for c, b in zip(allscat, big)]  # into the threads' accumulators
    vacc = [r and not c for r, c in zip(reduce, allscat)]  # reduced in vector accumulators
    rout = [a.size if c else int(np.size(a)) for a, c in zip(outputs, allscat)]
    roff = [int(sum(r for r, d in zip(rout[:j], vacc) if d)) if vacc[j] else -1 for j in range(len(outputs))]
    soff = [int(sum(r for r, c in zip(rout[:j], scat) if c)) if scat[j] else -2 if big[j] else -1 for j in range(len(outputs))]
    lanes = f"((vi){{{', '.join(map(str, range(W)))}}} + bm(# * W))"  # parenthesized: a macro argument
    lines, done = [f"  const vm keep_# = {_mcmp('lt', lanes, 'bm(m)', RB, 'i')};"] if any(vacc) else [], set()
    top[:] = [lines, done]
    for j, a in enumerate(outputs):  # per group of W entries: values, transposed to rows, stored
        if big[j]:  # each entry's value, and its position (as a double) where the driver builds the plan of the sums
            b, NE = sum(big[:j]), len(a.entries)  # (big[2 b + 1] not NULL), into the buffers, in chunks through arrays
            for c in range(0, NE, SCAT_CHUNK):
                chunk = a.entries[c:c + SCAT_CHUNK]
                vals = [todbl(x, value(x, lines, done)) for _, x in chunk]
                pl, pdone = [], set(done)  # (the positions' code in the branch only: not `done` after it)
                pis = [pos(f, ("pos", j), pl, pdone) for f, _ in chunk]
                at = f"((s0 / W + #) * {NE} + {c}) * W"
                lines += ["  {", f"  vd SI_#[{len(vals)}], SV_#[{len(vals)}];"] + [f"  SV_#[{e}] = {v};" for e, v in enumerate(vals)]
                lines += [f"  if (big[{2 * b + 1}]) {{"] + pl + [f"  SI_#[{e}] = {i};" for e, i in enumerate(pis)] + ["  }"]
                lines += [f"  bput((double*)big[{2 * b}] + {at}, big[{2 * b + 1}] ? (double*)big[{2 * b + 1}] + {at} : NULL, SV_#, SI_#, {len(vals)});", "  }"]
            continue
        if scat[j]:  # each entry into the thread's accumulator of the output, lane by lane (padding lanes skipped); many:
            for c in range(0, len(a.entries), SCAT_CHUNK):  # in chunks, through arrays (one call, not one loop each)
                vals = []
                for f, x in a.entries[c:c + SCAT_CHUNK]:  # (their code first, outside the chunk's scope)
                    v = todbl(x, value(x, lines, done))
                    vals.append((pos(f, ("pos", j), lines, done), v))
                at = f"sacc + (int64_t){P(soff[j], ('soff', j))}, (int64_t){P(rout[j], ('rout', j))}"
                args = f"{at}, # * W, m, srng + {2 * sum(scat[:j])}"
                if len(vals) == 1:
                    lines.append(f"  scat({at}, {vals[0][0]}, {vals[0][1]}, # * W, m, srng + {2 * sum(scat[:j])});")
                    continue
                lines += ["  {", f"  vd SI_#[{len(vals)}], SV_#[{len(vals)}];"]
                lines += [f"  SI_#[{e}] = {i}; SV_#[{e}] = {v};" for e, (i, v) in enumerate(vals)]
                lines += [f"  scatn({args}, SI_#, SV_#, {len(vals)});", "  }"]
            continue
        if reduce[j]:  # the sample's values into the thread's accumulators (padding lanes masked out; float32 values
            # before they become doubles: masked selects of doubles in two registers make gcc's combiner take minutes)
            lines += [f"  acc[{roff[j] + i}] += " + (f"__builtin_convertvector(sel_s(keep_#, bc_s(0), {value(x, lines, done)}), vd);"
                      if isinstance(x, Node) and x.t == "s" else f"sel_d(keep_#, bc_d(0.0), {todbl(x, value(x, lines, done))});")
                      for i, x in enumerate(np.asarray(a, dtype=object).ravel())]
    # the outputs stored as they are: in groups (W entries, or a whole output whose rows interleave), each group where
    # its entries are ready (the latest of their operands, in emission order) and not output by output: an unrolled
    # scan's ys per step, not after its final carry (their values live to the end: gcc took 3-4 times as long)
    flats = {j: list(np.asarray(a, dtype=object).ravel()) for j, a in enumerate(outputs) if not (allscat[j] or reduce[j])
             and np.size(a)}  # (outputs without elements: nothing to store)
    stores = [(j, None) if _contiguous(len(f), W) else (j, g) for j, f in flats.items() for g in
              ([None] if _contiguous(len(f), W) else range(0, len(f), W))]
    pos = {n.id: k for k, n in enumerate(schedule([x for f in flats.values() for x in f if isinstance(x, Node)], done))}
    ready = lambda x: max((pos.get(y.id, -1) for y in x.args if isinstance(y, Node)), default=-1) if isinstance(x, Node) else -1
    for j, g in sorted(stores, key=lambda jg: max(map(ready, flats[jg[0]] if jg[1] is None else flats[jg[0]][jg[1]:jg[1] + W]))):
        flat = flats[j]
        R = len(flat)
        row = lambda off: f"out[{j}] + {off} + # * {W * R}"
        if g is None:  # entries interleaved into the sub-block's R contiguous vectors
            vals = [tomem(x, value(x, lines, done), j) for x in flat]
            lines += [f"  st({row(u * W)}, {_lanes([divmod(u * W + t, R)[::-1] for t in range(W)], vals.__getitem__, W)}, W);"
                      for u in range(R)]
            continue
        xs = flat[g:g + W]
        vals = [tomem(x, value(x, lines, done), j) for x in xs] + ["(ve){}"] * (W - len(xs))
        lines += [f"  ve S{j}_{g}_#[W] = {{{', '.join(vals)}}};", f"  tr(S{j}_{g}_#);"]
        lines += [f"  st({row(s * R + g)}, S{j}_{g}_#[{s}], {min(W, R - g)});" for s in range(W)]
    pre = lines  # (before the copies per sub-block: parts the same for every K, _parts)
    lines = [line.replace("#", str(b)) for line in lines for b in (range(K) if "#" in line else [0])]
    rin = [sizes[k] if batched[k] else 0 for k in range(len(inputs))] + [0]  # (PRM: not batched)
    rt = lambda v, c, at: f"(int64_t){P(v, at)}" if c else str(v)  # (sizes of the scatter outputs, offsets: at run time)
    hsz = ", ".join(f"n * RIN[{k}]" if batched[k] else f"(int64_t){P(sizes[k], ('in', k))}" for h in hsrc for k in h)  # (elements
    routs = ", ".join(rt(r, c, ("rout", j)) for j, (r, c) in enumerate(zip(rout, allscat)))
    soffs = ", ".join(rt(o, c, ("soff", j)) for j, (o, c) in enumerate(zip(soff, scat)))
    scs = rt(sum(r for r, c in zip(rout, scat) if c), any(scat), ("sc",))
    params = [at[-1] for at in sorted(prm, key=prm.get)] or [0.0]
    split = _parts(pre, K, W, cheap, tables, len(params), frame_max) if split else None
    if split is None:
        decls, parts = "".join(f"static const double TAB{t}[{len(vec.table_pad(v))}] = {{{', '.join(map(_lit, vec.table_pad(v)))}}};\n"
                               for v, t in tables.items()), []
    else:
        lines, parts, extra, decls = split
        params += [params[g] for g in extra]
    src = (prelude(W, RB, E) + decls
            + f"static inline void block({_SIG}) {{\n{_VOID}\n"
            + f"  const double* restrict PRM = (const double*)in[{len(inputs)}]; (void)PRM;\n"
            + "\n".join(lines) + "\n}\n"
            + _DRIVER.format(K=K, NI=len(inputs) + 1, NO=len(outputs), RIN=", ".join(map(str, rin)),
                             OKIND=", ".join(str("fiu".index(kout[j]) if E == 4 else 0) for j in range(len(outputs))) or "0",
                             ROUT=routs, ROFF=", ".join(map(str, roff)),
                             SOFF=soffs,
                             RED=sum(r for r, d in zip(rout, vacc) if d), SC=scs, NS=sum(scat),
                             SOUT=sum(r for r, d in zip(rout, reduce) if not d), NB=sum(big), L3C=l3_per_core(),
                             BOUT=", ".join(str(j) for j in range(len(outputs)) if big[j]) or "0",
                             BE=", ".join(str(len(a.entries)) for a, b in zip(outputs, big) if b) or "0", WIDE=f"{WIDE_MIN}LL",
                             LOOP=_GUIDED if whiles and not any(vacc) and not any(scat) else _STATIC, GS=group * W,
                             LINK="static " if static else "", HUGE=int(not static), FPBACK="FPMODE_SET(fp0);" if static else "",
                             HK=", ".join(str(k) for h in hsrc for k in h) or "0",
                             HOFF=", ".join(str(sum(map(len, hsrc[:b]))) for b in range(len(hsrc) + 1)),
                             HSZ=hsz or "0", NPRM=len(params),
                             ISZ=", ".join(str(msz(k)) for k in range(len(inputs))) + ", 8",
                             S=sum(r * msz(k) for k, r in enumerate(rin[:-1])) + E * sum(r for r, d in zip(rout, reduce) if not d)))
    head = prelude(W, RB, E) + _SLOT
    return src + "".join(PART_MARK + head + p for p in parts), np.array(params)


_STATIC = """\
    const int64_t b0 = ng * t / m * (GS / L), b1 = t == m - 1 ? nb : ng * (t + 1) / m * (GS / L);  /* the thread's
       samples: whole groups, the same for every K (the sums over the samples, its accumulators', in the same order: their
       results do not depend on K, which the first large call chooses); the last one also the rest */
    for (int64_t b = b0; b < b1; ++b) {"""
_GUIDED = """\
    #pragma omp for schedule(guided)  /* (kernels with data-dependent loops and no accumulators per thread:
                                         deterministic; uneven loop counts balanced over the threads) */
    for (int64_t b = 0; b < nb; ++b) {"""


_DRIVER = """\
#include <omp.h>
#define K {K}
#define L (K * W)
#define GS {GS}  /* samples per group: the threads' samples are whole groups (lcm of the variants' K, times W) */
#define NI {NI}
#define NO {NO}
#define RED {RED}
#define NS {NS}  /* scatter outputs */
#define NB {NB}  /* scatter outputs into large arrays (through buffers) */
/* elements per sample (0: not batched), or per output (ROUT, in kernel: of scatter outputs at run time); reduced
   outputs: offset in the accumulators (else -1), scatter outputs (sums at sample-dependent positions): offset in the
   scalar accumulators (else -1; SOFF, in kernel) */
static const int64_t RIN[NI + 1] = {{{RIN}}}, ISZ[NI + 1] = {{{ISZ}}}, ROFF[NO + 1] = {{{ROFF}}};  /* ISZ: bytes of an
   input's elements */
#define PS(j) (ROFF[j] < 0 && SOFF[j] == -1)  /* output j per sample (SOFF -2: a sum through buffers) */
#define SOUT {SOUT}  /* elements per sample of the outputs that are not reduced */
/* n (a multiple of W) elements from the aligned buffer s to the aligned d, non-temporal: no read for ownership,
   no cache pollution */
static inline void put(elem* restrict d, const elem* restrict s, int64_t n) {{
  for (int64_t i = 0; i < n; i += W) STREAM(d + i, *(const ve*)(s + i));
}}
static const int OKIND[NO + 1] = {{{OKIND}}};  /* output elements: 0 elem, 1 int32, 2 uint32 */
/* large buffers read at random positions (the sums through buffers): in 2 MB pages where the system grants them on
   request (transparent huge pages "madvise", as numpy asks for its large arrays): fewer TLB misses */
static void* halloc(size_t n) {{
  const size_t h = (size_t)2 << 20;
  void* a = aligned_alloc(h, (n + h - 1) / h * h);
#ifdef MADV_HUGEPAGE
  if (a) madvise(a, (n + h - 1) / h * h, MADV_HUGEPAGE);
#endif
  return a;
}}
/* a hash of n bytes at a, continuing h: 64-bit words in two independent chains per thread (memory bound) */
static uint64_t hashmem(const void* a, int64_t n, int nt, uint64_t h) {{
  const uint64_t P = 1099511628211ULL, * w = a;
  const int64_t nw = n / 8;
  uint64_t hs[nt + 1], tail = 0;
  #pragma omp parallel num_threads(nw > (1 << 16) ? nt : 1)
  {{
    const int t = omp_get_thread_num(), m = omp_get_num_threads();
    const int64_t lo = nw * t / m, hi = nw * (t + 1) / m;
    uint64_t x = 1469598103934665603ULL, y = 7809847782465536322ULL;
    int64_t i = lo;
    for (; i + 1 < hi; i += 2) {{ x = (x ^ w[i]) * P; x ^= x >> 32; y = (y ^ w[i + 1]) * P; y ^= y >> 32; }}
    if (i < hi) {{ x = (x ^ w[i]) * P; x ^= x >> 32; }}  /* (mixed: a flip of the top bit alone survives * P) */
    hs[t] = x ^ (y >> 1);
    if (t == m - 1) for (int u = t + 1; u < nt; ++u) hs[u] = 0;
  }}
  for (int t = 0; t < nt; ++t) h = (h ^ hs[t]) * P;
  memcpy(&tail, (const char*)a + nw * 8, n % 8);
  return (h ^ tail ^ (uint64_t)n) * P;
}}
static inline void putv(void* o, int64_t i, double s, int kind) {{  /* an output's element i from a double */
  if (kind == 0) ((elem*)o)[i] = (elem)s;
  else if (kind == 1) ((int32_t*)o)[i] = (int32_t)(int64_t)s;
  else ((uint32_t*)o)[i] = (uint32_t)(int64_t)s;
}}
{LINK}void kernel(int nt, int64_t n, void** p) {{
  const double* PRM = (const double*)p[NI - 1];  /* run-time values (emit_map) */
  const int64_t ROUT[NO + 1] = {{{ROUT}}}, SOFF[NO + 1] = {{{SOFF}}}, SC = {SC};
  (void)PRM;
#if defined(MADV_HUGEPAGE) && {HUGE}
  /* outputs in 2 MB pages: callers like XLA hand over new buffers per call, whose page faults (4 KB pages,
     zeroed by the OS) would otherwise cost more than the kernel for streaming kernels) */
  for (int j = 0; j < NO; ++j) {{
    const uintptr_t h = (uintptr_t)2 << 20, a = (uintptr_t)p[NI + j], lo = (a + h - 1) & ~(h - 1);
    const uintptr_t hi = (a + (PS(j) ? n : 1) * ROUT[j] * sizeof(elem)) & ~(h - 1);
    if (hi > lo) madvise((void*)lo, hi - lo, MADV_HUGEPAGE);
  }}
#endif
  vd** red = RED ? calloc(nt, sizeof(vd*)) : NULL;  /* accumulators per thread: allocated and zeroed by the thread (in
                                                    parallel, in its NUMA node; none where OpenMP grants fewer threads) */
  /* scatter outputs: per thread SC accumulators, kept between calls and zero outside the ranges a call touched
     (zeroed again when they are summed), so that a call costs its entries and the ranges, not nt SC */
  static double* sbuf = NULL;
  static int64_t scap = 0;  /* (a library may serve kernels of several sizes: its buffers grow) */
  static int sbusy = 0;
  const int own = SC && !__atomic_exchange_n(&sbusy, 1, __ATOMIC_ACQUIRE);  /* else a concurrent call: own buffer */
  if (own && scap < nt * SC) {{ free(sbuf); sbuf = calloc((size_t)nt * SC, sizeof(double)); scap = nt * SC; }}
  double* sred = !SC ? NULL : own ? sbuf : calloc((size_t)nt * SC, sizeof(double));
  int64_t* srng = malloc(((size_t)nt * 2 * NS + 1) * sizeof(int64_t));
  for (int64_t e = 0; e < (int64_t)nt * NS; ++e) {{ srng[2 * e] = INT64_MAX; srng[2 * e + 1] = -1; }}
  /* sums into large arrays at sample-dependent positions (NB of them): per entry and sample its value and position
     into buffers (streamed), summed after the loop per position in the order of a plan (the entries by position),
     kept while the positions stay the same (their hash): no scattered read-modify-writes, no per-thread copies */
  static double* bval[NB + 1], * bpos[NB + 1]; static int64_t bcap[NB + 1];
  static uint64_t phash[NB + 1]; static int64_t ptot[NB + 1], psize[NB + 1], * pstart[NB + 1];
  static void* porder[NB + 1];  /* (entry numbers: uint32, uint64 from WIDE_MIN entries) */
  static int bbusy = 0;
  const int bown = NB && !__atomic_exchange_n(&bbusy, 1, __ATOMIC_ACQUIRE);  /* else a concurrent call: own buffers */
  const int64_t NG = (n + L - 1) / L * K;  /* groups of W samples, padded to whole blocks */
  static const int64_t BOUT[NB + 1] = {{{BOUT}}}, BE[NB + 1] = {{{BE}}};
  /* the positions are a function of the inputs HK[HOFF[b]] .. HK[HOFF[b + 1] - 1] (HSZ elements each), PRM and n: where
     their hash is the plan's, the plan stays, and the blocks skip the positions (big[2 b + 1] NULL) */
  static const int HK[] = {{{HK}}}, HOFF[NB + 1] = {{{HOFF}}};
  const int64_t HSZ[] = {{{HSZ}}};
  uint64_t bhash[NB + 1];
  int bplan[NB + 1];
  void* big[2 * NB + 1];
  double* lval[NB + 1], * lpos[NB + 1];
  for (int b = 0; b < NB; ++b) {{
    uint64_t h = hashmem(PRM, {NPRM} * sizeof(double), nt, 1469598103934665603ULL ^ (uint64_t)n);
    for (int i = HOFF[b]; i < HOFF[b + 1]; ++i) h = hashmem(p[HK[i]], HSZ[i] * ISZ[HK[i]], nt, h);
    bhash[b] = h;
    bplan[b] = bown && phash[b] == h && ptot[b] == NG * BE[b] * W && psize[b] == ROUT[BOUT[b]];
    const int64_t need = NG * BE[b] * W;
    if (bown && bcap[b] < need) {{
      free(bval[b]); free(bpos[b]);
      bval[b] = halloc(need * sizeof(double)); bpos[b] = halloc(need * sizeof(double));
      bcap[b] = need;
    }}
    lval[b] = bown ? bval[b] : halloc(need * sizeof(double));
    lpos[b] = bown ? bpos[b] : halloc(need * sizeof(double));
    big[2 * b] = lval[b]; big[2 * b + 1] = bplan[b] ? NULL : lpos[b];
  }}
  const uint64_t csr = FPMODE_GET();  /* the caller's floating-point mode (XLA's threads flush subnormals) in all
                                         threads, whichever thread created OpenMP's */
  int64_t obytes = 0;  /* outputs per sample, not reduced: through the thread buffer and non-temporal stores only
                          beyond half the L3 of the threads, else normal stores (2-2.2x faster for outputs well within
                          the L3, up to 10 % slower near its size) */
  for (int j = 0; j < NO; ++j) obytes += PS(j) ? ROUT[j] * (int64_t)sizeof(elem) : 0;
  const int nts = obytes * n > (int64_t)nt * {L3C} / 2;
  const int64_t nb = n / L, ng = n / GS;  /* blocks; groups of GS samples (a multiple of every variant's L = K W) */
  int mt = 1;  /* threads that ran (the last one's accumulators take the last block, below) */
  #pragma omp parallel num_threads(nt) if(ng > 1)  /* (no threads woken for one group or none: waking them costs
                                                    milliseconds on a loaded machine) */
  {{
    const int t = omp_get_thread_num(), m = omp_get_num_threads();
    if (t == 0) mt = m;
    const uint64_t fp0 = FPMODE_GET();  /* (the thread's own mode: back after the call, in an export: the host's
                                           OpenMP threads keep theirs) */
    FPMODE_SET(csr);
    (void)fp0;
    vd* acc = RED ? red[t] = aligned_alloc(sizeof(vd), RED * sizeof(vd)) : NULL;
    if (RED) memset(acc, 0, RED * sizeof(vd));
    double* sacc = SC ? sred + (int64_t)t * SC : NULL;
    int64_t* rng = srng + (int64_t)t * 2 * NS;
    /* outputs at a vector boundary: the block's rows into a buffer of the thread (cache resident), then streamed
       as whole aligned vectors (a block's rows are L * ROUT[j] contiguous elements at a multiple of L * ROUT[j]);
       other outputs: written by the block directly */
    elem* ob = SOUT && nts ? aligned_alloc(sizeof(ve), L * SOUT * sizeof(elem)) : NULL;
{LOOP}
      const void* in[NI + 1]; elem* out[NO + 1], * q = ob;
      for (int k = 0; k < NI; ++k) in[k] = (const char*)p[k] + b * L * RIN[k] * ISZ[k];
      for (int j = 0; j < NO; ++j) {{
        out[j] = !PS(j) ? NULL : !nts || (uintptr_t)p[NI + j] % sizeof(ve) ? (elem*)p[NI + j] + b * L * ROUT[j] : q;
        q += PS(j) ? L * ROUT[j] : 0;
      }}
      block(in, out, acc, sacc, rng, big, b * L, L);
      for (int j = 0; j < NO; ++j)
        if (nts && PS(j) && (uintptr_t)p[NI + j] % sizeof(ve) == 0) put((elem*)p[NI + j] + b * L * ROUT[j], out[j], L * ROUT[j]);
    }}
    STORE_FENCE();  /* the non-temporal stores before the results are read */
    free(ob);
    {FPBACK}
  }}
  const int64_t r = n % L, b0 = n - r;
  if (r) {{  /* the last r samples: a block padded with copies of the last sample */
    char* buf = malloc(L * {S} + 1), * q = buf;  /* ({S}: bytes per sample of the batched inputs and the outputs) */
    const void* in[NI + 1]; elem* out[NO + 1];
    for (int k = 0; k < NI; ++k) {{
      in[k] = p[k];
      if (!RIN[k]) continue;
      for (int s = 0; s < L; ++s)
        memcpy(q + s * RIN[k] * ISZ[k], (const char*)p[k] + (b0 + (s < r ? s : r - 1)) * RIN[k] * ISZ[k], RIN[k] * ISZ[k]);
      in[k] = q;
      q += L * RIN[k] * ISZ[k];
    }}
    for (int j = 0; j < NO; ++j) {{ out[j] = PS(j) ? (elem*)q : NULL; q += PS(j) ? L * ROUT[j] * sizeof(elem) : 0; }}
    block(in, out, RED ? red[mt - 1] : NULL, SC ? sred + (int64_t)(mt - 1) * SC : NULL, srng + (int64_t)(mt - 1) * 2 * NS, big, b0, r);
    for (int j = 0; j < NO; ++j)
      if (PS(j)) memcpy((elem*)p[NI + j] + b0 * ROUT[j], out[j], r * ROUT[j] * sizeof(elem));
    free(buf);
  }}
  for (int b = 0; b < NB; ++b) {{  /* sums into large arrays: the plan (built where the positions changed), then the sums */
    const int64_t tot = NG * BE[b] * W, size = ROUT[BOUT[b]];
    const double* pos = lpos[b], * val = lval[b];
    /* entry i: group i / (BE W), lane i % W: sample (i / (BE W)) W + i % W; padding samples (>= n) skipped */
    #define BPOS(i) ((i) / (BE[b] * W) * W + (i) % W >= n ? -1 : (int64_t)pos[i])
    int64_t* start = bplan[b] ? pstart[b] : NULL;
    void* order = start ? porder[b] : NULL;
    const int wide = tot >= {WIDE};
    if (!start) {{
      start = halloc((size + 1) * sizeof(int64_t));
      memset(start, 0, (size + 1) * sizeof(int64_t));
      for (int64_t i = 0; i < tot; ++i) {{ const int64_t k = BPOS(i); if ((uint64_t)k < (uint64_t)size) ++start[k + 1]; }}
      for (int64_t k = 0; k < size; ++k) start[k + 1] += start[k];
      int64_t* cur = malloc((size + 1) * sizeof(int64_t));
      memcpy(cur, start, (size + 1) * sizeof(int64_t));
      order = halloc((start[size] + 1) * (wide ? sizeof(uint64_t) : sizeof(uint32_t)));
      for (int64_t i = 0; i < tot; ++i) {{
        const int64_t k = BPOS(i);
        if ((uint64_t)k >= (uint64_t)size) continue;
        if (wide) ((uint64_t*)order)[cur[k]++] = i;
        else ((uint32_t*)order)[cur[k]++] = i;
      }}
      free(cur);
      if (bown) {{ free(pstart[b]); free(porder[b]); pstart[b] = start; porder[b] = order; phash[b] = bhash[b]; ptot[b] = tot; psize[b] = size; }}
    }}
    void* o = p[NI + BOUT[b]];
    #pragma omp parallel num_threads(nt)
    {{
      FPMODE_SET(csr);
      #pragma omp for schedule(static)
      for (int64_t k = 0; k < size; ++k) {{
        double s = 0.0;
        if (wide) for (int64_t q = start[k]; q < start[k + 1]; ++q) s += val[((const uint64_t*)order)[q]];
        else for (int64_t q = start[k]; q < start[k + 1]; ++q) s += val[((const uint32_t*)order)[q]];
        putv(o, k, s, OKIND[BOUT[b]]);
      }}
    }}
    if (!bown) {{ free(start); free(order); free(lval[b]); free(lpos[b]); }}
    #undef BPOS
  }}
  if (bown) __atomic_store_n(&bbusy, 0, __ATOMIC_RELEASE);
  if (RED) {{  /* reduced outputs: the accumulators summed over threads and lanes, the entries in parallel (each in
                 the same order: the sums do not depend on how the entries are split) */
    double* sum = malloc(RED * sizeof(double));
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int64_t e = 0; e < RED; ++e) {{
      FPMODE_SET(csr);
      double s = 0.0;
      for (int t = 0; t < nt; ++t)
        for (int l = 0; t < nt && red[t] && l < W; ++l) s += red[t][e][l];
      sum[e] = s;
    }}
    for (int j = 0; j < NO; ++j)
      for (int64_t e = 0; ROFF[j] >= 0 && e < ROUT[j]; ++e) putv(p[NI + j], e, sum[ROFF[j] + e], OKIND[j]);
    for (int t = 0; t < nt; ++t) free(red[t]);
    free(sum);
  }}
  for (int j = 0, k = 0; j < NO; ++j)  /* scatter outputs: the threads' ranges summed in a fixed order, and zeroed */
    if (SOFF[j] >= 0) {{
      const int64_t* rg = srng + 2 * k++;
      #pragma omp parallel num_threads(nt)
      {{
        FPMODE_SET(csr);
        #pragma omp for schedule(static)
        for (int64_t e = 0; e < ROUT[j]; ++e) {{
          double s = 0.0;
          for (int t = 0; t < nt; ++t)
            if (e >= rg[2 * NS * t] && e <= rg[2 * NS * t + 1]) {{
              double* a = sred + (int64_t)t * SC + SOFF[j] + e;
              s += *a;
              *a = 0.0;
            }}
          putv(p[NI + j], e, s, OKIND[j]);
        }}
      }}
    }}
  free(red);
  free(srng);
  if (own) __atomic_store_n(&sbusy, 0, __ATOMIC_RELEASE);
  else free(sred);
}}
"""
