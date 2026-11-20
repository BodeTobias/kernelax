# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Scalar DAG: hash-consed nodes whose constructors simplify (CSE, constant folding, algebra).

Constants are plain Python floats/bools, never Nodes. Booleans are masks (_MASK_OPS); select trees
with constant leaves (pivot indices, failure flags) are pushed through arithmetic (_push).
"""

import functools

import numpy as np
import scipy.special


_MASK_OPS = {"lt", "le", "gt", "ge", "eq", "ne", "and", "or", "not", "carrymask", "loopmask", "icmp"}  # boolean nodes
# integer nodes (int64 lanes; narrower types wrapped by iwrap): where integers need their bits (bitwise operations,
# shifts, wraparound, bitcasts). Other integers are carried as exact doubles (indices, counters).
_INT_OPS = {"ftoi", "fbits", "f32bits", "b2i", "iadd", "isub", "imul", "iand", "ior", "ixor", "ishl", "ishr", "ishru", "iwrap",
            "iselect", "iarr", "ipick", "threefry"}

class Node:
    """Hash-consed scalar expression. Constants are plain Python floats/bools, never Nodes. t: the type of its value,
    "d" (double; also the integers carried as exact doubles), "s" (float32), "m" (mask), "i" (integer node)."""
    __slots__ = ("op", "args", "id", "t")

    def __init__(self, op, args, id, t):
        self.op, self.args, self.id, self.t = op, args, id, t

    def __add__(a, b): return add(a, b)
    def __radd__(a, b): return add(b, a)
    def __sub__(a, b): return sub(a, b)
    def __rsub__(a, b): return sub(b, a)
    def __mul__(a, b): return mul(a, b)
    def __rmul__(a, b): return mul(b, a)
    def __truediv__(a, b): return div(a, b)
    def __rtruediv__(a, b): return div(b, a)
    def __neg__(a): return neg(a)


_table = {}
_WEIGHT = [0]  # (operations of the C code beyond one per node: a threefry node's, size)
THREEFRY_WEIGHT = 110  # a threefry node's operations, as its unrolled hash had nodes (decisions by size: a scan unrolled
# within a budget, frontend; CartPole's rollout 4 200 nodes would else unroll its 200 steps into 2.2 MB of C)


def size():
    """The nodes made so far, a threefry node as THREEFRY_WEIGHT (for decisions by the size of the code)."""
    return len(_table) + _WEIGHT[0]
_IN = {}  # the inputs (by index) whose entries are float32 ("s"; set per trace)
_F32 = []  # non-empty once there is a float32 node (per trace)
_F64 = []  # non-empty once an equation computes float64 values (per trace; gpu.emit: doubles kept so in 4-byte programs)
_KEEP_NANS = []  # non-empty: x * 0, x - x and 0 / x stay operations (NaN and inf propagate as in IEEE; per trace)
_TYPED = {"tof32": "s", "bitsf32": "s", "tof64": "d", "itof": "d", "bitsf64": "d", "b2f": "d", "sample": "d", "table": "d"}
_INDEXED = {"while", "pick", "load", "table"}  # (operands of other types: not promoted)


def _type(op, args):
    """The type of a node's value (Node.t): of its float operands (float32 only if all are), else by the op."""
    if op in _MASK_OPS:
        return "m"
    if op in _INT_OPS:
        return "i"
    if op in _TYPED:
        return _TYPED[op]
    if op in ("in", "load"):
        return _IN.get(args[0], "d")
    if op in ("carry", "pick"):  # (a carry: its type is its last argument)
        return args[-1] if op == "carry" else args[0].t
    if op == "loopout":
        return args[0].args[0][0][args[1]].t
    if op == "dense":  # (frontend._dense: float32 if the matrix and the operand are)
        return "s" if _IN.get(args[0][0][0]) == "s" and all(a.t == "s" for a in args[0][1]) else "d"
    if op == "denseout":
        return args[0].t
    return "s" if any(isinstance(a, Node) and a.t == "s" for a in args) else "d"


def _bits(c):  # a constant as its key: -0.0 apart from 0.0 (equal as floats, the same hash)
    return (c, np.signbit(c)) if isinstance(c, float) and c == 0 else c


def mk(op, *args):
    key = (op,) + tuple(map(_bits, args))
    node = _table.get(key)
    if node is None:
        ts = {a.t for a in args if isinstance(a, Node)}
        if "s" in ts and "d" in ts and op not in _INDEXED:  # float32 and double operands: in double (exact widening)
            return mk(op, *[tof64(a) for a in args])
        node = _table[key] = Node(op, args, len(_table), _type(op, args))
        if node.t == "s" and not _F32:
            _F32.append(True)
    return node


def tof32(x):
    """x rounded to float32 (masks as they are)."""
    if _isb(x):
        return x
    if is_c(x):
        return float(np.float32(x))
    x = to_float(x)
    return x if x.t == "s" else x.args[0] if x.op == "tof64" else mk("tof32", x)


def tof64(x):
    """x as a double (exact)."""
    return mk("tof64", x) if isinstance(x, Node) and x.t == "s" else x


def is_c(x):
    return not isinstance(x, Node)


def _sorted(a, b):
    """Canonical operand order of commutative ops (constants first, then by node id)."""
    ka = (0, a) if is_c(a) else (1, a.id)
    kb = (0, b) if is_c(b) else (1, b.id)
    return (a, b) if ka <= kb else (b, a)


def add(a, b):
    if _isi(a) or _isi(b) or _big(a) or _big(b):  # (an integer beyond 2^53: as int64, as select does)
        return iadd(a, b)
    if is_c(a) and is_c(b):
        return a + b
    if is_c(a) and a == 0:
        return b
    if is_c(b) and b == 0:
        return a
    if not is_c(b) and b.op == "neg":
        return sub(a, b.args[0])
    if not is_c(a) and a.op == "neg":
        return sub(b, a.args[0])
    if a is b:
        return mul(2.0, a)
    return mk("add", *_sorted(a, b))


def _inf_choice(x):  # a choice between constants, one of them inf or NaN: x - x and 0 * x make NaN on purpose there
    while not is_c(x) and x.op in ("tof64", "tof32"):  # (float32 values: their selects reach sub and mul as doubles)
        x = x.args[0]
    return not is_c(x) and x.op == "select" and any(is_c(c) and not np.isfinite(c) for c in x.args[1:])


def sub(a, b):
    if _isi(a) or _isi(b) or _big(a) or _big(b):
        return isub(a, b)
    if is_c(a) and is_c(b):
        return a - b
    if is_c(b) and b == 0:
        return a
    if is_c(a) and a == 0 and np.signbit(a):  # (-0.0 - x = -x; 0.0 - 0.0 is 0.0, as XLA keeps it)
        return neg(b)
    if a is b and not _KEEP_NANS and not _inf_choice(a):
        return 0.0
    if not is_c(b) and b.op == "neg":
        return add(a, b.args[0])
    return mk("sub", a, b)


def mul(a, b):
    if _isi(a) or _isi(b):
        return imul(a, b)
    if is_c(a) and is_c(b):
        return a * b
    if is_c(b):
        a, b = b, a
    if is_c(a):
        if a == 0 and not _KEEP_NANS and not _inf_choice(b):
            return 0.0
        if a == 1:
            return b
        if a == -1:
            return neg(b)
        if b.op == "neg":  # c * (-x) = (-c) * x
            return mul(-a, b.args[0])
        if b.op == "mul" and is_c(b.args[0]) and (_pow2(a) or _pow2(b.args[0])) and _normal(a * b.args[0], b.t):
            return mul(a * b.args[0], b.args[1])  # c1 * (c2 * x) = (c1 c2) * x: exact where one of them is +-2^k (else
            # XLA rounds twice) and c1 c2 is a normal number of x's type (2^600 * 2^600 is inf: x * inf, not 1.7e61)
    elif a is b:
        return mk("mul", a, a)
    elif a.op == "neg" or b.op == "neg":  # (-x) * y = -(x y), (-x) * (-y) = x y
        prod = mul(a.args[0] if a.op == "neg" else a, b.args[0] if b.op == "neg" else b)
        return prod if a.op == b.op else neg(prod)
    return mk("mul", *_sorted(a, b))


def _normal(c, t):  # whether the constant c is a normal number of type t ("s": float32, else double)
    return (2.0 ** -126 if t == "s" else 2.0 ** -1022) <= abs(c) < (2.0 ** 128 if t == "s" else np.inf)


def _pow2(c):  # whether the constant c is +-2^k: scaling by it is exact (short of overflow and subnormals)
    return np.isfinite(c) and c != 0 and np.frexp(c)[0] in (0.5, -0.5)


def div(a, b):
    a, b = to_float(a), to_float(b)
    if is_c(b):  # (IEEE: / 0 gives +-inf or NaN; a constant's reciprocal, as XLA's algebraic simplifier takes it)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return float(np.divide(a, b)) if is_c(a) else mul(a, float(np.divide(1.0, b)))
    if is_c(a) and a == 0 and not _KEEP_NANS:
        return 0.0
    return mul(a, mk("recip", b))  # one shared reciprocal per runtime denominator (beyond 2^1022, float32: 2^126, it
    # is subnormal, in XLA's mode zero: scaling such denominators cost up to 25 %)


def neg(a):
    if _isi(a):
        return isub(0, a)
    if is_c(a):
        return -a
    return a.args[0] if a.op == "neg" else mk("neg", a)


def ipow(x, y):
    if y < 0:
        return div(1.0, ipow(x, -y))
    result, base = 1.0, x
    while y:
        if y & 1:
            result = mul(result, base)
        y >>= 1
        if y:
            base = mul(base, base)
    return result


def powr(x, y):
    x, y = to_float(x), to_float(y)
    if is_c(x) and is_c(y):
        return float(np.power(x, y))
    if is_c(y) and float(y).is_integer() and abs(y) <= 64:  # (repeated squaring: few steps, accurate)
        return ipow(x, int(y))
    if is_c(y) and y == 0.5:  # (as pow: +0 at -0, inf at -inf)
        return select(cmp("eq", x, -np.inf), unary("abs", unary("sqrt", x)), np.inf)
    return mk("pow", x, y)


_NP_UNARY = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
             "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign, "tanh": np.tanh,
             "rsqrt": lambda x: 1.0 / np.sqrt(x), "floor": np.floor, "ceil": np.ceil, "trunc": np.trunc,
             "rint": np.rint, "sinh": np.sinh, "cosh": np.cosh, "asin": np.arcsin, "acos": np.arccos,
             "atan": np.arctan, "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
             "log1p": np.log1p, "expm1": np.expm1, "exp2": np.exp2, "cbrt": np.cbrt, "log2": np.log2, "log10": np.log10,
             "erf": scipy.special.erf, "erfc": scipy.special.erfc}

def unary(op, a):
    a = to_float(a)
    if is_c(a):
        return _NP_UNARY[op](float(a)).item()
    if op == "rsqrt":
        return mk("recip", mk("sqrt", a))
    return mk(op, a)


_NP_CMP = {"lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
           "eq": np.equal, "ne": np.not_equal}


def cmp(op, a, b):
    if _isi(a) or _isi(b) or _big(a) or _big(b):
        return icmp(op, a, b)
    if is_c(a) and is_c(b):
        return bool(_NP_CMP[op](a, b))
    if op in ("eq", "ne") and isinstance(a if is_c(a) else b, (bool, np.bool_)):  # mask vs. constant
        c, x = (a, b) if is_c(a) else (b, a)
        return x if bool(c) == (op == "eq") else bnot(x)
    if _isb(a) and _isb(b) and op in ("eq", "ne"):  # masks: equal where both or neither are set
        same = bor(band(a, b), band(bnot(a), bnot(b)))
        return same if op == "eq" else bnot(same)
    if _isb(a) or _isb(b):  # masks are not numbers in C (k registers with AVX-512): order them as 0.0/1.0
        a, b = (b2f(x) if _isb(x) else x for x in (a, b))
    return mk(op, a, b)


def band(a, b):
    if is_c(a) or is_c(b):
        c, x = (a, b) if is_c(a) else (b, a)
        return x if c else False
    return a if a is b else mk("and", *_sorted(a, b))


def bor(a, b):
    if is_c(a) or is_c(b):
        c, x = (a, b) if is_c(a) else (b, a)
        return True if c else x
    return a if a is b else mk("or", *_sorted(a, b))


def bnot(a):
    if is_c(a):
        return not a
    return a.args[0] if a.op == "not" else mk("not", a)


def _isb(x):
    return isinstance(x, (bool, np.bool_)) or (isinstance(x, Node) and x.op in _MASK_OPS)


def _big(x):  # an integer constant beyond the doubles' exact integers
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)) and abs(int(x)) >= 2 ** 53


def select(p, f, t):
    if _isi(f) or _isi(t) or _big(f) or _big(t):
        return iselect(p, f, t)
    if is_c(p):
        return t if p else f
    if f is t or (is_c(f) and is_c(t) and _bits(f) == _bits(t)):
        return f
    if _isb(f) and _isb(t):  # masks: no select node, so that masks stay masks
        return bor(band(p, t), band(bnot(p), f))
    return mk("select", p, f, t)


_LEAVES = {}  # unfolded leaf counts of select trees by node id (cleared per trace)


def _leaves(x):
    """Unfolded leaf count of a select tree with constant leaves, None for other nodes."""
    if is_c(x):
        return 1
    stack = [x]  # (iterative: a select tree may be deeper than Python's recursion, an argmin over 1000 entries)
    while stack:
        n = stack[-1]
        if n.id in _LEAVES:
            stack.pop()
        elif n.op != "select":
            _LEAVES[n.id] = None
        elif todo := [a for a in n.args[1:] if not is_c(a) and a.id not in _LEAVES]:
            stack += todo
        else:
            a, b = (1 if is_c(a) else _LEAVES[a.id] for a in n.args[1:])
            _LEAVES[n.id] = None if a is None or b is None else a + b
    return _LEAVES[x.id]


def _leafy(x):
    """A select tree with at most 32 constant leaves, unfolded. _push expands such trees; larger
    ones (the binary search of jnp.interp grows them exponentially) stay ordinary nodes."""
    n = _leaves(x)
    return n is not None and n <= 32


def _push(fn, *xs):
    """fn(*xs); if all operands are constants or select trees with constant leaves (pivot
    indices of small LU solves, failure flags), fn is pushed to the leaves, so that index
    arithmetic and comparisons fold and the result is again such a tree."""
    if all(_leafy(x) for x in xs):
        for i, x in enumerate(xs):
            if not is_c(x):
                p, f, t = x.args
                return select(p, _push(fn, *xs[:i], f, *xs[i + 1:]), _push(fn, *xs[:i], t, *xs[i + 1:]))
    return fn(*xs)


def maxmin(op, a, b):
    if _isi(a) or _isi(b) or _big(a) or _big(b):
        return imaxmin(op, a, b)
    if _isb(a) and _isb(b):
        return (bor if op == "max" else band)(a, b)
    if is_c(a) and is_c(b):  # (NaN wins, as XLA)
        return float(np.maximum(a, b) if op == "max" else np.minimum(a, b))
    if is_c(a) and a != a or is_c(b) and b != b:
        return np.nan
    return mk(op, *_sorted(a, b))


def table(values, i):
    """values[i] of a constant table (tuple of floats) at an integral index i, clamped to the table."""
    i = to_float(i)
    if is_c(i):
        return values[min(max(int(i), 0), len(values) - 1)]
    return mk("table", values, i)


def pick(i, opts):
    """opts[i] at an integral index i, clamped to the options: a constant table, or an array of the options
    (built where they are known, e.g. once before a loop that reads it at its counter) read per lane."""
    i = to_float(i)
    if is_c(i):
        return opts[min(max(int(i), 0), len(opts) - 1)]
    if all(_isi(o) or isinstance(o, (int, np.integer)) and not isinstance(o, (bool, np.bool_)) for o in opts) and any(map(_isi, opts)):
        return mk("ipick", mk("iarr", *opts), i)  # (integer options: only from frontend._pick with PICK_INTS, GPU kernels)
    if any(map(_big, opts)):  # (64-bit integers beyond doubles: selects on integers)
        return functools.reduce(lambda out, v: iselect(cmp("le", i, v), out, opts[v]), range(len(opts) - 2, -1, -1), opts[-1])
    if all(is_c(o) for o in opts):
        return table(tuple(float(o) for o in opts), i)
    return mk("pick", mk("arr", *opts), i)


def load(k, i):
    """Entry i of input k (the same for all samples, kept in memory) at an integral index i, clamped to it."""
    i = to_float(i)
    return mk("in", k, int(i)) if is_c(i) else mk("load", k, i)


# ----------------------------------------------------------------------------- integers

def _isi(x):
    return isinstance(x, Node) and x.op in _INT_OPS


def to_int(x):
    """x as an integer node (or Python int): masks as 0/1, doubles truncated."""
    if _isi(x):
        return x
    if is_c(x):
        return int(x)
    if x.op == "itof" and x.args[0].op == "iwrap" and x.args[0].args[1] <= 52:  # (a wrapped integer as a double and
        return x.args[0]  # back: exact)
    return mk("b2i", x) if _isb(x) else mk("ftoi", x)


def to_float(x, unsigned=False):
    """An integer node as a double (uint64: unsigned)."""
    return mk("itof", x, int(unsigned)) if _isi(x) else x


def iadd(a, b):
    a, b = to_int(a), to_int(b)
    if is_c(a) and is_c(b):
        return a + b
    if is_c(a) and a == 0 or is_c(b) and b == 0:
        return b if is_c(a) else a
    return mk("iadd", *_sorted(a, b))


def isub(a, b):
    a, b = to_int(a), to_int(b)
    if is_c(a) and is_c(b):
        return a - b
    if is_c(b) and b == 0:
        return a
    return 0 if a is b else mk("isub", a, b)


def imul(a, b):
    a, b = to_int(a), to_int(b)
    if is_c(a) and is_c(b):
        return a * b
    if is_c(b):
        a, b = b, a
    if is_c(a) and a in (0, 1):
        return 0 if a == 0 else b
    return mk("imul", *_sorted(a, b))


_IBITS = {"iand": lambda a, b: a & b, "ior": lambda a, b: a | b, "ixor": lambda a, b: a ^ b}


def ibit(op, a, b):
    """Bitwise and, or, xor of integers (two's complement)."""
    a, b = to_int(a), to_int(b)
    if is_c(a) and is_c(b):
        return _IBITS[op](a, b)
    if is_c(b):
        a, b = b, a
    if is_c(a) and a == 0:
        return 0 if op == "iand" else b
    if a is b:
        return 0 if op == "ixor" else a
    return mk(op, *_sorted(a, b))


def ishift(op, a, s):
    """a << s (ishl), a >> s arithmetic (ishr) or logical on 64 bits (ishru), 0 <= s < 64."""
    a, s = to_int(a), to_int(s)
    if is_c(s) and s == 0:
        return a
    if is_c(a) and is_c(s):
        return a << s if op == "ishl" else a >> s if op == "ishr" else (a % 2 ** 64) >> s
    return mk(op, a, s)


def iwrap(a, bits, signed):
    """a modulo 2^bits, as a signed or unsigned bits-bit integer (64 bits: the lanes wrap by themselves)."""
    a = to_int(a)
    if is_c(a):
        m = a % (1 << bits)
        return m - (1 << bits) if signed and m >> (bits - 1) else m
    if bits == 64 or a.op == "iwrap" and a.args[1:] == (bits, signed) or not signed and (  # (within the range already:
            a.op == "iand" and any(is_c(b) and 0 <= b < 1 << bits for b in a.args) or  # a mask, the high word of 64
            a.op == "ishru" and is_c(a.args[1]) and a.args[1] >= 64 - bits):  # bits)
        return a
    return mk("iwrap", a, bits, signed)


def threefry(k1, k2, x1, x2):
    """jax.random's threefry2x32 hash of key words k1, k2 and counter words x1, x2 (uint32 values) as one node, its two
    words packed into one 64-bit integer (x1's in the low bits): the kernels compute it in 32-bit lanes, a rotation one
    instruction, instead of as 64-bit integer operations with a wraparound after each (an RL rollout's random numbers
    were 74-94 % of its operations)."""
    n = len(_table)
    node = mk("threefry", *(to_int(a) if isinstance(a, Node) else int(a) % (1 << 32) for a in (k1, k2, x1, x2)))
    _WEIGHT[0] += (THREEFRY_WEIGHT - 1) * (len(_table) - n)  # (a new node, not one hash-consed)
    return node


def icmp(op, a, b, unsigned=False):
    a, b = to_int(a), to_int(b)
    if is_c(a) and is_c(b):
        u = (lambda x: x % 2 ** 64) if unsigned else (lambda x: x)
        return bool(_NP_CMP[op](u(a), u(b)))
    return mk("icmp", op, a, b, int(unsigned))


def iselect(p, f, t):
    f, t = to_int(f), to_int(t)
    if is_c(p):
        return t if p else f
    if f is t or is_c(f) and is_c(t) and f == t:
        return f
    return mk("iselect", p, f, t)


def imaxmin(op, a, b, unsigned=False):
    return iselect(icmp("gt" if op == "max" else "lt", a, b, unsigned), b, a)


def b2f(p):
    return float(p) if is_c(p) else mk("b2f", p)


def _first_max(xs, nan=False, least=False):  # argmax (least: argmin) as a select chain (the first maximum; nan: the
    # first NaN wins, as XLA's argmax; else never, as LAPACK's idamax for pivots)
    better, end = ("lt", np.inf) if least else ("gt", -np.inf)
    i0 = next((i for i, x in enumerate(xs) if not (is_c(x) and x == end)), 0)  # masked-out prefix
    best, idx = xs[i0], i0
    for i, x in enumerate(xs[i0 + 1:], i0 + 1):
        g = bor(cmp(better, x, best), band(cmp("ne", x, x), cmp("eq", best, best))) if nan else cmp(better, x, best)
        best, idx = select(g, best, x), select(g, idx, i)
    return select(cmp("eq", best, end), idx, 0) if i0 else idx  # (all at the end value: the first)


_dot = lambda xs, ys, init=0.0: functools.reduce(add, [mul(x, y) for x, y in zip(xs, ys)], init)


def loop_roots(loop):
    """Body outputs and condition of a loop node."""
    return (*loop.args[0][1], loop.args[0][2])


def hidden(n):
    """The nodes n needs that schedule does not traverse: a loop's body outputs and condition, a dense node's operand
    (frontend._dense: all of its columns)."""
    return loop_roots(n) if n.op == "while" else n.args[0][1] if n.op == "dense" else ()


def _high(u):  # x where u is the high part of x (Veltkamp's split, c - (c - x) with c = K x, or a bit mask), else None
    c, d = u.args[0] if u.args else None, u.args[1] if len(u.args) > 1 else None
    if u.op == "sub" and isinstance(c, Node) and isinstance(d, Node) and d.op == "sub" and d.args[0] is c and \
            c.op == "mul" and any(a is d.args[1] for a in c.args):
        return d.args[1]
    masked = False
    while isinstance(u, Node) and u.op in ("bitsf64", "bitsf32", "iwrap", "iand", "fbits", "f32bits"):
        if u.op in ("fbits", "f32bits"):
            return u.args[0] if masked else None
        masked = masked or u.op == "iand"
        u = next(a for a in u.args if isinstance(a, Node))
    return None


def exact_products(nodes):
    """Ids of the float products that error-free transformations need rounded (JAX's double-word arithmetic: Veltkamp's
    split c - (c - x) with c = K x, and two_prod's error hi(a) hi(b) - a b): emitted so that the C compiler does not
    contract them into their uses (-ffp-contract=fast fuses a product into all its additions, XLA's LLVM only a product
    used once; gamma, lgamma, digamma, Bessel functions of float64 were up to 1300 ulp off). Other products stay
    contracted (all products used twice rounded cost up to 1.34x: lennard_jones, spectrum256, inv3)."""
    out = set()
    for n in nodes:
        if n.op != "sub" or not all(isinstance(a, Node) for a in n.args):
            continue
        a, b = n.args
        if b.op == "sub" and b.args[0] is a and a.op == "mul":  # (Veltkamp: c - (c - x))
            out.add(a.id)
        for m, p in ((a, b), (b, a)):  # (two_prod: hi(a) hi(b) - a b)
            if m.op == "mul" and p.op == "mul" and all(isinstance(t, Node) for t in m.args + p.args):
                hs = [_high(t) for t in m.args]
                if None not in hs and {h.id for h in hs} == {t.id for t in p.args}:
                    out.add(p.id)
    return {n.id for n in nodes if n.id in out and n.t in ("d", "s")}


GUARD_MIN = 256  # operations that only one side of selects with the same condition needs: computed only where a lane
# of the block takes that side (guards; an RL rollout's auto-reset cost 10-73 % of its time, review.md §55), from this
# many (2048's 18 groups of 46 operations: 13 % slower)
GUARD_PER_OUT = 16  # and per value the block passes to its selects at least (each lives from the block to its select:
# xland-minigrid's grid after an action, 522 operations for 128 cells, 2.8 times slower on the GPU, spilled)
GUARD_DEPTH = 4  # guarded blocks inside each other at most (deeper ones: computed in the innermost, unguarded; a select
# tree as argmin's over 1500 entries nests as deep as it is)
_FIXED = ("in", "sample", "carry", "carrymask", "dense", "denseout")  # (at block or loop level, or reading values
# schedule does not traverse: never inside a guarded block)


def _loops(nodes):
    """{node id: the loops whose carries it reads (directly or through other nodes)}. Selects of one condition in
    different loops' bodies (or one inside, one around a loop) form different groups (guards): a group's work is
    computed in one place, a body's only in its loop."""
    loop_of = {p.id: n.id for n in nodes if n.op == "while" for p in n.args[0][0]}
    out = {}
    for root in nodes:
        stack = [(root, False)]
        while stack:
            n, post = stack.pop()
            if n.id in out:
                continue
            kids = [a for a in (*n.args, *hidden(n)) if isinstance(a, Node)]
            if post:
                s = frozenset().union(*(out[a.id] for a in kids))
                out[n.id] = (s - {n.id} if n.op == "while" else s) | ({loop_of[n.id]} if n.id in loop_of else set())
            else:
                stack += [(n, True)] + [(a, False) for a in kids if a.id not in out]
    return out


def guards(roots):
    """Selects in loops that share their condition, side and loops (_loops) whose operands need GUARD_MIN or more operations
    that nothing else needs (only those selects there, the work exclusive to them): {node id: groups whose work it is},
    a group (condition, side, the guarded operands, the ids of its work). Groups nest (one's selects inside another's
    work)."""
    nodes = every(roots)
    users = {n.id: set() for n in nodes}
    for n in nodes:
        for a in [*(a for a in n.args if isinstance(a, Node)), *(h for h in hidden(n) if isinstance(h, Node))]:
            users.setdefault(a.id, set()).add(n.id)
    for r in roots:
        users.setdefault(r.id, set()).add(-1)
    loops, by = _loops(nodes), {n.id: n for n in nodes}
    sels = {}
    for n in nodes:
        if n.op == "select" and isinstance(n.args[0], Node) and n.args[0].t == "m":
            for side in (1, 2):
                if isinstance(n.args[side], Node) and n.args[side] is not n.args[3 - side]:
                    sels.setdefault((n.args[0].id, side, loops[n.id]), []).append(n)
    found = []
    for (_, side, lp), group in sels.items():  # the work grows from the guarded operands to the values all of whose
        if not lp:  # (in loops only: outside, a lane of the block takes either side almost always, interp and expm3's
            continue  # Pade degrees 12-16 % slower; in loops a sample's side changes seldom, an RL rollout's reset)
        left, work, stack, seen = {}, set(), [], set()  # users are in it (so in the time of the work, not of the graph)

        def reach(v, u):  # (u: a user of v in the work, or a select of the group taking it; each once)
            if (u.id, v.id) not in seen:
                seen.add((u.id, v.id))
                left[v.id] = left.get(v.id, len(users[v.id])) - 1
                if not left[v.id] and v.op not in _FIXED:
                    work.add(v.id)
                    stack.append(v)
        for g in group:  # (selects that take their operand on that side only)
            o = g.args[side]
            if all(not isinstance(a, Node) or a.id != o.id for a in (g.args[0], g.args[3 - side])):
                reach(o, g)
        while stack:
            u = stack.pop()
            for a in u.args:
                if isinstance(a, Node):
                    reach(a, u)
        outs = list({o.id: o for o in (g.args[side] for g in group if g.id not in work) if o.id in work}.values())  # (the
        # operands of the selects outside the work: those inside it take theirs in the block, maybe in a nested one)
        weight = len(work) + (THREEFRY_WEIGHT - 1) * sum(by[v].op == "threefry" for v in work)  # (operations, size)
        if outs and weight >= max(GUARD_MIN, GUARD_PER_OUT * len(outs)):
            found.append((group[0].args[0], side, outs, frozenset(work)))
    out = {}  # (groups nested or apart, each block inside those it is part of: the largest first, then those inside
    for grp in sorted(found, key=lambda g: -len(g[3])):  # every one they touch; a group partly inside another, as
        if all(grp[3] < k[3] for k in {id(k): k for v in grp[3] for k in out.get(v, ())}.values()):  # those of both
            for v in grp[3]:  # sides of a binary search's condition (select(c, select(c, ..), ..)), stays unguarded)
                out.setdefault(v, []).append(grp)
    return out


def every(roots):
    """Every node the roots need, also those schedule does not traverse (hidden: loop bodies and conditions, dense
    operands): for analyses that must see all of them (which inputs positions depend on, emit._sources)."""
    out, seen, stack = [], set(), [r for r in roots if isinstance(r, Node)]
    while stack:
        order = schedule(stack, seen)
        seen.update(n.id for n in order)
        out += order
        stack = [h for n in order for h in hidden(n) if isinstance(h, Node) and h.id not in seen]
    return out


def schedule(roots, done=()):
    """Depth-first post order of the nodes reachable from ``roots`` (emission order),
    without descending into nodes whose id is in ``done`` (already emitted)."""
    order, seen = [], set()
    skip = lambda n: n.id in seen or n.id in done
    for root in roots:
        if is_c(root) or skip(root):
            continue
        stack = [(root, False)]
        while stack:
            node, post = stack.pop()
            if post:
                order.append(node)
                continue
            if skip(node):
                continue
            seen.add(node.id)
            stack.append((node, True))
            stack.extend((a, False) for a in reversed(node.args) if isinstance(a, Node) and not skip(a))
    return order
