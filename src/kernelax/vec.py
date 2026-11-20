# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""SIMD vectors of W lanes (one sample per lane) in C: the types of the kernels' values and their helpers.

Value types: vd (double), vf (float), vi (int64: integer nodes, a double's bits), vm (masks). A configuration is
the lane count W, the register size R in bytes (64: AVX-512, masks in k registers; 32: AVX2, masks as vectors of
-1/0 lanes) and the element size E of the arrays in memory (8: double, 4: float; vectors of them, ve, fill one
register: W = R / E). A value type is P registers wide (doubles in a kernel of 16 lanes: two) or narrower than one
(floats in a kernel of 8 lanes: a 32-byte register): its helpers apply the intrinsic of its register per piece
(_Type.each), so that the emitter's code is the same for every configuration (emit._c_expr: helper_<type>).
"""

import numpy as np

_PRE = {64: "_mm512", 32: "_mm256", 16: "_mm"}  # intrinsics by register bytes
_ROUND = {"floor": 0x09, "ceil": 0x0A, "trunc": 0x0B, "rint": 0x08}  # rounding mode | no exceptions
PRED = dict(lt="_CMP_LT_OQ", le="_CMP_LE_OQ", gt="_CMP_GT_OQ", ge="_CMP_GE_OQ", eq="_CMP_EQ_OQ", ne="_CMP_NEQ_UQ")
IPRED = dict(lt="_MM_CMPINT_LT", le="_MM_CMPINT_LE", gt="_MM_CMPINT_NLE", ge="_MM_CMPINT_NLT", eq="_MM_CMPINT_EQ",
             ne="_MM_CMPINT_NE")
ICMP = dict(lt="<", le="<=", gt=">", ge=">=", eq="==", ne="!=")
INLINE = {"d": ("exp", "expm1", "tanh")}  # inline where faster than libmvec: exp_d, ...
MVEC = ("sin", "cos", "tan", "exp", "log", "tanh", "sinh", "cosh", "asin", "acos", "atan", "asinh", "acosh",
        "atanh", "log1p", "expm1", "exp2", "cbrt", "erf", "erfc", "log2", "log10")  # unary libmvec (glibc >= 2.35)


def lit_d(c):
    c = float(c)
    if np.isfinite(c):
        return f"({c.hex()})"
    return f"({nan_bits(c)})" if np.isnan(c) else ("(1.0/0.0)" if c > 0 else "(-1.0/0.0)")


def nan_bits(c):
    """C of the double NaN c with its bits (sign, quiet bit, payload): 64-bit integers travel as the doubles of their
    bits (frontend), -1 and x // 0 among them (the canonical NaN would turn them into 9221120237041090560)."""
    b = int(np.float64(c).view(np.uint64))
    return f"{'-' if b >> 63 else ''}{'__builtin_nan' if b >> 51 & 1 else '__builtin_nans'}(\"{b & (1 << 51) - 1:#x}\")"


def lit_s(c):  # (rounded to the nearest float by the compiler)
    c = float(c)
    if np.isfinite(c):
        return f"((float){c.hex()})"
    return "(__builtin_nanf(\"\"))" if np.isnan(c) else ("(__builtin_inff())" if c > 0 else "(-__builtin_inff())")


def lit_i(c):
    c = int(c) % 2 ** 64
    c = c - 2 ** 64 if c >= 2 ** 63 else c
    return "(-9223372036854775807LL - 1)" if c == -2 ** 63 else f"({c}LL)"


class _Type:
    """A float type of the values (t: d or s) in a configuration: its vector vT of W lanes, made of P pieces nT
    (registers, or the part of one it fills) of n lanes each."""

    def __init__(self, t, W, R):
        self.t, self.size = t, 8 if t == "d" else 4
        self.v, self.nv, self.c, self.sfx = ("vd", "nd", "double", "pd") if t == "d" else ("vf", "nf", "float", "ps")
        self.B = min(R, self.size * W)  # bytes of a piece
        self.P, self.n = self.size * W // self.B, self.B // self.size
        if R == 16:  # (NEON: 128-bit registers, floats of a kernel of 2 lanes in half of one; no x86 intrinsics)
            self.pre, self.nat = None, f"float{8 * self.size}x{self.n}_t"
        else:
            self.pre, self.nat = _PRE[self.B], {64: "__m512", 32: "__m256", 16: "__m128"}[self.B] + ("d" if t == "d" else "")
        self.lit = lit_d if t == "d" else lit_s

    def piece(self, x, h):
        """C of piece h of the vector x (a name or a pure expression: it may be evaluated twice)."""
        return x if self.P == 1 else f"__builtin_shufflevector({x}, {x}, {', '.join(map(str, range(h * self.n, (h + 1) * self.n)))})"

    def join(self, parts, cast=None):
        """C of the vector of the pieces `parts`."""
        cast = cast or self.nv
        parts = [f"({cast})({p})" for p in parts]
        return parts[0] if self.P == 1 else f"__builtin_shufflevector({parts[0]}, {parts[1]}, {', '.join(map(str, range(2 * self.n)))})"

    def each(self, f, *xs):
        """C of the vector whose pieces are f(pieces of xs..., h) (f: C of one piece, as the native type)."""
        return self.join([f(*[f"({self.nat})({self.piece(x, h)})" for x in xs], h) for h in range(self.P)])

    def mask(self, m, h, kmask):
        """Piece h of mask m: k register bits (kmask), else the vector of -1/0 lanes of this type's width."""
        if kmask:
            return f"(__mmask{max(8, self.n)})(({m}) >> {h * self.n})" if self.P > 1 else f"(__mmask{max(8, self.n)})({m})"
        return self.piece(f"__builtin_convertvector({m}, {'vi' if self.t == 'd' else 'vi32'})", h)


_THREEFRY = ("static inline __attribute__((always_inline)) vi threefry_v(vi a, vi b, vi c, vi d) {  /* threefry2x32 (dag.threefry): 32-bit lanes, the words packed */\n"
               "  const vu32 k0 = __builtin_convertvector((vu)a, vu32), k1 = __builtin_convertvector((vu)b, vu32), k2 = k0 ^ k1 ^ 0x1BD11BDAu;\n"
               "  vu32 x = __builtin_convertvector((vu)c, vu32) + k0, y = __builtin_convertvector((vu)d, vu32) + k1;\n  x += y; y = x ^ ((y << 13) | (y >> 19)); x += y; y = x ^ ((y << 15) | (y >> 17)); x += y; y = x ^ ((y << 26) | (y >> 6)); x += y; y = x ^ ((y << 6) | (y >> 26)); x += k1; y += k2 + 1u; x += y; y = x ^ ((y << 17) | (y >> 15)); x += y; y = x ^ ((y << 29) | (y >> 3)); x += y; y = x ^ ((y << 16) | (y >> 16)); x += y; y = x ^ ((y << 24) | (y >> 8)); x += k2; y += k0 + 2u; x += y; y = x ^ ((y << 13) | (y >> 19)); x += y; y = x ^ ((y << 15) | (y >> 17)); x += y; y = x ^ ((y << 26) | (y >> 6)); x += y; y = x ^ ((y << 6) | (y >> 26)); x += k0; y += k1 + 3u; x += y; y = x ^ ((y << 17) | (y >> 15)); x += y; y = x ^ ((y << 29) | (y >> 3)); x += y; y = x ^ ((y << 16) | (y >> 16)); x += y; y = x ^ ((y << 24) | (y >> 8)); x += k1; y += k2 + 4u; x += y; y = x ^ ((y << 13) | (y >> 19)); x += y; y = x ^ ((y << 15) | (y >> 17)); x += y; y = x ^ ((y << 26) | (y >> 6)); x += y; y = x ^ ((y << 6) | (y >> 26)); x += k2; y += k0 + 5u;\n"
               "  return (vi)((__builtin_convertvector(y, vu) << 32) | __builtin_convertvector(x, vu));\n}\n")


def prelude(W, R, E):
    """C of the types and helpers of a configuration (see the module docstring)."""
    k = R == 64  # k registers
    assert (R, W) in ((64, 8), (64, 16), (32, 4), (32, 8), (16, 2), (16, 4)) and W * E == R, (W, R, E)
    D, S = _Type("d", W, R), _Type("s", W, R)
    lanes = ", ".join(["c"] * W)
    out = [_HEAD.format(W=W, R=R, BD=D.B, BS=S.B, lanes=lanes, isa=_NEON_HEAD if R == 16 else _X86_HEAD), _THREEFRY]
    if R == 16:
        return "".join(out + [_neon(D, S, W, E)])
    ones = (1 << W) - 1
    if k:
        out.append(f"typedef __mmask{W} vm;\n#define MNOT(a) ((vm)~(a))\n#define MALL ((vm){ones:#x})\n#define MNONE ((vm)0)\n"
                   "static inline int many(vm m) { return m != 0; }\n")
    else:
        m = " | ".join(f"({D.piece('m', h)})" for h in range(D.P))  # (masks of 8 lanes: two registers)
        out.append("typedef vi vm;\n#define MNOT(a) (~(a))\n#define MALL bm(-1)\n#define MNONE bm(0)\n"
                   f"static inline int many(vm m) {{ const __m256i o = (__m256i)({m}); return !_mm256_testz_si256(o, o); }}"
                   "  /* any lane set */\n")
    for T in (D, S):
        out.append(_float_helpers(T, k))
    out.append(_int_helpers(D, k))
    out.append(_fmod(D) + _fmod(S))
    out.append(_MATH.replace("@RD@", str(D.B)).replace("@LANES@", ", ".join(map(str, range(W))))
               .replace("@NLANES@", ", ".join(["c"] * D.n)))
    for T in (D, S):
        out.append(_calls(T))
    out.append(_gathers(D, S, k))
    out.append(_memory(W, R, E))
    return "".join(out)


_HEAD = """\
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
{isa}#define W {W}
typedef double vd __attribute__((vector_size(8 * W)));
typedef float vf __attribute__((vector_size(4 * W)));
typedef long long vi __attribute__((vector_size(8 * W)));
typedef unsigned long long vu __attribute__((vector_size(8 * W)));
typedef int vi32 __attribute__((vector_size(4 * W)));
typedef unsigned vu32 __attribute__((vector_size(4 * W)));
/* pieces: the part of a register a value type fills, or one of the registers it spans */
typedef double nd __attribute__((vector_size({BD})));
typedef long long ni __attribute__((vector_size({BD})));
typedef float nf __attribute__((vector_size({BS})));
typedef int ni32 __attribute__((vector_size({BS})));
static inline vd bc_d(double c) {{ return (vd){{{lanes}}}; }}  /* broadcast ((vd){{}} + c is an add: -0.0) */
static inline vf bc_s(float c) {{ return (vf){{{lanes}}}; }}
static inline vi bm(long long c) {{ return (vi){{{lanes}}}; }}
#define BLENDI(m, a, b) (((a) & (m)) | ((b) & ~(m)))  /* with vector masks (-1/0 lanes) */
#define BLENDD(m, a, b) ((nd)BLENDI(m, (ni)(a), (ni)(b)))
"""


_X86_HEAD = """#include <immintrin.h>
#define FPMODE_GET() ((uint64_t)_mm_getcsr())  /* the floating-point mode (MXCSR: XLA's threads flush subnormals) */
#define FPMODE_SET(c) _mm_setcsr((unsigned)(c))
#define STORE_FENCE() _mm_sfence()  /* the non-temporal stores before the results are read */
"""
_NEON_HEAD = """#include <arm_neon.h>
#define FPMODE_GET() __builtin_aarch64_get_fpcr64()  /* the floating-point mode (FPCR) */
#define FPMODE_SET(c) __builtin_aarch64_set_fpcr64(c)
#define STORE_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)
"""
MVEC_NEON = frozenset("sin cos tan exp exp2 exp10 expm1 log log2 log10 log1p asin acos atan atan2".split())  # glibc 2.39's
# AdvSIMD variants (_ZGVnN2v_sin, _ZGVnN4v_sinf, ...); the others per lane through libm


def _neon(D, S, W, E):
    """The helpers of a NEON configuration (R = 16: 2 doubles or 4 floats a register; kernels of 2 or 4 lanes), the
    names and semantics of the x86 ones: GCC's generic vector operations where they suffice (compares, selects, max/min
    as x86's: the second operand where one is NaN, gathers lane by lane, loads and stores by memcpy), NEON intrinsics for
    sqrt and rounding, libmvec's AdvSIMD variants where glibc has them."""
    pred = dict(_CMP_LT_OQ="(a) < (b)", _CMP_LE_OQ="(a) <= (b)", _CMP_GT_OQ="(a) > (b)", _CMP_GE_OQ="(a) >= (b)",
                _CMP_EQ_OQ="(a) == (b)", _CMP_NEQ_UQ="(a) != (b)", _CMP_NEQ_OQ="((a) < (b)) | ((a) > (b))",
                _CMP_UNORD_Q="((a) != (a)) | ((b) != (b))")
    out = ["typedef vi vm;\n#define MNOT(a) (~(a))\n#define MALL bm(-1)\n#define MNONE bm(0)\n"
           "static inline int many(vm m) { long long o = 0; for (int l = 0; l < W; ++l) o |= m[l]; return o != 0; }\n"]
    for T in (D, S):
        t, v = T.t, T.v
        it, one, f = ("vi", "0x7fffffffffffffffLL", "") if t == "d" else ("vi32", "0x7fffffff", "f")
        q, bits = "q" if T.B == 16 else "", 8 * T.size
        out += [f"#define CMP{t}_{p}(a, b) ((vm)__builtin_convertvector(({it})({e}), vi))\n" for p, e in pred.items()]
        rnd = {"0x09": "vrndm", "0x0A": "vrndp", "0x0B": "vrnd", "0x08": "vrndn"}
        each = lambda op: T.each(lambda x, h: f"{op}{q}_f{bits}({x})", "(x)")
        out.append(f"#define cmp_{t}(a, b, p) CMP{t}_##p(a, b)\n"
                   f"static inline {v} sel_{t}(vm m, {v} f, {v} t) {{ return ({v})BLENDI(__builtin_convertvector(m, {it}), ({it})t, ({it})f); }}\n"
                   f"static inline {v} b2f_{t}(vm m) {{ return ({v})(({it})bc_{t}(1) & __builtin_convertvector(m, {it})); }}\n"
                   f"static inline {v} sqrt_{t}({v} x) {{ return {each('vsqrt')}; }}\n"
                   f"static inline {v} max_{t}({v} a, {v} b) {{ return sel_{t}(cmp_{t}(a, b, _CMP_GT_OQ), b, a); }}\n"
                   f"static inline {v} min_{t}({v} a, {v} b) {{ return sel_{t}(cmp_{t}(a, b, _CMP_LT_OQ), b, a); }}\n"
                   f"#define round_{t}(x, mode) ({' : '.join(f'(mode) == {m} ? {each(op)}' for m, op in rnd.items())} : (x))\n"
                   f"static inline {v} abs_{t}({v} x) {{ return ({v})(({it})x & {one}); }}\n"
                   f"static inline {v} sign_{t}({v} x) {{ return sel_{t}(cmp_{t}(x, bc_{t}(0), _CMP_NEQ_OQ), x, ({v})((({it})x & ~{one}) | ({it})bc_{t}(1))); }}\n")
    out.append(_int_helpers(D, False))
    out.append(_fmod(D) + _fmod(S))
    out.append(_MATH.replace("@RD@", str(D.B)).replace("@LANES@", ", ".join(map(str, range(W)))).replace("@NLANES@", ", ".join(["c"] * D.n)))
    for T in (D, S):
        f = "" if T.t == "d" else "f"
        for name in MVEC + ("pow", "atan2"):
            two = name in ("pow", "atan2")
            args = ("a", "b") if two else ("x",)
            sig = f"static inline {T.v} {name}_{T.t}({', '.join(f'{T.v} {a}' for a in args)})"
            if name in INLINE.get(T.t, ()):
                out.append(f"{sig} {{ return {T.each(lambda x, h: f'v{name}((nd){x})', 'x')}; }}\n")
            elif name in MVEC_NEON:
                sym = f"_ZGVnN{T.n}v{'v' if two else ''}_{name}{f}"
                out.append(f"extern __attribute__((aarch64_vector_pcs)) {T.nat} {sym}({', '.join([T.nat] * len(args))});\n"  # (the
                           f"{sig} {{ return {T.each(lambda *p: f'{sym}({chr(44).join(p[:-1])})', *args)}; }}\n")  # vector call standard)
            else:  # (lane by lane)
                out.append(f"{sig} {{ {T.v} r; for (int l = 0; l < W; ++l) r[l] = {name}{f}({', '.join(f'{a}[l]' for a in args)}); return r; }}\n")
        out.append(f"static inline {T.v} gather_{T.t}(const {T.c}* t, vi i) {{ {T.v} r; for (int l = 0; l < W; ++l) r[l] = t[i[l]]; return r; }}\n"
                   f"static inline {T.v} lookup_{T.t}(const {T.c}* t, vd x, long long n) {{ return gather_{T.t}(t, clampi((vi)(x + MAGIC) - (vi)MAGIC, n)); }}\n"
                   f"static inline {T.v} pickv_{T.t}(const {T.v}* a, vd x, long long n) {{ return gather_{T.t}((const {T.c}*)a, clampi((vi)(x + MAGIC) - (vi)MAGIC, n) * W + LANES); }}\n"
                   f"static inline {T.v} lookup1_{T.t}(const {T.c}* t, vd x, long long n) {{ return bc_{T.t}(t[clampi((vi)(x + MAGIC) - (vi)MAGIC, n)[0]]); }}\n"
                   f"static inline {T.v} pick1_{T.t}(const {T.v}* a, vd x, long long n) {{ return a[clampi((vi)(x + MAGIC) - (vi)MAGIC, n)[0]]; }}\n")
    out.append(f"typedef {'double' if E == 8 else 'float'} elem;\ntypedef {'vd' if E == 8 else 'vf'} ve;\n"
               f"typedef {'vi' if E == 8 else 'vi32'} ive;\n#define ld ld{E}\n#define st st{E}\n#define tr tr{E}\n"
               "static inline void STREAM_d(double* p, vd v) { memcpy(p, &v, sizeof v); }\n"
               "#define STREAM(p, v) do { const ve v_ = (v); memcpy(p, &v_, sizeof v_); } while (0)\n")
    for size, v, c in ((8, "vd", "double"), (4, "vf", "float")):  # (n elements, the rest zeros: n mostly the constant W)
        out.append(f"static inline {v} ld{size}(const {c}* p, int n) {{ {v} v = {{0}}; memcpy(&v, p, n * {size}); return v; }}\n"
                   f"static inline void st{size}({c}* p, {v} v, int n) {{ memcpy(p, &v, n * {size}); }}\n")
    return "".join(out)


def _float_helpers(T, k):
    """Compare (cmp_T: a mask), select, mask as 0/1, sqrt, max/min (the intrinsics': the second operand where one
    is NaN), rounding, abs, sign (NaN: NaN, +-0: +-0) of a float type."""
    t, v, sfx, pre, nat = T.t, T.v, T.sfx, T.pre, T.nat
    it, one = ("vi", "0x7fffffffffffffffLL") if t == "d" else ("vi32", "0x7fffffff")
    if k:  # k registers: one per piece, the pieces' bits side by side
        cmp = f"#define cmp_{t}(a, b, p) " + _kjoin([f"{pre}_cmp_{sfx}_mask(({nat})({T.piece('(a)', h)}), ({nat})({T.piece('(b)', h)}), p)"
                                                    for h in range(T.P)]) + "\n"
        sel = T.join([f"{pre}_mask_blend_{sfx}({T.mask('m', h, k)}, ({nat}){T.piece('f', h)}, ({nat}){T.piece('t', h)})"
                      for h in range(T.P)])
        b2f = T.join([f"{pre}_maskz_mov_{sfx}({T.mask('m', h, k)}, ({nat}){T.piece(f'bc_{t}(1)', h)})" for h in range(T.P)])
    else:  # vectors of -1/0 lanes (vi), of the type's width for the blends (per piece: doubles of 8 lanes in two)
        ni = "ni" if t == "d" else "ni32"
        parts = T.join([f"{pre}_cmp_{sfx}(({nat})({T.piece('(a)', h)}), ({nat})({T.piece('(b)', h)}), p)" for h in range(T.P)], ni)
        cmp = f"#define cmp_{t}(a, b, p) ((vm)__builtin_convertvector(({it})({parts}), vi))\n"
        sel = f"({v})BLENDI(__builtin_convertvector(m, {it}), ({it})t, ({it})f)"
        b2f = f"({v})(({it})bc_{t}(1) & __builtin_convertvector(m, {it}))"
    rnd = (lambda x, h: f"_mm512_roundscale_{sfx}({x}, mode)") if T.B == 64 else (lambda x, h: f"{pre}_round_{sfx}({x}, mode)")
    return (cmp
            + f"static inline {v} sel_{t}(vm m, {v} f, {v} t) {{ return {sel}; }}\n"
            + f"static inline {v} b2f_{t}(vm m) {{ return {b2f}; }}\n"
            + f"static inline {v} sqrt_{t}({v} x) {{ return {T.each(lambda x, h: f'{pre}_sqrt_{sfx}({x})', 'x')}; }}\n"
            + "".join(f"static inline {v} {op}_{t}({v} a, {v} b) {{ return {T.each(lambda a, b, h: f'{pre}_{op}_{sfx}({a}, {b})', 'a', 'b')}; }}\n"
                      for op in ("max", "min"))
            + f"#define round_{t}(x, mode) ({T.each(rnd, '(x)')})\n"
            + f"static inline {v} abs_{t}({v} x) {{ return ({v})(({it})x & {one}); }}\n"
            + f"static inline {v} sign_{t}({v} x) {{ return sel_{t}(cmp_{t}(x, bc_{t}(0), _CMP_NEQ_OQ), x, ({v})((({it})x & ~{one}) | ({it})bc_{t}(1))); }}\n")


def _fmod(T):
    """fmod_T(x, y) as C's fmod: |x| - |y| trunc(|x| / |y|), fused (exact while the quotient is below 2^52, float32:
    2^23; -ffp-contract=fast), corrected by a step where the quotient rounded to the next integer, with x's sign; lanes
    beyond, at infinities, NaN or y = 0 by libm's fmod (rare: one test of the vector)."""
    t, v = T.t, T.v
    it, one, f, big = ("vi", "0x7fffffffffffffffLL", "", "0x1p52") if t == "d" else ("vi32", "0x7fffffff", "f", "0x1p23f")
    odd = f"!(ax[l] < ay[l] * {big}) || !(ay[l] < __builtin_inf{f}())"
    return (f"static inline {v} fmod_{t}({v} x, {v} y) {{\n"
            f"  const {v} ax = abs_{t}(x), ay = abs_{t}(y);\n"
            f"  {v} r = ax - ay * round_{t}(ax / ay, 0x0B);\n"
            f"  r = sel_{t}(cmp_{t}(r, bc_{t}(0), _CMP_LT_OQ), r, r + ay);\n"
            f"  r = sel_{t}(cmp_{t}(r, ay, _CMP_GE_OQ), r, r - ay);\n"
            f"  r = ({v})((({it})r & {one}) | (({it})x & ~{one}));\n"
            f"  if (many(MNOT(cmp_{t}(ax, ay * bc_{t}({big}), _CMP_LT_OQ)) | MNOT(cmp_{t}(ay, bc_{t}(__builtin_inf{f}()), _CMP_LT_OQ))))\n"
            f"    for (int l = 0; l < W; ++l) if ({odd}) r[l] = fmod{f}(x[l], y[l]);\n"
            f"  return r;\n}}\n")


def _kjoin(parts):
    """C of the mask (vm) of the k-register masks of a value's pieces: one as it is, two of 8 lanes by kunpackb. (Not
    lo | hi << 8: gcc 13 spilled that mask a byte wide and reloaded it two bytes wide, lanes 8-15 from old stack contents,
    GCC PR target/117159; the exported sources are built by the host's compiler.)"""
    return f"((vm)({parts[0]}))" if len(parts) == 1 else f"((vm)_mm512_kunpackb((__mmask16)({parts[1]}), (__mmask16)({parts[0]})))"


def _int_helpers(D, k):
    """Integer lanes (vi, int64; the pieces of doubles): compare (signed, unsigned), select, mask as 0/1, clamp."""
    if k:
        cmp = lambda f: _kjoin([f"{D.pre}_cmp_{f}_mask((__m512i)({D.piece('(a)', h)}), (__m512i)({D.piece('(b)', h)}), p)"
                                for h in range(D.P)])
        sel = D.join([f"{D.pre}_mask_blend_epi64({D.mask('m', h, k)}, (__m512i)({D.piece('f', h)}), (__m512i)({D.piece('t', h)}))"
                      for h in range(D.P)], "ni")
        b2i = D.join([f"{D.pre}_maskz_mov_epi64({D.mask('m', h, k)}, (__m512i)({D.piece('bm(1)', h)}))" for h in range(D.P)], "ni")
        clamp = D.join([f"_mm512_max_epi64(_mm512_min_epi64((__m512i)({D.piece('x', h)}), (__m512i)({D.piece('bm(n - 1)', h)})), "
                        f"_mm512_setzero_si512())" for h in range(D.P)], "ni")
        return (f"#define cmp_i(a, b, p) ((vm)({cmp('epi64')}))\n#define cmp_u(a, b, p) ((vm)({cmp('epu64')}))\n"
                f"static inline vi misel(vm m, vi f, vi t) {{ return {sel}; }}\n"
                f"static inline vi mb2i(vm m) {{ return {b2i}; }}\n"
                f"static inline vi clampi(vi x, long long n) {{ return {clamp}; }}  /* x in 0 .. n - 1 */\n")
    return ("static inline vi misel(vm m, vi f, vi t) { return BLENDI(m, t, f); }\n"
            "static inline vi mb2i(vm m) { return m & 1; }\n"
            "static inline vi clampi(vi x, long long n) { x = BLENDI(x < bm(0), bm(0), x); return BLENDI(x > bm(n - 1), bm(n - 1), x); }\n")


def _calls(T):
    """The elementary functions of a float type: libmvec's vector variant of its pieces (inline ones: _MATH)."""
    isa = {64: "e", 32: "d", 16: "b"}[T.B]
    f = "" if T.t == "d" else "f"
    out = []
    for name in MVEC + ("pow", "atan2"):
        two = name in ("pow", "atan2")
        sym = f"_ZGV{isa}N{T.n}v{'v' if two else ''}_{name}{f}"
        args = ("a", "b") if two else ("x",)
        if name in INLINE.get(T.t, ()):  # (piece-level inline functions: v<name>)
            out.append(f"static inline {T.v} {name}_{T.t}({T.v} x) {{ return {T.each(lambda x, h: f'v{name}((nd){x})', 'x')}; }}\n")
            continue
        out.append(f"extern {T.nat} {sym}({', '.join([T.nat] * len(args))});\n")
        out.append(f"static inline {T.v} {name}_{T.t}({', '.join(f'{T.v} {a}' for a in args)}) "
                   f"{{ return {T.each(lambda *p: f'{sym}({chr(44).join(p[:-1])})', *args)}; }}\n")
    return "".join(out)


def _gathers(D, S, k):
    """gather_T(t, i): t[i] per lane (i: int64 lanes in range); lookup_T(t, x, n): of a table of n entries at an
    integral double x, clamped (NaN included: no access outside the table); pickv_T(a, x, n): lane l of entry x of an
    array of n vectors; lookup1_T, pick1_T: the same for an x the same in all lanes (a load, no gather)."""
    out = []
    for T in (D, S):
        # one gather per piece of the indices (int64: as many lanes as a piece of doubles)
        if k:
            g = lambda i, h: f"_mm512_i64gather_{T.sfx}((__m512i)({D.piece(i, h)}), t, {T.size})"
        else:
            g = lambda i, h: f"_mm256_i64gather_{T.sfx}(t, (__m256i)({D.piece(i, h)}), {T.size})"
        parts = [g("i", h) for h in range(D.P)]
        if T.t == "d":
            body = D.join(parts)
        else:  # (a piece of indices gathers half a register of floats: the pieces' floats side by side)
            n = D.n
            body = parts[0] if D.P == 1 else f"__builtin_shufflevector(({_half(n)})({parts[0]}), ({_half(n)})({parts[1]}), {', '.join(map(str, range(2 * n)))})"
            body = f"(vf)({body})"
        out.append(f"static inline {T.v} gather_{T.t}(const {T.c}* t, vi i) {{ return {body}; }}\n")
        out.append(f"static inline {T.v} lookup_{T.t}(const {T.c}* t, vd x, long long n) "
                   f"{{ return gather_{T.t}(t, clampi((vi)(x + MAGIC) - (vi)MAGIC, n)); }}\n")
        out.append(f"static inline {T.v} pickv_{T.t}(const {T.v}* a, vd x, long long n) "
                   f"{{ return gather_{T.t}((const {T.c}*)a, clampi((vi)(x + MAGIC) - (vi)MAGIC, n) * W + LANES); }}\n")
        out.append(f"static inline {T.v} lookup1_{T.t}(const {T.c}* t, vd x, long long n) "  # (x the same in all lanes)
                   f"{{ return bc_{T.t}(t[clampi((vi)(x + MAGIC) - (vi)MAGIC, n)[0]]); }}\n")
        out.append(f"static inline {T.v} pick1_{T.t}(const {T.v}* a, vd x, long long n) "
                   f"{{ return a[clampi((vi)(x + MAGIC) - (vi)MAGIC, n)[0]]; }}\n")
    if k:  # tperm_d(t, x, n): lookup_d in a constant table of n <= 32 doubles (padded to 8, 16 or 32: TABLE_PAD), by
        # permutes of its registers (a gather costs ~5x as much)
        t = [f"_mm512_loadu_pd(t + {8 * j})" for j in range(4)]
        def perm(h):
            i = f"(__m512i)({D.piece('i', h)})"
            pair = lambda a, b: f"_mm512_permutex2var_pd({a}, {i}, {b})"
            return (f"(n <= 8 ? _mm512_permutexvar_pd({i}, {t[0]}) : n <= 16 ? {pair(t[0], t[1])} : _mm512_mask_blend_pd("
                    f"_mm512_cmpge_epi64_mask({i}, _mm512_set1_epi64(16)), {pair(t[0], t[1])}, {pair(t[2], t[3])}))")
        out.append(f"static inline vd tperm_d(const double* t, vd x, long long n) {{ const vi i = clampi((vi)(x + MAGIC) - (vi)MAGIC, n); "
                   f"return {D.join([perm(h) for h in range(D.P)])}; }}\n")
    return "".join(out)


TABLE_PERM = 32  # constant tables of at most this many entries: looked up by permutes (AVX-512, tperm_d)


def table_pad(v):
    """A constant table's entries, padded with zeros to the registers tperm_d loads (8, 16 or 32 entries)."""
    n = len(v)
    return list(v) + [0.0] * (next((m for m in (8, 16, 32) if n <= m), n) - n)


def _half(n):
    return f"float __attribute__((vector_size({4 * n})))"


def _memory(W, R, E):
    """The arrays in memory, of 8-byte elements (vd vectors: where W doubles fill a register) and of 4-byte ones (vf:
    float32, and int32 / uint32 by their bits): ld8 / ld4 load n <= W consecutive elements as a vector (zeros after),
    st8 / st4 store them back, masked (no access beyond p + n). The outputs' elements (`elem`, vectors `ve`, lane
    indices `ive`) have E bytes: ld, st, STREAM (non-temporal), tr (emit) of their size. STREAM_d: doubles in P pieces."""
    D = _Type("d", W, R)
    out = [f"typedef {'double' if E == 8 else 'float'} elem;\ntypedef {'vd' if E == 8 else 'vf'} ve;\n"
           f"typedef {'vi' if E == 8 else 'vi32'} ive;\n#define ld ld{E}\n#define st st{E}\n#define tr tr{E}\n",
           f"static inline void STREAM_d(double* p, vd v) {{ {' '.join(f'{D.pre}_stream_pd(p + {h * D.n}, ({D.nat})({D.piece("v", h)}));' for h in range(D.P))} }}\n"]
    for size, v, c, sfx in ((8, "vd", "double", "pd"), (4, "vf", "float", "ps")):
        B = size * W  # (bytes of a vector of W elements)
        if B > R:
            continue  # (8-byte elements in a kernel of 16 lanes: none, cpu.elements)
        nat = {64: "__m512", 32: "__m256", 16: "__m128"}[B] + ("d" if size == 8 else "")
        if R == 64:  # (masks with AVX512VL for 32-byte vectors)
            m = f"(__mmask{max(8, W)})((1u << n) - 1)"
            out.append(f"static inline {v} ld{size}(const {c}* p, int n) {{ return ({v}){_PRE[B]}_maskz_loadu_{sfx}({m}, p); }}\n"
                       f"static inline void st{size}({c}* p, {v} v, int n) {{ {_PRE[B]}_mask_storeu_{sfx}(p, {m}, ({nat})v); }}\n")
        else:  # (AVX2: masks as vectors, of the elements' width; whole vectors unmasked: a vmaskmov store costs ~6
            # stores: 1.6-2.4x on AVX2; n is mostly the constant W, and the branch folds)
            lanes = ("_mm256_cmpgt_epi64(_mm256_set1_epi64x(n), _mm256_setr_epi64x(0, 1, 2, 3))" if size == 8 else
                     f"{_PRE[B]}_cmpgt_epi32({_PRE[B]}_set1_epi32(n), {_PRE[B]}_setr_epi32({', '.join(map(str, range(W)))}))")
            out.append(f"static inline {v} ld{size}(const {c}* p, int n) {{ return n == {W} ? ({v}){_PRE[B]}_loadu_{sfx}(p) : "
                       f"({v}){_PRE[B]}_maskload_{sfx}(p, {lanes}); }}\n"
                       f"static inline void st{size}({c}* p, {v} v, int n) {{ if (n == {W}) {_PRE[B]}_storeu_{sfx}(p, ({nat})v); "
                       f"else {_PRE[B]}_maskstore_{sfx}(p, {lanes}, ({nat})v); }}\n")
        if size == E:
            out.append(f"#define STREAM(p, v) {_PRE[B]}_stream_{sfx}(p, ({nat})(v))\n")
    return "".join(out)


_MATH = """\
/* Inline elementary functions of doubles, per piece (nd), where they are faster than libmvec (log, log1p, sin and
   cos inline after fdlibm were slower). Special values (0, inf, NaN, subnormals) by blends. */
#define LANES ((vi){@LANES@})
#define MAGIC bc_d(0x1.8p52)  /* x + MAGIC - MAGIC: x rounded to an integer; bits(n + MAGIC) - bits(MAGIC) = n */
#define NMAGIC nbc(0x1.8p52)
#define SIGN ((long long)0x8000000000000000ULL)
static inline nd nbc(double c) { return (nd){@NLANES@}; }  /* broadcast to a piece */
#if @RD@ == 64
static inline nd vldexp(nd p, nd n) { return (nd)_mm512_scalef_pd((__m512d)p, (__m512d)n); }  /* p 2^n */
#else
static inline nd vldexp(nd p, nd n) {  /* p 2^n for integral |n| <= 2100: two normal powers of 2 */
  const ni k = (ni)(n + NMAGIC) - (ni)NMAGIC, k1 = k >> 1, k2 = k - k1;
  return p * (nd)((k1 + 1023) << 52) * (nd)((k2 + 1023) << 52);
}
#endif
/* exp: x = n ln2 + r (|r| <= ln2/2), exp(r) by a degree-13 Taylor polynomial (truncation < 1e-17), times
   2^n (overflow -> inf, underflow -> 0). NaN stays NaN (the clamp compares are false for NaN). */
static inline nd vexp(nd x) {
  x = BLENDD(x > nbc(709.79), nbc(709.79), x);
  x = BLENDD(x < nbc(-745.2), nbc(-745.2), x);
  nd n = (x * nbc(0x1.71547652b82fep0) + NMAGIC) - NMAGIC;
  nd r = (x - n * nbc(0x1.62e42fefa39efp-1)) - n * nbc(0x1.abc9e3b39803fp-56);
  nd p = nbc(1.0 / 6227020800.0);
  p = p * r + nbc(1.0 / 479001600.0); p = p * r + nbc(1.0 / 39916800.0); p = p * r + nbc(1.0 / 3628800.0);
  p = p * r + nbc(1.0 / 362880.0); p = p * r + nbc(1.0 / 40320.0); p = p * r + nbc(1.0 / 5040.0);
  p = p * r + nbc(1.0 / 720.0); p = p * r + nbc(1.0 / 120.0); p = p * r + nbc(1.0 / 24.0);
  p = p * r + nbc(1.0 / 6.0); p = p * r + nbc(0.5); p = p * r + nbc(1.0); p = p * r + nbc(1.0);
  return vldexp(p, n);
}
/* expm1 without cancellation: x = n ln2 + r, expm1(r) = r + r^2 q(r) (Taylor to degree 13), result
   2^n expm1(r) + (2^n - 1), exact for n = 0; 2^n (expm1(r) + 1) for n > 60 */
static inline nd vexpm1(nd x) {
  x = BLENDD(x > nbc(709.79), nbc(709.79), x);
  x = BLENDD(x < nbc(-745.2), nbc(-745.2), x);
  const nd n = (x * nbc(0x1.71547652b82fep0) + NMAGIC) - NMAGIC;
  const nd r = (x - n * nbc(0x1.62e42feep-1)) - n * nbc(0x1.a39ef35793c76p-33);
  nd q = nbc(1.0 / 6227020800.0);
  q = q * r + nbc(1.0 / 479001600.0); q = q * r + nbc(1.0 / 39916800.0); q = q * r + nbc(1.0 / 3628800.0);
  q = q * r + nbc(1.0 / 362880.0); q = q * r + nbc(1.0 / 40320.0); q = q * r + nbc(1.0 / 5040.0);
  q = q * r + nbc(1.0 / 720.0); q = q * r + nbc(1.0 / 120.0); q = q * r + nbc(1.0 / 24.0);
  q = q * r + nbc(1.0 / 6.0); q = q * r + nbc(0.5);
  const nd m = r + r * r * q;  /* for n > 60 the 1 is below the rounding, and 2^n alone could overflow */
  const nd y = BLENDD(n > nbc(60.0), vldexp(m + nbc(1.0), n), vldexp(m, n) + (vldexp(nbc(1.0), n) - nbc(1.0)));
  return BLENDD(x == nbc(0.0), x, y);  /* (-0: -0) */
}
/* tanh: sign(x) u / (u + 2), u = expm1(2 |x|); 1 for |x| > 22 */
static inline nd vtanh(nd x) {
  const nd a = (nd)((ni)x & ~SIGN), u = vexpm1(nbc(2.0) * a);
  return (nd)((ni)BLENDD(a > nbc(22.0), nbc(1.0), u / (u + nbc(2.0))) | ((ni)x & SIGN));
}
"""
