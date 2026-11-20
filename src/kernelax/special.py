# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Primitives XLA lowers in C++, as jnp functions of other primitives (frontend: _EXPANSIONS). digamma with XLA's
algorithm for polygamma (m = 0): the Lanczos approximation (g = 7, 9 terms), Euler's reflection below 0.5 (steps and
constants as in XLA's lowering, Apache-2.0); erf_inv with XLA's polynomials (JAX's flag jax_accurate_erf_inv off);
jax.random's threefry2x32 hash. Kept here rather than taken from JAX's internals, which move between versions
(internals.py)."""

import jax.numpy as jnp
import numpy as np
from jax import lax

G = 7.0
C0 = 0.99999999999980993227684700473478
C = (676.520368121885098567009190444019, -1259.13921672240287047156078755283, 771.3234287776530788486528258894,
     -176.61502916214059906584551354, 12.507343278686904814458936853, -0.13857109526572011689554707,
     9.984369578019570859563e-6, 1.50563273514931155834e-7)


def _lanczos(x):  # reflected?, z (x - 1 or -x), log(z + g + 1/2)
    reflect = x < 0.5
    z = jnp.where(reflect, -x, x - 1.0)
    return reflect, z, np.log(G + 0.5) + jnp.log1p(z / (G + 0.5))


def digamma(x):
    reflect, z, log_t = _lanczos(x)
    num = -sum(c / (z + (i + 1.0)) ** 2 for i, c in enumerate(C))
    den = C0 + sum(c / (z + (i + 1.0)) for i, c in enumerate(C))
    y = log_t + num / den - G / (z + G + 0.5)
    r = np.pi * (x + jnp.abs(jnp.floor(x + 0.5)))
    out = jnp.where(reflect, y - np.pi * jnp.cos(r) / jnp.sin(r), y)
    return jnp.where(x == 0, -jnp.inf, jnp.where((x < 0) & (x == jnp.floor(x)), jnp.nan, out))


def _unsigned(x):  # x's bits as the unsigned type of its width, the width
    u = jnp.dtype(f"uint{8 * x.dtype.itemsize}")
    return lax.bitcast_convert_type(x, u) if x.dtype != u else x, 8 * x.dtype.itemsize


def population_count(x):
    """Set bits per element (SWAR: sums of 2, 4, 8 bits, then the bytes summed by one multiply)."""
    u, b = _unsigned(x)
    c = lambda byte: np.array(int.from_bytes(bytes([byte]) * (b // 8), "little"), u.dtype)
    u = u - ((u >> 1) & c(0x55))
    u = (u & c(0x33)) + ((u >> 2) & c(0x33))
    u = (((u + (u >> 4)) & c(0x0F)) * c(0x01)) >> (b - 8)
    return lax.bitcast_convert_type(u, x.dtype) if x.dtype != u.dtype else u


def clz(x):
    """Leading zero bits: the bits below the highest set bit set, then the zeros counted."""
    u, b = _unsigned(x)
    s = 1
    while s < b:
        u, s = u | (u >> s), 2 * s
    return population_count(lax.bitcast_convert_type(~u, x.dtype) if x.dtype != u.dtype else ~u)


def nextafter(x, y):
    """The next representable value after x towards y (its bits plus or minus one)."""
    i = jnp.dtype(f"int{8 * x.dtype.itemsize}")
    bits = lax.bitcast_convert_type(x, i) + jnp.where((y > x) == (x > 0), 1, -1).astype(i)
    tiny = lax.bitcast_convert_type(jnp.ones((), i), x.dtype)  # the smallest subnormal
    out = jnp.where(x == 0, jnp.where(y > 0, tiny, -tiny), lax.bitcast_convert_type(bits, x.dtype))
    return jnp.where(jnp.isnan(x) | jnp.isnan(y), jnp.nan, jnp.where(x == y, y, out))


TOPK_SELECT = 8  # top_k of k <= n / TOPK_SELECT of n entries: k passes over them (the largest not yet taken each), else
# a sort of all of them (n^2 / 2 compare-swaps: an environment's reset drawing 2 positions of 100 cells by Gumbel top-k,
# jumanji's Maze, 189 000 nodes)


def top_k(x, k, axis=-1, total=True):
    """The k largest along `axis` and their indices, ties by lower index (a stable sort, then reversed): in XLA's total
    order as its TopK (floats by their bits: -NaN lowest, -0.0 below 0.0), or (not total: approx_max_k) in sort's
    (every NaN highest, -0.0 equal to 0.0)."""
    x = jnp.moveaxis(x, axis, -1)
    i = lax.broadcasted_iota(np.int32, x.shape, x.ndim - 1)
    key = x
    if total and jnp.issubdtype(x.dtype, jnp.floating):
        b = lax.bitcast_convert_type(x, jnp.dtype(f"int{8 * x.dtype.itemsize}"))
        key = b ^ ((b >> (8 * x.dtype.itemsize - 1)) & jnp.iinfo(b.dtype).max)  # (below 0: the lower bits flipped)
    if k * TOPK_SELECT <= x.shape[-1] and not jnp.issubdtype(key.dtype, jnp.floating):  # (integer keys: totally ordered)
        taken, vs, js = jnp.zeros(x.shape, bool), [], []
        for _ in range(k):
            m = jnp.max(jnp.where(taken, jnp.iinfo(key.dtype).min, key), axis=-1, keepdims=True)  # (the largest key
            j = jnp.argmax(~taken & (key == m), axis=-1).astype(np.int32)  # not taken, its first entry not taken)
            vs.append(m[..., 0] if key is x else lax.bitcast_convert_type(  # (the value from its key: the map of the
                m[..., 0] ^ ((m[..., 0] >> (8 * x.dtype.itemsize - 1)) & jnp.iinfo(m.dtype).max), x.dtype))  # bits is its own inverse)
            js.append(j)
            taken = taken | (i == j[..., None])
        return jnp.moveaxis(jnp.stack(vs, -1), -1, axis), jnp.moveaxis(jnp.stack(js, -1), -1, axis)
    _, j, v = lax.sort((key, -i, x), dimension=x.ndim - 1, num_keys=2)
    v, j = v[..., ::-1][..., :k], -j[..., ::-1][..., :k]
    return jnp.moveaxis(v, -1, axis), jnp.moveaxis(j, -1, axis)


def approx_top_k(x, k, axis, is_max_k):
    """approx_max_k / approx_min_k as XLA computes them on the CPU: exactly, in sort's order (NaN highest, -0.0 equal
    to 0.0; the k largest descending, the k smallest ascending)."""
    if is_max_k:
        return top_k(x, k, axis, total=False)
    x = jnp.moveaxis(x, axis, -1)  # (ascending, stable: ties by lower index, NaN last)
    v, i = lax.sort((x, lax.broadcasted_iota(np.int32, x.shape, x.ndim - 1)), dimension=x.ndim - 1, num_keys=1)
    return jnp.moveaxis(v[..., :k], -1, axis), jnp.moveaxis(i[..., :k], -1, axis)


def _bernoulli_over_factorial(k):  # B_2, B_4, ..., B_2k divided by (2j)!, exactly (fractions), as floats
    from fractions import Fraction
    from math import comb, factorial
    B = [Fraction(1)]
    for m in range(1, 2 * k + 1):
        B.append(-sum(comb(m + 1, j) * B[j] for j in range(m)) / (m + 1))
    return [float(B[2 * j] / factorial(2 * j)) for j in range(1, k + 1)]


_ZETA_EM = _bernoulli_over_factorial(12)


def zeta(x, q):
    """Hurwitz zeta: the first 10 terms, then Euler-Maclaurin at a = q + 10 (integral, half term and 12 corrections,
    as Cephes and XLA). Special values as XLA (jaxlib 0.12.1): x = 1: inf; q a non-positive integer or -inf (a pole):
    inf for even integers x, else NaN; other x < 1, x inf or NaN: NaN; q = inf: 0."""
    a, s = q + 10.0, sum((q + n) ** -x for n in range(10))
    t = a ** -x
    corr, fac = 0.0, x / a
    for k, c in enumerate(_ZETA_EM):
        corr = corr + c * fac
        fac = fac * (x + 2 * k + 1) * (x + 2 * k + 2) / (a * a)
    s = s + t * a / (x - 1.0) + 0.5 * t + t * corr
    even = (x == jnp.floor(x)) & (jnp.fmod(x, 2.0) == 0)
    pole = (q <= 0) & (q == jnp.floor(q))
    return jnp.where(x == 1.0, jnp.inf, jnp.where(pole, jnp.where(even, jnp.inf, jnp.nan), jnp.where(
        (x < 1.0) | ~jnp.isfinite(x), jnp.nan, jnp.where(q == jnp.inf, 0.0, s))))


def polygamma(m, x):
    """polygamma(m, x) = (-1)^(m + 1) m! zeta(m + 1, x) for integral m >= 1 (at the poles: inf for odd m, NaN for
    even ones, as XLA), digamma for m = 0, NaN else."""
    m = jnp.asarray(m, x.dtype)
    p = jnp.where(jnp.fmod(m, 2.0) == 0, -1.0, 1.0) * jnp.exp(lax.lgamma(m + 1.0)) * zeta(m + 1.0, x)
    p = jnp.where((x <= 0) & (x == jnp.floor(x)) & (jnp.fmod(m, 2.0) == 0), jnp.nan, p)  # (poles of even orders: NaN, as XLA)
    return jnp.where(m == 0, digamma(x), jnp.where((m < 0) | (m != jnp.floor(m)), jnp.nan, p))


def round_float(x, e, m, flush=False):
    """x (float32 or float64) rounded to a float of e exponent and m mantissa bits (e <= 8): to nearest even, too large
    to inf, NaN kept; subnormals of the format as a conversion to it keeps them (flush: zero, as XLA's
    reduce_precision)."""
    M, u = (23, jnp.uint32) if x.dtype == jnp.float32 else (52, jnp.uint64)
    bits = lax.bitcast_convert_type(x, u)
    sign, b = bits & u(1 << (M + (8 if M == 23 else 11))), bits & u((1 << (M + (8 if M == 23 else 11))) - 1)
    if M > m:  # (round half to even: add half an ulp minus one, plus the last kept bit)
        sh = M - m
        b = (b + u((1 << (sh - 1)) - 1) + ((b >> u(sh)) & u(1))) & ~u((1 << sh) - 1)
    r = lax.bitcast_convert_type(b, x.dtype)
    bias = 2 ** (e - 1) - 1
    if e < (8 if M == 23 else 11):
        r = jnp.where(r > (2 - 2.0 ** -m) * 2.0 ** bias, jnp.inf, r)
        tiny, a = 2.0 ** (1 - bias), jnp.abs(x)
        magic = 2.0 ** (1 - bias - m + M)  # (its ulp: the format's smallest subnormal)
        r = jnp.where(r < tiny, 0.0, r) if flush else jnp.where(a < tiny, (a + magic) - magic, r)
    out = lax.bitcast_convert_type(lax.bitcast_convert_type(r, u) | sign, x.dtype)
    return jnp.where(jnp.isnan(x), x, out)


# erf_inv as XLA lowers it (chlo_legalize_to_hlo.cc, Apache-2.0): Giles' polynomials in w = -log(1 - x^2), shifted, per
# range of w (float32: w < 5, else; float64: w < 6.25, w < 16, else), in Horner form from the highest coefficient
_ERF_INV = {np.dtype(np.float32): (
    (5.0, lambda w: w - 2.5, (2.81022636e-08, 3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
                              -0.00125372503, -0.00417768164, 0.246640727, 1.50140941)),
    (np.inf, lambda w: jnp.sqrt(w) - 3.0, (-0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844,
                                           0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682))),
    np.dtype(np.float64): (
    (6.25, lambda w: w - 3.125, (
        -3.6444120640178196996e-21, -1.685059138182016589e-19, 1.2858480715256400167e-18, 1.115787767802518096e-17,
        -1.333171662854620906e-16, 2.0972767875968561637e-17, 6.6376381343583238325e-15, -4.0545662729752068639e-14,
        -8.1519341976054721522e-14, 2.6335093153082322977e-12, -1.2975133253453532498e-11, -5.4154120542946279317e-11,
        1.051212273321532285e-09, -4.1126339803469836976e-09, -2.9070369957882005086e-08, 4.2347877827932403518e-07,
        -1.3654692000834678645e-06, -1.3882523362786468719e-05, 0.0001867342080340571352, -0.00074070253416626697512,
        -0.0060336708714301490533, 0.24015818242558961693, 1.6536545626831027356)),
    (16.0, lambda w: jnp.sqrt(w) - 3.25, (
        2.2137376921775787049e-09, 9.0756561938885390979e-08, -2.7517406297064545428e-07, 1.8239629214389227755e-08,
        1.5027403968909827627e-06, -4.013867526981545969e-06, 2.9234449089955446044e-06, 1.2475304481671778723e-05,
        -4.7318229009055733981e-05, 6.8284851459573175448e-05, 2.4031110387097893999e-05, -0.0003550375203628474796,
        0.00095328937973738049703, -0.0016882755560235047313, 0.0024914420961078508066, -0.0037512085075692412107,
        0.005370914553590063617, 1.0052589676941592334, 3.0838856104922207635)),
    (np.inf, lambda w: jnp.sqrt(w) - 5.0, (
        -2.7109920616438573243e-11, -2.5556418169965252055e-10, 1.5076572693500548083e-09, -3.7894654401267369937e-09,
        7.6157012080783393804e-09, -1.4960026627149240478e-08, 2.9147953450901080826e-08, -6.7711997758452339498e-08,
        2.2900482228026654717e-07, -9.9298272942317002539e-07, 4.5260625972231537039e-06, -1.9681778105531670567e-05,
        7.5995277030017761139e-05, -0.00021503011930044477347, -0.00013871931833623122026, 1.0103004648645343977,
        4.8499064014085844221)))}


def erf_inv(x):
    """erf_inv of float32 or float64 x as XLA computes it in C++ (_ERF_INV), with JAX's flag jax_accurate_erf_inv off."""
    w = -jnp.log1p(x * -x)
    p = None
    for bound, shift, cs in reversed(_ERF_INV[np.dtype(x.dtype)]):
        t, q = shift(w), jnp.full_like(x, cs[0])
        for c in cs[1:]:
            q = c + q * t
        p = q if p is None else jnp.where(w < bound, q, p)
    return jnp.where(jnp.abs(x) == 1.0, np.inf * x, p * x)


def threefry2x32(k1, k2, x1, x2):
    """The Threefry-2x32 hash of jax.random (Salmon et al. 2011, 20 rounds): key words k1, k2, counter words x1, x2
    (uint32) -> the two hashed words, unrolled as JAX's lowering."""
    ks = (k1, k2, k1 ^ k2 ^ np.uint32(0x1BD11BDA))
    x = [x1 + ks[0], x2 + ks[1]]
    for i in range(5):
        for r in ((13, 15, 26, 6), (17, 29, 16, 24))[i % 2]:
            x[0] = x[0] + x[1]
            x[1] = x[0] ^ ((x[1] << np.uint32(r)) | (x[1] >> np.uint32(32 - r)))
        x = [x[0] + ks[(i + 1) % 3], x[1] + ks[(i + 2) % 3] + np.uint32(i + 1)]
    return tuple(x)

def f16_bits(x):
    """The float16 encoding (int32, 0..0xffff) of x, a float32 holding a float16 value (exact: normal, subnormal, inf,
    NaN with its payload's top bits): bitcast_convert_type from float16."""
    b = lax.bitcast_convert_type(x, jnp.int32)
    sign, e, m = (b >> 16) & 0x8000, (b >> 23) & 0xFF, b & 0x7FFFFF
    normal = sign | ((e - 112) << 10) | (m >> 13)  # (exponent bias 127 -> 15)
    sub = sign | ((m | 0x800000) >> jnp.clip(126 - e, 0, 31))
    return jnp.where(e == 0xFF, sign | 0x7C00 | (m >> 13), jnp.where(e > 112, normal, jnp.where(e == 0, sign, sub)))


def f16_from_bits(h):
    """The float16 value (a float32) of the encoding in h's low 16 bits: bitcast_convert_type to float16."""
    h = h & 0xFFFF
    sign, e, m = (h & 0x8000) << 16, (h >> 10) & 0x1F, h & 0x3FF
    normal = lax.bitcast_convert_type(sign | ((e + 112) << 23) | (m << 13), jnp.float32)
    special = lax.bitcast_convert_type(sign | 0x7F800000 | (m << 13), jnp.float32)
    sub = jnp.where(sign != 0, -1.0, 1.0).astype(jnp.float32) * m.astype(jnp.float32) * jnp.float32(2.0 ** -24)
    return jnp.where(e == 31, special, jnp.where(e == 0, sub, normal))


def bf16_bits(x):
    """The bfloat16 encoding (int32, 0..0xffff) of x, a float32 holding a bfloat16 value: its upper 16 bits."""
    return (lax.bitcast_convert_type(x, jnp.int32) >> 16) & 0xFFFF


def bf16_from_bits(h):
    """The bfloat16 value (a float32) of the encoding in h's low 16 bits."""
    return lax.bitcast_convert_type((h & 0xFFFF) << 16, jnp.float32)
