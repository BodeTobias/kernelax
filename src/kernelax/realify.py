# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Complex numbers as pairs of real values: a jaxpr with complex values inside (not as inputs or outputs) rewritten
into one of real values only (realify), which the frontend traces as any other. Complex primitives become their real
arithmetic (complex division as Smith's, fft as products with constant twiddle matrices), complex values move through
reshapes, gathers, scatters, sums and control flow as their two parts."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core as jcore


class Cx(NamedTuple):
    """A complex value: its real and imaginary parts (arrays of the real dtype; im None: zero)."""
    re: Any
    im: Any


def _isc(x):  # whether an aval, array or dtype is complex
    return jax.dtypes.issubdtype(getattr(x, "dtype", x) if x is not None else np.float64, np.complexfloating)


def _subs(eqn):  # the jaxprs in eqn's params (also in tuples)
    return jcore.jaxprs_in_params(eqn.params)


def _sub_jaxpr(eqn):  # a call primitive's jaxpr and its constants (_CALLS)
    for key in ("jaxpr", "call_jaxpr"):
        if key in eqn.params:
            sub = eqn.params[key]
            return sub, sub.consts
    raise NotImplementedError(f"kernelax: the call primitive {eqn.primitive.name} is not supported (no sub-jaxpr)")


def has_complex(jaxpr):
    """Whether a value of jaxpr (or of a jaxpr nested in it) is complex."""
    return any(_isc(v.aval) for v in jaxpr.invars + jaxpr.outvars + jaxpr.constvars) or \
        any(any(_isc(v.aval) for v in e.invars + e.outvars) or any(has_complex(s) for s in _subs(e)) for e in jaxpr.eqns)


def realify(closed):
    """closed, with its complex values as pairs of real ones (closed itself if it has none). Complex inputs or outputs:
    NotImplementedError."""
    if not has_complex(closed):
        return closed
    if any(_isc(v.aval) for v in closed.invars + closed.outvars):
        raise NotImplementedError("kernelax: complex inputs or outputs are not supported here")
    return jax.make_jaxpr(lambda *a: _eval(closed, closed.consts, list(a)))(
        *[jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in closed.invars])


def complex_leaves(f, tree, cin):
    """f(*unflatten(tree, leaves)) on flat leaves, those marked in cin complex values passed as [re, im] on a trailing
    axis, its complex outputs returned so (a kernel's interface: real values; realify within)."""
    def g(*leaves):
        out = f(*jax.tree.unflatten(tree, [lax.complex(x[..., 0], x[..., 1]) if c else x for x, c in zip(leaves, cin)]))
        return jax.tree.map(lambda o: jnp.stack([jnp.real(o), jnp.imag(o)], -1) if jnp.iscomplexobj(o) else o, out)
    return g


def _part_dtype(dtype):
    return np.float32 if np.dtype(dtype) == np.complex64 else np.float64


def _lift(c, aval):  # a constant or literal: complex as Cx of numpy parts
    if _isc(aval):
        c = np.asarray(c)
        return Cx(np.real(c).astype(_part_dtype(c.dtype)), np.imag(c).astype(_part_dtype(c.dtype)))
    return c


def _eval(jaxpr, consts, args):
    env = {v: _lift(c, v.aval) for v, c in zip(jaxpr.constvars, consts)}
    env.update(zip(jaxpr.invars, args))
    read = lambda v: _lift(v.val, v.aval) if isinstance(v, jcore.Literal) else env[v]
    for eqn in jaxpr.eqns:
        outs = _apply(eqn, [read(v) for v in eqn.invars])
        env.update(zip(eqn.outvars, outs))
    return [read(v) for v in jaxpr.outvars]


def _bind(eqn, vals):
    out = eqn.primitive.bind(*vals, **eqn.primitive.get_bind_params(eqn.params))
    return list(out) if eqn.primitive.multiple_results else [out]


_CALLS = {"jit", "eval_jaxpr", "custom_jvp_call", "custom_vjp_call", "remat2"}  # call primitives (their jaxprs inlined;
# JAX's names; also frontend's)


def eqx_error(eqn):
    """Whether eqn is a pure_callback of equinox's runtime errors (eqx.error_if: its lax.cond's branch that raises)."""
    f = getattr(eqn.params.get("callback"), "callback_func", None) if eqn.primitive.name == "pure_callback" else None
    return getattr(f, "__module__", "") == "equinox._errors" and getattr(f, "__qualname__", "").startswith("_error.")


def eqx_nan(aval):
    """equinox's value of an error with EQX_ON_ERROR=nan (_nan_like): NaN, an integer's largest value, True."""
    dt = np.dtype(aval.dtype)
    c = np.nan if np.issubdtype(dt, np.inexact) else True if dt == bool else np.iinfo(dt).max
    return np.full(aval.shape, c, dt)


def _apply(eqn, vals):
    name = eqn.primitive.name
    if not any(isinstance(v, Cx) for v in vals) and not any(_isc(v.aval) for v in eqn.outvars) and \
            not any(has_complex(s) for s in _subs(eqn)):
        return _bind(eqn, vals)
    if eqx_error(eqn):  # (in a kernel nothing raises: the sample's values as with EQX_ON_ERROR=nan, frontend)
        return [Cx(jnp.full(v.aval.shape, np.nan, _part_dtype(v.aval.dtype)), jnp.full(v.aval.shape, np.nan, _part_dtype(v.aval.dtype)))
                if _isc(v.aval) else jnp.asarray(eqx_nan(v.aval)) for v in eqn.outvars]
    if name in _CALLS:  # (inlined: the derivatives are taken already)
        return _eval(*_sub_jaxpr(eqn), vals)
    if name in _CONTROL:
        return _CONTROL[name](vals, **eqn.params)
    if name in _PARTS:  # per part: the data operands' real parts, then their imaginary parts
        data = [i for i in range(len(vals)) if (i != 1 if name == "gather" else i < _INDEX.get(name, len(vals)))]
        ims = [_im(vals[i]) for i in data]
        part = lambda f: _bind(eqn, [f(v) if i in data else v for i, v in enumerate(vals)])
        re = part(_re)
        if all(m is None for m in ims):
            return [Cx(r, None) if _isc(o.aval) else r for r, o in zip(re, eqn.outvars)]
        im = part(lambda v: _dense(v).im if isinstance(v, Cx) else jnp.zeros_like(v))
        return [Cx(r, i) for r, i in zip(re, im)]
    rule = _RULES.get(name)
    if rule is None:
        from kernelax.jaxprs import _library  # (at the call: jaxprs imports frontend, which imports this module)
        impl = _library(eqn)
        if impl is not None:  # a library's primitive written in jax (equinox's while-loop buffers: maybe_set, dynamiqs)
            return _eval(impl, impl.consts, vals)
        raise NotImplementedError(f"kernelax: complex numbers in '{name}' are not supported")
    out = rule(*vals, **eqn.params)
    out = list(out) if isinstance(out, (list, tuple)) and not isinstance(out, Cx) else [out]
    shape = lambda x, v: None if x is None else jnp.broadcast_to(x, v.aval.shape)  # (a part from a scalar: as the result)
    return [Cx(shape(o.re, v), shape(o.im, v)) if isinstance(o, Cx) else o for o, v in zip(out, eqn.outvars)]


_re = lambda v: v.re if isinstance(v, Cx) else v
_im = lambda v: v.im if isinstance(v, Cx) else None


def _dense(v):  # a Cx with both parts (loop carries, operands of per-part primitives)
    if isinstance(v, Cx) and v.im is None:
        return Cx(v.re, jnp.zeros_like(v.re))
    return v


def _add(a, b):
    return a if b is None else b if a is None else a + b


def _sub(a, b):
    return a if b is None else -b if a is None else a - b


def _mul(a, b):
    return None if a is None or b is None else a * b


def _neg(a):
    return None if a is None else -a


def _cmul(x, y):
    (p, q), (r, s) = (_re(x), _im(x)), (_re(y), _im(y))
    return Cx(_sub(p * r, _mul(q, s)), _add(_mul(p, s), _mul(q, r)))


def _cdiv(x, y):
    (p, q), (r, s) = (_re(x), _im(x)), (_re(y), _im(y))
    if s is None:
        return Cx(p / r, None if q is None else q / r)
    q = jnp.zeros_like(p) if q is None else q
    big = jnp.abs(r) >= jnp.abs(s)  # Smith's algorithm: no overflow of r^2 + s^2
    t = jnp.where(big, s / jnp.where(big, r, 1.0), r / jnp.where(big, 1.0, s))
    den = jnp.where(big, r + s * t, r * t + s)
    zero, inf = (r == 0) & (s == 0), jnp.copysign(jnp.inf, r)  # (a zero denominator: the numerator times inf with its
    return Cx(jnp.where(zero, p * inf, jnp.where(big, p + q * t, p * t + q) / den),  # sign, as C99's __divdc3 and XLA)
              jnp.where(zero, q * inf, jnp.where(big, q - p * t, q * t - p) / den))


def _abs(x):
    return jnp.hypot(x.re, x.im) if isinstance(x, Cx) and x.im is not None else jnp.abs(_re(x))


def _exp(x):
    e = jnp.exp(x.re)
    if x.im is None:
        return Cx(e, None)
    z = x.im == 0  # (a zero imaginary part stays +0, as XLA: exp(710) * 0 is no NaN)
    return Cx(jnp.where(z, e, e * jnp.cos(x.im)), jnp.where(z, 0.0, e * jnp.sin(x.im)))


def _log(x):
    if x.im is None:
        return Cx(jnp.log(jnp.abs(x.re)), jnp.where(jnp.isnan(x.re), x.re, jnp.where(jnp.signbit(x.re), np.pi, 0.0)).astype(x.re.dtype))
        # (atan2(0, x): pi also for -0.0)
    return Cx(jnp.log(jnp.hypot(x.re, x.im)), jnp.arctan2(x.im, x.re))


def _sqrt(x):
    p, q = x.re, _dense(x).im
    t = jnp.sqrt((jnp.hypot(p, q) + jnp.abs(p)) / 2)  # the principal root, as C's csqrt (x + i inf: inf + i inf)
    h = jnp.where(t == 0, 1.0, 2 * t)
    re, im = jnp.where(p >= 0, t, jnp.abs(q) / h), jnp.where(p >= 0, q / h, jnp.where(q < 0, -t, t))
    im = jnp.where(q == 0, jnp.abs(im), im)  # (a zero imaginary part's sign does not count, as in XLA: -4 - 0j gives 2j)
    return Cx(jnp.where(jnp.isinf(q), jnp.inf, re), jnp.where(jnp.isinf(q), q, im))


def _ipow(x, y):
    if y == 0:
        return Cx(jnp.ones_like(x.re), None)
    out, base, k = None, x, abs(y)
    while k:  # binary powers, as XLA's integer_pow
        if k & 1:
            out = base if out is None else _cmul(out, base)
        k >>= 1
        if k:
            base = _cmul(base, base)
    return _cdiv(Cx(jnp.ones_like(x.re), None), out) if y < 0 else out


def _trig(x, f):  # sin, cos, sinh, cosh of a + ib
    a, b = x.re, _dense(x).im
    z = lambda v: jnp.where(b == 0, 0.0, v)  # (sinh, cosh: +0 where b is +-0, as XLA; inf * 0 is NaN for a large a)
    return {"sin": Cx(jnp.sin(a) * jnp.cosh(b), jnp.cos(a) * jnp.sinh(b)), "cos": Cx(jnp.cos(a) * jnp.cosh(b), -jnp.sin(a) * jnp.sinh(b)),
            "sinh": Cx(jnp.sinh(a) * jnp.cos(b), z(jnp.cosh(a) * jnp.sin(b))), "cosh": Cx(jnp.cosh(a) * jnp.cos(b), z(jnp.sinh(a) * jnp.sin(b)))}[f]


def _convert(x, *, new_dtype, **_):
    if _isc(new_dtype):
        d = _part_dtype(new_dtype)
        return Cx(lax.convert_element_type(_re(x), d), None if _im(x) is None else lax.convert_element_type(x.im, d))
    return lax.convert_element_type(_re(x), new_dtype)  # (complex to real: the real part, as JAX)


def _cpow(x, y):  # x ** y; a zero base as XLA: 1 for y = 0, 0 for a real y > 0, else NaN
    x, y = _full(x), _full(y)
    r = _exp(_cmul(y, _log(x)))
    zero, real = (x.re == 0) & (x.im == 0), y.im == 0
    at0 = jnp.where((y.re == 0) & real, 1.0, jnp.where((y.re > 0) & real, 0.0, jnp.nan))
    return Cx(jnp.where(zero, at0, r.re), jnp.where(zero, jnp.where(jnp.isnan(at0), jnp.nan, 0.0), r.im))


def _ctanh(x):  # tanh(a + ib) = (sinh a cosh a + i sin b cos b) / (sinh^2 a + cos^2 b) (no cancellation near the poles,
    a, b = x.re, _im(x)  # unlike cosh 2a + cos 2b); beyond |a| = 20 it is sign(a) (+-0 i)
    big = jnp.abs(a) > 20.0
    a1 = jnp.where(big, 0.0, a)
    sh, ch, s, c = jnp.sinh(a1), jnp.cosh(a1), jnp.sin(b), jnp.cos(b)
    d = sh * sh + c * c
    return Cx(jnp.where(big, jnp.sign(a), sh * ch / d), jnp.where(big, jnp.copysign(0.0, s * c), s * c / d))


def _full(v):  # a Cx with both parts, also of a real value
    return _dense(v if isinstance(v, Cx) else Cx(v, None))


def _cmp(x, y, eq):
    same = (_re(x) == _re(y)) & (_full(x).im == _full(y).im)
    return same if eq else ~same


def _select_n(p, *cases):
    ims = [_im(c) for c in cases]
    re = lax.select_n(p, *[_re(c) for c in cases])
    return Cx(re, None if all(m is None for m in ims) else lax.select_n(p, *[_dense(c).im for c in cases]))


def _dot(x, y, *, dimension_numbers, precision=None, preferred_element_type=None, **_):
    pet = _part_dtype(preferred_element_type) if _isc(preferred_element_type) \
        else preferred_element_type
    d = lambda a, b: None if a is None or b is None else lax.dot_general(a, b, dimension_numbers, precision=precision,
                                                                         preferred_element_type=pet)
    (p, q), (r, s) = (_re(x), _im(x)), (_re(y), _im(y))
    return Cx(_sub(d(p, r), d(q, s)), _add(d(p, s), d(q, r)))


def _scatter(f):  # scatter / scatter-add per part (lax's function: its update_jaxpr of the real dtype)
    def rule(x, idx, upd, *, dimension_numbers, indices_are_sorted, unique_indices, mode, **_):
        g = lambda a, b: f(a, idx, b, dimension_numbers, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
                           mode=mode)
        if _im(x) is None and _im(upd) is None:
            return Cx(g(_re(x), _re(upd)), None)
        x, upd = _full(x), _full(upd)
        return Cx(g(x.re, upd.re), g(x.im, upd.im))
    return rule


def _twiddles(n, rows, cols, sign, dtype):
    """cos and sign * sin of 2 pi j k / n, j < rows, k < cols: the argument reduced (j k mod n), multiples of pi / 2
    exact (products with them fold)."""
    jk = (np.arange(rows)[:, None] * np.arange(cols)[None, :]) % n
    c, s = np.cos(2 * np.pi * jk / n), np.sin(2 * np.pi * jk / n)
    quarter, r = (4 * jk) % n == 0, (4 * jk // n) % 4
    c, s = np.where(quarter, np.array([1.0, 0.0, -1.0, 0.0])[r], c), np.where(quarter, np.array([0.0, 1.0, 0.0, -1.0])[r], s)
    return c.astype(dtype), (sign * s).astype(dtype)


FFT_DIRECT = 16  # DFTs up to this length: products with the twiddle matrix; longer ones split (Cooley-Tukey)


def _split(n):  # a factor of n for a Cooley-Tukey split (radix 4 first), None for a prime
    return next((f for f in (4, 2, 3, 5, 7) if n % f == 0), None) or next((f for f in range(11, int(n ** 0.5) + 1, 2) if n % f == 0), None)


def _dft_last(p, q, sign):
    """The unscaled DFT of p + i q (q None: real) along the last axis, exp(sign 2 pi i j k / n): (re, im). Long ones
    by the four-step split n = n1 n2: DFTs of length n1 over x[n2 j1 + j2], twiddles, DFTs of length n2."""
    n = p.shape[-1]
    f = _split(n) if n > FFT_DIRECT else None
    mm = lambda a, M: jnp.matmul(a, M, precision=lax.Precision.HIGHEST)
    if f is None:
        C, S = _twiddles(n, n, n, sign, p.dtype)
        return (mm(p, C), mm(p, S)) if q is None else (mm(p, C) - mm(q, S), mm(p, S) + mm(q, C))
    n1, n2, lead = f, n // f, p.shape[:-1]
    cols = lambda a: None if a is None else jnp.swapaxes(a.reshape(lead + (n1, n2)), -1, -2)  # [j2, j1]
    br, bi = _dft_last(cols(p), cols(q), sign)  # [j2, k1]
    C, S = _twiddles(n, n2, n1, sign, p.dtype)  # exp(sign 2 pi i j2 k1 / n)
    cr, ci = _dft_last(jnp.swapaxes(br * C - bi * S, -1, -2), jnp.swapaxes(br * S + bi * C, -1, -2), sign)  # [k1, k2]
    return jnp.swapaxes(cr, -1, -2).reshape(lead + (n,)), jnp.swapaxes(ci, -1, -2).reshape(lead + (n,))  # X[k1 + n1 k2]


def _along(x, axis, M):  # x (along axis) times M, the result's new axis at axis
    return jnp.moveaxis(jnp.tensordot(jnp.moveaxis(x, axis, -1), M, axes=1, precision=lax.Precision.HIGHEST), -1, axis)


def _dft(x, axis, inverse, keep=None):
    """The DFT of the complex x along axis (inverse: conjugate twiddles, divided by n); keep: its first entries only."""
    p, q = jnp.moveaxis(_re(x), axis, -1), None if _im(x) is None else jnp.moveaxis(x.im, axis, -1)
    n = p.shape[-1]
    re, im = _dft_last(p, q, 1.0 if inverse else -1.0)
    if inverse:
        re, im = re * np.asarray(1 / n, p.dtype), im * np.asarray(1 / n, p.dtype)
    return Cx(jnp.moveaxis(re[..., :keep], -1, axis), jnp.moveaxis(im[..., :keep], -1, axis))


def _fft(x, *, fft_type, fft_lengths, **_):
    t, k = int(fft_type), len(fft_lengths)  # 0 FFT, 1 IFFT, 2 RFFT (real in), 3 IRFFT (real out)
    if t == 2:
        y = _dft(x, x.ndim - 1, False, x.shape[-1] // 2 + 1)  # (the unused half: not computed, dead code)
        for a in range(x.ndim - k, x.ndim - 1):
            y = _dft(y, a, False)
        return y
    if t == 3:
        nd = x.re.ndim
        for a in range(nd - k, nd - 1):
            x = _dft(x, a, True)
        n, m = fft_lengths[-1], x.re.shape[-1]
        if n > FFT_DIRECT:  # the Hermitian spectrum completed, its inverse DFT's real part (imaginary parts of X_0 and
            x = _full(x)  # X_n/2 ignored, as XLA)
            ends = (np.arange(m) == 0) | (2 * np.arange(m) == n)
            im = jnp.where(ends, 0.0, x.im)
            back = slice(n - m, 0, -1) if n % 2 == 0 else slice(m - 1, 0, -1)  # X_(n-k) = conj X_k, k = n - m + 1 ...
            re = jnp.concatenate([x.re, x.re[..., back][..., -(n - m):]], -1)
            im = jnp.concatenate([im, -im[..., back][..., -(n - m):]], -1)
            return _dft(Cx(re, im), nd - 1, True).re
        # x_j = sum over the half spectrum k < m of w_k (Re X_k cos - Im X_k sin) / n
        C, S = _twiddles(n, n, m, 1.0, np.float64)  # (C[j, k] = cos 2 pi j k / n; sin exactly 0 at k = 0 and k = n / 2:
        w = np.where((np.arange(m) == 0) | (2 * np.arange(m) == n), 1.0, 2.0)[:, None] / n  # the imaginary parts there
        A, B = (C.T * w).astype(x.re.dtype), (-S.T * w).astype(x.re.dtype)  # ignored)
        out = _along(x.re, nd - 1, A)
        return out if x.im is None else out + _along(x.im, nd - 1, B)
    nd = _re(x).ndim
    for a in range(nd - k, nd):
        x = _dft(x, a, t == 1)
    return x


_INDEX = {"dynamic_slice": 1, "dynamic_update_slice": 2}  # data operands: those before this one (gather: all but 1)
_PARTS = {"reshape", "broadcast_in_dim", "transpose", "squeeze", "slice", "dynamic_slice", "dynamic_update_slice",
          "pad", "concatenate", "gather", "rev", "copy", "reduce_sum", "cumsum", "split", "reduce_precision",
          "optimization_barrier", "stop_gradient", "stack", "unstack", "tile"}


def _while(vals, *, cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts, **_):
    cc, bc, init = vals[:cond_nconsts], vals[cond_nconsts:cond_nconsts + body_nconsts], vals[cond_nconsts + body_nconsts:]
    cond = lambda c: _eval(cond_jaxpr, cond_jaxpr.consts, cc + list(c))[0]
    body = lambda c: tuple(_dense(o) for o in _eval(body_jaxpr, body_jaxpr.consts, bc + list(c)))
    return list(lax.while_loop(cond, body, tuple(_dense(v) for v in init)))


def _scan(vals, *, jaxpr, ft_in, ft_out, length, reverse, unroll, **_):
    consts, init, xs = map(list, ft_in.update(list(vals)).unpack())  # (the operands: constants, carry, xs)

    def body(c, x):
        outs = [_dense(o) for o in _eval(jaxpr, jaxpr.consts, consts + list(c) + list(x))]
        carry, ys = map(tuple, ft_out.update(outs).unpack())
        return carry, ys
    carry, ys = lax.scan(body, tuple(_dense(v) for v in init), tuple(_dense(v) for v in xs), length=length, reverse=reverse,
                         unroll=unroll)
    return list(carry) + list(ys)


def _cond(vals, *, branches, **_):
    fs = [lambda *o, b=b: tuple(_dense(x) for x in _eval(b, b.consts, list(o))) for b in branches]
    return list(lax.switch(vals[0], fs, *[_dense(v) for v in vals[1:]]))


def _complex_code(f, *args):
    """f, JAX code on complex arrays (kernelax.schur's zgees and zgeev), evaluated on the Cx args: its outputs."""
    dt = lambda v: np.complex64 if _re(v).dtype == np.float32 else np.complex128
    closed = jax.make_jaxpr(f)(*[jax.ShapeDtypeStruct(jnp.shape(_re(v)), dt(v)) for v in args])
    return _eval(closed, closed.consts, list(args))


def _eig(a, *, compute_left_eigenvectors, compute_right_eigenvectors, **_):
    """The eigenvalues of a real matrix (schur.eigenvalues: dlahqr's order) or a complex one (zlahqr's, also of real
    values: as XLA's zgeev), and the left and right eigenvectors asked for (schur.eig_vectors: as dgeev's up to their
    signs, as zgeev's)."""
    from kernelax.schur import eig_vectors, eigenvalues
    left, right = compute_left_eigenvectors, compute_right_eigenvectors
    f = (lambda x: eig_vectors(x, left, right)) if left or right else eigenvalues
    out = _complex_code(f, a) if isinstance(a, Cx) else f(a)
    return [Cx(out[q], out[q + 1]) for q in range(0, len(out), 2)]


def _schur(a, *, compute_schur_vectors=True, **params):
    """The Schur form of a complex matrix (also of real values: XLA's zgees, its eigenvalues in zlahqr's order) by
    schur.schur's zgees; as LAPACK's up to the phase of each column of Z (rounding errors decide LAPACK's)."""
    from kernelax.schur import schur
    return _complex_code(lambda x: schur(x, compute_schur_vectors=compute_schur_vectors, **params), a)


def _sort(*ops, num_keys, **params):
    """sort with complex operands: each as its two parts (complex keys: lexicographic by real, then imaginary part, as
    XLA's comparator)."""
    parts, keys = [], 0
    for k, o in enumerate(ops):
        o = _full(o) if isinstance(o, Cx) else o
        parts += [o.re, o.im] if isinstance(o, Cx) else [o]
        keys += (2 if isinstance(o, Cx) else 1) if k < num_keys else 0
    outs, it = iter(lax.sort_p.bind(*parts, num_keys=keys, **params)), iter(ops)
    return [Cx(next(outs), next(outs)) if isinstance(o, Cx) else next(outs) for o in it]


def _batch(f, *xs):  # f on each matrix of xs (Cx, leading batch axes alike), the results stacked back
    lead = xs[0].re.shape[:-2]
    flat = [Cx(x.re.reshape((-1,) + x.re.shape[-2:]), x.im.reshape((-1,) + x.im.shape[-2:])) for x in xs]
    outs = [f(*[Cx(x.re[b], x.im[b]) for x in flat]) for b in range(flat[0].re.shape[0])]
    stack = lambda vals: jnp.stack(vals).reshape(lead + vals[0].shape)
    return [Cx(stack([o[k].re for o in outs]), stack([o[k].im for o in outs])) if isinstance(outs[0][k], Cx) else
            stack([o[k] for o in outs]) for k in range(len(outs[0]))]


def _lu(a):
    """lu of a complex matrix as LAPACK's zgetf2: the pivot the first largest |re| + |im| (izamax), whole rows swapped,
    the column below times the pivot's reciprocal (none if it is 0), then the trailing update; (lu, pivots,
    permutation)."""
    def one(x):
        R, I = x.re, x.im
        m, n = R.shape
        rows, perm, pivs = jnp.arange(m)[:, None], jnp.arange(m, dtype=jnp.int32), []
        for j in range(min(m, n)):
            p = j + jnp.argmax(jnp.abs(R[j:, j]) + jnp.abs(I[j:, j])).astype(jnp.int32)
            pivs.append(p)
            swap = lambda M: jnp.where(rows == j, jnp.take(M, p, axis=0), jnp.where(rows == p, M[j], M))
            R, I = swap(R), swap(I)
            perm = jnp.where(jnp.arange(m) == j, perm[p], jnp.where(jnp.arange(m) == p, perm[j], perm))
            nz = (R[j, j] != 0) | (I[j, j] != 0)
            rcp = _cdiv(Cx(jnp.ones_like(R[j, j]), None), Cx(jnp.where(nz, R[j, j], 1.0), jnp.where(nz, I[j, j], 0.0)))
            col = _cmul(Cx(R[j + 1:, j], I[j + 1:, j]), rcp)
            cr, ci = jnp.where(nz, col.re, R[j + 1:, j]), jnp.where(nz, col.im, I[j + 1:, j])
            R, I = R.at[j + 1:, j].set(cr), I.at[j + 1:, j].set(ci)
            ur, ui = R[j, j + 1:], I[j, j + 1:]
            R = R.at[j + 1:, j + 1:].add(-(jnp.outer(cr, ur) - jnp.outer(ci, ui)))
            I = I.at[j + 1:, j + 1:].add(-(jnp.outer(cr, ui) + jnp.outer(ci, ur)))
        return Cx(R, I), jnp.stack(pivs) if pivs else jnp.zeros(0, jnp.int32), perm
    return _batch(one, _full(a))


def _triangular_solve(a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal, **_):
    """op(a) x = b (left_side) or x op(a) = b, op(a) = a, a^T, conj(a) or a^H, by substitution, reading a's triangle."""
    def one(a, b):
        R, I = a.re, -a.im if conjugate_a else a.im
        lo = lower
        if transpose_a:
            R, I, lo = R.T, I.T, not lo
        Br, Bi = (b.re, b.im) if left_side else (b.re.T, b.im.T)
        if not left_side:  # x op(a) = b  <=>  op(a)^T x^T = b^T
            R, I, lo = R.T, I.T, not lo
        n, xr, xi = R.shape[0], [None] * R.shape[0], [None] * R.shape[0]
        for i in (range(n) if lo else reversed(range(n))):
            js = list(range(i)) if lo else list(range(i + 1, n))
            sr, si = Br[i], Bi[i]
            for j in js:
                sr, si = sr - (R[i, j] * xr[j] - I[i, j] * xi[j]), si - (R[i, j] * xi[j] + I[i, j] * xr[j])
            q = Cx(sr, si) if unit_diagonal else _cdiv(Cx(sr, si), Cx(R[i, i], I[i, i]))
            xr[i], xi[i] = q.re, q.im
        Xr, Xi = jnp.stack(xr), jnp.stack(xi)
        return (Cx(Xr, Xi) if left_side else Cx(Xr.T, Xi.T),)
    return _batch(one, _full(a), _full(b))


def _reduce_prod(x, *, axes, **_):  # the product over axes, one factor after the other
    x = _full(x)
    keep = [d for d in range(x.re.ndim) if d not in axes]
    move = lambda a: jnp.transpose(a, keep + list(axes)).reshape([a.shape[d] for d in keep] + [-1])
    R, I = move(x.re), move(x.im)
    out = Cx(jnp.ones_like(R[..., 0]), jnp.zeros_like(I[..., 0]))
    for k in range(R.shape[-1]):
        out = _cmul(out, Cx(R[..., k], I[..., k]))
    return out


def _linear_solve(vals, *, const_lengths, jaxprs, **_):  # custom_linear_solve: its solve
    n, solve = const_lengths, jaxprs.solve
    return _eval(solve, solve.consts, list(vals[n.matvec + n.vecmat:][:n.solve]) + list(vals[sum(n):]))


_CONTROL = {"while": _while, "scan": _scan, "cond": _cond, "custom_linear_solve": _linear_solve}
_P = lambda f: lambda *a, **_: f(*a)  # (a rule without parameters: accuracy and the like ignored)
_RULES = {
    "add": _P(lambda x, y: Cx(_re(x) + _re(y), _add(_im(x), _im(y)))), "add_any": _P(lambda x, y: Cx(_re(x) + _re(y), _add(_im(x), _im(y)))),
    "sub": _P(lambda x, y: Cx(_re(x) - _re(y), _sub(_im(x), _im(y)))), "mul": _P(_cmul), "div": _P(_cdiv),
    "neg": _P(lambda x: Cx(-x.re, _neg(x.im))), "conj": _P(lambda x: Cx(_re(x), _neg(_im(x)))), "real": _P(lambda x: x.re),
    "imag": _P(lambda x: jnp.zeros_like(x.re) if x.im is None else x.im), "complex": _P(lambda x, y: Cx(x, y)), "abs": _P(_abs),
    "sign": _P(lambda x: _cdiv(x, Cx(jnp.where(_abs(x) == 0, 1.0, _abs(x)), None))), "exp": _P(_exp), "log": _P(_log),
    "sqrt": _P(_sqrt), "rsqrt": _P(lambda x: _cdiv(Cx(jnp.ones_like(x.re), None), _sqrt(x))), "integer_pow": lambda x, *, y, **_: _ipow(x, y),
    "square": _P(lambda x: _cmul(x, x)), "pow": _P(_cpow),
    "sin": _P(lambda x: _trig(x, "sin")), "cos": _P(lambda x: _trig(x, "cos")), "sinh": _P(lambda x: _trig(x, "sinh")),
    "cosh": _P(lambda x: _trig(x, "cosh")), "tan": _P(lambda x: (lambda t: Cx(t.im, -t.re))(_ctanh(Cx(_neg(_im(x)), x.re)))),
    "tanh": _P(_ctanh), "eq": _P(lambda x, y: _cmp(x, y, True)),
    "ne": _P(lambda x, y: _cmp(x, y, False)), "convert_element_type": _convert, "select_n": _P(_select_n), "dot_general": _dot,
    "scatter-add": _scatter(lax.scatter_add), "scatter": _scatter(lax.scatter), "fft": _fft, "eig": _eig, "sort": _sort, "schur": _schur, "lu": _P(_lu), "triangular_solve": _triangular_solve, "reduce_prod": _reduce_prod,
}
