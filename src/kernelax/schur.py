# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""The real Schur decomposition as LAPACK's dgees computes it, without balancing: Householder reduction to Hessenberg
form (dgehd2, its Q as dorghr), then the double-shift QR algorithm of dlahqr (its deflation test, exceptional shifts
and bulge chase; 2 x 2 blocks standardized by dlanv2), so that the eigenvalues come in LAPACK's order. A JAX function
of one while loop over deflations and sweeps, its active window (l, i) dynamic, every access at static positions under
masks (the frontend traces it into a kernel loop). Complex matrices as zgees and zgeev (zgehd2, zlahqr's single-shift
sweeps, zgebal, ztrevc3): complex JAX code, which realify evaluates as pairs of real values."""

import jax.numpy as jnp
import numpy as np
from jax import lax


def _lapy2(x, y):  # sqrt(x^2 + y^2) without overflow
    w, z = jnp.maximum(jnp.abs(x), jnp.abs(y)), jnp.minimum(jnp.abs(x), jnp.abs(y))
    return jnp.where(z == 0, w, w * jnp.sqrt(1 + (z / jnp.where(w == 0, 1, w)) ** 2))


def _csqrt(z):  # Fortran's (C99's) complex SQRT: z - 0j gives the conjugate root (jnp.sqrt, as XLA, ignores the sign)
    r = jnp.sqrt(z)
    return jnp.where((jnp.imag(z) == 0) & jnp.signbit(jnp.imag(z)), jnp.conj(r), r)


def _sign(a, b):  # Fortran's SIGN(a, b) (gfortran: the sign bit of b)
    return jnp.copysign(jnp.abs(a), b)


def _larfg(alpha, x):
    """dlarfg: (beta, tau, v) with (I - tau [1 v] [1 v]^T) [alpha x] = [beta 0]; tau = 0 if x is zero."""
    xnorm = jnp.sqrt(sum(xi * xi for xi in x))
    zero = xnorm == 0
    beta = -_sign(_lapy2(alpha, xnorm), alpha)
    tau = jnp.where(zero, 0.0, (beta - alpha) / jnp.where(zero, 1.0, beta))
    scale = 1.0 / jnp.where(zero, 1.0, alpha - beta)
    return jnp.where(zero, alpha, beta), tau, [jnp.where(zero, xi, xi * scale) for xi in x]


def _hessenberg(a, wantz):
    """dgehd2 (and dorghr): H = Q^T a Q upper Hessenberg, Q (identity unless wantz)."""
    n = len(a)
    A, refl = [list(r) for r in a], []
    for i in range(n - 2):
        beta, tau, v = _larfg(A[i + 1][i], [A[r][i] for r in range(i + 2, n)])
        v = [1.0] + v  # rows i + 1 .. n - 1
        for r in range(n):  # from the right: A[:, i+1:] -= tau (A v) v^T
            w = tau * sum(A[r][i + 1 + t] * v[t] for t in range(len(v)))
            for t in range(len(v)):
                A[r][i + 1 + t] = A[r][i + 1 + t] - w * v[t]
        for c in range(i + 1, n):  # from the left: A[i+1:, i+1:] -= tau v (v^T A)
            w = tau * sum(v[t] * A[i + 1 + t][c] for t in range(len(v)))
            for t in range(len(v)):
                A[i + 1 + t][c] = A[i + 1 + t][c] - v[t] * w
        A[i + 1][i] = beta
        for r in range(i + 2, n):
            A[r][i] = jnp.zeros_like(beta)
        refl.append((v, tau))
    Q = [[float(r == c) for c in range(n)] for r in range(n)]
    if wantz:
        for i in reversed(range(n - 2)):  # Q = H_0 H_1 ... (from the left, the last first)
            v, tau = refl[i]
            for c in range(n):
                w = tau * sum(v[t] * Q[i + 1 + t][c] for t in range(len(v)))
                for t in range(len(v)):
                    Q[i + 1 + t][c] = Q[i + 1 + t][c] - v[t] * w
    return A, Q


def _lanv2(a, b, c, d, eps):
    """dlanv2: the standardized Schur factorization of [[a, b], [c, d]] (without its rescaling of extreme
    magnitudes): (a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn)."""
    zero, one = jnp.zeros_like(a), jnp.ones_like(a)
    # case c == 0: nothing; case b == 0: swap rows and columns; case a == d with b, c of opposite signs: nothing
    c0, b0 = c == 0, (c != 0) & (b == 0)
    std = (c != 0) & (b != 0) & (a - d == 0) & (_sign(one, b) != _sign(one, c))
    gen = ~(c0 | b0 | std)
    temp = a - d
    p = 0.5 * temp
    bcmax = jnp.maximum(jnp.abs(b), jnp.abs(c))
    bcmis = jnp.minimum(jnp.abs(b), jnp.abs(c)) * _sign(one, b) * _sign(one, c)
    scale = jnp.maximum(jnp.abs(p), bcmax)
    sc = jnp.where(scale == 0, 1.0, scale)
    z = (p / sc) * p + (bcmax / sc) * bcmis
    real = z >= 4.0 * eps
    # real eigenvalues
    zr = p + _sign(jnp.sqrt(sc) * jnp.sqrt(jnp.where(real, z, 0.0)), p)
    zr_ = jnp.where(zr == 0, 1.0, zr)
    r_a, r_d = d + zr, d - (bcmax / zr_) * bcmis
    tau = _lapy2(c, zr)
    tau_ = jnp.where(tau == 0, 1.0, tau)
    r_cs, r_sn, r_b, r_c = zr / tau_, c / tau_, b - c, zero
    # complex or nearly equal real eigenvalues: equal diagonal entries
    sigma = b + c
    tau2 = _lapy2(sigma, temp)
    tau2_ = jnp.where(tau2 == 0, 1.0, tau2)
    cs = jnp.sqrt(0.5 * (1.0 + jnp.abs(sigma) / tau2_))
    cs_ = jnp.where(cs == 0, 1.0, cs)
    sn = -(p / (tau2_ * cs_)) * _sign(one, sigma)
    aa, bb, cc, dd = a * cs + b * sn, -a * sn + b * cs, c * cs + d * sn, -c * sn + d * cs
    a2, b2, c2, d2 = aa * cs + cc * sn, bb * cs + dd * sn, -aa * sn + cc * cs, -bb * sn + dd * cs
    t2 = 0.5 * (a2 + d2)
    a2 = d2 = t2
    split = (c2 != 0) & (b2 != 0) & (_sign(one, b2) == _sign(one, c2))  # real: upper triangular
    sab, sac = jnp.sqrt(jnp.abs(b2)), jnp.sqrt(jnp.abs(c2))
    pp = _sign(sab * sac, c2)
    tau3 = 1.0 / jnp.sqrt(jnp.where(split, jnp.abs(b2 + c2), 1.0))
    cs1, sn1 = sab * tau3, sac * tau3
    s_a, s_d, s_b = t2 + pp, t2 - pp, b2 - c2
    s_cs, s_sn = cs * cs1 - sn * sn1, cs * sn1 + sn * cs1
    flip = (c2 != 0) & (b2 == 0)  # b = -c, c = 0, (cs, sn) = (-sn, cs)
    c_a = jnp.where(split, s_a, a2)
    c_d = jnp.where(split, s_d, d2)
    c_b = jnp.where(split, s_b, jnp.where(flip, -c2, b2))
    c_c = jnp.where(split | flip, zero, c2)
    c_cs = jnp.where(split, s_cs, jnp.where(flip, -sn, cs))
    c_sn = jnp.where(split, s_sn, jnp.where(flip, cs, sn))
    A = jnp.where(gen, jnp.where(real, r_a, c_a), jnp.where(b0, d, a))
    D = jnp.where(gen, jnp.where(real, r_d, c_d), jnp.where(b0, a, d))
    B = jnp.where(gen, jnp.where(real, r_b, c_b), jnp.where(b0, -c, b))
    C = jnp.where(gen, jnp.where(real, r_c, c_c), jnp.where(b0, zero, c))
    CS = jnp.where(gen, jnp.where(real, r_cs, c_cs), jnp.where(b0, zero, one))
    SN = jnp.where(gen, jnp.where(real, r_sn, c_sn), jnp.where(b0, one, zero))
    im = jnp.where(C == 0, zero, jnp.sqrt(jnp.abs(B)) * jnp.sqrt(jnp.abs(C)))
    return A, B, C, D, A, im, D, -im, CS, SN


def _get(M, i, j=None):  # M[i][j] (M[i][i] if j is None) at dynamic i (and j: offsets of i)
    n = len(M)
    out = 0.0
    for r in range(n):
        c = r if j is None else r + j
        if 0 <= c < n:
            out = jnp.where(i == r, M[r][c], out)
    return out


def _hqr(H, Z, wantz, nh):
    """dlahqr with wantt (all of T updated) on the Hessenberg rows H (lists of scalars; nh: the size of dgebal's window):
    (T, Z, wr, wi, ok)."""
    n = len(H)
    dt = H[0][0].dtype
    ulp, safmin = np.finfo(dt).eps, np.finfo(dt).tiny
    smlnum, itmax = safmin * (nh.astype(dt) / ulp), 30 * jnp.maximum(10, nh)
    stack = lambda M: jnp.stack([jnp.stack([jnp.asarray(x, dt) for x in r]) for r in M])
    rows = lambda M: [[M[r, c] for c in range(n)] for r in range(n)]

    def body(carry):
        Hm, Zm, wr, wi, i, its, kdefl = carry
        h, zz = rows(Hm), rows(Zm)
        l = jnp.asarray(0, i.dtype)
        for k in range(1, n):  # the largest k <= i with a negligible subdiagonal entry h[k][k-1]
            hkk1 = jnp.abs(h[k][k - 1])
            tst = jnp.abs(h[k - 1][k - 1]) + jnp.abs(h[k][k])
            extra = (jnp.abs(h[k - 1][k - 2]) if k - 2 >= 0 else 0.0) + (jnp.abs(h[k + 1][k]) if k + 1 <= n - 1 else 0.0)
            tst = jnp.where(tst == 0, tst + extra, tst)
            ab, ba = jnp.maximum(hkk1, jnp.abs(h[k - 1][k])), jnp.minimum(hkk1, jnp.abs(h[k - 1][k]))
            dk = jnp.abs(h[k - 1][k - 1] - h[k][k])
            aa, bb = jnp.maximum(jnp.abs(h[k][k]), dk), jnp.minimum(jnp.abs(h[k][k]), dk)
            s = aa + ab
            s_ = jnp.where(s == 0, 1.0, s)
            small = (hkk1 <= smlnum) | ((hkk1 <= ulp * tst) & (ba * (ab / s_) <= jnp.maximum(smlnum, ulp * (bb * (aa / s_)))))
            l = jnp.where((k <= i) & small, k, l)
        for k in range(1, n):  # (H(l, l-1) = 0)
            h[k][k - 1] = jnp.where(l == k, 0.0, h[k][k - 1])
        deflate = l >= i - 1
        # --- a sweep (l < i - 1): shifts, then the bulge chase from m to i - 1
        kd = kdefl + 1
        hii, hi1i1, hii1, hi1i = _get(h, i), _get(h, i - 1), _get(h, i, -1), _get(h, i - 1, 1)
        s1 = jnp.abs(hii1) + jnp.abs(_get(h, i - 1, -1))
        s2 = jnp.abs(_get(h, l + 1, -1)) + jnp.abs(_get(h, l + 2, -1))
        exc1, exc2 = kd % 20 == 0, (kd % 20 != 0) & (kd % 10 == 0)
        h11 = jnp.where(exc1, 0.75 * s1 + hii, jnp.where(exc2, 0.75 * s2 + _get(h, l), hi1i1))
        h12 = jnp.where(exc1, -0.4375 * s1, jnp.where(exc2, -0.4375 * s2, hi1i))
        h21 = jnp.where(exc1, s1, jnp.where(exc2, s2, hii1))
        h22 = jnp.where(exc1 | exc2, h11, hii)
        s = jnp.abs(h11) + jnp.abs(h12) + jnp.abs(h21) + jnp.abs(h22)
        s0, s_ = s == 0, jnp.where(s == 0, 1.0, s)
        h11, h12, h21, h22 = h11 / s_, h12 / s_, h21 / s_, h22 / s_
        tr = (h11 + h22) / 2.0
        det = (h11 - tr) * (h22 - tr) - h12 * h21
        rtdisc = jnp.sqrt(jnp.abs(det))
        cpx = det >= 0
        rt1, rt2 = tr + rtdisc, tr - rtdisc
        rr = jnp.where(jnp.abs(rt1 - h22) <= jnp.abs(rt2 - h22), rt1 * s_, rt2 * s_)
        rt1r = jnp.where(s0, 0.0, jnp.where(cpx, tr * s_, rr))
        rt2r = rt1r
        rt1i = jnp.where(s0 | ~cpx, 0.0, rtdisc * s_)
        rt2i = -rt1i
        m, v1, v2, v3 = l, 0.0, 0.0, 0.0
        for mm in range(n - 2):  # the largest m in [l, i - 2] with two consecutive small subdiagonal entries (or l)
            h21s = h[mm + 1][mm]
            sm = jnp.abs(h[mm][mm] - rt2r) + jnp.abs(rt2i) + jnp.abs(h21s)
            sm = jnp.where(sm == 0, 1.0, sm)
            h21s = h21s / sm
            a1 = h21s * h[mm][mm + 1] + (h[mm][mm] - rt1r) * ((h[mm][mm] - rt2r) / sm) - rt1i * (rt2i / sm)
            a2 = h21s * (h[mm][mm] + h[mm + 1][mm + 1] - rt1r - rt2r)
            a3 = h21s * h[mm + 2][mm + 1]
            sv = jnp.abs(a1) + jnp.abs(a2) + jnp.abs(a3)
            sv = jnp.where(sv == 0, 1.0, sv)
            a1, a2, a3 = a1 / sv, a2 / sv, a3 / sv
            if mm > 0:
                h00 = jnp.abs(h[mm][mm - 1]) * (jnp.abs(a2) + jnp.abs(a3))
                h01 = jnp.abs(a1) * (jnp.abs(h[mm - 1][mm - 1]) + jnp.abs(h[mm][mm]) + jnp.abs(h[mm + 1][mm + 1]))
                ok = (mm == l) | (h00 <= ulp * h01)
            else:
                ok = mm == l
            take = ok & (mm >= l) & (mm <= i - 2)
            m, v1, v2, v3 = (jnp.where(take, mm, m), jnp.where(take, a1, v1), jnp.where(take, a2, v2), jnp.where(take, a3, v3))
        g = [list(r) for r in h]
        gz = [list(r) for r in zz]
        for k in range(n - 1):
            act = ~deflate & (k >= m) & (k <= i - 1)
            nr3 = k <= i - 2
            first = k == m
            x1 = v1 if k == 0 else jnp.where(first, v1, g[k][k - 1])
            x2 = v2 if k == 0 else jnp.where(first, v2, g[k + 1][k - 1])
            x3 = (v3 if k == 0 else jnp.where(first, v3, g[k + 2][k - 1])) if k + 2 <= n - 1 else 0.0
            x3 = jnp.where(nr3, x3, 0.0)
            beta, t1, (w2, w3) = _larfg(x1, [x2, x3])
            if k > 0:
                g[k][k - 1] = jnp.where(act, jnp.where(first, jnp.where(m > l, g[k][k - 1] * (1.0 - t1), g[k][k - 1]), beta), g[k][k - 1])
                g[k + 1][k - 1] = jnp.where(act & ~first, 0.0, g[k + 1][k - 1])
                if k + 2 <= n - 1:
                    g[k + 2][k - 1] = jnp.where(act & ~first & nr3, 0.0, g[k + 2][k - 1])
            t2, t3 = t1 * w2, t1 * w3
            r3 = k + 2 <= n - 1
            for j in range(k, n):  # rows k .. k+2, columns k .. n-1
                sm = g[k][j] + w2 * g[k + 1][j] + (w3 * g[k + 2][j] if r3 else 0.0)
                g[k][j] = jnp.where(act, g[k][j] - sm * t1, g[k][j])
                g[k + 1][j] = jnp.where(act, g[k + 1][j] - sm * t2, g[k + 1][j])
                if r3:
                    g[k + 2][j] = jnp.where(act & nr3, g[k + 2][j] - sm * t3, g[k + 2][j])
            for j in range(min(k + 3, n - 1) + 1):  # columns k .. k+2, rows 0 .. min(k+3, i)
                sm = g[j][k] + w2 * g[j][k + 1] + (w3 * g[j][k + 2] if r3 else 0.0)
                ok = act & (j <= i)
                g[j][k] = jnp.where(ok, g[j][k] - sm * t1, g[j][k])
                g[j][k + 1] = jnp.where(ok, g[j][k + 1] - sm * t2, g[j][k + 1])
                if r3:
                    g[j][k + 2] = jnp.where(ok & nr3, g[j][k + 2] - sm * t3, g[j][k + 2])
            if wantz:
                for j in range(n):
                    sm = gz[j][k] + w2 * gz[j][k + 1] + (w3 * gz[j][k + 2] if r3 else 0.0)
                    gz[j][k] = jnp.where(act, gz[j][k] - sm * t1, gz[j][k])
                    gz[j][k + 1] = jnp.where(act, gz[j][k + 1] - sm * t2, gz[j][k + 1])
                    if r3:
                        gz[j][k + 2] = jnp.where(act & nr3, gz[j][k + 2] - sm * t3, gz[j][k + 2])
        # --- a deflation (l >= i - 1): one eigenvalue, or a 2 x 2 block standardized (dlanv2) and its rotation applied
        two = deflate & (l == i - 1)
        a_, b_, c_, d_, r1r, r1i, r2r, r2i, cs, sn = _lanv2(hi1i1, hi1i, hii1, hii, ulp)
        for r in range(n):
            for c in range(n):
                at = lambda di, dj: (i - 1 + di == r) & (i - 1 + dj == c)
                val = jnp.where(at(0, 0), a_, jnp.where(at(0, 1), b_, jnp.where(at(1, 0), c_, d_)))
                blk = two & (at(0, 0) | at(0, 1) | at(1, 0) | at(1, 1))
                g[r][c] = jnp.where(blk, val, g[r][c])
        h2 = [list(r) for r in g]
        for r in range(n):  # rows i-1, i at columns > i; columns i-1, i at rows < i-1
            for c in range(n):
                row_i1, row_i, col_i1, col_i = (r == i - 1) & (c > i), (r == i) & (c > i), (c == i - 1) & (r < i - 1), (c == i) & (r < i - 1)
                below = g[r + 1][c] if r + 1 < n else 0.0
                above = g[r - 1][c] if r > 0 else 0.0
                right = g[r][c + 1] if c + 1 < n else 0.0
                left = g[r][c - 1] if c > 0 else 0.0
                val = jnp.where(row_i1, cs * g[r][c] + sn * below, jnp.where(row_i, cs * g[r][c] - sn * above,
                                jnp.where(col_i1, cs * g[r][c] + sn * right, cs * g[r][c] - sn * left)))
                h2[r][c] = jnp.where(two & (row_i1 | row_i | col_i1 | col_i), val, g[r][c])
        if wantz:
            z2 = [list(r) for r in gz]
            for r in range(n):
                for c in range(n):
                    right = gz[r][c + 1] if c + 1 < n else 0.0
                    left = gz[r][c - 1] if c > 0 else 0.0
                    val = jnp.where(c == i - 1, cs * gz[r][c] + sn * right, cs * gz[r][c] - sn * left)
                    z2[r][c] = jnp.where(two & ((c == i - 1) | (c == i)), val, gz[r][c])
            gz = z2
        one_ = deflate & (l == i)
        idx = jnp.arange(n)
        wr = jnp.where(one_ & (idx == i), hii, jnp.where(two & (idx == i - 1), r1r, jnp.where(two & (idx == i), r2r, wr)))
        wi = jnp.where(one_ & (idx == i), 0.0, jnp.where(two & (idx == i - 1), r1i, jnp.where(two & (idx == i), r2i, wi)))
        i_new = jnp.where(deflate, l - 1, i)
        its_new = jnp.where(deflate, 0, its + 1)
        kd_new = jnp.where(deflate, 0, kd)
        return stack(h2), stack(gz), wr, wi, i_new, its_new, kd_new

    i0 = jnp.asarray(n - 1, jnp.int32)
    zero = jnp.asarray(0, jnp.int32)
    carry = (stack(H), stack(Z), jnp.zeros(n, dt), jnp.zeros(n, dt), i0, zero, zero)
    Hm, Zm, wr, wi, i, _, _ = lax.while_loop(lambda c: (c[4] >= 0) & (c[5] <= itmax), body, carry)
    return Hm, Zm, wr, wi, i < 0


def schur(a, *, compute_schur_vectors=True, sort_eig_vals=False, select_callable=None, **_):
    """lax.linalg.schur of a real or complex matrix (leading batch axes too): (T, Z), NaN if dlahqr (zlahqr) did not
    converge."""
    if sort_eig_vals or select_callable is not None:
        raise NotImplementedError("kernelax: schur with sorted eigenvalues is not supported")
    T, Z, _, _, ok, _, f = _batched(a, compute_schur_vectors)
    T = T / f[..., None, None]  # (dgees: T scaled back)
    nan = lambda M: jnp.where(ok[..., None, None], M, jnp.nan)
    return (nan(T), nan(Z)) if compute_schur_vectors else (nan(T),)


def eigenvalues(a):
    """The eigenvalues of a real or complex matrix (leading batch axes too) as dgeev's (zgeev's): (wr, wi)."""
    _, _, wr, wi, ok, _, _ = _batched(a, False, True)
    return jnp.where(ok[..., None], wr, jnp.nan), jnp.where(ok[..., None], wi, jnp.nan)


def _sq(x):  # |x|^2
    return jnp.real(x) ** 2 + jnp.imag(x) ** 2 if jnp.iscomplexobj(x) else x * x


def _amax(xs):  # |x| at LAPACK's i(d|z)amax: the first largest |x| (complex: the largest |re| + |im|)
    if not jnp.iscomplexobj(xs[0]):
        return jnp.max(jnp.abs(jnp.stack(xs)))
    best, at = _cabs1(xs[0]), xs[0]
    for x in xs[1:]:
        take = _cabs1(x) > best
        best, at = jnp.where(take, _cabs1(x), best), jnp.where(take, x, at)
    return jnp.abs(at)


def _sel(xs, j):  # xs[j] at a dynamic j
    out = xs[0]
    for p in range(1, len(xs)):
        out = jnp.where(j == p, xs[p], out)
    return out


def _exchange(m, i, j, on):
    """The rows m with rows i and j, then columns i and j exchanged where on (i static, j dynamic)."""
    n = len(m)
    rj = [_sel([m[r][c] for r in range(n)], j) for c in range(n)]
    m = [[jnp.where(on & (r == j), m[i][c], jnp.where(on & (r == i), rj[c], m[r][c])) for c in range(n)] for r in range(n)]
    cj = [_sel(m[r], j) for r in range(n)]
    return [[jnp.where(on & (c == j), m[r][i], jnp.where(on & (c == i), cj[r], m[r][c])) for c in range(n)] for r in range(n)]


def _balance(A, scale):
    """dgebal (zgebal) of LAPACK 3.12, job "B" (scale) or "P": the permutations that isolate eigenvalues (rows with
    zeros off the diagonal pushed down, then such columns left; the rows and columns ilo..ihi between remain), then on
    that window D^-1 A D with D a diagonal of powers of 2 that brings the norms of each row and column close. Returns
    (rows of the result, D's diagonal (1 outside the window), dgebak's row order (A's vectors: the result's, times D,
    at these rows), ihi - ilo + 1). The result is block upper triangular: dgehrd and dhseqr with ilo, ihi take the
    same steps as without (those outside the window change nothing; its eigenvalues deflate at once)."""
    n, dt = len(A), np.finfo(A[0][0].dtype).dtype
    i32 = lambda v: jnp.asarray(v, jnp.int32)
    stack = lambda M: jnp.stack([jnp.stack(q) for q in M])
    rows = lambda M: [[M[r, c] for c in range(n)] for r in range(n)]
    every = lambda zs: jnp.all(jnp.stack(zs)) if zs else jnp.asarray(True)

    def row_sweep(carry):  # rows isolating an eigenvalue, to l (which moves up); at l = 0 all are isolated (done)
        M, l, rec, _, done = carry
        m, rec, l0, noconv = rows(M), [rec[q] for q in range(n)], l, jnp.asarray(False)
        for i in range(n - 1, -1, -1):  # (i <= l throughout)
            can = (i <= l0) & ~done & every([(m[i][j] == 0) | (j > l) for j in range(n) if j != i])
            rec = [jnp.where(can & (l == q), i32(i), rec[q]) for q in range(n)]
            m = _exchange(m, i, l, can & (l != i))
            noconv, done = noconv | can, done | (can & (l == 0))
            l = jnp.where(can & (l > 0), l - 1, l)
        return stack(m), l, jnp.stack(rec), noconv & ~done, done

    some = lambda m, l: jnp.any(jnp.stack([(i <= l) & every([(m[i][j] == 0) | (j > l) for j in range(n) if j != i])
                                           for i in range(n)]))  # (a row (of m^T: column) to isolate; else no sweep: the
    # first would change nothing)
    M, l, rec, _, done = lax.while_loop(lambda c: c[3], row_sweep, (stack(A), i32(n - 1), jnp.arange(n, dtype=jnp.int32),
                                                                     some(A, n - 1), jnp.asarray(False)))

    def col_sweep(carry):  # columns isolating an eigenvalue, to k (which moves down)
        M, k, rec, _ = carry
        m, rec, k0, noconv = rows(M), [rec[q] for q in range(n)], k, jnp.asarray(False)
        for j in range(n):  # (j >= k throughout)
            can = (j >= k0) & (j <= l) & every([(m[i][j] == 0) | (i < k) | (i > l) for i in range(n) if i != j])
            rec = [jnp.where(can & (k == q), i32(j), rec[q]) for q in range(n)]
            m = _exchange(m, j, k, can & (k != j))
            noconv, k = noconv | can, jnp.where(can, k + 1, k)
        return stack(m), k, jnp.stack(rec), noconv

    M, k, rec, _ = lax.while_loop(lambda c: c[3], col_sweep, (M, i32(0), rec, ~done & some([[M[r, c] for r in range(n)]
                                                                                            for c in range(n)], l)))
    sfmin1 = np.finfo(dt).tiny / np.finfo(dt).eps
    sfmax1, sfmin2 = 1 / sfmin1, sfmin1 * 2
    sfmax2 = 1 / sfmin2
    win = [(q >= k) & (q <= l) for q in range(n)]

    def sweep(carry):  # (outside the window: zeros in the rows and columns scaled)
        M, scale, _ = carry
        m, sc, noconv = rows(M), [scale[i] for i in range(n)], jnp.asarray(False)
        for i in range(n):
            c = jnp.sqrt(sum(jnp.where(win[r], _sq(m[r][i]), 0.0) for r in range(n)))
            r = jnp.sqrt(sum(jnp.where(win[q], _sq(m[i][q]), 0.0) for q in range(n)))
            ca, ra = _amax([m[q][i] for q in range(n)]), _amax([m[i][q] for q in range(n)])
            skip = ~win[i] | (c == 0) | (r == 0) | jnp.isnan(c + ca + r + ra)
            s0 = c + r
            up = lambda v: (v[1] < v[4] / 2) & (jnp.maximum(jnp.maximum(v[0], v[1]), v[2]) < sfmax2) & \
                (jnp.minimum(jnp.minimum(v[3], v[4] / 2), v[5]) > sfmin2)  # (f, c, ca, r, 2 g, ra), g = r / 2 at first
            f, c, ca, r, _, ra = lax.while_loop(lambda v: up(v) & ~skip, lambda v: (v[0] * 2, v[1] * 2, v[2] * 2, v[3] / 2,
                                                                               v[4] / 2, v[5] / 2), (jnp.ones_like(c), c, ca, r, r, ra))
            down = lambda v: (v[1] / 2 >= v[3]) & (jnp.maximum(v[3], v[5]) < sfmax2) & \
                (jnp.minimum(jnp.minimum(v[0], v[1]), jnp.minimum(v[1] / 2, v[2])) > sfmin2)
            f, c, ca, r, _, ra = lax.while_loop(lambda v: down(v) & ~skip, lambda v: (v[0] / 2, v[1] / 2, v[2] / 2, v[3] * 2, v[4],
                                                                                  v[5] * 2), (f, c, ca, r, r, ra))
            go = ~skip & (c + r < 0.95 * s0) & ~((f < 1) & (sc[i] < 1) & (f * sc[i] <= sfmin1)) & \
                ~((f > 1) & (sc[i] > 1) & (sc[i] >= sfmax1 / f))
            f = jnp.where(go, f, 1.0)
            sc[i], noconv = sc[i] * f, noconv | go
            m[i] = [x / f for x in m[i]]  # (powers of 2: exact)
            for q in range(n):
                m[q][i] = m[q][i] * f
        return jnp.stack([jnp.stack(q) for q in m]), jnp.stack(sc), noconv
    D = jnp.ones(n, dt)
    if scale:
        M, D, _ = lax.while_loop(lambda c: c[2], sweep, (M, D, jnp.asarray(True)))
    order, rec = [i32(q) for q in range(n)], [rec[q] for q in range(n)]
    for ii in range(n):  # dgebak: the exchanges undone, rows ilo-1 .. 0, then ihi+1 .. n-1
        i = jnp.where(ii < k, k - 1 - ii, ii)
        on = (ii < k) | (ii > l)
        e = _sel(rec, i)
        oi, oe = _sel(order, i), _sel(order, e)
        order = [jnp.where(on & (q == i), oe, jnp.where(on & (q == e), oi, order[q])) for q in range(n)]
    return rows(M), D, order, l - k + 1


SCALE = [True]  # (A/B: scaled as dgeev)


def _batched(a, wantz, balance=False):
    """Per matrix (leading batch axes): (T, Z, wr, wi, ok, D, f) of dgees' chain (dgebal's permutations; dgeev's: also
    balanced by D), of zgees' (zgeev's) for a complex a. The matrix scaled by f first as dgeev and dgees do (dlascl:
    its largest |entry| into [smlnum, bignum], else f = 1, exact; without, float32 entries of 1e-15 lost
    digits, of 1e-17 gave NaN), the eigenvalues scaled back, T not (the caller's: T / f). T: the Schur form of the
    permuted (balanced) matrix; Z = P D Q and D = P D at A's rows (dgebak)."""
    a = jnp.asarray(a)
    n, rdt = a.shape[-1], np.finfo(a.dtype).dtype
    small = np.sqrt(np.finfo(rdt).tiny) / np.finfo(rdt).eps  # (dgeev's smlnum; bignum = 1 / smlnum)
    hessenberg, hqr = (_zhessenberg, _zhqr) if jnp.iscomplexobj(a) else (_hessenberg, _hqr)
    flat = a.reshape((-1, n, n))
    outs = []
    for b in range(flat.shape[0]):
        if n == 0:
            outs.append((flat[b], flat[b], jnp.zeros(0, rdt), jnp.zeros(0, rdt), jnp.asarray(True), jnp.ones(0, rdt), jnp.ones((), rdt)))
            continue
        anrm = jnp.max(jnp.abs(flat[b]))
        f = jnp.where((anrm > 0) & (anrm < small), small / anrm, jnp.where(anrm > 1 / small, 1 / small / anrm, 1.0)).astype(rdt) \
            if SCALE[0] else jnp.ones((), rdt)
        A, scale, order, nh = _balance([[flat[b, r, c] * f for c in range(n)] for r in range(n)], balance)
        H, Q = hessenberg(A, wantz)
        T, Z, wr, wi, ok = hqr(H, Q, wantz, nh)
        Z = [[Z[r, c] * scale[r] for c in range(n)] for r in range(n)]  # (dgebak: Z's rows by D, then in A's order)
        Z = jnp.stack([jnp.stack([_sel([Z[q][c] for q in range(n)], order[r]) for c in range(n)]) for r in range(n)])
        outs.append((T, Z, wr / f, wi / f, ok, jnp.stack([_sel([scale[q] for q in range(n)], order[r]) for r in range(n)]), f))
    T, Z, wr, wi, ok, D, f = (jnp.stack([o[k] for o in outs]) for k in range(7))
    lead = a.shape[:-2]
    return T.reshape(a.shape), Z.reshape(a.shape), wr.reshape(lead + (n,)), wi.reshape(lead + (n,)), ok.reshape(lead), \
        D.reshape(lead + (n,)), f.reshape(lead)


def rsf2csf(T, Z):
    """The complex Schur form of a real one (as scipy's rsf2csf): each 2 x 2 block (a nonzero subdiagonal entry) made
    upper triangular by a complex Givens rotation, from the bottom; (T.re, T.im, Z.re, Z.im), leading batch axes too."""
    n = T.shape[-1]
    lead = T.shape[:-2]
    Tf, Zf = T.reshape((-1, n, n)), Z.reshape((-1, n, n))
    outs = []
    for b in range(Tf.shape[0]):
        tr = [[Tf[b, r, c] for c in range(n)] for r in range(n)]
        ti = [[jnp.zeros_like(tr[0][0]) for c in range(n)] for r in range(n)]
        zr = [[Zf[b, r, c] for c in range(n)] for r in range(n)]
        zi = [[jnp.zeros_like(tr[0][0]) for c in range(n)] for r in range(n)]
        eps = np.finfo(Tf.dtype).eps
        for m in range(n - 1, 0, -1):
            sub = tr[m][m - 1]  # (real: the steps below m change columns m-1, m of rows <= m only)
            act = jnp.abs(sub) > eps * (jnp.abs(tr[m - 1][m - 1]) + jnp.abs(tr[m][m]))
            a, bb, c, d = tr[m - 1][m - 1], tr[m - 1][m], sub, tr[m][m]
            disc = ((a - d) / 2) ** 2 + bb * c  # (negative in a 2 x 2 block: complex eigenvalues)
            mr, mi = (a - d) / 2, jnp.sqrt(jnp.maximum(-disc, 0.0))  # mu = the first eigenvalue - T[m, m]
            rn = jnp.sqrt(mr * mr + mi * mi + sub * sub)
            rn = jnp.where(act, rn, 1.0)
            cr, ci, s = mr / rn, mi / rn, sub / rn
            for j in range(m - 1, n):  # rows m-1, m: [conj(c) x + s y, -s x + c y]
                xr, xi, yr, yi = tr[m - 1][j], ti[m - 1][j], tr[m][j], ti[m][j]
                nxr, nxi = cr * xr + ci * xi + s * yr, cr * xi - ci * xr + s * yi
                nyr, nyi = -s * xr + cr * yr - ci * yi, -s * xi + cr * yi + ci * yr
                tr[m - 1][j], ti[m - 1][j] = jnp.where(act, nxr, xr), jnp.where(act, nxi, xi)
                tr[m][j], ti[m][j] = jnp.where(act, nyr, yr), jnp.where(act, nyi, yi)
            for M, Mi, rows in ((tr, ti, range(m + 1)), (zr, zi, range(n))):  # columns m-1, m: [x c + y s, -x s + y conj(c)]
                for j in rows:
                    xr, xi, yr, yi = M[j][m - 1], Mi[j][m - 1], M[j][m], Mi[j][m]
                    nxr, nxi = xr * cr - xi * ci + s * yr, xr * ci + xi * cr + s * yi
                    nyr, nyi = -s * xr + yr * cr + yi * ci, -s * xi + yi * cr - yr * ci
                    M[j][m - 1], Mi[j][m - 1] = jnp.where(act, nxr, xr), jnp.where(act, nxi, xi)
                    M[j][m], Mi[j][m] = jnp.where(act, nyr, yr), jnp.where(act, nyi, yi)
            tr[m][m - 1], ti[m][m - 1] = jnp.zeros_like(sub), jnp.zeros_like(sub)
        outs.append([jnp.stack([jnp.stack(r) for r in M]) for M in (tr, ti, zr, zi)])
    return tuple(jnp.stack([o[k] for o in outs]).reshape(lead + (n, n)) for k in range(4))


def eig_vectors(a, left=False, right=True):
    """Eigenvalues and eigenvectors of a real matrix (leading batch axes too) as dgeev returns them, up to the sign of
    each vector: (wr, wi, then (re, im) of the left vectors (u^H a = w u^H) if left, of the right ones if right), the
    vectors as columns, unit 2-norm, their largest entry real (dlartg's rotation); a conjugate pair's vectors exactly
    conjugate. From the complex Schur form (rsf2csf): per eigenvalue a triangular solve with x_k = 1 (right: T, upward;
    left: T^H, downward), back-transformed by Z = P D Q (left: P D^-1 Q, dgebak's). A complex matrix: _zeig_vectors."""
    if jnp.iscomplexobj(a):
        return _zeig_vectors(a, left, right)
    T, Z, wr, wi, ok, D, _ = _batched(a, True, True)  # (T scaled: the vectors are the same)
    tr, ti, zr, zi = rsf2csf(T, Z)
    n = a.shape[-1]
    sides = [side for side, on in (("left", left), ("right", right)) if on]
    flat = lambda x: x.reshape((-1,) + x.shape[len(a.shape) - 2:])
    outs = []
    for b in range(flat(T).shape[0]):
        R, I = [[flat(tr)[b, r, c] for c in range(n)] for r in range(n)], [[flat(ti)[b, r, c] for c in range(n)] for r in range(n)]
        ZR, ZI = [[flat(zr)[b, r, c] for c in range(n)] for r in range(n)], [[flat(zi)[b, r, c] for c in range(n)] for r in range(n)]
        w_i = flat(wi)[b]
        Tb, Zb, Db = flat(T)[b], flat(Z)[b], flat(D)[b]
        fi = np.finfo(np.dtype(a.dtype))  # (at least dtrevc's smlnum: of the zero matrix eps * tiny was subnormal, 0 / 0)
        tiny = jnp.maximum(fi.eps * jnp.max(jnp.abs(flat(T)[b])), fi.tiny * (n / fi.eps))
        out = []
        for side in sides:
            cols = []
            for k in range(n):  # x_k = 1; right: (T - T_kk) x = 0, x_j = -(T_jk + sum_m T_jm x_m) / (T_jj - T_kk), j < k;
                xr, xi = [0.0] * n, [0.0] * n  # left: (T - T_kk)^H x = 0, x_j = -sum_m conj(T_mj) x_m / conj(T_jj - T_kk), j > k
                xr[k], xi[k] = 1.0, 0.0
                for j in (range(k - 1, -1, -1) if side == "right" else range(k + 1, n)):
                    if side == "right":
                        sr = R[j][k] + sum(R[j][m] * xr[m] - I[j][m] * xi[m] for m in range(j + 1, k))
                        si = I[j][k] + sum(R[j][m] * xi[m] + I[j][m] * xr[m] for m in range(j + 1, k))
                        dr, di = R[j][j] - R[k][k], I[j][j] - I[k][k]
                    else:
                        sr = sum(R[m][j] * xr[m] + I[m][j] * xi[m] for m in range(k, j))
                        si = sum(R[m][j] * xi[m] - I[m][j] * xr[m] for m in range(k, j))
                        dr, di = R[j][j] - R[k][k], I[k][k] - I[j][j]
                    small = jnp.abs(dr) + jnp.abs(di) < tiny  # (a repeated eigenvalue: perturbed as dtrevc's smin)
                    dr = jnp.where(small, tiny, dr)
                    sc = 1.0 / jnp.maximum(jnp.abs(dr), jnp.abs(di))  # (scaled as dlaln2: tiny^2 underflowed, 0 / 0)
                    pr, pi_ = dr * sc, di * sc
                    den = (pr * pr + pi_ * pi_) / sc
                    xr[j], xi[j] = -(sr * pr + si * pi_) / den, -(si * pr - sr * pi_) / den
                span = range(k + 1) if side == "right" else range(k, n)
                vr = [sum(ZR[r][j] * xr[j] - ZI[r][j] * xi[j] for j in span) for r in range(n)]
                vi = [sum(ZR[r][j] * xi[j] + ZI[r][j] * xr[j] for j in span) for r in range(n)]
                if side == "left":  # (P D^-1 Q = Z / D^2)
                    vr, vi = [x / Db[r] ** 2 for r, x in enumerate(vr)], [x / Db[r] ** 2 for r, x in enumerate(vi)]
                cols.append((vr, vi, I[k][k]))
            VR, VI = [], []
            for k in range(n):  # the vector of w_k: real; the first of a pair (w_i > 0): T_kk's or its conjugate's
                vr, vi, tk = cols[k]
                flip = (w_i[k] > 0) & (tk < 0)
                vi = [jnp.where(w_i[k] == 0, 0.0, jnp.where(flip, -c_, c_)) for c_ in vi]
                if k + 1 < n:  # the first of a pair: its phase as dtrevc's start in the real Schur basis Q (y = Q^T D^-1 v
                    zw = (lambda r, j: Zb[r, j] / Db[r] ** 2) if side == "right" else (lambda r, j: Zb[r, j])  # = Z^T D^-2 v;
                    yr = [sum(zw(r, j) * vr[r] for r in range(n)) for j in (k, k + 1)]  # left: Z^T v)
                    yi = [sum(zw(r, j) * vi[r] for r in range(n)) for j in (k, k + 1)]
                    first = jnp.abs(Tb[k, k + 1]) >= jnp.abs(Tb[k + 1, k])  # right: y_k = 1 if |T_k,k+1| >= |T_k+1,k|, else
                    if side == "left":  # y_k+1 = i; left: the other way round. The vector's factor c to that; times conj(c)
                        first = ~first  # (a positive factor left)
                    cr, ci = jnp.where(first, yr[0], yi[1]), jnp.where(first, yi[0], -yr[1])
                    cr, ci = jnp.where(w_i[k] > 0, cr, 1.0), jnp.where(w_i[k] > 0, ci, 0.0)
                    vr, vi = [x * cr + y * ci for x, y in zip(vr, vi)], [y * cr - x * ci for x, y in zip(vr, vi)]
                nrm = jnp.sqrt(sum(x * x for x in vr) + sum(x * x for x in vi))  # (dnrm2, dlapy2 of the two parts' norms)
                vr, vi = [x / nrm for x in vr], [x / nrm for x in vi]
                mag = [x * x + y * y for x, y in zip(vr, vi)]  # the largest entry real (idamax: the first maximum; dlartg)
                f, g, best, at = vr[0], vi[0], mag[0], 0
                for i, (x, y, m) in enumerate(zip(vr[1:], vi[1:], mag[1:]), 1):
                    take = m > best
                    f, g, best, at = jnp.where(take, x, f), jnp.where(take, y, g), jnp.where(take, m, best), jnp.where(take, i, at)
                d = jnp.sqrt(f * f + g * g)
                cs = jnp.where(g == 0, 1.0, jnp.abs(f) / d)
                sn = jnp.where(g == 0, 0.0, g / jnp.copysign(d, f))  # (drot)
                vr, vi = [cs * x + sn * y for x, y in zip(vr, vi)], [jnp.where(at == i, 0.0, cs * y - sn * x) for i, (x, y) in enumerate(zip(vr, vi))]
                if k > 0:  # (the second of a pair: the first's, conjugated)
                    vr = [jnp.where(w_i[k] < 0, x, y) for x, y in zip(VR[k - 1], vr)]
                    vi = [jnp.where(w_i[k] < 0, -x, y) for x, y in zip(VI[k - 1], vi)]
                VR.append(vr)
                VI.append(vi)
            out += [jnp.stack([jnp.stack(c) for c in VR], 1), jnp.stack([jnp.stack(c) for c in VI], 1)]
        outs.append(out)
    lead = a.shape[:-2]
    bad = ~ok[..., None, None]
    vecs = [jnp.where(bad, jnp.nan, jnp.stack([o[q] for o in outs]).reshape(lead + (n, n))) for q in range(2 * len(sides))]
    return (jnp.where(ok[..., None], wr, jnp.nan), jnp.where(ok[..., None], wi, jnp.nan), *vecs)


# ------------------------------------------------------------------ complex matrices (LAPACK's zgees, zgeev)
# Complex JAX code: realify evaluates it as pairs of real values (its _schur and _eig rules).

def _zeig_vectors(a, left, right):
    """zgeev's eigenvalues and eigenvectors (wr, wi, then (re, im) of the left vectors if left, of the right ones if
    right): per eigenvalue T_kk ztrevc3's triangular solve with x_k = 1, right (T - T_kk) x = 0 column by column as
    ztrsv, left (T - T_kk)^H x = 0 row by row as ztrsv's conjugate transpose (a diagonal entry within smin of T_kk
    replaced by smin), back-transformed by Z = P D Q (left: P D^-1 Q; zgebak), then unit 2-norm and the first largest
    entry real positive (zgeev). Independent of the Schur form's column phases."""
    T, Z, wr, wi, ok, D, _ = _batched(a, True, True)
    n = a.shape[-1]
    rdt = np.finfo(a.dtype).dtype
    ulp, smlnum = np.finfo(rdt).eps, np.finfo(rdt).tiny * (n / np.finfo(rdt).eps)
    sides = [side for side, on in (("left", left), ("right", right)) if on]
    lead = a.shape[:-2]
    Tf, Zf, Df = T.reshape((-1, n, n)), Z.reshape((-1, n, n)), D.reshape((-1, n))
    outs = []
    for b in range(Tf.shape[0]):
        t, z = [[Tf[b, r, c] for c in range(n)] for r in range(n)], [[Zf[b, r, c] for c in range(n)] for r in range(n)]
        out = []
        for side in sides:
            cols = []
            for k in range(n):
                smin = jnp.maximum(ulp * _cabs1(t[k][k]), smlnum)
                dg = lambda j: (lambda d: jnp.where(_cabs1(d) < smin, smin + 0j, d))(t[j][j] - t[k][k])
                x = [jnp.zeros_like(t[k][k])] * n
                x[k] = jnp.ones_like(t[k][k])
                if side == "right":
                    rhs = [-t[j][k] for j in range(k)]
                    for j in range(k - 1, -1, -1):
                        x[j] = rhs[j] / dg(j)
                        for r in range(j):
                            rhs[r] = rhs[r] - x[j] * t[r][j]
                    v = [sum(z[r][j] * x[j] for j in range(k + 1)) for r in range(n)]
                else:
                    for j in range(k + 1, n):
                        tmp = -jnp.conj(t[k][j])
                        for i in range(k + 1, j):
                            tmp = tmp - jnp.conj(t[i][j]) * x[i]
                        x[j] = tmp / jnp.conj(dg(j))
                    v = [sum(z[r][j] * x[j] for j in range(k, n)) / Df[b, r] ** 2 for r in range(n)]
                remax = 1 / jnp.max(jnp.stack([_cabs1(c) for c in v]))  # (ztrevc3's: no overflow in the 2-norm)
                v = [c * remax for c in v]
                scl = 1 / jnp.sqrt(sum(_sq(c) for c in v))
                v = [c * scl for c in v]
                mag = [_sq(c) for c in v]
                best, at, vk = mag[0], 0, v[0]
                for i in range(1, n):  # (idamax: the first largest)
                    take = mag[i] > best
                    best, at, vk = jnp.where(take, mag[i], best), jnp.where(take, i, at), jnp.where(take, v[i], vk)
                tmp = jnp.conj(vk) / jnp.sqrt(best)
                cols.append([jnp.where(at == i, jnp.real(c * tmp) + 0j, c * tmp) for i, c in enumerate(v)])
            V = jnp.stack([jnp.stack(c) for c in cols], 1)
            out += [jnp.real(V), jnp.imag(V)]
        outs.append(out)
    bad = ~ok[..., None, None]
    vecs = [jnp.where(bad, jnp.nan, jnp.stack([o[q] for o in outs]).reshape(lead + (n, n))) for q in range(2 * len(sides))]
    return (jnp.where(ok[..., None], wr, jnp.nan), jnp.where(ok[..., None], wi, jnp.nan), *vecs)


def _cabs1(z):  # |re| + |im| (LAPACK's CABS1)
    return jnp.abs(jnp.real(z)) + jnp.abs(jnp.imag(z))


def _zlarfg(alpha, x):
    """zlarfg: (beta (real, as complex), tau, v): (I - tau [1 v][1 v]^H)^H [alpha x] = [beta 0]."""
    xnorm = jnp.sqrt(sum(jnp.real(xi) ** 2 + jnp.imag(xi) ** 2 for xi in x)) if x else 0.0 * jnp.real(alpha)
    ar, ai = jnp.real(alpha), jnp.imag(alpha)
    zero = (xnorm == 0) & (ai == 0)
    beta = -_sign(jnp.sqrt(ar * ar + ai * ai + xnorm * xnorm), ar)
    b_ = jnp.where(zero, 1.0, beta)
    tau = jnp.where(zero, 0.0, (beta - ar) / b_ - 1j * ai / b_)
    scale = 1.0 / jnp.where(zero, 1.0, alpha - beta)
    return jnp.where(zero, alpha, beta + 0j), tau, [jnp.where(zero, xi, xi * scale) for xi in x]


def _zhessenberg(a, wantz):
    """zgehd2 (and zunghr): H = Q^H a Q upper Hessenberg with a real subdiagonal, Q (identity unless wantz)."""
    n = len(a)
    A, refl = [list(r) for r in a], []
    for i in range(n - 1):  # (to n - 2: a reflector of one entry still makes the last subdiagonal entry real)
        beta, tau, v = _zlarfg(A[i + 1][i], [A[r][i] for r in range(i + 2, n)])
        v = [1.0 + 0j] + v
        for r in range(n):  # from the right: A[:, i+1:] -= tau (A v) v^H
            w = tau * sum(A[r][i + 1 + t] * v[t] for t in range(len(v)))
            for t in range(len(v)):
                A[r][i + 1 + t] = A[r][i + 1 + t] - w * jnp.conj(v[t])
        for c in range(i + 1, n):  # from the left with conj(tau): A[i+1:, i+1:] -= conj(tau) v (v^H A)
            w = jnp.conj(tau) * sum(jnp.conj(v[t]) * A[i + 1 + t][c] for t in range(len(v)))
            for t in range(len(v)):
                A[i + 1 + t][c] = A[i + 1 + t][c] - v[t] * w
        A[i + 1][i] = beta
        for r in range(i + 2, n):
            A[r][i] = jnp.zeros_like(beta)
        refl.append((v, tau))
    Q = [[(1.0 if r == c else 0.0) + 0j for c in range(n)] for r in range(n)]
    if wantz:
        for i in reversed(range(n - 1)):  # Q = H_0 H_1 ... (from the left, the last first)
            v, tau = refl[i]
            for c in range(n):
                w = tau * sum(jnp.conj(v[t]) * Q[i + 1 + t][c] for t in range(len(v)))
                for t in range(len(v)):
                    Q[i + 1 + t][c] = Q[i + 1 + t][c] - v[t] * w
    return A, Q


def _zhqr(H, Z, wantz, nh):
    """zlahqr with wantt on the complex Hessenberg rows H (lists of scalars, real subdiagonal): (T, Z, wr, wi, ok). Its
    Schur form is LAPACK's up to the phase of each column of Z (the last sweep before a deflation makes H(i, i-1), by
    then of the size of rounding errors, real by a diagonal unitary: LAPACK's phases change under a change of A in the
    last bit)."""
    n = len(H)
    cdt = H[0][0].dtype
    rdt = jnp.real(H[0][0]).dtype
    ulp, safmin = np.finfo(rdt).eps, np.finfo(rdt).tiny
    smlnum, itmax = safmin * (nh.astype(rdt) / ulp), 30 * jnp.maximum(10, nh)
    stack = lambda M: jnp.stack([jnp.stack([jnp.asarray(x, cdt) for x in r]) for r in M])
    rows = lambda M: [[M[r, c] for c in range(n)] for r in range(n)]
    re = jnp.real
    h, zz = [list(r) for r in H], [list(r) for r in Z]
    for i in range(1, n):  # real subdiagonal entries (zgehrd's already are)
        hi = h[i][i - 1]
        nz = jnp.imag(hi) != 0
        sc = hi / jnp.where(nz, _cabs1(hi), 1.0)
        sc = jnp.conj(sc) / jnp.where(nz, jnp.abs(sc), 1.0)
        sc = jnp.where(nz, sc, 1.0 + 0j)
        h[i][i - 1] = jnp.where(nz, jnp.abs(hi) + 0j, hi)
        for c in range(i, n):
            h[i][c] = h[i][c] * sc
        for r in range(min(n - 1, i + 1) + 1):
            h[r][i] = h[r][i] * jnp.conj(sc)
        for r in range(n):
            zz[r][i] = zz[r][i] * jnp.conj(sc)

    def body(carry):
        Hm, Zm, w, i, its, kdefl = carry
        h, zz = rows(Hm), rows(Zm)
        l = jnp.asarray(0, i.dtype)
        for k in range(1, n):  # the largest k <= i with a negligible subdiagonal entry h[k][k-1]
            hk = h[k][k - 1]
            tst = _cabs1(h[k - 1][k - 1]) + _cabs1(h[k][k])
            extra = (jnp.abs(re(h[k - 1][k - 2])) if k - 2 >= 0 else 0.0) + (jnp.abs(re(h[k + 1][k])) if k + 1 <= n - 1 else 0.0)
            tst = jnp.where(tst == 0, tst + extra, tst)
            ab, ba = jnp.maximum(_cabs1(hk), _cabs1(h[k - 1][k])), jnp.minimum(_cabs1(hk), _cabs1(h[k - 1][k]))
            dk = _cabs1(h[k - 1][k - 1] - h[k][k])
            aa, bb = jnp.maximum(_cabs1(h[k][k]), dk), jnp.minimum(_cabs1(h[k][k]), dk)
            s = aa + ab
            s_ = jnp.where(s == 0, 1.0, s)
            small = (_cabs1(hk) <= smlnum) | ((jnp.abs(re(hk)) <= ulp * tst) & (ba * (ab / s_) <= jnp.maximum(smlnum, ulp * (bb * (aa / s_)))))
            l = jnp.where((k <= i) & small, k, l)
        for k in range(1, n):  # (H(l, l-1) = 0)
            h[k][k - 1] = jnp.where(l == k, 0.0 + 0j, h[k][k - 1])
        deflate = l >= i
        kd = kdefl + 1
        hii, hi1i1, hii1, hi1i = _get(h, i), _get(h, i - 1), _get(h, i, -1), _get(h, i - 1, 1)
        exc1, exc2 = kd % 20 == 0, (kd % 20 != 0) & (kd % 10 == 0)
        t_e1 = 0.75 * jnp.abs(re(hii1)) + hii
        t_e2 = 0.75 * jnp.abs(re(_get(h, l + 1, -1))) + _get(h, l)
        u = _csqrt(hi1i) * _csqrt(hii1)  # (Wilkinson's shift)
        su = _cabs1(u)
        x = 0.5 * (hi1i1 - hii)
        sx = _cabs1(x)
        s = jnp.maximum(su, sx)
        s_ = jnp.where(s == 0, 1.0, s)
        y = s * _csqrt((x / s_) ** 2 + (u / s_) ** 2)
        xs = x / jnp.where(sx == 0, 1.0, sx)
        y = jnp.where((sx > 0) & (re(xs) * re(y) + jnp.imag(xs) * jnp.imag(y) < 0), -y, y)
        xy = x + y
        t_w = jnp.where(su != 0, hii - u * (u / jnp.where(xy == 0, 1.0 + 0j, xy)), hii)
        t = jnp.where(exc1, t_e1, jnp.where(exc2, t_e2, t_w))
        vstart = lambda h11, h22, h21: (lambda s: ((h11 - t) / s, (h21 / s) + 0j))(
            jnp.where(_cabs1(h11 - t) + jnp.abs(h21) == 0, 1.0, _cabs1(h11 - t) + jnp.abs(h21)))
        m = l
        v1, v2 = vstart(_get(h, l), _get(h, l + 1), re(_get(h, l + 1, -1)))
        for mm in range(1, n - 1):  # the largest m in [l+1, i-1] where starting there makes h[m][m-1] negligible
            h11, h22, h21 = h[mm][mm], h[mm + 1][mm + 1], re(h[mm + 1][mm])
            a1, a2 = vstart(h11, h22, h21)
            ok = jnp.abs(re(h[mm][mm - 1])) * jnp.abs(re(a2)) <= ulp * (_cabs1(a1) * (_cabs1(h11) + _cabs1(h22)))
            take = ok & (mm >= l + 1) & (mm <= i - 1)
            m, v1, v2 = jnp.where(take, mm, m), jnp.where(take, a1, v1), jnp.where(take, a2, v2)
        g, gz = [list(r) for r in h], [list(r) for r in zz]
        for k in range(n - 1):  # the single-shift sweep from m to i - 1
            act = ~deflate & (k >= m) & (k <= i - 1)
            first = k == m
            x1 = v1 if k == 0 else jnp.where(first, v1, g[k][k - 1])
            x2 = v2 if k == 0 else jnp.where(first, v2, g[k + 1][k - 1])
            beta, t1, (w2,) = _zlarfg(x1, [x2])
            if k > 0:
                g[k][k - 1] = jnp.where(act & ~first, beta, g[k][k - 1])
                g[k + 1][k - 1] = jnp.where(act & ~first, 0.0 + 0j, g[k + 1][k - 1])
            t2 = re(t1 * w2)
            for j in range(k, n):  # rows k, k+1, columns k .. n-1
                sm = jnp.conj(t1) * g[k][j] + t2 * g[k + 1][j]
                g[k][j], g[k + 1][j] = jnp.where(act, g[k][j] - sm, g[k][j]), jnp.where(act, g[k + 1][j] - sm * w2, g[k + 1][j])
            for j in range(min(k + 2, n - 1) + 1):  # columns k, k+1, rows 0 .. min(k+2, i)
                sm = t1 * g[j][k] + t2 * g[j][k + 1]
                ok = act & (j <= i)
                g[j][k], g[j][k + 1] = jnp.where(ok, g[j][k] - sm, g[j][k]), jnp.where(ok, g[j][k + 1] - sm * jnp.conj(w2), g[j][k + 1])
            if wantz:
                for j in range(n):
                    sm = t1 * gz[j][k] + t2 * gz[j][k + 1]
                    gz[j][k], gz[j][k + 1] = jnp.where(act, gz[j][k] - sm, gz[j][k]), jnp.where(act, gz[j][k + 1] - sm * jnp.conj(w2), gz[j][k + 1])
            sc = act & first & (m > l)  # (a start below l: H(m, m-1) kept real by a diagonal scaling)
            tmp = 1.0 - t1
            tmp = tmp / jnp.where(jnp.abs(tmp) == 0, 1.0, jnp.abs(tmp))
            g[k + 1][k] = jnp.where(sc, g[k + 1][k] * jnp.conj(tmp), g[k + 1][k])
            if k + 2 <= n - 1:
                g[k + 2][k + 1] = jnp.where(sc & (k + 2 <= i), g[k + 2][k + 1] * tmp, g[k + 2][k + 1])
            for j in range(k, n):
                if j == k + 1:
                    continue
                on = sc & (j <= i)
                for c in range(j + 1, n):
                    g[j][c] = jnp.where(on, g[j][c] * tmp, g[j][c])
                for r in range(j):
                    g[r][j] = jnp.where(on, g[r][j] * jnp.conj(tmp), g[r][j])
                if wantz:
                    for r in range(n):
                        gz[r][j] = jnp.where(on, gz[r][j] * jnp.conj(tmp), gz[r][j])
        # H(i, i-1) real again (after a sweep)
        hi = _get(g, i, -1)
        nzi = ~deflate & (jnp.imag(hi) != 0)
        rt = jnp.abs(hi)
        tn = hi / jnp.where(nzi, rt, 1.0)
        for r in range(n):
            for c in range(n):
                g[r][c] = jnp.where(nzi & (r == i) & (c == i - 1), rt + 0j,
                                    jnp.where(nzi & (r == i) & (c > i), g[r][c] * jnp.conj(tn),
                                              jnp.where(nzi & (c == i) & (r < i), g[r][c] * tn, g[r][c])))
        if wantz:
            for r in range(n):
                for c in range(n):
                    gz[r][c] = jnp.where(nzi & (c == i), gz[r][c] * tn, gz[r][c])
        idx = jnp.arange(n)
        w = jnp.where(deflate & (idx == i), hii, w)
        return (stack(g), stack(gz), w, jnp.where(deflate, l - 1, i), jnp.where(deflate, 0, its + 1), jnp.where(deflate, 0, kd))

    i0, zero = jnp.asarray(n - 1, jnp.int32), jnp.asarray(0, jnp.int32)
    Hm, Zm, w, i, _, _ = lax.while_loop(lambda c: (c[3] >= 0) & (c[4] <= itmax), body,
                                        (stack(h), stack(zz), jnp.zeros(n, cdt), i0, zero, zero))
    return Hm, Zm, jnp.real(w), jnp.imag(w), i < 0
