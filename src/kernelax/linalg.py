# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Small dense linear algebra on lists of DAG scalars, as LAPACK computes it (signs, pivots, order)."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from kernelax.dag import Node, _dot, _first_max, _push, add, band, cmp, div, mul, neg, pick, select, sub, unary


MORE_SWEEPS = 30  # Jacobi sweeps at most, per sample until converged (clustered eigenvalues or singular values: fixed
# sweeps, 5 to 9, left couplings of 1e-12 to 1e-9 between clusters)


def _rot(th):  # tan of a Jacobi rotation for theta = (a_qq - a_pp) / (2 a_pq) (the smaller root)
    return jnp.where(th >= 0, 1.0, -1.0) / (jnp.abs(th) + jnp.sqrt(th * th + 1.0))


def _eigh_sweeps(A, V):
    """Cyclic Jacobi sweeps (as _eigh's) while an off-diagonal entry exceeds n eps times A's norm (rounding leaves
    about eps; clusters left 1e-12 to 1e-9)."""
    n, eps = A.shape[0], jnp.finfo(A.dtype).eps
    tol = n * eps * jnp.sqrt(jnp.sum(A * A))

    def sweep(c):
        A, V, k = c
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                t = jnp.where(apq == 0, 0.0, _rot((A[q, q] - A[p, p]) / (2.0 * apq)))
                cs = 1.0 / jnp.sqrt(t * t + 1.0)
                sn = t * cs
                app, aqq = A[p, p] - t * apq, A[q, q] + t * apq
                cols = lambda X: X.at[:, p].set(cs * X[:, p] - sn * X[:, q]).at[:, q].set(sn * X[:, p] + cs * X[:, q])
                A, V = cols(A), cols(V)
                A = A.at[p].set(A[:, p]).at[q].set(A[:, q])  # (symmetric)
                A = A.at[p, p].set(app).at[q, q].set(aqq).at[p, q].set(0.0).at[q, p].set(0.0)
        return A, V, k + 1
    off = lambda A: jnp.max(jnp.abs(A - jnp.diag(jnp.diag(A))))
    return lax.while_loop(lambda c: (off(c[0]) > tol) & (c[2] < MORE_SWEEPS), sweep, (A, V, 0))[:2]


def _svd_sweeps(A, V):
    """One-sided Jacobi sweeps (as _svd's, A's rows the columns) while a pair of columns is not orthogonal to eps
    (dgesvj: |a_p . a_q| <= eps |a_p| |a_q|)."""
    n, eps = A.shape[0], jnp.finfo(A.dtype).eps

    def sweep(c):
        A, V, k = c
        for p in range(n - 1):
            for q in range(p + 1, n):
                al, be, ga = A[p] @ A[p], A[q] @ A[q], A[p] @ A[q]
                t = jnp.where(ga == 0, 0.0, _rot((be - al) / (2.0 * ga)))
                cs = 1.0 / jnp.sqrt(t * t + 1.0)
                sn = t * cs
                rows = lambda X: X.at[p].set(cs * X[p] - sn * X[q]).at[q].set(sn * X[p] + cs * X[q])
                A, V = rows(A), rows(V)
        return A, V, k + 1

    def open_(A):
        G, nrm = A @ A.T, jnp.sqrt(jnp.sum(A * A, axis=1))
        return jnp.any(jnp.abs(G - jnp.diag(jnp.diag(G))) > eps * nrm[:, None] * nrm[None, :])
    return lax.while_loop(lambda c: open_(c[0]) & (c[2] < MORE_SWEEPS), sweep, (A, V, 0))[:2]


def _more(f, A, V):
    """f (a while loop of sweeps, in jnp) on the DAG's matrices A and V: traced and evaluated by the frontend, a loop
    in the kernel whose lanes wait once converged."""
    from kernelax import frontend
    dt = np.float64 if any(isinstance(x, Node) and x.t == "d" for row in A + V for x in row) or \
        not any(isinstance(x, Node) for row in A + V for x in row) else np.float32
    with jax.enable_x64(True):
        closed = jax.make_jaxpr(f)(*[jax.ShapeDtypeStruct((len(M), len(M[0])), dt) for M in (A, V)])
    A, V = frontend._eval(closed, closed.consts, [np.array(M, dtype=object) for M in (A, V)])
    return [list(r) for r in np.asarray(A, dtype=object)], [list(r) for r in np.asarray(V, dtype=object)]


def _qr(a, *, full_matrices, pivoting=False, **_):
    """Householder QR as LAPACK dgeqr2/dorg2r, with its signs: beta = -sign(alpha) |x|, and no
    reflection (tau = 0, R_jj = alpha) if the column is zero below the diagonal."""
    if pivoting:
        raise NotImplementedError("kernelax: qr with pivoting is not supported")
    R = [list(r) for r in a]
    m, n = len(R), len(R[0])
    k, hs = min(m, n), []
    for j in range(k):
        alpha, x = R[j][j], [R[i][j] for i in range(j + 1, m)]
        s = _dot(x, x)
        z = cmp("eq", s, 0.0)
        nrm = unary("sqrt", add(mul(alpha, alpha), s))
        beta = select(z, select(cmp("ge", alpha, 0.0), nrm, neg(nrm)), alpha)
        tau = select(z, div(sub(beta, alpha), select(z, beta, 1.0)), 0.0)
        v = [1.0] + [div(xi, select(z, sub(alpha, beta), 1.0)) for xi in x]
        for c in range(j + 1, n):
            w = mul(tau, _dot(v, [R[i][c] for i in range(j, m)]))
            for i in range(j, m):
                R[i][c] = sub(R[i][c], mul(v[i - j], w))
        R[j][j] = beta
        for i in range(j + 1, m):
            R[i][j] = 0.0
        hs.append((v, tau))
    cols = m if full_matrices else k
    Q = [[float(i == c) for c in range(cols)] for i in range(m)]
    for j in reversed(range(k)):
        v, tau = hs[j]
        for c in range(cols):
            w = mul(tau, _dot(v, [Q[i][c] for i in range(j, m)]))
            for i in range(j, m):
                Q[i][c] = sub(Q[i][c], mul(v[i - j], w))
    return Q, R[:cols]


def _lu(a, **_):
    """LU with partial pivoting as LAPACK getrf (first maximum = idamax); pivots and permutation
    are index trees with constant leaves, so that gathers by them fold."""
    A = [list(r) for r in a]
    m, n = len(A), len(A[0])
    perm, piv = [float(i) for i in range(m)], []
    for j in range(min(m, n)):
        p = _push(lambda c: add(float(j), c), _first_max([unary("abs", A[i][j]) for i in range(j, m)]))
        piv.append(p)
        row, pj = A[j], perm[j]
        for i in range(j + 1, m):  # swap rows j and p
            h = _push(lambda c: cmp("eq", c, float(i)), p)
            A[j], A[i] = [select(h, x, y) for x, y in zip(A[j], A[i])], [select(h, y, x) for x, y in zip(row, A[i])]
            perm[j], perm[i] = select(h, perm[j], perm[i]), select(h, perm[i], pj)
        inv = select(cmp("eq", A[j][j], 0.0), div(1.0, A[j][j]), 1.0)  # (a zero pivot: no scaling, as LAPACK)
        for i in range(j + 1, m):
            A[i][j] = mul(A[i][j], inv)
            A[i][j + 1:] = [sub(x, mul(A[i][j], y)) for x, y in zip(A[i][j + 1:], A[j][j + 1:])]
    return A, piv, perm


def _cholesky(a, **_):
    """Lower Cholesky factor (reads the lower triangle); NaN if not positive definite, as JAX."""
    n = len(a)
    L, ok = [[0.0] * n for _ in range(n)], True
    for j in range(n):
        d = sub(a[j][j], _dot(L[j][:j], L[j][:j]))
        ok = band(ok, cmp("gt", d, 0.0))
        L[j][j] = unary("sqrt", d)
        for i in range(j + 1, n):
            L[i][j] = div(sub(a[i][j], _dot(L[i][:j], L[j][:j])), L[j][j])
    return ([[select(ok, np.nan, x) for x in r] for r in L],)


def _triangular_solve(a, b, *, left_side, lower, transpose_a, unit_diagonal, **_):
    """Forward/back substitution on op(a) x = b (left) or x op(a) = b (right), reading a's triangle."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    if not left_side:  # x op(a) = b  <=>  op(a)^T x^T = b^T
        b, transpose_a = b.T, not transpose_a
    if transpose_a:
        a, lower = a.T, not lower
    n, x = len(a), [None] * len(a)
    for i in (range(n) if lower else reversed(range(n))):
        js = range(i) if lower else range(i + 1, n)
        r = [sub(bv, _dot([a[i][j] for j in js], [x[j][c] for j in js])) for c, bv in enumerate(b[i])]
        x[i] = r if unit_diagonal else [div(v, a[i][i]) for v in r]
    x = np.asarray(x, dtype=object)
    return (x if left_side else x.T,)


def _eigh(a, *, lower=True, sort_eigenvalues=True, subset_by_index=None, **_):
    """Symmetric eigendecomposition as LAPACK syevd (eigenvectors as columns, eigenvalues
    ascending, the given triangle read): cyclic Jacobi sweeps per sample until converged (_eigh_sweeps: a loop of the
    kernel; fixed sweeps unrolled were no faster and left clustered eigenvalues coupled), then an odd-even sorting
    network."""
    n = len(a)
    if n > 8 or subset_by_index is not None:
        raise NotImplementedError("kernelax: eigh is supported up to 8 x 8 and for all eigenvalues (no index subset)")
    A = [[a[max(i, j)][min(i, j)] if lower else a[min(i, j)][max(i, j)] for j in range(n)] for i in range(n)]
    V = [[float(i == j) for j in range(n)] for i in range(n)]
    if n > 1:
        A, V = _more(_eigh_sweeps, A, V)
    w = [A[i][i] for i in range(n)]
    for rnd in range(n if sort_eigenvalues else 0):
        for i in range(rnd % 2, n - 1, 2):
            g = cmp("gt", w[i], w[i + 1])  # swap
            w[i], w[i + 1] = select(g, w[i], w[i + 1]), select(g, w[i + 1], w[i])
            for r in range(n):
                V[r][i], V[r][i + 1] = select(g, V[r][i], V[r][i + 1]), select(g, V[r][i + 1], V[r][i])
    return V, w


def _svd(a, *, full_matrices, compute_uv, subset_by_index=None, **_):
    """Singular value decomposition (s descending; u, vt as LAPACK gesdd up to the signs of singular vector pairs):
    one-sided Jacobi (Hestenes) on the columns, sweeps per sample until converged (_svd_sweeps, as eigh), then an
    odd-even sorting network. Thin factors only (full_matrices: square matrices)."""
    m, n = len(a), len(a[0])
    if subset_by_index is not None or min(m, n) > 8 or compute_uv and full_matrices and m != n:
        raise NotImplementedError("kernelax: svd is supported up to 8 columns, for all singular values (no index "
                                  "subset) and with full_matrices only for square matrices")
    if m < n:  # (of the transpose: u and vt swap)
        out = _svd([[a[i][j] for i in range(m)] for j in range(n)], full_matrices=full_matrices, compute_uv=compute_uv)
        return out if not compute_uv else (out[0], [list(r) for r in zip(*out[2])], [list(r) for r in zip(*out[1])])
    A = [[a[i][j] for i in range(m)] for j in range(n)]  # columns
    V = [[float(i == j) for i in range(n)] for j in range(n)]  # columns of v
    if n > 1:
        A, V = _more(_svd_sweeps, A, V)
    s = [unary("sqrt", _dot(col, col)) for col in A]
    for rnd in range(n):  # (descending)
        for i in range(rnd % 2, n - 1, 2):
            g = cmp("lt", s[i], s[i + 1])  # swap
            s[i], s[i + 1] = select(g, s[i], s[i + 1]), select(g, s[i + 1], s[i])
            for M in (A, V):
                M[i], M[i + 1] = ([select(g, x, y) for x, y in zip(M[i], M[i + 1])], [select(g, y, x) for x, y in zip(M[i], M[i + 1])])
    if not compute_uv:
        return (s,)
    eps = float(np.finfo(np.float32 if any(isinstance(x, Node) and x.t == "s" for x in s) else np.float64).eps)
    U = []  # (columns of u)
    for j in range(n):  # a singular value that vanishes (to n eps of the largest): its column completes u's
        # orthonormal basis, as LAPACK's (a zero column would make, e.g., Kabsch's rotation of planar
        # point sets singular), by Gram-Schmidt of the unit vector least in the span of the columns before
        col = [mul(A[j][i], div(1.0, select(cmp("eq", s[j], 0.0), s[j], 1.0))) for i in range(m)]
        k = _first_max([sub(1.0, _dot([c[i] for c in U], [c[i] for c in U])) for i in range(m)])
        at = [pick(k, c) for c in U]
        r = [sub(select(cmp("eq", k, float(i)), 0.0, 1.0), _dot(at, [c[i] for c in U])) for i in range(m)]
        nr = div(1.0, unary("sqrt", _dot(r, r)))
        null = cmp("le", s[j], mul(float(n) * eps, s[0]))
        U.append([select(null, x, mul(y, nr)) for x, y in zip(col, r)])
    u = [[U[j][i] for j in range(n)] for i in range(m)]
    return s, u, V  # (vt: the rows of vt are the columns of v)
