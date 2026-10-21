"""Geometric Complexity criterion for the number of principal components.

Implements the codelength of Theorem 3.5 of the paper (Gaussian Geometric
Complexity) for the zero-mean Gaussian family with free covariance, and selects
the reduced dimension m that minimises it.  Also implements Tavory's NML
criterion (the baseline), used for the comparisons in Section 4.

The codelength of a reduced model of dimension m is

    L*(x^N; m) = -log p(x^N | Q_hat)                                  (fit)
               + m(m+1)/4 * log(N / 2 pi)                            (dimension penalty)
               + log vol_g(M(s))                                     (volume)
               - m(m+2)(m-1) / (24 N)                                (curvature correction)

with Q_hat the covariance of the reduced model, s the precision of the
representation (the smallest integer with Sigma <= 2^(2s) I_d), and vol_g(M(s))
the Riemannian volume of the model space.

Reference: "Geometric Complexity, applications to optimal PCA and Causality".
"""
import numpy as np
from scipy.special import gammaln as lgm
from sklearn.decomposition import PCA

# Precision of the representation used throughout the paper (32-bit float).
S_DEFAULT = 32


def zeromean(data):
    """Center the columns (features) of a data matrix."""
    return np.asarray(data, float) - np.mean(data, axis=0)


def changebasis(data, components):
    """Rotate the data onto the principal directions."""
    return np.asarray(data, float) @ np.asarray(components, float).T


def _sum_log_terms(d):
    """Precompute sum_{a=1}^{i} (log a + log Gamma(a/2)) for i = 1..d."""
    out, acc = [], 0.0
    for a in range(1, d + 1):
        acc += np.log(a) + lgm(a / 2.0)
        out.append(acc)
    return out


def _log_gauss(N, d, k, lambdas, S):
    """-log p(x^N | Q_hat): negative log-likelihood of the reduced model."""
    if k == d:
        lambda_bar = 1.0
    else:
        lambda_bar = np.sum(lambdas[k:]) / (d - k)
    Q = np.diag([1.0 / e for e in lambdas[:k]] + [1.0 / lambda_bar] * (d - k))
    r = d * np.log(2 * np.pi) + np.sum(np.log(lambdas[:k]))
    r += (d - k) * np.log(lambda_bar) + np.trace(Q @ S)
    return r * N / 2.0


def gc_optk(data, s=S_DEFAULT, return_scores=False):
    """Optimal number of principal components by Geometric Complexity.

    Parameters
    ----------
    data : array (N, d)
        Data matrix; constant columns should be removed beforehand.
    s : int
        Precision of the representation.  Fixed at 32 in the paper.
    return_scores : bool
        If True also return the codelength of every candidate dimension.

    Returns
    -------
    m* : int
        The dimension that minimises the codelength.
    """
    zm = zeromean(data)
    pca = PCA().fit(zm)
    d, N = zm.shape[1], zm.shape[0]
    rotdata = changebasis(zm, pca.components_)
    S = rotdata.T @ rotdata
    lambdas = pca.explained_variance_
    sumlog = _sum_log_terms(d)

    best_m, best_score, scores = -1, np.inf, {}
    for m in range(1, d + 1):
        score = _log_gauss(N, d, m, lambdas, S)
        # dimension penalty
        score += m * (m + 1) / 4.0 * np.log(N / (2 * np.pi))
        # curvature correction
        score += m * (m + 2) * (m - 1) / (24.0 * N)
        # volume of the model space
        vol = -m / 2.0 * np.log(2) + m * (m + 1) / 4.0 * np.log(np.pi) - sumlog[m - 1]
        vol += m * np.log(s * np.log(2)) + (2 * s + 1) * m * (m - 1) / 4.0 * np.log(2)
        score += vol
        scores[m] = score
        if score < best_score:
            best_m, best_score = m, score
    return (best_m, scores) if return_scores else best_m


def nml_optk(data):
    """Optimal number of principal components by Tavory's NML criterion.

    Baseline of Section 4.3: the normalised maximum likelihood (minimax regret)
    code length of each candidate dimension.
    """
    data = zeromean(data)
    n, m = data.shape
    p = PCA().fit(data)
    log_sigma = np.log(np.dot(data.T, data).sum())
    best_k, best = -1, np.inf
    for k in range(1, m - 1):
        tau = np.sum(p.explained_variance_[k:])
        if tau <= 0:
            continue
        s = (n * m - k * m) * np.log(tau) + k * m * log_sigma
        s += (m * n - k * n - 1) * np.log(m * n / (m * n - k * m))
        s -= (m * k + 1) * np.log(m * k)
        smax = s + m * k * np.log(m / np.sqrt(np.e)) + (k - 1) * np.log(1 / m / np.pi)
        if smax < best:
            best_k, best = k, smax
    return best_k


def knee_optk(data, S=1.0):
    """Optimal number of principal components by the knee/elbow heuristic.

    Kneedle (Satopaa et al. 2011) on the explained variance ratio curve.  Uses the
    ``kneed`` package when it is installed and a built-in implementation of the
    same rule otherwise, so that the paper's knee column can be reproduced
    without it.  Baseline of Section 4.3.
    """
    zm = zeromean(data)
    ev = PCA().fit(zm).explained_variance_ratio_
    x = np.arange(1, len(ev) + 1)
    try:
        from kneed import KneeLocator
        kl = KneeLocator(x, ev, S=S, curve="convex", direction="decreasing")
        return int(kl.elbow) if kl.elbow is not None else 1
    except ImportError:
        return _knee_fallback(x, ev)


def _knee_fallback(x, y):
    """Kneedle's rule without the dependency: maximal distance to the chord."""
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # chord from the first to the last point of the normalised curve
    chord = yn[0] + (yn[-1] - yn[0]) * xn
    dist = chord - yn            # positive where the curve lies below the chord
    return int(x[int(np.argmax(dist))])
