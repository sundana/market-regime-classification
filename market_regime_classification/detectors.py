"""Regime detectors.

Every detector implements the same minimal interface::

    fit(features, feature_cols, vol_col) -> self
    predict(features, feature_cols, init_probs=None) -> (n_rows, n_states)

Predictions are **filtered** probabilities: the value at row `t` uses only rows
`<= t`. The hand-rolled HMM below is deliberately forward-only; there is no
forward-backward smoothing at prediction time. A causality test in
`tests/test_detectors.py` guards this invariant.

States are sorted by their mean value of the volatility feature, so state 0 is
the calmest regime in every model and after every refit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Stable log-sum-exp that tolerates ``-inf`` entries."""
    a = np.asarray(a, dtype=float)
    if axis is None:
        m = float(np.max(a))
        if not np.isfinite(m):
            return m
        return float(m + np.log(np.sum(np.exp(a - m))))
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Cholesky with escalating jitter, for near-singular covariances."""
    d = cov.shape[0]
    jitter = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(d))
        except np.linalg.LinAlgError:
            jitter = 1e-8 if jitter == 0.0 else jitter * 10.0
    # Last resort: eigen-decompose and floor the eigenvalues.
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    vals = np.maximum(vals, 1e-8)
    return np.linalg.cholesky((vecs * vals) @ vecs.T + 1e-8 * np.eye(d))


def _gaussian_logpdf(
    X: np.ndarray, means: np.ndarray, covars: np.ndarray
) -> np.ndarray:
    """Log N(x_t | mu_k, Sigma_k) for a full-covariance mixture.

    Returns an array of shape (n_samples, n_components).
    """
    n, d = X.shape
    k = means.shape[0]
    out = np.empty((n, k), dtype=float)
    constant = d * np.log(2.0 * np.pi)
    for j in range(k):
        diff = X - means[j]
        chol = _safe_cholesky(covars[j])
        sol = np.linalg.solve(chol, diff.T)  # (d, n)
        maha = np.sum(sol * sol, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        out[:, j] = -0.5 * (constant + logdet + maha)
    return out


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseDetector(ABC):
    name = "base"

    def __init__(self, n_states: int) -> None:
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.n_states = int(n_states)

    @staticmethod
    def _matrix(X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        missing = [c for c in feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")
        M = X.loc[:, feature_cols].to_numpy(dtype=float)
        if not np.isfinite(M).all():
            raise ValueError("feature matrix contains NaN or infinite values")
        return M

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, feature_cols: list[str], vol_col: str
    ) -> "BaseDetector":
        ...

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        feature_cols: list[str],
        init_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Rule baseline: trailing quantiles of realized volatility
# ---------------------------------------------------------------------------
class VolThresholdDetector(BaseDetector):
    """Classify regimes by trailing-window realized-vol quantiles.

    This is the null model the ML detectors must beat. Thresholds are estimated
    only on the training window and then frozen, so no look-ahead.
    """

    name = "vol_threshold"

    def fit(
        self, X: pd.DataFrame, feature_cols: list[str], vol_col: str
    ) -> "VolThresholdDetector":
        if vol_col not in X.columns:
            raise KeyError(f"vol_col={vol_col!r} missing")
        v = X[vol_col].to_numpy(dtype=float)
        qs = np.linspace(0.0, 1.0, self.n_states + 1)[1:-1]
        self.thresholds_ = np.quantile(v, qs)
        self.vol_col_ = vol_col
        return self

    def predict(
        self,
        X: pd.DataFrame,
        feature_cols: list[str],
        init_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        v = X[self.vol_col_].to_numpy(dtype=float)
        states = np.searchsorted(self.thresholds_, v, side="right")
        states = np.clip(states, 0, self.n_states - 1)
        P = np.zeros((len(v), self.n_states), dtype=float)
        P[np.arange(len(v)), states] = 1.0
        return P


# ---------------------------------------------------------------------------
# Gaussian mixture
# ---------------------------------------------------------------------------
class GMMDetector(BaseDetector):
    """Gaussian mixture on standardized features, fitted per training window."""

    name = "gmm"

    def __init__(
        self,
        n_states: int = 3,
        random_state: int = 42,
        n_init: int = 3,
        reg_covar: float = 1e-4,
    ) -> None:
        super().__init__(n_states)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.reg_covar = float(reg_covar)

    def fit(
        self, X: pd.DataFrame, feature_cols: list[str], vol_col: str
    ) -> "GMMDetector":
        if vol_col not in feature_cols:
            raise ValueError(f"vol_col={vol_col!r} must be in feature_cols")
        self.feature_cols_ = list(feature_cols)
        self.vol_idx_ = self.feature_cols_.index(vol_col)

        M = self._matrix(X, self.feature_cols_)
        self.scaler_ = StandardScaler().fit(M)
        Z = self.scaler_.transform(M)

        self.gmm_ = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            n_init=self.n_init,
            random_state=self.random_state,
            reg_covar=self.reg_covar,
            max_iter=200,
        ).fit(Z)

        # Order components by mean volatility (scaled space is monotone in raw
        # volatility because the scaler's scale is positive).
        scores = self.gmm_.means_[:, self.vol_idx_]
        self.order_ = np.argsort(scores, kind="stable")
        return self

    def predict(
        self,
        X: pd.DataFrame,
        feature_cols: list[str],
        init_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        M = self._matrix(X, self.feature_cols_)
        Z = self.scaler_.transform(M)
        P = self.gmm_.predict_proba(Z)
        return P[:, self.order_]


# ---------------------------------------------------------------------------
# Causal Gaussian HMM
# ---------------------------------------------------------------------------
class GaussianHMMDetector(BaseDetector):
    """Full-covariance Gaussian HMM fitted with EM, filtered at prediction time.

    Notes
    -----
    * EM uses the forward-backward algorithm *on the training window* — that is
      legitimate: the whole training window is in the past.
    * Prediction uses a forward-only recursion. When called block-by-block with
      ``init_probs`` equal to the previous block's last filtered state, the
      filter is continuous across refits.
    * States are permuted after fitting so that state 0 has the lowest mean
      volatility feature.
    """

    name = "hmm"

    def __init__(
        self,
        n_states: int = 3,
        random_state: int = 42,
        n_init: int = 3,
        max_iter: int = 50,
        tol: float = 1e-4,
        reg_covar: float = 1e-2,
    ) -> None:
        super().__init__(n_states)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)

    # -- public API ---------------------------------------------------------
    def fit(
        self, X: pd.DataFrame, feature_cols: list[str], vol_col: str
    ) -> "GaussianHMMDetector":
        if vol_col not in feature_cols:
            raise ValueError(f"vol_col={vol_col!r} must be in feature_cols")
        self.feature_cols_ = list(feature_cols)
        self.vol_idx_ = self.feature_cols_.index(vol_col)

        M = self._matrix(X, self.feature_cols_)
        if len(M) < max(2 * self.n_states, 20):
            raise ValueError(f"not enough training rows for HMM: {len(M)}")
        self.scaler_ = StandardScaler().fit(M)
        Z = self.scaler_.transform(M)

        best: tuple[float, tuple] | None = None
        for r in range(self.n_init):
            params, ll = self._fit_once(Z, seed=self.random_state + r)
            if best is None or (np.isfinite(ll) and ll > best[0]):
                best = (ll, params)
        if best is None:
            raise RuntimeError("HMM fitting failed for every restart")

        _, (pi, A, means, covars) = best

        # Canonical state ordering: low vol -> high vol.
        order = np.argsort(means[:, self.vol_idx_], kind="stable")
        self.startprob_ = pi[order]
        self.transmat_ = A[np.ix_(order, order)]
        self.means_ = means[order]
        self.covars_ = covars[order]
        self.log_likelihood_ = float(best[0])
        return self

    def predict(
        self,
        X: pd.DataFrame,
        feature_cols: list[str],
        init_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        M = self._matrix(X, self.feature_cols_)
        Z = self.scaler_.transform(M)
        return self._filter(Z, init_probs)

    # -- EM -----------------------------------------------------------------
    def _fit_once(
        self, Z: np.ndarray, seed: int
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]:
        n, d = Z.shape
        k = self.n_states

        km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(Z)
        labels = km.labels_

        # Initial transition matrix from hard labels, with smoothing.
        counts = np.zeros((k, k), dtype=float)
        np.add.at(counts, (labels[:-1], labels[1:]), 1.0)
        trans = counts + 0.1
        trans /= trans.sum(axis=1, keepdims=True)

        # Initial state distribution favors the first observation's cluster.
        start = np.full(k, 0.1, dtype=float)
        start[labels[0]] += 1.0
        start /= start.sum()

        means = np.empty((k, d), dtype=float)
        covars = np.empty((k, d, d), dtype=float)
        eye = np.eye(d)
        for j in range(k):
            pts = Z[labels == j]
            if len(pts) >= 2:
                means[j] = pts.mean(axis=0)
                cov = np.cov(pts, rowvar=False, bias=True)
            else:
                means[j] = Z.mean(axis=0)
                cov = np.cov(Z, rowvar=False, bias=True)
            covars[j] = np.atleast_2d(cov) + self.reg_covar * eye

        prev_ll = -np.inf
        ll = -np.inf
        for _ in range(self.max_iter):
            logB = _gaussian_logpdf(Z, means, covars)
            log_alpha, log_beta, ll = self._forward_backward(logB, trans, start)

            log_gamma = log_alpha + log_beta - ll
            gamma = np.exp(log_gamma)
            gamma = np.clip(gamma, 1e-300, None)
            gamma /= gamma.sum(axis=1, keepdims=True)

            log_xi = (
                log_alpha[:-1, :, None]
                + np.log(np.maximum(trans, 1e-300))[None, :, :]
                + logB[1:, None, :]
                + log_beta[1:, None, :]
            )
            xi = np.exp(log_xi - ll)
            xi /= np.maximum(xi.sum(axis=(1, 2), keepdims=True), 1e-300)

            start = gamma[0] / max(gamma[0].sum(), 1e-300)
            denom = np.maximum(gamma[:-1].sum(axis=0), 1e-300)
            trans = xi.sum(axis=0) / denom[:, None]

            nk = gamma.sum(axis=0)
            means = (gamma.T @ Z) / np.maximum(nk[:, None], 1e-300)
            for j in range(k):
                diff = Z - means[j]
                cov = (gamma[:, j, None] * diff).T @ diff / max(nk[j], 1e-300)
                cov = 0.5 * (cov + cov.T)
                covars[j] = cov + self.reg_covar * eye

            if np.isfinite(ll) and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return (start, trans, means, covars), float(ll)

    def _forward_backward(
        self,
        logB: np.ndarray,
        trans: np.ndarray,
        start: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Forward-backward on a training window (used only inside EM)."""
        n, k = logB.shape
        logA = np.log(np.maximum(trans, 1e-300))
        logpi = np.log(np.maximum(start, 1e-300))

        log_alpha = np.empty((n, k), dtype=float)
        log_alpha[0] = logpi + logB[0]
        for t in range(1, n):
            log_alpha[t] = logB[t] + _logsumexp(
                log_alpha[t - 1][:, None] + logA, axis=0
            )

        log_beta = np.zeros((n, k), dtype=float)
        for t in range(n - 2, -1, -1):
            log_beta[t] = _logsumexp(
                logA + (logB[t + 1] + log_beta[t + 1])[None, :], axis=1
            )

        ll = float(_logsumexp(log_alpha[-1], axis=0))
        return log_alpha, log_beta, ll

    def _filter(
        self, Z: np.ndarray, init_probs: np.ndarray | None = None
    ) -> np.ndarray:
        """Forward-only (causal) filtering.

        P(state_t | x_1..x_t). No future observations are touched.
        """
        if len(Z) == 0:
            return np.empty((0, self.n_states), dtype=float)

        logB = _gaussian_logpdf(Z, self.means_, self.covars_)
        k = self.n_states

        if init_probs is None:
            prior = self.startprob_.copy()
        else:
            prior = np.asarray(init_probs, dtype=float).reshape(-1)
            if prior.shape != (k,):
                raise ValueError(
                    f"init_probs must have shape ({k},), got {prior.shape}"
                )
            prior = np.clip(prior, 0.0, None)

        s = prior.sum()
        prior = prior / s if s > 0 else np.full(k, 1.0 / k)

        first = np.log(np.maximum(prior, 1e-300)) + logB[0]
        alpha = np.exp(first - np.max(first))
        alpha /= max(alpha.sum(), 1e-300)

        out = np.empty((len(Z), k), dtype=float)
        out[0] = alpha
        for t in range(1, len(Z)):
            alpha = (alpha @ self.transmat_) * np.exp(logB[t] - np.max(logB[t]))
            s = alpha.sum()
            if not np.isfinite(s) or s <= 0:
                alpha = np.full(k, 1.0 / k)
            else:
                alpha /= s
            out[t] = alpha
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_detector(
    name: str,
    n_states: int = 3,
    random_state: int = 42,
    *,
    gmm_n_init: int = 3,
    hmm_n_init: int = 3,
    hmm_max_iter: int = 50,
):
    """Construct a detector by CLI name."""
    name = name.strip().lower()
    if name in {"vol_threshold", "threshold", "rule"}:
        return VolThresholdDetector(n_states=n_states)
    if name in {"gmm", "mixture"}:
        return GMMDetector(
            n_states=n_states, random_state=random_state, n_init=gmm_n_init
        )
    if name in {"hmm", "markov"}:
        return GaussianHMMDetector(
            n_states=n_states,
            random_state=random_state,
            n_init=hmm_n_init,
            max_iter=hmm_max_iter,
        )
    raise ValueError(
        f"unknown detector {name!r}; choose from vol_threshold, gmm, hmm"
    )
