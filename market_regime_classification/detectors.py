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
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

MODEL_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _index_label(value: object) -> str:
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _as_float_array(
    value: object, *, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_prob_vector(value: object, *, name: str, size: int) -> np.ndarray:
    array = _as_float_array(value, name=name, shape=(size,))
    if (array < 0.0).any() or not np.isclose(array.sum(), 1.0, atol=1e-8):
        raise ValueError(f"{name} must be a probability vector of length {size}")
    return array


def _as_prob_matrix(value: object, *, name: str, size: int) -> np.ndarray:
    array = _as_float_array(value, name=name, shape=(size, size))
    if (array < 0.0).any() or not np.allclose(
        array.sum(axis=1), 1.0, atol=1e-8
    ):
        raise ValueError(f"{name} must be a {size}x{size} row-stochastic matrix")
    return array


def _as_permutation(value: object, *, name: str, size: int) -> np.ndarray:
    array = _as_float_array(value, name=name, shape=(size,))
    as_int = array.astype(int)
    if not np.array_equal(array, as_int) or sorted(as_int.tolist()) != list(
        range(size)
    ):
        raise ValueError(f"{name} must be a permutation of 0..{size - 1}")
    return as_int


def _check_symmetric_covars(covars: np.ndarray, *, name: str) -> None:
    if not np.allclose(covars, np.swapaxes(covars, 1, 2), atol=1e-8):
        raise ValueError(f"{name} must contain symmetric covariance matrices")


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Stable log-sum-exp that tolerates ``-inf`` entries."""
    a = np.asarray(a, dtype=float)
    if axis is None:
        scalar_max = float(np.max(a))
        if not np.isfinite(scalar_max):
            return scalar_max
        return float(scalar_max + np.log(np.sum(np.exp(a - scalar_max))))
    row_max = np.max(a, axis=axis, keepdims=True)
    safe_max = np.where(np.isfinite(row_max), row_max, 0.0)
    total = safe_max + np.log(
        np.sum(np.exp(a - safe_max), axis=axis, keepdims=True)
    )
    return np.squeeze(total, axis=axis)


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
        self.feature_cols_: list[str] = []
        self.vol_col_: str = ""
        self.fitted_on_: dict[str, str | int] | None = None

    @staticmethod
    def _matrix(X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        missing = [c for c in feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")
        M = X.loc[:, feature_cols].to_numpy(dtype=float)
        if not np.isfinite(M).all():
            raise ValueError("feature matrix contains NaN or infinite values")
        return M

    def _record_fit_window(self, X: pd.DataFrame) -> None:
        if len(X) == 0:
            self.fitted_on_ = None
            return
        self.fitted_on_ = {
            "start": _index_label(X.index[0]),
            "end": _index_label(X.index[-1]),
            "rows": len(X),
        }

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fitted model to a JSON-safe dictionary."""
        if not self.feature_cols_ or not self.vol_col_:
            raise RuntimeError("detector must be fitted before serialization")
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "detector": self.name,
            "n_states": self.n_states,
            "feature_cols": list(self.feature_cols_),
            "vol_col": self.vol_col_,
            "fitted_on": self.fitted_on_,
            "params": self._params_dict(),
        }

    @abstractmethod
    def _params_dict(self) -> dict[str, Any]:
        ...

    @classmethod
    def _validate_payload(
        cls, payload: Mapping[str, Any]
    ) -> tuple[
        int, list[str], str, Mapping[str, Any], dict[str, Any] | None
    ]:
        if not isinstance(payload, Mapping):
            raise TypeError("model payload must be a mapping")
        version = payload.get("schema_version")
        if version != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported model schema_version={version!r}; "
                f"expected {MODEL_SCHEMA_VERSION}"
            )
        detector = payload.get("detector")
        if detector != cls.name:
            raise ValueError(
                f"payload detector={detector!r} does not match {cls.name!r}"
            )
        n_states = payload.get("n_states")
        if (
            not isinstance(n_states, int)
            or isinstance(n_states, bool)
            or n_states < 2
        ):
            raise ValueError("n_states must be an integer >= 2")
        feature_cols = payload.get("feature_cols")
        if (
            not isinstance(feature_cols, list)
            or not feature_cols
            or not all(isinstance(col, str) for col in feature_cols)
        ):
            raise ValueError("feature_cols must be a non-empty list of strings")
        if len(set(feature_cols)) != len(feature_cols):
            raise ValueError("feature_cols must be unique")
        vol_col = payload.get("vol_col")
        if not isinstance(vol_col, str) or vol_col not in feature_cols:
            raise ValueError("vol_col must be one of feature_cols")
        params = payload.get("params")
        if not isinstance(params, Mapping):
            raise ValueError("params must be a mapping")
        fitted_on = payload.get("fitted_on")
        if fitted_on is not None and not isinstance(fitted_on, Mapping):
            raise ValueError("fitted_on must be a mapping or null")
        return (
            n_states,
            list(feature_cols),
            vol_col,
            params,
            dict(fitted_on) if fitted_on is not None else None,
        )

    def _serialize_scaler(self) -> dict[str, list[float]]:
        return {
            "scaler_mean": np.asarray(self.scaler_.mean_, dtype=float).tolist(),
            "scaler_scale": np.asarray(self.scaler_.scale_, dtype=float).tolist(),
        }

    def _restore_scaler(
        self, params: Mapping[str, Any], n_features: int
    ) -> None:
        mean = _as_float_array(
            params.get("scaler_mean"),
            name="scaler_mean",
            shape=(n_features,),
        )
        scale = _as_float_array(
            params.get("scaler_scale"),
            name="scaler_scale",
            shape=(n_features,),
        )
        if (scale <= 0.0).any():
            raise ValueError("scaler_scale must be strictly positive")
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = scale
        scaler.var_ = scale**2
        scaler.n_features_in_ = n_features
        self.scaler_ = scaler

    @classmethod
    @abstractmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BaseDetector:
        ...

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, feature_cols: list[str], vol_col: str
    ) -> BaseDetector:
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
    ) -> VolThresholdDetector:
        if vol_col not in feature_cols:
            raise ValueError(f"vol_col={vol_col!r} must be in feature_cols")
        if vol_col not in X.columns:
            raise KeyError(f"vol_col={vol_col!r} missing")
        self.feature_cols_ = list(feature_cols)
        self.vol_col_ = vol_col
        self._record_fit_window(X)
        v = X[vol_col].to_numpy(dtype=float)
        qs = np.linspace(0.0, 1.0, self.n_states + 1)[1:-1]
        self.thresholds_ = np.quantile(v, qs)
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

    def _params_dict(self) -> dict[str, Any]:
        return {
            "thresholds": np.asarray(
                self.thresholds_, dtype=float
            ).tolist()
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VolThresholdDetector:
        (
            n_states,
            feature_cols,
            vol_col,
            params,
            fitted_on,
        ) = cls._validate_payload(payload)
        thresholds = _as_float_array(
            params.get("thresholds"),
            name="thresholds",
            shape=(n_states - 1,),
        )
        if (np.diff(thresholds) < 0.0).any():
            raise ValueError("thresholds must be sorted ascending")
        model = cls(n_states=n_states)
        model.feature_cols_ = feature_cols
        model.vol_col_ = vol_col
        model.fitted_on_ = fitted_on
        model.thresholds_ = thresholds
        return model


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
    ) -> GMMDetector:
        if vol_col not in feature_cols:
            raise ValueError(f"vol_col={vol_col!r} must be in feature_cols")
        self.feature_cols_ = list(feature_cols)
        self.vol_col_ = vol_col
        self.vol_idx_ = self.feature_cols_.index(vol_col)
        self._record_fit_window(X)

        M = self._matrix(X, self.feature_cols_)
        self.scaler_ = StandardScaler().fit(M)
        Z = self.scaler_.transform(M)

        gmm = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            n_init=self.n_init,
            random_state=self.random_state,
            reg_covar=self.reg_covar,
            max_iter=200,
        ).fit(Z)

        # Keep plain arrays so a fitted model and a reloaded model behave
        # identically without depending on sklearn internals at inference time.
        self.weights_ = np.asarray(gmm.weights_, dtype=float).copy()
        self.means_ = np.asarray(gmm.means_, dtype=float).copy()
        self.covars_ = np.asarray(gmm.covariances_, dtype=float).copy()

        # Order components by mean volatility (scaled space is monotone in raw
        # volatility because the scaler's scale is positive).
        scores = self.means_[:, self.vol_idx_]
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
        log_prob = _gaussian_logpdf(Z, self.means_, self.covars_)
        log_prob = log_prob + np.log(np.maximum(self.weights_, 1e-300))
        log_norm = _logsumexp(log_prob, axis=1)
        assert isinstance(log_norm, np.ndarray)
        P = np.exp(log_prob - log_norm[:, None])
        return P[:, self.order_]

    def _params_dict(self) -> dict[str, Any]:
        return {
            "weights": np.asarray(self.weights_, dtype=float).tolist(),
            "means": np.asarray(self.means_, dtype=float).tolist(),
            "covariances": np.asarray(self.covars_, dtype=float).tolist(),
            "order": np.asarray(self.order_, dtype=int).tolist(),
            **self._serialize_scaler(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GMMDetector:
        (
            n_states,
            feature_cols,
            vol_col,
            params,
            fitted_on,
        ) = cls._validate_payload(payload)
        n_features = len(feature_cols)
        weights = _as_prob_vector(
            params.get("weights"), name="weights", size=n_states
        )
        means = _as_float_array(
            params.get("means"),
            name="means",
            shape=(n_states, n_features),
        )
        covars = _as_float_array(
            params.get("covariances"),
            name="covariances",
            shape=(n_states, n_features, n_features),
        )
        _check_symmetric_covars(covars, name="covariances")
        order = _as_permutation(
            params.get("order"), name="order", size=n_states
        )
        model = cls(n_states=n_states)
        model.feature_cols_ = feature_cols
        model.vol_col_ = vol_col
        model.vol_idx_ = feature_cols.index(vol_col)
        model.fitted_on_ = fitted_on
        model.weights_ = weights
        model.means_ = means
        model.covars_ = covars
        model.order_ = order
        model._restore_scaler(params, n_features)
        return model


# ---------------------------------------------------------------------------
# Causal Gaussian HMM
# ---------------------------------------------------------------------------
class GaussianHMMDetector(BaseDetector):
    """Full-covariance Gaussian HMM fitted with EM, filtered at prediction time.

    Notes
    -----
    * EM uses the forward-backward algorithm *on the training window* — that is
      legitimate: the whole training window is in the past.
    * States are initialized using quantiles of the primary volatility feature
      (with randomized quantile jitter on restarts > 0), preventing multi-feature
      KMeans from collapsing rare crisis windows into low-volatility clusters.
    * The M-step applies a Dirichlet sticky self-transition prior (``stickiness``,
      Fox et al.) to prevent spurious regime chatter and shrinks state sample
      covariances toward their diagonal (``cov_shrinkage``) to improve
      conditioning in low-occupancy states.
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
        stickiness: float = 15.0,
        cov_shrinkage: float = 0.25,
    ) -> None:
        super().__init__(n_states)
        if stickiness < 0.0 or not np.isfinite(stickiness):
            raise ValueError("stickiness must be a finite non-negative number")
        if (
            cov_shrinkage < 0.0
            or cov_shrinkage > 1.0
            or not np.isfinite(cov_shrinkage)
        ):
            raise ValueError("cov_shrinkage must be in [0, 1]")
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.stickiness = float(stickiness)
        self.cov_shrinkage = float(cov_shrinkage)

    # -- public API ---------------------------------------------------------
    def fit(
        self, X: pd.DataFrame, feature_cols: list[str], vol_col: str
    ) -> GaussianHMMDetector:
        if vol_col not in feature_cols:
            raise ValueError(f"vol_col={vol_col!r} must be in feature_cols")
        self.feature_cols_ = list(feature_cols)
        self.vol_col_ = vol_col
        self.vol_idx_ = self.feature_cols_.index(vol_col)
        self._record_fit_window(X)

        M = self._matrix(X, self.feature_cols_)
        if len(M) < max(2 * self.n_states, 20):
            raise ValueError(f"not enough training rows for HMM: {len(M)}")
        self.scaler_ = StandardScaler().fit(M)
        Z = self.scaler_.transform(M)

        best: tuple[float, tuple] | None = None
        for r in range(self.n_init):
            params, obj, ll = self._fit_once(Z, restart=r)
            if best is None or (np.isfinite(obj) and obj > best[0]):
                best = (obj, (*params, ll))
        if best is None:
            raise RuntimeError("HMM fitting failed for every restart")

        _, (pi, A, means, covars, best_ll) = best

        # Canonical state ordering: low vol -> high vol.
        order = np.argsort(means[:, self.vol_idx_], kind="stable")
        self.startprob_ = pi[order]
        self.transmat_ = A[np.ix_(order, order)]
        self.means_ = means[order]
        self.covars_ = covars[order]
        self.log_likelihood_ = float(best_ll)
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

    def _params_dict(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "startprob": np.asarray(
                self.startprob_, dtype=float
            ).tolist(),
            "transmat": np.asarray(
                self.transmat_, dtype=float
            ).tolist(),
            "means": np.asarray(self.means_, dtype=float).tolist(),
            "covars": np.asarray(self.covars_, dtype=float).tolist(),
            **self._serialize_scaler(),
        }
        log_likelihood = getattr(self, "log_likelihood_", None)
        if log_likelihood is not None and np.isfinite(log_likelihood):
            params["log_likelihood"] = float(log_likelihood)
        return params

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GaussianHMMDetector:
        (
            n_states,
            feature_cols,
            vol_col,
            params,
            fitted_on,
        ) = cls._validate_payload(payload)
        n_features = len(feature_cols)
        startprob = _as_prob_vector(
            params.get("startprob"), name="startprob", size=n_states
        )
        transmat = _as_prob_matrix(
            params.get("transmat"), name="transmat", size=n_states
        )
        means = _as_float_array(
            params.get("means"),
            name="means",
            shape=(n_states, n_features),
        )
        covars = _as_float_array(
            params.get("covars"),
            name="covars",
            shape=(n_states, n_features, n_features),
        )
        _check_symmetric_covars(covars, name="covars")

        model = cls(n_states=n_states)
        model.feature_cols_ = feature_cols
        model.vol_col_ = vol_col
        model.vol_idx_ = feature_cols.index(vol_col)
        model.fitted_on_ = fitted_on
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = covars
        log_likelihood = params.get("log_likelihood")
        if log_likelihood is not None:
            model.log_likelihood_ = float(
                _as_float_array(log_likelihood, name="log_likelihood")
            )
        model._restore_scaler(params, n_features)
        return model

    # -- EM -----------------------------------------------------------------
    def _init_labels(self, Z: np.ndarray, restart: int) -> np.ndarray:
        k = self.n_states
        v = Z[:, self.vol_idx_]
        qs = np.linspace(0.0, 1.0, k + 1)[1:-1]
        if restart > 0:
            rng = np.random.default_rng(self.random_state + restart)
            jitter = rng.uniform(-0.12, 0.12, size=len(qs))
            qs = np.sort(np.clip(qs + jitter, 0.05, 0.95))
        thresholds = np.quantile(v, qs)
        return np.clip(np.searchsorted(thresholds, v, side="right"), 0, k - 1)

    def _fit_once(
        self, Z: np.ndarray, restart: int
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float, float
    ]:
        _, d = Z.shape
        k = self.n_states

        labels = self._init_labels(Z, restart)

        # Initial transition matrix from hard labels, with sticky Dirichlet prior.
        eye_k = np.eye(k)
        counts = np.zeros((k, k), dtype=float)
        np.add.at(counts, (labels[:-1], labels[1:]), 1.0)
        trans = counts + 0.1 + self.stickiness * eye_k
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
            cov = np.atleast_2d(cov)
            if self.cov_shrinkage > 0.0:
                cov = (
                    (1.0 - self.cov_shrinkage) * cov
                    + self.cov_shrinkage * np.diag(np.diag(cov))
                )
            covars[j] = cov + self.reg_covar * eye

        prev_obj = -np.inf
        obj = -np.inf
        ll = -np.inf
        for _ in range(self.max_iter):
            logB = _gaussian_logpdf(Z, means, covars)
            log_alpha, log_beta, ll = self._forward_backward(logB, trans, start)

            prior_ll = self.stickiness * float(
                np.sum(np.log(np.maximum(np.diag(trans), 1e-300)))
            )
            obj = ll + prior_ll

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
            xi_sum = xi.sum(axis=0) + self.stickiness * eye_k
            denom = np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
            trans = xi_sum / denom

            nk = gamma.sum(axis=0)
            means = (gamma.T @ Z) / np.maximum(nk[:, None], 1e-300)
            for j in range(k):
                diff = Z - means[j]
                cov = (gamma[:, j, None] * diff).T @ diff / max(nk[j], 1e-300)
                cov = 0.5 * (cov + cov.T)
                if self.cov_shrinkage > 0.0:
                    cov = (
                        (1.0 - self.cov_shrinkage) * cov
                        + self.cov_shrinkage * np.diag(np.diag(cov))
                    )
                covars[j] = cov + self.reg_covar * eye

            if np.isfinite(obj) and abs(obj - prev_obj) < self.tol:
                break
            prev_obj = obj

        return (start, trans, means, covars), float(obj), float(ll)

    def _forward_backward(
        self,
        logB: np.ndarray,
        trans: np.ndarray,
        start: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Rabiner-scaled forward-backward on a training window (used only inside EM)."""
        n, k = logB.shape
        pi = np.maximum(start, 1e-300)
        pi = pi / pi.sum()
        A = np.maximum(trans, 1e-300)
        A = A / A.sum(axis=1, keepdims=True)

        row_max = np.max(logB, axis=1)
        B = np.exp(logB - row_max[:, None])

        alpha = np.empty((n, k), dtype=float)
        scales = np.empty(n, dtype=float)

        a0 = pi * B[0]
        s0 = float(a0.sum())
        if s0 <= 0.0:
            a0 = np.full(k, 1.0 / k)
            s0 = 1.0
        else:
            a0 /= s0
        alpha[0] = a0
        scales[0] = s0

        for t in range(1, n):
            at = (alpha[t - 1] @ A) * B[t]
            st = float(at.sum())
            if st <= 0.0:
                at = np.full(k, 1.0 / k)
                st = 1e-300
            else:
                at /= st
            alpha[t] = at
            scales[t] = st

        log_scales = np.log(np.maximum(scales, 1e-300)) + row_max
        ll = float(np.sum(log_scales))

        beta = np.empty((n, k), dtype=float)
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            bt = A @ (B[t + 1] * beta[t + 1])
            beta[t] = bt / max(scales[t + 1], 1e-300)

        cum_log_scales = np.cumsum(log_scales)
        log_alpha = np.log(np.maximum(alpha, 1e-300)) + cum_log_scales[:, None]
        rev_cum_log_scales = np.cumsum(log_scales[::-1])[::-1] - log_scales
        log_beta = (
            np.log(np.maximum(beta, 1e-300)) + rev_cum_log_scales[:, None]
        )
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
            s = prior.sum()
            prior = prior / s if s > 0 else np.full(k, 1.0 / k)
            first = np.log(np.maximum(prior, 1e-300)) + logB[0]
            alpha = np.exp(first - np.max(first))
        else:
            prior = np.asarray(init_probs, dtype=float).reshape(-1)
            if prior.shape != (k,):
                raise ValueError(
                    f"init_probs must have shape ({k},), got {prior.shape}"
                )
            prior = np.clip(prior, 0.0, None)
            s = prior.sum()
            prior = prior / s if s > 0 else np.full(k, 1.0 / k)
            alpha = (prior @ self.transmat_) * np.exp(
                logB[0] - np.max(logB[0])
            )

        s0 = alpha.sum()
        alpha = (
            alpha / s0
            if (np.isfinite(s0) and s0 > 0)
            else np.full(k, 1.0 / k)
        )

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
# Statistical Jump Model (SJM)
# ---------------------------------------------------------------------------
class JumpModelDetector(BaseDetector):
    """Statistical Jump Model with L1 jump penalty and causal forward filtering.

    Fits regime centroids and diagonal variances on the standardized training
    window by minimizing negative diagonal Gaussian log-likelihood plus an L1
    state-switch penalty ``jump_penalty`` via coordinate descent (alternating
    dynamic-programming path updates with parameter updates).

    At prediction time, runs a causal forward-only filter using the fitted
    diagonal-Gaussian emissions and sticky transition matrix, carrying state
    across blocks via ``init_probs``.
    """

    name = "jump"

    def __init__(
        self,
        n_states: int = 3,
        jump_penalty: float = 10.0,
        random_state: int = 42,
        n_init: int = 3,
        max_iter: int = 30,
    ) -> None:
        super().__init__(n_states)
        if jump_penalty < 0.0 or not np.isfinite(jump_penalty):
            raise ValueError(
                "jump_penalty must be a finite non-negative number"
            )
        self.jump_penalty = float(jump_penalty)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)

    def fit(
        self, X: pd.DataFrame, feature_cols: list[str], vol_col: str
    ) -> JumpModelDetector:
        if vol_col not in feature_cols:
            raise ValueError(f"vol_col={vol_col!r} must be in feature_cols")
        self.feature_cols_ = list(feature_cols)
        self.vol_col_ = vol_col
        self.vol_idx_ = self.feature_cols_.index(vol_col)
        self._record_fit_window(X)

        M = self._matrix(X, self.feature_cols_)
        if len(M) < max(2 * self.n_states, 20):
            raise ValueError(
                f"not enough training rows for JumpModel: {len(M)}"
            )
        self.scaler_ = StandardScaler().fit(M)
        Z = self.scaler_.transform(M)

        best_cost = np.inf
        best_params: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        for r in range(self.n_init):
            centers, vars_, states, cost = self._fit_once(Z, restart=r)
            if np.isfinite(cost) and (
                best_params is None or cost < best_cost
            ):
                best_cost = cost
                best_params = (centers, vars_, states)

        if best_params is None:
            raise RuntimeError("JumpModel fitting failed for every restart")

        centers, vars_, states = best_params
        order = np.argsort(centers[:, self.vol_idx_], kind="stable")
        self.means_ = centers[order]
        self.vars_ = vars_[order]

        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(self.n_states)
        ordered_states = inv_order[states]

        k = self.n_states
        counts = np.zeros((k, k), dtype=float)
        np.add.at(counts, (ordered_states[:-1], ordered_states[1:]), 1.0)
        trans = counts + 0.1 + self.jump_penalty * np.eye(k)
        trans /= trans.sum(axis=1, keepdims=True)
        self.transmat_ = trans

        start = np.bincount(ordered_states, minlength=k).astype(float) + 1.0
        start /= start.sum()
        self.startprob_ = start
        self.cost_ = float(best_cost)
        return self

    def _init_labels(self, Z: np.ndarray, restart: int) -> np.ndarray:
        k = self.n_states
        v = Z[:, self.vol_idx_]
        qs = np.linspace(0.0, 1.0, k + 1)[1:-1]
        if restart > 0:
            rng = np.random.default_rng(self.random_state + restart)
            jitter = rng.uniform(-0.12, 0.12, size=len(qs))
            qs = np.sort(np.clip(qs + jitter, 0.05, 0.95))
        thresholds = np.quantile(v, qs)
        return np.clip(np.searchsorted(thresholds, v, side="right"), 0, k - 1)

    def _fit_once(
        self, Z: np.ndarray, restart: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        n, d = Z.shape
        k = self.n_states
        states = self._init_labels(Z, restart)

        centers = np.empty((k, d), dtype=float)
        vars_ = np.ones((k, d), dtype=float)
        global_var = np.maximum(np.var(Z, axis=0), 0.05)

        lam = self.jump_penalty
        off_diag = lam * (1.0 - np.eye(k))
        prev_states = np.full(n, -1, dtype=int)

        cost = np.inf
        for _ in range(self.max_iter):
            for j in range(k):
                mask = states == j
                if mask.sum() >= 5:
                    pts = Z[mask]
                    centers[j] = pts.mean(axis=0)
                    vars_[j] = np.maximum(
                        0.7 * np.var(pts, axis=0) + 0.3 * global_var, 0.05
                    )
                elif mask.any():
                    centers[j] = Z[mask].mean(axis=0)
                    vars_[j] = global_var
                else:
                    centers[j] = Z.mean(axis=0)
                    vars_[j] = global_var

            diff = Z[:, None, :] - centers[None, :, :]
            logdet = 0.5 * np.sum(np.log(vars_), axis=1)
            maha = 0.5 * np.sum((diff * diff) / vars_[None, :, :], axis=2)
            loss = maha + logdet[None, :]

            dp = np.empty((n, k), dtype=float)
            back = np.empty((n, k), dtype=int)
            dp[0] = loss[0]
            back[0] = 0
            for t in range(1, n):
                scores = dp[t - 1][:, None] + off_diag
                best_prev = np.argmin(scores, axis=0)
                dp[t] = loss[t] + scores[best_prev, np.arange(k)]
                back[t] = best_prev

            states = np.empty(n, dtype=int)
            states[-1] = int(np.argmin(dp[-1]))
            cost = float(dp[-1, states[-1]])
            for t in range(n - 2, -1, -1):
                states[t] = back[t + 1, states[t + 1]]

            if np.array_equal(states, prev_states):
                break
            prev_states = states.copy()

        return centers, vars_, states, cost

    def predict(
        self,
        X: pd.DataFrame,
        feature_cols: list[str],
        init_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        M = self._matrix(X, self.feature_cols_)
        Z = self.scaler_.transform(M)
        n = len(Z)
        k = self.n_states
        if n == 0:
            return np.empty((0, k), dtype=float)

        diff = Z[:, None, :] - self.means_[None, :, :]
        logdet = np.sum(np.log(2.0 * np.pi * self.vars_), axis=1)
        maha = np.sum((diff * diff) / self.vars_[None, :, :], axis=2)
        logB = -0.5 * (logdet[None, :] + maha)

        if init_probs is None:
            prior = self.startprob_.copy()
            s = prior.sum()
            prior = prior / s if s > 0 else np.full(k, 1.0 / k)
            first = np.log(np.maximum(prior, 1e-300)) + logB[0]
            alpha = np.exp(first - np.max(first))
        else:
            prior = np.asarray(init_probs, dtype=float).reshape(-1)
            if prior.shape != (k,):
                raise ValueError(
                    f"init_probs must have shape ({k},), got {prior.shape}"
                )
            prior = np.clip(prior, 0.0, None)
            s = prior.sum()
            prior = prior / s if s > 0 else np.full(k, 1.0 / k)
            alpha = (prior @ self.transmat_) * np.exp(
                logB[0] - np.max(logB[0])
            )

        s0 = alpha.sum()
        alpha = (
            alpha / s0
            if (np.isfinite(s0) and s0 > 0)
            else np.full(k, 1.0 / k)
        )

        out = np.empty((n, k), dtype=float)
        out[0] = alpha
        for t in range(1, n):
            alpha = (alpha @ self.transmat_) * np.exp(
                logB[t] - np.max(logB[t])
            )
            s = alpha.sum()
            if not np.isfinite(s) or s <= 0:
                alpha = np.full(k, 1.0 / k)
            else:
                alpha /= s
            out[t] = alpha
        return out

    def _params_dict(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "jump_penalty": float(self.jump_penalty),
            "startprob": np.asarray(
                self.startprob_, dtype=float
            ).tolist(),
            "transmat": np.asarray(
                self.transmat_, dtype=float
            ).tolist(),
            "means": np.asarray(self.means_, dtype=float).tolist(),
            "vars": np.asarray(self.vars_, dtype=float).tolist(),
            **self._serialize_scaler(),
        }
        cost = getattr(self, "cost_", None)
        if cost is not None and np.isfinite(cost):
            params["cost"] = float(cost)
        return params

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JumpModelDetector:
        (
            n_states,
            feature_cols,
            vol_col,
            params,
            fitted_on,
        ) = cls._validate_payload(payload)
        n_features = len(feature_cols)
        jump_penalty = float(
            _as_float_array(params.get("jump_penalty"), name="jump_penalty")
        )
        if jump_penalty < 0.0:
            raise ValueError("jump_penalty must be non-negative")
        startprob = _as_prob_vector(
            params.get("startprob"), name="startprob", size=n_states
        )
        transmat = _as_prob_matrix(
            params.get("transmat"), name="transmat", size=n_states
        )
        means = _as_float_array(
            params.get("means"),
            name="means",
            shape=(n_states, n_features),
        )
        vars_ = _as_float_array(
            params.get("vars"),
            name="vars",
            shape=(n_states, n_features),
        )
        if (vars_ <= 0.0).any():
            raise ValueError("vars must be strictly positive")

        model = cls(n_states=n_states, jump_penalty=jump_penalty)
        model.feature_cols_ = feature_cols
        model.vol_col_ = vol_col
        model.vol_idx_ = feature_cols.index(vol_col)
        model.fitted_on_ = fitted_on
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.vars_ = vars_
        cost = params.get("cost")
        if cost is not None:
            model.cost_ = float(_as_float_array(cost, name="cost"))
        model._restore_scaler(params, n_features)
        return model


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
    hmm_stickiness: float = 15.0,
    hmm_cov_shrinkage: float = 0.25,
    jump_penalty: float = 10.0,
    jump_n_init: int = 3,
    jump_max_iter: int = 30,
) -> BaseDetector:
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
            stickiness=hmm_stickiness,
            cov_shrinkage=hmm_cov_shrinkage,
        )
    if name in {"jump", "sjm", "jump_model"}:
        return JumpModelDetector(
            n_states=n_states,
            jump_penalty=jump_penalty,
            random_state=random_state,
            n_init=jump_n_init,
            max_iter=jump_max_iter,
        )
    raise ValueError(
        f"unknown detector {name!r}; choose from vol_threshold, gmm, hmm, jump"
    )
