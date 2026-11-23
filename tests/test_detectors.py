from __future__ import annotations

import numpy as np
import pytest

from market_regime_classification.data import synthetic_market
from market_regime_classification.detectors import (
    GaussianHMMDetector,
    GMMDetector,
    VolThresholdDetector,
    make_detector,
)
from market_regime_classification.features import build_features

VOL_COL = "rv_21"
FEATURE_COLS = [
    "rv_5",
    "rv_21",
    "rv_63",
    "vol_ratio",
    "trend_21",
    "trend_63",
    "drawdown_63",
    "corr_21",
]


@pytest.fixture(scope="module")
def data():
    prices, states = synthetic_market(n_days=1_500, seed=11)
    features = build_features(prices)
    return features, states


def _check_probs(P: np.ndarray, n_rows: int, n_states: int) -> None:
    assert P.shape == (n_rows, n_states)
    assert np.isfinite(P).all()
    assert (P >= -1e-12).all()
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-8)


def test_vol_threshold(data):
    features, _ = data
    train, test = features.iloc[:800], features.iloc[800:1100]
    det = VolThresholdDetector(n_states=3).fit(train, FEATURE_COLS, VOL_COL)

    assert np.all(np.diff(det.thresholds_) >= 0)
    P = det.predict(test, FEATURE_COLS)
    _check_probs(P, len(test), 3)
    assert set(np.unique(P)) <= {0.0, 1.0}

    # Hard labels must be monotone in realized vol.
    labels = P.argmax(axis=1)
    means = test[VOL_COL].groupby(labels).mean()
    assert means.is_monotonic_increasing


def test_gmm(data):
    features, _ = data
    train, test = features.iloc[:800], features.iloc[800:1100]
    det = GMMDetector(n_states=3, random_state=0, n_init=2).fit(
        train, FEATURE_COLS, VOL_COL
    )
    P = det.predict(test, FEATURE_COLS)
    _check_probs(P, len(test), 3)

    ordered_means = det.means_[det.order_, det.vol_idx_]
    assert np.all(np.diff(ordered_means) >= 0)


def test_hmm(data):
    features, _ = data
    train, test = features.iloc[:800], features.iloc[800:1100]
    det = GaussianHMMDetector(
        n_states=3, random_state=0, n_init=2, max_iter=40
    ).fit(train, FEATURE_COLS, VOL_COL)
    P = det.predict(test, FEATURE_COLS)
    _check_probs(P, len(test), 3)

    assert np.all(np.diff(det.means_[:, det.vol_idx_]) > 0)
    assert np.allclose(det.startprob_.sum(), 1.0)
    assert np.allclose(det.transmat_.sum(axis=1), 1.0)


def test_hmm_prediction_is_causal(data):
    """Filtered probabilities must not depend on future rows."""
    features, _ = data
    train, test = features.iloc[:800], features.iloc[800:1000]
    det = GaussianHMMDetector(
        n_states=3, random_state=3, n_init=1, max_iter=30
    ).fit(train, FEATURE_COLS, VOL_COL)

    full = det.predict(test, FEATURE_COLS)
    prefix = det.predict(test.iloc[:40], FEATURE_COLS)
    np.testing.assert_allclose(full[:40], prefix, atol=1e-12)


def test_hmm_block_filtering_matches_concatenation(data):
    """Carrying init_probs across blocks must match one continuous filter."""
    features, _ = data
    train, test = features.iloc[:800], features.iloc[800:1000]
    det = GaussianHMMDetector(
        n_states=3, random_state=4, n_init=1, max_iter=30
    ).fit(train, FEATURE_COLS, VOL_COL)

    P_all = det.predict(test, FEATURE_COLS)
    P_a = det.predict(test.iloc[:50], FEATURE_COLS)
    P_b = det.predict(test.iloc[50:], FEATURE_COLS, init_probs=P_a[-1])
    np.testing.assert_allclose(P_b, P_all[50:], atol=1e-12)


def test_hmm_recovers_latent_regimes_roughly():
    """Not a quality guarantee, just protection against a broken implementation."""
    prices, states = synthetic_market(n_days=3_000, seed=5)
    features = build_features(prices)
    train = features.iloc[:2_000]
    test = features.iloc[2_000:]

    det = GaussianHMMDetector(
        n_states=3, random_state=0, n_init=3, max_iter=50
    ).fit(train, FEATURE_COLS, VOL_COL)
    labels = det.predict(test, FEATURE_COLS).argmax(axis=1)

    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score(states.loc[test.index], labels)
    # A backward-looking, noisy detector will not recover the latent chain
    # exactly; this threshold only catches gross breakage.
    assert ari > 0.15, f"HMM failed to recover regimes (ARI={ari:.3f})"


def test_factory():
    assert isinstance(make_detector("vol_threshold"), VolThresholdDetector)
    assert isinstance(make_detector("gmm"), GMMDetector)
    assert isinstance(make_detector("hmm"), GaussianHMMDetector)
    with pytest.raises(ValueError):
        make_detector("nope")
