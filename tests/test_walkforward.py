from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_classification.data import synthetic_market
from market_regime_classification.detectors import GMMDetector, GaussianHMMDetector, VolThresholdDetector
from market_regime_classification.evaluate import (
    compare_results,
    linear_weight_map,
    pairwise_agreement,
    perf_stats,
    regime_stats,
)
from market_regime_classification.features import build_features, daily_returns
from market_regime_classification.walkforward import run_walkforward

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
VOL_COL = "rv_21"


@pytest.fixture(scope="module")
def market():
    prices, states = synthetic_market(n_days=1_400, seed=21)
    features = build_features(prices)
    returns = daily_returns(prices)
    return prices, features, returns, states


def test_walkforward_structure(market):
    prices, features, _, _ = market
    train_days, refit_every = 300, 50

    result = run_walkforward(
        features,
        lambda: VolThresholdDetector(n_states=3),
        name="vol_threshold",
        train_days=train_days,
        refit_every=refit_every,
        feature_cols=FEATURE_COLS,
        vol_col=VOL_COL,
    )

    n_oos = len(features) - train_days
    assert len(result.probs) == n_oos
    assert len(result.labels) == n_oos
    assert result.probs.index[0] == features.index[train_days]
    assert result.probs.index.is_monotonic_increasing
    assert result.probs.index.is_unique
    expected_refits = int(np.ceil(n_oos / refit_every))
    assert len(result.refit_dates) == expected_refits
    assert np.allclose(result.probs.sum(axis=1), 1.0)
    assert set(result.labels.unique()) <= {0, 1, 2}


def test_walkforward_has_no_lookahead(market):
    """Predictions for a date must not change when later data is removed."""
    _, features, _, _ = market
    kwargs = dict(
        name="vol_threshold",
        train_days=300,
        refit_every=50,
        feature_cols=FEATURE_COLS,
        vol_col=VOL_COL,
    )
    full = run_walkforward(features, lambda: VolThresholdDetector(3), **kwargs)

    cutoff = features.index[900]
    truncated = run_walkforward(
        features.loc[:cutoff], lambda: VolThresholdDetector(3), **kwargs
    )

    # The truncated run stops at `cutoff`; everything up to there must match.
    common = truncated.probs.index
    pd.testing.assert_frame_equal(full.probs.loc[common], truncated.probs)


def test_walkforward_gmm_and_hmm(market):
    _, features, _, _ = market
    train_days = 350

    gmm = run_walkforward(
        features,
        lambda: GMMDetector(3, random_state=0, n_init=1),
        name="gmm",
        train_days=train_days,
        refit_every=100,
        feature_cols=FEATURE_COLS,
        vol_col=VOL_COL,
    )
    assert np.allclose(gmm.probs.sum(axis=1), 1.0)

    hmm = run_walkforward(
        features,
        lambda: GaussianHMMDetector(3, random_state=0, n_init=1, max_iter=25),
        name="hmm",
        train_days=train_days,
        refit_every=150,
        feature_cols=FEATURE_COLS,
        vol_col=VOL_COL,
    )
    assert np.allclose(hmm.probs.sum(axis=1), 1.0)
    assert hmm.probs.shape == gmm.probs.shape


def test_regime_stats(market):
    prices, features, returns, _ = market
    labels = pd.Series(
        np.tile([0, 1, 2], len(features) // 3 + 1)[: len(features)],
        index=features.index,
    )
    stats = regime_stats(labels, returns, prices["EQ"], 3)
    assert list(stats["state"]) == [0, 1, 2]
    assert np.isclose(stats["share"].sum(), 1.0)
    assert (stats["avg_run_days"] >= 1).all()
    assert np.isfinite(stats["fwd_vol_21d"].dropna()).all()


def test_perf_stats():
    r = pd.Series(np.full(252, 0.001))
    stats = perf_stats(r)
    assert stats["n_days"] == 252
    assert stats["ann_vol"] == pytest.approx(0.0)
    assert stats["sharpe"] == pytest.approx(0.0)
    assert stats["max_drawdown"] == pytest.approx(0.0)
    assert stats["hit_rate"] == 1.0

    empty = perf_stats(pd.Series(dtype=float))
    assert empty["n_days"] == 0


def test_compare_results_and_agreement(market):
    _, features, returns, _ = market
    kwargs = dict(
        train_days=300,
        refit_every=100,
        feature_cols=FEATURE_COLS,
        vol_col=VOL_COL,
    )
    results = {
        "vol_threshold": run_walkforward(
            features, lambda: VolThresholdDetector(3), name="vol_threshold", **kwargs
        ),
        "gmm": run_walkforward(
            features,
            lambda: GMMDetector(3, random_state=0, n_init=1),
            name="gmm",
            **kwargs,
        ),
    }

    table = compare_results(results, returns, 3)
    assert set(table["detector"]) == {"vol_threshold", "gmm", "buy_hold"}
    assert table["n_days"].gt(0).all()
    assert table.loc[table["detector"] == "buy_hold", "turnover"].iloc[0] == 0.0

    agreement = pairwise_agreement(results)
    assert np.allclose(np.diag(agreement), 1.0)
    assert -1.0 <= agreement.loc["vol_threshold", "gmm"] <= 1.0

    weights = linear_weight_map(3)
    assert weights == {0: 1.0, 1: 0.5, 2: 0.0}
