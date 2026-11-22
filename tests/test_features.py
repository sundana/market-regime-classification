from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_classification.data import synthetic_market
from market_regime_classification.features import build_features, daily_returns


@pytest.fixture(scope="module")
def market():
    return synthetic_market(n_days=1_500, seed=7)


def test_synthetic_prices_are_positive_and_dated(market):
    prices, states = market
    assert prices.shape == (1_500, 2)
    assert list(prices.columns) == ["EQ", "BD"]
    assert (prices > 0).all().all()
    assert prices.index.is_monotonic_increasing
    assert len(states) == len(prices)
    assert set(states.unique()) <= {0, 1, 2}


def test_features_have_expected_columns_and_no_nans(market):
    prices, _ = market
    f = build_features(prices)

    expected = {
        "rv_5",
        "rv_21",
        "rv_63",
        "vol_ratio",
        "trend_21",
        "trend_63",
        "drawdown_63",
        "corr_21",
    }
    assert expected <= set(f.columns)
    assert not f.isna().any().any()
    assert np.isfinite(f.to_numpy()).all()
    assert len(f) == len(prices) - 63

    assert (f["rv_5"] > 0).all()
    assert (f["rv_21"] > 0).all()
    assert (f["rv_63"] > 0).all()
    assert (f["vol_ratio"] > 0).all()
    assert (f["drawdown_63"] <= 1e-12).all()
    assert f["corr_21"].between(-1.0, 1.0).all()


def test_features_are_causal(market):
    """Truncating future prices must not change past feature values."""
    prices, _ = market
    full = build_features(prices)
    cut = 900
    part = build_features(prices.iloc[:cut])
    common = full.index.intersection(part.index)
    pd.testing.assert_frame_equal(full.loc[common], part.loc[common])


def test_daily_returns(market):
    prices, _ = market
    r = daily_returns(prices)
    assert r.name == "EQ_ret"
    assert np.isnan(r.iloc[0])
    assert np.isfinite(r.iloc[1:]).all()
    assert (r.iloc[1:] > -1).all()
