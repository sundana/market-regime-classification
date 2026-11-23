from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime_classification.data import synthetic_market
from market_regime_classification.detectors import (
    MODEL_SCHEMA_VERSION,
    BaseDetector,
    GaussianHMMDetector,
    GMMDetector,
    JumpModelDetector,
    VolThresholdDetector,
)
from market_regime_classification.features import build_features
from market_regime_classification.models import (
    detector_from_dict,
    load_model,
    save_model,
)

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

FACTORIES: dict[str, Callable[[], BaseDetector]] = {
    "vol_threshold": lambda: VolThresholdDetector(3),
    "gmm": lambda: GMMDetector(3, random_state=0, n_init=1),
    "hmm": lambda: GaussianHMMDetector(
        3, random_state=0, n_init=1, max_iter=25
    ),
    "jump": lambda: JumpModelDetector(
        3, jump_penalty=10.0, random_state=0, n_init=1, max_iter=20
    ),
}


@pytest.fixture(scope="module")
def market() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices, _ = synthetic_market(n_days=1_200, seed=5)
    features = build_features(prices)
    return features.iloc[:800], features.iloc[800:950]


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_roundtrip_predictions_match(
    tmp_path: Path, market: tuple[pd.DataFrame, pd.DataFrame], name: str
) -> None:
    train, test = market
    model = FACTORIES[name]().fit(train, FEATURE_COLS, VOL_COL)
    path = save_model(model, tmp_path / "nested" / f"{name}.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == MODEL_SCHEMA_VERSION
    assert payload["detector"] == name
    assert payload["n_states"] == 3
    assert payload["feature_cols"] == FEATURE_COLS
    assert payload["vol_col"] == VOL_COL
    assert payload["fitted_on"]["rows"] == len(train)
    assert payload["artifact"]["created_by"].startswith(
        "market-regime-classification"
    )
    assert "NaN" not in path.read_text(encoding="utf-8")
    assert "Infinity" not in path.read_text(encoding="utf-8")

    loaded = load_model(path)
    assert isinstance(loaded, type(model))
    assert loaded.n_states == model.n_states
    assert loaded.feature_cols_ == model.feature_cols_
    assert loaded.vol_col_ == model.vol_col_
    np.testing.assert_allclose(
        loaded.predict(test, FEATURE_COLS),
        model.predict(test, FEATURE_COLS),
        atol=1e-12,
    )

    # A loaded model must itself be serializable (idempotent round trip).
    loaded_again = load_model(
        save_model(loaded, tmp_path / "again" / f"{name}.json")
    )
    np.testing.assert_allclose(
        loaded_again.predict(test, FEATURE_COLS),
        model.predict(test, FEATURE_COLS),
        atol=1e-12,
    )


def test_loaded_hmm_keeps_filter_continuous(
    tmp_path: Path, market: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, test = market
    model = FACTORIES["hmm"]().fit(train, FEATURE_COLS, VOL_COL)
    loaded = load_model(save_model(model, tmp_path / "hmm.json"))

    full = loaded.predict(test, FEATURE_COLS)
    first = loaded.predict(test.iloc[:40], FEATURE_COLS)
    rest = loaded.predict(
        test.iloc[40:], FEATURE_COLS, init_probs=first[-1]
    )
    np.testing.assert_allclose(rest, full[40:], atol=1e-12)


def test_load_rejects_unknown_detector(market) -> None:
    train, _ = market
    payload = VolThresholdDetector(3).fit(
        train, FEATURE_COLS, VOL_COL
    ).to_dict()
    payload["detector"] = "nope"
    with pytest.raises(ValueError, match="unknown detector"):
        detector_from_dict(payload)


def test_load_rejects_wrong_schema_version(market) -> None:
    train, _ = market
    payload = VolThresholdDetector(3).fit(
        train, FEATURE_COLS, VOL_COL
    ).to_dict()
    payload["schema_version"] = MODEL_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        detector_from_dict(payload)


def test_load_rejects_unsorted_thresholds(market) -> None:
    train, _ = market
    payload = VolThresholdDetector(3).fit(
        train, FEATURE_COLS, VOL_COL
    ).to_dict()
    payload["params"]["thresholds"] = [2.0, 1.0]
    with pytest.raises(ValueError, match="sorted ascending"):
        detector_from_dict(payload)


def test_load_rejects_bad_transition_matrix(market) -> None:
    train, _ = market
    payload = GaussianHMMDetector(
        3, random_state=0, n_init=1, max_iter=25
    ).fit(train, FEATURE_COLS, VOL_COL).to_dict()
    payload["params"]["transmat"][0] = [0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match="row-stochastic"):
        detector_from_dict(payload)


def test_load_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_model(path)


def test_save_model_requires_fitted_detector(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        save_model(VolThresholdDetector(3), tmp_path / "model.json")

    with pytest.raises(TypeError, match="BaseDetector"):
        save_model(object(), tmp_path / "model.json")


def test_save_model_is_atomic(tmp_path: Path, market) -> None:
    train, _ = market
    model = FACTORIES["vol_threshold"]().fit(
        train, FEATURE_COLS, VOL_COL
    )
    save_model(model, tmp_path / "model.json")
    assert {path.name for path in tmp_path.iterdir()} == {"model.json"}
