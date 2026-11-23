from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime_classification.data import synthetic_market
from market_regime_classification.detectors import GaussianHMMDetector
from market_regime_classification.features import build_features
from market_regime_classification.metadata import sha256_file
from market_regime_classification.models import save_model
from market_regime_classification.score import main

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
STATE_COLS = ["state_0", "state_1", "state_2"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):
    prices, _ = synthetic_market(n_days=600, seed=3)
    features = build_features(prices)
    model = GaussianHMMDetector(
        3, random_state=0, n_init=1, max_iter=25
    ).fit(features.iloc[:400], FEATURE_COLS, VOL_COL)

    model_path = save_model(
        model, tmp_path_factory.mktemp("model") / "hmm.json"
    )
    prices_path = tmp_path_factory.mktemp("data") / "prices.csv"
    prices.reset_index(names="date").to_csv(prices_path, index=False)
    return model, model_path, prices_path


def _score_args(
    model_path: Path,
    prices_path: Path,
    outdir: Path,
    *extra: str,
) -> list[str]:
    return [
        "--model",
        str(model_path),
        "--csv",
        str(prices_path),
        "--outdir",
        str(outdir),
        *extra,
    ]


def test_score_cli_end_to_end(tmp_path: Path, trained) -> None:
    model, model_path, prices_path = trained
    outdir = tmp_path / "scores"

    rc = main(_score_args(model_path, prices_path, outdir))

    assert rc == 0
    assert {path.name for path in outdir.iterdir()} == {
        "scores.csv",
        "features.csv",
        "last_probs.json",
        "run_metadata.json",
    }

    scores = pd.read_csv(outdir / "scores.csv", index_col=0, parse_dates=True)
    assert list(scores.columns) == [*STATE_COLS, "label"]
    assert np.allclose(scores[STATE_COLS].sum(axis=1), 1.0)
    assert set(scores["label"].unique()) <= {0, 1, 2}

    last = json.loads((outdir / "last_probs.json").read_text())
    assert np.allclose(last["probabilities"], scores[STATE_COLS].iloc[-1])

    features = pd.read_csv(
        outdir / "features.csv", index_col=0, parse_dates=True
    )
    expected = model.predict(features, feature_cols=model.feature_cols_)
    np.testing.assert_allclose(
        scores[STATE_COLS].to_numpy(), expected, atol=1e-12
    )

    metadata = json.loads((outdir / "run_metadata.json").read_text())
    assert metadata["config"]["detector"] == "hmm"
    assert metadata["config"]["n_states"] == 3
    assert metadata["config"]["model_sha256"] == sha256_file(model_path)
    assert {item["path"] for item in metadata["outputs"]} == {
        "scores.csv",
        "features.csv",
        "last_probs.json",
    }


def test_score_cli_carries_state_forward(tmp_path: Path, trained) -> None:
    model, model_path, prices_path = trained
    first_out = tmp_path / "first"
    assert main(_score_args(model_path, prices_path, first_out)) == 0

    second_out = tmp_path / "second"
    rc = main(
        _score_args(
            model_path,
            prices_path,
            second_out,
            "--init-probs",
            str(first_out / "last_probs.json"),
        )
    )
    assert rc == 0

    features = pd.read_csv(
        second_out / "features.csv", index_col=0, parse_dates=True
    )
    init = json.loads((first_out / "last_probs.json").read_text())[
        "probabilities"
    ]
    expected = model.predict(
        features, feature_cols=model.feature_cols_, init_probs=np.array(init)
    )
    scores = pd.read_csv(
        second_out / "scores.csv", index_col=0, parse_dates=True
    )
    np.testing.assert_allclose(
        scores[STATE_COLS].to_numpy(), expected, atol=1e-12
    )


def test_score_cli_accepts_bare_list_init_probs(
    tmp_path: Path, trained
) -> None:
    _, model_path, prices_path = trained
    init_path = tmp_path / "init.json"
    init_path.write_text(json.dumps([0.2, 0.3, 0.5]), encoding="utf-8")

    rc = main(
        _score_args(
            model_path,
            prices_path,
            tmp_path / "scores",
            "--init-probs",
            str(init_path),
        )
    )
    assert rc == 0


def test_score_cli_rejects_bad_init_probs(
    tmp_path: Path, trained, capsys: pytest.CaptureFixture[str]
) -> None:
    _, model_path, prices_path = trained
    init_path = tmp_path / "init.json"
    init_path.write_text("[0.5, 0.5]", encoding="utf-8")

    rc = main(
        _score_args(
            model_path,
            prices_path,
            tmp_path / "scores",
            "--init-probs",
            str(init_path),
        )
    )

    assert rc == 2
    assert "3 entries" in capsys.readouterr().err


def test_score_cli_rejects_missing_feature_contract(
    tmp_path: Path, trained, capsys: pytest.CaptureFixture[str]
) -> None:
    _, model_path, _ = trained
    prices, _ = synthetic_market(n_days=400, seed=4)
    single_asset = tmp_path / "single.csv"
    prices[["EQ"]].reset_index(names="date").to_csv(
        single_asset, index=False
    )

    rc = main(
        _score_args(model_path, single_asset, tmp_path / "scores")
    )

    assert rc == 2
    assert "corr_21" in capsys.readouterr().err


def test_score_cli_rejects_missing_model(
    tmp_path: Path, trained, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, prices_path = trained
    rc = main(
        _score_args(
            tmp_path / "missing.json", prices_path, tmp_path / "scores"
        )
    )
    assert rc == 2
    assert "error:" in capsys.readouterr().err
