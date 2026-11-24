from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime_classification.regime_quality import (
    calibration_metrics,
    consensus_metrics,
    consensus_series,
    current_run_length,
    evaluate_run_dir,
    event_recall,
    jsonable,
    label_runs,
    main,
    recovery_metrics,
    stability_metrics,
    switches_in_window,
)


def _dates(n: int = 7) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _write_run_dir(tmp_path: Path) -> tuple[Path, pd.DatetimeIndex]:
    root = tmp_path / "run"
    root.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=30)
    features = pd.DataFrame(
        {
            "rv_5": np.linspace(0.08, 0.50, 30),
            "rv_21": np.linspace(0.10, 0.31, 30),
            "rv_63": np.linspace(0.12, 0.20, 30),
            "vol_ratio": np.linspace(0.7, 1.6, 30),
            "trend_21": np.linspace(0.02, -0.05, 30),
            "trend_63": np.linspace(0.04, -0.06, 30),
            "drawdown_63": np.linspace(0.0, -0.08, 30),
            "corr_21": np.linspace(-0.1, -0.2, 30),
        },
        index=dates,
    )
    features.index.name = "date"
    features.to_csv(root / "features.csv")

    labels = {
        "jump": [0] * 10 + [1] * 14 + [2] * 6,
        "hmm": [0] * 10 + [1] * 14 + [2] * 6,
        "gmm": [0] * 24 + [1] * 6,
    }
    for name, values in labels.items():
        pd.DataFrame({"state": values}, index=dates).to_csv(
            root / f"{name}_labels.csv"
        )
        probs = np.full((30, 3), 0.05)
        for index, state in enumerate(values):
            probs[index, state] = 0.9
        pd.DataFrame(
            probs,
            index=dates,
            columns=["state_0", "state_1", "state_2"],
        ).to_csv(root / f"{name}_probabilities.csv")
        pd.DataFrame(
            {
                "state": [0, 1, 2],
                "share": [0.4, 0.4, 0.2],
                "avg_run_days": [20, 15, 10],
                "n_runs": [2, 2, 1],
                "fwd_vol_21d": [0.12, 0.16, 0.28],
                "fwd_ret_21d": [0.006, 0.014, 0.009],
            }
        ).to_csv(root / f"{name}_regime_stats.csv", index=False)
    pd.DataFrame({"true_state": labels["jump"]}, index=dates).to_csv(
        root / "true_states.csv"
    )
    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2024-02-01T00:00:00+00:00",
                "config": {
                    "n_states": 3,
                    "train_days": 10,
                    "refit_every": 10,
                },
                "source": {
                    "type": "synthetic",
                    "n_days": 60,
                    "seed": 1,
                },
            }
        )
    )
    return root, dates


# ---------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------
def test_recovery_perfect() -> None:
    dates = _dates()
    truth = pd.Series([0, 0, 1, 1, 1, 2, 2], index=dates)
    metrics = recovery_metrics(truth, truth.copy())

    assert metrics["exact_accuracy"] == 1.0
    assert metrics["within_one_state"] == 1.0
    assert metrics["adjusted_rand_index"] == 1.0
    assert metrics["median_lag_days"] == 0.0
    assert metrics["churn_ratio"] == 1.0
    assert metrics["recall_at_21d"] == 1.0
    assert metrics["n_true_transitions"] == 2


def test_recovery_measures_lag() -> None:
    dates = _dates(5)
    truth = pd.Series([0, 0, 1, 1, 1], index=dates)
    labels = pd.Series([0, 0, 0, 1, 1], index=dates)

    metrics = recovery_metrics(truth, labels)

    assert metrics["exact_accuracy"] == 0.8
    assert metrics["median_lag_days"] == 1.0
    assert metrics["recall_at_5d"] == 1.0
    assert metrics["transitions_missed"] == 0


def test_consensus_requires_strict_majority() -> None:
    dates = _dates(3)
    models = {
        "a": pd.Series([0, 0, 1], index=dates),
        "b": pd.Series([0, 1, 1], index=dates),
        "c": pd.Series([0, 0, 0], index=dates),
    }
    consensus = consensus_series(models)

    assert list(consensus) == [0.0, 0.0, 1.0]
    assert consensus_metrics(models)["all_agree_share"] == pytest.approx(1 / 3)

    tied = {
        "a": pd.Series([0, 0], index=dates[:2]),
        "b": pd.Series([1, 1], index=dates[:2]),
    }
    assert consensus_series(tied).isna().all()


def test_calibration_ranks_forward_volatility() -> None:
    rng = np.random.default_rng(7)
    n = 240
    forward_vol = pd.Series(
        rng.lognormal(mean=-1.8, sigma=0.6, size=n),
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    scaled = (forward_vol - forward_vol.min()) / (
        forward_vol.max() - forward_vol.min()
    )
    probs = pd.DataFrame(
        {"state_0": 1.0 - scaled, "state_1": scaled},
        index=forward_vol.index,
    )

    metrics = calibration_metrics(probs, forward_vol)

    assert metrics["auc"] > 0.9
    assert metrics["spearman"] > 0.9
    assert 0.0 <= metrics["brier"] <= 1.0
    assert len(metrics["reliability"]) >= 2


def test_event_recall_flags_stress_window() -> None:
    dates = pd.bdate_range("2024-01-01", periods=30)
    values = [0] * 10 + [2] * 5 + [0] * 15
    labels = pd.Series(values, index=dates)

    report = event_recall(
        labels,
        events={"Test event": "2024-01-15"},
        window_days=5,
    )

    assert report["recall"] == 1.0
    assert report["events"][0]["hit"] is True
    assert report["events"][0]["first_offset_days"] is not None


def test_stability_refit_churn() -> None:
    dates = _dates(10)
    labels = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], index=dates)
    refit_dates = [dates[2], dates[4], dates[6], dates[8]]

    metrics = stability_metrics(labels, refit_dates)

    assert metrics["refit_day_switch_rate"] == 1.0
    assert metrics["non_refit_switch_rate"] == 0.0
    assert metrics["switches"] == 4


def test_run_helpers() -> None:
    dates = _dates(6)
    labels = pd.Series([0, 0, 1, 1, 1, 1], index=dates)

    assert current_run_length(labels) == 4
    assert switches_in_window(labels, window=4) == 1
    runs = label_runs(labels)
    assert list(runs["length"]) == [2, 4]


def test_jsonable_replaces_non_finite() -> None:
    payload = jsonable({"ok": 1.0, "bad": float("nan"), "n": np.int64(3)})
    assert payload == {"ok": 1.0, "bad": None, "n": 3}


# ---------------------------------------------------------------------------
# Run directory and CLI
# ---------------------------------------------------------------------------
def test_evaluate_run_dir(tmp_path: Path) -> None:
    root, _ = _write_run_dir(tmp_path)

    report = evaluate_run_dir(root)

    assert report["n_states"] == 3
    assert report["state_names"] == ["CALM", "NORMAL", "STRESSED"]
    assert set(report["detectors"]) == {"jump", "hmm", "gmm"}
    assert report["recovery"]["jump"]["exact_accuracy"] == 1.0
    assert report["stability"]["jump"]["switches_per_year"] > 0
    assert 0.0 < report["consensus"]["all_agree_share"] < 1.0
    assert report["separation"]["jump"]
    assert report["calibration"] == {}


def test_evaluate_run_dir_with_prices(tmp_path: Path) -> None:
    root, dates = _write_run_dir(tmp_path)
    prices = pd.DataFrame(
        {"EQ": np.linspace(100.0, 120.0, len(dates))},
        index=dates,
    )

    report = evaluate_run_dir(root, prices=prices)

    assert report["separation"]["jump"]
    assert "jump" in report["calibration"]


def test_quality_cli_writes_reports(tmp_path: Path) -> None:
    root, _ = _write_run_dir(tmp_path)
    outdir = tmp_path / "quality"

    rc = main(["--run-dir", str(root), "--outdir", str(outdir)])

    assert rc == 0
    json_path = outdir / "regime_quality.json"
    markdown_path = outdir / "regime_quality.md"
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] == 1
    assert "Regime quality report" in markdown_path.read_text()
