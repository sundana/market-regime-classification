from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_regime_classification.brief import generate_brief, main, render_brief
from market_regime_classification.regime_labels import (
    state_color_names,
    state_description,
    state_names,
)


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


def test_generate_brief_uses_default_detectors(tmp_path: Path) -> None:
    root, _ = _write_run_dir(tmp_path)

    brief = generate_brief(root)

    assert brief["primary_model"] == "jump"
    assert brief["reference_model"] == "jump"
    assert brief["regime"]["state"] == 2
    assert brief["regime"]["name"] == "STRESSED"
    assert brief["regime"]["confidence"] == 0.9
    assert brief["regime"]["stability"] == "stable"
    assert brief["regime"]["run_length"] == 6
    assert brief["regime"]["typical_run_days"] == 10.0
    assert brief["consensus"]["agreement"] == "unanimous"
    assert brief["consensus"]["n_agree"] == 2
    assert brief["consensus"]["n_models"] == 2
    assert brief["forward_context"]["fwd_vol_21d"] == 0.28
    assert brief["forward_context"]["fwd_ret_21d"] == 0.009
    assert brief["evidence"]["rv_21"] > 0.3
    assert any("Nowcast" in caveat for caveat in brief["caveats"])


def test_generate_brief_reports_split(tmp_path: Path) -> None:
    root, _ = _write_run_dir(tmp_path)

    brief = generate_brief(root, detectors=["jump", "gmm"])

    assert brief["consensus"]["agreement"] == "split"
    assert brief["consensus"]["label"] is None
    assert brief["consensus"]["disagreeing"] == ["gmm"]
    assert brief["regime"]["state"] == 2
    assert "provisional" in brief["read"]
    gmm = next(model for model in brief["models"] if model["name"] == "gmm")
    assert gmm["label"] == 1
    assert gmm["agrees"] is False


def test_brief_markdown_and_cli(tmp_path: Path) -> None:
    root, _ = _write_run_dir(tmp_path)

    rc = main(["--run-dir", str(root)])

    assert rc == 0
    json_path = root / "regime_brief.json"
    markdown_path = root / "regime_brief.md"
    assert json_path.exists()
    assert markdown_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] == 1
    text = markdown_path.read_text()
    assert "MARKET REGIME BRIEF" in text
    assert "STRESSED" in text
    assert "Model agreement" in text
    assert "Caveats" in text
    assert "Nowcast" in text


def test_render_brief_split_is_flagged(tmp_path: Path) -> None:
    root, _ = _write_run_dir(tmp_path)
    brief = generate_brief(root, detectors=["jump", "gmm"])

    text = render_brief(brief)

    assert "MODEL DISAGREEMENT" in text
    assert "provisional" in text


def test_state_labels_are_readable() -> None:
    assert state_names(2) == ["CALM", "STRESSED"]
    assert state_names(3) == ["CALM", "NORMAL", "STRESSED"]
    assert state_names(4) == [
        "CALM",
        "NORMAL",
        "ELEVATED",
        "STRESSED",
    ]
    assert len(state_color_names(5)) == 5
    assert state_color_names(3) == ["green", "amber", "red"]
    assert "high realised volatility" in state_description(2, 3)
