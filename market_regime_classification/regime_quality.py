"""Regime-quality diagnostics for a walk-forward run directory.

The economic scorecard answers "would this have made money?" This module
answers the question the project is actually built for: "is the regime call
accurate, timely, stable, and calibrated?"

There is no ground-truth regime in real markets, so the report combines:

* recovery against the synthetic latent state (exact and ordinal accuracy,
  transition lag, churn);
* separation of forward volatility and returns across states;
* probability calibration against forward realised volatility;
* label stability and refit-boundary churn;
* cross-model consensus; and
* recall around a configurable list of stress events.

The functions are pure where possible so they can be used in tests and from
notebooks. ``market-regime-quality`` wraps them for a run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from .config import TRADING_DAYS
from .evaluate import regime_stats as _regime_stats
from .features import daily_returns
from .regime_labels import state_names

QUALITY_SCHEMA_VERSION = 1
QUALITY_FILENAME = "regime_quality.json"
QUALITY_MARKDOWN_FILENAME = "regime_quality.md"

#: Dates are deliberately explicit and configurable. They are validation
#: episodes, not a trading calendar.
DEFAULT_STRESS_EVENTS: dict[str, str] = {
    "Lehman": "2008-09-15",
    "Flash crash": "2010-05-06",
    "US downgrade": "2011-08-08",
    "China devaluation": "2015-08-24",
    "Volmageddon": "2018-02-05",
    "Christmas 2018": "2018-12-24",
    "COVID": "2020-03-16",
    "CPI shock 2022": "2022-06-13",
    "Yen carry unwind": "2024-08-05",
    "Tariff shock 2025": "2025-04-07",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def finite_or_none(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when unavailable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas values into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is pd.NaT:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    return value


def load_indexed_csv(path: str | Path) -> pd.DataFrame:
    """Read a date-indexed CSV written by the run/score CLIs."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_detectors(run_dir: str | Path) -> list[str]:
    """Return detector names that have ``<name>_labels.csv`` in ``run_dir``."""
    root = Path(run_dir)
    suffix = "_labels.csv"
    return sorted(
        path.name[: -len(suffix)]
        for path in root.glob(f"*{suffix}")
    )


def label_runs(labels: pd.Series) -> pd.DataFrame:
    """Return one row per contiguous run of equal labels."""
    series = pd.Series(labels).dropna()
    if series.empty:
        return pd.DataFrame(columns=["state", "length", "start", "end"])
    groups = (series != series.shift()).cumsum()
    rows: list[dict[str, Any]] = []
    for _, group in series.groupby(groups, sort=False):
        rows.append(
            {
                "state": int(group.iloc[0]),
                "length": len(group),
                "start": group.index[0],
                "end": group.index[-1],
            }
        )
    return pd.DataFrame(rows)


def current_run_length(labels: pd.Series) -> int:
    """Return the number of trailing observations sharing the latest label."""
    series = pd.Series(labels).dropna()
    if series.empty:
        return 0
    latest = series.iloc[-1]
    length = 0
    for value in series.iloc[::-1]:
        if value != latest:
            break
        length += 1
    return length


def switches_in_window(labels: pd.Series, window: int = 20) -> int:
    """Return the number of label changes in the last ``window`` changes."""
    series = pd.Series(labels).dropna().tail(max(int(window), 1) + 1)
    if len(series) < 2:
        return 0
    return int((series.diff().iloc[1:] != 0).sum())


def _run_stats(labels: pd.Series) -> dict[str, Any]:
    runs = label_runs(labels)
    if runs.empty:
        return {
            "median_run_days": None,
            "p10_run_days": None,
            "p90_run_days": None,
            "max_run_days": None,
            "share_runs_under_5d": None,
            "share_days_in_runs_under_5d": None,
        }
    lengths = runs["length"].astype(float)
    return {
        "median_run_days": float(lengths.median()),
        "p10_run_days": float(lengths.quantile(0.1)),
        "p90_run_days": float(lengths.quantile(0.9)),
        "max_run_days": int(lengths.max()),
        "share_runs_under_5d": float((lengths < 5).mean()),
        "share_days_in_runs_under_5d": float(
            lengths[lengths < 5].sum() / lengths.sum()
        ),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recovery_metrics(
    truth: pd.Series,
    labels: pd.Series,
    horizons: Sequence[int] = (5, 10, 21, 42),
) -> dict[str, Any]:
    """Compare detected labels with a known latent state series.

    States are assumed to be ordinal (0 = calmest). ``within_one`` therefore
    counts adjacent-state errors, which are less harmful than calling a calm
    market stressed.
    """
    frame = pd.concat(
        [pd.Series(truth).rename("truth"), pd.Series(labels).rename("label")],
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        return {"n_days": 0}

    truth_values = frame["truth"].to_numpy(dtype=int)
    label_values = frame["label"].to_numpy(dtype=int)
    n_days = len(frame)
    error = np.abs(label_values - truth_values)
    transitions = np.flatnonzero(truth_values[1:] != truth_values[:-1]) + 1
    switches = int(np.sum(label_values[1:] != label_values[:-1]))
    max_horizon = max(int(h) for h in horizons)

    lags: list[int] = []
    recall = {int(h): 0 for h in horizons}
    for start in transitions:
        new_state = truth_values[start]
        window = label_values[start : min(start + max_horizon + 1, n_days)]
        hits = np.flatnonzero(window == new_state)
        if hits.size:
            lags.append(int(hits[0]))
        for horizon in horizons:
            end = min(start + int(horizon) + 1, n_days)
            recall[int(horizon)] += int(
                bool(np.any(label_values[start:end] == new_state))
            )

    ari: float | None
    if len(np.unique(truth_values)) > 1 and len(np.unique(label_values)) > 1:
        ari = float(adjusted_rand_score(truth_values, label_values))
    else:
        ari = None

    result: dict[str, Any] = {
        "n_days": n_days,
        "exact_accuracy": float(np.mean(error == 0)),
        "within_one_state": float(np.mean(error <= 1)),
        "off_by_two_or_more": float(np.mean(error >= 2)),
        "adjusted_rand_index": ari,
        "n_true_transitions": len(transitions),
        "n_detected_switches": switches,
        "churn_ratio": float(switches / len(transitions))
        if len(transitions)
        else None,
        "median_lag_days": float(np.median(lags)) if lags else None,
        "transitions_detected": len(lags),
        "transitions_missed": int(len(transitions) - len(lags)),
    }
    for horizon in horizons:
        total = len(transitions)
        result[f"recall_at_{int(horizon)}d"] = (
            float(recall[int(horizon)] / total) if total else None
        )
    result.update(_run_stats(pd.Series(label_values)))
    return result


def stability_metrics(
    labels: pd.Series,
    refit_dates: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Summarise persistence and model-refit churn."""
    series = pd.Series(labels).dropna()
    if series.empty:
        return {"n_days": 0}
    switches = int((series.diff().iloc[1:] != 0).sum())
    years = len(series) / TRADING_DAYS
    result: dict[str, Any] = {
        "n_days": len(series),
        "switches": switches,
        "switches_per_year": float(switches / years) if years else None,
        **_run_stats(series),
    }
    if refit_dates is not None:
        refit_set = {pd.Timestamp(value) for value in refit_dates}
        is_refit = pd.Series(
            [timestamp in refit_set for timestamp in series.index],
            index=series.index,
        )
        changed = series.diff().fillna(0) != 0
        on_refit = (
            float(changed[is_refit].mean())
            if bool(is_refit.any())
            else None
        )
        off_refit = (
            float(changed[~is_refit].mean())
            if bool((~is_refit).any())
            else None
        )
        result["refit_day_switch_rate"] = on_refit
        result["non_refit_switch_rate"] = off_refit
        result["refit_churn_ratio"] = (
            float(on_refit / off_refit)
            if on_refit is not None and off_refit not in (None, 0.0)
            else None
        )
    return result


def _row_consensus(row: pd.Series) -> float:
    counts = row.dropna().value_counts()
    if counts.empty:
        return math.nan
    total = int(counts.sum())
    if int(counts.iloc[0]) * 2 <= total:
        return math.nan
    return float(counts.index[0])


def consensus_series(labels_by_model: Mapping[str, pd.Series]) -> pd.Series:
    """Return the strict-majority label per date, or NaN on a tie."""
    if not labels_by_model:
        raise ValueError("at least one model is required for consensus")
    frame = pd.concat(
        {name: pd.Series(series) for name, series in labels_by_model.items()},
        axis=1,
        join="inner",
    )
    values = frame.apply(_row_consensus, axis=1)
    return pd.Series(values, index=frame.index, name="consensus", dtype=float)


def consensus_metrics(labels_by_model: Mapping[str, pd.Series]) -> dict[str, Any]:
    """Return all-agree share and pairwise exact-agreement rates."""
    names = list(labels_by_model)
    if not names:
        return {"n_models": 0, "n_days": 0}
    frame = pd.concat(
        {name: pd.Series(labels_by_model[name]) for name in names},
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        return {"n_models": len(names), "n_days": 0}
    pairwise: dict[str, float] = {}
    values: list[float] = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            agreement = float((frame[first] == frame[second]).mean())
            pairwise[f"{first}|{second}"] = agreement
            values.append(agreement)
    return {
        "n_models": len(names),
        "n_days": len(frame),
        "all_agree_share": float((frame.nunique(axis=1) == 1).mean()),
        "mean_pairwise_agreement": float(np.mean(values)) if values else 1.0,
        "pairwise": pairwise,
    }


def event_recall(
    labels: pd.Series,
    events: Mapping[str, str] | None = None,
    window_days: int = 21,
    stressed_state: int | None = None,
) -> dict[str, Any]:
    """Measure how often the stressed state appears around known episodes."""
    series = pd.Series(labels).dropna()
    if series.empty:
        return {"n_events_in_range": 0, "recall": None, "events": []}
    if events is None:
        events = DEFAULT_STRESS_EVENTS
    if stressed_state is None:
        stressed_state = int(series.max())

    rows: list[dict[str, Any]] = []
    hits = 0
    in_range = 0
    for name, date_text in events.items():
        event_date = pd.Timestamp(date_text)
        start = event_date - pd.Timedelta(days=int(window_days))
        end = event_date + pd.Timedelta(days=int(window_days))
        window = series.loc[(series.index >= start) & (series.index <= end)]
        if window.empty:
            rows.append(
                {
                    "event": name,
                    "date": event_date.date().isoformat(),
                    "in_range": False,
                    "hit": False,
                    "first_offset_days": None,
                }
            )
            continue
        in_range += 1
        stressed = window[window == stressed_state]
        hit = not stressed.empty
        first_offset = (
            int((stressed.index[0] - event_date).days) if hit else None
        )
        if hit:
            hits += 1
        rows.append(
            {
                "event": name,
                "date": event_date.date().isoformat(),
                "in_range": True,
                "hit": bool(hit),
                "first_offset_days": first_offset,
            }
        )
    return {
        "stressed_state": int(stressed_state),
        "window_days": int(window_days),
        "n_events_in_range": in_range,
        "recall": float(hits / in_range) if in_range else None,
        "events": rows,
    }


def separation_metrics(
    labels: pd.Series,
    returns: pd.Series,
    primary_prices: pd.Series,
    n_states: int,
    forward_days: int = 21,
) -> pd.DataFrame:
    """Forward volatility / return by state, plus the high-vol ratio."""
    stats = _regime_stats(
        labels,
        returns,
        primary_prices,
        n_states,
        forward_days=forward_days,
    )
    if not stats.empty:
        calm = finite_or_none(stats["fwd_vol_21d"].iloc[0])
        stats["vol_ratio_vs_calm"] = (
            stats["fwd_vol_21d"] / calm if calm and calm > 0 else np.nan
        )
    return stats


def calibration_metrics(
    probs: pd.DataFrame,
    forward_vol: pd.Series,
    state: int | str | None = None,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Calibrate a state probability against forward 21-day volatility."""
    if probs.empty:
        return {"n": 0}
    column: str
    if state is None:
        column = str(probs.columns[-1])
    elif isinstance(state, int):
        column = f"state_{state}"
    else:
        column = state
    if column not in probs.columns:
        return {"n": 0, "error": f"missing probability column {column!r}"}

    frame = pd.concat(
        [
            pd.Series(probs[column]).rename("probability"),
            pd.Series(forward_vol).rename("forward_vol"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(frame) < 10:
        return {"n": len(frame)}

    probability = frame["probability"].astype(float)
    realised = frame["forward_vol"].astype(float)
    threshold = float(realised.quantile(2 / 3))
    target = (realised > threshold).astype(int)
    auc = (
        float(roc_auc_score(target, probability))
        if target.nunique() == 2
        else None
    )
    brier = float(np.mean((probability - target) ** 2))
    spearman = finite_or_none(
        probability.corr(realised, method="spearman")
    )

    reliability: list[dict[str, Any]] = []
    q = min(int(n_bins), int(probability.nunique()))
    if q >= 2:
        bins = pd.qcut(probability, q=q, duplicates="drop")
        grouped = (
            pd.DataFrame(
                {
                    "probability": probability,
                    "forward_vol": realised,
                    "bin": bins,
                }
            )
            .groupby("bin", observed=True)
            .agg(
                mean_probability=("probability", "mean"),
                mean_forward_vol=("forward_vol", "mean"),
                n=("probability", "size"),
            )
            .reset_index(drop=True)
        )
        reliability = grouped.to_dict(orient="records")

    return {
        "n": len(frame),
        "state": column,
        "high_vol_threshold": threshold,
        "auc": auc,
        "brier": brier,
        "spearman": spearman,
        "reliability": reliability,
    }


# ---------------------------------------------------------------------------
# Run-directory report
# ---------------------------------------------------------------------------
def _metadata_config(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    config = metadata.get("config", {})
    return config if isinstance(config, Mapping) else {}


def _read_regime_stats(root: Path, detector: str) -> pd.DataFrame | None:
    path = root / f"{detector}_regime_stats.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def evaluate_run_dir(
    run_dir: str | Path,
    *,
    detectors: Sequence[str] | None = None,
    prices: pd.DataFrame | None = None,
    events: Mapping[str, str] | None = None,
    include_default_events: bool = True,
) -> dict[str, Any]:
    """Build a regime-quality report for one walk-forward output directory."""
    root = Path(run_dir)
    features_path = root / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found")

    features = load_indexed_csv(features_path)
    available = discover_detectors(root)
    if detectors is None:
        selected = available
    else:
        selected = [str(name) for name in detectors]
        missing = [name for name in selected if name not in available]
        if missing:
            raise ValueError(
                f"missing labels for detectors {missing}; available: {available}"
            )
    if not selected:
        raise ValueError(f"no *_labels.csv files found in {root}")

    labels = {
        name: load_indexed_csv(root / f"{name}_labels.csv")
        .iloc[:, 0]
        .dropna()
        for name in selected
    }
    probs = {
        name: load_indexed_csv(root / f"{name}_probabilities.csv")
        for name in selected
        if (root / f"{name}_probabilities.csv").exists()
    }

    metadata: Mapping[str, Any] = {}
    metadata_path = root / "run_metadata.json"
    if metadata_path.exists():
        loaded = load_json(metadata_path)
        if isinstance(loaded, Mapping):
            metadata = loaded
    config = _metadata_config(metadata)

    n_states = int(config.get("n_states", 0) or 0)
    if n_states <= 0:
        n_states = max(int(series.max()) for series in labels.values()) + 1
    refit_every = int(config.get("refit_every", 0) or 0)

    truth: pd.Series | None = None
    truth_path = root / "true_states.csv"
    if truth_path.exists():
        truth = load_indexed_csv(truth_path).iloc[:, 0].dropna()

    recovery: dict[str, Any] = {}
    if truth is not None:
        recovery = {
            name: recovery_metrics(truth, series)
            for name, series in labels.items()
        }

    stability: dict[str, Any] = {}
    for name, series in labels.items():
        refit_dates = (
            list(series.index[::refit_every]) if refit_every > 0 else None
        )
        stability[name] = stability_metrics(series, refit_dates)

    consensus = consensus_metrics(labels)

    if events is None and include_default_events:
        events = DEFAULT_STRESS_EVENTS
    event_report = (
        {
            name: event_recall(series, events=events)
            for name, series in labels.items()
        }
        if events
        else {}
    )

    separation: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    if prices is not None and not prices.empty:
        returns = daily_returns(prices)
        primary_prices = prices[prices.columns[0]]
        forward_vol = (
            returns.rolling(21).std().shift(-21) * np.sqrt(TRADING_DAYS)
        )
        for name, series in labels.items():
            separation[name] = separation_metrics(
                series, returns, primary_prices, n_states
            ).to_dict(orient="records")
            if name in probs:
                calibration[name] = calibration_metrics(
                    probs[name], forward_vol
                )
    else:
        for name in selected:
            stats = _read_regime_stats(root, name)
            if stats is not None and not stats.empty:
                calm = finite_or_none(stats["fwd_vol_21d"].iloc[0])
                stats["vol_ratio_vs_calm"] = (
                    stats["fwd_vol_21d"] / calm
                    if calm and calm > 0
                    else np.nan
                )
                separation[name] = stats.to_dict(orient="records")

    caveats = [
        "A regime is a volatility state, not a direction forecast.",
        "Filtered probabilities are nowcasts; turns are identified with a lag.",
        "Real markets have no ground-truth regime; read recovery, separation, "
        "stability, calibration, and event recall together.",
        "Informational only; not investment advice.",
    ]
    if not calibration:
        caveats.append(
            "Calibration was skipped; pass --prices-csv to compute it from "
            "raw returns."
        )

    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(root),
        "data_through": features.index[-1].date().isoformat(),
        "detectors": selected,
        "n_states": n_states,
        "state_names": state_names(n_states),
        "recovery": recovery,
        "stability": stability,
        "consensus": consensus,
        "events": event_report,
        "separation": separation,
        "calibration": calibration,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _pct(value: Any, digits: int = 0) -> str:
    number = finite_or_none(value)
    return "n/a" if number is None else f"{100 * number:.{digits}f}%"


def _num(value: Any, digits: int = 2) -> str:
    number = finite_or_none(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _ratio(value: Any, digits: int = 1) -> str:
    number = finite_or_none(value)
    return "n/a" if number is None else f"{number:.{digits}f}x"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_quality_markdown(report: Mapping[str, Any]) -> str:
    """Render a quality report as Markdown."""
    lines = [
        "# Regime quality report",
        "",
        f"- Run: `{report.get('run_dir', '')}`",
        f"- Data through: {report.get('data_through', 'n/a')}",
        f"- Detectors: {', '.join(report.get('detectors', []))}",
        f"- States: {report.get('n_states', 'n/a')} "
        f"({', '.join(report.get('state_names', []))})",
        "",
    ]

    stability = report.get("stability", {})
    if stability:
        lines += [
            "## Stability",
            "",
            _table(
                [
                    "Detector",
                    "Switches/yr",
                    "Median run",
                    "Runs <5d",
                    "Days in runs <5d",
                    "Refit-day churn",
                ],
                [
                    [
                        name,
                        _num(values.get("switches_per_year"), 1),
                        _num(values.get("median_run_days"), 0),
                        _pct(values.get("share_runs_under_5d")),
                        _pct(values.get("share_days_in_runs_under_5d")),
                        _ratio(values.get("refit_churn_ratio")),
                    ]
                    for name, values in stability.items()
                ],
            ),
            "",
            "A useful regime persists for weeks. High switch counts or a "
            "refit-day churn well above 1x means the model is reacting to "
            "estimation noise, not the market.",
            "",
        ]

    consensus = report.get("consensus", {})
    if consensus:
        lines += [
            "## Cross-model consensus",
            "",
            f"- All selected models agree on **{_pct(consensus.get('all_agree_share'))}** of days.",
            f"- Mean pairwise exact agreement: "
            f"**{_pct(consensus.get('mean_pairwise_agreement'))}**.",
            "",
        ]

    recovery = report.get("recovery", {})
    if recovery:
        lines += [
            "## Recovery against synthetic truth",
            "",
            _table(
                [
                    "Detector",
                    "Exact",
                    "Within 1",
                    "ARI",
                    "Median lag",
                    "Recall @21d",
                    "Churn vs truth",
                ],
                [
                    [
                        name,
                        _pct(values.get("exact_accuracy")),
                        _pct(values.get("within_one_state")),
                        _num(values.get("adjusted_rand_index"), 3),
                        _num(values.get("median_lag_days"), 0) + "d",
                        _pct(values.get("recall_at_21d")),
                        _ratio(values.get("churn_ratio"), 2),
                    ]
                    for name, values in recovery.items()
                ],
            ),
            "",
            "States are ordinal by volatility, so *within 1* is the fairer "
            "day-level metric. *Recall @21d* is the share of true transitions "
            "detected within 21 trading days.",
            "",
        ]

    events = report.get("events", {})
    if events:
        lines += [
            "## Stress-event recall",
            "",
            _table(
                ["Detector", "Events in range", "Recall"],
                [
                    [
                        name,
                        values.get("n_events_in_range", 0),
                        _pct(values.get("recall")),
                    ]
                    for name, values in events.items()
                ],
            ),
            "",
            "Recall uses the highest-volatility state within +/-21 calendar "
            "days of each event. It measures coincidence, not prediction.",
            "",
        ]

    separation = report.get("separation", {})
    if separation:
        lines += ["## Regime separation", ""]
        for name, records in separation.items():
            if not records:
                continue
            lines += [
                f"### {name}",
                "",
                _table(
                    ["State", "Share", "Avg run", "Fwd vol 21d", "Fwd ret 21d"],
                    [
                        [
                            row.get("state"),
                            _pct(row.get("share")),
                            _num(row.get("avg_run_days"), 0),
                            _pct(row.get("fwd_vol_21d")),
                            _pct(row.get("fwd_ret_21d"), 2),
                        ]
                        for row in records
                    ],
                ),
                "",
            ]

    calibration = report.get("calibration", {})
    if calibration:
        lines += [
            "## Probability calibration",
            "",
            _table(
                ["Detector", "N", "AUC", "Brier", "Spearman"],
                [
                    [
                        name,
                        values.get("n", 0),
                        _num(values.get("auc"), 3),
                        _num(values.get("brier"), 3),
                        _num(values.get("spearman"), 3),
                    ]
                    for name, values in calibration.items()
                ],
            ),
            "",
            "AUC is for detecting top-tercile forward volatility; 0.5 is "
            "chance. Spearman measures whether the probability rises "
            "monotonically with future volatility.",
            "",
        ]

    lines += ["## Notes", ""]
    lines += [f"- {caveat}" for caveat in report.get("caveats", [])]
    lines.append("")
    return "\n".join(lines)


def write_quality_report(
    report: Mapping[str, Any],
    outdir: str | Path,
) -> tuple[Path, Path]:
    """Write ``regime_quality.json`` and ``regime_quality.md``."""
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / QUALITY_FILENAME
    markdown_path = destination / QUALITY_MARKDOWN_FILENAME
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_quality_markdown(report), encoding="utf-8"
    )
    return json_path, markdown_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="market-regime-quality",
        description=(
            "Evaluate regime recovery, stability, calibration, consensus, "
            "and stress-event recall for a walk-forward run"
        ),
    )
    parser.add_argument(
        "--run-dir",
        default="outputs",
        help="walk-forward output directory (default: outputs)",
    )
    parser.add_argument(
        "--detectors",
        default="",
        help="comma-separated detector subset (default: all with labels)",
    )
    parser.add_argument(
        "--prices-csv",
        default=None,
        help="optional price CSV for calibration and separation from returns",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="skip the built-in stress-event recall scan",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="report directory (default: --run-dir)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    detectors = [
        name.strip() for name in args.detectors.split(",") if name.strip()
    ]
    try:
        prices = None
        if args.prices_csv:
            from .data import load_csv

            prices = load_csv(args.prices_csv)
        report = evaluate_run_dir(
            args.run_dir,
            detectors=detectors or None,
            prices=prices,
            events=None if not args.no_events else {},
            include_default_events=not args.no_events,
        )
        outdir = Path(args.outdir) if args.outdir else Path(args.run_dir)
        json_path, markdown_path = write_quality_report(report, outdir)
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"market-regime-quality: error: {exc}", file=sys.stderr)
        return 2

    print(render_quality_markdown(report))
    print(f"\nWrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
