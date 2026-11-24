"""Trader-readable regime brief.

The brief is the product layer on top of the walk-forward outputs. It answers,
in plain language:

* what volatility regime the models say we are in now;
* how confident the reference model is, and how stable the call has been;
* whether the selected models agree;
* the evidence (realised vol, drawdown, trend, correlation);
* the historical average forward volatility and return for this state; and
* the caveats: this is a nowcast, not a direction forecast or investment advice.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .regime_labels import (
    state_color_names,
    state_description,
    state_names,
)
from .regime_quality import (
    consensus_series,
    current_run_length,
    discover_detectors,
    finite_or_none,
    jsonable,
    label_runs,
    load_indexed_csv,
    load_json,
    switches_in_window,
)

BRIEF_SCHEMA_VERSION = 1
BRIEF_JSON_FILENAME = "regime_brief.json"
BRIEF_MARKDOWN_FILENAME = "regime_brief.md"
DEFAULT_STABILITY_MIN_RUN = 5
DEFAULT_FLAP_WINDOW = 20
DEFAULT_FLAP_SWITCHES = 3
PREFERRED_DETECTORS = ("jump", "hmm", "vol_threshold")
FEATURE_KEYS = (
    "rv_5",
    "rv_21",
    "rv_63",
    "vol_ratio",
    "trend_21",
    "trend_63",
    "drawdown_63",
    "corr_21",
)
_COLOR_EMOJI = {"green": "🟢", "amber": "🟡", "red": "🔴"}


def _select_detectors(
    available: Sequence[str],
    requested: Sequence[str] | None,
) -> list[str]:
    if requested:
        selected = [str(name) for name in requested]
        missing = [name for name in selected if name not in available]
        if missing:
            raise ValueError(
                f"missing labels for detectors {missing}; available: {list(available)}"
            )
        return selected
    preferred = [name for name in PREFERRED_DETECTORS if name in available]
    return preferred or list(available[:1])


def _asset_label(source: Any) -> str:
    if not isinstance(source, Mapping):
        return "unknown"
    kind = source.get("type")
    if kind == "yfinance":
        return str(source.get("tickers", "unknown"))
    if kind == "csv":
        return Path(str(source.get("path", "prices.csv"))).name
    if kind == "synthetic":
        return "synthetic"
    return str(kind or "unknown")


def _probability(
    probs: pd.DataFrame | None,
    date: pd.Timestamp,
    state: int,
) -> float | None:
    if probs is None or date not in probs.index:
        return None
    column = f"state_{state}"
    if column not in probs.columns:
        return None
    return finite_or_none(probs.loc[date, column])


def _regime_stats_row(
    root: Path,
    detector: str,
    state: int,
) -> pd.Series | None:
    path = root / f"{detector}_regime_stats.csv"
    if not path.exists():
        return None
    stats = pd.read_csv(path)
    if "state" not in stats.columns:
        return None
    rows = stats.loc[stats["state"] == state]
    if rows.empty:
        return None
    return rows.iloc[0]


def _stability_badge(
    run_length: int,
    switches_recent: int,
    minimum_run: int,
    flap_switches: int,
) -> str:
    if run_length >= minimum_run and switches_recent <= flap_switches:
        return "stable"
    if run_length >= 3:
        return "building"
    return "flapping"


def _read_text(name: str, agreement: str) -> str:
    if name == "STRESSED":
        text = (
            "Risk is elevated. This is not a directional sell call: "
            "high-volatility states can occur in rallies and selloffs, and "
            "they describe risk, not expected direction."
        )
    elif name == "ELEVATED":
        text = (
            "Volatility is above normal, so risk is rising. The state does "
            "not tell you which way the market will move."
        )
    elif name == "CALM":
        text = (
            "Realised volatility is low. Calm regimes can persist, but they "
            "do not imply direction or a durable trend."
        )
    elif name == "NORMAL":
        text = "Volatility is near its normal range; there is no strong risk signal."
    else:
        text = (
            f"The model places the market in {name}. This is a volatility "
            "state, not a direction forecast."
        )
    if agreement == "split":
        text += " Selected models disagree, so treat the call as provisional."
    return text


def generate_brief(
    run_dir: str | Path,
    *,
    detectors: Sequence[str] | None = None,
    primary: str | None = None,
    stability_min_run: int = DEFAULT_STABILITY_MIN_RUN,
    flap_window: int = DEFAULT_FLAP_WINDOW,
    flap_switches: int = DEFAULT_FLAP_SWITCHES,
) -> dict[str, Any]:
    """Build a trader-readable brief from a walk-forward output directory."""
    if stability_min_run < 1:
        raise ValueError("stability_min_run must be >= 1")
    if flap_window < 1:
        raise ValueError("flap_window must be >= 1")
    if flap_switches < 0:
        raise ValueError("flap_switches must be >= 0")
    root = Path(run_dir)
    features_path = root / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found")
    features = load_indexed_csv(features_path)
    if features.empty:
        raise ValueError(f"{features_path} is empty")

    available = discover_detectors(root)
    if not available:
        raise ValueError(f"no *_labels.csv files found in {root}")
    selected = _select_detectors(available, detectors)

    if primary is None:
        primary = selected[0]
    elif primary not in selected:
        raise ValueError(
            f"primary={primary!r} must be one of the selected detectors {selected}"
        )

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

    common = labels[selected[0]].index
    for name in selected[1:]:
        common = common.intersection(labels[name].index)
    if common.empty:
        raise ValueError("selected detectors have no overlapping out-of-sample dates")
    common = common.sort_values()
    labels = {name: series.reindex(common) for name, series in labels.items()}
    latest_date = pd.Timestamp(common[-1])
    latest_labels = {
        name: int(series.loc[latest_date]) for name, series in labels.items()
    }

    metadata: Mapping[str, Any] = {}
    metadata_path = root / "run_metadata.json"
    if metadata_path.exists():
        loaded = load_json(metadata_path)
        if isinstance(loaded, Mapping):
            metadata = loaded
    config = metadata.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    n_states = int(config.get("n_states", 0) or 0)
    n_states = max(
        n_states,
        max(int(series.max()) for series in labels.values()) + 1,
    )
    names = state_names(n_states)
    color_names = state_color_names(n_states)

    def name_for(state: int) -> str:
        return names[state] if 0 <= state < len(names) else f"STATE {state}"

    consensus = consensus_series(labels)
    latest_consensus = consensus.loc[latest_date]
    unique_latest = set(latest_labels.values())
    if len(unique_latest) == 1:
        agreement = "unanimous"
        headline = int(next(iter(unique_latest)))
    elif pd.notna(latest_consensus):
        agreement = "majority"
        headline = int(latest_consensus)
    else:
        agreement = "split"
        headline = latest_labels[primary]

    if latest_labels[primary] == headline:
        reference = primary
    else:
        reference = next(
            (name for name in selected if latest_labels[name] == headline),
            primary,
        )

    reference_labels = labels[reference]
    run_length = current_run_length(reference_labels)
    recent_switches = switches_in_window(reference_labels, flap_window)
    badge = _stability_badge(
        run_length, recent_switches, stability_min_run, flap_switches
    )

    runs = label_runs(reference_labels)
    since: str | None = None
    if not runs.empty and int(runs.iloc[-1]["state"]) == headline:
        since = pd.Timestamp(runs.iloc[-1]["start"]).date().isoformat()

    stats_row = _regime_stats_row(root, reference, headline)
    typical_run: float | None = None
    forward_vol: float | None = None
    forward_return: float | None = None
    if stats_row is not None:
        typical_run = finite_or_none(stats_row.get("avg_run_days"))
        forward_vol = finite_or_none(stats_row.get("fwd_vol_21d"))
        forward_return = finite_or_none(stats_row.get("fwd_ret_21d"))
    if typical_run is None or typical_run <= 0:
        state_runs = runs.loc[runs["state"] == headline]
        typical_run = (
            finite_or_none(state_runs["length"].mean())
            if not state_runs.empty
            else None
        )

    models: list[dict[str, Any]] = []
    for name in selected:
        own_state = latest_labels[name]
        models.append(
            {
                "name": name,
                "label": own_state,
                "label_name": name_for(own_state),
                "color": color_names[own_state]
                if 0 <= own_state < len(color_names)
                else "amber",
                "probability_of_call": _probability(
                    probs.get(name), latest_date, own_state
                ),
                "probability_of_headline": _probability(
                    probs.get(name), latest_date, headline
                ),
                "run_length": current_run_length(labels[name]),
                "switches_recent": switches_in_window(labels[name], flap_window),
                "agrees": own_state == headline,
            }
        )

    latest_features = (
        features.loc[latest_date]
        if latest_date in features.index
        else features.iloc[-1]
    )
    evidence = {
        key: finite_or_none(latest_features[key])
        for key in FEATURE_KEYS
        if key in features.columns
    }

    confidence = _probability(probs.get(reference), latest_date, headline)
    source = metadata.get("source", {})
    asset = _asset_label(source)
    caveats = [
        f"Nowcast: uses data through {latest_date.date().isoformat()}; regime "
        "turns are typically identified 6-12 trading sessions late.",
        "A regime is a volatility state, not a direction forecast. "
        "High-volatility states can occur in rallies and selloffs.",
        "Probabilities are model scores, not calibrated real-world probabilities.",
        "Informational only; not investment advice.",
    ]
    if len(selected) == 1:
        caveats.append(
            "No cross-model consensus: only one detector was selected."
        )
    if agreement == "split":
        caveats.append(
            "Selected models disagree; the headline is the primary model's call."
        )
    if "gmm" in selected:
        caveats.append(
            "GMM has no temporal persistence and was poorly calibrated in "
            "review; treat its call as a weak signal."
        )

    return {
        "schema_version": BRIEF_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of": latest_date.date().isoformat(),
        "data_through": latest_date.date().isoformat(),
        "asset": asset,
        "n_states": n_states,
        "primary_model": primary,
        "reference_model": reference,
        "regime": {
            "state": headline,
            "name": name_for(headline),
            "color": color_names[headline]
            if 0 <= headline < len(color_names)
            else "amber",
            "description": state_description(headline, n_states),
            "confidence": confidence,
            "stability": badge,
            "run_length": run_length,
            "switches_recent": recent_switches,
            "stability_window": flap_window,
            "since": since,
            "typical_run_days": typical_run,
        },
        "consensus": {
            "label": int(latest_consensus)
            if pd.notna(latest_consensus)
            else None,
            "agreement": agreement,
            "n_models": len(selected),
            "n_agree": sum(
                1 for state in latest_labels.values() if state == headline
            ),
            "disagreeing": [
                name
                for name in selected
                if latest_labels[name] != headline
            ],
        },
        "models": models,
        "evidence": evidence,
        "forward_context": {
            "state": headline,
            "fwd_vol_21d": forward_vol,
            "fwd_ret_21d": forward_return,
        },
        "freshness": {
            "run_generated_at_utc": metadata.get("generated_at_utc"),
            "source": source,
            "train_days": config.get("train_days"),
            "refit_every": config.get("refit_every"),
        },
        "read": _read_text(name_for(headline), agreement),
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _pct(value: Any, digits: int = 0, signed: bool = False) -> str:
    number = finite_or_none(value)
    if number is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{100 * number:{sign}.{digits}f}%"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def _format_feature(key: str, value: Any) -> str:
    number = finite_or_none(value)
    if number is None:
        return "n/a"
    if key in {"trend_21", "trend_63", "drawdown_63"}:
        return _pct(number, 1, signed=True)
    if key in {"rv_5", "rv_21", "rv_63"}:
        return _pct(number, 1)
    if key == "corr_21":
        return f"{number:+.2f}"
    if key == "vol_ratio":
        return f"{number:.2f}x"
    return f"{number:.2f}"


def render_brief(brief: Mapping[str, Any]) -> str:
    """Render a brief as Markdown suitable for a terminal or a daily note."""
    regime = brief.get("regime", {})
    consensus = brief.get("consensus", {})
    name = str(regime.get("name", "UNKNOWN"))
    color = str(regime.get("color", "amber"))
    emoji = _COLOR_EMOJI.get(color, "🟡")
    agreement = str(consensus.get("agreement", "unknown"))
    models = brief.get("models", [])
    evidence = brief.get("evidence", {})
    forward = brief.get("forward_context", {})
    freshness = brief.get("freshness", {})
    source = freshness.get("source", {})
    asset = brief.get("asset", "unknown")

    typical_run = finite_or_none(regime.get("typical_run_days"))
    typical_text = (
        "n/a"
        if typical_run is None
        else _plural(round(typical_run), "session")
    )
    stability_window = regime.get("stability_window", DEFAULT_FLAP_WINDOW)
    run_length = int(regime.get("run_length", 0) or 0)
    recent_switches = int(regime.get("switches_recent", 0) or 0)

    lines = [
        f"# MARKET REGIME BRIEF — {asset} — {brief.get('as_of', 'n/a')}",
        "",
    ]
    if agreement == "split":
        lines += [
            "## ⚠️ MODEL DISAGREEMENT",
            "",
            f"Primary model **{brief.get('primary_model', 'n/a')}** calls "
            f"**{emoji} {name}**, but the selected models do not agree.",
        ]
    else:
        lines += [
            f"## {emoji} {name} — {regime.get('description', '')}",
            "",
            f"State {regime.get('state', 'n/a')} of "
            f"{brief.get('n_states', 'n/a')} (calm → stressed).",
        ]

    lines += [
        "",
        f"- **Reference model:** {brief.get('reference_model', 'n/a')}",
        f"- **Confidence:** {_pct(regime.get('confidence'))} "
        "(model score, not a calibrated probability)",
        f"- **Stability:** **{str(regime.get('stability', 'n/a')).upper()}** — "
        f"{_plural(run_length, 'session')} in this run, "
        f"{_plural(recent_switches, 'switch', 'switches')} in the last "
        f"{stability_window} sessions",
        f"- **Since:** {regime.get('since') or 'n/a'}",
        f"- **Typical run in this state:** {typical_text}",
        "",
        "## Model agreement",
        "",
        f"Agreement: **{agreement}** "
        f"({consensus.get('n_agree', 0)}/{consensus.get('n_models', 0)} agree).",
        "",
    ]

    lines += [
        "| Model | Call | P(call) | P(headline) | Current run | Switches/20d |",
        "|---|---|---|---|---|---|",
    ]
    for model in models:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(model.get("name", "")),
                    f"{_COLOR_EMOJI.get(str(model.get('color', 'amber')), '🟡')} "
                    f"{model.get('label_name', model.get('label', ''))}",
                    _pct(model.get("probability_of_call")),
                    _pct(model.get("probability_of_headline")),
                    f"{model.get('run_length', 0)}d",
                    str(model.get("switches_recent", 0)),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## What we see",
        "",
        f"- 21d realised vol: **{_format_feature('rv_21', evidence.get('rv_21'))}**",
        f"- 5d realised vol: {_format_feature('rv_5', evidence.get('rv_5'))}",
        f"- 63d realised vol: {_format_feature('rv_63', evidence.get('rv_63'))}",
        f"- Drawdown from 63d high: "
        f"{_format_feature('drawdown_63', evidence.get('drawdown_63'))}",
        f"- 21d trend: {_format_feature('trend_21', evidence.get('trend_21'))}",
        f"- 63d trend: {_format_feature('trend_63', evidence.get('trend_63'))}",
        f"- 21d stock/bond correlation: "
        f"{_format_feature('corr_21', evidence.get('corr_21'))}",
        "",
        "## Historical context",
        "",
        f"- Average forward 21d volatility in this state: "
        f"**{_pct(forward.get('fwd_vol_21d'))}**",
        f"- Average forward 21d return in this state: "
        f"**{_pct(forward.get('fwd_ret_21d'), 2, signed=True)}**",
        "",
        "These are historical averages for the state, not forecasts.",
        "",
        "## Read",
        "",
        str(brief.get("read", "")),
        "",
        "## Caveats",
        "",
    ]
    lines += [f"- {caveat}" for caveat in brief.get("caveats", [])]
    lines += [
        "",
        "---",
        f"Data through **{brief.get('data_through', 'n/a')}** | "
        f"Run generated {freshness.get('run_generated_at_utc') or 'n/a'} | "
        f"Source: {source.get('type', 'n/a') if isinstance(source, Mapping) else 'n/a'}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="market-regime-brief",
        description="Write a plain-language regime brief from walk-forward outputs",
    )
    parser.add_argument(
        "--run-dir",
        default="outputs",
        help="walk-forward output directory (default: outputs)",
    )
    parser.add_argument(
        "--detectors",
        default="",
        help=(
            "comma-separated detector subset; default uses jump, hmm, and "
            "vol_threshold when present (GMM is excluded by default)"
        ),
    )
    parser.add_argument(
        "--primary",
        default=None,
        help="primary model for split decisions (default: first selected)",
    )
    parser.add_argument(
        "--stability-min-run",
        type=int,
        default=DEFAULT_STABILITY_MIN_RUN,
        help="minimum sessions for a STABLE badge (default: 5)",
    )
    parser.add_argument(
        "--flap-window",
        type=int,
        default=DEFAULT_FLAP_WINDOW,
        help="window in sessions for the flapping check (default: 20)",
    )
    parser.add_argument(
        "--flap-switches",
        type=int,
        default=DEFAULT_FLAP_SWITCHES,
        help="switches in the window before the badge drops (default: 3)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="brief directory (default: --run-dir)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    detectors = [
        name.strip() for name in args.detectors.split(",") if name.strip()
    ]
    try:
        brief = generate_brief(
            args.run_dir,
            detectors=detectors or None,
            primary=args.primary,
            stability_min_run=args.stability_min_run,
            flap_window=args.flap_window,
            flap_switches=args.flap_switches,
        )
        outdir = Path(args.outdir) if args.outdir else Path(args.run_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        json_path = outdir / BRIEF_JSON_FILENAME
        markdown_path = outdir / BRIEF_MARKDOWN_FILENAME
        json_path.write_text(
            json.dumps(jsonable(brief), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        markdown = render_brief(brief)
        markdown_path.write_text(markdown, encoding="utf-8")
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"market-regime-brief: error: {exc}", file=sys.stderr)
        return 2

    print(markdown)
    print(f"\nWrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
