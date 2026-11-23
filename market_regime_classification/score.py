"""Score new price data with a saved fitted regime model.

This is the inference half of the pipeline: it never refits. It rebuilds the
feature frame from price history, applies the model's frozen feature contract,
and writes per-date state probabilities, an argmax label, and a
``last_probs.json`` vector that can be carried into the next scoring run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import DEFAULT_MAX_STALE_DAYS, load_csv, load_yfinance
from .detectors import BaseDetector
from .features import build_features
from .metadata import collect_run_metadata, sha256_file, write_run_metadata
from .models import load_model

LAST_PROBS_FILENAME = "last_probs.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="market-regime-score",
        description="Score new price data with a saved fitted regime model",
    )
    p.add_argument(
        "--model",
        required=True,
        help="path to a model JSON artifact written by --save-models",
    )
    p.add_argument(
        "--source",
        choices=("csv", "yfinance"),
        default="csv",
        help="data source (default: csv)",
    )
    p.add_argument(
        "--csv", default="data/prices.csv", help="CSV path for --source csv"
    )
    p.add_argument(
        "--tickers",
        default="SPY,TLT",
        help="comma-separated tickers for --source yfinance",
    )
    p.add_argument("--start", default="2005-01-01", help="yfinance start date")
    p.add_argument("--end", default=None, help="yfinance end date")
    p.add_argument(
        "--max-stale-days",
        type=int,
        default=DEFAULT_MAX_STALE_DAYS,
        help=(
            "maximum consecutive missing price days to forward-fill; "
            f"longer gaps are an error (default: {DEFAULT_MAX_STALE_DAYS})"
        ),
    )
    p.add_argument(
        "--init-probs",
        default=None,
        help=(
            "optional JSON probability vector (or a previous last_probs.json) "
            "used to continue HMM filtering from the prior scoring block"
        ),
    )
    p.add_argument(
        "--outdir", default="scores", help="output directory (default: scores)"
    )
    return p.parse_args(argv)


def _load_prices(args: argparse.Namespace) -> pd.DataFrame:
    if args.source == "csv":
        return load_csv(args.csv, max_stale_days=args.max_stale_days)
    return load_yfinance(
        args.tickers,
        start=args.start,
        end=args.end,
        max_stale_days=args.max_stale_days,
    )


def _source_info(args: argparse.Namespace) -> dict[str, object]:
    if args.source == "csv":
        return {"type": "csv", "path": str(args.csv)}
    return {
        "type": "yfinance",
        "tickers": args.tickers,
        "start": args.start,
        "end": args.end,
    }


def _load_init_probs(path: str | Path, n_states: int) -> np.ndarray:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("probabilities")
    try:
        probs = np.asarray(payload, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "init-probs must be a numeric probability vector"
        ) from exc
    if probs.shape != (n_states,):
        raise ValueError(
            f"init-probs must have {n_states} entries, got {probs.size}"
        )
    if not np.isfinite(probs).all() or (probs < 0.0).any():
        raise ValueError("init-probs must be finite and non-negative")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("init-probs must have a positive sum")
    return probs / total


def _check_feature_contract(
    model: BaseDetector, features: pd.DataFrame
) -> None:
    missing = [col for col in model.feature_cols_ if col not in features.columns]
    if missing:
        raise ValueError(
            f"model expects feature columns {missing} that this data cannot "
            f"produce (available: {list(features.columns)}). Multi-asset "
            "models need the same asset columns and enough history for the "
            "63-day feature warmup."
        )


def _run(args: argparse.Namespace, run_argv: list[str]) -> int:
    model_path = Path(args.model)
    model = load_model(model_path)

    init_probs = (
        None
        if args.init_probs is None
        else _load_init_probs(args.init_probs, model.n_states)
    )

    prices = _load_prices(args)
    features = build_features(prices)
    _check_feature_contract(model, features)

    probs = np.asarray(
        model.predict(
            features,
            feature_cols=model.feature_cols_,
            init_probs=init_probs,
        ),
        dtype=float,
    )
    expected_shape = (len(features), model.n_states)
    if probs.shape != expected_shape:
        raise RuntimeError(
            f"model returned probabilities with shape {probs.shape}, "
            f"expected {expected_shape}"
        )
    if not np.isfinite(probs).all() or not np.allclose(
        probs.sum(axis=1), 1.0, atol=1e-6
    ):
        raise RuntimeError("model returned invalid probabilities")

    scores = pd.DataFrame(
        probs,
        index=features.index,
        columns=[f"state_{j}" for j in range(model.n_states)],
    )
    scores["label"] = probs.argmax(axis=1)
    scores.index.name = "date"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores_path = outdir / "scores.csv"
    features_path = outdir / "features.csv"
    last_probs_path = outdir / LAST_PROBS_FILENAME
    scores.to_csv(scores_path)
    features.to_csv(features_path)
    last_probs_path.write_text(
        json.dumps(
            {
                "date": str(scores.index[-1]),
                "probabilities": probs[-1].tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written = [scores_path, features_path, last_probs_path]

    print(f"Model: {model.name} ({model.n_states} states) from {model_path}")
    print(
        f"Data: {len(prices)} price rows -> {len(features)} feature rows, "
        f"{features.index[0].date()} -> {features.index[-1].date()}"
    )
    print(
        f"Latest: state {int(probs[-1].argmax())} "
        f"P={np.round(probs[-1], 4).tolist()}"
    )

    metadata = collect_run_metadata(
        argv=run_argv,
        config={
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "detector": model.name,
            "n_states": model.n_states,
            "feature_cols": list(model.feature_cols_),
            "vol_col": model.vol_col_,
            "init_probs": (
                None if init_probs is None else init_probs.tolist()
            ),
        },
        source=_source_info(args),
        outputs=written,
        base_dir=outdir,
    )
    metadata_path = write_run_metadata(outdir, metadata)

    print(f"Outputs written to {outdir.resolve()}")
    print(f"Metadata: {metadata_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    run_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(run_argv)
    try:
        return _run(args, run_argv)
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"market-regime-score: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
