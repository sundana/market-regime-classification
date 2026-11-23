"""End-to-end CLI.

Examples
--------
    uv run market-regime --source synthetic --outdir outputs
    uv run market-regime --source yfinance --tickers SPY,TLT --start 2005-01-01
    uv run market-regime --source csv --csv data/prices.csv --outdir outputs

``uv run python -m market_regime_classification.run ...`` is equivalent to the console script.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path

import pandas as pd

from .config import DEFAULT_DETECTORS, Config
from .data import DEFAULT_MAX_STALE_DAYS, load_csv, load_yfinance, synthetic_market
from .detectors import make_detector
from .evaluate import (
    compare_results,
    linear_weight_map,
    pairwise_agreement,
    regime_stats,
)
from .features import build_features, daily_returns
from .metadata import collect_run_metadata, write_run_metadata
from .models import save_model
from .plotting import plot_results
from .walkforward import run_walkforward


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline market-regime detection with walk-forward evaluation"
    )
    p.add_argument(
        "--source",
        choices=("synthetic", "yfinance", "csv"),
        default="synthetic",
        help="data source (default: synthetic, no network required)",
    )
    p.add_argument("--tickers", default="SPY,TLT", help="comma-separated tickers")
    p.add_argument("--start", default="2005-01-01", help="yfinance start date")
    p.add_argument("--end", default=None, help="yfinance end date")
    p.add_argument("--csv", default="data/prices.csv", help="CSV path for --source csv")
    p.add_argument(
        "--n-days", type=int, default=5_000, help="synthetic history length"
    )
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument(
        "--train-days", type=int, default=756, help="trailing training window"
    )
    p.add_argument(
        "--refit-every", type=int, default=63, help="out-of-sample days between refits"
    )
    p.add_argument("--n-states", type=int, default=3, help="number of regimes")
    p.add_argument(
        "--detectors",
        default=",".join(DEFAULT_DETECTORS),
        help="comma-separated: vol_threshold,gmm,hmm,jump",
    )
    p.add_argument("--outdir", default="outputs", help="output directory")
    p.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help=(
            "one-way transaction cost in basis points charged on the absolute "
            "weight change at each rebalance (default: 0)"
        ),
    )
    p.add_argument(
        "--save-models",
        action="store_true",
        help="write the final fitted model per detector to <outdir>/models/",
    )
    p.add_argument(
        "--max-stale-days",
        type=int,
        default=DEFAULT_MAX_STALE_DAYS,
        help=(
            "maximum consecutive missing price days to forward-fill; "
            f"longer gaps are an error (default: {DEFAULT_MAX_STALE_DAYS})"
        ),
    )
    p.add_argument("--no-plot", action="store_true", help="skip the diagnostic figure")
    return p.parse_args(argv)


def _load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series | None]:
    if args.source == "synthetic":
        return synthetic_market(n_days=args.n_days, seed=args.seed)
    if args.source == "yfinance":
        return (
            load_yfinance(
                args.tickers,
                start=args.start,
                end=args.end,
                max_stale_days=args.max_stale_days,
            ),
            None,
        )
    if args.source == "csv":
        return load_csv(args.csv, max_stale_days=args.max_stale_days), None
    raise ValueError(args.source)


def _source_metadata(args: argparse.Namespace) -> dict[str, object]:
    if args.source == "synthetic":
        return {"type": "synthetic", "n_days": args.n_days, "seed": args.seed}
    if args.source == "yfinance":
        return {
            "type": "yfinance",
            "tickers": args.tickers,
            "start": args.start,
            "end": args.end,
        }
    return {"type": "csv", "path": str(args.csv)}


def _print_regime_tables(
    results: dict, returns: pd.Series, primary_prices: pd.Series, n_states: int
) -> None:
    for name, result in results.items():
        stats = regime_stats(result.labels, returns, primary_prices, n_states)
        print(f"\n--- {name}: regime diagnostics ---")
        print(stats.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))


def main(argv: list[str] | None = None) -> int:
    run_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(run_argv)
    if not math.isfinite(args.cost_bps) or args.cost_bps < 0.0:
        raise SystemExit("--cost-bps must be a finite non-negative number")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prices, true_state = _load_data(args)
    if prices.shape[1] == 0:
        raise SystemExit("no price columns loaded")

    features = build_features(prices)
    cfg = Config(
        n_states=args.n_states,
        train_days=args.train_days,
        refit_every=args.refit_every,
        random_state=args.seed,
    )
    feature_cols = cfg.feature_columns(list(features.columns))

    if cfg.train_days >= len(features):
        raise SystemExit(
            f"train_days={cfg.train_days} leaves no out-of-sample data "
            f"({len(features)} feature rows). Lower --train-days or load more history."
        )

    primary = prices.columns[0]
    primary_prices = prices[primary]
    returns = daily_returns(prices, primary)

    print(
        f"Data: {primary}+{prices.shape[1] - 1} asset(s), "
        f"{prices.index[0].date()} -> {prices.index[-1].date()} "
        f"({len(prices)} rows, {len(features)} feature rows)"
    )
    print(
        f"Walk-forward: train={cfg.train_days}d, refit every {cfg.refit_every}d, "
        f"{cfg.n_states} states, features={feature_cols}"
    )

    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    if not detectors:
        raise SystemExit("no detectors selected")

    results = {}
    for name in detectors:
        print(f"Fitting {name} across walk-forward windows ...", flush=True)
        factory = partial(
            make_detector,
            name,
            n_states=cfg.n_states,
            random_state=cfg.random_state,
            gmm_n_init=cfg.gmm_n_init,
            hmm_n_init=cfg.hmm_n_init,
            hmm_max_iter=cfg.hmm_max_iter,
            hmm_stickiness=cfg.hmm_stickiness,
            hmm_cov_shrinkage=cfg.hmm_cov_shrinkage,
            jump_penalty=cfg.jump_penalty,
            jump_n_init=cfg.jump_n_init,
            jump_max_iter=cfg.jump_max_iter,
        )
        results[name] = run_walkforward(
            features,
            factory,
            name=name,
            train_days=cfg.train_days,
            refit_every=cfg.refit_every,
            feature_cols=feature_cols,
            vol_col=cfg.vol_col,
        )

    written: list[Path] = []

    if args.save_models:
        for name, result in results.items():
            if result.final_model is None:
                raise RuntimeError(f"{name}: no fitted model to save")
            written.append(
                save_model(
                    result.final_model,
                    outdir / "models" / f"{name}.json",
                )
            )

    # Persist features and per-detector outputs.
    features_path = outdir / "features.csv"
    features.to_csv(features_path)
    written.append(features_path)
    for name, result in results.items():
        probs_path = outdir / f"{name}_probabilities.csv"
        labels_path = outdir / f"{name}_labels.csv"
        stats_path = outdir / f"{name}_regime_stats.csv"
        result.probs.to_csv(probs_path)
        result.labels.to_frame().to_csv(labels_path)
        regime_stats(
            result.labels, returns, primary_prices, cfg.n_states
        ).to_csv(stats_path, index=False)
        written.extend([probs_path, labels_path, stats_path])

    if true_state is not None:
        true_states_path = outdir / "true_states.csv"
        true_state.to_csv(true_states_path)
        written.append(true_states_path)

    metrics = compare_results(
        results, returns, cfg.n_states, cost_bps=args.cost_bps
    )
    metrics_csv = outdir / "metrics.csv"
    metrics_json = outdir / "metrics.json"
    metrics.to_csv(metrics_csv, index=False)
    metrics.to_json(metrics_json, orient="records", indent=2)
    written.extend([metrics_csv, metrics_json])

    agreement = pairwise_agreement(results)
    agreement_path = outdir / "agreement.csv"
    agreement.to_csv(agreement_path)
    written.append(agreement_path)

    print("\n=== Economic scorecard (walk-forward, out-of-sample) ===")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    if len(results) > 1:
        print("\n=== Pairwise label agreement (adjusted Rand index) ===")
        print(agreement.to_string(float_format=lambda v: f"{v:0.3f}"))

    _print_regime_tables(results, returns, primary_prices, cfg.n_states)

    if true_state is not None:
        from sklearn.metrics import adjusted_rand_score

        print("\n=== Sanity check vs synthetic latent state ===")
        print(
            "ARI is expected to be modest (roughly 0.1-0.3): features are "
            "trailing windows, so detected regimes lag the latent switch."
        )
        for name, result in results.items():
            common = result.labels.index.intersection(true_state.index)
            ari = adjusted_rand_score(
                true_state.loc[common], result.labels.loc[common]
            )
            print(f"{name:14s} ARI = {ari:.3f}")

    if not args.no_plot:
        weight_map = linear_weight_map(cfg.n_states)
        outpath = plot_results(
            primary_prices,
            returns,
            results,
            outdir / "regimes.png",
            weight_map=weight_map,
            n_states=cfg.n_states,
            cost_bps=args.cost_bps,
        )
        written.append(Path(outpath))
        print(f"\nFigure: {outpath}")

    metadata = collect_run_metadata(
        argv=run_argv,
        config={
            **asdict(cfg),
            "feature_cols": feature_cols,
            "detectors": detectors,
            "max_stale_days": args.max_stale_days,
            "cost_bps": args.cost_bps,
            "save_models": args.save_models,
        },
        source=_source_metadata(args),
        outputs=written,
        base_dir=outdir,
    )
    metadata_path = write_run_metadata(outdir, metadata)
    print(f"Metadata: {metadata_path}")

    print(f"\nOutputs written to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
