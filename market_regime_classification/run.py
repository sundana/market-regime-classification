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
import sys
from pathlib import Path

import pandas as pd

from .config import DEFAULT_DETECTORS, Config
from .data import load_csv, load_yfinance, synthetic_market
from .detectors import make_detector
from .evaluate import (
    compare_results,
    linear_weight_map,
    pairwise_agreement,
    regime_stats,
)
from .features import build_features, daily_returns
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
        help="comma-separated: vol_threshold,gmm,hmm",
    )
    p.add_argument("--outdir", default="outputs", help="output directory")
    p.add_argument("--no-plot", action="store_true", help="skip the diagnostic figure")
    return p.parse_args(argv)


def _load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series | None]:
    if args.source == "synthetic":
        return synthetic_market(n_days=args.n_days, seed=args.seed)
    if args.source == "yfinance":
        return load_yfinance(args.tickers, start=args.start, end=args.end), None
    if args.source == "csv":
        return load_csv(args.csv), None
    raise ValueError(args.source)


def _print_regime_tables(
    results: dict, returns: pd.Series, primary_prices: pd.Series, n_states: int
) -> None:
    for name, result in results.items():
        stats = regime_stats(result.labels, returns, primary_prices, n_states)
        print(f"\n--- {name}: regime diagnostics ---")
        print(stats.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
        factory = lambda name=name: make_detector(
            name,
            n_states=cfg.n_states,
            random_state=cfg.random_state,
            gmm_n_init=cfg.gmm_n_init,
            hmm_n_init=cfg.hmm_n_init,
            hmm_max_iter=cfg.hmm_max_iter,
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

    # Persist features and per-detector outputs.
    features.to_csv(outdir / "features.csv")
    for name, result in results.items():
        result.probs.to_csv(outdir / f"{name}_probabilities.csv")
        result.labels.to_frame().to_csv(outdir / f"{name}_labels.csv")
        regime_stats(
            result.labels, returns, primary_prices, cfg.n_states
        ).to_csv(outdir / f"{name}_regime_stats.csv", index=False)

    if true_state is not None:
        true_state.to_csv(outdir / "true_states.csv")

    metrics = compare_results(results, returns, cfg.n_states)
    metrics.to_csv(outdir / "metrics.csv", index=False)
    metrics.to_json(outdir / "metrics.json", orient="records", indent=2)

    agreement = pairwise_agreement(results)
    agreement.to_csv(outdir / "agreement.csv")

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
        )
        print(f"\nFigure: {outpath}")

    print(f"\nOutputs written to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
