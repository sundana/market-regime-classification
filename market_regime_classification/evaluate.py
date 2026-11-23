"""Evaluation: regime diagnostics + a deliberately simple economic scorecard.

There is no ground-truth regime label in real markets, so classification
accuracy is not a meaningful primary metric. Instead:

* describe each regime (occupancy, persistence, forward vol/return);
* run a toy allocation rule through each detector's labels;
* compare Sharpe, drawdown and turnover against buy & hold;
* report pairwise agreement between detectors (adjusted Rand index).

The toy allocation can charge a flat transaction cost per unit of absolute
weight change. It still ignores market impact, borrow costs, cash yield and
slippage. It is a diagnostic for whether the regimes separate risk, not a
profitable strategy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .config import TRADING_DAYS
from .walkforward import WalkForwardResult


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def _run_lengths(values: np.ndarray) -> list[tuple[int, int]]:
    """Return (value, length) for each contiguous run."""
    if len(values) == 0:
        return []
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((int(values[start]), i - start))
            start = i
    return out


def regime_stats(
    labels: pd.Series,
    returns: pd.Series,
    primary_prices: pd.Series,
    n_states: int,
    forward_days: int = 21,
) -> pd.DataFrame:
    """Occupancy, persistence and forward outcomes by detected regime."""
    idx = labels.index
    fwd_vol = (
        returns.rolling(forward_days).std().shift(-forward_days)
        * np.sqrt(TRADING_DAYS)
    ).reindex(idx)
    fwd_ret = primary_prices.pct_change(forward_days).shift(-forward_days).reindex(
        idx
    )

    runs = _run_lengths(labels.to_numpy())
    durations: dict[int, list[int]] = {j: [] for j in range(n_states)}
    for value, length in runs:
        if 0 <= value < n_states:
            durations[value].append(length)

    rows = []
    for j in range(n_states):
        mask = labels == j
        d = durations[j]
        rows.append(
            {
                "state": j,
                "share": float(mask.mean()),
                "avg_run_days": float(np.mean(d)) if d else 0.0,
                "n_runs": len(d),
                "fwd_vol_21d": float(fwd_vol[mask].mean()),
                "fwd_ret_21d": float(fwd_ret[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Economic scorecard
# ---------------------------------------------------------------------------
def perf_stats(returns: pd.Series) -> dict[str, float]:
    """CAGR / vol / Sharpe / max drawdown / Calmar for a return series."""
    r = pd.Series(returns).dropna().astype(float)
    n = len(r)
    if n == 0:
        return {
            "n_days": 0,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "hit_rate": np.nan,
        }

    curve = (1.0 + r).cumprod()
    final = float(curve.iloc[-1])
    cagr = final ** (TRADING_DAYS / n) - 1.0 if final > 0 else np.nan
    sd = float(r.std(ddof=1)) if n > 1 else np.nan
    # Guard against floating-point dust in near-constant return series.
    sd_is_zero = (not np.isfinite(sd)) or sd < 1e-12
    vol = 0.0 if sd_is_zero and n > 1 else float(sd * np.sqrt(TRADING_DAYS))
    sharpe = (
        0.0
        if sd_is_zero
        else float(r.mean() / sd * np.sqrt(TRADING_DAYS))
    )
    drawdown = float((curve / curve.cummax() - 1.0).min())
    calmar = cagr / abs(drawdown) if np.isfinite(cagr) and drawdown < 0 else np.nan

    return {
        "n_days": n,
        "cagr": float(cagr),
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": float(calmar) if np.isfinite(calmar) else np.nan,
        "hit_rate": float((r > 0).mean()),
    }


def linear_weight_map(n_states: int) -> dict[int, float]:
    """State 0 -> 100% risk asset, state K-1 -> 0% (linear in between)."""
    if n_states == 1:
        return {0: 1.0}
    return {j: 1.0 - j / (n_states - 1) for j in range(n_states)}


def _turnover(weights: pd.Series) -> pd.Series:
    """Absolute weight change at each decision date; zero on the first date."""
    return weights.diff().abs().fillna(0.0)


def strategy_returns(
    labels: pd.Series,
    returns: pd.Series,
    weight_map: dict[int, float],
    *,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Apply weights decided at close ``t`` to the next day's return.

    The signal at ``t`` uses features through ``t``; the return realized is
    ``returns.shift(-1)[t]``. No look-ahead.

    ``cost_bps`` is a flat one-way cost in basis points charged on the absolute
    weight change at each rebalance. The opening position at the first
    observation is not charged, so the strategy and buy & hold follow the same
    convention.
    """
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("cost_bps must be a finite non-negative number")

    next_ret = returns.shift(-1).reindex(labels.index)
    weights = labels.map(weight_map)
    if weights.isna().any():
        unknown = sorted(
            {repr(value) for value in labels[weights.isna()].unique()}
        )
        raise ValueError(f"labels not covered by weight_map: {unknown}")
    weights = weights.astype(float)
    cost_rate = cost_bps / 10_000.0
    net = weights * next_ret - _turnover(weights) * cost_rate
    return net.dropna()


def compare_results(
    results: dict[str, WalkForwardResult],
    returns: pd.Series,
    n_states: int,
    weight_map: dict[int, float] | None = None,
    *,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Economic comparison across detectors plus buy & hold.

    ``sharpe``/``cagr``/``max_drawdown`` are net of ``cost_bps``; the
    ``gross_*`` columns repeat the headline metrics before costs, and
    ``ann_cost`` is the annualized cost drag as a fraction of the portfolio.
    """
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("cost_bps must be a finite non-negative number")
    if weight_map is None:
        weight_map = linear_weight_map(n_states)
    cost_rate = cost_bps / 10_000.0

    rows: list[dict[str, Any]] = []
    all_dates: pd.Index | None = None
    for name, result in results.items():
        weights = result.labels.map(weight_map)
        if weights.isna().any():
            unknown = sorted(
                {
                    repr(value)
                    for value in result.labels[weights.isna()].unique()
                }
            )
            raise ValueError(
                f"{name}: labels not covered by weight_map: {unknown}"
            )
        weights = weights.astype(float)
        turnover = _turnover(weights)

        net = strategy_returns(
            result.labels, returns, weight_map, cost_bps=cost_bps
        )
        gross = strategy_returns(
            result.labels, returns, weight_map, cost_bps=0.0
        )
        net_stats = perf_stats(net)
        gross_stats = perf_stats(gross)
        switches = int((result.labels.diff().fillna(0) != 0).sum())
        years = max(len(result.labels) / TRADING_DAYS, 1e-9)
        row: dict[str, Any] = {
            **net_stats,
            "detector": name,
            "turnover": float(turnover.mean()),
            "switches_per_year": switches / years,
            "avg_state": float(result.labels.mean()),
            "cost_bps": float(cost_bps),
            "gross_cagr": gross_stats["cagr"],
            "gross_sharpe": gross_stats["sharpe"],
            "ann_cost": float(
                turnover.mean() * cost_rate * TRADING_DAYS
            ),
        }
        rows.append(row)
        dates = net.index
        all_dates = dates if all_dates is None else all_dates.intersection(dates)

    if all_dates is None:
        raise ValueError("no results supplied")

    bh_returns = returns.shift(-1).reindex(all_dates).dropna()
    bh_stats = perf_stats(bh_returns)
    bh_row: dict[str, Any] = {
        **bh_stats,
        "detector": "buy_hold",
        "turnover": 0.0,
        "switches_per_year": 0.0,
        "avg_state": np.nan,
        "cost_bps": float(cost_bps),
        "gross_cagr": bh_stats["cagr"],
        "gross_sharpe": bh_stats["sharpe"],
        "ann_cost": 0.0,
    }
    rows.append(bh_row)

    cols = [
        "detector",
        "n_days",
        "cagr",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "turnover",
        "switches_per_year",
        "avg_state",
        "cost_bps",
        "gross_cagr",
        "gross_sharpe",
        "ann_cost",
    ]
    return pd.DataFrame(rows)[cols]


def pairwise_agreement(results: dict[str, WalkForwardResult]) -> pd.DataFrame:
    """Adjusted Rand index between detector labels on shared OOS dates."""
    names = list(results)
    mat = pd.DataFrame(
        np.eye(len(names)), index=names, columns=names, dtype=float
    )
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = results[a].labels.index.intersection(results[b].labels.index)
            score = adjusted_rand_score(
                results[a].labels.loc[common], results[b].labels.loc[common]
            )
            mat.loc[a, b] = score
            mat.loc[b, a] = score
    return mat
