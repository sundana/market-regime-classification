"""Diagnostic figure: price shading, filtered probabilities, equity curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_hex

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt

from .evaluate import linear_weight_map, strategy_returns
from .walkforward import WalkForwardResult

_STATE_COLORS = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
_LINE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def _colors(n_states: int) -> list[str]:
    if n_states <= len(_STATE_COLORS):
        return _STATE_COLORS[:n_states]
    cmap = plt.get_cmap("tab10")
    return [to_hex(cmap(i % 10)) for i in range(n_states)]


def _shade_regimes(ax: Axes, labels: pd.Series, colors: list[str]) -> None:
    """Shade contiguous runs of the same state."""
    if labels.empty:
        return
    values = labels.to_numpy()
    idx = labels.index
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            state = int(values[start])
            ax.axvspan(
                idx[start],
                idx[i - 1],
                color=colors[state % len(colors)],
                alpha=0.15,
                linewidth=0,
            )
            start = i


def plot_results(
    primary_prices: pd.Series,
    returns: pd.Series,
    results: dict[str, WalkForwardResult],
    outpath: str | Path,
    weight_map: dict[int, float] | None = None,
    n_states: int = 3,
    cost_bps: float = 0.0,
) -> Path:
    if not results:
        raise ValueError("no results to plot")
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Prefer the HMM for the regime panel; otherwise the first detector.
    primary_name = "hmm" if "hmm" in results else next(iter(results))
    primary = results[primary_name]
    colors = _colors(n_states)
    if weight_map is None:
        weight_map = linear_weight_map(n_states)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 12),
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.6]},
        sharex=False,
    )

    # 1) Price with regime shading over the out-of-sample window.
    ax = axes[0]
    oos = primary.labels.index
    ax.plot(
        primary_prices.loc[oos].index,
        primary_prices.loc[oos].to_numpy(),
        color="black",
        linewidth=1.1,
    )
    _shade_regimes(ax, primary.labels, colors)
    ax.set_yscale("log")
    ax.set_title(
        f"Primary asset with {primary_name} regimes "
        f"({oos[0].date()} to {oos[-1].date()})"
    )
    ax.set_ylabel("Price (log scale)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[j], alpha=0.35)
        for j in range(n_states)
    ]
    ax.legend(
        handles, [f"state {j}" for j in range(n_states)], loc="upper left", ncol=n_states
    )

    # 2) Filtered probabilities (causal).
    ax = axes[1]
    P = primary.probs
    ax.stackplot(
        P.index,
        [P[f"state_{j}"].to_numpy() for j in range(n_states)],
        colors=colors,
        alpha=0.85,
        labels=[f"P(state {j})" for j in range(n_states)],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Filtered probability")
    ax.set_title(f"{primary_name}: filtered regime probabilities (no smoothing)")
    ax.legend(loc="upper left", ncol=n_states, fontsize=8)

    # 3) Equity curves of the toy allocation.
    ax = axes[2]
    used_dates = None
    for i, (name, result) in enumerate(results.items()):
        strat = strategy_returns(
            result.labels, returns, weight_map, cost_bps=cost_bps
        )
        equity = (1.0 + strat).cumprod()
        ax.plot(
            equity.index,
            equity.to_numpy(),
            label=f"{name} (Sharpe {_sharpe(strat):.2f})",
            color=_LINE_COLORS[i % len(_LINE_COLORS)],
            linewidth=1.3,
        )
        used_dates = equity.index if used_dates is None else used_dates.intersection(
            equity.index
        )
    if used_dates is not None and len(used_dates):
        bh = (1.0 + returns.shift(-1).reindex(used_dates).fillna(0.0)).cumprod()
        ax.plot(
            bh.index,
            bh.to_numpy(),
            label=f"buy & hold (Sharpe {_sharpe(returns.shift(-1).reindex(used_dates)):.2f})",
            color="black",
            linestyle="--",
            linewidth=1.2,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log scale)")
    cost_note = f", net of {cost_bps:g} bps" if cost_bps else ""
    ax.set_title(
        "Toy allocation: state 0 = 100% risk asset ... last state = 0%"
        f"{cost_note}"
    )
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(252))
