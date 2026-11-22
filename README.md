# Market Regime Classification — Baseline Pipeline

[![PyPI](https://img.shields.io/pypi/v/market-regime-classification.svg)](https://pypi.org/project/market-regime-classification/)
[![Python versions](https://img.shields.io/pypi/pyversions/market-regime-classification.svg)](https://pypi.org/project/market-regime-classification/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A deliberately simple, honest baseline for **detecting market regimes** with
statistical and machine-learning models, plus the evaluation discipline that
usually decides whether a regime model is real or an artifact.

The point of this scaffold is not to produce a trading system. It is to give
you a runnable reference for:

1. **Filtered (causal) inference** — never forward-backward-smoothed
   probabilities, which secretly use future data.
2. **Walk-forward evaluation** — refit on a trailing window, predict the next
   block out-of-sample.
3. **Economic evaluation** — because "regime" has no ground truth, accuracy is
   mostly meaningless. What matters is whether regime-conditional allocations
   improve risk-adjusted return with acceptable turnover.

Distribution name: `market-regime-classification`.
Import package: `market_regime_classification`.
CLI command: `market-regime`.

## Models included

| Name | Type | Notes |
|---|---|---|
| `vol_threshold` | Rule baseline | Trailing-window quantiles of 21-day realized vol. Hard labels, one-hot "probabilities". |
| `gmm` | Unsupervised ML | Gaussian mixture on standardized features; per-sample posterior, no temporal persistence. |
| `hmm` | Unsupervised ML | Full-covariance Gaussian HMM with EM (Baum-Welch), **forward-filtered only** at prediction time. States sorted by volatility. |

All three run through the same walk-forward harness and the same economic
scorecard. If the ML models cannot beat a simple vol quantile rule
out-of-sample, that is a finding, not a bug.

## Install from PyPI

```bash
pip install market-regime-classification

market-regime --help
market-regime --source synthetic --outdir outputs
```

## Development setup (uv)

This project is managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml`
declares the dependencies and the `market-regime` console script; `uv.lock`
pins the full resolution.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv, install the project + dev dependencies from uv.lock
uv sync
```

## Quick start

```bash
# No network needed: synthetic market with known latent regimes
# (~1–2 minutes for the default 5,000 days, mostly the HMM refits)
uv run market-regime --source synthetic --outdir outputs

# Real data (yfinance is a project dependency)
uv run market-regime --source yfinance --tickers SPY,TLT \
    --start 2005-01-01 --outdir outputs

# Your own price CSV (date index, one or more price columns)
uv run market-regime --source csv --csv data/prices.csv --outdir outputs
```

`uv run market-regime` invokes the console script declared in `pyproject.toml`:

```toml
[project.scripts]
market-regime = "market_regime_classification.run:main"
```

`uv run python -m market_regime_classification.run ...` is equivalent.

Run the tests:

```bash
uv run pytest -q
```

If you need a plain requirements file for a non-uv environment:

```bash
uv export --format requirements-txt --no-hashes > requirements.txt
```

## What gets written

```
outputs/
├── features.csv
├── vol_threshold_probabilities.csv
├── vol_threshold_labels.csv
├── vol_threshold_regime_stats.csv
├── gmm_*.csv
├── hmm_*.csv
├── metrics.csv              # economic scorecard, all detectors + buy & hold
├── metrics.json
├── agreement.csv            # pairwise adjusted Rand index between detectors
├── regimes.png              # price shading, filtered probabilities, equity curves
└── true_states.csv          # synthetic source only
```

## Features

Built from daily prices only (no look-ahead):

- `rv_5`, `rv_21`, `rv_63` — annualized realized volatility
- `vol_ratio` — `rv_5 / rv_63` (vol acceleration)
- `trend_21`, `trend_63` — log-price momentum
- `drawdown_63` — drawdown from the trailing 63-day high
- `corr_21` — average pairwise correlation (when >= 2 assets)

The reference feature for ordering/sorting regimes is `rv_21`. States are
always sorted low → high volatility so labels are comparable across refits.

## How to read the results

- **`share` / `avg_run_days`** — a regime that flips every few days is not a
  regime. Real vol regimes persist for weeks to months.
- **`fwd_vol` / `fwd_ret`** — forward 21-day realized volatility and return by
  regime. Good separation here is the main evidence the detector found
  something.
- **Sharpe / MaxDD / turnover** — the toy allocator maps state 0 → 100%
  equity, state 1 → 50%, state 2 → 0%. It ignores costs, cash yield and
  slippage; treat it as a diagnostic, not a strategy.
- **ARI between detectors** — how much the models agree. Very high agreement
  plus little economic difference means the HMM is not earning its complexity.

On the synthetic sample, ARI against the *latent* state is typically only
~0.1–0.3 even when forward-vol separation is strong. That is expected, not a
bug: every feature is a trailing window, so detection necessarily lags each
switch, and ARI penalizes every lagged day. The informative diagnostics are
regime-conditional forward volatility/return and the economic scorecard.

## Deliberate limitations

This is a baseline, not a production system:

- Single asset target; multi-asset regime work needs covariance/correlation
  state modeling.
- No transaction costs, borrow costs or cash yield in the scorecard.
- No hyperparameter search. `n_states`, `train_days` and `refit_every` are
  fixed by CLI flags; choosing them on the test period would be data snooping.
- HMM is full-covariance with light regularization. It can still find local
  optima; `n_init` restarts mitigate but do not eliminate this.
- The synthetic generator is a sanity check, not evidence about real markets.

## The trap this pipeline is built to avoid

An HMM fitted on the full sample and queried with `predict_proba` returns
**smoothed** state probabilities. At time `t` they incorporate observations
after `t`. Plots look excellent; live results do not.
`market_regime_classification.detectors` runs its own forward recursion and
only ever consumes data up to `t`. `tests/` includes a causality test that
verifies truncating future rows does not change past probabilities.

## Publishing to PyPI

Build and validate locally:

```bash
uv build
uvx twine check dist/*
```

Publish with an API token:

```bash
UV_PUBLISH_TOKEN=pypi-... uv publish
```

Or use the included GitHub Actions workflow
(`.github/workflows/publish.yml`) with PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/): configure the
PyPI project to trust this repository and the `publish` workflow, then publish
a GitHub release. The workflow builds, runs `twine check`, and uploads via OIDC
— no long-lived token in the repository.

## License

MIT. See [LICENSE](LICENSE).

## Layout

```
pyproject.toml       # metadata, deps, console script, pytest/uv config
uv.lock              # pinned dependency resolution (commit this)
.python-version      # interpreter uv should use
LICENSE              # MIT
MANIFEST.in          # files included in the sdist
.github/workflows/
└── publish.yml      # build + PyPI trusted publishing
market_regime_classification/
├── config.py        # CLI/config defaults
├── data.py          # synthetic generator, yfinance loader, CSV loader
├── features.py      # feature engineering
├── detectors.py     # vol-threshold, GMM, causal Gaussian HMM
├── walkforward.py   # rolling refit / out-of-sample harness
├── evaluate.py      # regime stats + economic scorecard
├── plotting.py      # diagnostic figure
└── run.py           # end-to-end CLI
tests/
├── test_features.py
├── test_detectors.py
└── test_walkforward.py
```
