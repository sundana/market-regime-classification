# Market Regime Classification

[![PyPI](https://img.shields.io/pypi/v/market-regime-classification.svg)](https://pypi.org/project/market-regime-classification/)
[![Python versions](https://img.shields.io/pypi/pyversions/market-regime-classification.svg)](https://pypi.org/project/market-regime-classification/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A causal, walk-forward baseline for **detecting market regimes** with a
volatility rule, a Gaussian mixture, and a Gaussian HMM — plus the evaluation
discipline that separates a real signal from a look-ahead artifact.

This is not a trading system. It is a runnable reference for the three things
regime research usually gets wrong:

- **Filtered inference.** State probabilities at time `t` use only data up to
  `t`; no forward-backward smoothing.
- **Walk-forward evaluation.** Fit on a trailing window, predict the next block
  out-of-sample, refit, repeat.
- **Economic evaluation.** "Regime" has no ground truth, so classification
  accuracy is mostly meaningless. The question is whether regime-conditional
  allocations improve risk-adjusted return at acceptable turnover.

| | |
|:---|:---|
| Distribution | `market-regime-classification` |
| Import package | `market_regime_classification` |
| CLI command | `market-regime` |
| Python | 3.10+ |

## Demo

![Walk-forward diagnostics: detected regimes shaded on price, filtered state probabilities, and equity curves](docs/regimes-example.png)

*Default synthetic run: HMM regimes shaded on price, causal filtered
probabilities (no smoothing), and walk-forward equity curves for all three
detectors plus buy & hold.*

```text
uv run market-regime --source synthetic --outdir outputs
Data: EQ+1 asset(s), 2007-09-24 -> 2026-11-20 (5000 rows, 4937 feature rows)
Walk-forward: train=756d, refit every 63d, 3 states, features=[...]
Fitting vol_threshold across walk-forward windows ...
Fitting gmm across walk-forward windows ...
Fitting hmm across walk-forward windows ...

=== Economic scorecard (walk-forward, out-of-sample) ===
     detector  n_days   cagr  ann_vol  sharpe  max_drawdown  turnover  switches_per_year
vol_threshold    4180 0.0272   0.0858  0.3554       -0.2373    0.0324            16.2736
          gmm    4180 0.0549   0.1288  0.4791       -0.3458    0.0507            20.7338
          hmm    4180 0.0612   0.1296  0.5235       -0.3245    0.0169             7.5944
     buy_hold    4180 0.0493   0.2298  0.3241       -0.5626    0.0000             0.0000
...
Figure: outputs/regimes.png
```

That excerpt is one seeded synthetic run (`--seed 42`), not a performance
claim. On real data the ordering can and does change.

## Models

| Name | Type | Notes |
|---|---|---|
| `vol_threshold` | Rule baseline | Trailing-window quantiles of 21-day realized vol. Hard labels, one-hot "probabilities". |
| `gmm` | Unsupervised ML | Gaussian mixture on standardized features; per-sample posterior, no temporal persistence. |
| `hmm` | Unsupervised ML | Full-covariance Gaussian HMM with EM (Baum-Welch), **forward-filtered only** at prediction time. |

All three run through the same walk-forward harness and the same economic
scorecard. If the ML models cannot beat a simple vol-quantile rule
out-of-sample, that is a finding, not a bug. States are ordered by mean
volatility after every refit, so `state_0` is always the calmest regime.

## Contents

- [Demo](#demo)
- [Models](#models)
- [Install](#install)
- [Development setup (uv)](#development-setup-uv)
- [Quick start](#quick-start)
- [CLI options](#cli-options)
- [Outputs](#outputs)
- [Engineered features](#engineered-features)
- [Python API](#python-api)
- [Reading the results](#reading-the-results)
- [Deliberate limitations](#deliberate-limitations)
- [The look-ahead trap](#the-look-ahead-trap)
- [Project layout](#project-layout)
- [Testing and contributing](#testing-and-contributing)
- [Publishing to PyPI](#publishing-to-pypi)
- [License](#license)

## Install

Requires Python 3.10 or newer. The `yfinance` data source additionally needs
network access; the default `synthetic` source runs fully offline.

```bash
pip install market-regime-classification
market-regime --help
```

## Development setup (uv)

The project is managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml`
declares dependencies and the `market-regime` console script; `uv.lock` pins
the full resolution.

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv and install the project plus dev dependencies from uv.lock
uv sync
```

## Quick start

```bash
# No network needed: synthetic market with known latent regimes
# (a couple of minutes for the default 5,000 days, mostly HMM refits)
uv run market-regime --source synthetic --outdir outputs

# Real data via yfinance
uv run market-regime --source yfinance --tickers SPY,TLT \
    --start 2005-01-01 --outdir outputs

# Your own price CSV (date in the first column, then one or more price columns)
uv run market-regime --source csv --csv data/prices.csv --outdir outputs
```

`uv run market-regime` invokes the console script declared in `pyproject.toml`:

```toml
[project.scripts]
market-regime = "market_regime_classification.run:main"
```

`python -m market_regime_classification.run ...` is equivalent. Run the tests
with:

```bash
uv run pytest -q
```

For a non-uv environment, export a plain requirements file:

```bash
uv export --format requirements-txt --no-hashes > requirements.txt
```

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--source {synthetic,yfinance,csv}` | `synthetic` | Data source. `synthetic` needs no network. |
| `--tickers` | `SPY,TLT` | Comma-separated tickers for `--source yfinance`. |
| `--start` | `2005-01-01` | Start date for `yfinance`. |
| `--end` | now | End date for `yfinance`. |
| `--csv` | `data/prices.csv` | Price CSV for `--source csv`. |
| `--n-days` | `5000` | Synthetic history length (minimum 300). |
| `--seed` | `42` | Random seed for synthetic data and model initialization. |
| `--train-days` | `756` | Trailing training window (~3 years). |
| `--refit-every` | `63` | Out-of-sample days between refits (~quarterly). |
| `--n-states` | `3` | Number of regimes; states sorted lowest → highest volatility. |
| `--detectors` | `vol_threshold,gmm,hmm` | Comma-separated detector subset. |
| `--outdir` | `outputs` | Output directory (created if missing). |
| `--no-plot` | off | Skip the diagnostic figure. |

## Outputs

```text
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

`metrics.csv` reports `cagr`, `ann_vol`, `sharpe`, `max_drawdown`, `calmar`,
`hit_rate`, `turnover`, `switches_per_year`, and `avg_state` for each detector
and for buy & hold. `*_regime_stats.csv` reports occupancy, run length, and
forward 21-day volatility/return per state.

## Engineered features

Built from daily prices only, with trailing windows and no look-ahead:

- `rv_5`, `rv_21`, `rv_63` — annualized realized volatility
- `vol_ratio` — `rv_5 / rv_63` (vol acceleration)
- `trend_21`, `trend_63` — log-price momentum
- `drawdown_63` — drawdown from the trailing 63-day high
- `corr_21` — average pairwise correlation (when ≥ 2 assets)

`rv_21` is the reference feature for ordering states and for the rule
baseline's thresholds. The final frame is `dropna`'d, so the first 63 rows are
lost to the warmup for every model equally.

## Python API

Every stage is importable, so the CLI is optional:

```python
from market_regime_classification.config import Config
from market_regime_classification.data import synthetic_market
from market_regime_classification.detectors import GaussianHMMDetector
from market_regime_classification.features import build_features
from market_regime_classification.walkforward import run_walkforward

prices, true_state = synthetic_market(n_days=3000, seed=0)
features = build_features(prices)

cfg = Config(n_states=3, train_days=756, refit_every=63, random_state=0)
cols = cfg.feature_columns(list(features.columns))

result = run_walkforward(
    features,
    lambda: GaussianHMMDetector(n_states=cfg.n_states, random_state=cfg.random_state),
    name="hmm",
    train_days=cfg.train_days,
    refit_every=cfg.refit_every,
    feature_cols=cols,
    vol_col=cfg.vol_col,
)
print(result.labels.tail())
print(result.probs.tail())
```

`result.probs` is indexed by out-of-sample date with one `state_k` column per
regime; `result.labels` is the per-day argmax.

## Reading the results

- **`share` / `avg_run_days`** — a regime that flips every few days is not a
  regime. Real volatility regimes persist for weeks to months.
- **`fwd_vol_21d` / `fwd_ret_21d`** — forward 21-day realized volatility and
  return by regime. Good separation here is the main evidence the detector
  found something.
- **Sharpe / MaxDD / turnover** — the toy allocator maps state 0 → 100%
  equity, middle states linearly, last state → 0%. It ignores costs, cash
  yield, and slippage; treat it as a diagnostic, not a strategy.
- **ARI between detectors** — how much the models agree. Very high agreement
  with little economic difference means the HMM is not earning its complexity.

On the synthetic sample, ARI against the *latent* state is typically only
~0.1–0.3 even when forward-vol separation is strong. That is expected: every
feature is a trailing window, so detection necessarily lags each switch, and
ARI penalizes every lagged day. The informative diagnostics are
regime-conditional forward volatility/return and the economic scorecard.

## Deliberate limitations

This is a baseline, not a production system:

- Single asset target; multi-asset regime work needs covariance/correlation
  state modeling.
- No transaction costs, borrow costs, or cash yield in the scorecard.
- No hyperparameter search. `n_states`, `train_days`, and `refit_every` are
  fixed by CLI flags; choosing them on the test period would be data snooping.
- The HMM is full-covariance with light regularization. It can still find
  local optima; `n_init` restarts mitigate but do not eliminate this.
- The synthetic generator is a sanity check, not evidence about real markets.

## The look-ahead trap

An HMM fitted on the full sample and queried with `predict_proba` returns
**smoothed** state probabilities. At time `t` they incorporate observations
after `t`. Plots look excellent; live results do not.

`market_regime_classification.detectors` runs its own forward recursion and
only ever consumes data up to `t`. The HMM's EM uses forward-backward on the
training window only, which is legitimate because that window is entirely in
the past. `tests/test_detectors.py` and `tests/test_walkforward.py` verify that
truncating future rows leaves past probabilities unchanged.

## Project layout

```text
pyproject.toml       # metadata, deps, console script, pytest/uv config
uv.lock              # pinned dependency resolution (commit this)
.python-version      # interpreter uv should use
LICENSE              # MIT
MANIFEST.in          # files included in the sdist
.github/workflows/
└── publish.yml      # tests, build, PyPI trusted publishing
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
├── test_package.py  # metadata + console-script registration
├── test_features.py
├── test_detectors.py
└── test_walkforward.py
```

## Testing and contributing

Bug reports, feature requests, and pull requests are welcome at
[github.com/sundana/market-regime-classification](https://github.com/sundana/market-regime-classification).

Before opening a pull request:

1. Fork the repo and create a branch from `main`.
2. Run `uv sync`, make your change, and add or update tests.
3. Run `uv run pytest -q`; the suite must pass.
4. Keep changes focused and explain *why* in the PR description, especially
   for anything that affects evaluation or could introduce look-ahead.

The suite covers feature causality, walk-forward structure, HMM filtering
continuity, scorecard math, and package/console-script registration.

To add a detector, implement the `BaseDetector` interface in
`market_regime_classification/detectors.py`, register it in `make_detector()`,
and add a causality test showing that truncating future rows does not change
past labels or probabilities.

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

Or use `.github/workflows/publish.yml` with PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/). The workflow
runs on `v*` tags, published GitHub releases, or manual dispatch; it installs
locked dependencies, runs the tests, builds, runs `twine check`, and uploads
via OIDC — no long-lived token in the repository. Configure the PyPI project to
trust this repository and the `publish` workflow before tagging a release.

## License

MIT. See [LICENSE](LICENSE).
