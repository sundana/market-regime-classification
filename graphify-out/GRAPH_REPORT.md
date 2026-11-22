# Graph Report - market-regime-classification  (2026-11-22)

## Corpus Check
- 18 files · ~8,065 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 4, .in 1, .lock 1)

## Summary
- 213 nodes · 452 edges · 12 communities (8 shown, 4 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d4d22d7d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Evaluation & Scorecard
- Config & Data Loading
- Synthetic Market & Features
- PyPI Publish Workflow
- Detector Concepts & Evaluation
- detectors.py
- Diagnostic Plotting
- CLAUDE.md
- run_walkforward
- Synthetic Generator Module
- Project Root

## God Nodes (most connected - your core abstractions)
1. `GaussianHMMDetector` - 18 edges
2. `build_features()` - 15 edges
3. `main()` - 15 edges
4. `synthetic_market()` - 15 edges
5. `run_walkforward()` - 14 edges
6. `VolThresholdDetector` - 13 edges
7. `GMMDetector` - 13 edges
8. `compare_results()` - 13 edges
9. `regime_stats()` - 12 edges
10. `plot_results()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `test_factory()` --uses--> `VolThresholdDetector`  [INFERRED]
  tests/test_detectors.py → market_regime_classification/detectors.py
- `test_vol_threshold()` --calls--> `VolThresholdDetector`  [EXTRACTED]
  tests/test_detectors.py → market_regime_classification/detectors.py
- `test_compare_results_and_agreement()` --calls--> `VolThresholdDetector`  [EXTRACTED]
  tests/test_walkforward.py → market_regime_classification/detectors.py
- `test_walkforward_has_no_lookahead()` --calls--> `VolThresholdDetector`  [EXTRACTED]
  tests/test_walkforward.py → market_regime_classification/detectors.py
- `test_factory()` --uses--> `GaussianHMMDetector`  [INFERRED]
  tests/test_detectors.py → market_regime_classification/detectors.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Evaluation Discipline for Regime Models** — readme_filtered_causal_inference, readme_walk_forward_evaluation, readme_economic_evaluation [EXTRACTED 1.00]
- **Look-Ahead Avoidance Mechanism** — readme_detectors_module, readme_causal_forward_filtering, readme_forward_backward_smoothing, readme_causality_test [EXTRACTED 1.00]
- **Regime Detector Suite Sharing One Harness and Scorecard** — readme_vol_threshold, readme_gmm, readme_hmm, readme_walk_forward_harness, readme_economic_scorecard [EXTRACTED 1.00]

## Communities (12 total, 4 thin omitted)

### Community 0 - "Evaluation & Scorecard"
Cohesion: 0.16
Nodes (10): compare_results(), linear_weight_map(), pairwise_agreement(), perf_stats(), regime_stats(), _run_lengths(), strategy_returns(), test_compare_results_and_agreement() (+2 more)

### Community 1 - "Config & Data Loading"
Cohesion: 0.12
Nodes (7): Config, load_csv(), load_yfinance(), _load_data(), main(), parse_args(), _print_regime_tables()

### Community 2 - "Synthetic Market & Features"
Cohesion: 0.10
Nodes (16): synthetic_market(), build_features(), daily_returns(), _check_probs(), data(), test_gmm(), test_hmm(), test_hmm_block_filtering_matches_concatenation() (+8 more)

### Community 3 - "PyPI Publish Workflow"
Cohesion: 0.09
Nodes (21): environment, permissions, runs-on, steps, name, url, jobs, build-and-publish (+13 more)

### Community 4 - "Detector Concepts & Evaluation"
Cohesion: 0.16
Nodes (17): Adjusted Rand Index, Causal Forward Filtering, Causality Test, market_regime_classification.detectors, Economic Scorecard, Regime-Conditional Forward Volatility and Return Separation, GMM Detector, HMM Detector (+9 more)

### Community 5 - "detectors.py"
Cohesion: 0.10
Nodes (10): BaseDetector, _gaussian_logpdf(), GaussianHMMDetector, GMMDetector, _logsumexp(), make_detector(), _safe_cholesky(), VolThresholdDetector (+2 more)

### Community 6 - "Diagnostic Plotting"
Cohesion: 0.26
Nodes (5): _colors(), plot_results(), _shade_regimes(), _sharpe(), WalkForwardResult

### Community 13 - "run_walkforward"
Cohesion: 0.18
Nodes (4): _Detector, run_walkforward(), test_walkforward_gmm_and_hmm(), test_walkforward_has_no_lookahead()

## Knowledge Gaps
- **5 isolated node(s):** `market-regime-classification`, `market_regime_classification.data`, `Synthetic Market Generator`, `HMM Local Optima Risk`, `PyPI Trusted Publishing`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 82 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GaussianHMMDetector` connect `detectors.py` to `Evaluation & Scorecard`, `Synthetic Market & Features`, `run_walkforward`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **What connects `market-regime-classification`, `market_regime_classification.data`, `Synthetic Market Generator` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Config & Data Loading` be split into smaller, more focused modules?**
  _Cohesion score 0.1168091168091168 - nodes in this community are weakly interconnected._
- **Why does `synthetic_market()` connect `Synthetic Market & Features` to `Evaluation & Scorecard`, `Config & Data Loading`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Should `Synthetic Market & Features` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._
- **Why does `run_walkforward()` connect `run_walkforward` to `Evaluation & Scorecard`, `Config & Data Loading`, `Synthetic Market & Features`, `detectors.py`, `Diagnostic Plotting`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Should `PyPI Publish Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.09090909090909091 - nodes in this community are weakly interconnected._