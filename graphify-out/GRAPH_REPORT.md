# Graph Report - market-regime-classification  (2026-11-24)

## Corpus Check
- 32 files · ~24,125 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 4, .in 1, .lock 1)

## Summary
- 515 nodes · 1424 edges · 22 communities (11 shown, 11 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 26 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `51183fb6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run.py
- json
- Synthetic Market & Features
- PyPI Publish Workflow
- Detector Concepts & Evaluation
- detectors.py
- test_models.py
- CLAUDE.md
- pytest
- ci.yml
- Synthetic Generator Module
- Project Root
- pandas
- brief.py

## God Nodes (most connected - your core abstractions)
1. `generate_brief()` - 28 edges
2. `GaussianHMMDetector` - 26 edges
3. `BaseDetector` - 25 edges
4. `load_csv()` - 25 edges
5. `evaluate_run_dir()` - 24 edges
6. `VolThresholdDetector` - 24 edges
7. `main()` - 22 edges
8. `build_features()` - 21 edges
9. `synthetic_market()` - 20 edges
10. `JumpModelDetector` - 18 edges

## Surprising Connections (you probably didn't know these)
- `test_cli_synthetic_end_to_end()` --uses--> `VolThresholdDetector`  [INFERRED]
  tests/test_cli.py → market_regime_classification/detectors.py
- `test_factory()` --uses--> `VolThresholdDetector`  [INFERRED]
  tests/test_detectors.py → market_regime_classification/detectors.py
- `test_jsonable_replaces_non_finite()` --calls--> `jsonable()`  [EXTRACTED]
  tests/test_regime_quality.py → market_regime_classification/regime_quality.py
- `test_factory()` --uses--> `GaussianHMMDetector`  [INFERRED]
  tests/test_detectors.py → market_regime_classification/detectors.py
- `test_hmm()` --calls--> `GaussianHMMDetector`  [EXTRACTED]
  tests/test_detectors.py → market_regime_classification/detectors.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Evaluation Discipline for Regime Models** — readme_filtered_causal_inference, readme_walk_forward_evaluation, readme_economic_evaluation [EXTRACTED 1.00]
- **Look-Ahead Avoidance Mechanism** — readme_detectors_module, readme_causal_forward_filtering, readme_forward_backward_smoothing, readme_causality_test [EXTRACTED 1.00]
- **Regime Detector Suite Sharing One Harness and Scorecard** — readme_vol_threshold, readme_gmm, readme_hmm, readme_walk_forward_harness, readme_economic_scorecard [EXTRACTED 1.00]

## Communities (22 total, 11 thin omitted)

### Community 0 - "run.py"
Cohesion: 0.06
Nodes (31): compare_results(), linear_weight_map(), pairwise_agreement(), perf_stats(), regime_stats(), _run_lengths(), strategy_returns(), _turnover() (+23 more)

### Community 1 - "json"
Cohesion: 0.08
Nodes (22): Config, collect_run_metadata(), _git(), _repo_root(), sha256_file(), write_run_metadata(), parse_args(), _load_data() (+14 more)

### Community 2 - "Synthetic Market & Features"
Cohesion: 0.11
Nodes (18): synthetic_market(), build_features(), daily_returns(), main(), data(), market(), test_daily_returns(), test_features_are_causal() (+10 more)

### Community 3 - "PyPI Publish Workflow"
Cohesion: 0.09
Nodes (21): environment, permissions, runs-on, steps, name, url, jobs, build-and-publish (+13 more)

### Community 4 - "Detector Concepts & Evaluation"
Cohesion: 0.16
Nodes (17): Adjusted Rand Index, Causal Forward Filtering, Causality Test, market_regime_classification.detectors, Economic Scorecard, Regime-Conditional Forward Volatility and Return Separation, GMM Detector, HMM Detector (+9 more)

### Community 5 - "detectors.py"
Cohesion: 0.06
Nodes (21): _as_float_array(), _as_permutation(), _as_prob_matrix(), _as_prob_vector(), BaseDetector, _check_symmetric_covars(), _gaussian_logpdf(), GaussianHMMDetector (+13 more)

### Community 6 - "test_models.py"
Cohesion: 0.13
Nodes (14): VolThresholdDetector, detector_from_dict(), load_model(), save_model(), market(), test_load_rejects_bad_transition_matrix(), test_load_rejects_non_object_json(), test_load_rejects_unknown_detector() (+6 more)

### Community 8 - "pytest"
Cohesion: 0.16
Nodes (24): _extract_close_prices(), load_csv(), load_yfinance(), _validate_price_frame(), _frame(), test_extract_close_prices_from_flat_columns(), test_extract_close_prices_from_multiindex(), test_extract_close_prices_reports_missing_ticker() (+16 more)

### Community 9 - "ci.yml"
Cohesion: 0.08
Nodes (28): concurrency, cancel-in-progress, group, jobs, lint, test, runs-on, steps (+20 more)

### Community 16 - "pandas"
Cohesion: 0.09
Nodes (39): calibration_metrics(), consensus_metrics(), consensus_series(), current_run_length(), discover_detectors(), evaluate_run_dir(), event_recall(), finite_or_none() (+31 more)

### Community 17 - "brief.py"
Cohesion: 0.10
Nodes (23): _asset_label(), _format_feature(), generate_brief(), main(), parse_args(), _pct(), _plural(), _probability() (+15 more)

## Knowledge Gaps
- **5 isolated node(s):** `market-regime-classification`, `market_regime_classification.data`, `Synthetic Market Generator`, `HMM Local Optima Risk`, `PyPI Trusted Publishing`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 162 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseDetector` connect `detectors.py` to `json`, `test_models.py`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **What connects `market-regime-classification`, `market_regime_classification.data`, `Synthetic Market Generator` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06010230179028133 - nodes in this community are weakly interconnected._
- **Why does `GaussianHMMDetector` connect `detectors.py` to `run.py`, `Synthetic Market & Features`, `test_models.py`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Should `json` be split into smaller, more focused modules?**
  _Cohesion score 0.08082706766917293 - nodes in this community are weakly interconnected._
- **Why does `JumpModelDetector` connect `detectors.py` to `run.py`, `test_models.py`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Should `Synthetic Market & Features` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._