# Graph Report - market-regime-classification  (2026-11-23)

## Corpus Check
- 27 files · ~15,271 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 4, .in 1, .lock 1)

## Summary
- 387 nodes · 1006 edges · 16 communities (10 shown, 6 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 24 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ef05bb80`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run.py
- metadata.py
- Synthetic Market & Features
- PyPI Publish Workflow
- Detector Concepts & Evaluation
- test_detectors.py
- test_models.py
- CLAUDE.md
- pandas
- ci.yml
- run.py
- Synthetic Generator Module
- Project Root

## God Nodes (most connected - your core abstractions)
1. `GaussianHMMDetector` - 25 edges
2. `BaseDetector` - 24 edges
3. `VolThresholdDetector` - 24 edges
4. `load_csv()` - 23 edges
5. `main()` - 22 edges
6. `build_features()` - 21 edges
7. `synthetic_market()` - 20 edges
8. `GMMDetector` - 17 edges
9. `compare_results()` - 16 edges
10. `collect_run_metadata()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `test_cli_synthetic_end_to_end()` --uses--> `VolThresholdDetector`  [INFERRED]
  tests/test_cli.py → market_regime_classification/detectors.py
- `test_factory()` --uses--> `VolThresholdDetector`  [INFERRED]
  tests/test_detectors.py → market_regime_classification/detectors.py
- `trained()` --calls--> `GaussianHMMDetector`  [EXTRACTED]
  tests/test_score.py → market_regime_classification/detectors.py
- `trained()` --calls--> `save_model()`  [EXTRACTED]
  tests/test_score.py → market_regime_classification/models.py
- `test_score_cli_rejects_missing_feature_contract()` --calls--> `synthetic_market()`  [EXTRACTED]
  tests/test_score.py → market_regime_classification/data.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Evaluation Discipline for Regime Models** — readme_filtered_causal_inference, readme_walk_forward_evaluation, readme_economic_evaluation [EXTRACTED 1.00]
- **Look-Ahead Avoidance Mechanism** — readme_detectors_module, readme_causal_forward_filtering, readme_forward_backward_smoothing, readme_causality_test [EXTRACTED 1.00]
- **Regime Detector Suite Sharing One Harness and Scorecard** — readme_vol_threshold, readme_gmm, readme_hmm, readme_walk_forward_harness, readme_economic_scorecard [EXTRACTED 1.00]

## Communities (16 total, 6 thin omitted)

### Community 0 - "run.py"
Cohesion: 0.07
Nodes (25): compare_results(), linear_weight_map(), pairwise_agreement(), perf_stats(), regime_stats(), _run_lengths(), strategy_returns(), _turnover() (+17 more)

### Community 1 - "metadata.py"
Cohesion: 0.10
Nodes (23): collect_run_metadata(), _git(), _repo_root(), sha256_file(), write_run_metadata(), _check_feature_contract(), _load_init_probs(), _load_prices() (+15 more)

### Community 2 - "Synthetic Market & Features"
Cohesion: 0.08
Nodes (17): synthetic_market(), build_features(), daily_returns(), _check_probs(), data(), test_gmm(), test_hmm(), test_hmm_block_filtering_matches_concatenation() (+9 more)

### Community 3 - "PyPI Publish Workflow"
Cohesion: 0.09
Nodes (21): environment, permissions, runs-on, steps, name, url, jobs, build-and-publish (+13 more)

### Community 4 - "Detector Concepts & Evaluation"
Cohesion: 0.16
Nodes (17): Adjusted Rand Index, Causal Forward Filtering, Causality Test, market_regime_classification.detectors, Economic Scorecard, Regime-Conditional Forward Volatility and Return Separation, GMM Detector, HMM Detector (+9 more)

### Community 5 - "test_detectors.py"
Cohesion: 0.07
Nodes (14): _as_float_array(), _as_permutation(), _as_prob_matrix(), _as_prob_vector(), BaseDetector, _check_symmetric_covars(), _gaussian_logpdf(), GaussianHMMDetector (+6 more)

### Community 6 - "test_models.py"
Cohesion: 0.15
Nodes (14): VolThresholdDetector, detector_from_dict(), load_model(), save_model(), market(), test_load_rejects_bad_transition_matrix(), test_load_rejects_non_object_json(), test_load_rejects_unknown_detector() (+6 more)

### Community 8 - "pandas"
Cohesion: 0.17
Nodes (24): _extract_close_prices(), load_csv(), load_yfinance(), _validate_price_frame(), _frame(), test_extract_close_prices_from_flat_columns(), test_extract_close_prices_from_multiindex(), test_extract_close_prices_reports_missing_ticker() (+16 more)

### Community 9 - "ci.yml"
Cohesion: 0.08
Nodes (28): concurrency, cancel-in-progress, group, jobs, lint, test, runs-on, steps (+20 more)

### Community 12 - "run.py"
Cohesion: 0.15
Nodes (7): Config, _load_data(), main(), parse_args(), _print_regime_tables(), _source_metadata(), test_cli_synthetic_end_to_end()

## Knowledge Gaps
- **5 isolated node(s):** `market-regime-classification`, `market_regime_classification.data`, `Synthetic Market Generator`, `HMM Local Optima Risk`, `PyPI Trusted Publishing`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 123 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseDetector` connect `test_detectors.py` to `metadata.py`, `test_models.py`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `BaseDetector` (e.g. with `detector_from_dict()` and `load_model()`) actually correct?**
  _`BaseDetector` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `market-regime-classification`, `market_regime_classification.data`, `Synthetic Market Generator` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07103825136612021 - nodes in this community are weakly interconnected._
- **Why does `GaussianHMMDetector` connect `test_detectors.py` to `run.py`, `metadata.py`, `Synthetic Market & Features`, `test_models.py`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Should `metadata.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10374149659863946 - nodes in this community are weakly interconnected._
- **Why does `synthetic_market()` connect `Synthetic Market & Features` to `run.py`, `metadata.py`, `test_models.py`, `pandas`, `run.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._