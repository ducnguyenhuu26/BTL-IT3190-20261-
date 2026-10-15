# Optimization regression and timing

One CPU wall-time measurement per implementation on this machine; includes function execution, excludes data loading. Speedups are local measurements, not general hardware guarantees. No model retraining or new test evaluation.

| Operation | Rows | Before (s) | After (s) | Speedup | Verification |
| --- | ---: | ---: | ---: | ---: | --- |
| Features: all Favorita rows | 3,000,888 | 11.711 | 6.338 | 1.85x | Exact equality, all columns and rows |
| Online backtest: full validation | 162,162 | 31.110 | 0.461 | 67.47x | max prediction difference 8.73e-10; states/diagnostics match |
| Metrics: full validation | 162,162 | 0.084 | 0.053 | 1.59x | All metrics match within 1e-10 |

Original implementation: `provenance/ede2f98b9ae4/`; SHA-256 verified before use. Historical model/prediction/lock files are retained. Report-only audits may verify that archived source; prediction/test stages still require unchanged current code.

Publish verification on 2026-10-15:

- Local suite: 43 tests passed, including real Beijing coverage.
- Clean staged checkout: 43 tests ran successfully; one optional Beijing-data integration test was skipped because raw datasets are excluded from Git.
- All 71 protected experiment artifacts retained their original SHA-256.
- Git export preserved every historical source checksum. No raw/processed datasets, trained models, large predictions, virtual environment, or Python caches were staged.
