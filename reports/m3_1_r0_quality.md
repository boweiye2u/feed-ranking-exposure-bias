# M3.1 R0 quality report

Machine-readable report: `~/recom/predictions/r0/v1/data_quality.json`.

## Contract and alignment

- Scoring rule: `r0_score = video__hist_long_view_rate`.
- Source: frozen FS2 `Float64` feature from strictly prior-calendar-day normal-policy outcomes.
- Smoothing: fixed `alpha = 20`; burn-in prior `0.3364927893253054` from Apr 9–11 normal rows.
- Outcome columns projected by scorer/loader: none.
- Randomized outcomes used as history: no.
- Validation or final-test outcomes accessed: no.
- Prediction IDs and scores exactly equal the aligned FS2 IDs and source-rate values for both splits.
- Actual AUC or LogLoss computed: no.

## Score diagnostics

| Diagnostic | Normal train | Random train |
|---|---:|---:|
| Rows / unique IDs | 722,481 / 722,481 | 337,933 / 337,933 |
| Missing / NaN scores | 0 / 0 | 0 / 0 |
| Minimum | 0.01073591 | 0.01073591 |
| p01 | 0.04127149 | 0.06980974 |
| p05 | 0.09920291 | 0.11781589 |
| p25 | 0.23145192 | 0.21621902 |
| Median | 0.33346525 | 0.28629096 |
| Mean | 0.33150897 | 0.29554846 |
| p75 | 0.43183717 | 0.36365736 |
| p95 | 0.55499240 | 0.50948875 |
| p99 | 0.62535470 | 0.59401531 |
| Maximum | 0.73464501 | 0.73464501 |
| Standard deviation | 0.13826432 | 0.11478424 |
| Distinct scores | 28,098 | 16,577 |
| Zero video history | 514 (0.07114%) | 1,785 (0.52821%) |

All scores lie in `[0,1]`. Zero-history rows receive the same frozen prior through FS2; R0 adds no fallback or imputation of its own.

## Resources

- Runtime: 0.325 seconds.
- Peak RSS: 305.2 MiB.
- Prediction artifacts and manifests: 4,460,419 bytes (approximately 4.25 MiB).
- Focused R0 tests: 8 passed in 0.37 seconds.
- Complete test suite: 77 passed in 4.06 seconds.
