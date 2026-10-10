# M2.2 FS2 quality report

Machine-readable report: `~/recom/processed/fs2/v1/data_quality.json`.

## Output integrity

| Check | Normal train | Random train |
|---|---:|---:|
| Rows / unique impression IDs | 722,481 / 722,481 | 337,933 / 337,933 |
| Zero user-history rows | 7,816 (1.0818%) | 2,876 (0.8511%) |
| Zero video-history rows | 514 (0.0711%) | 1,785 (0.5282%) |
| Rows with any null aggregate | 0 | 0 |
| Missing FS1 impression IDs | 0 | 0 |
| FS1 schema/order preserved | Yes | Yes |

The complete FS2 schema is identical across training sources. Labels remain separate and checksum-verified after copying. No shipped aggregate-video statistics are used.

## Aggregate schema

The following suffixes exist under both `user__hist_` and `video__hist_`:

| Suffix | Type |
|---|---|
| `impression_count` | `Int64` |
| `long_view_positive_count` | `Int64` |
| `long_view_rate` | `Float64` |
| `like_positive_count` | `Int64` |
| `like_rate` | `Float64` |
| `valid_play_impression_count` | `Int64` |
| `valid_play_positive_count` | `Int64` |
| `valid_play_rate` | `Float64` |
| `literal_click_impression_count` | `Int64` |
| `literal_click_positive_count` | `Int64` |
| `literal_click_rate` | `Float64` |

The tab-1 and tab-0 click features have separate denominators and priors. No homogeneous `click_rate` field exists.

## Leakage verification

Synthetic tests establish that changing an impression's own outcome, another same-day outcome, or a future outcome cannot change its FS2 features. Tests also verify correct counts/rates, zero-history smoothing, separate click semantics, exact FS1 carry-through, source-schema equality, rejection of randomized feature history, and continued validation/test embargoes.

Production construction reads normal outcomes only from Apr 9–29. Randomized outcomes are not accepted as a feature-history source. Feature Parquet files contain no label columns, and no validation or test FS2 table is produced.

## Resources

- Runtime: 7.89 seconds.
- Peak resident memory: 1,870.9 MiB.
- Four FS2 Parquet artifacts: 55,908,535 bytes (approximately 53.3 MiB).
