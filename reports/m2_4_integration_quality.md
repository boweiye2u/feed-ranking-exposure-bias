# M2.4a feature integration and freeze report

Machine-readable outputs:

- `~/recom/processed/m2/v1/feature_manifest.json`
- `~/recom/processed/m2/v1/integration_quality.json`

## Alignment and nesting

| Check | Normal train | Random train |
|---|---:|---:|
| Rows at each of FS1/FS2/FS3 | 722,481 | 337,933 |
| Positional feature/label matches before repair | 0 | 337,933 |
| Positional feature/label matches after repair | 722,481 | 337,933 |
| Unique, non-null, ascending IDs | Pass | Pass |
| FS2 exact FS1 prefix | Pass | Pass |
| FS3 exact FS2 prefix | Pass | Pass |
| ID-keyed feature equality to snapshot, all stages | Pass | Pass |
| ID-keyed label equality to snapshot, all stages | Pass | Pass |
| ID-keyed sequence equality to snapshot | Pass | Pass |

No outcome columns occur in feature or sequence tables. Labels remain separate and byte-identical across stages within each rebuilt source.
Direct keyed comparison against the released user and basic-video dimension tables found zero static-value mismatches for either source, confirming that the original join reordering did not transfer attributes between impression IDs.

## Sequence audit

| Check | Normal train | Random train |
|---|---:|---:|
| Rows / unique target IDs | 722,481 / 722,481 | 337,933 / 337,933 |
| Valid history positions | 13,111,147 | 5,873,143 |
| Array length | 20 | 20 |
| Mask/length mismatches | 0 | 0 |
| Non-left-padded rows | 0 | 0 |
| Padding ID/time violations | 0 | 0 |
| Timestamp violations | 0 | 0 |
| Target-time mismatches | 0 | 0 |
| Missing normal-history tuple matches | 0 | 0 |
| Unknown-video positions | 0 | 0 |
| Tie-boundary groups dropped deterministically | 47,446 | 14,648 |

Padding ID 0, unknown ID 1, and known encoded IDs 2–7,584 are disjoint. Every valid history tuple matches the permitted Apr 9–29 normal structural history. Random interactions do not supply sequence events.

## Categories and selectors

The approved canonical complete-tag-set representation is unchanged. All category keys and atomic counts match reconstruction from `video_features_basic_pure.csv`: 111 keys across 7,583 videos, including 96 missing-category videos. Normal training observes 92 of those missing-category videos; random training observes all 96.

The baseline and RQ3 random-only selectors exclude `context__is_rand`; only RQ3 joint includes it. The machine-readable manifest lists all 89 persisted FS3 fields and all seven sequence fields with type, source, allowed model families, missing-value policy, and temporal-provenance status.

## Static provenance and resources

Released video upload dates span Apr 9–11 and never postdate an authorized training impression. The release does not document the snapshot cutoff for mutable user counts/status or other basic-video fields; they remain included under the frozen FS1 decision and are flagged in the manifest. Shipped aggregate video statistics remain excluded.

- Full M2 artifact footprint: 190,448,685 bytes (approximately 181.6 MiB).
- Integration-gate runtime: 7.87 seconds.
- Integration-gate peak RSS: 3,703.8 MiB.
- Complete test suite: 69 passed in 3.37 seconds.
- Held-out outcomes accessed: no.
