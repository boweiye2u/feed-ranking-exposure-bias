# M2.3 FS3 quality report

Machine-readable report: `~/recom/processed/fs3/v1/data_quality.json`.

## Output integrity and coverage

| Check | Normal train | Random train |
|---|---:|---:|
| Rows / unique impression IDs | 722,481 / 722,481 | 337,933 / 337,933 |
| Zero prior user-category affinity | 195,208 (27.0191%) | 177,295 (52.4645%) |
| Missing prior-day user recency | 7,948 (1.1001%) | 2,879 (0.8519%) |
| Missing prior-day category recency | 195,279 (27.0289%) | 177,295 (52.4645%) |
| Empty sequence | 2,971 (0.4112%) | 2,352 (0.6960%) |
| Mean sequence length | 18.1474 | 17.3796 |
| Full 20-event sequence | 535,412 (74.1074%) | 235,953 (69.8224%) |
| Tie-boundary group dropped | 47,446 | 14,648 |
| Temporal violations | 0 | 0 |
| Missing sequence impression IDs | 0 | 0 |

Every FS2 column is an exact prefix of FS3, and the production value/order hash matches. FS3 schemas are identical across the two training sources. Labels remain separate and are byte-identical to the corresponding FS2 label files. Feature files contain no outcome columns.

## Leakage controls

- Aggregate affinity and recency use only normal-policy rows from strictly earlier calendar days.
- Neural histories use only normal-policy events with timestamps strictly earlier than the target timestamp.
- Randomized outcomes are not loaded as history; the sequence does not contain outcomes from any source.
- Same-day aggregate outcomes cannot influence one another.
- Equal timestamps cannot enter one another's history, and a last-N boundary never splits a timestamp tie group.
- Existing authorization checks continue to reject randomized feature-history outcomes, randomized validation outcomes, and all final-test outcomes.
- No validation or test FS3 artifact was created.

Synthetic tests cover FS2 preservation, own/same-day/future outcome invariance, normal-only history, correct affinity and smoothing, deterministic multi-tag and missing-tag handling, recency defaults, cold starts, schema alignment, strict sequence cutoff, tie invariance, explicit unknown-video encoding, and the pre-existing embargo safeguards.

## Resources and artifacts

- Runtime: 32.14 seconds.
- Peak resident memory: 2,752.6 MiB.
- Total M2.3 artifact size: 103,493,922 bytes (approximately 98.7 MiB), including labels copied for alignment and the JSON audit.
- Complete test suite: 59 passed in 3.65 seconds.

## Unresolved decisions

1. The release provides no semantic codebook for the 46 numeric tag IDs. They remain opaque; the complete tag-set representation should be reviewed before modeling if atomic-tag pooling is desired.
2. Engagement signals are intentionally absent from sequence tensors. Adding long-view/like or interface-masked click history would require an explicit neural-input decision before model implementation.
3. `time_ms` supports strict numeric ordering, but the dataset still does not document its timezone/event-clock semantics. Recency is therefore reported only as elapsed milliseconds, not local-time behavior.
