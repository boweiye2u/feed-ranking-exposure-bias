# M1.2 engagement-label and metric-feasibility report

Machine-readable aggregates: `~/recom/processed/audits/m1_2_engagement_diagnostics.json`.

## Access boundary

Engagement columns were analyzed only for randomized impressions dated April 22–29, 2022. April 30–May 2 randomized outcomes and every May 3–8 outcome were excluded. Test overlap calculations projected only `date`, `user_id`, and `video_id`. No predictive model was trained.

## Semantics: facts, observations, and inferences

- **[DOCUMENTED]** `long_view` is 1 when `play_time_ms >= duration_ms` for videos no longer than 18 seconds, or when play reaches 18 seconds for longer videos.
- **[DOCUMENTED]** `is_click` is a literal click in the two-column UI. In the autoplay single-column UI it instead represents valid play: completion for videos no longer than 7 seconds, or play beyond 7 seconds for longer videos.
- **[DOCUMENTED]** `is_like` indicates use of the like action; `play_time_ms` and `duration_ms` are milliseconds.
- **[DOCUMENTED]** `tab` is a scenario code. The dataset owner later stated that the most frequent `tab` is single-column, the second most frequent is two-column, and other tabs are sub-scenarios.
- **[OBSERVED]** Across both complete normal logs, tab 1 is most frequent and tab 0 second. Applying the owner's rule identifies tab 1 as single-column and tab 0 as two-column.
- **[OBSERVED]** The permitted randomized slice contains tabs 1, 11, 2, and 14, but no tab 0. Thus it has no identified two-column impressions under that rule.
- **[INFERRED]** Numeric mapping tab 1 → single-column and tab 0 → two-column is strong release-specific evidence, but is not a published universal codebook for all Kuaishou policies.

Official sources: [dataset field documentation](https://github.com/chongminggao/KuaiRand#data-descriptions), [KuaiRand paper](https://arxiv.org/abs/2208.08696), and the [dataset owner's interface clarification](https://github.com/chongminggao/KuaiRand/issues/3).

## Permitted-slice label frequencies

The slice contains 337,933 impressions from 20,997 users and 7,552 videos.

| Outcome | Positive | Negative | Positive rate |
|---|---:|---:|---:|
| Long view | 27,197 | 310,736 | 8.0480% |
| `is_click` | 56,484 | 281,449 | 16.7146% |
| Like | 1,583 | 336,350 | 0.4684% |

Long view is inside the plan's 2%–98% feasibility interval. Like is sparse but present.

## Results by randomized `tab`

| Tab | Frequency rank | Impressions | Long-view rate | Click rate | Like rate | Likes without click | Click/valid-play formula match |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 334,290 | 8.1139% | 16.8850% | 0.4714% | 657 / 1,576 | 96.6712% |
| 11 | 2 | 3,159 | 1.5828% | 0% | 0.2216% | 7 / 7 | 93.2890% |
| 2 | 3 | 377 | 3.9788% | 10.3448% | 0% | 0 / 0 | 95.7560% |
| 14 | 4 | 107 | 7.4766% | 0% | 0% | 0 / 0 | 79.4393% |

The documented long-view formula matches the released `long_view` label for roughly 96.8% of dominant tab-1 rows, rather than exactly 100%. The documented single-column valid-play formula similarly matches `is_click` for 96.7% of tab-1 rows. The residual discrepancies are unresolved and argue against silently recomputing either label.

Across the slice, 664 of 1,583 likes (41.95%) occur with `is_click=0`. The data therefore do **not** support the prerequisite interpretation “like requires click,” so the optional ESMM configuration is not supported by M1.2 evidence. This conclusion remains proposed for review.

The random evaluation domain is 98.92% tab 1 and contains no inferred two-column tab 0. Consequently, there is no clean literal-click evaluation population in T-random under the available mapping. Proposed review choices are to treat `is_click` explicitly as valid play for tab 1, model click/valid-play separately by known interface when possible, or remove the nominal click task from comparisons requiring common semantics. No choice is finalized here.

## Long-view pair feasibility

- Users in slice: 20,997.
- Users with both classes: 7,678.
- Total within-user positive-negative pairs: 1,515,626.
- Top 10% definition: top 2,100 of all slice users by pair count.
- Top 10% contribution: 1,402,976 pairs, or 92.5674%.

| Distribution | Minimum | P25 | Median | P75 | P90 | P95 | P99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Impressions, all users | 1 | 3 | 6 | 15 | 38 | 67 | 152 | 741 |
| Pairs, all users | 0 | 0 | 0 | 10 | 90 | 294 | 1,554.24 | 16,168 |
| Pairs, eligible users | 1 | 7 | 22 | 106 | 450 | 950.30 | 2,725.69 | 16,168 |

The proposed sensitivity cap is 952, the higher-observation 95th percentile among eligible users. The uncapped metric remains primary; see `docs/metric_spec_m1_2.md`.

## Cold-start overlap using identifiers only

Each cell gives the share of test impressions with either an unseen user or unseen video under the named training source.

| Training identifiers | T-normal May 3–8 | T-random May 3–8 |
|---|---:|---:|
| Normal Apr 11–29 | 2.0311% | 3.2115% |
| Random Apr 22–29 | 25.5319% | 20.6823% |
| Joint normal + random | 1.8208% | 1.2861% |

For the ordinary normal-only workflow on T-random, 640 of 26,186 test users (2.4441%) and 100 of 7,333 test videos (1.3637%) are unseen; at impression level, 1.8694% have an unseen user, 1.3693% an unseen video, and 3.2115% either. These calculations did not access outcomes.

## Timestamp findings

- **[DOCUMENTED]** `date` is the interaction date, `time_ms` is a millisecond timestamp, and `hourmin` is described ambiguously as format “HHSS.” Whether `time_ms` is request time or exposure time remains unanswered in the [official issue](https://github.com/chongminggao/KuaiRand/issues/4).
- **[OBSERVED]** `hourmin` has only 24 values, 0 through 2300 in steps of 100. It is an hourly bucket, not an HHMM timestamp.
- **[OBSERVED]** UTC+8 conversion of `time_ms` agrees with `date` for 99.19%–99.27% of rows, versus 83.84%–85.62% for UTC.
- **[OBSERVED]** The encoded hour equals the UTC+8 hour for 82.28%–83.07% of rows and is within one hour for 99.980%–99.996%.
- **[INFERRED]** UTC+8/local-China time is strongly supported, but `date`, hourly bucket, and `time_ms` are not exact deterministic transforms. Use `time_ms` for strict event ordering and `date` for frozen calendar splits; do not reconstruct precise event time from `hourmin`.

## Missing April 8 and burn-in alternatives

There are zero April 8 rows. The frozen Apr 11–29 normal training window contains 1,001,316 rows.

1. **Preserve Apr 11 start:** retain all 1,001,316 rows but accept only two observed burn-in days (Apr 9–10), contrary to the stated three-day minimum.
2. **Start training Apr 12:** use Apr 9–11 as the three observed burn-in days. This removes 278,835 Apr 11 rows and leaves 722,481 training rows, a 27.85% reduction. It changes the frozen split.

Starting Apr 12 better preserves the three-day feature-history rule, but neither option is adopted in M1.2 because either requires an explicit protocol decision.

## Shipped video statistics

The official documentation says the statistics are averages by day and scenario over one month, but gives no exact month or cutoff. No metadata verifies that all source events predate April 8. The file remains **excluded** under the frozen plan.

## Unresolved issues for review

1. Select the Apr 11 two-day-burn-in exception or the Apr 12 training start.
2. Decide whether the nominal click task becomes tab-1 valid-play, is restricted by interface, or is removed where semantics must be common.
3. Confirm that the 952-pair sensitivity cap should be frozen.
4. Explain the roughly 3% mismatch between documented formulas and released `long_view`/`is_click` fields.
5. Obtain an authoritative numeric tab codebook and timestamp-event definition.
