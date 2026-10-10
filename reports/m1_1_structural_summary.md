# M1.1 KuaiRand-Pure structural summary

Generated from the checksum-verified official archive by `scripts/audit_dataset.py`. The complete machine-readable report is stored outside Git at `~/recom/processed/audits/m1_1_structural_audit.json`.

This is a structural audit only. It did not compute engagement values, label rates, outcome distributions, or positive/negative pair counts. In particular, no May 3–8, 2022 engagement outcomes were inspected.

## Release files

| File | Bytes | Rows | Unique users | Unique videos |
|---|---:|---:|---:|---:|
| `log_standard_4_08_to_4_21_pure.csv` | 83,961,282 | 1,141,112 | 26,210 | 7,538 |
| `log_standard_4_22_to_5_08_pure.csv` | 21,765,075 | 295,497 | 25,877 | 6,618 |
| **Combined normal logs** | 105,726,357 | **1,436,609** | **27,077** | **7,551** |
| `log_random_4_22_to_5_08_pure.csv` | 87,086,116 | **1,186,059** | **27,285** | **7,583** |
| `user_features_pure.csv` | 3,519,028 | 27,285 | 27,285 | — |
| `video_features_basic_pure.csv` | 626,669 | 7,583 | — | 7,583 |
| `video_features_statistic_pure.csv` | 6,559,217 | 7,583 | — | 7,583 |

## Timestamp coverage and protocol branch

| Log | Minimum `date` | Maximum `date` | Minimum `time_ms` (UTC rendering) | Maximum `time_ms` (UTC rendering) |
|---|---:|---:|---|---|
| Early normal | 2022-04-09 | 2022-04-21 | 2022-04-09 03:46:03.278Z | 2022-04-21 15:52:08.855Z |
| Later normal | 2022-04-22 | 2022-05-08 | 2022-04-21 14:52:47.170Z | 2022-05-08 15:52:10.564Z |
| Random | 2022-04-22 | 2022-05-08 | 2022-04-22 02:42:12.831Z | 2022-05-08 15:52:08.763Z |

The later normal table contains every calendar date from April 22 through May 8 and reaches May 8. **Branch B1 applies.** The `date` field is the study split key. The UTC rendering of `time_ms` can fall on the preceding UTC date, so the timezone/local-calendar relationship must be resolved before timestamp-level feature work.

Despite its filename and official description, the early normal file has no observed `date=20220408`; it begins on April 9. This reduces the planned Apr 8–10 burn-in to two observed calendar days unless the protocol is explicitly amended in a later authorized planning step. The frozen plan was not modified during M1.1.

## Schemas

All three log files share 19 columns, each inferred as `Int64`:

`user_id`, `video_id`, `date`, `hourmin`, `time_ms`, `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`, `play_time_ms`, `duration_ms`, `profile_stay_time`, `comment_stay_time`, `is_profile_enter`, `is_rand`, `tab`.

`user_features_pure.csv` has 31 columns:

- `String`: `user_active_degree`, `follow_user_num_range`, `fans_user_num_range`, `friend_user_num_range`, `register_days_range`.
- `Float64` because nulls are present: `onehot_feat4`, `onehot_feat12`–`onehot_feat17`.
- `Int64`: `user_id`, `is_lowactive_period`, `is_live_streamer`, `is_video_author`, `follow_user_num`, `fans_user_num`, `friend_user_num`, `register_days`, `onehot_feat0`–`onehot_feat3`, and `onehot_feat5`–`onehot_feat11`.

`video_features_basic_pure.csv` has 12 columns:

- `Int64`: `video_id`, `author_id`, `music_id`.
- `Float64`: `visible_status`, `video_duration`, `server_width`, `server_height`, `music_type`.
- `String`: `video_type`, `upload_dt`, `upload_type`, `tag`.

`video_features_statistic_pure.csv` has 52 columns. `video_id` and `counts` are `Int64`; its other 50 aggregate-statistic columns are `Float64`. Per the frozen plan, this file remains excluded unless its pre-April-8 provenance is established.

## Missing values

- All three logs: zero nulls in every column.
- User features: 874 nulls in `onehot_feat4`; 714 each in `onehot_feat12` through `onehot_feat17`; zero elsewhere.
- Basic video features: 203 nulls in `music_type`, 96 in `tag`, and 239 in `video_duration`; zero elsewhere.
- Video statistics: zero nulls in every column.

## Exposure structure

- The random table contains only `is_rand=1`; both normal tables contain only `is_rand=0`.
- Normal logs contain all `tab` codes 0–14. Random logs contain only tabs 1, 2, 11, and 14.
- Official documentation calls `tab` a scenario indicator, with recommendation/main-page examples, but does not publish a code-to-interface or code-to-policy mapping.
- Official documentation says random interactions result from random intervention and describes equal-probability replacement by an item sampled from a pool. This supports randomized-insertion analysis within the observed eligible scenarios and released candidate pool. It does not establish unconditional random exposure across all 15 scenarios, users, feed positions, or the full Kuaishou catalog. Per-row propensities and candidate-pool snapshots are absent.
