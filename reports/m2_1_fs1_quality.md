# M2.1 FS1 schema and data-quality report

Machine-readable report: `~/recom/processed/fs1/v1/data_quality.json`.

## Integrity results

| Check | Normal train | Random train |
|---|---:|---:|
| Input/output and label rows | 722,481 | 337,933 |
| Unique impression IDs | 722,481 | 337,933 |
| Null impression user/video keys | 0 / 0 | 0 / 0 |
| Unknown static user/video keys | 0 / 0 | 0 / 0 |
| Feature rows without labels | 0 | 0 |
| Label rows without features | 0 | 0 |
| Shipped aggregate statistics used | No | No |

The complete feature schemas are identical across sources. All expected dates are present, joins are many-to-one, and no duplicate expansion occurred.

## Feature groups

FS1 has 41 shipped static attributes, two missing-entity indicators, and two contextual fields. Five alignment fields (`impression_id`, `user_id`, `video_id`, `date`, `time_ms`) are metadata rather than model features.

- User categorical: `user_active_degree`, three `is_*` flags, four range fields, and `onehot_feat0`–`onehot_feat17` (26 fields).
- User numeric: follow, fan, friend, and registration-day counts (4 fields).
- Video categorical: author, video type, upload date/type, visibility, music ID/type, and tag (8 fields).
- Video numeric: duration, server width, and server height (3 fields).
- Missing indicators: `user__static_missing`, `video__static_missing` (2 Boolean fields).
- Context: `context__tab` and restricted `context__is_rand` (2 integer fields).

Prefixes are added to static fields in the output. The baseline and random-only selectors expose 44 model fields; the RQ3-joint selector exposes 45 by adding `context__is_rand`.

## Preserved missing values

| Field group | Normal null rows | Random null rows |
|---|---:|---:|
| `user__onehot_feat4` | 24,550 | 12,582 |
| Each of `user__onehot_feat12`–`17` | 14,320 | 8,669 |
| `video__video_duration` | 17,040 | 10,663 |
| `video__music_type` | 20,668 | 8,710 |
| `video__tag` | 6,757 | 4,384 |

These are released static-feature nulls, not unknown joins. They are deliberately retained for later model-specific handling.

## Resource use

The finalized full run took 10.99 seconds and recorded 1,248.2 MiB peak resident memory. The four Parquet outputs total 30,942,065 bytes (approximately 29.5 MiB). No model was trained.
