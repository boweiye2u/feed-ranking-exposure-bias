# KuaiRand-Pure data card (Week-0 v1)

Status: structural and permitted training-slice diagnostics complete through M1.3. This document follows the frozen `docs/study_plan_v5.md` plus the approved, separately versioned `docs/week0_amendment_v1.md`.

## Source and license

- Dataset: KuaiRand-Pure, from the authors' [official repository](https://github.com/chongminggao/KuaiRand).
- Archive: [Zenodo record 10439422](https://zenodo.org/records/10439422), `KuaiRand-Pure.tar.gz` only.
- Downloaded archive MD5: `0820331067a3784d9691136f772b35a7` (matches the authors' published checksum).
- Extracted release license: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
- Redistribution caveat: the Zenodo metadata did not clearly display dataset-specific redistribution terms during review. The archive itself includes CC BY-SA 4.0 and the official repository links it, which supports authorized local research use. Raw data will not be committed or redistributed. Any future sharing must preserve attribution and license conditions and should first confirm dataset-specific redistribution scope.

## Storage and integrity

- Archive: `~/recom/raw/KuaiRand-Pure.tar.gz` (47,432,272 bytes).
- Extracted data: `~/recom/raw/KuaiRand-Pure/data/`.
- Structural audit: `~/recom/processed/audits/m1_1_structural_audit.json`.
- Raw data, derived artifacts, and audit outputs are excluded from Git.

The archive member list was checked for absolute paths and `..` traversal before extraction. File-level SHA-256 values, schemas, missingness, structural counts, and daily row counts are recorded in the machine-readable audit.

## Contents and populations

| Exposure source | Rows | Users | Videos | Observed `date` coverage |
|---|---:|---:|---:|---|
| Normal, combined | 1,436,609 | 27,077 | 7,551 | 2022-04-09 to 2022-05-08 |
| Random intervention | 1,186,059 | 27,285 | 7,583 | 2022-04-22 to 2022-05-08 |

Static tables contain 27,285 user rows and 7,583 video rows. Detailed per-file schemas and missingness are in `reports/m1_1_structural_summary.md`.

## Branch determination

**Branch B1 applies.** The observed later normal log contains dates through May 8, 2022. The approved Week-0 B1 split is:

- Burn-in: Apr 9–11 normal logs, features only.
- Train: Apr 12–29 normal rows; Apr 22–29 random rows only for the specified RQ3 training arms.
- Validation: Apr 30–May 2 normal rows.
- Test: May 3–8 T-normal and T-random.

Observed discrepancy: the early normal file is named `4_08_to_4_21` but begins at `date=20220409`. The outcome-free M1.3 audit found no blocking April 11 structural issue, so the amendment uses April 9–11 as the required three observed feature-only days and begins normal training April 12.

## Candidate fields and documented semantics

These are candidate fields confirmed by official documentation, not re-inferred from label values:

| Role | Column | Documented meaning |
|---|---|---|
| Long view | `long_view` | Binary; 1 when play reaches duration for videos ≤18 seconds, or reaches 18 seconds for longer videos. |
| Click | `is_click` | Two-column UI: click opens video. Single-column UI: documented as a derived valid-play signal with a 7-second threshold. |
| Like | `is_like` | Binary indication that the user used the like action. |
| Play time | `play_time_ms` | User viewing time in milliseconds. |
| Video duration | `duration_ms` | Video duration in milliseconds for the impression. |
| Random exposure | `is_rand` | Binary indication that the video came from random intervention. |
| Scenario/interface candidate | `tab` | Scenario code in 0–14; exact code-to-policy/interface mapping is not supplied. |

The basic video table also has `video_duration`, documented in milliseconds, but it has 239 nulls. Reconciliation with impression-level `duration_ms` is deferred.

## Randomization support and limits

The official documentation states that an eligible recommended video had an equal probability of replacement by a random video sampled from an item pool. The released structure is consistent with that description: all rows in the random table have `is_rand=1`, all rows in normal tables have `is_rand=0`, and the random table covers all 7,583 released candidate-pool videos.

This supports the plan's use of the random table as a randomized-exposure evaluation distribution, subject to important limits:

- random rows occur only for observed `tab` values 1, 2, 11, and 14, whereas normal rows span all 0–14;
- the exact eligible feed positions, replacement mechanism, pool composition over time, and per-row assignment probabilities are not released;
- the mapping from `tab` to interface and recommendation policy is undocumented;
- KuaiRand-Pure retains only candidate-pool videos and therefore has incomplete sequential histories;
- users were filtered based on random-exposure availability, limiting population generalization;
- randomized insertion does not imply random sampling of users, contexts, interfaces, or all catalog items.

Consequently, T-random can evaluate ranking on the observed randomized-insertion domain, but it does not justify claims of unbiased value for arbitrary deployment policies or interfaces.

## Outcome-data embargo

M1.1 did not inspect engagement values or distributions. No label rates, long-view/click/like counts, user-level positive/negative pair counts, or May 3–8 outcome statistics were computed. Only schemas, null counts, identifiers, exposure metadata, timestamps, and row counts were audited.

## M1.2 permitted diagnostics

M1.2 inspected engagement outcomes only on the April 22–29 randomized training slice. Long view has an 8.0480% positive rate; 7,678 users have both classes and contribute 1,515,626 within-user positive-negative pairs. M1.3 freezes long view and uncapped pair-weighted within-user AUC as primary, with the 952-pair cap used only for sensitivity.

The inferred single-column tab 1 accounts for 98.92% of the permitted random slice; the inferred two-column tab 0 is absent. `is_click` is therefore masked and named by interface: valid play on tab 1 and literal click on tab 0, without pooling. Because 41.95% of likes occur without `is_click`, ESMM is excluded. Detailed findings are in `reports/m1_2_engagement_diagnostics.md`.

The outcome embargo remains in force for randomized April 30–May 2 and all May 3–8 engagement values until the scheduled frozen evaluation.

## Unresolved questions

1. What timezone defines `date` and `hourmin` relative to `time_ms`, and which event does `time_ms` timestamp?
2. Is there an authoritative mapping from all `tab` codes to interfaces and recommendation policies? The release-specific frequency rule supports tab 1 as single-column and tab 0 as two-column only.
3. Are random replacements uniform over the full 7,583-video pool for every eligible `tab`, date, and context, and are assignment propensities available?
4. Why do the documented long-view and valid-play formulas disagree with roughly 3% of released labels?
5. What exact calendar month and cutoff produced `video_features_statistic_pure.csv`? It remains excluded unless proven pre-study.
6. Why does the normal population have 27,077 users while the random and user-feature tables have 27,285?
7. The official `hourmin` description says “HHSS,” but observed values are hourly buckets; the naming remains undocumented.

## Reproduction

From the repository with the project environment active:

```bash
source .venv/bin/activate
python scripts/audit_dataset.py
pytest
```
