# Analysis protocol

Version: Week-0 v1, frozen after M1.3. The original `study_plan_v5.md` is preserved; approved changes are recorded in `week0_amendment_v1.md`.

## Populations and splits

- Feature-only burn-in: normal impressions, 2022-04-09 through 2022-04-11.
- Normal training: 2022-04-12 through 2022-04-29.
- Random training: 2022-04-22 through 2022-04-29, used only for the prespecified RQ3 arms and frozen Week-0 diagnostics.
- Model selection: normal impressions, 2022-04-30 through 2022-05-02.
- Final T-normal and T-random: 2022-05-03 through 2022-05-08.
- Random outcomes for 2022-04-30 through 2022-05-02 are not a tuning or diagnostic set and remain embargoed.

Splits use the released integer `date`. Within-window event ordering uses `time_ms`; `hourmin` is only an hourly bucket. The observed timestamp evidence supports UTC+8 approximately, but no exact timezone transform is assumed.

## Targets and interface handling

Long view is the primary binary target, using the shipped `long_view` value. Documented label formulas are not used to silently rewrite released values.

`is_click` is not pooled across interfaces:

- tab 1: call the target **valid play**, evaluated and trained only within tab 1;
- tab 0: call the target **literal click**, evaluated and trained only within tab 0;
- other tabs: no click task until an authoritative interface codebook exists.

Any table or model name must carry the target and tab mask. Because T-random has no observed tab-0 support in the audited release, literal-click T-random performance is not claimed. Like remains a separate secondary binary target. ESMM is excluded; like is not conditioned on click.

## Features and model population

Feature histories use normal exposure only. Burn-in is feature-only; its rows are not training labels. Shipped `video_features_statistic_pure.csv` is excluded because its temporal cutoff is unknown. Test labels never enter histories. The configuration population remains bounded at 12 and excludes ESMM. No feature engineering or training was performed in M1.3.

## Metrics and inference

The primary metric is uncapped pair-weighted within-user AUC for long view. The frozen 952-pair capped version is sensitivity-only. Secondary metrics are per-user-average AUC, pooled AUC, NDCG@5, and LogLoss, as defined in `metric_spec.md`.

For neural models, seeds are 0, 1, and 2; seed 0 is primary and results across all three are reported. Paired user-bootstrap inference uses 10,000 resamples with seed 0. RQ2 uses 50 within-user candidate splits per run: split seed 0 is primary and seeds 1 and 2 are stability checks. Regret bootstrap uses 2,000 resamples. Six prespecified primary comparisons use Holm correction.

## Outcome-access controls

Outcome-bearing loaders must call `authorize_outcome_access`. The guard allows only the configured source, dates, workflow, and purpose. It rejects:

- May 3–8 outcomes outside the final-evaluation metric workflow;
- May 3–8 outcomes requested for features even during final evaluation;
- randomized April 30–May 2 outcomes in every pre-final workflow;
- outcome windows outside their configured training or validation split.

Structural and identifier-only audits may inspect held-out rows but must project only approved non-outcome columns. Final evaluation is a one-time frozen operation after training and selection are complete.
