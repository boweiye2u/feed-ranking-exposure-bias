# Feed Ranking under Exposure Bias (KuaiRand-Pure)
## Multi-Task Engagement Ranking and Exposure-Bias-Aware Model Selection

**Date:** 2026-10-09
**Version:** v5, frozen planning draft (wording finalized 2026-10-09). Preserve v4 unchanged. Freeze the experiment protocol at the end of Week 0, after permitted dataset-dependent choices are resolved and before model-result inspection.
**Scope:** small. About 3.5–4 weeks, 1 GPU, up to 3 GPUs only for parallel sweeps.

Tag key: [KNOWN] from the KuaiRand paper (arXiv 2208.08696) or other cited facts · [COMPUTED] calculated · [INFERRED] deduction · [COMMON] standard field knowledge · [GUESS] no solid basis.

---

## 0. What changed

### v4 → v5 (methodological clarification)

| # | Revision | Sections | Rationale |
|---|---|---|---|
| V1 | Rename **SNIPS-IPS** to **marginal-exposure reweighting (SNIPS-style heuristic)** | 3, 9, 10, 15–17 | Normal-log item frequency is not the conditional exposure propensity needed for causal IPS; present as a heuristic, not unbiased counterfactual estimation. |
| V2 | Match **optimizer steps** rather than epochs in neural RQ3 arms; fix optimizer, batch size, schedule and boosting rounds | 10, 11, 14 | Otherwise different sample sizes imply different gradient-update budgets. |
| V3 | Describe the joint arm as **random-exposure data fusion / target-domain adaptation** | 3, 10, 15–17 | A domain indicator and extra randomized observations do not establish causal removal of logging-policy bias. |
| V4 | Clarify that cross-fitting mitigates **oracle-selection evaluation optimism** | 9, 17 | It evaluates a model chosen on one random-test user half on the other half; it does not make the oracle target noise-free. |
| V5 | Align the Week 0 data-use and diagnostic rules with the original split | 5, 11 | Earlier v4 wording inaccurately limited pre-validation randomized rows to joint training even though random-only also uses them. |

### v3 → v4 (retained historical change log)

| # | Fix | Sections | Why |
|---|---|---|---|
| M1 | **Random-only arm replaces the size-matched control;** RQ3 now has 3 arms, and the Holm family has 6 comparisons | 3, 9, 10 | A size-matched control would drop ≈558K of ≈910K normal training rows, leaving 39%, so it would mostly test "less data." [COMPUTED, rough; assumes Branch B1 and uniform daily volume] |
| M2 | **Regret averaged over K = 50 random A/B splits** inside each bootstrap resample; computed from precomputed per-user pair counts | 9, 13 | A single split makes the result depend on the split seed. [INFERRED, HIGH] |
| M3 | **RQ1 uses within-user AUC;** pooled AUC is secondary | 3 | Consistency with the v3 primary-metric change. |
| m1 | Section 5 header: "Branch B confirmed; B1 vs B2 open" | 5 | Header wasn't updated in v3. |
| m2 | Leakage test wording: "a random sample of rows" | 14 | Avoids confusion with random-exposure rows. |
| m3 | Contenders defined by a paired-difference CI that includes 0 (minimum top 3) | 9 | The v3 wording compared point scores to one configuration's interval. |
| m4 | Bootstrap p-value defined | 9 | It was unspecified. |
| m5 | Thin user embeddings: small dimensions + weight decay | 8 | ≈33 training rows per user, ≈120 per video. [COMPUTED, rough] |
| m6 | Heavy-tailed users: report the top-10% share of pairs; per-user pair-cap sensitivity check | 9, 11 | Mean ≈43.5 random exposures per user, while 55.1% of users have 10–25. [COMPUTED] |

### v2 → v3

| # | Fix | Sections | Why |
|---|---|---|---|
| C1 | **Features come from normal logs only.** Random-log events never enter features, except the exposure indicator in the RQ3 joint arm | 6 | Otherwise every baseline uses the RQ3 random slice, and T-random rows get features built from earlier T-random labels. [INFERRED, HIGH] |
| C2 | **Primary metric is pair-weighted within-user AUC** on T-random (long view) | 4, 9 | Pooled AUC mostly rewards learning which users long-view more, not ranking within a user's feed. [INFERRED, HIGH] |
| C3 | **Paired user-level bootstrap replaces DeLong;** 10,000 resamples | 9 | DeLong assumes independent impressions; impressions cluster within users. [COMMON, HIGH] More resamples are needed to resolve p-values near Holm's 0.01 cutoff. [COMPUTED] |
| I1 | **Second regret reference:** random pick among contenders (within the validation CI of the best, minimum top 3) | 9 | Weak configurations make the all-configuration random pick look bad by design. [INFERRED, MED-HIGH] |
| I2 | **Seed rule for regret and τ:** seed 0; seeds 1–2 rerun the whole computation | 9 | Regret had no seed rule. |
| I3 | **Tagged GitHub commit** of the data card and analysis plan at the end of Week 0; "pre-registered" used only if it exists | 11, 16 | Otherwise "pre-registered" is an overclaim. |
| I4 | **Size-matched control** for the RQ3 joint-training arm | 10 | Joint training gets extra rows from the T-random distribution, which confounds data volume with debiasing. [INFERRED, MED] |
| F1 | Item-mix caveat weakened: both test sets share Pure's candidate pool | 9 | Pure keeps only candidate-pool videos. [KNOWN] |
| F2 | History is thin in Pure; a null "no history" ablation isn't read as "history doesn't matter" | 8 | [INFERRED, MED-HIGH] |
| F3 | B2 caveat: test-day features stop at Apr 21 | 5 | Follows from C1. |
| F4 | Header wording: "Branch B confirmed; B1 vs B2 open"; 7,582 vs 7,583 off-by-one noted | 0, 2 | The paper states both numbers. |

### v1 → v2

| Area | v1 | v2 | Why |
|---|---|---|---|
| Features | Aggregates computed over the training period | **Point-in-time:** day d sees only days < d; rates shrunk toward a prior, plus raw counts | Training-period aggregates leak each row's own label and its future. [INFERRED, HIGH] |
| Positioning | "Feed and ads ranking" | **Short-video feed ranking**; ads only as a talking point (calibration) | No bids, auctions, or conversions in the data. [INFERRED, MED] |
| RQ2 | Kendall τ over 4 models | **Selection regret** (primary, cross-fit) over ≤13 configurations; τ secondary, split by model family | τ over 6 pairs carries almost no information. [COMPUTED] |
| Primary metric | Unspecified | Long-view AUC on T-random (v3: pair-weighted within-user), declared at the end of Week 0 | Click semantics vary by interface. [KNOWN + INFERRED] |
| Dataset | Small and mid-size variants | **KuaiRand-Pure only** | Recommended by the authors for debiasing and multi-task work. [KNOWN] |
| Split | Generic | **Dated split inside the random-exposure window** (Branch B confirmed; B1 vs B2 open) | Random logs exist only from Apr 22 to May 8. [KNOWN] |
| Retrieval, re-ranking | In scope | **Stretch goals** | With 7.5K videos, candidate generation is nearly trivial. [INFERRED, HIGH] |
| Baseline fairness | Not stated | Trees and neural networks get the same dense features; extras listed; dense-only ablation | Otherwise the comparison measures feature sets, not architectures. [INFERRED, MED-HIGH] |
| RQ3 | Open-ended | Two arms (SNIPS-IPS, joint training); v4 adds a random-only arm | Keeps the scope small. |
| PLE, Schnabel propensities | Optional | **Dropped** | Scope. |

---

## 1. Why this project and how to position it

**Gap it closes.** Ranking and personalization is the top Applied Scientist theme in our job-posting sample [COMPUTED, rough]. Your portfolio has document retrieval and ranking, but no user modeling or feed ranking.

**Frame:** "Short-video feed ranking: multi-task engagement prediction and exposure-bias-aware evaluation."

**Employers it maps to:** TikTok, Meta Reels, Roblox, YouTube, Snap, Pinterest.

**Ads:** mention only in interviews, via calibration. Never on the resume.

**Hook:** almost all offline recommendation results come from data that the old recommender chose to show users. KuaiRand includes videos inserted at random, which lets you test whether the standard way of picking a model still picks the right one when the evaluation data is close to unbiased. This fits the rigor-first story from your biomedical and audio projects.

---

## 2. Dataset facts (from the KuaiRand paper)

| Fact | Value | Tag |
|---|---|---|
| Version used | KuaiRand-Pure: 184 MB logs + 10 MB features | [KNOWN] |
| Users | 27,285 (users with fewer than 10 random exposures removed) | [KNOWN] |
| Normal logs (Pure) | 1,436,609 interactions over 7,551 videos | [KNOWN] |
| Random logs | 1,186,059 interactions over 7,583 videos. The paper also says "7,582 videos" in its appendix, most likely 0-based IDs running 0–7,582 [INFERRED, MED]; Week 0 counts it | [KNOWN] |
| Normal-history window | 2022-04-08 to 2022-04-21 | [KNOWN] |
| Random-exposure window | 2022-04-22 to 2022-05-08 (17 days) | [KNOWN] |
| Feedback signals | 12, e.g., click, like, view time | [KNOWN] |
| Recommendation policies | 15 scenarios, each with its own policy and interface | [KNOWN] |
| Click meaning | Two-column feed: click opens a video, so it's meaningful. Single-column feed: videos autoplay, so view time is the meaningful signal | [KNOWN] |
| Random exposures per user (whole window) | 55.1% of users have 10–25 | [KNOWN, from the 27K figure; Pure shares the same random log] |
| Not used | KuaiRand-1K (≈4.4M videos, so item embeddings would mostly go untrained), KuaiRand-27K (23 GB logs + 23 GB features) | [KNOWN + INFERRED] |

**Still unknown, checked in Week 0:**
- Whether the normal logs continue into Apr 22–May 8. I believe the release has a separate normal-log file for that period [KNOWN, MED, from memory].
- The license and terms of use.

---

## 3. Research questions

| RQ | Question | Primary output |
|---|---|---|
| RQ1 | Does a shared multi-task model help the sparse task (like) compared with separate models, without hurting the dense tasks? | Per-task within-user AUC on T-random (pooled AUC secondary): separate LightGBM vs shared-bottom vs MMoE |
| **RQ2 (headline)** | Does the standard workflow (pick the model on normal validation logs) choose a model that is also good on random exposure? | **Selection regret** (cross-fit) vs a random-pick reference |
| RQ3 | Do pragmatic exposure-frequency reweighting and randomized-data adaptation improve T-random ranking, and how does randomized-only training compare with normal-only training? | Marginal-exposure SNIPS-style heuristic, normal+random data fusion, and random-only vs normal-only baseline on T-random |

---

## 4. Primary metric (declared at the end of Week 0)

**Default:** pair-weighted within-user long-view AUC on T-random. [INFERRED, MED-HIGH]

```text
Within-user AUC = (# correctly ordered (positive, negative) pairs from the same user)
                  / (# such pairs), pooled over all users with ≥1 positive and ≥1 negative
```

- Algebraically, this is GAUC weighted by n_pos × n_neg per user. [COMPUTED]
- **Why:** it measures ranking *within* a user's feed, which is what a feed ranker does. Pooled AUC also compares user A's impressions against user B's, so a model that only learns "heavy users long-view more" scores well without ranking anything. [INFERRED, HIGH]
- **Why pair-weighted:** unlike the per-user average, it keeps low-volume users instead of letting each one count equally despite having very few pairs.
- **Enough data:** a user with about 6 random impressions and a 30% long-view rate contributes about 8 pairs; across ~27K users that's on the order of 100K+ pairs. [COMPUTED, rough]

- **Why not click:** what a click means depends on the interface (see Section 2).
- **Why not like:** it is probably too sparse on random exposures.
- **Why not a mean across tasks:** it mixes a dense and a sparse signal into one hard-to-interpret number.
- **Why not per-user-average GAUC as primary:** a 6-day test window gives most users only about 3.5–9 random impressions [COMPUTED, rough: 10–25 × 6/17], so averaging per-user AUCs gives very noisy users equal weight.

**Secondary metrics:**
- **Per-user-average GAUC:** over users with both outcome classes; report how many users qualify.
- **Pooled AUC:** reported alongside. If within-user and pooled results disagree, that is itself informative.
- **NDCG@K:** only over users with at least 5 random test impressions [GUESS on the threshold].
- **Calibration:** LogLoss and calibration error.

**How to decide.** At the end of Week 0, compute label rates on the **random slice before validation** (Section 5), never on T-random itself. Same distribution, no peeking at the test set. If long-view's positive rate is unusable (below about 2% or above about 98% [GUESS]), switch the primary task and record why, before any model is trained.

---

## 5. Split (Branch B confirmed; B1 vs B2 open)

### Branch B1 (expected): normal logs continue through May 8

| Period | Dates | Normal logs | Random logs |
|---|---|---|---|
| Burn-in (features only) | Apr 8–10 | Feature source | none |
| Train | Apr 11–Apr 29 | Training rows | **Apr 22–29: RQ3 random slice only** (8 days) |
| Validation | Apr 30–May 2 | Model selection (standard workflow) | Not used for any decision |
| Test | May 3–May 8 | **T-normal** | **T-random** |

[COMPUTED] The random window holds 8 days of slice + 3 days of validation + 6 days of test = 17 days.

### Branch B2: normal logs end Apr 21

- Validation: normal logs Apr 19–21. Training: Apr 11–18. Burn-in: Apr 8–10.
- T-random: May 3–8. RQ3 random slice: Apr 22–May 2.
- T-normal: no normal data exists in the random window. Report this, and treat τ between T-normal and T-random as confounded by time.
- Selection regret still works, since it only needs normal validation plus T-random.
- **Stale features:** because features come from normal logs only (Section 6), test-day features stop at Apr 21, more than 12 days before the test window. Report this as a limitation.

### Hard rules

- No random-log data from the test days touches any decision.
- The pre-validation RQ3 random slice may be used **only** for pre-frozen Week 0 diagnostics and for training the RQ3 joint and random-only arms; it does not enter the ordinary baseline models, feature histories, or normal-validation selection. No random-validation or T-random outcomes are used to tune model choices.

---

## 6. Features (point-in-time)

**Source rule (all configurations):** aggregates and history come from **normal logs only**. Random-log events never enter features. The only exception is the exposure-type indicator in the RQ3 joint-training arm.

- **Aggregate rule:** every aggregate for an impression on day d uses only days before d. Use an expanding window with a 3-day minimum burn-in. Burn-in days are a feature source only, never training rows.
- **History rule:** a user's last-N history uses only events strictly before the impression's timestamp.
- **Shrunk rates:** rate = (positives + α·global_rate) / (impressions + α), with α tuned on validation. This damps the mismatch where early training rows have rates built from about 3 days but test rows from about 24 days. [COMMON, HIGH]
- **Raw counts:** every rate is paired with its raw impression count, so the model knows how much support a rate has.
- **Test days:** test-day features use labels from earlier test days (day d sees days < d). That is legitimate, because a production system would have those labels. Stated in the README so reviewers don't read it as leakage.
- **Shipped statistics:** the dataset's own video-statistics file is dropped, unless Week 0 proves it was computed before Apr 8.

**Feature sets (same for all models):**

| Set | Contents |
|---|---|
| FS1 | Static user and video features (as shipped, excluding aggregate statistics) |
| FS2 | FS1 + point-in-time user and video aggregates (shrunk rates + counts) |
| FS3 | FS2 + user × category affinity, recency, interface/policy indicator |

Neural networks also get user and video ID embeddings and pooled last-N history. Both are listed explicitly in the README.

---

## 7. Task definitions

- All tasks are P(outcome | impression).
- **Tasks:** long view (primary), click, like.
- **Click:** checked per interface/policy in Week 0. Drop click rows where the video autoplays (click is meaningless there), or model click separately by interface.
- **Like:** checked in Week 0 whether a like requires a click. If it does, add one ESMM configuration, P(click) × P(like | click), to the population.

---

## 8. Configuration population (frozen before any results, capped at 13)

| # | Configuration |
|---|---|
| 1 | R0: popularity / shrunk video long-view rate |
| 2–7 | R1: LightGBM, one model per task, {FS1, FS2, FS3} × {shallow, deep} |
| 8 | R2: shared-bottom MLP (FS3 + IDs + history) |
| 9 | R3: MMoE (FS3 + IDs + history) |
| 10 | Ablation: MMoE with dense FS3 features only (no IDs, no history). This is the fair architecture comparison against LightGBM-FS3 |
| 11 | Ablation: MMoE without history |
| 12 | Ablation: MMoE without aggregates |
| 13 | ESMM (only if Week 0 shows a like requires a click) |

**Training:** each configuration is tuned on normal validation only, with a small fixed grid. 3 seeds for the neural networks.

**Thin ID embeddings.** About 910K normal training rows over 27,285 users gives ≈33 rows per user; videos get ≈120 rows each. [COMPUTED, rough; Branch B1] User embeddings will overfit easily: use small embedding dimensions (e.g., 16–32 [GUESS]) and weight decay. This also makes "LightGBM wins" the likelier outcome, which the risk table covers. [INFERRED, MED]

**History is thin in Pure.** Pure keeps only candidate-pool videos, so a user's last-N history covers a small slice of their real feed. [INFERRED, MED-HIGH] Expect ablation #11 (no history) to show little difference, and don't read that as 'history doesn't matter.'

---

## 9. Evaluation and statistics

**Selection regret (RQ2 primary).** Cross-fitting reduces *oracle-selection evaluation optimism*: the configuration chosen for its random-exposure score on one user half is evaluated on disjoint users. The split-half best is a noisy, finite-sample reference, **not** the population-optimal model; report regret relative to this reference, not as ground-truth causal regret.
1. The standard workflow picks the configuration that is best on normal validation, using the primary metric.
2. **Cross-fit to avoid reusing the same users for random-exposure reference selection and evaluation:**
   1. Split the T-random users randomly into halves A and B.
   2. Find the configuration that is best on A.
   3. Regret on B = best-on-A's score on B − the workflow pick's score on B.
   4. Swap A and B, then average the two regrets.
   5. **Repeat over K = 50 random A/B splits [GUESS on K] and average,** so the result doesn't depend on one split seed.
3. **Random-pick reference:** the expected regret of choosing a configuration uniformly at random.
4. **Contender reference:** the expected regret of choosing at random among *contenders*. A configuration is a contender if the bootstrap 95% CI of its **paired difference** from the validation best (on normal validation, resampling users) includes 0. The set always includes at least the **top 3** by validation. This is the realistic question: among models a practitioner would actually consider, does normal validation pick the right one? [INFERRED, MED-HIGH]
5. **Bootstrap** the whole procedure over users (2,000 resamples for regret CIs; each resample contains the K = 50 splits). Report a 95% CI for each regret.
6. **Seed rule:** regret and τ use seed 0 for every neural configuration. Seeds 1–2 each rerun the whole regret computation, as a stability check.
7. **Heavy-tailed users.** The mean is ≈43.5 random exposures per user [COMPUTED: 1,186,059 / 27,285], while 55.1% of users have 10–25 [KNOWN], so a few heavy users contribute many of the within-user pairs. Report the share of pairs from the top 10% of users. As a sensitivity check, recompute the primary metric and regret with pairs capped per user (cap set in Week 0). [INFERRED, MED]

**τ (RQ2 secondary).**
- Report Kendall τ separately for between-family pairs and within-family pairs.
- Give it a bootstrap CI but **no significance claim.** Within-family near-ties push τ down and between-family agreement pushes it up, so the net direction is unknown. [INFERRED, MED-HIGH]

**Flip analysis.** Only for the pre-declared pairs below: does the winner on T-normal lose on T-random?

**Absolute gaps between T-normal and T-random.** Reported descriptively only. Both test sets come from Pure's single candidate pool [KNOWN], so they share one item set; they differ in how often each item is shown. The gaps therefore mix exposure bias with that frequency difference and are not a clean bias estimate, though the confound is weaker than in datasets with different item sets.

**Pre-declared paired comparisons** (primary metric on T-random; **paired user-level bootstrap**, 10,000 resamples; Holm correction across all 6). DeLong is not used: it assumes independent impressions, and impressions cluster within users. [COMMON, HIGH]

1. R3 (MMoE) vs best R1 (LightGBM)
2. R3 vs R2 (shared-bottom)
3. R3-dense-only (#10) vs LightGBM-FS3 (fair architecture comparison)
4. RQ3 marginal-exposure SNIPS-style heuristic arm vs its baseline
5. RQ3 normal + random data-fusion arm vs its baseline
6. RQ3 random-only arm vs its baseline

**Bootstrap p-value (approximate):** two-sided p = 2 × min(share of resampled paired differences ≤ 0, share ≥ 0), capped at 1. This tail-share approximation is a pragmatic descriptive inference method rather than a rigorously null-centered bootstrap test; report CIs and Holm-adjusted results with that limitation. [COMMON, MED]

**Seeds.** For neural configurations, every test uses the pre-declared seed 0. Seeds 1–2 are reported as a stability check: an effect counts as stable if at least 2 of 3 seeds, including seed 0, show the same sign and the 3-seed mean has the same sign.

---

## 10. RQ3: exposure-aware heuristics and random-data adaptation (three arms)

**Interpretation:** these arms are practical distribution-shift interventions, **not** validated causal off-policy estimators. In particular, the random-insertion policy, interface and population restrictions limit what T-random can represent. All gains are offline gains on the observed T-random evaluation distribution; no online causal uplift is claimed.

1. **Marginal-exposure reweighting (SNIPS-style heuristic):**
   - Estimate the **marginal item exposure frequency** from normal **training** logs only; derive inverse-frequency observation weights, clip them, and normalize them to mean one within the training data (a SNIPS-style weighted objective).
   - Freeze clipping thresholds and weight construction using normal training/validation only before inspecting any T-random model performance. Record weight quantiles and effective sample size, ESS = (sum w)^2 / sum(w^2).
   - **Identification limit:** item frequency p(i) is **not** the required user/context/policy-dependent exposure propensity p(i | u, x, policy). Consequently, these weights do **not** justify an unbiased IPS estimate of policy value. Treat this as a popularity/exposure-frequency correction heuristic; do not claim causal debiasing. [COMMON + INFERRED, HIGH]
2. **Normal + random data fusion (target-domain adaptation):**
   - Train on normal logs plus the **pre-validation** random slice, with an exposure-type indicator feature identifying which source produced each row.
   - At T-random prediction time, set the indicator to `random`, reflecting the intended evaluation domain. This is supervised domain adaptation to the random-exposure distribution, not proof that exposure bias has been removed.
   - **Data-volume confound (stated, not controlled):** this arm gets ≈558K extra rows from the randomized domain under the B1 rough count. A size-matched normal-only control would discard ≈61% of the normal rows and answer a different question. Gains reflect an inseparable combination of randomized-sample information, domain adaptation and extra data; do not attribute them to any one mechanism. [COMPUTED, rough; Branch B1]
3. **Random-only:**
   - Train on the **pre-validation** random slice alone (≈558K rows over 8 days under the rough B1 estimate), with features still built exclusively from normal logs (Section 6).
   - **Question:** does a shorter period of randomized exposures beat a longer period of normal-policy exposures on T-random under the same model and optimization budget?
   - **Interpretation:** the training source is closer to the evaluation exposure distribution, but it has a different number of observations and different calendar coverage. Report the magnitude of the offline gap; do **not** interpret the gap as a causal exchange rate between unbiased and biased samples. [INFERRED, MED]

All three arms reuse the **best R1 or R3 base configuration**, chosen on normal validation and fixed before RQ3. The comparison baseline is the unmodified normal-only fit of that base configuration. All feature histories and aggregate statistics continue to use normal logs only, even for rows in the random training slice (Section 6).

**Frozen optimization budget (all three arms):**
- **Neural networks:** use the base model's fixed optimizer, batch size, regularization, learning-rate schedule and **total optimizer-update count**, *not* its epoch count. Different dataset sizes therefore do not automatically grant more or fewer gradient steps. Training examples may repeat when an arm is smaller; record this explicitly.
- **LightGBM:** reuse the base model's fixed number of boosting rounds and its other hyperparameters, without arm-specific early stopping. This matches boosting-round budget, not total CPU cost.
- **No arm-specific selection on randomized outcomes:** training-budget and weight rules are fixed on the normal-validation base workflow before running RQ3. The pre-validation random slice supplies training examples to the joint/random-only arms and permitted Week 0 descriptive checks, not a new tuning set. Report the remaining difference in unique examples as a limitation. [INFERRED, HIGH]

---

## 11. Week 0 checklist (about 3 days)

1. License and terms of use.
2. Normal-log coverage of Apr 22–May 8. This decides Branch B1 vs B2.
3. Exact field names: long view, click, like, play time, interface/policy ID.
4. Click meaning per interface. Decide which click rows to drop.
5. Whether a like requires a click (decides the ESMM configuration).
6. Whether the shipped video statistics were computed before Apr 8 (keep or drop).
7. Label rates for long view, click, and like on the **pre-validation random slice**. Then declare the primary metric.
8. **Before the freeze:** estimate positive–negative within-user pair counts, impression-count distribution, and top-10%-of-users pair share using the **pre-validation random slice only**; use these to predefine a pair-cap sensitivity rule and a feasibility/fallback criterion. **After the freeze, during final evaluation:** report the corresponding T-random counts and shares. Do not inspect T-random outcomes during Week 0.
9. Cold-start rate: the share of test impressions with users or videos unseen in training.
10. Write the data card. Freeze the split, feature rules, ≤13-configuration population, primary metric, six hypothesis comparisons, marginal-frequency weight construction/clipping protocol, RQ3 optimizer-step/boosting-round budget, and analysis plan.
11. **Tag a GitHub commit** containing the data card and analysis plan, and link the tag in the README. Use “frozen before results” only if the tag predates model-result inspection.
12. Before model training, verify which fields identify policy/interface and which randomized rows are actually random within the relevant candidate pool; document limits on exposure-policy generalization.

---

## 12. Timeline (about 3.5–4 weeks)

| Week | Work | Exit criterion |
|---|---|---|
| 0 | Checklist above | Data card + frozen protocol |
| 1 | Point-in-time feature pipeline (the largest engineering task); R0 and R1 configurations | Validation-only baseline diagnostics; hold out both test sets until frozen evaluation |
| 2 | R2, R3, ablations; finish the configuration population (≤3 GPUs in parallel) | Configuration population trained and frozen; normal-validation selections completed; RQ1/RQ2 reported only at the scheduled held-out evaluation |
| 3 | RQ3 arms; freeze; final evaluation | All results |
| 3.5–4 | README, figures, write-up. Stretch goals only if early: light retrieval (popularity + two-tower Recall@N), an artist/category diversity cap | Done |

Durations are [GUESS, MED].

---

## 13. Resources

- **Compute:** LightGBM runs on CPU. The neural networks fit easily on 1 L40S. Up to 3 GPUs only for parallel configurations and seeds.
- **Data:** about 200 MB. Process with polars or DuckDB.
- **Libraries:** polars/DuckDB, LightGBM, PyTorch, scikit-learn. The bootstrap is a small custom user-level resampler.
- **Bootstrap cost is small.** Within-user AUC decomposes over users: it is (sum of correct pairs) / (sum of all pairs). Precompute, per user and per configuration, the two counts once. Every A/B split and bootstrap resample is then a weighted sum over users, with no re-scoring. [COMPUTED] Rough size: 13 configurations × ~27K users × 2 halves × 50 splits × 2,000 resamples ≈ 7×10¹⁰ multiply-adds, one matrix multiply on a GPU (seconds) or minutes on a CPU. [COMPUTED, rough]

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| Leakage in point-in-time features | Unit test: for a random sample of rows, recompute features from raw logs restricted to days < d and assert they match exactly |
| Too few long-view positives on T-random | Week 0 label-rate check on the random slice; documented switch rule |
| Too few within-user pairs | Week 0 pair count; pooled AUC as a fallback, documented before the freeze |
| Branch B2 (no normal logs in the random window) | Regret still works; τ reported as confounded by time |
| MMoE doesn't beat LightGBM | A valid finding; strong tree models are common in practice |
| Regret near zero | Also a finding: the standard workflow is robust here. Report it with its CI |
| Scope creep | PLE, full propensity reconstruction, retrieval, and re-ranking remain out of the core plan |
| Marginal item frequency mistaken for causal propensity | Label the arm as SNIPS-style *heuristic* and report ESS/weight diagnostics; make no unbiased IPS claims |
| Different RQ3 sample counts change gradient-update budgets | Fix neural optimizer steps (not epochs), learning-rate schedule, and batch size across arms |
| Random-domain training improvement overinterpreted | Label it target-domain adaptation, acknowledge data-volume and time-window confounding, avoid causal uplift claims |

---

## 15. What this project can and can't show (for the README)

**Can:**
- multi-task engagement ranking;
- leakage-free point-in-time features;
- calibration;
- exposure bias and model selection under random-exposure evaluation;
- marginal-exposure reweighting as a heuristic and random-exposure data adaptation (not causal proof of debiasing).

**Can't:**
- identification of unbiased counterfactual policy value from marginal item frequencies;
- online A/B testing;
- production serving;
- long-term feedback loops;
- ads auctions;
- a two-stage system, unless the stretch retrieval stage is built.

---

## 16. Resume bullet (fill in after results)

> **Feed Ranking under Exposure Bias (KuaiRand):** Built multi-task engagement rankers (LightGBM, shared-bottom, MMoE) for long-view, click, and like with leakage-free point-in-time features; across 13 model configurations frozen in a tagged protocol before results, measured whether selecting on standard logs picks well on randomly-exposed logs (cross-fit selection regret = X within-user AUC vs Y for random choice among contenders); tested marginal-exposure reweighting, random-exposure data fusion, and random-only training for robustness under logged-policy selection bias.

Use "frozen in a tagged protocol" only if the Week 0 tagged commit exists; otherwise cut that phrase.

---

## 17. Interview talking points

- Why offline gains can fail online, and what randomly exposed data fixes.
- Point-in-time features: the leakage trap in aggregate features and how it was avoided.
- When MMoE helps compared with separate per-task models, and why strong tree models are hard to beat.
- Within-user vs pooled AUC: why pooled AUC rewards learning user base rates, and why pair-weighting suits users with few impressions.
- Why clustered impressions call for a user-level bootstrap instead of DeLong.
- Why selecting a reference model on one random-exposure user half and evaluating it on the other mitigates selection/evaluation optimism, without identifying population-optimal regret.
- Why marginal item frequencies are not user/context-conditional propensities; what a SNIPS-style training heuristic can and cannot claim.
- Why matching optimizer steps matters when normal, random, and joint training sets differ in size.
- Why random-exposure data fusion is supervised domain adaptation, not proof of causal debiasing.
- Calibration: why predicted probabilities matter in feed systems, and in ads systems.
