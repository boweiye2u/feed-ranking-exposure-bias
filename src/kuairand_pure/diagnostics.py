"""M1.2 engagement diagnostics with explicit held-out outcome safeguards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from kuairand_pure.paths import get_artifact_paths
from kuairand_pure.protocol import authorize_outcome_access, load_protocol

RANDOM_TRAIN_START = 20220422
RANDOM_TRAIN_END = 20220429
NORMAL_TRAIN_START = 20220411
NORMAL_TRAIN_END = 20220429
TEST_START = 20220503
TEST_END = 20220508
OUTCOME_COLUMNS = ("long_view", "is_click", "is_like")
PERMITTED_COLUMNS = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "tab",
    "long_view",
    "is_click",
    "is_like",
    "play_time_ms",
    "duration_ms",
)
IDENTIFIER_COLUMNS = ("user_id", "video_id", "date")


def _scan(path: Path, columns: Iterable[str]) -> pl.LazyFrame:
    """Project the named columns at scan time."""
    return pl.scan_csv(path).select(*columns)


def load_permitted_random_slice(path: Path) -> pl.DataFrame:
    """Load outcomes only for the protocol-permitted Apr 22--29 slice."""
    authorize_outcome_access(
        load_protocol(),
        source="random",
        start=RANDOM_TRAIN_START,
        end=RANDOM_TRAIN_END,
        workflow="week0",
        purpose="diagnostics",
    )
    frame = (
        _scan(path, PERMITTED_COLUMNS)
        .filter(pl.col("date").is_between(RANDOM_TRAIN_START, RANDOM_TRAIN_END))
        .collect()
    )
    if frame.is_empty():
        raise ValueError("The permitted randomized training slice is empty")
    if frame["date"].min() < RANDOM_TRAIN_START or frame["date"].max() > RANDOM_TRAIN_END:
        raise AssertionError("Outcome access escaped the permitted randomized training slice")
    return frame


def load_id_window(path: Path, start: int, end: int) -> pl.DataFrame:
    """Load identifiers only; safe for held-out windows."""
    return (
        _scan(path, IDENTIFIER_COLUMNS)
        .filter(pl.col("date").is_between(start, end))
        .collect()
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0}
    quantiles = np.quantile(
        values,
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
        method="linear",
    )
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "minimum": int(quantiles[0]),
        "p10": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p90": float(quantiles[5]),
        "p95": float(quantiles[6]),
        "p99": float(quantiles[7]),
        "maximum": int(quantiles[8]),
    }


def engagement_diagnostics(frame: pl.DataFrame) -> dict[str, Any]:
    """Compute permitted label, interface, and within-user diagnostics."""
    row_count = frame.height
    label_counts = frame.select(
        *(pl.col(column).sum().alias(column) for column in OUTCOME_COLUMNS)
    ).row(0, named=True)
    labels = {
        column: {
            "positive_count": int(label_counts[column]),
            "negative_count": row_count - int(label_counts[column]),
            "positive_rate": _rate(int(label_counts[column]), row_count),
        }
        for column in OUTCOME_COLUMNS
    }

    valid_play = (
        pl.when(pl.col("duration_ms") <= 7_000)
        .then(pl.col("play_time_ms") >= pl.col("duration_ms"))
        .otherwise(pl.col("play_time_ms") > 7_000)
        .cast(pl.Int64)
    )
    derived_long_view = (
        pl.when(pl.col("duration_ms") <= 18_000)
        .then(pl.col("play_time_ms") >= pl.col("duration_ms"))
        .otherwise(pl.col("play_time_ms") >= 18_000)
        .cast(pl.Int64)
    )
    enriched = frame.with_columns(
        (valid_play != pl.col("is_click")).cast(pl.Int64).alias("click_formula_mismatch"),
        (derived_long_view != pl.col("long_view"))
        .cast(pl.Int64)
        .alias("long_view_formula_mismatch"),
        ((pl.col("is_like") == 1) & (pl.col("is_click") == 1))
        .cast(pl.Int64)
        .alias("click_and_like"),
        ((pl.col("is_like") == 1) & (pl.col("is_click") == 0))
        .cast(pl.Int64)
        .alias("like_without_click"),
    )
    tab_rows = (
        enriched.group_by("tab")
        .agg(
            pl.len().alias("impressions"),
            *(pl.col(column).sum().alias(f"{column}_positives") for column in OUTCOME_COLUMNS),
            pl.col("click_and_like").sum(),
            pl.col("like_without_click").sum(),
            pl.col("click_formula_mismatch").sum(),
            pl.col("long_view_formula_mismatch").sum(),
        )
        .sort("impressions", descending=True)
    )

    by_tab: list[dict[str, Any]] = []
    for rank, row in enumerate(tab_rows.iter_rows(named=True), start=1):
        impressions = int(row["impressions"])
        clicks = int(row["is_click_positives"])
        likes = int(row["is_like_positives"])
        click_and_like = int(row["click_and_like"])
        like_without_click = int(row["like_without_click"])
        by_tab.append(
            {
                "frequency_rank": rank,
                "tab": int(row["tab"]),
                "impressions": impressions,
                "impression_share": impressions / row_count,
                "long_view_positive_count": int(row["long_view_positives"]),
                "long_view_rate": int(row["long_view_positives"]) / impressions,
                "click_positive_count": clicks,
                "click_rate": clicks / impressions,
                "like_positive_count": likes,
                "like_rate": likes / impressions,
                "click_and_like_count": click_and_like,
                "like_without_click_count": like_without_click,
                "share_of_likes_without_click": _rate(like_without_click, likes),
                "like_rate_given_click": _rate(click_and_like, clicks),
                "like_rate_given_no_click": _rate(like_without_click, impressions - clicks),
                "click_formula_mismatch_count": int(row["click_formula_mismatch"]),
                "click_formula_match_rate": 1
                - int(row["click_formula_mismatch"]) / impressions,
                "long_view_formula_mismatch_count": int(row["long_view_formula_mismatch"]),
                "long_view_formula_match_rate": 1
                - int(row["long_view_formula_mismatch"]) / impressions,
            }
        )

    per_user = enriched.group_by("user_id").agg(
        pl.len().alias("impressions"),
        pl.col("long_view").sum().cast(pl.Int64).alias("positives"),
    ).with_columns(
        (pl.col("impressions") - pl.col("positives")).alias("negatives")
    ).with_columns(
        (pl.col("positives") * pl.col("negatives")).alias("pair_count")
    )
    impressions = per_user["impressions"].to_numpy()
    pairs = per_user["pair_count"].to_numpy()
    eligible_pairs = pairs[pairs > 0]
    total_pairs = int(pairs.sum())
    top_user_count = math.ceil(0.10 * len(pairs))
    top_pairs = int(np.sort(pairs)[::-1][:top_user_count].sum())
    proposed_cap = int(np.quantile(eligible_pairs, 0.95, method="higher"))
    capped_pairs = int(np.minimum(pairs, proposed_cap).sum())

    return {
        "permitted_slice": {
            "start_date": RANDOM_TRAIN_START,
            "end_date": RANDOM_TRAIN_END,
            "row_count": row_count,
            "unique_users": frame["user_id"].n_unique(),
            "unique_videos": frame["video_id"].n_unique(),
        },
        "label_frequencies": labels,
        "by_tab": by_tab,
        "long_view_pairs": {
            "all_users": len(pairs),
            "users_with_both_classes": int((pairs > 0).sum()),
            "total_positive_negative_pairs": total_pairs,
            "top_10_percent_definition": "ceil(10% of all slice users), ranked by pair_count descending",
            "top_10_percent_user_count": top_user_count,
            "top_10_percent_pair_count": top_pairs,
            "top_10_percent_pair_share": _rate(top_pairs, total_pairs),
            "impression_count_distribution_all_users": _distribution(impressions),
            "pair_count_distribution_all_users": _distribution(pairs),
            "pair_count_distribution_eligible_users": _distribution(eligible_pairs),
            "proposed_pair_cap": {
                "status": "proposed_for_review",
                "rule": "95th percentile of nonzero pre-validation per-user pair counts, nearest observed value at or above the quantile",
                "cap": proposed_cap,
                "uncapped_total_weight": total_pairs,
                "capped_total_weight": capped_pairs,
                "weight_retained": _rate(capped_pairs, total_pairs),
                "metric_definition": "weight each eligible user's AUC by min(n_positive*n_negative, cap)",
            },
        },
    }


def _cold_start(test: pl.DataFrame, seen: pl.DataFrame) -> dict[str, Any]:
    seen_users = set(seen["user_id"].to_list())
    seen_videos = set(seen["video_id"].to_list())
    user_unseen = ~test["user_id"].is_in(seen_users)
    video_unseen = ~test["video_id"].is_in(seen_videos)
    either_unseen = user_unseen | video_unseen
    test_users = set(test["user_id"].to_list())
    test_videos = set(test["video_id"].to_list())
    unseen_users = test_users - seen_users
    unseen_videos = test_videos - seen_videos
    return {
        "test_impressions": test.height,
        "test_unique_users": len(test_users),
        "test_unique_videos": len(test_videos),
        "unseen_unique_users": len(unseen_users),
        "unseen_unique_user_rate": _rate(len(unseen_users), len(test_users)),
        "unseen_unique_videos": len(unseen_videos),
        "unseen_unique_video_rate": _rate(len(unseen_videos), len(test_videos)),
        "impressions_with_unseen_user": int(user_unseen.sum()),
        "impression_unseen_user_rate": float(user_unseen.mean()),
        "impressions_with_unseen_video": int(video_unseen.sum()),
        "impression_unseen_video_rate": float(video_unseen.mean()),
        "impressions_with_either_unseen": int(either_unseen.sum()),
        "impression_either_unseen_rate": float(either_unseen.mean()),
    }


def cold_start_diagnostics(data_dir: Path) -> dict[str, Any]:
    """Compare training and test identifiers without loading test outcomes."""
    early = data_dir / "log_standard_4_08_to_4_21_pure.csv"
    later = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    random = data_dir / "log_random_4_22_to_5_08_pure.csv"
    normal_train = pl.concat(
        [
            load_id_window(early, NORMAL_TRAIN_START, NORMAL_TRAIN_END),
            load_id_window(later, NORMAL_TRAIN_START, NORMAL_TRAIN_END),
        ]
    )
    random_train = load_id_window(random, RANDOM_TRAIN_START, RANDOM_TRAIN_END)
    joint_train = pl.concat([normal_train, random_train])
    tests = {
        "t_normal": load_id_window(later, TEST_START, TEST_END),
        "t_random": load_id_window(random, TEST_START, TEST_END),
    }
    sources = {
        "normal_only_apr11_29": normal_train,
        "random_only_apr22_29": random_train,
        "joint_normal_and_random": joint_train,
    }
    return {
        test_name: {
            source_name: _cold_start(test, source)
            for source_name, source in sources.items()
        }
        for test_name, test in tests.items()
    }


def timestamp_consistency(path: Path) -> dict[str, Any]:
    """Compare calendar fields with epoch time without loading outcomes."""
    frame = _scan(path, ("date", "hourmin", "time_ms"))
    utc = pl.from_epoch(pl.col("time_ms"), time_unit="ms")
    china = utc + pl.duration(hours=8)
    encoded_hour = (pl.col("hourmin") // 100).cast(pl.Int64)
    china_hour = china.dt.hour().cast(pl.Int64)
    hour_distance = (encoded_hour - china_hour).abs()
    circular_hour_distance = pl.min_horizontal(hour_distance, 24 - hour_distance)
    checked = frame.select(
        pl.len().alias("rows"),
        (pl.col("date") == utc.dt.strftime("%Y%m%d").cast(pl.Int64))
        .sum()
        .alias("utc_date_matches"),
        (pl.col("date") == china.dt.strftime("%Y%m%d").cast(pl.Int64))
        .sum()
        .alias("utc_plus_8_date_matches"),
        (
            pl.col("hourmin")
            == (utc.dt.hour().cast(pl.Int64) * 100 + utc.dt.minute().cast(pl.Int64))
        )
        .sum()
        .alias("utc_hhmm_matches"),
        (
            pl.col("hourmin")
            == (china.dt.hour().cast(pl.Int64) * 100 + china.dt.minute().cast(pl.Int64))
        )
        .sum()
        .alias("utc_plus_8_hhmm_matches"),
        (pl.col("hourmin") == utc.dt.hour().cast(pl.Int64) * 100)
        .sum()
        .alias("utc_hour_bucket_matches"),
        (pl.col("hourmin") == china.dt.hour().cast(pl.Int64) * 100)
        .sum()
        .alias("utc_plus_8_hour_bucket_matches"),
        (circular_hour_distance <= 1)
        .sum()
        .alias("utc_plus_8_hour_within_one_bucket"),
        (
            (pl.col("date") == china.dt.strftime("%Y%m%d").cast(pl.Int64))
            & (pl.col("hourmin") == china.dt.hour().cast(pl.Int64) * 100)
        )
        .sum()
        .alias("utc_plus_8_date_and_hour_bucket_matches"),
    ).collect().row(0, named=True)
    rows = checked["rows"]
    return {
        **checked,
        "utc_date_match_rate": checked["utc_date_matches"] / rows,
        "utc_plus_8_date_match_rate": checked["utc_plus_8_date_matches"] / rows,
        "utc_hhmm_match_rate": checked["utc_hhmm_matches"] / rows,
        "utc_plus_8_hhmm_match_rate": checked["utc_plus_8_hhmm_matches"] / rows,
        "utc_hour_bucket_match_rate": checked["utc_hour_bucket_matches"] / rows,
        "utc_plus_8_hour_bucket_match_rate": checked["utc_plus_8_hour_bucket_matches"] / rows,
        "utc_plus_8_hour_within_one_bucket_rate": checked[
            "utc_plus_8_hour_within_one_bucket"
        ]
        / rows,
        "utc_plus_8_date_and_hour_bucket_match_rate": checked[
            "utc_plus_8_date_and_hour_bucket_matches"
        ]
        / rows,
    }


def burn_in_diagnostics(data_dir: Path) -> dict[str, Any]:
    """Count structural rows under the two reviewable burn-in alternatives."""
    early = data_dir / "log_standard_4_08_to_4_21_pure.csv"
    later = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    dates = pl.concat(
        [_scan(early, ("date",)), _scan(later, ("date",))]
    ).collect().get_column("date")
    count = lambda start, end: int(dates.is_between(start, end).sum())
    return {
        "observed_minimum_date": int(dates.min()),
        "april_8_rows": int((dates == 20220408).sum()),
        "frozen_train_apr11_29_rows": count(20220411, 20220429),
        "apr11_rows": int((dates == 20220411).sum()),
        "alternative_train_apr12_29_rows": count(20220412, 20220429),
        "alternatives": [
            {
                "name": "preserve_apr11_start",
                "status": "requires_review",
                "effect": "Keeps the frozen row dates but supplies only two observed burn-in days (Apr 9-10), violating the stated three-day minimum.",
            },
            {
                "name": "start_training_apr12",
                "status": "proposed_for_review",
                "effect": "Uses Apr 9-11 as three observed burn-in days and removes Apr 11 rows from training; this changes the frozen split.",
            },
        ],
    }


def interface_frequency_evidence(data_dir: Path) -> dict[str, Any]:
    """Apply the author's frequency-ranking guidance without reading outcomes."""
    normal = pl.concat(
        [
            _scan(data_dir / "log_standard_4_08_to_4_21_pure.csv", ("tab",)),
            _scan(data_dir / "log_standard_4_22_to_5_08_pure.csv", ("tab",)),
        ]
    )
    rows = (
        normal.group_by("tab")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .collect()
        .to_dicts()
    )
    return {
        "documented_rule": "Dataset owner states that the most frequent tab is single-column, the second most frequent is two-column, and remaining tabs are sub-scenarios.",
        "scope_used_for_ranking": "both complete standard-log files; tab only",
        "normal_tab_frequencies": rows,
        "frequency_based_mapping": {
            "single_column_tab": rows[0]["tab"],
            "two_column_tab": rows[1]["tab"],
            "status": "documented frequency rule applied to observed release; numeric mapping is inferred",
        },
    }


def run_diagnostics(data_dir: Path) -> dict[str, Any]:
    random_path = data_dir / "log_random_4_22_to_5_08_pure.csv"
    permitted = load_permitted_random_slice(random_path)
    log_names = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    )
    return {
        "milestone": "M1.2",
        "branch": "B1",
        "outcome_access_policy": {
            "outcome_columns_loaded_only_for": "randomized rows dated 2022-04-22 through 2022-04-29",
            "randomized_apr30_may2_outcomes_accessed": False,
            "may3_may8_outcomes_accessed": False,
            "test_overlap_columns": list(IDENTIFIER_COLUMNS),
        },
        "engagement": engagement_diagnostics(permitted),
        "cold_start": cold_start_diagnostics(data_dir),
        "timestamp_consistency": {
            name: timestamp_consistency(data_dir / name) for name in log_names
        },
        "interface_frequency_evidence": interface_frequency_evidence(data_dir),
        "burn_in": burn_in_diagnostics(data_dir),
        "video_statistics": {
            "status": "excluded",
            "reason": "Official documentation says statistics are daily/scenario averages over one month but provides no exact source window or pre-April-8 cutoff.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    paths = get_artifact_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=paths.raw / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.processed / "audits" / "m1_2_engagement_diagnostics.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_diagnostics(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
