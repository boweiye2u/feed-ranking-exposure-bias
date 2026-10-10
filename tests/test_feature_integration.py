import polars as pl

from kuairand_pure.data.fs1 import model_feature_columns
from kuairand_pure.features.integration import build_feature_manifest


def test_manifest_lists_every_nested_feature_and_sequence_field() -> None:
    schemas = {
        "fs1": pl.Schema({"impression_id": pl.UInt64, "context__is_rand": pl.Int64, "user__x": pl.Int64}),
        "fs2": pl.Schema({"impression_id": pl.UInt64, "context__is_rand": pl.Int64, "user__x": pl.Int64, "user__hist_impression_count": pl.Int64}),
        "fs3": pl.Schema({"impression_id": pl.UInt64, "context__is_rand": pl.Int64, "user__x": pl.Int64, "user__hist_impression_count": pl.Int64, "category__key": pl.String}),
    }
    sequence = pl.Schema({"impression_id": pl.UInt64, "history_video_ids": pl.Array(pl.Int32, 20), "history_mask": pl.Array(pl.UInt8, 20), "history_length": pl.Int16})

    manifest = build_feature_manifest(schemas, sequence)

    assert [row["name"] for row in manifest["feature_fields"]] == schemas["fs3"].names()
    assert [row["name"] for row in manifest["sequence_fields"]] == sequence.names()
    assert manifest["feature_counts"] == {"fs1": 3, "fs2": 4, "fs3": 5}
    category = next(row for row in manifest["feature_fields"] if row["name"] == "category__key")
    assert category["missing_value_policy"] == "explicit___MISSING___category"


def test_exposure_selector_contract_applies_to_nested_feature_sets() -> None:
    names = ["impression_id", "context__tab", "context__is_rand", "user__x", "user__hist_impression_count", "category__key"]

    assert "context__is_rand" not in model_feature_columns(names, "baseline")
    assert "context__is_rand" not in model_feature_columns(names, "rq3_random_only")
    assert "context__is_rand" in model_feature_columns(names, "rq3_joint")
