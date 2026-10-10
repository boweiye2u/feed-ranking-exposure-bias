import polars as pl

from kuairand_pure.week0 import STRUCTURAL_COLUMNS, structural_summary


def test_structural_summary_counts_duplicates_without_outcomes() -> None:
    rows = [
        (1, 10, 20220409, 100, 1, 1),
        (1, 10, 20220411, 200, 2, 1),
        (1, 10, 20220411, 200, 2, 1),
        (2, 11, 20220411, 300, 3, 0),
        (2, 12, 20220412, 400, 4, 1),
    ]
    frame = pl.DataFrame(rows, schema=STRUCTURAL_COLUMNS, orient="row")

    report = structural_summary(frame)

    assert report["scope"]["outcome_columns_loaded"] is False
    assert report["april_11"]["rows"] == 3
    assert report["april_11"]["unique_structural_rows"] == 2
    assert report["april_11"]["duplicate_excess_rows"] == 1
    assert report["april_11"]["duplicate_groups"] == 1
    assert report["april_11"]["maximum_duplicate_multiplicity"] == 2
