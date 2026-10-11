import json
from pathlib import Path

import pytest

from kuairand_pure.results_package import EXPECTED_M6_HASHES, load_verified_results, result_tables, sha256


FROZEN_RESULTS = (
    Path.home() / "recom" / "predictions" / "final" / "v1" / "m6_1_manifest.json",
    Path.home() / "recom" / "evaluation" / "final" / "v1" / "m6_2_results.json",
    Path.home() / "recom" / "evaluation" / "final" / "v1" / "m6_3_rq2_results.json",
)
pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in FROZEN_RESULTS),
    reason="external frozen M6 result artifacts are unavailable",
)


def test_final_table_provenance_and_candidate_population() -> None:
    results=load_verified_results(); tables=result_tables(results)
    assert results["source_hashes"] == EXPECTED_M6_HASHES
    assert len(results["candidate_ids"]) == 12
    assert tables["rq1_long_view"].height == 24
    assert tables["rq3"].height == 4
    assert tables["paired_comparisons"].height == 6


def test_report_values_come_from_frozen_machine_results() -> None:
    results=load_verified_results(); tables=result_tables(results)
    fs1=tables["rq1_long_view"].filter((tables["rq1_long_view"]["candidate"]=="fs1_deep") & (tables["rq1_long_view"]["source"]=="random"))
    assert fs1["pair_weighted_auc"].item() == results["m6_2"]["rq3"][0]["metrics"]["pair_weighted_auc"]
    assert results["m6_3"]["results"]["contenders"]["normal_selection_regret_percentile_95_ci"][1] == 0.0


def test_final_report_and_figure_manifest_are_provenance_bound() -> None:
    root=Path(__file__).resolve().parents[1]
    report=(root/"reports"/"final_research_report.md").read_text()
    assert "0.027714 [0.024180, 0.030318]" in report
    assert "-0.000527 [-0.001580, 0]" in report
    assert "not an exact p-value of zero" in report
    manifest=json.loads((root/"reports"/"figures"/"manifest.json").read_text())
    assert manifest["source_hashes"] == EXPECTED_M6_HASHES
    assert len([name for name in manifest["outputs"] if name.endswith(".svg")]) == 6
    assert all(sha256(root/name) == digest for name,digest in manifest["outputs"].items())
