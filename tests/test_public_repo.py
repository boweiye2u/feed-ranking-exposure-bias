import re
import subprocess
from pathlib import Path

import pytest

from kuairand_pure.results_package import load_verified_results
from scripts.check_public_repo import local_links

ROOT=Path(__file__).resolve().parents[1]


def test_readme_local_links_figures_and_commands_exist() -> None:
    readme=ROOT/"README.md"; text=readme.read_text()
    assert all(path.exists() for path in local_links(readme))
    figures=re.findall(r"!\[[^]]*\]\((reports/figures/[^)]+\.png)\)",text)
    assert len(figures)==3 and all((ROOT/path).is_file() for path in figures)
    commands=set(re.findall(r"python (scripts/[A-Za-z0-9_./-]+\.py)",text))
    assert commands and all((ROOT/command).is_file() for command in commands)


def test_readme_frozen_metrics_match_sources() -> None:
    required=(
        Path.home()/"recom"/"predictions"/"final"/"v1"/"m6_1_manifest.json",
        Path.home()/"recom"/"evaluation"/"final"/"v1"/"m6_2_results.json",
        Path.home()/"recom"/"evaluation"/"final"/"v1"/"m6_3_rq2_results.json",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("external frozen M6 result artifacts are unavailable")
    results=load_verified_results(); text=(ROOT/"README.md").read_text()
    all_result=results["m6_3"]["results"]["all_candidates"]
    contender=results["m6_3"]["results"]["contenders"]
    assert f"{all_result['normal_selection_regret']:+.6f} AUC" in text
    assert f"{contender['normal_selection_regret']:.6f}" in text
    rq3={row["arm"]:row["metrics"]["pair_weighted_auc"] for row in results["m6_2"]["rq3"]}
    assert all(f"{value:.6f}" in text for value in rq3.values())


def test_no_tracked_large_or_model_data_files() -> None:
    forbidden={".parquet",".arrow",".feather",".pt",".pth",".ckpt",".pkl",".joblib"}
    names=subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).split(b"\0")
    paths=[ROOT/name.decode() for name in names if name]
    assert not [path for path in paths if path.suffix in forbidden or path.stat().st_size>5*1024*1024]
