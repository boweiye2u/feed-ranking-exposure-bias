"""Verified M7.1 research tables, figures, and reproducibility inventory."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightgbm
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import polars as pl
import pyarrow
import sklearn
import torch

from kuairand_pure.paths import get_artifact_paths

EXPECTED_M6_HASHES = {
    "m6_1_manifest": "de70509f06e190ade3317fb93565a20fcc74c90634c15c388a2a0c92ac02f532",
    "m6_2_results": "dab05b76c9830e4d4edb4f4d197397ed73819edc89f18a196332531bcab91214",
    "m6_3_results": "8f34cdb9791e123dcfc1bc1d5131ba2aac51147d13d8d5d3ca40b74d7e782ab9",
}
EXPECTED_TAG_COMMITS = {
    "week0-protocol-v1": "84dde526189270a8df71cc100bf3f9a3c4c768ea",
    "m2-features-v1": "5f0b0ea12fc89f0c61c8b4947428b39320b6a19b",
    "m4-candidates-v1": "6f30ce489cfc5f5009e6373e2e10de0f4a515b6a",
    "m5-prefinal-v1": "c936a93cf2c81efb95840a1e8a52e42437f8fb54",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_results() -> dict[str, Any]:
    paths = get_artifact_paths()
    final_predictions = paths.predictions / "final" / "v1"
    final_evaluation = paths.root / "evaluation" / "final" / "v1"
    sources = {
        "m6_1_manifest": final_predictions / "m6_1_manifest.json",
        "m6_2_results": final_evaluation / "m6_2_results.json",
        "m6_3_results": final_evaluation / "m6_3_rq2_results.json",
    }
    observed = {name: sha256(path) for name, path in sources.items()}
    if observed != EXPECTED_M6_HASHES:
        raise ValueError(f"Frozen M6 result hashes changed: {observed}")
    m61, m62, m63 = (json.loads(sources[name].read_text()) for name in sources)
    tag_commits = {tag: subprocess.check_output(
        ["git", "rev-list", "-n", "1", tag], cwd=_repo_root(), text=True
    ).strip() for tag in EXPECTED_TAG_COMMITS}
    if tag_commits != EXPECTED_TAG_COMMITS:
        raise ValueError("A frozen Git tag target changed")
    protocol_paths = {
        "study_plan_v5": _repo_root() / "docs" / "study_plan_v5.md",
        "week0_amendment_v1": _repo_root() / "docs" / "week0_amendment_v1.md",
        "feature_manifest": paths.processed / "m2" / "v1" / "feature_manifest.json",
    }
    protocol_hashes = {name: sha256(path) for name, path in protocol_paths.items()}
    if protocol_hashes != m61["protocol_hashes"]:
        raise ValueError("A frozen protocol or feature manifest changed")
    m62_manifest = json.loads((final_evaluation / "manifest.json").read_text())
    if any(sha256(final_evaluation / name) != digest for name, digest in m62_manifest["outputs"].items()):
        raise ValueError("An M6.2 output changed")
    m63_manifest = json.loads((final_evaluation / "m6_3_manifest.json").read_text())
    m63_paths = {"results": final_evaluation / "m6_3_rq2_results.json",
                 "user_statistics": final_evaluation / "rq2_user_statistics.parquet",
                 "selection_frequencies": final_evaluation / "rq2_selection_frequencies.parquet"}
    if any(sha256(m63_paths[name]) != digest for name, digest in m63_manifest["outputs"].items()):
        raise ValueError("An M6.3 output changed")
    prediction_report = m61["predictions"]
    if prediction_report["record_count"] != 146:
        raise ValueError("M6.1 prediction population is incomplete")
    for record in prediction_report["records"]:
        if sha256(Path(record["path"])) != record["prediction_sha256"]:
            raise ValueError("An M6.1 prediction checksum changed")
    candidates = sorted({r["candidate"] for r in m62["rq1_long_view_primary"]})
    if len(candidates) != 12:
        raise ValueError("RQ1/RQ2 candidate population is not exactly 12")
    contenders = m63["results"]["contenders"]["candidate_population"]
    if contenders != ["fs1_deep", "fs1_shallow", "mmoe_no_aggregates"]:
        raise ValueError("Frozen contender set changed")
    if [r["arm"] for r in m62["rq3"]] != ["normal_only_baseline", "marginal_reweighting", "joint", "random_only"]:
        raise ValueError("RQ3 arm population changed")
    expected_comparisons = [
        "mmoe_full_seed0_minus_fs1_deep", "mmoe_full_seed0_minus_shared_bottom_seed0",
        "mmoe_dense_only_seed0_minus_fs3_deep", "rq3_marginal_reweighting_minus_fs1_deep",
        "rq3_joint_minus_fs1_deep", "rq3_random_only_minus_fs1_deep",
    ]
    if [r["comparison"] for r in m62["paired_comparisons"]] != expected_comparisons:
        raise ValueError("Six predefined comparisons changed")
    return {"m6_1": m61, "m6_2": m62, "m6_3": m63, "source_hashes": observed,
            "candidate_ids": candidates, "contenders": contenders,
            "tag_commits": tag_commits, "protocol_hashes": protocol_hashes}


def result_tables(results: dict[str, Any]) -> dict[str, pl.DataFrame]:
    m62, m63 = results["m6_2"], results["m6_3"]
    rq1 = pl.DataFrame([{"source": r["source"], "candidate": r["candidate"], **r["metrics"]}
                        for r in m62["rq1_long_view_primary"]]).sort("candidate", "source")
    rq2 = pl.DataFrame([{"population": name, **{k: v for k, v in values.items()
                         if k != "reference_selection_frequency" and k != "candidate_population"}}
                        for name, values in m63["results"].items()])
    rq3 = pl.DataFrame([{"arm": r["arm"], "training_rows": r["training_rows"],
                         "difference_vs_baseline": r["pair_weighted_auc_difference_vs_normal_only"], **r["metrics"]}
                        for r in m62["rq3"]])
    comparisons = pl.DataFrame(m62["paired_comparisons"])
    populations = pl.DataFrame([{"source": source, "task": task, **values}
        for source, tasks in m62["population_diagnostics"].items() for task, values in tasks.items()])
    return {"rq1_long_view": rq1, "rq2_regret": rq2, "rq3": rq3,
            "paired_comparisons": comparisons, "populations": populations}


def _save(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight", metadata={"Software": "kuairand_pure M7.1"})
    plt.close(fig)


def generate_figures(results: dict[str, Any], tables: dict[str, pl.DataFrame], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.titlesize": 12,
                         "svg.hashsalt": "kuairand-pure-m7-v1"})
    # 1: chronology.
    fig, ax = plt.subplots(figsize=(11, 3.2))
    spans = [("Burn-in", 0, 3, "#9ecae1"), ("Normal train", 3, 18, "#3182bd"),
             ("Random train", 13, 8, "#74c476"), ("Normal validation", 21, 3, "#fdae6b"),
             ("T-normal / T-random", 24, 6, "#de2d26")]
    for y, (label, start, width, color) in enumerate(spans):
        ax.broken_barh([(start, width)], (y - .35, .7), facecolors=color)
    ax.set_yticks(range(len(spans)), [x[0] for x in spans]); ax.set_xlim(0, 30)
    ax.set_xticks([0, 3, 13, 21, 24, 30], ["Apr 9", "Apr 12", "Apr 22", "Apr 30", "May 3", "May 9"])
    ax.set_title("Frozen chronological design (end ticks are exclusive)"); ax.grid(axis="x", alpha=.25)
    _save(fig, output / "01_dataset_chronology")
    # 2: RQ1 paired bars.
    rq1 = tables["rq1_long_view"]
    order = ["r0", "fs1_shallow", "fs1_deep", "fs2_shallow", "fs2_deep", "fs3_shallow", "fs3_deep",
             "shared_bottom", "mmoe_full", "mmoe_dense_only", "mmoe_no_history", "mmoe_no_aggregates"]
    values = {(r["candidate"], r["source"]): r["pair_weighted_auc"] for r in rq1.to_dicts()}
    x = numpy.arange(len(order)); width = .38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x-width/2, [values[(c,"normal")] for c in order], width, label="T-normal")
    ax.bar(x+width/2, [values[(c,"random")] for c in order], width, label="T-random")
    ax.axhline(.5, color="black", lw=.8, ls="--", label="Chance AUC")
    ax.set_ylim(.5, .68); ax.set_ylabel("Pair-weighted within-user AUC")
    ax.set_xticks(x, [c.replace("_", "\n") for c in order]); ax.legend(ncol=3); ax.grid(axis="y", alpha=.2)
    ax.set_title("RQ1 frozen candidate population (neural seed 0)")
    _save(fig, output / "02_rq1_auc_comparison")
    # 3: RQ2 regret.
    r2 = results["m6_3"]["results"]
    labels = ["All 12\nNormal selection", "All 12\nRandom pick", "Contenders\nNormal selection", "Contenders\nRandom pick"]
    points = [r2["all_candidates"]["normal_selection_regret"], r2["all_candidates"]["random_pick_regret"],
              r2["contenders"]["normal_selection_regret"], r2["contenders"]["random_pick_regret"]]
    cis = [r2["all_candidates"]["normal_selection_regret_percentile_95_ci"], r2["all_candidates"]["random_pick_regret_percentile_95_ci"],
           r2["contenders"]["normal_selection_regret_percentile_95_ci"], r2["contenders"]["random_pick_regret_percentile_95_ci"]]
    fig, ax = plt.subplots(figsize=(8, 4.5)); xx=numpy.arange(4)
    ax.errorbar(xx, points, yerr=[[p-c[0] for p,c in zip(points,cis)],[c[1]-p for p,c in zip(points,cis)]], fmt="o", capsize=5)
    ax.axhline(0,color="black",lw=.8); ax.set_xticks(xx,labels); ax.set_ylabel("Cross-fitted regret (AUC)")
    ax.set_title("RQ2 percentile intervals; references selected on disjoint user halves"); ax.grid(axis="y",alpha=.2)
    _save(fig, output / "03_rq2_selection_regret")
    # 4: RQ3.
    rq3=tables["rq3"].to_dicts(); arm_labels=["Normal-only","Marginal","Joint","Random-only"]
    fig,(a,b)=plt.subplots(1,2,figsize=(10,4.2)); xx=numpy.arange(4)
    a.bar(xx,[r["pair_weighted_auc"] for r in rq3]); a.axhline(.5,color="black",ls="--",lw=.8); a.set_ylim(.5,.70)
    a.set_ylabel("T-random pair-weighted AUC"); b.bar(xx,[r["log_loss"] for r in rq3],color="#e6550d"); b.set_ylabel("T-random LogLoss (lower is better)")
    for axis in (a,b): axis.set_xticks(xx,arm_labels,rotation=20,ha="right"); axis.grid(axis="y",alpha=.2)
    fig.suptitle("RQ3 frozen training strategies (not a causal decomposition)")
    _save(fig, output / "04_rq3_auc_logloss")
    # 5: prespecified paired comparisons.
    comps=tables["paired_comparisons"].to_dicts(); names=[r["comparison"].replace("_minus_"," − ").replace("_"," ") for r in comps]
    points=[r["observed_pair_weighted_auc_difference"] for r in comps]; cis=[r["percentile_95_ci"] for r in comps]
    fig,ax=plt.subplots(figsize=(9,5.2)); yy=numpy.arange(6)
    ax.errorbar(points,yy,xerr=[[p-c[0] for p,c in zip(points,cis)],[c[1]-p for p,c in zip(points,cis)]],fmt="o",capsize=4)
    ax.axvline(0,color="black",lw=.8); ax.set_yticks(yy,names); ax.invert_yaxis(); ax.set_xlabel("T-random pair-weighted AUC difference")
    ax.set_title("Six prespecified paired comparisons: 95% percentile intervals"); ax.grid(axis="x",alpha=.2)
    _save(fig, output / "05_six_paired_comparisons")
    # 6: concentration and weighting sensitivity.
    pop=results["m6_2"]["population_diagnostics"]
    fig,(a,b)=plt.subplots(1,2,figsize=(11,4.2))
    a.bar([0,1],[pop["normal"]["long_view"]["top_10_percent_pair_share"],pop["random"]["long_view"]["top_10_percent_pair_share"]])
    a.set_xticks([0,1],["T-normal","T-random"]); a.set_ylim(0,1); a.set_ylabel("Share of long-view pairs"); a.set_title("Pairs contributed by top 10% of users")
    random_rows={r["candidate"]:r for r in rq1.to_dicts() if r["source"]=="random"}
    yy=numpy.arange(len(order)); equal_delta=[random_rows[c]["pair_weighted_auc"]-random_rows[c]["per_user_average_auc"] for c in order]
    cap_delta=[random_rows[c]["capped_pair_weighted_auc"]-random_rows[c]["pair_weighted_auc"] for c in order]
    b.scatter(equal_delta,yy,label="Primary − equal-user",marker="o"); b.scatter(cap_delta,yy,label="952-capped − primary",marker="s")
    b.axvline(0,color="black",lw=.7); b.set_yticks(yy,[c.replace("mmoe_","").replace("_"," ") for c in order],fontsize=7)
    b.invert_yaxis(); b.set_xlabel("AUC difference"); b.set_title("T-random weighting sensitivity"); b.legend(fontsize=7); b.grid(axis="x",alpha=.2)
    fig.suptitle("User activity concentration and frozen sensitivity metrics")
    _save(fig, output / "06_pair_concentration_sensitivity")


def build_results_package() -> dict[str, Any]:
    results = load_verified_results(); repo = _repo_root()
    table_dir, figure_dir = repo / "reports" / "tables", repo / "reports" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    tables = result_tables(results)
    for name, frame in tables.items():
        safe = frame.with_columns(
            *(pl.col(column).map_elements(
                lambda value: json.dumps(value.to_list() if hasattr(value, "to_list") else value),
                return_dtype=pl.String)
              for column, dtype in frame.schema.items() if dtype.is_nested())
        )
        safe.write_csv(table_dir / f"{name}.csv")
    generate_figures(results, tables, figure_dir)
    inventory = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {"numpy": numpy.__version__, "polars": pl.__version__, "pyarrow": pyarrow.__version__,
                     "pandas": pandas.__version__, "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                     "torch": torch.__version__, "matplotlib": matplotlib.__version__},
        "hardware": {"cpu": "AMD EPYC 7573X; 2 sockets; 64 physical cores; 128 logical CPUs", "cuda_available": torch.cuda.is_available()},
        "execution": {"lightgbm_threads": 1, "neural_device": "cpu", "neural_seeds": [0,1,2], "lightgbm_seed": 202211},
        "frozen_tags": results["tag_commits"], "protocol_hashes": results["protocol_hashes"],
        "source_hashes": results["source_hashes"],
        "excluded_large_artifacts": ["raw data","processed Parquet tables","model checkpoints","prediction Parquet files"],
    }
    inventory_path=table_dir/"reproducibility_inventory.json"; inventory_path.write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
    outputs=[*table_dir.glob("*"),*figure_dir.glob("*.svg"),*figure_dir.glob("*.png")]
    manifest={"source_hashes":results["source_hashes"],"outputs":{str(p.relative_to(repo)):sha256(p) for p in sorted(outputs)}}
    (figure_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return {"tables":len(tables),"figures":6,"manifest":manifest,"inventory":inventory}
