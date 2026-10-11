#!/usr/bin/env python3
"""Validate all M4.4 bundles and write the aggregate checkpoint manifest."""

import hashlib
import json
from pathlib import Path

import torch

from kuairand_pure.models.neural import MMOE_ABLATION_SPECS, MMoE
from kuairand_pure.paths import get_artifact_paths
from kuairand_pure.training.neural import load_checkpoint
from kuairand_pure.training.neural_data import NeuralPreprocessor, load_neural_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


config = load_neural_config()
budget = config["m4_2_training_budget"]
paths = get_artifact_paths()
root = paths.models / "neural" / "ablations" / "v1"
feature_manifest_path = paths.processed / "m2" / "v1" / "feature_manifest.json"
preprocessor_path = paths.models / "neural" / "infrastructure" / "v1" / "preprocessor.json"
feature_manifest_sha256 = sha256(feature_manifest_path)
preprocessing_sha256 = sha256(preprocessor_path)
preprocessor = NeuralPreprocessor.load(
    preprocessor_path, expected_manifest_sha256=feature_manifest_sha256
)
dimensions = {"dense_only": 238, "no_history": 270, "no_aggregates": 220}
entries = []
for ablation, dimension in dimensions.items():
    for seed in (0, 1, 2):
        directory = root / ablation / f"seed_{seed}"
        report_path = directory / "run_report.json"
        checkpoint_path = directory / "checkpoint.pt"
        checkpoint_manifest_path = directory / "checkpoint_manifest.json"
        if not all(path.is_file() for path in (report_path, checkpoint_path, checkpoint_manifest_path)):
            raise ValueError(f"Incomplete M4.4 bundle: {ablation} seed {seed}")
        report = json.loads(report_path.read_text())
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
        expected_architecture = f"mmoe_{ablation}"
        checks = {
            "ablation": report["ablation"] == ablation,
            "architecture": report["architecture"] == expected_architecture,
            "seed": report["seed"] == seed,
            "dimension": report["input_dimension"] == dimension,
            "updates": report["optimizer_updates"] == budget["optimizer_updates"],
            "examples": report["examples_processed"] == budget["examples_processed"],
            "reload": report["checkpoint_reload_exact"] and report["reload_max_absolute_difference"] == 0,
            "checkpoint_hash": sha256(checkpoint_path) == report["checkpoint_sha256"],
            "manifest_hash": checkpoint_manifest["checkpoint_sha256"] == report["checkpoint_sha256"],
            "manifest_architecture": checkpoint_manifest["architecture"] == expected_architecture,
        }
        if not all(checks.values()):
            raise ValueError(f"Invalid M4.4 bundle {ablation} seed {seed}: {checks}")
        model = MMoE(preprocessor, config, **MMOE_ABLATION_SPECS[ablation]["kwargs"])
        loaded_manifest = load_checkpoint(
            directory, model=model, expected_architecture=expected_architecture,
            expected_preprocessing_sha256=preprocessing_sha256,
            expected_feature_manifest_sha256=feature_manifest_sha256,
            device=torch.device("cpu"),
        )
        if loaded_manifest["production_metadata"]["ablation"] != ablation:
            raise ValueError(f"Loaded checkpoint has wrong ablation: {ablation} seed {seed}")
        entries.append({
            "ablation": ablation,
            "architecture": expected_architecture,
            "seed": seed,
            "input_dimension": dimension,
            "parameter_count": report["parameter_count"],
            "optimizer_updates": report["optimizer_updates"],
            "examples_processed": report["examples_processed"],
            "training_seconds": report["training_seconds"],
            "peak_rss_mib": report["peak_rss_mib"],
            "loss_summary": report["loss_summary"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "checkpoint_bytes": report["checkpoint_bytes"],
            "checkpoint_reload_exact": True,
            "run_report": str(report_path),
        })
if len({entry["checkpoint_sha256"] for entry in entries}) != 9:
    raise ValueError("M4.4 checkpoint hashes are not unique")
manifest = {
    "manifest_version": "m4.4-ablations-v1",
    "protocol_tag": "week0-protocol-v1",
    "feature_tag": "m2-features-v1",
    "training_dates": [20220412, 20220429],
    "training_rows": 722481,
    "common_budget": {
        "optimizer_updates": budget["optimizer_updates"],
        "examples_processed": budget["examples_processed"],
        "batch_size": config["neural"]["batch_size"],
        "drop_last": False,
        "optimizer": config["neural"]["optimizer"],
        "learning_rate": config["neural"]["learning_rate"],
        "weight_decay": config["neural"]["weight_decay"],
        "learning_rate_schedule": config["neural"]["learning_rate_schedule"],
    },
    "feature_manifest_sha256": feature_manifest_sha256,
    "preprocessing_sha256": preprocessing_sha256,
    "bundle_count": len(entries),
    "bundles": entries,
    "validation_or_test_outcomes_accessed": False,
}
root.mkdir(parents=True, exist_ok=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
