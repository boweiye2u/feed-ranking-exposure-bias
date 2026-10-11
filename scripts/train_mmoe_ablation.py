#!/usr/bin/env python3
"""Train one frozen production MMoE ablation and seed."""

import argparse
import hashlib
import json
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kuairand_pure.data.alignment import load_aligned_model_data
from kuairand_pure.models.neural import MMOE_ABLATION_SPECS, MMoE
from kuairand_pure.paths import get_artifact_paths
from kuairand_pure.training.neural import (
    load_checkpoint,
    move_batch,
    resolve_device,
    save_checkpoint,
    set_deterministic_seed,
    train_updates,
)
from kuairand_pure.training.neural_data import (
    NeuralDataset,
    NeuralPreprocessor,
    build_neural_targets,
    load_neural_config,
    make_dataloader,
)


INPUT_GROUPS = {
    "dense_only": {
        "active_input_groups": ["fs3_dense_numerical", "fs3_dense_categorical"],
        "excluded_input_groups": ["user_id_embedding", "target_video_embedding", "pooled_history_embedding"],
    },
    "no_history": {
        "active_input_groups": ["fs3_dense", "user_id_embedding", "target_video_embedding"],
        "excluded_input_groups": ["pooled_history_embedding"],
    },
    "no_aggregates": {
        "active_input_groups": ["fs3_nonaggregate_dense", "user_id_embedding", "target_video_embedding", "pooled_history_embedding"],
        "excluded_input_groups": ["dense_fields_with___hist_"],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_losses(logs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for name in ("loss", "long_view_loss", "like_loss", "valid_play_loss"):
        values = np.asarray([row[name] for row in logs], dtype=np.float64)
        result[name] = {
            "first": float(values[0]), "last": float(values[-1]),
            "mean": float(values.mean()), "last_100_mean": float(values[-100:].mean()),
        }
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--ablation", choices=sorted(MMOE_ABLATION_SPECS), required=True)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()
spec = {**MMOE_ABLATION_SPECS[args.ablation], **INPUT_GROUPS[args.ablation]}
overall_started = time.perf_counter()

config = load_neural_config()
neural = config["neural"]
budget = config["m4_2_training_budget"]
if budget["status"] != "approved" or args.seed not in neural["seeds"]:
    raise ValueError("Ablation seed or common neural budget is not approved")
expected_settings = {
    "batch_size": 512, "optimizer": "adamw", "learning_rate": 0.001,
    "learning_rate_schedule": "constant", "weight_decay": 0.00001,
}
for name, expected in expected_settings.items():
    if neural[name] != expected:
        raise ValueError(f"Frozen neural setting changed: {name}")
if neural["task_weights"] != [1.0, 1.0, 1.0] or config["mmoe"] != {
    "num_experts": 4, "expert_dimension": 64, "tower_dimension": 32, "dropout": 0.1,
}:
    raise ValueError("Frozen task loss or MMoE architecture changed")

paths = get_artifact_paths()
source = paths.processed / "fs3" / "v1" / "normal_train"
manifest_path = paths.processed / "m2" / "v1" / "feature_manifest.json"
preprocessor_path = paths.models / "neural" / "infrastructure" / "v1" / "preprocessor.json"
manifest_sha256 = sha256(manifest_path)
preprocessor_sha256 = sha256(preprocessor_path)
load_started = time.perf_counter()
features, labels, sequence = load_aligned_model_data(
    source / "features.parquet", source / "labels.parquet",
    sequence_path=source / "sequence.parquet",
    expected_stage="fs3", expected_split="normal_train",
)
if sequence is None:
    raise ValueError("Ablation training requires aligned sequence validation")
load_seconds = time.perf_counter() - load_started
if features.height != 722_481 or features["date"].min() != 20220412 or features["date"].max() != 20220429:
    raise ValueError("Normal-training population or dates changed")
preprocessor = NeuralPreprocessor.load(
    preprocessor_path, expected_manifest_sha256=manifest_sha256
)
if preprocessor.include_is_rand:
    raise ValueError("Normal-only ablations cannot use context__is_rand")
transform_started = time.perf_counter()
arrays = preprocessor.transform(features, sequence)
targets = build_neural_targets(labels)
dataset = NeuralDataset(arrays, targets)
transform_seconds = time.perf_counter() - transform_started

loader = make_dataloader(
    dataset, batch_size=int(neural["batch_size"]), seed=args.seed, shuffle=True
)
updates = int(budget["optimizer_updates"])
expected_examples = int(budget["examples_processed"])
if loader.drop_last or len(loader) * int(budget["full_training_passes"]) != updates:
    raise ValueError("Ablation loader differs from the approved five-pass budget")
device = resolve_device(neural["device"])
set_deterministic_seed(args.seed)
model = MMoE(preprocessor, config, **spec["kwargs"])
if model.encoder.output_dimension != spec["input_dimension"]:
    raise ValueError(
        f"{args.ablation} input width changed: {model.encoder.output_dimension}"
    )
parameter_count = sum(parameter.numel() for parameter in model.parameters())

training_started = time.perf_counter()
counters, logs, optimizer = train_updates(
    model, loader, device=device,
    learning_rate=float(neural["learning_rate"]),
    weight_decay=float(neural["weight_decay"]),
    task_weights=[float(value) for value in neural["task_weights"]],
    max_updates=updates,
)
training_seconds = time.perf_counter() - training_started
if counters.optimizer_updates != updates or counters.training_examples != expected_examples:
    raise ValueError("Ablation update/example accounting differs from approval")
gradients = [parameter.grad for parameter in model.parameters()]
if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
    raise ValueError("Ablation contains absent or non-finite active-parameter gradients")
gradient_norm = float(torch.sqrt(sum(gradient.square().sum() for gradient in gradients)))
if gradient_norm == 0:
    raise ValueError("Ablation received zero total gradient")

valid_mask = labels["click_semantics"] == "valid_play"
task_population = {
    "long_view": {"eligible": labels.height, "positive": int(labels["long_view"].sum())},
    "like": {"eligible": labels.height, "positive": int(labels["is_like"].sum())},
    "valid_play": {
        "eligible": int(valid_mask.sum()),
        "positive": int(labels.filter(valid_mask)["is_click"].sum()),
    },
}
loss_summary = summarize_losses(logs)
versions = {
    "python": sys.version.split()[0], "torch": version("torch"),
    "numpy": version("numpy"), "polars": version("polars"),
}
metadata = {
    "milestone": "M4.4", "production_model": True,
    "ablation": args.ablation, "architecture_family": "mmoe",
    "architecture_configuration": config["mmoe"],
    "selector_kwargs": spec["kwargs"],
    "input_dimension": model.encoder.output_dimension,
    "active_input_groups": spec["active_input_groups"],
    "excluded_input_groups": spec["excluded_input_groups"],
    "training_dates": [20220412, 20220429], "training_rows": len(dataset),
    "task_population": task_population,
    "task_mask_policy": {
        "long_view": "all normal-training impressions",
        "like": "all normal-training impressions",
        "valid_play": "click_semantics == valid_play (tab 1) only",
    },
    "optimizer_updates": counters.optimizer_updates,
    "examples_processed": counters.training_examples,
    "full_training_passes": int(budget["full_training_passes"]),
    "drop_last": False, "training_seconds": training_seconds,
    "parameter_count": parameter_count, "loss_summary": loss_summary,
    "gradient_norm_after_final_update": gradient_norm,
    "software_versions": versions,
    "learning_rate_schedule": neural["learning_rate_schedule"],
    "class_weighting": False, "additional_training_objectives": False,
    "validation_early_stopping": False,
    "nonfinite_losses_outputs_or_gradients": False,
    "validation_or_test_outcomes_accessed": False,
}

destination = (
    paths.models / "neural" / "ablations" / "v1" /
    args.ablation / f"seed_{args.seed}"
)
fixed_batch = move_batch(next(iter(make_dataloader(
    dataset, batch_size=1024, seed=0, shuffle=False
))), device)
model.eval()
with torch.no_grad():
    before = model.probabilities(fixed_batch).cpu()
if not torch.isfinite(before).all() or not ((before >= 0) & (before <= 1)).all():
    raise ValueError("Ablation probabilities are invalid")
architecture_name = f"mmoe_{args.ablation}"
checkpoint_manifest = save_checkpoint(
    destination, model=model, optimizer=optimizer, architecture=architecture_name,
    preprocessing_sha256=preprocessor_sha256,
    feature_manifest_sha256=manifest_sha256, seed=args.seed,
    counters=counters, config=config, extra_metadata=metadata,
)
(destination / "training_log.jsonl").write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in logs)
)
reloaded = MMoE(preprocessor, config, **spec["kwargs"])
load_checkpoint(
    destination, model=reloaded, expected_architecture=architecture_name,
    expected_preprocessing_sha256=preprocessor_sha256,
    expected_feature_manifest_sha256=manifest_sha256, device=device,
)
reloaded.eval()
with torch.no_grad():
    after = reloaded.probabilities(fixed_batch).cpu()
exact_reload = torch.equal(before, after)
max_reload_difference = float(torch.max(torch.abs(before - after)))
if not exact_reload or max_reload_difference != 0.0:
    raise ValueError("Ablation predictions differ after exact reload")

report = {
    **metadata, "architecture": architecture_name, "seed": args.seed,
    "device": str(device), "data_load_seconds": load_seconds,
    "preprocessing_seconds": transform_seconds,
    "overall_runtime_seconds": time.perf_counter() - overall_started,
    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "feature_manifest_sha256": manifest_sha256,
    "preprocessing_sha256": preprocessor_sha256,
    "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
    "checkpoint_bytes": (destination / "checkpoint.pt").stat().st_size,
    "checkpoint_reload_exact": exact_reload,
    "reload_max_absolute_difference": max_reload_difference,
    "comparison_impression_ids_sha256": hashlib.sha256(
        np.ascontiguousarray(arrays.impression_ids[:1024]).tobytes()
    ).hexdigest(),
    "comparison_rows": 1024,
    "probability_min": float(after.min()), "probability_max": float(after.max()),
}
(destination / "run_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
