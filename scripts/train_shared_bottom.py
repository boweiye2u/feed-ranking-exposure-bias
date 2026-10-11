#!/usr/bin/env python3
"""Train one approved production shared-bottom seed on normal-policy data."""

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
from kuairand_pure.models.neural import SharedBottomMLP
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_losses(logs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in ("loss", "long_view_loss", "like_loss", "valid_play_loss"):
        values = np.asarray([row[name] for row in logs], dtype=np.float64)
        result[name] = {
            "first": float(values[0]),
            "last": float(values[-1]),
            "mean": float(values.mean()),
            "last_100_mean": float(values[-100:].mean()),
        }
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

overall_started = time.perf_counter()
config = load_neural_config()
neural = config["neural"]
budget = config["m4_2_training_budget"]
if budget["status"] != "approved":
    raise ValueError("M4.2 optimizer-update budget is not approved")
if args.seed not in neural["seeds"]:
    raise ValueError(f"Seed {args.seed} is not one of the approved seeds")
expected_settings = {
    "batch_size": 512,
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "learning_rate_schedule": "constant",
    "weight_decay": 0.00001,
}
for name, expected in expected_settings.items():
    if neural[name] != expected:
        raise ValueError(f"Frozen neural setting changed: {name}")
if neural["task_weights"] != [1.0, 1.0, 1.0]:
    raise ValueError("Frozen equal task weights changed")

paths = get_artifact_paths()
source = paths.processed / "fs3" / "v1" / "normal_train"
manifest_path = paths.processed / "m2" / "v1" / "feature_manifest.json"
preprocessor_path = paths.models / "neural" / "infrastructure" / "v1" / "preprocessor.json"
manifest_sha256 = sha256(manifest_path)
preprocessor_sha256 = sha256(preprocessor_path)

load_started = time.perf_counter()
features, labels, sequence = load_aligned_model_data(
    source / "features.parquet",
    source / "labels.parquet",
    sequence_path=source / "sequence.parquet",
    expected_stage="fs3",
    expected_split="normal_train",
)
if sequence is None:
    raise ValueError("Production shared-bottom training requires sequence history")
load_seconds = time.perf_counter() - load_started
if features.height != 722_481 or features["date"].min() != 20220412 or features["date"].max() != 20220429:
    raise ValueError("Normal-training population or dates changed")

preprocessor = NeuralPreprocessor.load(
    preprocessor_path, expected_manifest_sha256=manifest_sha256
)
if preprocessor.include_is_rand:
    raise ValueError("Normal-only model cannot use context__is_rand")
transform_started = time.perf_counter()
arrays = preprocessor.transform(features, sequence)
targets = build_neural_targets(labels)
dataset = NeuralDataset(arrays, targets)
transform_seconds = time.perf_counter() - transform_started

batch_size = int(neural["batch_size"])
updates = int(budget["optimizer_updates"])
expected_examples = int(budget["examples_processed"])
loader = make_dataloader(dataset, batch_size=batch_size, seed=args.seed, shuffle=True)
if loader.drop_last:
    raise ValueError("Production DataLoader must use drop_last=False")
if len(loader) * int(budget["full_training_passes"]) != updates:
    raise ValueError("Approved update budget no longer equals five complete passes")

device = resolve_device(neural["device"])
set_deterministic_seed(args.seed)
model = SharedBottomMLP(preprocessor, config)
training_started = time.perf_counter()
counters, logs, optimizer = train_updates(
    model,
    loader,
    device=device,
    learning_rate=float(neural["learning_rate"]),
    weight_decay=float(neural["weight_decay"]),
    task_weights=[float(value) for value in neural["task_weights"]],
    max_updates=updates,
)
training_seconds = time.perf_counter() - training_started
if counters.optimizer_updates != updates or counters.training_examples != expected_examples:
    raise ValueError(
        f"Budget accounting mismatch: {counters.optimizer_updates} updates, "
        f"{counters.training_examples} examples"
    )

loss_summary = summarize_losses(logs)
versions = {
    "python": sys.version.split()[0],
    "torch": version("torch"),
    "numpy": version("numpy"),
    "polars": version("polars"),
}
valid_mask = labels["click_semantics"] == "valid_play"
task_population = {
    "long_view": {"eligible": labels.height, "positive": int(labels["long_view"].sum())},
    "like": {"eligible": labels.height, "positive": int(labels["is_like"].sum())},
    "valid_play": {
        "eligible": int(valid_mask.sum()),
        "positive": int(labels.filter(valid_mask)["is_click"].sum()),
    },
}

destination = paths.models / "neural" / "shared_bottom" / "v1" / f"seed_{args.seed}"
fixed_loader = make_dataloader(dataset, batch_size=1024, seed=0, shuffle=False)
fixed_batch = move_batch(next(iter(fixed_loader)), device)
model.eval()
with torch.no_grad():
    before = model.probabilities(fixed_batch).cpu()
parameter_count = sum(parameter.numel() for parameter in model.parameters())
metadata = {
    "milestone": "M4.2",
    "production_model": True,
    "training_dates": [20220412, 20220429],
    "training_rows": len(dataset),
    "task_population": task_population,
    "task_mask_policy": {
        "long_view": "all normal-training impressions",
        "like": "all normal-training impressions",
        "valid_play": "click_semantics == valid_play (tab 1) only",
    },
    "optimizer_updates": counters.optimizer_updates,
    "examples_processed": counters.training_examples,
    "full_training_passes": int(budget["full_training_passes"]),
    "drop_last": False,
    "training_seconds": training_seconds,
    "parameter_count": parameter_count,
    "loss_summary": loss_summary,
    "software_versions": versions,
    "learning_rate_schedule": neural["learning_rate_schedule"],
    "validation_early_stopping": False,
    "validation_or_test_outcomes_accessed": False,
}
checkpoint_manifest = save_checkpoint(
    destination,
    model=model,
    optimizer=optimizer,
    architecture="shared_bottom",
    preprocessing_sha256=preprocessor_sha256,
    feature_manifest_sha256=manifest_sha256,
    seed=args.seed,
    counters=counters,
    config=config,
    extra_metadata=metadata,
)
(destination / "training_log.jsonl").write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in logs)
)

reloaded = SharedBottomMLP(preprocessor, config)
load_checkpoint(
    destination,
    model=reloaded,
    expected_architecture="shared_bottom",
    expected_preprocessing_sha256=preprocessor_sha256,
    expected_feature_manifest_sha256=manifest_sha256,
    device=device,
)
reloaded.eval()
with torch.no_grad():
    after = reloaded.probabilities(fixed_batch).cpu()
exact_reload = torch.equal(before, after)
max_reload_difference = float(torch.max(torch.abs(before - after)))
if not exact_reload or max_reload_difference != 0.0:
    raise ValueError("Production checkpoint predictions differ after exact reload")
if not torch.isfinite(after).all() or not ((after >= 0) & (after <= 1)).all():
    raise ValueError("Production shared-bottom probabilities are invalid")

report = {
    **metadata,
    "architecture": "shared_bottom",
    "seed": args.seed,
    "device": str(device),
    "data_load_seconds": load_seconds,
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
    "probability_min": float(after.min()),
    "probability_max": float(after.max()),
}
(destination / "run_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
