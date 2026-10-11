#!/usr/bin/env python3
"""Train one approved production MMoE seed on normal-policy data."""

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
from kuairand_pure.models.neural import MMoE
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
    TASK_NAMES,
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
            "first": float(values[0]), "last": float(values[-1]),
            "mean": float(values.mean()), "last_100_mean": float(values[-100:].mean()),
        }
    return result


def gate_diagnostics(
    model: MMoE, dataset: NeuralDataset, device: torch.device
) -> dict[str, dict[str, object]]:
    loader = make_dataloader(dataset, batch_size=4096, seed=0, shuffle=False)
    experts = len(model.experts)
    accumulator = {
        task: {
            "count": 0, "sum": torch.zeros(experts, dtype=torch.float64),
            "sum_squares": torch.zeros(experts, dtype=torch.float64),
            "argmax": torch.zeros(experts, dtype=torch.int64),
            "entropy_sum": 0.0, "minimum": 1.0, "maximum": 0.0,
        }
        for task in TASK_NAMES
    }
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            routing = model.gate_weights(batch)
            for task, device_weights in routing.items():
                if not torch.isfinite(device_weights).all():
                    raise ValueError("MMoE produced non-finite gate weights")
                if torch.any(device_weights < 0) or not torch.allclose(
                    device_weights.sum(dim=1),
                    torch.ones(device_weights.shape[0], device=device),
                    atol=1e-6,
                ):
                    raise ValueError("MMoE gate weights are not normalized")
                weights = device_weights.cpu().to(torch.float64)
                values = accumulator[task]
                values["count"] += weights.shape[0]
                values["sum"] += weights.sum(dim=0)
                values["sum_squares"] += weights.square().sum(dim=0)
                values["argmax"] += torch.bincount(
                    weights.argmax(dim=1), minlength=experts
                )
                values["entropy_sum"] += float(
                    (-(weights * weights.clamp_min(1e-15).log()).sum(dim=1)).sum()
                )
                values["minimum"] = min(values["minimum"], float(weights.min()))
                values["maximum"] = max(values["maximum"], float(weights.max()))
    result: dict[str, dict[str, object]] = {}
    for task, values in accumulator.items():
        count = int(values["count"])
        means = values["sum"] / count
        variances = (values["sum_squares"] / count - means.square()).clamp_min(0)
        result[task] = {
            "rows": count,
            "mean_weight_by_expert": means.tolist(),
            "std_weight_by_expert": variances.sqrt().tolist(),
            "argmax_fraction_by_expert": (values["argmax"] / count).tolist(),
            "mean_entropy": values["entropy_sum"] / count,
            "minimum_weight": values["minimum"],
            "maximum_weight": values["maximum"],
        }
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

overall_started = time.perf_counter()
config = load_neural_config()
neural = config["neural"]
budget = config["m4_2_training_budget"]
architecture = config["mmoe"]
if budget["status"] != "approved" or args.seed not in neural["seeds"]:
    raise ValueError("MMoE seed or training budget is not approved")
expected_settings = {
    "batch_size": 512, "optimizer": "adamw", "learning_rate": 0.001,
    "learning_rate_schedule": "constant", "weight_decay": 0.00001,
}
for name, expected in expected_settings.items():
    if neural[name] != expected:
        raise ValueError(f"Frozen neural setting changed: {name}")
if neural["task_weights"] != [1.0, 1.0, 1.0]:
    raise ValueError("Frozen equal task weights changed")
if architecture != {
    "num_experts": 4, "expert_dimension": 64,
    "tower_dimension": 32, "dropout": 0.1,
}:
    raise ValueError("Frozen MMoE architecture changed")

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
    raise ValueError("Production MMoE training requires sequence history")
load_seconds = time.perf_counter() - load_started
if features.height != 722_481 or features["date"].min() != 20220412 or features["date"].max() != 20220429:
    raise ValueError("Normal-training population or dates changed")
preprocessor = NeuralPreprocessor.load(
    preprocessor_path, expected_manifest_sha256=manifest_sha256
)
if preprocessor.include_is_rand:
    raise ValueError("Normal-only MMoE cannot use context__is_rand")
transform_started = time.perf_counter()
arrays = preprocessor.transform(features, sequence)
targets = build_neural_targets(labels)
dataset = NeuralDataset(arrays, targets)
transform_seconds = time.perf_counter() - transform_started

batch_size = int(neural["batch_size"])
updates = int(budget["optimizer_updates"])
expected_examples = int(budget["examples_processed"])
loader = make_dataloader(dataset, batch_size=batch_size, seed=args.seed, shuffle=True)
if loader.drop_last or len(loader) * int(budget["full_training_passes"]) != updates:
    raise ValueError("MMoE loader differs from the approved five-pass budget")
device = resolve_device(neural["device"])
set_deterministic_seed(args.seed)
model = MMoE(preprocessor, config)
if model.encoder.output_dimension != 286 or len(model.experts) != 4 or len(model.gates) != 3 or len(model.towers) != 3:
    raise ValueError("Production MMoE architecture dimensions changed")
parameter_count = sum(parameter.numel() for parameter in model.parameters())
if parameter_count != 687_479:
    raise ValueError(f"Production MMoE parameter count changed: {parameter_count}")

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
    raise ValueError("MMoE optimizer-update or example accounting differs from approval")

gradient_norms = {}
for family, module in (("experts", model.experts), ("gates", model.gates), ("towers", model.towers)):
    gradients = [parameter.grad for parameter in module.parameters()]
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError(f"MMoE {family} gradients are absent or non-finite")
    gradient_norms[family] = float(torch.sqrt(sum(gradient.square().sum() for gradient in gradients)))
    if gradient_norms[family] == 0:
        raise ValueError(f"MMoE {family} received zero gradient")

diagnostic_started = time.perf_counter()
routing = gate_diagnostics(model, dataset, device)
diagnostic_seconds = time.perf_counter() - diagnostic_started
loss_summary = summarize_losses(logs)
versions = {
    "python": sys.version.split()[0], "torch": version("torch"),
    "numpy": version("numpy"), "polars": version("polars"),
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

destination = paths.models / "neural" / "mmoe" / "v1" / f"seed_{args.seed}"
fixed_batch = move_batch(next(iter(make_dataloader(
    dataset, batch_size=1024, seed=0, shuffle=False
))), device)
model.eval()
with torch.no_grad():
    before = model.probabilities(fixed_batch).cpu()
if not torch.isfinite(before).all() or not ((before >= 0) & (before <= 1)).all():
    raise ValueError("Production MMoE probabilities are invalid")
metadata = {
    "milestone": "M4.3", "production_model": True,
    "training_dates": [20220412, 20220429], "training_rows": len(dataset),
    "task_population": task_population,
    "task_mask_policy": {
        "long_view": "all normal-training impressions",
        "like": "all normal-training impressions",
        "valid_play": "click_semantics == valid_play (tab 1) only",
    },
    "architecture_configuration": architecture,
    "input_dimension": model.encoder.output_dimension,
    "optimizer_updates": counters.optimizer_updates,
    "examples_processed": counters.training_examples,
    "full_training_passes": int(budget["full_training_passes"]),
    "drop_last": False, "training_seconds": training_seconds,
    "parameter_count": parameter_count, "loss_summary": loss_summary,
    "gate_diagnostics": routing, "gradient_norms_after_final_update": gradient_norms,
    "software_versions": versions,
    "learning_rate_schedule": neural["learning_rate_schedule"],
    "class_weighting": False, "additional_training_objectives": False,
    "validation_early_stopping": False,
    "nonfinite_losses_outputs_or_gradients": False,
    "validation_or_test_outcomes_accessed": False,
}
checkpoint_manifest = save_checkpoint(
    destination, model=model, optimizer=optimizer, architecture="mmoe",
    preprocessing_sha256=preprocessor_sha256,
    feature_manifest_sha256=manifest_sha256, seed=args.seed,
    counters=counters, config=config, extra_metadata=metadata,
)
(destination / "training_log.jsonl").write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in logs)
)

reloaded = MMoE(preprocessor, config)
load_checkpoint(
    destination, model=reloaded, expected_architecture="mmoe",
    expected_preprocessing_sha256=preprocessor_sha256,
    expected_feature_manifest_sha256=manifest_sha256, device=device,
)
reloaded.eval()
with torch.no_grad():
    after = reloaded.probabilities(fixed_batch).cpu()
exact_reload = torch.equal(before, after)
max_reload_difference = float(torch.max(torch.abs(before - after)))
if not exact_reload or max_reload_difference != 0.0:
    raise ValueError("Production MMoE predictions differ after exact reload")

report = {
    **metadata, "architecture": "mmoe", "seed": args.seed,
    "device": str(device), "data_load_seconds": load_seconds,
    "preprocessing_seconds": transform_seconds,
    "gate_diagnostic_seconds": diagnostic_seconds,
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
