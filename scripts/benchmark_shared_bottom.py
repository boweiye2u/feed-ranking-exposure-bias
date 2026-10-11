#!/usr/bin/env python3
"""Benchmark the full shared-bottom training pipeline without saving a model."""

import hashlib
import json
import math
import resource
import subprocess
import time
from pathlib import Path

import torch

from kuairand_pure.data.alignment import load_aligned_model_data
from kuairand_pure.models.neural import SharedBottomMLP
from kuairand_pure.paths import get_artifact_paths
from kuairand_pure.training.neural import (
    masked_multitask_bce,
    move_batch,
    resolve_device,
    set_deterministic_seed,
)
from kuairand_pure.training.neural_data import (
    NeuralDataset,
    NeuralPreprocessor,
    build_neural_targets,
    load_neural_config,
    make_dataloader,
)


def gpu_utility_status() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False, capture_output=True, text=True, timeout=10,
        )
        return {
            "available": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}


def available_memory_mib() -> float | None:
    status = Path("/proc/meminfo")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return None


overall_started = time.perf_counter()
paths = get_artifact_paths()
config = load_neural_config()
benchmark_config = config["m4_2_benchmark"]
device = resolve_device(config["neural"]["device"])
gpu_status = gpu_utility_status()
memory_before = available_memory_mib()

load_started = time.perf_counter()
source = paths.processed / "fs3" / "v1" / "normal_train"
features, labels, sequence = load_aligned_model_data(
    source / "features.parquet", source / "labels.parquet",
    sequence_path=source / "sequence.parquet",
    expected_stage="fs3", expected_split="normal_train",
)
if sequence is None:
    raise ValueError("Shared-bottom benchmark requires sequence history")
load_seconds = time.perf_counter() - load_started

preprocessor_path = paths.models / "neural" / "infrastructure" / "v1" / "preprocessor.json"
manifest_path = paths.processed / "m2" / "v1" / "feature_manifest.json"
manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
preprocessor = NeuralPreprocessor.load(
    preprocessor_path, expected_manifest_sha256=manifest_sha256
)
transform_started = time.perf_counter()
arrays = preprocessor.transform(features, sequence)
targets = build_neural_targets(labels)
dataset = NeuralDataset(arrays, targets)
transform_seconds = time.perf_counter() - transform_started

seed = int(benchmark_config["temporary_seed"])
set_deterministic_seed(seed)
model = SharedBottomMLP(preprocessor, config).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=float(config["neural"]["learning_rate"]),
    weight_decay=float(config["neural"]["weight_decay"]),
)
weights = torch.tensor(config["neural"]["task_weights"], dtype=torch.float32, device=device)
loader = make_dataloader(
    dataset, batch_size=int(config["neural"]["batch_size"]), seed=seed, shuffle=True
)
iterator = iter(loader)


def one_update() -> tuple[int, float, dict[str, float]]:
    global iterator
    try:
        raw = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        raw = next(iterator)
    batch = move_batch(raw, device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss, task_losses = masked_multitask_bce(
        logits, batch["targets"], batch["task_masks"], weights
    )
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        int(batch["targets"].shape[0]), float(loss.detach().cpu()),
        {task: float(value.detach().cpu()) for task, value in task_losses.items()},
    )


model.train()
warmup_updates = int(benchmark_config["warmup_updates"])
for _ in range(warmup_updates):
    one_update()
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
measured_updates = int(benchmark_config["measured_updates"])
measured_examples = 0
losses = []
task_losses = {task: [] for task in ("long_view", "like", "valid_play")}
measured_started = time.perf_counter()
for _ in range(measured_updates):
    examples, loss, tasks = one_update()
    measured_examples += examples
    losses.append(loss)
    for task, value in tasks.items():
        task_losses[task].append(value)
measured_seconds = time.perf_counter() - measured_started

updates_per_full_pass = math.ceil(len(dataset) / int(config["neural"]["batch_size"]))
seconds_per_update = measured_seconds / measured_updates
report = {
    "milestone": "M4.2_resource_benchmark",
    "status": "benchmark_complete_budget_not_authorized",
    "temporary_run": True,
    "temporary_seed": seed,
    "architecture": "shared_bottom",
    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    "training_rows": len(dataset),
    "batch_size": int(config["neural"]["batch_size"]),
    "data_load_seconds": load_seconds,
    "full_preprocessing_seconds": transform_seconds,
    "warmup_updates": warmup_updates,
    "measured_updates": measured_updates,
    "measured_examples": measured_examples,
    "measured_seconds": measured_seconds,
    "updates_per_second": measured_updates / measured_seconds,
    "examples_per_second": measured_examples / measured_seconds,
    "seconds_per_update": seconds_per_update,
    "updates_per_full_training_pass": updates_per_full_pass,
    "estimated_seconds_per_full_training_pass": updates_per_full_pass * seconds_per_update,
    "maximum_projected_seed_seconds": float(
        benchmark_config["maximum_projected_seed_seconds"]
    ),
    "loss_diagnostics": {
        "first_overall": losses[0], "last_overall": losses[-1],
        "mean_overall": sum(losses) / len(losses),
        **{f"mean_{task}": sum(values) / len(values) for task, values in task_losses.items()},
    },
    "device": str(device),
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_device_count": torch.cuda.device_count(),
    "nvidia_smi": gpu_status,
    "cuda_peak_allocated_mib": (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else None
    ),
    "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "system_available_memory_mib_before": memory_before,
    "system_available_memory_mib_after": available_memory_mib(),
    "overall_runtime_seconds": time.perf_counter() - overall_started,
    "optimizer": "adamw",
    "learning_rate": float(config["neural"]["learning_rate"]),
    "learning_rate_schedule": config["neural"]["learning_rate_schedule"],
    "weight_decay": float(config["neural"]["weight_decay"]),
    "preprocessing_sha256": hashlib.sha256(preprocessor_path.read_bytes()).hexdigest(),
    "feature_manifest_sha256": manifest_sha256,
    "validation_or_test_outcomes_accessed": False,
    "checkpoint_saved": False,
}
output = paths.models / "neural" / "benchmarks" / "m4_2"
output.mkdir(parents=True, exist_ok=True)
(output / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
