#!/usr/bin/env python3
"""Fit the normal-training neural adapter and run tiny deterministic smoke updates."""

import hashlib
import json
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import polars as pl
import torch

from kuairand_pure.data.alignment import load_aligned_model_data
from kuairand_pure.models.neural import MMoE, SharedBottomMLP
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


started = time.perf_counter()
paths = get_artifact_paths()
config = load_neural_config()
source = paths.processed / "fs3" / "v1" / "normal_train"
feature_manifest_path = paths.processed / "m2" / "v1" / "feature_manifest.json"
features, labels, sequence = load_aligned_model_data(
    source / "features.parquet", source / "labels.parquet",
    sequence_path=source / "sequence.parquet",
    expected_stage="fs3", expected_split="normal_train",
)
assert sequence is not None
fit_started = time.perf_counter()
preprocessor = NeuralPreprocessor.fit(
    features, feature_manifest_path=feature_manifest_path, include_is_rand=False
)
fit_seconds = time.perf_counter() - fit_started
output_root = paths.models / "neural" / "infrastructure" / "v1"
preprocessor_path = output_root / "preprocessor.json"
preprocessor.save(preprocessor_path)
preprocessor_sha256 = sha256(preprocessor_path)
feature_manifest_sha256 = sha256(feature_manifest_path)
reloaded_preprocessor = NeuralPreprocessor.load(
    preprocessor_path, expected_manifest_sha256=feature_manifest_sha256
)
if reloaded_preprocessor.to_dict() != preprocessor.to_dict():
    raise ValueError("Reloaded neural preprocessor differs")

rows = int(config["smoke"]["rows"])
sample_features, sample_labels, sample_sequence = (
    features.head(rows), labels.head(rows), sequence.head(rows)
)
arrays = preprocessor.transform(sample_features, sample_sequence)
targets = build_neural_targets(sample_labels)
dataset = NeuralDataset(arrays, targets)
seed = int(config["smoke"]["seed"])
device = resolve_device(config["neural"]["device"])
architecture_reports = {}
constructors = {"shared_bottom": SharedBottomMLP, "mmoe": MMoE}
for architecture, constructor in constructors.items():
    set_deterministic_seed(seed)
    model = constructor(preprocessor, config)
    loader = make_dataloader(
        dataset, batch_size=int(config["neural"]["batch_size"]), seed=seed, shuffle=True
    )
    counters, logs, optimizer = train_updates(
        model, loader, device=device,
        learning_rate=float(config["neural"]["learning_rate"]),
        weight_decay=float(config["neural"]["weight_decay"]),
        task_weights=[float(value) for value in config["neural"]["task_weights"]],
        max_updates=int(config["smoke"]["updates"]),
    )
    fixed_batch = move_batch(next(iter(make_dataloader(
        dataset, batch_size=32, seed=seed, shuffle=False
    ))), device)
    model.eval()
    with torch.no_grad():
        before = model.probabilities(fixed_batch).cpu()
    destination = output_root / architecture / "smoke"
    checkpoint_manifest = save_checkpoint(
        destination, model=model, optimizer=optimizer, architecture=architecture,
        preprocessing_sha256=preprocessor_sha256,
        feature_manifest_sha256=feature_manifest_sha256, seed=seed,
        counters=counters, config=config,
    )
    (destination / "training_log.json").write_text(
        json.dumps(logs, indent=2, sort_keys=True) + "\n"
    )
    reloaded = constructor(preprocessor, config)
    load_checkpoint(
        destination, model=reloaded, expected_architecture=architecture,
        expected_preprocessing_sha256=preprocessor_sha256,
        expected_feature_manifest_sha256=feature_manifest_sha256, device=device,
    )
    reloaded.eval()
    with torch.no_grad():
        after = reloaded.probabilities(fixed_batch).cpu()
    if not torch.equal(before, after):
        raise ValueError(f"{architecture} checkpoint predictions differ after reload")
    if not torch.isfinite(after).all() or not ((after >= 0) & (after <= 1)).all():
        raise ValueError(f"{architecture} returned invalid probabilities")
    architecture_reports[architecture] = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_examples": counters.training_examples,
        "optimizer_updates": counters.optimizer_updates,
        "first_loss": logs[0]["loss"],
        "last_loss": logs[-1]["loss"],
        "checkpoint_reload_exact": True,
        "probability_min": float(after.min()),
        "probability_max": float(after.max()),
        "checkpoint_manifest": checkpoint_manifest,
    }

valid_mask = labels["click_semantics"] == "valid_play"
literal_mask = labels["click_semantics"] == "literal_click"
population = {
    "long_view": {
        "rows": labels.height,
        "positives": int(labels["long_view"].sum()),
    },
    "like": {"rows": labels.height, "positives": int(labels["is_like"].sum())},
    "valid_play": {
        "rows": int(valid_mask.sum()),
        "positives": int(labels.filter(valid_mask)["is_click"].sum()),
    },
    "literal_click_supplementary_excluded": {
        "rows": int(literal_mask.sum()),
        "positives": int(labels.filter(literal_mask)["is_click"].sum()),
    },
}
for values in population.values():
    values["prevalence"] = values["positives"] / values["rows"]

report = {
    "milestone": "M4.1",
    "normal_training_rows": features.height,
    "normal_training_dates": [20220412, 20220429],
    "adapter_fit_seconds": fit_seconds,
    "smoke_rows": rows,
    "numerical_input_dimension": len(preprocessor.numerical_output_names),
    "raw_numerical_feature_count": len(preprocessor.numerical_features),
    "numeric_missing_indicator_count": len(preprocessor.missing_indicator_features),
    "categorical_feature_count": len(preprocessor.categorical_features),
    "vocabulary_sizes": preprocessor.vocabulary_sizes,
    "task_population": population,
    "task_weights": config["neural"]["task_weights"],
    "history": preprocessor.to_dict()["history"],
    "architectures": architecture_reports,
    "device": str(device),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "runtime_seconds": time.perf_counter() - started,
    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "preprocessor_path": str(preprocessor_path),
    "preprocessor_sha256": preprocessor_sha256,
    "feature_manifest_sha256": feature_manifest_sha256,
    "software_versions": {
        "python": sys.version.split()[0], "torch": version("torch"),
        "numpy": version("numpy"), "polars": version("polars"),
    },
    "validation_or_test_outcomes_accessed": False,
    "production_models_trained": False,
}
(output_root / "smoke_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
