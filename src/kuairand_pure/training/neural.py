"""Deterministic masked multi-task neural training and checkpoint utilities."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from kuairand_pure.data.alignment import M2_BUILD_VERSION
from kuairand_pure.training.neural_data import TASK_NAMES


@dataclass(slots=True)
class TrainingCounters:
    training_examples: int = 0
    optimizer_updates: int = 0


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown neural device: {requested}")
    return torch.device(requested)


def masked_multitask_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    task_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.shape != targets.shape or targets.shape != masks.shape:
        raise ValueError("Logits, targets, and masks must have equal shapes")
    if logits.ndim != 2 or logits.shape[1] != len(TASK_NAMES):
        raise ValueError("Neural outputs must have exactly three task columns")
    elementwise = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    task_losses: dict[str, torch.Tensor] = {}
    weighted: list[torch.Tensor] = []
    active_weights: list[torch.Tensor] = []
    for index, task in enumerate(TASK_NAMES):
        eligible = masks[:, index].sum()
        if eligible.item() == 0:
            task_losses[task] = logits[:, index].sum() * 0
            continue
        task_loss = (elementwise[:, index] * masks[:, index]).sum() / eligible
        task_losses[task] = task_loss
        weighted.append(task_loss * task_weights[index])
        active_weights.append(task_weights[index])
    if not weighted:
        raise ValueError("Batch has no eligible neural targets")
    return torch.stack(weighted).sum() / torch.stack(active_weights).sum(), task_losses


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def train_updates(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    task_weights: list[float],
    max_updates: int,
) -> tuple[TrainingCounters, list[dict[str, float]], torch.optim.Optimizer]:
    if max_updates <= 0:
        raise ValueError("max_updates must be positive")
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    weights = torch.tensor(task_weights, dtype=torch.float32, device=device)
    counters = TrainingCounters()
    logs: list[dict[str, float]] = []
    iterator = iter(loader)
    started = time.perf_counter()
    while counters.optimizer_updates < max_updates:
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        if not torch.isfinite(logits).all():
            raise ValueError("Neural training produced non-finite outputs")
        loss, task_losses = masked_multitask_bce(
            logits, batch["targets"], batch["task_masks"], weights
        )
        if not torch.isfinite(loss):
            raise ValueError("Neural training produced a non-finite loss")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ValueError("Neural training produced non-finite gradients")
        optimizer.step()
        counters.training_examples += int(batch["targets"].shape[0])
        counters.optimizer_updates += 1
        logs.append({
            "optimizer_update": float(counters.optimizer_updates),
            "training_examples": float(counters.training_examples),
            "elapsed_seconds": time.perf_counter() - started,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "batch_examples": float(batch["targets"].shape[0]),
            "loss": float(loss.detach().cpu()),
            **{f"{task}_loss": float(value.detach().cpu()) for task, value in task_losses.items()},
        })
    return counters, logs, optimizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    directory: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    architecture: str,
    preprocessing_sha256: str,
    feature_manifest_sha256: str,
    seed: int,
    counters: TrainingCounters,
    config: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "architecture": architecture,
            "seed": seed,
            "counters": asdict(counters),
        },
        checkpoint_path,
    )
    manifest = {
        "checkpoint_version": "neural-infra-v1",
        "input_build_version": M2_BUILD_VERSION,
        "architecture": architecture,
        "seed": seed,
        "preprocessing_sha256": preprocessing_sha256,
        "feature_manifest_sha256": feature_manifest_sha256,
        "protocol_identifiers": {
            "study": "Feed Ranking under Exposure Bias v5",
            "week0_tag": "week0-protocol-v1",
            "feature_tag": "m2-features-v1",
        },
        "training_counters": asdict(counters),
        "optimizer": "adamw",
        "task_weights": config["neural"]["task_weights"],
        "training_configuration": {
            "neural": config["neural"],
            "embeddings": config["embeddings"],
            "shared_bottom": config["shared_bottom"],
            "mmoe": config["mmoe"],
        },
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    if extra_metadata is not None:
        manifest["production_metadata"] = extra_metadata
    (directory / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_checkpoint(
    directory: Path,
    *,
    model: nn.Module,
    expected_architecture: str,
    expected_preprocessing_sha256: str,
    expected_feature_manifest_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = directory / "checkpoint.pt"
    manifest_path = directory / "checkpoint_manifest.json"
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise ValueError("Incomplete neural checkpoint bundle")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("checkpoint_version") != "neural-infra-v1" or manifest.get("input_build_version") != M2_BUILD_VERSION:
        raise ValueError("Neural checkpoint version is stale")
    expected = {
        "architecture": expected_architecture,
        "preprocessing_sha256": expected_preprocessing_sha256,
        "feature_manifest_sha256": expected_feature_manifest_sha256,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Neural checkpoint {field} is incompatible")
    if _sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("Neural checkpoint checksum mismatch")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return manifest
