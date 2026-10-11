"""Shared-bottom and MMoE multi-task models with pooled video history."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kuairand_pure.training.neural_data import NeuralPreprocessor, TASK_NAMES


MMOE_ABLATION_SPECS = {
    "dense_only": {
        "kwargs": {"use_ids": False, "use_history": False, "use_aggregates": True},
        "input_dimension": 238,
    },
    "no_history": {
        "kwargs": {"use_ids": True, "use_history": False, "use_aggregates": True},
        "input_dimension": 270,
    },
    "no_aggregates": {
        "kwargs": {"use_ids": True, "use_history": True, "use_aggregates": False},
        "input_dimension": 220,
    },
}


def masked_mean_history(
    embeddings: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool valid history positions; empty histories map to zero."""
    weights = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
    numerator = (embeddings * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


class NeuralInputEncoder(nn.Module):
    def __init__(
        self,
        preprocessor: NeuralPreprocessor,
        config: dict[str, Any],
        *,
        use_ids: bool = True,
        use_history: bool = True,
        use_aggregates: bool = True,
    ) -> None:
        super().__init__()
        embedding = config["embeddings"]
        self.use_ids = use_ids
        self.use_history = use_history
        self.use_aggregates = use_aggregates
        self.numerical_indices = [
            index for index, name in enumerate(preprocessor.numerical_output_names)
            if use_aggregates or "__hist_" not in name
        ]
        self.categorical_indices = [
            index for index, name in enumerate(preprocessor.categorical_features)
            if use_aggregates or "__hist_" not in name
        ]
        self.categorical_names = [
            preprocessor.categorical_features[index] for index in self.categorical_indices
        ]
        cat_dim = int(embedding["categorical_dimension"])
        self.categorical_embeddings = nn.ModuleDict({
            name: nn.Embedding(
                preprocessor.vocabulary_sizes[name], cat_dim, padding_idx=0
            )
            for name in self.categorical_names
        })
        user_dim = int(embedding["user_dimension"])
        video_dim = int(embedding["video_dimension"])
        self.user_embedding = (
            nn.Embedding(preprocessor.vocabulary_sizes["user_id"], user_dim, padding_idx=0)
            if use_ids else None
        )
        self.video_embedding = (
            nn.Embedding(preprocessor.vocabulary_sizes["video_id"], video_dim, padding_idx=0)
            if use_ids or use_history else None
        )
        self.output_dimension = len(self.numerical_indices) + len(self.categorical_names) * cat_dim
        if use_ids:
            self.output_dimension += user_dim + video_dim
        if use_history:
            self.output_dimension += video_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = [batch["numerical"][:, self.numerical_indices]]
        for column_index, name in zip(self.categorical_indices, self.categorical_names):
            parts.append(self.categorical_embeddings[name](batch["categorical"][:, column_index]))
        if self.use_ids:
            assert self.user_embedding is not None and self.video_embedding is not None
            parts.extend((
                self.user_embedding(batch["user_id"]),
                self.video_embedding(batch["video_id"]),
            ))
        if self.use_history:
            assert self.video_embedding is not None
            history = self.video_embedding(batch["history_video_ids"])
            parts.append(masked_mean_history(history, batch["history_mask"]))
        return torch.cat(parts, dim=1)


def _mlp(input_dim: int, hidden: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden:
        layers.extend((nn.Linear(current, width), nn.ReLU(), nn.Dropout(dropout)))
        current = width
    return nn.Sequential(*layers)


class SharedBottomMLP(nn.Module):
    def __init__(
        self,
        preprocessor: NeuralPreprocessor,
        config: dict[str, Any],
        *,
        use_ids: bool = True,
        use_history: bool = True,
        use_aggregates: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = NeuralInputEncoder(
            preprocessor, config, use_ids=use_ids,
            use_history=use_history, use_aggregates=use_aggregates,
        )
        settings = config["shared_bottom"]
        hidden = [int(value) for value in settings["hidden_dimensions"]]
        self.bottom = _mlp(self.encoder.output_dimension, hidden, float(settings["dropout"]))
        last = hidden[-1] if hidden else self.encoder.output_dimension
        self.heads = nn.ModuleDict({task: nn.Linear(last, 1) for task in TASK_NAMES})

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        shared = self.bottom(self.encoder(batch))
        return torch.cat([self.heads[task](shared) for task in TASK_NAMES], dim=1)

    def probabilities(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self(batch))


class MMoE(nn.Module):
    def __init__(
        self,
        preprocessor: NeuralPreprocessor,
        config: dict[str, Any],
        *,
        use_ids: bool = True,
        use_history: bool = True,
        use_aggregates: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = NeuralInputEncoder(
            preprocessor, config, use_ids=use_ids,
            use_history=use_history, use_aggregates=use_aggregates,
        )
        settings = config["mmoe"]
        experts = int(settings["num_experts"])
        expert_dim = int(settings["expert_dimension"])
        tower_dim = int(settings["tower_dimension"])
        dropout = float(settings["dropout"])
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.encoder.output_dimension, expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(experts)
        ])
        self.gates = nn.ModuleDict({
            task: nn.Linear(self.encoder.output_dimension, experts) for task in TASK_NAMES
        })
        self.towers = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(expert_dim, tower_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(tower_dim, 1),
            )
            for task in TASK_NAMES
        })

    def gate_weights(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the task-specific expert routing probabilities."""
        encoded = self.encoder(batch)
        return {
            task: torch.softmax(self.gates[task](encoded), dim=1)
            for task in TASK_NAMES
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.encoder(batch)
        expert_outputs = torch.stack([expert(encoded) for expert in self.experts], dim=1)
        logits: list[torch.Tensor] = []
        for task in TASK_NAMES:
            weights = torch.softmax(self.gates[task](encoded), dim=1).unsqueeze(-1)
            mixture = (expert_outputs * weights).sum(dim=1)
            logits.append(self.towers[task](mixture))
        return torch.cat(logits, dim=1)

    def probabilities(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self(batch))
