import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from kuairand_pure.models.neural import (
    MMOE_ABLATION_SPECS,
    MMoE,
    SharedBottomMLP,
    masked_mean_history,
)
from kuairand_pure.training.neural import (
    load_checkpoint,
    masked_multitask_bce,
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


def _field(name, dtype, role="dense_model_feature"):
    return {
        "name": name, "data_type": dtype, "present_in": ["fs3"],
        "model_role": role, "allowed_model_families": ["shared_bottom_mlp", "mmoe"],
    }


def _manifest(path: Path) -> Path:
    fields = [
        _field("impression_id", "UInt64", "alignment_only"),
        _field("user_id", "Int64", "neural_id_embedding_input"),
        _field("video_id", "Int64", "neural_id_embedding_input"),
        _field("date", "Int64", "alignment_only"),
        _field("time_ms", "Int64", "alignment_only"),
        _field("context__tab", "Int64"),
        _field("context__is_rand", "Int64", "rq3_joint_only"),
        _field("user__user_active_degree", "String"),
        _field("user__onehot_feat0", "Int64"),
        _field("video__video_duration", "Float64"),
        _field("user__hist_impression_count", "Int64"),
    ]
    path.write_text(json.dumps({"manifest_version": "m2-features-v1", "feature_fields": fields}))
    return path


def _features() -> pl.DataFrame:
    return pl.DataFrame({
        "impression_id": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
        "user_id": [10, 10, 11, 12], "video_id": [20, 21, 20, 22],
        "date": [20220412] * 4, "time_ms": [100, 200, 300, 400],
        "context__tab": [1, 0, 1, 2], "context__is_rand": [0] * 4,
        "user__user_active_degree": ["a", "b", None, "a"],
        "user__onehot_feat0": [1, 2, 1, 3],
        "video__video_duration": [10.0, None, 30.0, 20.0],
        "user__hist_impression_count": [0, 1, 2, 3],
    })


def _sequence() -> pl.DataFrame:
    videos = [[0] * 20 for _ in range(4)]
    times = [[0] * 20 for _ in range(4)]
    masks = [[0] * 20 for _ in range(4)]
    videos[1][-1], times[1][-1], masks[1][-1] = 22, 100, 1  # raw video 20 + 2
    videos[2][-1], times[2][-1], masks[2][-1] = 1001, 200, 1  # unseen raw video 999
    return pl.DataFrame(
        {
            "impression_id": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
            "target_time_ms": [100, 200, 300, 400],
            "history_video_ids": videos, "history_time_ms": times,
            "history_mask": masks, "history_length": [0, 1, 1, 0],
            "history_tie_boundary_dropped": [False] * 4,
        },
        schema_overrides={
            "history_video_ids": pl.Array(pl.Int32, 20),
            "history_time_ms": pl.Array(pl.Int64, 20),
            "history_mask": pl.Array(pl.UInt8, 20),
            "history_length": pl.Int16,
        },
    )


def _labels() -> pl.DataFrame:
    return pl.DataFrame({
        "impression_id": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
        "long_view": [1, 0, 1, 0], "is_like": [0, 1, 0, 0],
        "is_click": [1, 1, 0, 1],
        "click_semantics": ["valid_play", "literal_click", "valid_play", "unsupported_interface"],
    })


def _adapter(tmp_path: Path) -> NeuralPreprocessor:
    return NeuralPreprocessor.fit(
        _features(), feature_manifest_path=_manifest(tmp_path / "manifest.json")
    )


def test_neural_preprocessing_unknowns_missingness_and_history(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    assert "context__is_rand" not in adapter.categorical_features
    assert "video__video_duration" in adapter.missing_indicator_features
    assert adapter.missing_indicator_features == adapter.numerical_features
    arrays = adapter.transform(_features(), _sequence())
    assert arrays.numerical.shape[1] == len(adapter.numerical_output_names)
    assert arrays.history_video_ids[0].tolist() == [0] * 20
    assert arrays.history_video_ids[1, -1] == adapter.video_vocabulary["i:20"]
    assert arrays.history_video_ids[2, -1] == 1

    changed = _features().with_columns(
        pl.when(pl.col("impression_id") == 1).then(999).otherwise(pl.col("user_id")).alias("user_id"),
        pl.when(pl.col("impression_id") == 1).then(pl.lit("unseen")).otherwise(pl.col("user__user_active_degree")).alias("user__user_active_degree"),
    )
    unknown = adapter.transform(changed, _sequence())
    assert unknown.user_ids[0] == 1
    category_index = adapter.categorical_features.index("user__user_active_degree")
    assert unknown.categorical[0, category_index] == 1
    joint_adapter = NeuralPreprocessor.fit(
        _features(), feature_manifest_path=tmp_path / "manifest.json", include_is_rand=True
    )
    assert "context__is_rand" in joint_adapter.categorical_features


def test_three_head_masks_loss_and_history_pooling() -> None:
    targets = build_neural_targets(_labels())
    assert targets.masks.tolist() == [[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0]]
    assert targets.values[1, 2] == 0  # literal-click value cannot enter valid-play loss
    logits = torch.zeros((4, 3), requires_grad=True)
    loss, per_task = masked_multitask_bce(
        logits, torch.from_numpy(targets.values), torch.from_numpy(targets.masks),
        torch.ones(3),
    )
    assert loss.item() == pytest.approx(np.log(2))
    assert set(per_task) == {"long_view", "like", "valid_play"}
    loss.backward()
    assert logits.grad is not None

    embeddings = torch.tensor([[[9.0, 9.0], [1.0, 3.0], [3.0, 5.0]]])
    pooled = masked_mean_history(embeddings, torch.tensor([[0.0, 1.0, 1.0]]))
    assert torch.equal(pooled, torch.tensor([[2.0, 4.0]]))


def test_approved_m4_2_budget_and_complete_batches(tmp_path: Path) -> None:
    config = load_neural_config()
    budget = config["m4_2_training_budget"]
    assert budget["status"] == "approved"
    assert budget["optimizer_updates"] == 7060
    assert budget["examples_processed"] == 3_612_405
    dataset = NeuralDataset(
        _adapter(tmp_path).transform(_features(), _sequence()),
        build_neural_targets(_labels()),
    )
    loader = make_dataloader(dataset, batch_size=3, seed=0, shuffle=False)
    assert loader.drop_last is False
    assert [batch["targets"].shape[0] for batch in loader] == [3, 1]


def test_predefined_ablation_inputs_are_supported(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    config = load_neural_config()
    full = MMoE(adapter, config)
    dense_only = MMoE(adapter, config, use_ids=False, use_history=False)
    no_history = MMoE(adapter, config, use_history=False)
    no_aggregates = MMoE(adapter, config, use_aggregates=False)

    assert {name: spec["input_dimension"] for name, spec in MMOE_ABLATION_SPECS.items()} == {
        "dense_only": 238, "no_history": 270, "no_aggregates": 220,
    }
    assert full.encoder.output_dimension - dense_only.encoder.output_dimension == 48
    assert full.encoder.output_dimension - no_history.encoder.output_dimension == 16
    assert full.encoder.output_dimension - no_aggregates.encoder.output_dimension == 2
    assert dense_only.encoder.user_embedding is None
    assert dense_only.encoder.video_embedding is None
    assert no_history.encoder.user_embedding is not None
    assert no_history.encoder.video_embedding is not None
    assert no_history.encoder.use_history is False
    assert no_aggregates.encoder.use_aggregates is False
    assert all(
        "__hist_" not in adapter.numerical_output_names[index]
        for index in no_aggregates.encoder.numerical_indices
    )


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("dense_only", {"use_ids": False, "use_history": False}),
        ("no_history", {"use_history": False}),
        ("no_aggregates", {"use_aggregates": False}),
    ],
)
def test_mmoe_ablation_gradients_and_checkpoint_identity(
    tmp_path: Path, name: str, kwargs: dict
) -> None:
    manifest_path = _manifest(tmp_path / "manifest.json")
    adapter = NeuralPreprocessor.fit(_features(), feature_manifest_path=manifest_path)
    dataset = NeuralDataset(
        adapter.transform(_features(), _sequence()), build_neural_targets(_labels())
    )
    config = load_neural_config()
    set_deterministic_seed(0)
    model = MMoE(adapter, config, **kwargs)
    loader = make_dataloader(dataset, batch_size=4, seed=0, shuffle=False)
    counters, _, optimizer = train_updates(
        model, loader, device=torch.device("cpu"), learning_rate=0.001,
        weight_decay=1e-5, task_weights=[1, 1, 1], max_updates=1,
    )
    assert counters.optimizer_updates == 1 and counters.training_examples == 4
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    adapter_path = tmp_path / "adapter.json"
    adapter.save(adapter_path)
    adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    batch = next(iter(make_dataloader(dataset, batch_size=4, seed=0, shuffle=False)))
    model.eval()
    before = model.probabilities(batch)
    destination = tmp_path / name
    architecture = f"mmoe_{name}"
    save_checkpoint(
        destination, model=model, optimizer=optimizer, architecture=architecture,
        preprocessing_sha256=adapter_hash, feature_manifest_sha256=manifest_hash,
        seed=0, counters=counters, config=config,
        extra_metadata={"ablation": name},
    )
    reloaded = MMoE(adapter, config, **kwargs)
    manifest = load_checkpoint(
        destination, model=reloaded, expected_architecture=architecture,
        expected_preprocessing_sha256=adapter_hash,
        expected_feature_manifest_sha256=manifest_hash, device=torch.device("cpu"),
    )
    reloaded.eval()
    assert manifest["production_metadata"]["ablation"] == name
    assert torch.equal(before, reloaded.probabilities(batch))
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(
            destination, model=reloaded, expected_architecture="mmoe_wrong_ablation",
            expected_preprocessing_sha256=adapter_hash,
            expected_feature_manifest_sha256=manifest_hash, device=torch.device("cpu"),
        )


def test_mmoe_gate_normalization_and_module_gradients(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    config = load_neural_config()
    dataset = NeuralDataset(
        adapter.transform(_features(), _sequence()), build_neural_targets(_labels())
    )
    batch = next(iter(make_dataloader(dataset, batch_size=4, seed=0, shuffle=False)))
    model = MMoE(adapter, config)
    weights = model.gate_weights(batch)
    assert set(weights) == {"long_view", "like", "valid_play"}
    for task_weights in weights.values():
        assert task_weights.shape == (4, 4)
        assert torch.all(task_weights >= 0)
        assert torch.allclose(task_weights.sum(dim=1), torch.ones(4))

    logits = model(batch)
    loss, _ = masked_multitask_bce(
        logits, batch["targets"], batch["task_masks"], torch.ones(3)
    )
    loss.backward()
    for name, module in (
        ("experts", model.experts), ("gates", model.gates), ("towers", model.towers)
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients), name
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0, name


def test_seed_independence_and_alignment_rejection(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    config = load_neural_config()
    arrays = adapter.transform(_features(), _sequence())
    targets = build_neural_targets(_labels())
    set_deterministic_seed(0)
    seed_zero = SharedBottomMLP(adapter, config)
    set_deterministic_seed(1)
    seed_one = SharedBottomMLP(adapter, config)
    batch = next(iter(make_dataloader(
        NeuralDataset(arrays, targets), batch_size=4, seed=0, shuffle=False
    )))
    seed_zero.eval(); seed_one.eval()
    assert not torch.equal(seed_zero(batch), seed_one(batch))

    targets.impression_ids = targets.impression_ids[::-1].copy()
    with pytest.raises(ValueError, match="differ by impression ID"):
        NeuralDataset(arrays, targets)


@pytest.mark.parametrize(("constructor", "name"), [(SharedBottomMLP, "shared_bottom"), (MMoE, "mmoe")])
def test_forward_updates_seed_and_checkpoint_reload(tmp_path: Path, constructor, name) -> None:
    manifest_path = _manifest(tmp_path / "manifest.json")
    adapter = NeuralPreprocessor.fit(_features(), feature_manifest_path=manifest_path)
    arrays = adapter.transform(_features(), _sequence())
    dataset = NeuralDataset(arrays, build_neural_targets(_labels()))
    config = load_neural_config()

    set_deterministic_seed(0)
    first = constructor(adapter, config)
    set_deterministic_seed(0)
    second = constructor(adapter, config)
    batch = next(iter(make_dataloader(dataset, batch_size=4, seed=0, shuffle=False)))
    first.eval(); second.eval()
    assert torch.equal(first(batch), second(batch))

    loader = make_dataloader(dataset, batch_size=2, seed=0, shuffle=True)
    counters, logs, optimizer = train_updates(
        first, loader, device=torch.device("cpu"), learning_rate=0.001,
        weight_decay=1e-5, task_weights=[1, 1, 1], max_updates=2,
    )
    assert counters.training_examples == 4 and counters.optimizer_updates == 2
    assert len(logs) == 2

    adapter_path = tmp_path / "adapter.json"
    adapter.save(adapter_path)
    adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    first.eval()
    before = first(batch)
    directory = tmp_path / name
    save_checkpoint(
        directory, model=first, optimizer=optimizer, architecture=name,
        preprocessing_sha256=adapter_hash, feature_manifest_sha256=manifest_hash,
        seed=0, counters=counters, config=config,
    )
    reloaded = constructor(adapter, config)
    load_checkpoint(
        directory, model=reloaded, expected_architecture=name,
        expected_preprocessing_sha256=adapter_hash,
        expected_feature_manifest_sha256=manifest_hash, device=torch.device("cpu"),
    )
    reloaded.eval()
    assert torch.equal(before, reloaded(batch))
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(
            directory, model=reloaded, expected_architecture="wrong",
            expected_preprocessing_sha256=adapter_hash,
            expected_feature_manifest_sha256=manifest_hash, device=torch.device("cpu"),
        )
    checkpoint_path = directory / "checkpoint.pt"
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_checkpoint(
            directory, model=reloaded, expected_architecture=name,
            expected_preprocessing_sha256=adapter_hash,
            expected_feature_manifest_sha256=manifest_hash, device=torch.device("cpu"),
        )
