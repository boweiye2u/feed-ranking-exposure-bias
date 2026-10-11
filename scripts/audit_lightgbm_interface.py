#!/usr/bin/env python3
"""Fit training-only categorical adapters and audit LightGBM task populations."""

import json
import resource
import time
from importlib.metadata import version

from kuairand_pure.models.lightgbm import (
    FEATURE_SETS,
    TASKS,
    LightGBMAdapter,
    load_training_artifacts,
    select_task,
)
from kuairand_pure.paths import get_artifact_paths


def main() -> None:
    started = time.perf_counter()
    paths = get_artifact_paths()
    feature_manifest = paths.processed / "m2" / "v1" / "feature_manifest.json"
    output = paths.models / "lightgbm" / "infrastructure" / "v1"
    output.mkdir(parents=True, exist_ok=True)
    adapters = {}
    task_counts = None
    for feature_set in FEATURE_SETS:
        features, labels = load_training_artifacts(paths.processed, feature_set)
        adapter = LightGBMAdapter.fit(
            features, feature_set=feature_set, feature_manifest_path=feature_manifest
        )
        adapter_path = output / f"{feature_set}_adapter.json"
        adapter.save(adapter_path)
        encoded_sample = adapter.transform(features.head(4096))
        adapters[feature_set] = {
            "adapter_path": str(adapter_path),
            "feature_count": len(adapter.feature_names),
            "categorical_count": len(adapter.categorical_features),
            "numerical_count": len(adapter.numerical_features),
            "vocabulary_cardinalities": {
                name: len(vocabulary) for name, vocabulary in adapter.vocabularies.items()
            },
            "sample_rows_transformed": len(encoded_sample),
            "context_is_rand_excluded": "context__is_rand" not in adapter.feature_names,
        }
        if task_counts is None:
            encoded = adapter.transform(features.head(1))
            task_counts = {}
            # Masks do not depend on encoded values; provide a correctly sized placeholder.
            import pandas as pd
            placeholder = pd.DataFrame({"x": [0.0] * features.height})
            for task in TASKS:
                task_counts[task] = select_task(placeholder, features, labels, task).rows
    dependencies = {
        name: version(name) for name in (
            "numpy", "polars", "pyarrow", "pandas", "scikit-learn",
            "lightgbm", "torch", "matplotlib", "pytest",
        )
    }
    report = {
        "milestone": "M3.2a",
        "training_source": "normal_train_only",
        "feature_manifest": str(feature_manifest),
        "adapters": adapters,
        "task_counts": task_counts,
        "pooled_click_model": False,
        "esmm": False,
        "validation_or_test_outcomes_accessed": False,
        "models_trained": False,
        "dependencies": dependencies,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (output / "data_quality.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    print(task_counts)
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")
    print(f"peak_rss_mib={report['peak_rss_mib']:.1f}")


if __name__ == "__main__":
    main()
