#!/usr/bin/env python3
"""Build normal-validation features and evaluate frozen M3 candidates."""

from kuairand_pure.evaluation import (
    evaluate_normal_validation,
    materialize_normal_validation_features,
)


feature_report = materialize_normal_validation_features()
evaluation_report = evaluate_normal_validation()
print(f"validation_rows={feature_report['rows']}")
print(f"feature_runtime_seconds={feature_report['runtime_seconds']:.3f}")
print(f"candidate_task_records={len(evaluation_report['records'])}")
print(f"selected={evaluation_report['selected_lightgbm']['candidate']}")
print(f"evaluation_runtime_seconds={evaluation_report['runtime_seconds']:.3f}")
