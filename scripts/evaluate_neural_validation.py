#!/usr/bin/env python3
"""Run M4.5 normal-validation neural evaluation and freeze candidates."""

from kuairand_pure.neural_evaluation import (
    evaluate_neural_normal_validation,
    freeze_candidate_population,
    materialize_neural_validation_sequence,
)


sequence = materialize_neural_validation_sequence()
result = evaluate_neural_normal_validation()
manifest = freeze_candidate_population(result)
print(f"validation_rows={result['rows']}")
print(f"sequence_mean_length={sequence['mean_history_length']:.3f}")
print(f"neural_task_records={len(result['records'])}")
print(f"selected={manifest['selected_overall']['candidate']}")
print(f"runtime_seconds={result['runtime_seconds']:.3f}")
