#!/usr/bin/env python3
"""Materialize and audit the three M5.1 RQ3 training populations."""

from kuairand_pure.rq3 import prepare_rq3_training_data


report = prepare_rq3_training_data()
print(f"normal_rows={report['populations']['normal']['rows']}")
print(f"random_rows={report['populations']['random']['rows']}")
print(f"joint_rows={report['populations']['joint']['rows']}")
print(f"weight_ess={report['weighting']['diagnostics']['effective_sample_size']:.6f}")
print(f"runtime_seconds={report['runtime_seconds']:.3f}")
