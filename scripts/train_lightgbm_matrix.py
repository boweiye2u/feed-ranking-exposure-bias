#!/usr/bin/env python3
"""Run the M3.2b benchmark or the gated full LightGBM matrix."""

import argparse

from kuairand_pure.models.lightgbm_matrix import run_benchmark, run_full_matrix


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("benchmark", "full"))
args = parser.parse_args()

report = run_benchmark() if args.mode == "benchmark" else run_full_matrix()
print(report["milestone"])
if args.mode == "benchmark":
    print(report["record"])
    print(f"resource_gate_passed={report['resource_gate_passed']}")
else:
    print(f"models={report['model_count']}")
    print(f"runtime_seconds={report['total_runtime_seconds']:.3f}")
    print(f"peak_rss_mib={report['process_peak_rss_mib']:.1f}")
