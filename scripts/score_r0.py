#!/usr/bin/env python3
"""Materialize authorized R0 training-split predictions."""

from kuairand_pure.baselines.r0 import materialize_r0_predictions


if __name__ == "__main__":
    result = materialize_r0_predictions()
    print(result["baseline"], result["scoring_rule"])
    print(f"runtime_seconds={result['runtime_seconds']:.3f}")
    print(f"peak_rss_mib={result['peak_rss_mib']:.1f}")
