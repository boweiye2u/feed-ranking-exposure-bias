#!/usr/bin/env python3
"""Materialize authorized training-only FS2 tables."""

from kuairand_pure.features.fs2 import materialize_training_fs2


if __name__ == "__main__":
    report = materialize_training_fs2()
    print(report["output_dir"])
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")
    print(f"peak_rss_mib={report['peak_rss_mib']:.1f}")
