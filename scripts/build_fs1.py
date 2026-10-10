#!/usr/bin/env python3
"""Materialize authorized M2.1 FS1 training tables."""

from kuairand_pure.data.fs1 import materialize_training_fs1


if __name__ == "__main__":
    report = materialize_training_fs1()
    print(report["output_dir"])
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")
    print(f"peak_rss_mib={report['peak_rss_mib']:.1f}")
