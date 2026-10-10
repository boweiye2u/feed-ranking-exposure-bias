#!/usr/bin/env python3
"""Materialize authorized training-only FS3 and neural histories."""

from kuairand_pure.features.fs3 import materialize_training_fs3


if __name__ == "__main__":
    report = materialize_training_fs3()
    print(report["output_dir"])
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")
    print(f"peak_rss_mib={report['peak_rss_mib']:.1f}")
