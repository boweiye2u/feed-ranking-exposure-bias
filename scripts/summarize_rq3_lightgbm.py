#!/usr/bin/env python3
"""Verify and summarize all three frozen M5.2 bundles."""

from kuairand_pure.rq3_training import summarize_rq3_models


report = summarize_rq3_models()
print(f"models={report['model_count']}")
print(f"complete={report['complete']}")
print(f"baseline_preserved={report['baseline_preserved']}")
