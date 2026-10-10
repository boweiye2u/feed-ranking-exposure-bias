#!/usr/bin/env python3
"""Run the M2 integration gate and write external machine-readable reports."""

from kuairand_pure.features.integration import review_m2_features


if __name__ == "__main__":
    result = review_m2_features()
    print(result["freeze_version"])
    for split, report in result["splits"].items():
        print(split, report["rows"])
