#!/usr/bin/env python3
"""Train one frozen M5.2 RQ3 LightGBM arm."""

import argparse

from kuairand_pure.rq3 import STRATEGIES
from kuairand_pure.rq3_training import train_rq3_arm


parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True, choices=STRATEGIES)
arguments = parser.parse_args()
record = train_rq3_arm(arguments.arm)
print(f"arm={record['arm']}")
print(f"rows={record['training_rows']}")
print(f"seconds={record['training_seconds']:.3f}")
print(f"model_sha256={record['model_sha256']}")
