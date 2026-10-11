#!/usr/bin/env python3
"""Freeze contenders and verify final-evaluation readiness without held-out outcomes."""

from kuairand_pure.prefinal import build_readiness_manifest, freeze_normal_validation_contenders


contenders = freeze_normal_validation_contenders()
manifest = build_readiness_manifest(contenders)
print(f"candidates={manifest['candidate_count']}")
print(f"contenders={','.join(contenders['contenders'])}")
print(f"rq3_arms={len(manifest['rq3_arms'])}")
print(f"held_out_outcomes_accessed={manifest['held_out_outcomes_accessed']}")
