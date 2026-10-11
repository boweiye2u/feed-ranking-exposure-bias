"""Build M6.1 held-out features and score every frozen model."""

from kuairand_pure.final_predictions import audit_m6_1_outputs, build_m6_1_manifest, materialize_final_features, score_final_predictions, verify_m6_1_preflight

preflight = verify_m6_1_preflight()
features = materialize_final_features()
predictions = score_final_predictions()
audit = audit_m6_1_outputs()
manifest = build_m6_1_manifest(features, predictions, preflight, audit)
print(f"normal_rows={features['populations']['normal']['rows']}")
print(f"random_rows={features['populations']['random']['rows']}")
print(f"prediction_artifacts={predictions['record_count']}")
print(f"manifest_version={manifest['version']}")
