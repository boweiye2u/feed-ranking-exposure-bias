"""Execute the authorized M6.2 RQ1/RQ3 held-out evaluation."""

from kuairand_pure.final_results import run_m6_2

result = run_m6_2()
print(f"rq1_records={len(result['rq1_long_view_primary'])}")
print(f"rq3_arms={len(result['rq3'])}")
print(f"paired_comparisons={len(result['paired_comparisons'])}")
print(f"runtime_seconds={result['runtime_seconds']:.3f}")
