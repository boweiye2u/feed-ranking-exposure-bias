"""Execute frozen M6.3 RQ2 cross-fitted regret analysis."""

from kuairand_pure.rq2 import run_m6_3

result = run_m6_3()
for name, values in result["results"].items():
    print(name, values["normal_selection_regret"], values["random_pick_regret"])
print(f"runtime_seconds={result['runtime_seconds']:.3f}")
