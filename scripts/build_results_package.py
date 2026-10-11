"""Generate the verified M7.1 tables and publication figures."""

from kuairand_pure.results_package import build_results_package

result=build_results_package()
print(f"tables={result['tables']}")
print(f"figures={result['figures']}")
