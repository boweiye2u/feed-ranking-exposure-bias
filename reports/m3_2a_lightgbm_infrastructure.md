# M3.2a LightGBM infrastructure quality report

Machine-readable report: `~/recom/models/lightgbm/infrastructure/v1/data_quality.json`.

## Production adapter audit

- Training vocabulary source: authorized normal training only, 722,481 aligned rows.
- FS1: 44 model fields, 35 categorical, 9 numerical.
- FS2: 66 model fields, 35 categorical, 31 numerical.
- FS3: 83 model fields, 36 categorical, 47 numerical.
- `context__is_rand`: excluded from every adapter.
- FS3 canonical category cardinality: 111 whole tag-set values.
- Each adapter successfully transformed the first 4,096 canonical training rows.
- Pooled click model: disabled.
- ESMM: disabled.
- Models trained by the production audit: none.
- Validation or test outcomes accessed: none.

Task-mask counts are 722,481 long-view, 722,481 like, 535,569 valid-play, and 78,575 literal-click rows. Counts describe eligibility only; no label rates or performance metrics were calculated.

## Tests and safeguards

Synthetic tests cover categorical dtype preservation, anonymous numeric categories, deterministic null/unseen codes, numerical missing values, complete tag-set non-expansion, FS1–FS3 schemas, task masks, pooled-click rejection, row misalignment, fixed capacity configurations, deterministic repeated training, probability bounds, serialization/reload, checksum compatibility, dtype drift, and missing fields.

Production adapters record the frozen feature-manifest checksum and exact input dtypes. Model bundles record model and adapter checksums and reject stale build/configuration versions.

## Runtime and dependencies

- Adapter audit runtime: 25.584 seconds.
- Peak RSS: 1,473.4 MiB.
- Adapter/report artifacts: 1,219,826 bytes (approximately 1.16 MiB).
- Focused LightGBM tests: 10 passed in 1.33 seconds.
- Complete test suite: 87 passed in 4.24 seconds.

| Dependency | Version |
|---|---|
| Python | 3.12.7 |
| NumPy | 2.5.3 |
| Polars | 2.0.0 |
| PyArrow | 26.0.0 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| LightGBM | 4.7.0 |
| PyTorch | 2.14.1+cpu |
| matplotlib | 3.11.2 |
| pytest | 9.1.1 |

No full experiment, normal-validation selection, or held-out evaluation was run.
