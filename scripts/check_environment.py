#!/usr/bin/env python3
"""Report whether the M0 runtime dependencies can be imported."""

from __future__ import annotations

import importlib

IMPORT_NAMES = (
    "numpy",
    "polars",
    "pyarrow",
    "pandas",
    "sklearn",
    "lightgbm",
    "torch",
    "matplotlib",
    "pytest",
)


def main() -> None:
    failures: list[str] = []
    for name in IMPORT_NAMES:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            failures.append(f"{name}: {exc}")
        else:
            print(f"{name}: {getattr(module, '__version__', 'unknown')}")
    if failures:
        raise SystemExit("Missing dependencies:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
