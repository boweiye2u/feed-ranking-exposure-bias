"""Central definitions for large artifact locations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ARTIFACT_ROOT_ENV = "KUAIRAND_ARTIFACT_ROOT"
DEFAULT_ARTIFACT_ROOT = Path("~/recom").expanduser()


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """All large, local artifact directories used by the project."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    def directories(self) -> tuple[Path, ...]:
        """Return artifact directories in a stable order."""
        return (self.raw, self.processed, self.models, self.predictions, self.cache)

    def create(self) -> None:
        """Create the artifact root and its standard subdirectories."""
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)


def get_artifact_paths(root: str | Path | None = None) -> ArtifactPaths:
    """Build paths from an explicit root, the environment, or ``~/recom``."""
    configured_root = root if root is not None else os.environ.get(ARTIFACT_ROOT_ENV)
    path = Path(configured_root).expanduser() if configured_root else DEFAULT_ARTIFACT_ROOT
    return ArtifactPaths(root=path)


def main() -> None:
    """Create and display the configured artifact directories."""
    paths = get_artifact_paths()
    paths.create()
    for directory in paths.directories():
        print(directory)


if __name__ == "__main__":
    main()
