from pathlib import Path

from kuairand_pure.paths import ARTIFACT_ROOT_ENV, get_artifact_paths


def test_explicit_artifact_paths(tmp_path: Path) -> None:
    paths = get_artifact_paths(tmp_path)

    assert paths.root == tmp_path
    assert paths.directories() == (
        tmp_path / "raw",
        tmp_path / "processed",
        tmp_path / "models",
        tmp_path / "predictions",
        tmp_path / "cache",
    )


def test_environment_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path))

    assert get_artifact_paths().root == tmp_path


def test_create_artifact_tree(tmp_path: Path) -> None:
    paths = get_artifact_paths(tmp_path)
    paths.create()

    assert all(path.is_dir() for path in paths.directories())


def test_default_root(monkeypatch) -> None:
    monkeypatch.delenv(ARTIFACT_ROOT_ENV, raising=False)

    assert get_artifact_paths().root == Path.home() / "recom"
