"""Audit reachable Git history for common public-release hazards."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BLOB_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".arrow", ".ckpt", ".db", ".feather", ".joblib", ".key", ".parquet",
    ".pem", ".pkl", ".pt", ".pth", ".sqlite",
}
SECRET_PATTERNS = {
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(rb"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "Git/service token": re.compile(rb"(?:ghp|github_pat|glpat|sk)-[A-Za-z0-9_-]{16,}"),
    "credential assignment": re.compile(rb"(?i)(?:api[_-]?key|password)[ \t]*[:=][ \t]*[^\s$<{][^\r\n]{5,}"),
}
PRIVATE_PATH = re.compile(rb"/(?:home|Users)/[^/\s]+/(?:Dropbox|Documents|Desktop)/")


def git(*args: str, text: bool = False) -> bytes | str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=text)


def reachable_blobs() -> dict[str, set[str]]:
    blobs: dict[str, set[str]] = {}
    for line in git("rev-list", "--objects", "--all", text=True).splitlines():
        object_id, _, path = line.partition(" ")
        if git("cat-file", "-t", object_id, text=True).strip() == "blob":
            blobs.setdefault(object_id, set()).add(path or "<unnamed>")
    return blobs


def audit() -> dict[str, int]:
    violations: list[str] = []
    blobs = reachable_blobs()
    largest = 0
    for object_id, paths in blobs.items():
        size = int(git("cat-file", "-s", object_id, text=True))
        largest = max(largest, size)
        labels = ", ".join(sorted(paths))
        if size > MAX_BLOB_BYTES:
            violations.append(f"large historical blob ({size} bytes): {labels}")
        if any(Path(path).suffix.lower() in FORBIDDEN_SUFFIXES for path in paths):
            violations.append(f"forbidden artifact extension: {labels}")
        if any(Path(path).name == ".env" for path in paths):
            violations.append(f"private environment file: {labels}")
        content = git("cat-file", "-p", object_id)
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                violations.append(f"{name} pattern: {labels}")
        if PRIVATE_PATH.search(content):
            violations.append(f"machine-specific private path: {labels}")
    if violations:
        raise SystemExit("Release audit failed:\n- " + "\n- ".join(sorted(set(violations))))
    result = {
        "commits": int(git("rev-list", "--all", "--count", text=True)),
        "blobs": len(blobs),
        "largest_blob_bytes": largest,
    }
    print(" ".join(f"{key}={value}" for key, value in result.items()), "history_audit=PASS")
    return result


if __name__ == "__main__":
    audit()
