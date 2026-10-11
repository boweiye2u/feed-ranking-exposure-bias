"""Validate public documentation, commands, figures, and tracked-file hygiene."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MARKDOWN=[ROOT/"README.md",ROOT/"docs"/"reproducibility.md",ROOT/"reports"/"final_research_report.md"]


def local_links(path: Path) -> list[Path]:
    targets=[]
    for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)",path.read_text()):
        if target.startswith(("http://","https://","#","mailto:")): continue
        targets.append((path.parent/target.split("#",1)[0]).resolve())
    return targets


def main() -> None:
    missing=[str(target) for path in MARKDOWN for target in local_links(path) if not target.exists()]
    if missing: raise SystemExit(f"Missing local documentation targets: {missing}")
    readme=(ROOT/"README.md").read_text()
    commands=set(re.findall(r"python (scripts/[A-Za-z0-9_./-]+\.py)",readme))
    absent=[command for command in commands if not (ROOT/command).is_file()]
    if absent: raise SystemExit(f"README commands do not exist: {absent}")
    tracked=subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).split(b"\0")
    forbidden=(".parquet",".arrow",".feather",".pt",".pth",".ckpt",".pkl",".joblib")
    violations=[]
    for raw in tracked:
        if not raw: continue
        path=ROOT/raw.decode()
        if path.suffix in forbidden or path.stat().st_size>5*1024*1024: violations.append(str(path.relative_to(ROOT)))
    if violations: raise SystemExit(f"Tracked large/artifact files: {violations}")
    print(f"markdown_files={len(MARKDOWN)} links=PASS commands={len(commands)} tracked_hygiene=PASS")


if __name__=="__main__": main()
