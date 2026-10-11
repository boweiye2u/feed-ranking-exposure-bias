# M7.3 release-readiness audit

## Verdict

The source repository is technically ready to become a release candidate. No secret, raw-record, large-artifact, or frozen-research integrity blocker was found. Public release remains intentionally blocked on owner decisions for the GitHub owner/repository name and code license; no remote was configured and nothing was pushed.

## Frozen integrity

All six pre-M7.3 annotated tags resolve to their recorded commits. Live checksum verification passed for the M6.1 prediction manifest, M6.2 results, M6.3 results, frozen protocol documents, M2 feature manifest, all 146 prediction records, and downstream evaluation artifacts. The verified populations remain 12 RQ1/RQ2 candidates, contenders `fs1_deep`, `fs1_shallow`, and `mmoe_no_aggregates`, four RQ3 strategies, and six predefined paired comparisons.

The study plan, Week-0 amendment, and M5.3 statistical amendment are unchanged from `m7-readme-v1`. No model was trained, no prediction was regenerated, and no scientific result was recomputed during M7.3.

## Security and history audit

The complete reachable history at audit time contained six commits, 267 Git objects, and no packed data. Every historical path and blob was inspected. The largest blob was a publication figure of 97,122 bytes. Checks found:

- no API keys, passwords, service tokens, private keys, or private configuration;
- no raw/processed interaction data, model checkpoints, prediction artifacts, environment directories, or caches;
- no historical file with a prohibited artifact extension or blob over 5 MiB;
- no machine-specific workstation-document path or operational private absolute path;
- no unnecessary personal information identified by path/content review;
- no vendored dependency or third-party dataset content; and
- only compact aggregate CSV result tables and project-generated figures.

The tracked `.env.example` contains only the portable `KUAIRAND_ARTIFACT_ROOT=~/recom/` example and no credential. Documentation uses benign `~/recom/` and `/absolute/path/to/recom` placeholders. Git history was not rewritten.

The automated audit is `python scripts/audit_release.py`. It checks every reachable historical blob, including content deleted from the current tree, for high-confidence credential patterns, private workstation paths, prohibited artifact extensions, and files over 5 MiB. Pattern scanning is defense in depth, not a guarantee against every possible secret format.

## Licensing and attribution

The README and data card link the official KuaiRand repository, Zenodo record, arXiv paper, archive checksum, and the archive's included CC BY-SA 4.0 text. Dataset-specific redistribution scope remains insufficiently clear, so raw and processed data must remain outside Git and users must obtain the archive from the official source.

No project code license exists. Apache-2.0 is recommended for a public ML/research implementation because it is permissive and includes an explicit patent grant, but selecting a license is an owner decision. No license file was added. Python dependencies are not vendored and remain governed by their own licenses.

## Clean-checkout test

The source checkout was cloned from annotated tag `m7-readme-v1` into `/tmp/kuairand-m7.3.aSi9XK/repo`, with `HOME=/tmp/kuairand-m7.3.aSi9XK/home` and `KUAIRAND_ARTIFACT_ROOT=/tmp/kuairand-m7.3.aSi9XK/no-artifacts`. The artifact directory remained absent.

An initial new-venv probe intentionally used `--no-deps` because network dependency download was unavailable. Package metadata installed, but test collection correctly failed because that base environment lacks Polars, LightGBM, and PyTorch. This is not counted as a successful fresh-environment dependency test.

The definitive clean-source test reused the already validated project dependency environment and ran:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
python scripts/check_public_repo.py
python scripts/check_environment.py
python scripts/train_lightgbm_matrix.py --help
python scripts/train_shared_bottom.py --help
python scripts/train_mmoe.py --help
python scripts/train_mmoe_ablation.py --help
python scripts/train_rq3_lightgbm.py --help
```

Results: editable installation passed in 1.20 seconds; **135 tests passed and 4 external-M6 tests skipped** in 7.89 seconds (8.89 seconds wall time including startup); documentation links, 24 documented script paths, and tracked-file hygiene passed; five argument-bearing training entry points returned help successfully; all declared imports succeeded. The four skips are intentional when frozen external artifacts are unavailable.

This establishes that a clean source checkout works without `~/recom/` when dependencies are already installed. It does not establish offline dependency installation, a clean-machine network installation, or end-to-end research reproduction from raw data.

## GitHub presentation review

The README opening states the motivation and all three research questions. It prominently reports both positive all-candidate regret and the inconclusive contender-only result. All three selected PNG figures use valid relative paths. Installation, dataset acquisition, artifact configuration, entry points, the full report, the reproducibility guide, and scientific limitations are discoverable.

Known placeholders are the public clone URL, GitHub owner, repository name, and code license. These must be resolved before publication.

## Required approvals and publication commands

Required owner decisions:

1. GitHub account or organization.
2. Public repository name (suggested: `kuairand-exposure-bias`).
3. Code license (recommended: Apache-2.0; MIT is a simpler permissive alternative).
4. Confirmation that the README clone placeholder has been replaced.

After those decisions, review the chosen license and execute—not during M7.3—the equivalent of:

```bash
# Add the approved LICENSE file and replace <published-repository-url> in README.md.
git add LICENSE README.md
git commit -m "docs: add approved public repository details"
gh repo create <owner>/<repository-name> --public --source=. --remote=origin
git remote -v
git push -u origin main
git push origin --follow-tags
```

Before pushing, rerun `python -m pytest -q`, `python scripts/check_public_repo.py`, `python scripts/audit_release.py`, inspect `git status`, and verify GitHub's rendered README. Creating the remote and pushing require separate explicit authorization.
