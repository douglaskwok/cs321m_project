"""Overnight benchmark runner.

Runs kudge.py on Qwen3.5 models, then kudge_pairwise.py on Ministral
and Qwen3.5 models, sequentially.  Continues on failure and prints a
summary at the end.

Usage::

    python benchmarks/kudge/run_overnight.py [--dry-run]

Or via modal directly for each job — see JOBS list below.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

KUDGE_MODELS = [
    "qwen3.5-27b",
    "qwen3.5-9b",
    "qwen3.5-4b",
    "qwen3.5-2b",
]

PAIRWISE_MODELS = [
    "ministral-14b",
    "ministral-8b",
    "ministral-3b",
    "qwen3.5-27b",
    "qwen3.5-9b",
    "qwen3.5-4b",
    "qwen3.5-2b",
]

JOBS: list[tuple[str, str]] = (
    [("benchmarks/kudge/kudge.py", m) for m in KUDGE_MODELS]
    + [("benchmarks/kudge/kudge_pairwise.py", m) for m in PAIRWISE_MODELS]
)


_RESULTS_SUBDIR = {
    "kudge": "kudge_challenge_easy_hard",
    "kudge_pairwise": "kudge_judge_easy_hard",
}


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


def _result_files(script: str, model: str) -> list[Path]:
    slug = _model_slug(model)
    stem = Path(script).stem  # "kudge" or "kudge_pairwise"
    results_dir = Path(__file__).parent / "results" / _RESULTS_SUBDIR[stem]
    return [
        results_dir / f"{stem}_{slug}.npz",
        results_dir / f"{stem}_{slug}_responses.jsonl",
    ]


def git_commit_push(script: str, model: str, *, dry_run: bool) -> None:
    files = [f for f in _result_files(script, model) if f.exists()]
    if not files:
        print("  git: no result files found, skipping commit", flush=True)
        return

    label = f"{Path(script).stem} / {model}"
    msg = f"results: {label}"
    rel_files = [str(f.relative_to(ROOT)) for f in files]

    cmds = [
        ["git", "add"] + rel_files,
        ["git", "commit", "-m", msg],
        ["git", "push"],
    ]

    for cmd in cmds:
        if dry_run:
            print(f"  dry-run git: {' '.join(cmd)}")
            continue
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"  git warning: {' '.join(cmd)} exited {result.returncode}", flush=True)
            break


def _banner(msg: str) -> None:
    print(f"\n{'=' * 64}\n{msg}\n{'=' * 64}", flush=True)


def run_job(script: str, model: str, *, dry_run: bool) -> bool:
    cmd = ["modal", "run", script, "--model", model]
    label = f"{Path(script).stem} / {model}"
    _banner(f"[{datetime.now():%H:%M:%S}]  START  {label}")
    if dry_run:
        print(f"  dry-run: would execute: {' '.join(cmd)}")
        return True
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.monotonic() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (rc={result.returncode})"
    print(
        f"\n[{datetime.now():%H:%M:%S}]  {status}  {label}  ({elapsed / 60:.1f} min)",
        flush=True,
    )
    return ok


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    overall_start = time.monotonic()
    failures: list[str] = []

    for script, model in JOBS:
        ok = run_job(script, model, dry_run=dry_run)
        if ok:
            git_commit_push(script, model, dry_run=dry_run)
        else:
            failures.append(f"{Path(script).stem} / {model}")

    total_min = (time.monotonic() - overall_start) / 60
    _banner(f"SUMMARY  (total: {total_min:.0f} min)")
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  FAILED: {f}")
        sys.exit(1)
    else:
        print(f"All {len(JOBS)} runs completed successfully.")


if __name__ == "__main__":
    main()
