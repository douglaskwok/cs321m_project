"""Run sampled CodeJudgeBench CodeGen verdict-mode jobs sequentially.

This wrapper launches the Modal runner once per model, in a fixed order, so a
single approved command can keep working through the queue. Each model run
uses codejudgebench_pairwise_faithful.py, which checkpoints JSONL and NPZ
outputs after every completed batch and resumes from existing JSONL records.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODELS = [
    "qwen3.5-27b",
    "ministral-14b",
    "qwen3.5-9b",
    "ministral-8b",
    "qwen3.5-4b",
    "ministral-3b",
    "qwen3.5-2b",
]

EXPECTED_RECORDS = 2103
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"
LOG_DIR = REPO_ROOT / "logs"
RUNNER = SCRIPT_DIR / "codejudgebench_pairwise_faithful.py"


def model_slug(model: str) -> str:
    return "".join(ch for ch in model.lower() if ch.isalnum())


def result_paths(model: str, seed: int, max_tokens: int) -> tuple[Path, Path]:
    slug = f"{model_slug(model)}_oneorder_seed{seed}_verdict_tok{max_tokens}"
    if not getattr(result_paths, "prefill_verdict", True):
        slug += "_noprefill"
    return (
        RESULTS_DIR / f"codejudgebench_pairwise_{slug}.npz",
        RESULTS_DIR / f"codejudgebench_pairwise_{slug}_responses.jsonl",
    )


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            json.loads(line)
            count += 1
    return count


def run_model(model: str, args: argparse.Namespace) -> int:
    npz_path, jsonl_path = result_paths(model, args.seed, args.max_tokens)
    existing = count_jsonl_records(jsonl_path)
    if existing >= EXPECTED_RECORDS and not args.force:
        print(f"[skip] {model}: {existing}/{EXPECTED_RECORDS} records already saved")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_log = LOG_DIR / f"codejudgebench_{model_slug(model)}_verdict_tok{args.max_tokens}.out.log"
    err_log = LOG_DIR / f"codejudgebench_{model_slug(model)}_verdict_tok{args.max_tokens}.err.log"

    cmd = [
        sys.executable,
        "-m",
        "modal",
        "run",
        str(RUNNER.relative_to(REPO_ROOT)),
        "--model",
        model,
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--prompt-style",
        "verdict",
        "--max-tokens",
        str(args.max_tokens),
    ]
    if not args.prefill_verdict:
        cmd.append("--no-prefill-verdict")

    print(f"[start] {model}: existing={existing}/{EXPECTED_RECORDS}")
    print(f"[cmd] {' '.join(cmd)}")
    print(f"[logs] {out_log}")
    print(f"[logs] {err_log}")
    with out_log.open("a", encoding="utf-8") as stdout, err_log.open("a", encoding="utf-8") as stderr:
        stdout.write(f"\n\n=== START {model} ===\n")
        stderr.write(f"\n\n=== START {model} ===\n")
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        stdout.write(f"\n=== END {model} rc={proc.returncode} ===\n")
        stderr.write(f"\n=== END {model} rc={proc.returncode} ===\n")

    final_count = count_jsonl_records(jsonl_path)
    print(f"[done] {model}: rc={proc.returncode}, records={final_count}/{EXPECTED_RECORDS}")
    print(f"[out] {npz_path}")
    print(f"[out] {jsonl_path}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true", default=True)
    parser.add_argument("--prefill-verdict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--models", nargs="*", default=MODELS)
    args = parser.parse_args()
    result_paths.prefill_verdict = args.prefill_verdict  # type: ignore[attr-defined]

    failures: list[tuple[str, int]] = []
    for model in args.models:
        rc = run_model(model, args)
        if rc != 0:
            failures.append((model, rc))
            print(f"[error] {model} failed with rc={rc}")
            if not args.keep_going:
                break

    if failures:
        print("[summary] failures:")
        for model, rc in failures:
            print(f"  {model}: rc={rc}")
        return 1

    print("[summary] all requested model runs completed or were skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
