"""LiveCodeBench v6 solver benchmark (Jain et al., 2024).

Runs an OpenAI or Anthropic model on LiveCodeBench v6 problems and evaluates
generated code solutions against public test cases.  Score is the fraction of
problems for which the model's solution passes all provided test cases (pass@1).

Usage
-----
One-time secret setup in Modal::

    modal secret create openai-secret OPENAI_API_KEY=sk-...
    modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...

Then run::

    modal run benchmarks/code/livecodebench.py [--model MODEL]

``MODEL`` defaults to ``gpt-5.4-nano``.  Pass any OpenAI or Anthropic model
name — models whose name starts with ``claude-`` are routed to Anthropic
automatically, e.g.::

    modal run benchmarks/code/livecodebench.py --model claude-haiku-4-5-20251001
    modal run benchmarks/code/livecodebench.py --model gpt-4o-mini

Results are written to two files (slug derived from the model name):

* ``benchmarks/code/results/livecodebench_<slug>.npz`` — response matrix and
  metadata (``response_matrix``, ``item_ids``, ``difficulties``, ``platforms``).
  ``response_matrix`` has shape ``(1, n_items)``, dtype int8.
* ``benchmarks/code/results/livecodebench_<slug>_responses.jsonl`` — one JSON
  record per problem with ``id``, ``difficulty``, ``platform``, ``correct``,
  and the full ``raw`` model response.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import os
from pathlib import Path

import modal
import numpy as np

# ---------------------------------------------------------------------------
# Modal image & app
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.20,<3.0",
        "openai>=1.30",
        "anthropic>=0.25",
        "numpy>=1.24",
    )
    .add_local_file(
        Path(__file__).parent.parent / "llm_client.py",
        remote_path="/root/llm_client.py",
    )
)

app = modal.App("livecodebench", image=image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "livecodebench/code_generation_lite"
VERSION_TAG = "release_v6"
DEFAULT_MODEL = "gpt-5.4-nano"
TEST_LIMIT: int | None = 2  # set to None to run all 1,055 problems
EXEC_TIMEOUT = 10.0  # seconds per test case


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


def _out_paths(model: str) -> tuple[Path, Path]:
    slug = _model_slug(model)
    base = Path(__file__).parent / "results"
    return (
        base / f"livecodebench_{slug}.npz",
        base / f"livecodebench_{slug}_responses.jsonl",
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SOLVER_SYSTEM = (
    "You are an expert competitive programmer. "
    "Write a complete Python 3 solution that reads from stdin and writes to stdout. "
    "Output only the code — no explanation, no markdown fences."
)


def _build_solver_prompt(problem_content: str, starter_code: str) -> str:
    prompt = problem_content.strip()
    if starter_code and starter_code.strip():
        prompt += f"\n\nStarter code:\n{starter_code.strip()}"
    return prompt


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> str:
    """Strip markdown fences; return raw code if none found."""
    m = _CODE_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Code execution helpers (run inside Modal containers)
# ---------------------------------------------------------------------------


def _run_code(code: str, test_input: str, timeout: float = EXEC_TIMEOUT) -> tuple[str, bool]:
    """Execute *code* with *test_input* as stdin.

    Returns (stdout_stripped, success).  success is False on timeout,
    non-zero exit code, or any exception.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(code)
        tmpfile = fh.name

    try:
        result = subprocess.run(
            ["python3", tmpfile],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", False
    except Exception:
        return "", False
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


def _passes_all_tests(code: str, test_cases: list[dict]) -> bool:
    """Return True if *code* produces the expected output for every test case."""
    if not test_cases:
        return False
    for tc in test_cases:
        expected = tc.get("output", tc.get("expected_output", "")).strip()
        inp = tc.get("input", "")
        actual, ok = _run_code(code, inp)
        if not ok or actual != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# Modal function: load dataset
# ---------------------------------------------------------------------------


@app.function(image=image)
def fetch_items() -> list[dict]:
    """Download LiveCodeBench and return v6 problems with public test cases."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, version_tag=VERSION_TAG, split="test", trust_remote_code=True)
    items = []
    for row in ds:
        raw_tests = row.get("public_test_cases", "[]")
        try:
            test_cases = json.loads(raw_tests) if isinstance(raw_tests, str) else raw_tests
        except (json.JSONDecodeError, TypeError):
            test_cases = []
        if not test_cases:
            continue
        items.append(
            {
                "id": str(row["question_id"]),
                "difficulty": str(row.get("difficulty", "unknown")),
                "platform": str(row.get("platform", "unknown")),
                "prompt": _build_solver_prompt(
                    row["question_content"], row.get("starter_code", "")
                ),
                "test_cases": test_cases,
            }
        )

    items.sort(key=lambda x: x["id"])
    if TEST_LIMIT is not None:
        items = items[:TEST_LIMIT]
    print(f"Loaded {len(items)} v6 items")
    return items


# ---------------------------------------------------------------------------
# Modal function: score one item
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ],
    max_containers=20,
    retries=2,
    timeout=300,
)
def score_item(item: dict) -> dict:
    """Generate a solution and test it against public test cases."""
    from llm_client import query_model

    model = item["model"]
    raw = query_model(
        model,
        item["prompt"],
        max_tokens=2048,
        max_attempts=3,
    )
    code = _extract_code(raw)
    correct = int(_passes_all_tests(code, item["test_cases"]))

    return {
        "id": item["id"],
        "difficulty": item["difficulty"],
        "platform": item["platform"],
        "correct": correct,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(model: str = DEFAULT_MODEL) -> None:
    out_path, out_jsonl = _out_paths(model)
    print(f"Model: {model}  (slug: {_model_slug(model)})")

    items = fetch_items.remote()
    for item in items:
        item["model"] = model
    n = len(items)
    print(f"Scoring {n} problems")
    if n == 0:
        print("No items matched the v6 date range — check dataset split/columns.")
        return

    results = list(score_item.map(items, order_outputs=True))

    responses = np.array([r["correct"] for r in results], dtype=np.int8)
    response_matrix = responses.reshape(1, -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        response_matrix=response_matrix,
        item_ids=np.array([r["id"] for r in results]),
        difficulties=np.array([r["difficulty"] for r in results]),
        platforms=np.array([r["platform"] for r in results]),
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "difficulty": r["difficulty"],
                        "platform": r["platform"],
                        "correct": r["correct"],
                        "raw": r["raw"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"].lower() == diff]
        if subset:
            acc = np.mean([r["correct"] for r in subset])
            print(f"{diff:<8}: {acc:.3f}  (n={len(subset)})")

    print(f"\nOverall  : {responses.mean():.3f}  ({int(responses.sum())}/{n})")
    print(f"\nSaved → {out_path}")
    print(f"Saved → {out_jsonl}")
