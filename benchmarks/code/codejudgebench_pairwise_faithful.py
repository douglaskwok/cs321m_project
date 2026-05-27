"""CodeJudgeBench CodeGen pairwise judge runner.

This is a cost-conscious CodeJudgeBench CodeGen runner:

* uses all CodeGen dataset rows by default;
* randomly samples exactly one ordering per row, forward or backward;
* batches multiple judgments inside each Modal function call;
* runs Hugging Face models on a B200 worker;
* appends JSONL results and writes an NPZ checkpoint after every completed
  batch, so interrupted runs can resume without redoing completed judgments.

Examples:

    modal run benchmarks/code/codejudgebench_pairwise_faithful.py --model qwen3.5-4b --limit 1
    modal run benchmarks/code/codejudgebench_pairwise_faithful.py --model mistral-3b --split gemini_2.5_pro --limit 4
    modal run benchmarks/code/codejudgebench_pairwise_faithful.py --model qwen3.5-4b --batch-size 8 --seed 123
    modal run benchmarks/code/codejudgebench_pairwise_faithful.py --model qwen3.5-4b --prompt-style verdict --max-tokens 64
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import modal
import numpy as np


# ---------------------------------------------------------------------------
# Modal image and app
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

hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers==5.9.0",
        "accelerate>=1.0",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
        "torchvision",
        "pillow",
        "mistral-common>=1.8.6",
        "kernels",
        "numpy>=1.24",
    )
    .add_local_file(
        Path(__file__).parent.parent / "llm_client.py",
        remote_path="/root/llm_client.py",
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("codejudgebench-codegen-sampled", image=image)


# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

DATASET_ID = "mattymchen/codejudgebench"
CODEGEN_CONFIG = "codegen"
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_SEED = 123
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_TOKENS = 4096
DEFAULT_PROMPT_STYLE = "official"
DEFAULT_PREFILL_VERDICT = True


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


def _run_slug(
    model: str,
    split: str,
    limit: int | None,
    seed: int,
    max_tokens: int,
    prompt_style: str,
    prefill_verdict: bool,
) -> str:
    slug = _model_slug(model)
    if split != "all":
        slug += f"_{_model_slug(split)}"
    if limit is not None:
        slug += f"_smoke{limit}"
    slug += f"_oneorder_seed{seed}_{_model_slug(prompt_style)}_tok{max_tokens}"
    if prompt_style == "verdict" and not prefill_verdict:
        slug += "_noprefill"
    return slug


def _out_paths(
    model: str,
    split: str,
    limit: int | None,
    seed: int,
    max_tokens: int,
    prompt_style: str,
    prefill_verdict: bool,
) -> tuple[Path, Path]:
    slug = _run_slug(
        model,
        split,
        limit,
        seed,
        max_tokens,
        prompt_style,
        prefill_verdict,
    )
    base = Path(__file__).parent / "results"
    return (
        base / f"codejudgebench_pairwise_{slug}.npz",
        base / f"codejudgebench_pairwise_{slug}_responses.jsonl",
    )


# ---------------------------------------------------------------------------
# Official-style pairwise judge prompt and parser
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is better: [[A>B]]
2. Assistant B is better: [[B>A]]

Example output: "My final verdict is Assistant A is better: [[A>B]]".

<|User Prompt|>
{query}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>
""".strip()

_VERDICT_ONLY_PROMPT = """
You are judging two candidate code solutions for the same programming problem.

Choose the solution that is more likely to be correct and pass all tests.

Output exactly one label and no other text:
[[A>B]] if Assistant A is better.
[[B>A]] if Assistant B is better.

<|User Prompt|>
{query}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>
""".strip()


def _build_judge_messages(
    problem: str,
    solution_a: str,
    solution_b: str,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    prefill_verdict: bool = DEFAULT_PREFILL_VERDICT,
) -> list[dict]:
    prompt_template = _VERDICT_ONLY_PROMPT if prompt_style == "verdict" else _JUDGE_PROMPT
    messages = [
        {
            "role": "user",
            "content": prompt_template.format(
                query=problem,
                answer_a=solution_a,
                answer_b=solution_b,
            ),
        }
    ]
    if prompt_style == "verdict" and prefill_verdict:
        messages.append({"role": "assistant", "content": "[["})
    return messages


def _parse_winner(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    pred_a = max(text.rfind(p) for p in ("[[A>B]]", "boxed{A>B}"))
    pred_b = max(text.rfind(p) for p in ("[[B>A]]", "boxed{B>A}"))
    if pred_a > pred_b:
        return "A"
    if pred_b > pred_a:
        return "B"
    return None


# ---------------------------------------------------------------------------
# Dataset loading and one-order sampling
# ---------------------------------------------------------------------------


@app.function(image=image)
def fetch_items(
    split: str = "all",
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    prefill_verdict: bool = DEFAULT_PREFILL_VERDICT,
) -> list[dict]:
    """Download CodeJudgeBench CodeGen and sample one ordering per row."""
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset(DATASET_ID, CODEGEN_CONFIG, trust_remote_code=True)
    split_names = list(ds.keys()) if split == "all" else [split]
    missing = [s for s in split_names if s not in ds]
    if missing:
        raise ValueError(f"Unknown split(s): {missing}; available={list(ds.keys())}")

    print(f"Splits: {split_names}")
    print(f"Dataset columns: {ds[split_names[0]].column_names}")

    items: list[dict] = []
    for split_name in split_names:
        split_ds = ds[split_name]
        rows = split_ds if limit is None else split_ds.select(range(min(limit, len(split_ds))))
        for row_index, row in enumerate(rows):
            qid = str(row.get("question_id", row.get("id", row_index)))
            problem = row.get("question_content", row.get("problem", row.get("prompt", "")))
            sol_correct = row.get(
                "pos_response",
                row.get(
                    "solution_correct",
                    row.get("solution_1", row.get("solution_a", row.get("chosen", ""))),
                ),
            )
            sol_incorrect = row.get(
                "neg_response",
                row.get(
                    "solution_incorrect",
                    row.get("solution_2", row.get("solution_b", row.get("rejected", ""))),
                ),
            )
            difficulty = str(row.get("difficulty", "unknown"))
            pair_id = f"{split_name}:{qid}:{row_index}"

            use_forward = rng.random() < 0.5
            ordering = "fwd" if use_forward else "bwd"
            gold = "A" if use_forward else "B"
            solution_a = sol_correct if use_forward else sol_incorrect
            solution_b = sol_incorrect if use_forward else sol_correct

            items.append(
                {
                    "item_id": f"{pair_id}:{ordering}",
                    "pair_id": pair_id,
                    "question_id": qid,
                    "split": split_name,
                    "ordering": ordering,
                    "difficulty": difficulty,
                    "judge_messages": _build_judge_messages(
                        problem,
                        solution_a,
                        solution_b,
                        prompt_style=prompt_style,
                        prefill_verdict=prefill_verdict,
                    ),
                    "gold": gold,
                }
            )

    n_fwd = sum(1 for item in items if item["ordering"] == "fwd")
    n_bwd = len(items) - n_fwd
    print(f"Built {len(items)} sampled judgment items")
    print(f"Ordering sample: fwd={n_fwd}, bwd={n_bwd}, seed={seed}")
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_item_impl(item: dict) -> dict:
    from llm_client import query_model

    prefill = ""
    messages = item["judge_messages"]
    if messages and messages[-1]["role"] == "assistant":
        prefill = messages[-1]["content"]

    raw = query_model(
        item["model"],
        messages=messages,
        max_tokens=item.get("max_tokens", DEFAULT_MAX_TOKENS),
    )
    full_text = prefill + raw
    predicted = _parse_winner(full_text)
    correct = int(predicted == item["gold"]) if predicted is not None else -1

    return {
        "item_id": item["item_id"],
        "pair_id": item["pair_id"],
        "question_id": item["question_id"],
        "split": item["split"],
        "ordering": item["ordering"],
        "difficulty": item["difficulty"],
        "gold": item["gold"],
        "predicted": predicted or "",
        "correct": correct,
        "raw": full_text,
    }


def _score_batch_impl(items: list[dict]) -> list[dict]:
    return [_score_item_impl(item) for item in items]


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("anthropic-secret")],
    max_containers=5,
    retries=2,
    timeout=1800,
)
def score_batch_api(items: list[dict]) -> list[dict]:
    """Score a batch with an API model."""
    return _score_batch_impl(items)


@app.function(
    image=hf_image,
    gpu="B200",
    volumes={"/root/.cache/huggingface": hf_cache},
    max_containers=1,
    retries=2,
    timeout=7200,
)
def score_batch_hf_b200(items: list[dict]) -> list[dict]:
    """Score a batch with a local HF model on a single B200 worker."""
    return _score_batch_impl(items)


# ---------------------------------------------------------------------------
# Local checkpointing and entrypoint
# ---------------------------------------------------------------------------


def _chunks(items: list[dict], batch_size: int) -> list[list[dict]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _jsonl_record(r: dict) -> dict:
    return {
        "item_id": r["item_id"],
        "pair_id": r["pair_id"],
        "question_id": r["question_id"],
        "split": r["split"],
        "ordering": r["ordering"],
        "difficulty": r["difficulty"],
        "gold": r["gold"],
        "predicted": r["predicted"],
        "correct": r["correct"],
        "raw": r["raw"],
    }


def _load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    results = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                results.append(json.loads(line))
    return results


def _write_npz(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    responses = np.array([r["correct"] for r in results], dtype=np.int8)
    np.savez(
        path,
        response_matrix=responses.reshape(1, -1),
        item_ids=np.array([r["item_id"] for r in results]),
        pair_ids=np.array([r["pair_id"] for r in results]),
        question_ids=np.array([r["question_id"] for r in results]),
        splits=np.array([r["split"] for r in results]),
        difficulties=np.array([r["difficulty"] for r in results]),
        gold=np.array([r["gold"] for r in results]),
        predicted=np.array([r["predicted"] for r in results]),
        orderings=np.array([r["ordering"] for r in results]),
    )


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    split: str = "all",
    limit: int = 0,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    prefill_verdict: bool = DEFAULT_PREFILL_VERDICT,
) -> None:
    smoke_limit = limit if limit > 0 else None
    batch_size = max(1, batch_size)
    max_tokens = max(1, max_tokens)
    prompt_style = prompt_style.lower().strip()
    if prompt_style not in {"official", "verdict"}:
        raise ValueError("prompt_style must be 'official' or 'verdict'")
    out_path, out_jsonl = _out_paths(
        model,
        split,
        smoke_limit,
        seed,
        max_tokens,
        prompt_style,
        prefill_verdict,
    )

    print(
        "Model: "
        f"{model} (slug: "
        f"{_run_slug(model, split, smoke_limit, seed, max_tokens, prompt_style, prefill_verdict)})"
    )
    print(f"Task: codegen  Split: {split}  Limit per split: {smoke_limit or 'none'}")
    print(f"Ordering: one randomly sampled order per row  Seed: {seed}")
    print(f"Prompt style: {prompt_style}")
    print(f"Verdict prefill: {prefill_verdict}")
    print(f"Batch size: {batch_size}")
    print(f"Max new tokens: {max_tokens}")

    items = fetch_items.remote(
        split=split,
        limit=smoke_limit,
        seed=seed,
        prompt_style=prompt_style,
        prefill_verdict=prefill_verdict,
    )
    for item in items:
        item["model"] = model
        item["max_tokens"] = max_tokens
    n = len(items)
    print(f"Loaded {n} sampled judgment items")
    if n == 0:
        print("No items loaded; check dataset id and split.")
        return

    resolved = re.sub(r"[^a-z0-9.\-]", "", model.lower())
    is_api = resolved.startswith(("gpt-", "o1", "o2", "o3", "o4", "chatgpt-", "claude-"))
    scorer = score_batch_api if is_api else score_batch_hf_b200
    print(f"Modal scorer: {'API batch' if is_api else 'HF batch on B200'}")

    existing_results = _load_existing_results(out_jsonl)
    completed = {r["item_id"] for r in existing_results}
    pending_items = [item for item in items if item["item_id"] not in completed]
    results = existing_results[:]
    print(f"Existing results: {len(existing_results)}  Pending: {len(pending_items)}")

    if existing_results:
        _write_npz(out_path, results)
    if not pending_items:
        print("Nothing to do; all sampled judgments already exist.")
        return

    batches = _chunks(pending_items, batch_size)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as fh:
        for batch_results in scorer.map(batches, order_outputs=True):
            for r in batch_results:
                results.append(r)
                fh.write(json.dumps(_jsonl_record(r), ensure_ascii=False) + "\n")
            fh.flush()
            _write_npz(out_path, results)
            print(f"Saved checkpoint: {len(results)}/{n} judgments")

    _write_npz(out_path, results)
    responses = np.array([r["correct"] for r in results], dtype=np.int8)

    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"].lower() == diff]
        if subset:
            acc = np.mean([r["correct"] == 1 for r in subset])
            print(f"{diff:<8}: {acc:.3f} (n={len(subset)} judgments)")

    n_correct = int(np.sum(responses == 1))
    n_parse_fail = int(np.sum(responses == -1))
    print(f"\nOverall  : {n_correct / len(results):.3f} ({n_correct}/{len(results)} judgments)")
    if n_parse_fail:
        print(f"Parse failures: {n_parse_fail}/{len(results)} judgments")
    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {out_jsonl}")
