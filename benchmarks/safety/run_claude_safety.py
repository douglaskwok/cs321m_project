"""Run Claude on SafetySuite prompts with Modal.

This script reads ``benchmarks/safety/safety_solver.json`` locally, resumes from
the model-specific output JSON when present, sends the next unanswered prompts to
Claude from Modal, and writes model responses to JSONL plus an updated JSON copy.

One-time Modal secret setup::

    modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...

Run all remaining prompts with Haiku 4.5::

    modal run benchmarks/safety/run_claude_safety.py

Smoke test on one prompt with Haiku 4.5::

    modal run benchmarks/safety/run_claude_safety.py --limit 1

Choose a model alias or full Anthropic model id::

    modal run benchmarks/safety/run_claude_safety.py --model haiku-4.5 --limit 1
    modal run benchmarks/safety/run_claude_safety.py --model sonnet-4.6 --limit 10
    modal run benchmarks/safety/run_claude_safety.py --model opus-4.7 --limit 10
    modal run benchmarks/safety/run_claude_safety.py --model claude-haiku-4-5-20251001 --limit 1
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "anthropic>=0.25",
    )
    .add_local_file(
        Path(__file__).parent.parent / "llm_client.py",
        remote_path="/root/llm_client.py",
    )
)

app = modal.App("safety-claude", image=image)


DEFAULT_INPUT_PATH = Path(__file__).parent / "safety_solver.json"
DEFAULT_MODEL = "haiku-4.5"
DEFAULT_LIMIT = -1
DEFAULT_TARGET_MODEL_NAME = ""

MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "haiku-4.5": "claude-haiku-4-5-20251001",
    "haiku45": "claude-haiku-4-5-20251001",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet46": "claude-sonnet-4-6",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
    "opus-4.7": "claude-opus-4-7",
    "opus47": "claude-opus-4-7",
    "claude-opus-4.7": "claude-opus-4-7",
}


def _model_id(model: str) -> str:
    """Resolve friendly model aliases to Anthropic API model ids."""
    model = model.strip()
    return MODEL_ALIASES.get(model.lower(), model)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _default_output_paths(model_id: str) -> tuple[Path, Path]:
    out_dir = Path(__file__).parent / "results"
    slug = _slug(model_id)
    return (
        out_dir / f"safety_solver_{slug}_responses.jsonl",
        out_dir / f"safety_solver_{slug}.json",
    )


def _default_target_model_name(model_id: str) -> str:
    if model_id.lower().startswith("claude-"):
        return "Claude"
    return model_id


def _retarget_prompt(prompt: str, target_model_name: str) -> str:
    if not target_model_name:
        return prompt
    return re.sub(
        r"\bvicuna(?:\s*[-_ ]\s*\d+b)?(?:\s*[-_ ]\s*v?\d+(?:[._]\d+)*)?\b",
        target_model_name,
        prompt,
        flags=re.IGNORECASE,
    )


def _retarget_items(items: list[dict], target_model_name: str) -> None:
    for item in items:
        prompt = str(item.get("harmful_prompt", ""))
        item["harmful_prompt"] = _retarget_prompt(prompt, target_model_name)


def _read_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        items = json.load(fh)
    if not isinstance(items, list):
        raise TypeError(f"Expected {path} to contain a JSON list")
    return items


def _load_all_items(input_path: Path, resume_path: Path) -> tuple[list[dict], Path]:
    """Load previous output when available, otherwise load the base input."""
    if resume_path.exists():
        return _read_json_list(resume_path), resume_path
    return _read_json_list(input_path), input_path


def _pending_items(items: list[dict], limit: int | None) -> list[dict]:
    # Only run non-empty prompts that do not already have a response.
    pending = [
        {**item, "_input_index": idx}
        for idx, item in enumerate(items)
        if str(item.get("harmful_prompt", "")).strip()
        and not str(item.get("response", "")).strip()
        and not str(item.get("error", "")).strip()
    ]
    if limit is not None and limit >= 0:
        pending = pending[:limit]
    return pending


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("anthropic-secret")],
    retries=2,
    timeout=300,
)
def run_prompt(item: dict) -> dict:
    """Query Claude for one safety prompt."""
    from llm_client import query_model

    model_id = item["model_id"]
    prompt = item["harmful_prompt"]
    error = ""
    try:
        response = query_model(
            model_id,
            prompt,
            max_tokens=item["max_tokens"],
            temperature=item["temperature"],
            max_attempts=3,
        )
    except Exception as exc:
        response = ""
        error = f"{type(exc).__name__}: {exc}"

    return {
        "input_index": item["_input_index"],
        "harmful_prompt": prompt,
        "source": item.get("source", ""),
        "model": model_id,
        "response": response,
        "error": error,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    limit: int = DEFAULT_LIMIT,
    input_path: str = str(DEFAULT_INPUT_PATH),
    output_jsonl: str = "",
    output_json: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    target_model_name: str = DEFAULT_TARGET_MODEL_NAME,
) -> None:
    model_id = _model_id(model)
    input_file = Path(input_path)
    resolved_target_model_name = target_model_name.strip() or _default_target_model_name(model_id)
    out_jsonl, out_json = _default_output_paths(model_id)
    if output_jsonl:
        out_jsonl = Path(output_jsonl)
    if output_json:
        out_json = Path(output_json)

    all_items, loaded_from = _load_all_items(input_file, out_json)
    _retarget_items(all_items, resolved_target_model_name)
    items = _pending_items(all_items, limit=limit)

    print(f"Model alias: {model}")
    print(f"Model id   : {model_id}")
    print(f"Base input : {input_file}")
    print(f"Loaded from: {loaded_from}")
    print(f"Output JSON: {out_json}")
    print(f"Limit      : {limit}")
    print(f"Retarget   : Vicuna -> {resolved_target_model_name}")
    if items:
        print(f"First pending index: {items[0]['_input_index']}")
    for item in items:
        item["model_id"] = model_id
        item["max_tokens"] = max_tokens
        item["temperature"] = temperature

    if not items:
        print("No pending items to run.")
        return

    print(f"Running {len(items)} prompt(s)...")
    results = list(run_prompt.map(items, order_outputs=True))

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    for result in results:
        idx = result["input_index"]
        all_items[idx]["response"] = result["response"]
        all_items[idx]["model"] = result["model"]
        all_items[idx]["created_at_utc"] = result["created_at_utc"]
        if result.get("error"):
            all_items[idx]["error"] = result["error"]
        else:
            all_items[idx].pop("error", None)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(all_items, fh, ensure_ascii=False, indent=2)

    print(f"Saved JSONL responses -> {out_jsonl}")
    print(f"Saved updated JSON     -> {out_json}")
