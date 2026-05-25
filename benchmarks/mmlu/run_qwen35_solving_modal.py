import json
import os
import re
from pathlib import Path

import modal


DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
MODAL_GPU = os.environ.get("MODAL_GPU", "B200+")
DATASET_ID = "ScalerLab/JudgeBench"
MMLU_PRO_ID = "TIGER-Lab/MMLU-Pro"
OUT_DIR = Path("qwen35_solving_outputs")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=0.34.0",
        "datasets>=2.20.0",
        "mistral-common>=1.8.6",
        "pandas>=2.2.0",
        "torch>=2.4.0",
        "transformers>=4.56.0",
        "huggingface_hub[hf_transfer]>=0.24.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("qwen35-08b-judgebench-solving")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


def extract_final_letter(text: str) -> str | None:
    """Extract the model's final multiple-choice answer letter."""
    if not text:
        return None

    text = re.sub(r"</s>|<eos>|<\\|endoftext\\|>", "", text, flags=re.IGNORECASE)
    bare = text.upper().strip()
    bare_letter = re.fullmatch(r"\(?([A-K])\)?\.?", bare)
    if bare_letter:
        return bare_letter.group(1)

    # JudgeBench/MMLU-Pro prompts ask for duplicated final letters, e.g. CCCCC.
    repeated = re.findall(r"\b([A-K])\1{2,}\b", text.upper())
    if repeated:
        return repeated[-1]

    # Fallbacks for models that ignore the duplicate-letter instruction.
    patterns = [
        r"(?:ANSWER|FINAL ANSWER)\s*(?:IS|:)?\s*\(?([A-K])\)?",
        r"\bOPTION\s+([A-K])\b",
        r"\(([A-K])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        if matches:
            return matches[-1]
    return None


def winning_response(row: dict) -> str:
    if row["label"] == "A>B":
        return row["response_A"]
    if row["label"] == "B>A":
        return row["response_B"]
    raise ValueError(f"Unexpected label: {row['label']}")


def make_solving_prompt(question: str) -> str:
    return question


def make_mmlu_pro_prompt(row: dict) -> str:
    options = "\n".join(
        f"({chr(65 + i)}) {option}" for i, option in enumerate(row["options"])
    )
    return (
        "Answer the following multiple-choice question. "
        "Return only the correct option letter.\n\n"
        f"{row['question']}\n{options}"
    )


def strip_judgebench_suffix(question: str) -> str:
    question = question.replace(
        "If you cannot determine the correct multiple-choice answer, take your best guess.",
        "",
    )
    question = re.sub(
        r"Once you have your answer, please duplicate that letter five times in a single string\."
        r"\s*For example, if the answer is K, then write KKKKK\."
        r"\s*Let's think step by step\.?",
        "",
        question,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", question).strip()


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=3 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_split(
    split: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 16,
    limit: int | None = None,
    prompt_style: str = "mmlu_pro",
) -> list[dict]:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if split not in {"gpt", "claude"}:
        raise ValueError("split must be 'gpt' or 'claude'")

    is_mistral3 = "Ministral-3" in model_id
    if is_mistral3:
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

        tokenizer = MistralCommonBackend.from_pretrained(model_id)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    ds = load_dataset(DATASET_ID, split=split)
    mmlu_pro = load_dataset(MMLU_PRO_ID, split="test")
    mmlu_by_id = {int(row["question_id"]): dict(row) for row in mmlu_pro}
    rows = [dict(r) for r in ds if str(r["source"]).startswith("mmlu-pro")]
    if limit is not None:
        rows = rows[:limit]

    results = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        print(f"{split}: solving {index}/{total} original_id={row['original_id']}", flush=True)
        mmlu_row = mmlu_by_id[int(row["original_id"])]
        if prompt_style == "judgebench":
            prompt = make_solving_prompt(row["question"])
        elif prompt_style == "mmlu_pro":
            prompt = make_mmlu_pro_prompt(mmlu_row)
        else:
            raise ValueError("prompt_style must be 'mmlu_pro' or 'judgebench'")
        if is_mistral3:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )[0]

            completion_ids = generated[inputs["input_ids"].shape[-1] :]
            completion = tokenizer.decode(completion_ids)
        else:
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            completion_ids = generated[0][inputs["input_ids"].shape[-1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        gold_letter = mmlu_row["answer"]
        judgebench_winner_letter = extract_final_letter(winning_response(row))
        pred_letter = extract_final_letter(completion)

        results.append(
            {
                "split": split,
                "pair_id": row["pair_id"],
                "original_id": row["original_id"],
                "source": row["source"],
                "response_model": row["response_model"],
                "label": row["label"],
                "gold_letter": gold_letter,
                "judgebench_winner_letter": judgebench_winner_letter,
                "judgebench_winner_matches_gold": judgebench_winner_letter == gold_letter,
                "pred_letter": pred_letter,
                "correct": pred_letter == gold_letter if pred_letter and gold_letter else None,
                "question": row["question"],
                "prompt_style": prompt_style,
                "model_id": model_id,
                "model_prompt": prompt,
                "normalized_question_for_matching": strip_judgebench_suffix(row["question"]),
                "completion": completion,
            }
        )

    return results


@app.local_entrypoint()
def main(
    split: str = "both",
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 16,
    limit: int | None = None,
    prompt_style: str = "mmlu_pro",
):
    OUT_DIR.mkdir(exist_ok=True)
    splits = ["gpt", "claude"] if split == "both" else [split]

    for current_split in splits:
        results = run_split.remote(
            current_split,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            limit=limit,
            prompt_style=prompt_style,
        )
        model_slug = model_id.split("/")[-1].lower().replace(".", "_").replace("-", "_")
        out_path = OUT_DIR / f"{model_slug}_solving_{current_split}_{prompt_style}.jsonl"
        if limit is not None:
            out_path = OUT_DIR / f"{model_slug}_solving_{current_split}_{prompt_style}_limit{limit}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        scored = [r for r in results if r["correct"] is not None]
        acc = sum(r["correct"] for r in scored) / len(scored) if scored else float("nan")
        print(f"{current_split}: wrote {len(results)} rows to {out_path}")
        print(f"{current_split}: parsed {len(scored)}/{len(results)} predictions, accuracy={acc:.3f}")
