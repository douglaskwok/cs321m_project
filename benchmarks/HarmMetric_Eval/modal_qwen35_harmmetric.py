import json
import subprocess
from pathlib import Path

import modal


REPO_DIR = Path(__file__).resolve().parent
REMOTE_REPO_DIR = "/root/HarmMetric_Eval"
METRIC_NAME = "Qwen3.5-0.8B-HarmJudge"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
SAFETY_SOLVER_PATH = Path(
    r"C:\Users\tanji\Documents\Codex\2026-05-24\help-me-pull-this-github-repo\cs321m_project\benchmarks\safety\safety_solver.json"
)
REMOTE_SAFETY_SOLVER_PATH = f"{REMOTE_REPO_DIR}/data/safety_solver.json"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "accelerate==1.1.0",
        "mistral-common>=1.8.6",
        "pandas==2.2.3",
        "pillow>=11.0.0",
        "qwen-vl-utils>=0.0.10",
        "sentencepiece>=0.2.0",
        "torch==2.9.0",
        "torchvision>=0.24.0",
        "git+https://github.com/huggingface/transformers.git",
    )
    .add_local_dir(REPO_DIR, REMOTE_REPO_DIR, copy=True)
    .add_local_file(SAFETY_SOLVER_PATH, REMOTE_SAFETY_SOLVER_PATH, copy=True)
)

app = modal.App("harmmetric-eval-qwen35-08b")
hf_cache = modal.Volume.from_name("harmmetric-qwen35-hf-cache", create_if_missing=True)


def run_command(command, cwd):
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + completed.stdout
            + "\n\nSTDERR:\n"
            + completed.stderr
        )
    return completed.stdout + completed.stderr


def run_eval_command(command, cwd):
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env=None,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 6,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_qwen35_harmmetric(
    limit=0,
    safety_solver_only=False,
    model_id=DEFAULT_MODEL_ID,
    model_label="Qwen3.5-0.8B",
    prompt_limit=0,
):
    repo = Path(REMOTE_REPO_DIR)
    metric_name = f"{model_label}-HarmJudge"
    output_metric_name = metric_name
    metric_args = ["--metrics", METRIC_NAME]
    if safety_solver_only:
        output_metric_name = f"{metric_name}-safety_solver"
        if prompt_limit:
            output_metric_name = f"{output_metric_name}_smoke{prompt_limit}"
        metric_args.extend(
            [
                "--prompt-file",
                REMOTE_SAFETY_SOLVER_PATH,
                "--output-name",
                output_metric_name,
            ]
        )
        if prompt_limit:
            metric_args.extend(["--prompt-limit", str(prompt_limit)])
    if limit:
        metric_args.extend(["--limit", str(limit)])

    import os

    env = os.environ.copy()
    env["QWEN35_HARMJUDGE_MODEL_ID"] = model_id
    completed = subprocess.run(
        ["python", "eval_with_metrics.py", *metric_args],
        cwd=repo / "metrics_codes",
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    eval_result = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    eval_log = eval_result["stdout"] + eval_result["stderr"]
    if eval_result["returncode"] != 0:
        raise RuntimeError(
            "Command failed:\npython eval_with_metrics.py "
            + " ".join(metric_args)
            + "\n\nSTDOUT:\n"
            + eval_result["stdout"]
            + "\n\nSTDERR:\n"
            + eval_result["stderr"]
        )
    score_result = run_eval_command(
        ["python", "scoring.py", "--metric", output_metric_name],
        cwd=repo / "benchmark_codes",
    )
    score_log = score_result["stdout"] + score_result["stderr"]

    result_csv = repo / "results" / f"{output_metric_name}.csv"
    temp_jsonl = repo / "temp_results" / f"{output_metric_name}.jsonl"
    return {
        "metric": output_metric_name,
        "model_id": model_id,
        "scoring_returncode": score_result["returncode"],
        "result_csv": result_csv.read_text(encoding="utf-8") if result_csv.exists() else "",
        "temp_jsonl": temp_jsonl.read_text(encoding="utf-8") if temp_jsonl.exists() else "",
        "eval_log_tail": eval_log[-8000:],
        "score_log_tail": score_log[-4000:],
    }


@app.local_entrypoint()
def main(
    limit: int = 0,
    safety_solver_only: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    model_label: str = "Qwen3.5-0.8B",
    prompt_limit: int = 0,
):
    result = run_qwen35_harmmetric.remote(
        limit=limit,
        safety_solver_only=safety_solver_only,
        model_id=model_id,
        model_label=model_label,
        prompt_limit=prompt_limit,
    )

    results_dir = REPO_DIR / "results"
    temp_dir = REPO_DIR / "temp_results"
    results_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)

    output_metric_name = result["metric"]
    (results_dir / f"{output_metric_name}.csv").write_text(result["result_csv"], encoding="utf-8")
    (temp_dir / f"{output_metric_name}.jsonl").write_text(result["temp_jsonl"], encoding="utf-8")
    (results_dir / f"{output_metric_name}_modal_run.json").write_text(
        json.dumps(
            {
                "metric": result["metric"],
                "model_id": result["model_id"],
                "scoring_returncode": result["scoring_returncode"],
                "eval_log_tail": result["eval_log_tail"],
                "score_log_tail": result["score_log_tail"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(result["result_csv"])
    if result["scoring_returncode"] != 0:
        print(result["score_log_tail"])
    print(f"Wrote {results_dir / (output_metric_name + '.csv')}")
    print(f"Wrote {temp_dir / (output_metric_name + '.jsonl')}")
