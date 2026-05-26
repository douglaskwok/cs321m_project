import json
import subprocess
from pathlib import Path

import modal


REPO_DIR = Path(__file__).resolve().parent
REMOTE_REPO_DIR = "/root/HarmMetric_Eval"
SAFETY_SOLVER_PATH = Path(
    r"C:\Users\tanji\Documents\Codex\2026-05-24\help-me-pull-this-github-repo\cs321m_project\benchmarks\safety\safety_solver.json"
)
REMOTE_SAFETY_SOLVER_PATH = f"{REMOTE_REPO_DIR}/data/safety_solver.json"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("anthropic>=0.75.0", "pandas==2.2.3")
    .add_local_dir(REPO_DIR, REMOTE_REPO_DIR, copy=True)
    .add_local_file(SAFETY_SOLVER_PATH, REMOTE_SAFETY_SOLVER_PATH, copy=True)
)

app = modal.App("harmmetric-eval-claude")


def safe_write_text(path, content):
    try:
        path.write_text(content, encoding="utf-8")
        return path
    except PermissionError:
        fallback = path.with_name(path.name + ".new")
        fallback.write_text(content, encoding="utf-8")
        return fallback


@app.function(
    image=image,
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("anthropic-secret", required_keys=["ANTHROPIC_API_KEY"])],
)
def run_claude_harmmetric(
    model_id,
    model_label,
    prompt_limit=0,
    response_ids="",
    output_suffix="",
    skip_exports=False,
):
    command = [
        "python",
        "run_claude_harmmetric.py",
        "--model-id",
        model_id,
        "--model-label",
        model_label,
    ]
    if prompt_limit:
        command.extend(["--prompt-limit", str(prompt_limit)])
    if response_ids:
        command.extend(["--response-ids", response_ids])
    if output_suffix:
        command.extend(["--output-suffix", output_suffix])
    if skip_exports:
        command.append("--skip-exports")

    completed = subprocess.run(
        command,
        cwd=REMOTE_REPO_DIR,
        text=True,
        capture_output=True,
        check=False,
        env={"SAFETY_SOLVER_PATH": REMOTE_SAFETY_SOLVER_PATH, **__import__("os").environ},
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

    suffix = "safety_solver" if not prompt_limit else f"safety_solver_smoke{prompt_limit}"
    if output_suffix:
        suffix = f"{suffix}_{output_suffix}"
    metric_stem = f"{model_label}-HarmJudge-{suffix}"
    repo = Path(REMOTE_REPO_DIR)
    files = {
        "aggregate_csv": repo / "results" / f"{metric_stem}.csv",
        "prompt_scores_csv": repo / "results" / f"{metric_stem}_prompt_scores.csv",
        "irt_items_csv": repo / "results" / f"{metric_stem}_irt_items.csv",
        "temp_jsonl": repo / "temp_results" / f"{metric_stem}.jsonl",
    }
    return {
        "metric_stem": metric_stem,
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-4000:],
        "files": {name: path.read_text(encoding="utf-8") for name, path in files.items() if path.exists()},
    }


@app.local_entrypoint()
def main(
    model_id: str,
    model_label: str,
    prompt_limit: int = 0,
    response_ids: str = "",
    output_suffix: str = "",
    skip_exports: bool = False,
):
    result = run_claude_harmmetric.remote(
        model_id=model_id,
        model_label=model_label,
        prompt_limit=prompt_limit,
        response_ids=response_ids,
        output_suffix=output_suffix,
        skip_exports=skip_exports,
    )

    results_dir = REPO_DIR / "results"
    temp_dir = REPO_DIR / "temp_results"
    results_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)

    metric_stem = result["metric_stem"]
    file_targets = {
        "aggregate_csv": results_dir / f"{metric_stem}.csv",
        "prompt_scores_csv": results_dir / f"{metric_stem}_prompt_scores.csv",
        "irt_items_csv": results_dir / f"{metric_stem}_irt_items.csv",
        "temp_jsonl": temp_dir / f"{metric_stem}.jsonl",
    }
    written_paths = []
    for name, content in result["files"].items():
        written_paths.append(safe_write_text(file_targets[name], content))

    modal_log_path = safe_write_text(
        results_dir / f"{metric_stem}_modal_run.json",
        json.dumps(
            {
                "metric_stem": metric_stem,
                "model_id": model_id,
                "stdout_tail": result["stdout_tail"],
                "stderr_tail": result["stderr_tail"],
            },
            indent=2,
        ),
    )
    written_paths.append(modal_log_path)

    print(result["files"].get("aggregate_csv", ""))
    for path in written_paths:
        print(f"Wrote {path}")
