from pathlib import Path
import subprocess
import sys

import modal


REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "HarmBench").exists():
    raise RuntimeError(
        "Run this script from the cs321m_project repo root so HarmBench can be mounted."
    )

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm==0.10.2",
        "transformers==4.57.3",
        "accelerate",
        "sentencepiece",
        "pandas",
        "pyyaml",
        "tqdm",
        "ray",
    )
    .add_local_dir(REPO_ROOT / "HarmBench", remote_path="/root/HarmBench")
    .add_local_file(
        REPO_ROOT / "benchmarks" / "safety" / "safety_solver_behaviors.csv",
        remote_path="/root/benchmarks/safety/safety_solver_behaviors.csv",
    )
)

app = modal.App("harmbench-attack-generation-heavy", image=image)
attack_cases_volume = modal.Volume.from_name("harmbench-attack-cases", create_if_missing=True)

METHODS = {
    "pap": ("PAP", "top_5", "PAP-top5", "A100"),
    "gcg": ("GCG", "vicuna_7b_v1_5", "GCG-vicuna_7b_v1_5", "A100"),
    "autodan": ("AutoDAN", "vicuna_7b_v1_5", "AutoDAN-vicuna_7b_v1_5", "A100"),
}


@app.function(
    gpu="A100",
    timeout=24 * 60 * 60,
    volumes={"/outputs": attack_cases_volume},
)
def generate_attack(method_key: str):
    method_name, experiment_name, out_name, _gpu = METHODS[method_key]
    save_dir = f"/outputs/{out_name}"

    cmd = [
        "python",
        "generate_test_cases.py",
        "--method_name", method_name,
        "--experiment_name", experiment_name,
        "--behaviors_path", "/root/benchmarks/safety/safety_solver_behaviors.csv",
        "--save_dir", save_dir,
        "--overwrite",
    ]

    result = subprocess.run(
        cmd,
        cwd="/root/HarmBench",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout)
    sys.stdout.flush()
    if result.returncode != 0:
        raise RuntimeError(
            f"HarmBench exited with status {result.returncode}. "
            "See subprocess output above."
        )
    attack_cases_volume.commit()
    return save_dir


@app.local_entrypoint()
def main(method: str = "pap"):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}. Choose from: {sorted(METHODS)}")
    save_dir = generate_attack.remote(method)
    output_name = Path(save_dir).name
    print(f"Saved to Modal volume harmbench-attack-cases: {save_dir}")
    print("Download with:")
    print(
        "  modal volume get harmbench-attack-cases "
        f"/{output_name} benchmarks/safety/attack_cases/{output_name}"
    )
