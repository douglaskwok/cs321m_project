from pathlib import Path
import subprocess
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
        "torch>=2.4",
        "transformers==4.57.3",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "pandas",
        "pyyaml",
        "tqdm",
        "fschat",
        "ray",
        "vllm",
    )
    .add_local_dir(REPO_ROOT / "HarmBench", remote_path="/root/HarmBench")
    .add_local_file(
        REPO_ROOT / "benchmarks" / "safety" / "safety_solver_behaviors.csv",
        remote_path="/root/benchmarks/safety/safety_solver_behaviors.csv",
    )
)

app = modal.App("harmbench-attack-generation", image=image)

METHODS = {
    "direct": ("DirectRequest", "default", "DirectRequest", "A10G"),
    "human": ("HumanJailbreaks", "random_subset_5", "HumanJailbreaks", "A10G"),
    "pap": ("PAP", "top_5", "PAP-top5", "A100"),
    "gcg": ("GCG", "vicuna_7b_v1_5", "GCG-vicuna_7b_v1_5", "A100"),
    "autodan": ("AutoDAN", "vicuna_7b_v1_5", "AutoDAN-vicuna_7b_v1_5", "A100"),
    "pair": ("PAIR", "vicuna_7b_v1_5", "PAIR-vicuna_7b_v1_5", "A100"),
    "gptfuzz": ("GPTFuzz", "vicuna_7b_v1_5", "GPTFuzz-vicuna_7b_v1_5", "A100"),
}


@app.function(
    gpu="A100",
    timeout=24 * 60 * 60,
)
def generate_attack(method_key: str):
    method_name, experiment_name, out_name, _gpu = METHODS[method_key]
    save_dir = f"/root/attack_cases/{out_name}"

    cmd = [
        "python",
        "generate_test_cases.py",
        "--method_name", method_name,
        "--experiment_name", experiment_name,
        "--behaviors_path", "/root/benchmarks/safety/safety_solver_behaviors.csv",
        "--save_dir", save_dir,
        "--overwrite",
    ]

    subprocess.run(cmd, cwd="/root/HarmBench", check=True)
    return save_dir


@app.local_entrypoint()
def main(method: str = "human"):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}. Choose from: {sorted(METHODS)}")
    save_dir = generate_attack.remote(method)
    print(f"Saved inside Modal container: {save_dir}")
