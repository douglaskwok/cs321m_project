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
    .pip_install(
        "pyyaml",
        "tqdm",
        "numpy",
    )
    .add_local_dir(REPO_ROOT / "HarmBench", remote_path="/root/HarmBench")
    .add_local_file(
        REPO_ROOT / "benchmarks" / "safety" / "safety_solver_behaviors.csv",
        remote_path="/root/benchmarks/safety/safety_solver_behaviors.csv",
    )
)

app = modal.App("harmbench-attack-generation", image=image)
attack_cases_volume = modal.Volume.from_name("harmbench-attack-cases", create_if_missing=True)

METHODS = {
    "direct": ("DirectRequest", "default", "DirectRequest", "A10G"),
    "human": ("HumanJailbreaks", "random_subset_5", "HumanJailbreaks", "A10G"),
    "pap": ("PAP", "top_5", "PAP-top5", "A100"),
    "gcg": ("GCG", "vicuna_7b_v1_5", "GCG-vicuna_7b_v1_5", "A100"),
    "autodan": ("AutoDAN", "vicuna_7b_v1_5", "AutoDAN-vicuna_7b_v1_5", "A100"),
    "pair": ("PAIR", "vicuna_7b_v1_5", "PAIR-vicuna_7b_v1_5", "A100"),
    "gptfuzz": ("GPTFuzz", "vicuna_7b_v1_5", "GPTFuzz-vicuna_7b_v1_5", "A100"),
}

LIGHTWEIGHT_METHODS = {"direct", "human"}


@app.function(
    gpu="A100",
    timeout=24 * 60 * 60,
    volumes={"/outputs": attack_cases_volume},
)
def generate_attack(method_key: str):
    if method_key not in LIGHTWEIGHT_METHODS:
        raise ValueError(
            f"{method_key} needs the heavy model-generation image. "
            f"This lightweight runner supports: {sorted(LIGHTWEIGHT_METHODS)}"
        )

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

    subprocess.run(cmd, cwd="/root/HarmBench", check=True)
    attack_cases_volume.commit()
    return save_dir


@app.local_entrypoint()
def main(method: str = "human"):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}. Choose from: {sorted(METHODS)}")
    save_dir = generate_attack.remote(method)
    print(f"Saved to Modal volume harmbench-attack-cases: {save_dir}")
    print("Download with:")
    print(f"  modal volume get harmbench-attack-cases /{Path(save_dir).name} benchmarks/safety/attack_cases/{Path(save_dir).name}")
