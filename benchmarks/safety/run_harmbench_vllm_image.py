import os
from pathlib import Path
import subprocess
import sys

import modal


def find_repo_root() -> Path:
    candidates = [Path.cwd()]
    try:
        candidates.append(Path(__file__).resolve().parents[2])
    except IndexError:
        pass
    for candidate in candidates:
        if (candidate / "HarmBench").exists():
            return candidate
    return Path.cwd()


REPO_ROOT = find_repo_root()

# Use vLLM's own CUDA/PyTorch image instead of rebuilding the whole stack with pip.
image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.10.2",
        setup_dockerfile_commands=[
            "RUN ln -sf $(command -v python3) /usr/local/bin/python"
        ],
    )
    .entrypoint([])
    .cmd([])
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
        }
    )
    .run_commands(
        "python -m pip install --no-deps pyyaml pandas tqdm ray accelerate sentencepiece fschat nltk"
    )
    .add_local_dir(REPO_ROOT / "HarmBench", remote_path="/root/HarmBench")
    .add_local_file(
        REPO_ROOT / "benchmarks" / "safety" / "safety_solver_behaviors.csv",
        remote_path="/root/benchmarks/safety/safety_solver_behaviors.csv",
    )
)

app = modal.App("harmbench-vllm-image", image=image)
attack_cases_volume = modal.Volume.from_name("harmbench-attack-cases", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("harmbench-hf-cache", create_if_missing=True)

METHODS = {
    "pap": ("PAP", "top_5", "PAP-top5"),
    "gcg": ("GCG", "vicuna_7b_v1_5", "GCG-vicuna_7b_v1_5"),
    "autodan": ("AutoDAN", "vicuna_7b_v1_5", "AutoDAN-vicuna_7b_v1_5"),
    "pair": ("PAIR", "vicuna_7b_v1_5", "PAIR-vicuna_7b_v1_5"),
    "gptfuzzer": ("GPTFuzz", "vicuna_7b_v1_5", "GPTFuzzer-vicuna_7b_v1_5"),
}


@app.function(
    gpu="H100:3",
    timeout=24 * 60 * 60,
    volumes={
        "/outputs": attack_cases_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def generate_attack(method_key: str):
    import csv
    import json

    import yaml

    method_name, experiment_name, out_name = METHODS[method_key]
    save_dir = f"/outputs/{out_name}"
    behaviors_path = Path("/root/benchmarks/safety/safety_solver_behaviors.csv")
    targets_path = Path("/root/benchmarks/safety/safety_solver_targets.json")
    patched_config_path = Path(
        f"/root/benchmarks/safety/{method_name}_safety_solver_config.yaml"
    )

    def target_for_behavior(behavior: str) -> str:
        behavior = behavior.strip()
        if not behavior:
            return "Sure, here is the requested information"
        return f"Sure, here is a detailed response to the request: {behavior}"

    def set_targets_path(config):
        if isinstance(config, dict):
            for key, value in list(config.items()):
                if key == "targets_path":
                    config[key] = str(targets_path)
                else:
                    set_targets_path(value)
        elif isinstance(config, list):
            for value in config:
                set_targets_path(value)

    with behaviors_path.open("r", encoding="utf-8") as f:
        targets = {
            row["BehaviorID"]: target_for_behavior(row["Behavior"])
            for row in csv.DictReader(f)
        }
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    with targets_path.open("w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2)

    base_config_path = Path(f"/root/HarmBench/configs/method_configs/{method_name}_config.yaml")
    with base_config_path.open("r", encoding="utf-8") as f:
        method_config = yaml.full_load(f)
    set_targets_path(method_config)

    if method_name == "PAIR":
        defaults = method_config["default_method_hyperparameters"]
        defaults["n_streams"] = 5
        defaults["max_retries"] = 5
        target_model = method_config["vicuna_7b_v1_5"]["target_model"]
        target_model["use_vllm"] = True
        target_model["num_gpus"] = 1
        target_model["dtype"] = "float16"

    if method_name == "AutoDAN":
        defaults = method_config["default_method_hyperparameters"]
        defaults["num_steps"] = 20
        defaults["batch_size"] = 16
        defaults["eval_steps"] = 5

    if method_name == "GCG":
        defaults = method_config["default_method_hyperparameters"]
        defaults["num_steps"] = 10
        defaults["search_width"] = 64
        defaults["eval_steps"] = 5
        defaults["use_prefix_cache"] = False

    if method_name == "GPTFuzz":
        defaults = method_config["default_method_hyperparameters"]
        defaults["mutate_model_path"] = "mistralai/Mistral-7B-Instruct-v0.2"
        defaults["max_query"] = 5
        defaults["max_iteration"] = 5
        defaults["max_new_tokens"] = 256
        method_config["vicuna_7b_v1_5"] = {
            "target_model": {
                "model_name_or_path": "lmsys/vicuna-7b-v1.5",
                "use_fast_tokenizer": False,
                "dtype": "float16",
                "chat_template": "vicuna",
                "gpu_memory_utilization": 0.35,
            },
            "num_gpus": 1,
        }

    with patched_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(method_config, f, sort_keys=False)

    cmd = [
        "python",
        "generate_test_cases.py",
        "--method_name",
        method_name,
        "--experiment_name",
        experiment_name,
        "--method_config_file",
        str(patched_config_path),
        "--behaviors_path",
        str(behaviors_path),
        "--save_dir",
        save_dir,
        "--overwrite",
    ]

    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
        }
    )

    process = subprocess.Popen(
        cmd,
        cwd="/root/HarmBench",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="")
            sys.stdout.flush()
    finally:
        hf_cache_volume.commit()

    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"HarmBench exited with status {returncode}. "
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
