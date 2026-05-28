"""Run reduced IRT fitting on Modal.

This keeps 1PL JMLE/MMLE and higher-parameter JMLE fits, while skipping
2PL/3PL item-marginal MMLE.

Example:
    modal run IRT/run_irt_modal_reduced_mmle.py --matrix code_judge --seed 123 --heldout-repeats 5 --gpu H100 --output-dir IRT/results_modal/code_judge_reduced_mmle
"""

from __future__ import annotations

from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_OUTPUT_DIR = REPO_ROOT / "IRT" / "results_modal"
REMOTE_OUTPUT_DIR = Path("/root/IRT/results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "pandas>=2.3",
        "pyarrow>=0.12",
        "pyro-ppl>=1.8",
        "huggingface_hub>=0.16",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "tueplots>=0.0.14",
        "tqdm>=4.66",
        "torchmetrics>=1.4",
        "sentence-transformers>=5.4",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_file(REPO_ROOT / "IRT" / "irt.py", remote_path="/root/IRT/irt.py")
    .add_local_file(REPO_ROOT / "IRT" / "irt_reduced_mmle.py", remote_path="/root/IRT/irt_reduced_mmle.py")
    .add_local_dir(REPO_ROOT / "src", remote_path="/root/src")
    .add_local_dir(
        REPO_ROOT / "benchmarks" / "mmlu" / "response_matrices",
        remote_path="/root/benchmarks/mmlu/response_matrices",
    )
    .add_local_dir(
        REPO_ROOT / "benchmarks" / "safety" / "final_solver" / "response_matrices",
        remote_path="/root/benchmarks/safety/final_solver/response_matrices",
    )
    .add_local_dir(
        REPO_ROOT / "benchmarks" / "kudge" / "results" / "kudge_challenge_easy_hard" / "response_matrices",
        remote_path="/root/benchmarks/kudge/results/kudge_challenge_easy_hard/response_matrices",
    )
    .add_local_dir(
        REPO_ROOT / "benchmarks" / "kudge" / "results" / "kudge_judge_easy_hard" / "response_matrices",
        remote_path="/root/benchmarks/kudge/results/kudge_judge_easy_hard/response_matrices",
    )
    .add_local_dir(
        REPO_ROOT / "benchmarks" / "code" / "response_matrices",
        remote_path="/root/benchmarks/code/response_matrices",
    )
)

app = modal.App("irt-reduced-mmle", image=image)


def _run_irt_impl(
    matrix: str,
    seed: int,
    heldout_repeats: int,
    train_frac: float,
    device: str,
) -> dict[str, bytes]:
    import subprocess

    remote_output_dir = REMOTE_OUTPUT_DIR / f"{matrix}_reduced_mmle"
    args = [
        "python",
        "/root/IRT/irt_reduced_mmle.py",
        "--matrix",
        matrix,
        "--seed",
        str(seed),
        "--heldout-repeats",
        str(heldout_repeats),
        "--train-frac",
        str(train_frac),
        "--device",
        device,
        "--output-dir",
        str(remote_output_dir),
    ]

    subprocess.run(args, cwd="/root", check=True)

    outputs: dict[str, bytes] = {}
    for path in sorted(remote_output_dir.glob("*")):
        if path.is_file():
            outputs[path.name] = path.read_bytes()
    return outputs


@app.function(image=image, timeout=8 * 60 * 60)
def run_irt_cpu(matrix: str, seed: int, heldout_repeats: int, train_frac: float) -> dict[str, bytes]:
    return _run_irt_impl(matrix, seed, heldout_repeats, train_frac, device="cpu")


@app.function(image=image, gpu="A10G", timeout=8 * 60 * 60)
def run_irt_a10g(matrix: str, seed: int, heldout_repeats: int, train_frac: float) -> dict[str, bytes]:
    return _run_irt_impl(matrix, seed, heldout_repeats, train_frac, device="cuda")


@app.function(image=image, gpu="H100", timeout=8 * 60 * 60)
def run_irt_h100(matrix: str, seed: int, heldout_repeats: int, train_frac: float) -> dict[str, bytes]:
    return _run_irt_impl(matrix, seed, heldout_repeats, train_frac, device="cuda")


@app.function(image=image, gpu="B200", timeout=8 * 60 * 60)
def run_irt_b200(matrix: str, seed: int, heldout_repeats: int, train_frac: float) -> dict[str, bytes]:
    return _run_irt_impl(matrix, seed, heldout_repeats, train_frac, device="cuda")


@app.local_entrypoint()
def main(
    matrix: str = "code_judge",
    seed: int = 123,
    heldout_repeats: int = 5,
    train_frac: float = 0.8,
    output_dir: str = str(DEFAULT_LOCAL_OUTPUT_DIR / "code_judge_reduced_mmle"),
    gpu: str = "H100",
) -> None:
    allowed_matrices = {
        "solver",
        "judging",
        "safety",
        "kudge_challenge",
        "kudge_judge",
        "code_solver",
        "code_judge",
    }
    if matrix not in allowed_matrices:
        choices = ", ".join(sorted(allowed_matrices))
        raise ValueError(f"matrix must be one of: {choices}")

    gpu_normalized = gpu.strip().lower()
    if gpu_normalized in {"none", "cpu"}:
        runner = run_irt_cpu
    elif gpu_normalized == "a10g":
        runner = run_irt_a10g
    elif gpu_normalized == "h100":
        runner = run_irt_h100
    elif gpu_normalized == "b200":
        runner = run_irt_b200
    else:
        raise ValueError("gpu must be one of: cpu, A10G, H100, B200")

    outputs = runner.remote(
        matrix=matrix,
        seed=seed,
        heldout_repeats=heldout_repeats,
        train_frac=train_frac,
    )

    local_output_dir = Path(output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs.items():
        out_path = local_output_dir / filename
        out_path.write_bytes(content)
        print(f"Saved {out_path}")
