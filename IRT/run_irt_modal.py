"""Run IRT fitting on Modal.

Example:
    modal run IRT/run_irt_modal.py --seed 123 --heldout-repeats 5 --gpu A10G
    modal run IRT/run_irt_modal.py --seed 123 --heldout-repeats 5 --gpu H100
    modal run IRT/run_irt_modal.py --matrix judging --seed 123 --heldout-repeats 5 --gpu A10G
    modal run IRT/run_irt_modal.py --matrix safety --seed 123 --heldout-repeats 5 --gpu H100
    modal run IRT/run_irt_modal.py --matrix kudge_challenge --seed 123 --heldout-repeats 5 --gpu H100
    modal run IRT/run_irt_modal.py --matrix kudge_judge --seed 123 --heldout-repeats 5 --gpu H100
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
)

app = modal.App("irt-mmlu", image=image)


def _run_irt_impl(
    matrix: str,
    seed: int,
    heldout_repeats: int,
    train_frac: float,
    run_item_bootstrap: bool,
    n_boot: int,
    device: str,
) -> dict[str, bytes]:
    import subprocess

    remote_output_dir = REMOTE_OUTPUT_DIR / matrix
    args = [
        "python",
        "/root/IRT/irt.py",
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
    if run_item_bootstrap:
        args.extend(["--run-item-bootstrap", "--n-boot", str(n_boot)])

    subprocess.run(args, cwd="/root", check=True)

    outputs: dict[str, bytes] = {}
    for path in sorted(remote_output_dir.glob("*")):
        if path.is_file():
            outputs[path.name] = path.read_bytes()
    return outputs


@app.function(image=image, timeout=8 * 60 * 60)
def run_irt_cpu(
    matrix: str,
    seed: int,
    heldout_repeats: int,
    train_frac: float,
    run_item_bootstrap: bool,
    n_boot: int,
) -> dict[str, bytes]:
    return _run_irt_impl(
        matrix=matrix,
        seed=seed,
        heldout_repeats=heldout_repeats,
        train_frac=train_frac,
        run_item_bootstrap=run_item_bootstrap,
        n_boot=n_boot,
        device="cpu",
    )


@app.function(image=image, gpu="A10G", timeout=8 * 60 * 60)
def run_irt_a10g(
    matrix: str,
    seed: int,
    heldout_repeats: int,
    train_frac: float,
    run_item_bootstrap: bool,
    n_boot: int,
) -> dict[str, bytes]:
    return _run_irt_impl(
        matrix=matrix,
        seed=seed,
        heldout_repeats=heldout_repeats,
        train_frac=train_frac,
        run_item_bootstrap=run_item_bootstrap,
        n_boot=n_boot,
        device="cuda",
    )


@app.function(image=image, gpu="H100", timeout=8 * 60 * 60)
def run_irt_h100(
    matrix: str,
    seed: int,
    heldout_repeats: int,
    train_frac: float,
    run_item_bootstrap: bool,
    n_boot: int,
) -> dict[str, bytes]:
    return _run_irt_impl(
        matrix=matrix,
        seed=seed,
        heldout_repeats=heldout_repeats,
        train_frac=train_frac,
        run_item_bootstrap=run_item_bootstrap,
        n_boot=n_boot,
        device="cuda",
    )


@app.function(image=image, gpu="B200", timeout=8 * 60 * 60)
def run_irt_b200(
    matrix: str,
    seed: int,
    heldout_repeats: int,
    train_frac: float,
    run_item_bootstrap: bool,
    n_boot: int,
) -> dict[str, bytes]:
    return _run_irt_impl(
        matrix=matrix,
        seed=seed,
        heldout_repeats=heldout_repeats,
        train_frac=train_frac,
        run_item_bootstrap=run_item_bootstrap,
        n_boot=n_boot,
        device="cuda",
    )


@app.local_entrypoint()
def main(
    matrix: str = "solver",
    seed: int = 123,
    heldout_repeats: int = 5,
    train_frac: float = 0.8,
    output_dir: str = str(DEFAULT_LOCAL_OUTPUT_DIR),
    gpu: str = "A10G",
    run_item_bootstrap: bool = False,
    n_boot: int = 50,
) -> None:
    allowed_matrices = {"solver", "judging", "safety", "kudge_challenge", "kudge_judge"}
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
        run_item_bootstrap=run_item_bootstrap,
        n_boot=n_boot,
    )

    local_output_dir = Path(output_dir)
    if local_output_dir == DEFAULT_LOCAL_OUTPUT_DIR:
        local_output_dir = local_output_dir / matrix
    local_output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs.items():
        out_path = local_output_dir / filename
        out_path.write_bytes(content)
        print(f"Saved {out_path}")
