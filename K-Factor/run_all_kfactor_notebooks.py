import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TMP = ROOT

BINARY_MATRICES = [
    "kudge_challenge",
    "kudge_judge",
    "mmlu_solver",
    "mmlu_judging",
    "safety_all_attacks",
    "code_solver",
    "code_judge",
]

CONTINUOUS_MATRICES = [
    "harmmetric_eval",
]


def parameterized_notebook(source_path: Path, matrix_name: str, out_path: Path) -> None:
    nb = json.loads(source_path.read_text())
    replaced = False
    needle = 'KFACTOR_MATRIX = "'
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if needle not in src:
            continue
        lines = []
        for line in src.splitlines():
            if line.strip().startswith("KFACTOR_MATRIX = "):
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(f'{indent}KFACTOR_MATRIX = "{matrix_name}"  # set by run_all_kfactor_notebooks.py')
                replaced = True
            else:
                lines.append(line)
        cell["source"] = [line + "\n" for line in lines]
        break
    if not replaced:
        raise RuntimeError(f"Could not find KFACTOR_MATRIX assignment in {source_path}")
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


def run_notebook(source_name: str, matrix_name: str) -> None:
    source_path = ROOT / source_name
    temp_path = TMP / f".tmp_{source_path.stem}_{matrix_name}.ipynb"
    parameterized_notebook(source_path, matrix_name, temp_path)

    print(f"\n=== Running {source_name} for {matrix_name} ===", flush=True)
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.kernel_name=cs321m-project",
            "--ExecutePreprocessor.timeout=-1",
            "--output-dir",
            str(TMP),
            "--output",
            temp_path.name,
            str(temp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"=== Finished {matrix_name} ===", flush=True)


def main() -> None:
    for matrix in BINARY_MATRICES:
        run_notebook("kfactor.ipynb", matrix)
    for matrix in CONTINUOUS_MATRICES:
        run_notebook("kfactor_continuous.ipynb", matrix)


if __name__ == "__main__":
    main()
