#!/usr/bin/env python
"""Plot solver-vs-judge IRT ability scatter plots with standard-error bars."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


CASES = [
    {
        "name": "mmlu",
        "title": "MMLU",
        "csv": Path("IRT/IRT - MMLU.csv"),
        "solver_label": "Solver ability (1PL JMLE)",
        "judge_label": "Judge ability (2PL JMLE)",
    },
    {
        "name": "kudge",
        "title": "Kudge",
        "csv": Path("IRT/IRT - Kudge.csv"),
        "solver_label": "Solver ability (1PL JMLE)",
        "judge_label": "Judge ability (1PL JMLE)",
    },
    {
        "name": "safety",
        "title": "Safety",
        "csv": Path("IRT/IRT - Safety.csv"),
        "solver_label": "Solver ability (1PL MMLE)",
        "judge_label": "Judge ability (3PL JMLE)",
    },
    {
        "name": "coding",
        "title": "Coding",
        "csv": Path("IRT/IRT - Coding.csv"),
        "solver_label": "Solver ability (1PL JMLE)",
        "judge_label": "Judge ability (2PL JMLE)",
    },
]

HIGH_CONTRAST_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E41A1C",  # red
    "#377EB8",  # deep blue
    "#4DAF4A",  # medium green
    "#984EA3",  # purple
    "#A65628",  # brown
    "#000000",  # black
    "#F781BF",  # pink
    "#666666",  # gray
]


def normalize_model_name(raw_name: str) -> tuple[str, str]:
    compact = (
        str(raw_name)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
    )

    if "opus" in compact and ("47" in compact or "4" in compact):
        return "claude_opus_4_7", "Claude Opus 4.7"
    if "sonnet" in compact and ("46" in compact or "4" in compact):
        return "claude_sonnet_4_6", "Claude Sonnet 4.6"
    if "haiku" in compact and ("45" in compact or "4" in compact):
        return "claude_haiku_4_5", "Claude Haiku 4.5"

    qwen_sizes = [
        ("3508", "0.8B"),
        ("08b", "0.8B"),
        ("08", "0.8B"),
        ("0.8", "0.8B"),
        ("27b", "27B"),
        ("27", "27B"),
        ("9b", "9B"),
        ("9", "9B"),
        ("4b", "4B"),
        ("4", "4B"),
        ("2b", "2B"),
        ("2", "2B"),
    ]
    if "qwen" in compact:
        for token, size in qwen_sizes:
            if token.replace(".", "") in compact:
                return f"qwen35_{size.lower().replace('.', '').replace('b', 'b')}", f"Qwen3.5 {size}"

    if "mistral" in compact or "ministral" in compact:
        for size in ["14", "8", "3"]:
            if size in compact:
                return f"mistral_{size}b", f"Mistral {size}B"

    display = str(raw_name).strip()
    return display.lower().replace(" ", "_"), display


def load_side_by_side(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)

    # The first CSV row is consumed by pandas as the header, so the visible
    # "Model Name" row is raw index 1 and model data starts at raw index 2.
    solver = raw.iloc[2:, [1, 2, 3]].copy()
    solver.columns = ["model", "solver_ability", "solver_se"]
    judge = raw.iloc[2:, [4, 5, 6]].copy()
    judge.columns = ["model", "judge_ability", "judge_se"]

    for table, ability_col, se_col in [
        (solver, "solver_ability", "solver_se"),
        (judge, "judge_ability", "judge_se"),
    ]:
        table["model_raw"] = table["model"].astype(str).str.strip()
        normalized = table["model_raw"].apply(normalize_model_name)
        table["model_key"] = normalized.apply(lambda x: x[0])
        table["model"] = normalized.apply(lambda x: x[1])
        table[ability_col] = pd.to_numeric(table[ability_col], errors="coerce")
        table[se_col] = pd.to_numeric(table[se_col], errors="coerce")
        table.dropna(subset=["model_key", "model", ability_col, se_col], inplace=True)

    solver.rename(columns={"model_raw": "solver_model_raw"}, inplace=True)
    judge.rename(columns={"model_raw": "judge_model_raw"}, inplace=True)
    merged = solver.merge(
        judge,
        on="model_key",
        how="inner",
        suffixes=("_solver", "_judge"),
    )
    merged["model"] = merged["model_solver"]
    merged.drop(columns=["model_solver", "model_judge"], inplace=True)
    merged["solver_rank"] = merged["solver_ability"].rank(ascending=False, method="average")
    merged["judge_rank"] = merged["judge_ability"].rank(ascending=False, method="average")
    return merged


def spearman_text(df: pd.DataFrame) -> str:
    try:
        from scipy.stats import spearmanr

        rho, p_value = spearmanr(df["solver_rank"], df["judge_rank"])
        return f"Spearman rho={rho:.3f}, p={p_value:.3g}"
    except Exception:
        rho = df["solver_rank"].corr(df["judge_rank"], method="spearman")
        return f"Spearman rho={rho:.3f}"


def build_model_colors(all_data: dict[str, pd.DataFrame]) -> dict[str, tuple]:
    models = sorted({model for df in all_data.values() for model in df["model"]})
    return {model: HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)] for i, model in enumerate(models)}


def model_legend(model_colors: dict[str, tuple], models: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=model_colors[model],
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            markersize=6,
            label=model,
        )
        for model in models
    ]


def _label_offsets(df: pd.DataFrame) -> tuple[float, float]:
    x_span = df["solver_ability"].max() - df["solver_ability"].min()
    y_span = df["judge_ability"].max() - df["judge_ability"].min()
    x_offset = 0.015 * x_span if x_span else 0.03
    y_offset = 0.015 * y_span if y_span else 0.03
    return x_offset, y_offset


def plot_case(case: dict, df: pd.DataFrame, out_dir: Path, model_colors: dict[str, tuple]) -> None:
    stats_text = spearman_text(df)

    fig, ax = plt.subplots(figsize=(9.1, 5.4), dpi=180)
    x_offset, y_offset = _label_offsets(df)
    for _, row in df.iterrows():
        color = model_colors[row["model"]]
        ax.errorbar(
            row["solver_ability"],
            row["judge_ability"],
            xerr=row["solver_se"],
            yerr=row["judge_se"],
            fmt="o",
            markersize=5,
            capsize=2.5,
            elinewidth=1,
            color=color,
            ecolor=color,
            alpha=0.9,
            markeredgecolor="#222222",
            markeredgewidth=0.4,
        )
        ax.text(
            row["solver_ability"] + x_offset,
            row["judge_ability"] + y_offset,
            row["model"],
            fontsize=7,
            color=color,
            alpha=0.9,
        )

    ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.35)
    ax.axvline(0, color="#999999", linewidth=0.8, alpha=0.35)
    ax.set_xlabel(case["solver_label"])
    ax.set_ylabel(case["judge_label"])
    ax.set_title(f"{case['title']}: Solver vs Judge IRT Ability\n{stats_text}")
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.legend(
        handles=model_legend(model_colors, list(df["model"])),
        title="Model",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
    )
    fig.tight_layout()

    png_path = out_dir / f"irt_{case['name']}_solver_judge_scatter_se.png"
    pdf_path = out_dir / f"irt_{case['name']}_solver_judge_scatter_se.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    data_path = out_dir / f"irt_{case['name']}_solver_judge_scatter_data.csv"
    df.to_csv(data_path, index=False)

    print(f"{case['title']}: {stats_text}")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {data_path}")


def plot_combined(all_data: dict[str, pd.DataFrame], out_dir: Path, model_colors: dict[str, tuple]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.2), dpi=180)
    axes_flat = axes.ravel()
    for ax, case in zip(axes_flat, CASES):
        df = all_data[case["name"]]
        x_offset, y_offset = _label_offsets(df)
        for _, row in df.iterrows():
            color = model_colors[row["model"]]
            ax.errorbar(
                row["solver_ability"],
                row["judge_ability"],
                xerr=row["solver_se"],
                yerr=row["judge_se"],
                fmt="o",
                markersize=4.5,
                capsize=2,
                elinewidth=0.9,
                color=color,
                ecolor=color,
                alpha=0.9,
                markeredgecolor="#222222",
                markeredgewidth=0.4,
            )
            ax.text(
                row["solver_ability"] + x_offset,
                row["judge_ability"] + y_offset,
                row["model"],
                fontsize=5.5,
                color=color,
                alpha=0.9,
            )
        ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.35)
        ax.axvline(0, color="#999999", linewidth=0.8, alpha=0.35)
        ax.set_title(f"{case['title']}\n{spearman_text(df)}")
        ax.set_xlabel(case["solver_label"])
        ax.set_ylabel(case["judge_label"])
        ax.grid(True, linewidth=0.5, alpha=0.25)

    legend_models = sorted(model_colors)
    axes_flat[-1].legend(
        handles=model_legend(model_colors, legend_models),
        title="Model",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
    )
    fig.tight_layout()
    png_path = out_dir / "irt_all_solver_judge_scatter_se.png"
    pdf_path = out_dir / "irt_all_solver_judge_scatter_se.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined:\n  {png_path}\n  {pdf_path}")


def main() -> None:
    out_dir = Path("IRT/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_data = {case["name"]: load_side_by_side(case["csv"]) for case in CASES}
    model_colors = build_model_colors(all_data)
    for case in CASES:
        plot_case(case, all_data[case["name"]], out_dir, model_colors)
    plot_combined(all_data, out_dir, model_colors)


if __name__ == "__main__":
    main()
