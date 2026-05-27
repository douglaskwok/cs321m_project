#!/usr/bin/env python
"""Compute Spearman correlation between side-by-side ranking tables.

This is meant for CSV exports like ``IRT - MMLU.csv`` where the solver table
and judge table are stored next to each other, each with model name, ability,
and standard error columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="IRT/IRT - MMLU.csv",
        help="Path to side-by-side ranking CSV.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save matched ranks as CSV.",
    )
    parser.add_argument(
        "--solver-cols",
        nargs=3,
        default=None,
        metavar=("MODEL", "ABILITY", "SE"),
        help="Solver columns. Defaults to columns 1, 2, 3 by position.",
    )
    parser.add_argument(
        "--judge-cols",
        nargs=3,
        default=None,
        metavar=("MODEL", "ABILITY", "SE"),
        help="Judge columns. Defaults to columns 4, 5, 6 by position.",
    )
    parser.add_argument(
        "--data-start-row",
        type=int,
        default=2,
        help="Zero-indexed row where model data starts. Default: 2.",
    )
    return parser.parse_args()


def _extract_table(
    raw: pd.DataFrame,
    cols: list[str] | None,
    positions: list[int],
    prefix: str,
    data_start_row: int,
) -> pd.DataFrame:
    if cols is None:
        table = raw.iloc[data_start_row:, positions].copy()
    else:
        table = raw.loc[data_start_row:, cols].copy()

    table.columns = ["model", f"{prefix}_ability", f"{prefix}_se"]
    table["model"] = table["model"].astype(str).str.strip()
    table[f"{prefix}_ability"] = pd.to_numeric(table[f"{prefix}_ability"], errors="coerce")
    table[f"{prefix}_se"] = pd.to_numeric(table[f"{prefix}_se"], errors="coerce")
    return table.dropna(subset=["model", f"{prefix}_ability"])


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    raw = pd.read_csv(csv_path)

    solver = _extract_table(
        raw,
        args.solver_cols,
        positions=[1, 2, 3],
        prefix="solver",
        data_start_row=args.data_start_row,
    )
    judge = _extract_table(
        raw,
        args.judge_cols,
        positions=[4, 5, 6],
        prefix="judge",
        data_start_row=args.data_start_row,
    )

    merged = solver.merge(judge, on="model", how="inner")
    merged["solver_rank"] = merged["solver_ability"].rank(ascending=False, method="average")
    merged["judge_rank"] = merged["judge_ability"].rank(ascending=False, method="average")
    merged["rank_gap"] = merged["solver_rank"] - merged["judge_rank"]

    rho = merged["solver_rank"].corr(merged["judge_rank"], method="spearman")

    print(f"File: {csv_path}")
    print(f"Matched models: {len(merged)}")
    print(f"Spearman rho: {rho:.6f}")

    try:
        from scipy.stats import spearmanr

        _, p_value = spearmanr(merged["solver_rank"], merged["judge_rank"])
        print(f"p-value: {p_value:.6g}")
    except Exception:
        pass

    print("\nMatched ranks:")
    print(
        merged.sort_values("solver_rank")[
            [
                "model",
                "solver_ability",
                "solver_se",
                "judge_ability",
                "judge_se",
                "solver_rank",
                "judge_rank",
                "rank_gap",
            ]
        ].to_string(index=False)
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.sort_values("solver_rank").to_csv(output_path, index=False)
        print(f"\nSaved matched ranks to {output_path}")


if __name__ == "__main__":
    main()
