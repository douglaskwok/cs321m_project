from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
import torch

from irt import (
    DEFAULT_OUTPUT_DIR,
    RESPONSE_MATRICES,
    Rasch,
    ThreePL,
    TwoPL,
    add_aic_bic_to_summary,
    add_regular_se_columns,
    conditional_bernoulli_log_likelihood,
    count_trainable_parameters,
    evaluate_heldout_auc,
    fit_1pl_item_marginal,
    fit_regular_models,
    flipped_1pl_item_marginal_log_likelihood,
    information_criterion_row,
    load_response_matrix,
    save_table,
    set_seed,
    summarize_heldout_eval,
)


REDUCED_FIT_ORDER = [
    "1pl_regular",
    "1pl_item_marginal_mmle",
    "2pl_regular",
    "3pl_regular",
]


def build_reduced_capability_table(df, rm, regular_fits, theta_1pl_item_marginal):
    accuracy = df.mean(axis=1, skipna=True)
    n_observed = df.notna().sum(axis=1)

    capability_df = pd.DataFrame(
        {
            "model": rm.subject_ids,
            "accuracy": accuracy.values,
            "n_observed": n_observed.values,
            "1pl_regular": regular_fits["1pl_regular"]["model"].ability.detach().cpu().numpy(),
            "1pl_item_marginal_mmle": theta_1pl_item_marginal.numpy(),
            "2pl_regular": regular_fits["2pl_regular"]["model"].ability.detach().cpu().numpy(),
            "3pl_regular": regular_fits["3pl_regular"]["model"].ability.detach().cpu().numpy(),
        }
    )

    for col in REDUCED_FIT_ORDER:
        capability_df[f"{col}_rank"] = capability_df[col].rank(ascending=False, method="min").astype(int)

    return capability_df.sort_values("1pl_regular", ascending=False)


def build_reduced_information_criteria_table(rm, regular_fits, model_1pl_item_marginal):
    n_observed = int(torch.isfinite(rm.data).sum().item())
    rows = []

    for fit_name in ["1pl_regular", "2pl_regular", "3pl_regular"]:
        model = regular_fits[fit_name]["model"]
        rows.append(
            information_criterion_row(
                fit_name,
                conditional_bernoulli_log_likelihood(model, rm.data),
                count_trainable_parameters(model),
                n_observed,
                bic_n=n_observed,
                likelihood_type="conditional_bernoulli",
            )
        )

    rows.append(
        information_criterion_row(
            "1pl_item_marginal_mmle",
            flipped_1pl_item_marginal_log_likelihood(
                rm.data,
                model_1pl_item_marginal,
                n_quadrature=31,
            ),
            rm.n_subjects,
            n_observed,
            bic_n=rm.n_items,
            likelihood_type="item_marginal_bernoulli",
        )
    )

    criteria_df = pd.DataFrame(rows)
    criteria_df["fit"] = pd.Categorical(criteria_df["fit"], categories=REDUCED_FIT_ORDER, ordered=True)
    return criteria_df.sort_values("fit").assign(fit=lambda x: x["fit"].astype(str)).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Fit reduced IRT model set for large response matrices.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--matrix",
        choices=sorted(RESPONSE_MATRICES),
        default="code_judge",
        help="Which response matrix to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved tables. Defaults to IRT/results/{matrix}_reduced_mmle.",
    )
    parser.add_argument("--heldout-repeats", type=int, default=5)
    parser.add_argument("--train-frac", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR / f"{args.matrix}_reduced_mmle"

    warnings.filterwarnings("ignore")
    set_seed(args.seed)
    print(f"Using device: {args.device}")
    print(f"Using seed: {args.seed}")
    print(f"Using matrix: {args.matrix}")
    print("Reduced fit set: 1PL JMLE, 1PL MMLE, 2PL JMLE, 3PL JMLE")

    df, _, _, rm = load_response_matrix(args.matrix)
    print(f"{rm.n_subjects} models x {rm.n_items} items, density = {rm.density:.1%}")

    regular_specs = {
        "1pl_regular": (Rasch, {"max_epochs": 10000, "lr": 0.01}),
        "2pl_regular": (TwoPL, {"max_epochs": 10000, "lr": 0.005}),
        "3pl_regular": (ThreePL, {"max_epochs": 10000, "lr": 0.003}),
    }

    regular_fits = fit_regular_models(rm, regular_specs, args.device)
    theta_1pl_item_marginal, model_1pl_item_marginal, history_1pl_item_marginal = fit_1pl_item_marginal(
        rm,
        args.device,
    )

    capability_df = build_reduced_capability_table(df, rm, regular_fits, theta_1pl_item_marginal)
    save_table(capability_df, args.output_dir / "capability_scores")

    capability_with_se_df = add_regular_se_columns(capability_df, regular_fits, rm.data)
    save_table(capability_with_se_df, args.output_dir / "capability_scores_with_uncertainty")

    item_marginal_fits = {
        "1pl_item_marginal_mmle": {
            "theta": theta_1pl_item_marginal,
            "model": model_1pl_item_marginal,
            "history": history_1pl_item_marginal,
        }
    }
    information_criteria_df = build_reduced_information_criteria_table(
        rm,
        regular_fits,
        model_1pl_item_marginal,
    )
    save_table(information_criteria_df, args.output_dir / "information_criteria")

    item_marginal_specs = {
        "1pl_item_marginal_mmle": {
            "kind": "flipped_1pl",
            "fit_kwargs": {"max_epochs": 500, "lr": 0.01, "n_quadrature": 31},
        }
    }

    heldout_eval_raw = evaluate_heldout_auc(
        regular_specs,
        item_marginal_specs,
        rm.data,
        device=args.device,
        n_repeats=args.heldout_repeats,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    save_table(heldout_eval_raw, args.output_dir / "heldout_eval_raw")

    heldout_eval_raw["fit"] = pd.Categorical(heldout_eval_raw["fit"], categories=REDUCED_FIT_ORDER, ordered=True)
    heldout_eval_raw = (
        heldout_eval_raw.sort_values(["rep", "fit"])
        .assign(fit=lambda x: x["fit"].astype(str))
        .reset_index(drop=True)
    )
    heldout_eval_summary = summarize_heldout_eval(heldout_eval_raw, REDUCED_FIT_ORDER)
    heldout_eval_summary = add_aic_bic_to_summary(heldout_eval_summary, information_criteria_df)
    save_table(heldout_eval_summary, args.output_dir / "heldout_eval_summary")

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["matrix"] = args.matrix
    config["fit_order"] = REDUCED_FIT_ORDER
    config["skipped_fits"] = ["2pl_item_marginal_mmle", "3pl_item_marginal_mmle"]
    config["benchmark_id"] = rm.info.get("benchmark_id", "")
    config["n_subjects"] = rm.n_subjects
    config["n_items"] = rm.n_items
    config["density"] = rm.density
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved {args.output_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
