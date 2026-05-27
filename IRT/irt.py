from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_measure.data import ResponseMatrix, random_mask
from torch_measure.metrics.functional import compute_all
from torch_measure.models import Rasch, ThreePL, TwoPL, predict_dense
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "IRT" / "results"
RESPONSE_MATRICES = {
    "solver": {
        "prefix": "mmlu_pro_solver",
        "matrix_dir": PROJECT_ROOT / "benchmarks/mmlu/response_matrices",
        "benchmark_id": "mmlu_pro_solver",
        "item_id_field": "original_id",
        "item_content_field": "source",
        "value": "correct: 1.0=true, 0.0=false, NaN=unparsed",
    },
    "judging": {
        "prefix": "mmlu_pro_judging",
        "matrix_dir": PROJECT_ROOT / "benchmarks/mmlu/response_matrices",
        "benchmark_id": "mmlu_pro_judging",
        "item_id_field": "original_id:order",
        "item_content_field": "source",
        "value": "correct: 1.0=true, 0.0=false, NaN=unparsed",
    },
    "safety": {
        "prefix": "safety_all_attacks",
        "matrix_dir": PROJECT_ROOT / "benchmarks/safety/final_solver/response_matrices",
        "benchmark_id": "safety_all_attacks",
        "item_id_field": "attack_family:input_index",
        "item_content_field": "attack_family",
        "value": "score: 1.0=attack failed/safe, 0.0=attack succeeded/harmful, NaN=missing",
    },
    "kudge_challenge": {
        "prefix": "kudge_challenge_easy_hard",
        "matrix_dir": PROJECT_ROOT / "benchmarks/kudge/results/kudge_challenge_easy_hard/response_matrices",
        "benchmark_id": "kudge_challenge_easy_hard",
        "item_id_field": "id",
        "item_content_field": "subset",
        "value": "correct: 1.0=true, 0.0=false",
    },
    "kudge_judge": {
        "prefix": "kudge_judge_easy_hard",
        "matrix_dir": PROJECT_ROOT / "benchmarks/kudge/results/kudge_judge_easy_hard/response_matrices",
        "benchmark_id": "kudge_judge_easy_hard",
        "item_id_field": "id",
        "item_content_field": "subset",
        "value": "correct: 1.0=true, 0.0=false",
    },
}
ABILITY_COLS = [
    "1pl_regular",
    "1pl_item_marginal_mmle",
    "2pl_regular",
    "2pl_item_marginal_mmle",
    "3pl_regular",
    "3pl_item_marginal_mmle",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_table(df: pd.DataFrame, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    df.to_json(path_stem.with_suffix(".json"), orient="records", indent=2)
    print(f"Saved {path_stem.with_suffix('.csv')}")
    print(f"Saved {path_stem.with_suffix('.json')}")


def center_item_difficulty(model):
    with torch.no_grad():
        shift = model.difficulty.mean()
        model.difficulty.sub_(shift)
        model.ability.sub_(shift)
    return model


def load_response_matrix(matrix_kind: str):
    if matrix_kind not in RESPONSE_MATRICES:
        choices = ", ".join(sorted(RESPONSE_MATRICES))
        raise ValueError(f"Unknown matrix kind {matrix_kind!r}. Choose one of: {choices}")

    matrix_config = RESPONSE_MATRICES[matrix_kind]
    prefix = matrix_config["prefix"]
    matrix_dir = matrix_config["matrix_dir"]
    matrix_path = matrix_dir / f"{prefix}_response_matrix.csv"
    subject_meta_path = matrix_dir / f"{prefix}_subject_metadata.csv"
    item_meta_path = matrix_dir / f"{prefix}_item_metadata.csv"

    df = pd.read_csv(matrix_path, index_col=0)
    subject_meta = pd.read_csv(subject_meta_path)
    item_meta = pd.read_csv(item_meta_path)

    rm = ResponseMatrix(
        data=torch.tensor(df.values, dtype=torch.float32),
        subject_ids=list(df.index.astype(str)),
        item_ids=list(df.columns.astype(str)),
        item_contents=list(item_meta[matrix_config["item_content_field"]].astype(str)),
        subject_metadata=subject_meta.to_dict("records"),
        info={
            "benchmark_id": matrix_config["benchmark_id"],
            "item_id_field": matrix_config["item_id_field"],
            "value": matrix_config["value"],
        },
    )
    return df, subject_meta, item_meta, rm


def fit_regular_models(rm: ResponseMatrix, regular_specs: dict, device: str):
    regular_fits = {}
    for fit_name, (model_cls, fit_kwargs) in regular_specs.items():
        print(f"\nFitting {fit_name}")
        model = model_cls(rm.n_subjects, rm.n_items, device=device)
        history = model.fit(
            rm.data,
            method="mle",
            verbose=True,
            **fit_kwargs,
        )
        center_item_difficulty(model)
        regular_fits[fit_name] = {"model": model, "history": history}
        print(f"{fit_name} final loss: {history['losses'][-1]:.4f}")
    return regular_fits


def fit_1pl_item_marginal(rm: ResponseMatrix, device: str):
    rasch_item_marginal = Rasch(
        n_subjects=rm.n_items,
        n_items=rm.n_subjects,
        device=device,
    )
    history = rasch_item_marginal.fit(
        rm.data.T,
        method="em",
        max_epochs=500,
        lr=0.01,
        n_quadrature=31,
        verbose=True,
    )

    theta = -rasch_item_marginal.difficulty.detach().cpu()
    theta = theta - theta.mean()
    print(f"1PL item-marginal final loss: {history['losses_ability'][-1]:.4f}")
    return theta, rasch_item_marginal, history


def fit_abilities_item_marginalized_pl(
    data,
    pl=2,
    n_samples=1024,
    max_epochs=1000,
    lr=0.03,
    b_sd=1.5,
    log_a_mean=0.0,
    log_a_sd=0.5,
    c_alpha=1.0,
    c_beta=9.0,
    ability_prior_sd=3.0,
    device="cpu",
    seed=0,
):
    assert pl in {2, 3}
    torch.manual_seed(seed)

    y_matrix = data.to(device).float()
    mask = ~torch.isnan(y_matrix) & (y_matrix != -1)

    n_models, n_items = y_matrix.shape
    ability = torch.nn.Parameter(torch.zeros(n_models, device=device))

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    b_samples = torch.randn(n_samples, generator=gen, device=device) * b_sd
    log_a_samples = torch.randn(n_samples, generator=gen, device=device) * log_a_sd + log_a_mean
    a_samples = torch.exp(log_a_samples)

    if pl == 3:
        beta = torch.distributions.Beta(
            torch.tensor(c_alpha, device=device),
            torch.tensor(c_beta, device=device),
        )
        c_samples = beta.sample((n_samples,))
    else:
        c_samples = torch.zeros(n_samples, device=device)

    log_weight = -math.log(n_samples)
    n_obs = mask.sum().clamp_min(1).float()

    opt = torch.optim.Adam([ability], lr=lr)
    losses = []

    for _ in tqdm(range(max_epochs), desc=f"{pl}PL item-marginal MMLE"):
        opt.zero_grad()
        item_terms = []

        for item_idx in range(n_items):
            obs = mask[:, item_idx]
            if not obs.any():
                continue

            y = y_matrix[obs, item_idx]
            theta = ability[obs]
            logits = a_samples[None, :] * (theta[:, None] - b_samples[None, :])

            if pl == 2:
                logp = y[:, None] * F.logsigmoid(logits) + (1 - y[:, None]) * F.logsigmoid(-logits)
            else:
                p = c_samples[None, :] + (1 - c_samples[None, :]) * torch.sigmoid(logits)
                p = p.clamp(1e-7, 1 - 1e-7)
                logp = y[:, None] * torch.log(p) + (1 - y[:, None]) * torch.log1p(-p)

            item_terms.append(torch.logsumexp(log_weight + logp.sum(dim=0), dim=0))

        log_likelihood = torch.stack(item_terms).sum()
        prior = 0.5 * (ability / ability_prior_sd).pow(2).mean()
        loss = -log_likelihood / n_obs + prior

        loss.backward()
        opt.step()
        losses.append(loss.item())

    return ability.detach(), losses


def _observed_mask(data: torch.Tensor) -> torch.Tensor:
    return ~torch.isnan(data) & (data != -1)


def _bernoulli_log_prob(probs: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-7, 1 - 1e-7)
    return observed * torch.log(probs) + (1 - observed) * torch.log1p(-probs)


def count_trainable_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def information_criterion_row(fit_name, log_likelihood, n_parameters, n_observed, bic_n, likelihood_type):
    aic = -2 * log_likelihood + 2 * n_parameters
    bic = -2 * log_likelihood + math.log(max(bic_n, 1)) * n_parameters
    return {
        "fit": fit_name,
        "likelihood_type": likelihood_type,
        "log_likelihood": float(log_likelihood),
        "n_parameters": int(n_parameters),
        "n_observed": int(n_observed),
        "bic_n": int(bic_n),
        "aic": float(aic),
        "bic": float(bic),
    }


def conditional_bernoulli_log_likelihood(model, data: torch.Tensor) -> float:
    data = data.to(model.device).float()
    mask = _observed_mask(data)
    with torch.no_grad():
        probs = predict_dense(model)
        logp = _bernoulli_log_prob(probs[mask], data[mask])
    return float(logp.sum().item())


def flipped_1pl_item_marginal_log_likelihood(data: torch.Tensor, flipped_model, n_quadrature=31) -> float:
    y_matrix = data.T.to(flipped_model.device).float()
    mask = _observed_mask(y_matrix)

    theta_nodes, weights = np.polynomial.hermite_e.hermegauss(n_quadrature)
    theta_nodes = torch.tensor(theta_nodes, dtype=torch.float32, device=flipped_model.device)
    weights = torch.tensor(weights, dtype=torch.float32, device=flipped_model.device)
    weights = weights / weights.sum()
    log_weights = torch.log(weights)

    with torch.no_grad():
        original_ability = flipped_model.ability.detach().clone()
        item_terms = []
        for subject_idx in range(y_matrix.shape[0]):
            obs = mask[subject_idx]
            if not obs.any():
                continue

            y = y_matrix[subject_idx, obs]
            log_terms = []
            for q, theta in enumerate(theta_nodes):
                flipped_model.ability.fill_(theta.item())
                subject = torch.full((int(obs.sum().item()),), subject_idx, dtype=torch.long, device=flipped_model.device)
                item = torch.arange(y_matrix.shape[1], device=flipped_model.device)[obs]
                probs = flipped_model.predict({"subject_idx": subject, "item_idx": item})
                log_terms.append(log_weights[q] + _bernoulli_log_prob(probs, y).sum())
            item_terms.append(torch.logsumexp(torch.stack(log_terms), dim=0))

        flipped_model.ability.copy_(original_ability)

    if not item_terms:
        return float("nan")
    return float(torch.stack(item_terms).sum().item())


def prior_item_marginal_log_likelihood(
    data: torch.Tensor,
    theta: torch.Tensor,
    pl: int,
    device: str,
    n_samples=8192,
    b_sd=1.5,
    log_a_mean=0.0,
    log_a_sd=0.5,
    c_alpha=1.0,
    c_beta=9.0,
    seed=0,
) -> float:
    assert pl in {2, 3}
    y_matrix = data.to(device).float()
    mask = _observed_mask(y_matrix)
    theta = theta.to(device).float()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    b_samples = torch.randn(n_samples, generator=gen, device=device) * b_sd
    log_a_samples = torch.randn(n_samples, generator=gen, device=device) * log_a_sd + log_a_mean
    a_samples = torch.exp(log_a_samples)

    if pl == 3:
        beta = torch.distributions.Beta(
            torch.tensor(c_alpha, device=device),
            torch.tensor(c_beta, device=device),
        )
        c_samples = beta.sample((n_samples,))
    else:
        c_samples = torch.zeros(n_samples, device=device)

    log_weight = -math.log(n_samples)
    item_terms = []
    with torch.no_grad():
        for item_idx in range(y_matrix.shape[1]):
            obs = mask[:, item_idx]
            if not obs.any():
                continue

            y = y_matrix[obs, item_idx]
            logits = a_samples[None, :] * (theta[obs, None] - b_samples[None, :])
            if pl == 2:
                logp = y[:, None] * F.logsigmoid(logits) + (1 - y[:, None]) * F.logsigmoid(-logits)
            else:
                probs = c_samples[None, :] + (1 - c_samples[None, :]) * torch.sigmoid(logits)
                logp = _bernoulli_log_prob(probs, y[:, None])
            item_terms.append(torch.logsumexp(log_weight + logp.sum(dim=0), dim=0))

    if not item_terms:
        return float("nan")
    return float(torch.stack(item_terms).sum().item())


def build_information_criteria_table(
    rm: ResponseMatrix,
    regular_fits: dict,
    item_marginal_fits: dict,
    device: str,
    seed: int,
):
    n_observed = int(_observed_mask(rm.data).sum().item())
    rows = []

    for fit_name in ["1pl_regular", "2pl_regular", "3pl_regular"]:
        model = regular_fits[fit_name]["model"]
        log_likelihood = conditional_bernoulli_log_likelihood(model, rm.data)
        rows.append(
            information_criterion_row(
                fit_name,
                log_likelihood,
                count_trainable_parameters(model),
                n_observed,
                bic_n=n_observed,
                likelihood_type="conditional_bernoulli",
            )
        )

    log_likelihood = flipped_1pl_item_marginal_log_likelihood(
        rm.data,
        item_marginal_fits["1pl_item_marginal_mmle"]["model"],
        n_quadrature=31,
    )
    rows.append(
        information_criterion_row(
            "1pl_item_marginal_mmle",
            log_likelihood,
            rm.n_subjects,
            n_observed,
            bic_n=rm.n_items,
            likelihood_type="item_marginal_bernoulli",
        )
    )

    for fit_name, pl, n_samples in [
        ("2pl_item_marginal_mmle", 2, 8192),
        ("3pl_item_marginal_mmle", 3, 8192),
    ]:
        kwargs = {"pl": pl, "n_samples": n_samples, "b_sd": 1.5}
        if pl == 3:
            kwargs.update({"c_alpha": 1.0, "c_beta": 9.0})
        log_likelihood = prior_item_marginal_log_likelihood(
            rm.data,
            item_marginal_fits[fit_name]["theta"],
            device=device,
            seed=seed + 200_000 + pl,
            **kwargs,
        )
        rows.append(
            information_criterion_row(
                fit_name,
                log_likelihood,
                rm.n_subjects,
                n_observed,
                bic_n=rm.n_items,
                likelihood_type=f"{pl}pl_item_marginal_bernoulli_mc",
            )
        )

    criteria_df = pd.DataFrame(rows)
    criteria_df["fit"] = pd.Categorical(criteria_df["fit"], categories=ABILITY_COLS, ordered=True)
    return criteria_df.sort_values("fit").assign(fit=lambda x: x["fit"].astype(str)).reset_index(drop=True)


def build_capability_table(
    df: pd.DataFrame,
    rm: ResponseMatrix,
    regular_fits: dict,
    theta_1pl_item_marginal: torch.Tensor,
    theta_2pl_item_marginal: torch.Tensor,
    theta_3pl_item_marginal: torch.Tensor,
):
    accuracy = df.mean(axis=1, skipna=True)
    n_observed = df.notna().sum(axis=1)

    capability_df = pd.DataFrame(
        {
            "model": rm.subject_ids,
            "accuracy": accuracy.loc[rm.subject_ids].values,
            "n_observed": n_observed.loc[rm.subject_ids].values,
            "1pl_regular": regular_fits["1pl_regular"]["model"].ability.detach().cpu().numpy(),
            "1pl_item_marginal_mmle": theta_1pl_item_marginal.cpu().numpy(),
            "2pl_regular": regular_fits["2pl_regular"]["model"].ability.detach().cpu().numpy(),
            "2pl_item_marginal_mmle": theta_2pl_item_marginal.cpu().numpy(),
            "3pl_regular": regular_fits["3pl_regular"]["model"].ability.detach().cpu().numpy(),
            "3pl_item_marginal_mmle": theta_3pl_item_marginal.cpu().numpy(),
        }
    )

    for col in ABILITY_COLS:
        capability_df[f"{col}_rank"] = capability_df[col].rank(ascending=False, method="min").astype(int)

    return capability_df.sort_values("1pl_regular", ascending=False)


def ability_laplace_se(model, data):
    data = data.to(model.device).float()
    mask = ~torch.isnan(data) & (data != -1)

    with torch.no_grad():
        theta = model.ability.detach()
        beta = model.difficulty.detach()

        if hasattr(model, "discrimination"):
            a = model.discrimination.detach()
        else:
            a = torch.ones_like(beta)

        logits = a.unsqueeze(0) * (theta.unsqueeze(1) - beta.unsqueeze(0))
        base_prob = torch.sigmoid(logits)

        if hasattr(model, "guessing"):
            c = model.guessing.detach()
            prob = c.unsqueeze(0) + (1 - c.unsqueeze(0)) * base_prob
            dprob_dtheta = (1 - c.unsqueeze(0)) * a.unsqueeze(0) * base_prob * (1 - base_prob)
        else:
            prob = base_prob
            dprob_dtheta = a.unsqueeze(0) * base_prob * (1 - base_prob)

        prob = prob.clamp(1e-7, 1 - 1e-7)
        info_matrix = dprob_dtheta.pow(2) / (prob * (1 - prob))
        info_matrix = torch.where(mask, info_matrix, torch.zeros_like(info_matrix))
        info = info_matrix.sum(dim=1)
        se = 1 / torch.sqrt(info.clamp_min(1e-8))

    return se.detach().cpu()


def add_regular_se_columns(capability_df: pd.DataFrame, regular_fits: dict, data: torch.Tensor):
    capability_df = capability_df.copy()
    for fit_name in ["1pl_regular", "2pl_regular", "3pl_regular"]:
        se = ability_laplace_se(regular_fits[fit_name]["model"], data)
        capability_df[f"{fit_name}_se"] = se.numpy()
        capability_df[f"{fit_name}_lo"] = capability_df[fit_name] - 1.96 * capability_df[f"{fit_name}_se"]
        capability_df[f"{fit_name}_hi"] = capability_df[fit_name] + 1.96 * capability_df[f"{fit_name}_se"]
    return capability_df


def fit_1pl_item_marginal_capability(data, max_epochs=200, lr=0.01, n_quadrature=31, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = Rasch(
        n_subjects=data.shape[1],
        n_items=data.shape[0],
        device=device,
    )
    _ = model.fit(
        data.T,
        method="em",
        max_epochs=max_epochs,
        lr=lr,
        n_quadrature=n_quadrature,
        verbose=False,
    )
    theta = -model.difficulty.detach().cpu()
    return theta - theta.mean()


def bootstrap_item_capabilities(data, fit_fn, fit_kwargs, n_boot=50, seed=0):
    rng = np.random.default_rng(seed)
    _, n_items = data.shape
    boot = []

    for boot_idx in tqdm(range(n_boot), desc="Bootstrap items"):
        cols = rng.integers(0, n_items, size=n_items)
        data_b = data[:, cols]
        theta_b = fit_fn(data_b, **fit_kwargs, seed=seed + boot_idx)
        if isinstance(theta_b, tuple):
            theta_b = theta_b[0]
        theta_b = theta_b.detach().cpu()
        theta_b = theta_b - theta_b.mean()
        boot.append(theta_b.numpy())

    return np.vstack(boot)


def add_bootstrap_summary(capability_df, boot, subject_ids, col):
    summary = pd.DataFrame(
        {
            "model": subject_ids,
            f"{col}_boot_mean": boot.mean(axis=0),
            f"{col}_boot_se": boot.std(axis=0, ddof=1),
            f"{col}_boot_lo": np.percentile(boot, 2.5, axis=0),
            f"{col}_boot_hi": np.percentile(boot, 97.5, axis=0),
        }
    )
    return capability_df.merge(summary, on="model", how="left")


def maybe_add_item_bootstrap(capability_df, rm, device, run_item_bootstrap=False, n_boot=50, seed=0):
    if not run_item_bootstrap:
        return capability_df

    boot_1pl = bootstrap_item_capabilities(
        rm.data,
        fit_1pl_item_marginal_capability,
        fit_kwargs={"max_epochs": 200, "lr": 0.01, "n_quadrature": 31, "device": device},
        n_boot=n_boot,
        seed=seed + 101,
    )
    capability_df = add_bootstrap_summary(capability_df, boot_1pl, rm.subject_ids, "1pl_item_marginal_mmle")

    boot_2pl = bootstrap_item_capabilities(
        rm.data,
        fit_abilities_item_marginalized_pl,
        fit_kwargs={"pl": 2, "n_samples": 1024, "max_epochs": 300, "lr": 0.03, "b_sd": 1.5, "device": device},
        n_boot=n_boot,
        seed=seed + 202,
    )
    capability_df = add_bootstrap_summary(capability_df, boot_2pl, rm.subject_ids, "2pl_item_marginal_mmle")

    boot_3pl = bootstrap_item_capabilities(
        rm.data,
        fit_abilities_item_marginalized_pl,
        fit_kwargs={
            "pl": 3,
            "n_samples": 2048,
            "max_epochs": 300,
            "lr": 0.03,
            "b_sd": 1.5,
            "c_alpha": 1.0,
            "c_beta": 9.0,
            "device": device,
        },
        n_boot=n_boot,
        seed=seed + 303,
    )
    return add_bootstrap_summary(capability_df, boot_3pl, rm.subject_ids, "3pl_item_marginal_mmle")


def _heldout_metric_row(fit_name, rep, metrics, data, test_mask):
    y_test = data[test_mask].float()
    return {
        "fit": fit_name,
        "rep": rep,
        "n_test": int(test_mask.sum().item()),
        "test_positive_rate": y_test.mean().item(),
        "heldout_auc": metrics["auc"],
        "heldout_log_likelihood": metrics["log_likelihood"],
        "heldout_brier": metrics["brier"],
        "heldout_ece": metrics["ece"],
    }


def heldout_auc_for_regular_model(model_cls, data, fit_kwargs, train_mask, test_mask, device):
    model = model_cls(data.shape[0], data.shape[1], device=device)
    history = model.fit(
        data,
        mask=train_mask,
        method="mle",
        verbose=False,
        **fit_kwargs,
    )
    center_item_difficulty(model)

    probs = predict_dense(model).detach()
    metrics = compute_all(probs, data, mask=test_mask)
    return metrics, model, history


def heldout_auc_for_1pl_item_marginal(data, fit_kwargs, train_mask, test_mask, device):
    flipped = Rasch(
        n_subjects=data.shape[1],
        n_items=data.shape[0],
        device=device,
    )
    history = flipped.fit(
        data.T,
        mask=train_mask.T,
        method="em",
        verbose=False,
        **fit_kwargs,
    )
    center_item_difficulty(flipped)

    probs = predict_dense(flipped).detach().T
    metrics = compute_all(probs, data, mask=test_mask)
    return metrics, flipped, history


def item_marginal_prior_probs(
    theta,
    n_items,
    pl,
    device,
    n_samples=8192,
    b_sd=1.5,
    log_a_mean=0.0,
    log_a_sd=0.5,
    c_alpha=1.0,
    c_beta=9.0,
    seed=0,
):
    theta = theta.to(device).float()
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    b_samples = torch.randn(n_samples, generator=gen, device=device) * b_sd
    log_a_samples = torch.randn(n_samples, generator=gen, device=device) * log_a_sd + log_a_mean
    a_samples = torch.exp(log_a_samples)

    logits = a_samples[None, :] * (theta[:, None] - b_samples[None, :])
    if pl == 2:
        p_samples = torch.sigmoid(logits)
    elif pl == 3:
        beta = torch.distributions.Beta(
            torch.tensor(c_alpha, device=device),
            torch.tensor(c_beta, device=device),
        )
        c_samples = beta.sample((n_samples,))
        p_samples = c_samples[None, :] + (1 - c_samples[None, :]) * torch.sigmoid(logits)
    else:
        raise ValueError(f"Expected pl=2 or pl=3, got {pl!r}")

    p_model = p_samples.mean(dim=1)
    return p_model[:, None].expand(-1, n_items)


def heldout_auc_for_prior_item_marginal(data, fit_kwargs, predict_kwargs, train_mask, test_mask, device, seed=0):
    train_data = data.clone()
    train_data[~train_mask] = float("nan")

    theta, history = fit_abilities_item_marginalized_pl(
        train_data,
        **fit_kwargs,
        seed=seed,
    )
    theta = theta - theta.mean()

    probs = item_marginal_prior_probs(
        theta,
        n_items=data.shape[1],
        seed=seed + 100_000,
        device=device,
        **predict_kwargs,
    ).detach()
    metrics = compute_all(probs, data, mask=test_mask)
    return metrics, theta, history


def evaluate_heldout_auc(regular_specs, item_marginal_specs, data, device, n_repeats=5, train_frac=0.8, seed=0):
    data = data.to(device).float()
    observed = ~torch.isnan(data) & (data != -1)
    rows = []

    for rep in range(n_repeats):
        torch.manual_seed(seed + rep)
        train_mask, test_mask = random_mask(observed, train_frac=train_frac)

        for fit_name, (model_cls, fit_kwargs) in regular_specs.items():
            metrics, _, _ = heldout_auc_for_regular_model(
                model_cls,
                data,
                fit_kwargs,
                train_mask=train_mask,
                test_mask=test_mask,
                device=device,
            )
            rows.append(_heldout_metric_row(fit_name, rep, metrics, data, test_mask))

        for fit_name, spec in item_marginal_specs.items():
            if spec["kind"] == "flipped_1pl":
                metrics, _, _ = heldout_auc_for_1pl_item_marginal(
                    data,
                    spec["fit_kwargs"],
                    train_mask=train_mask,
                    test_mask=test_mask,
                    device=device,
                )
            elif spec["kind"] == "prior_marginal":
                metrics, _, _ = heldout_auc_for_prior_item_marginal(
                    data,
                    spec["fit_kwargs"],
                    spec["predict_kwargs"],
                    train_mask=train_mask,
                    test_mask=test_mask,
                    device=device,
                    seed=seed + rep,
                )
            else:
                raise ValueError(f"Unknown item-marginal spec kind: {spec['kind']!r}")

            rows.append(_heldout_metric_row(fit_name, rep, metrics, data, test_mask))

    return pd.DataFrame(rows)


def summarize_heldout_eval(heldout_eval_raw: pd.DataFrame, fit_order: list[str]):
    summary = (
        heldout_eval_raw.groupby("fit")
        .agg(
            n_test_mean=("n_test", "mean"),
            test_positive_rate_mean=("test_positive_rate", "mean"),
            heldout_auc_mean=("heldout_auc", "mean"),
            heldout_auc_sd=("heldout_auc", "std"),
            heldout_log_likelihood_mean=("heldout_log_likelihood", "mean"),
            heldout_log_likelihood_sd=("heldout_log_likelihood", "std"),
            heldout_brier_mean=("heldout_brier", "mean"),
            heldout_brier_sd=("heldout_brier", "std"),
            heldout_ece_mean=("heldout_ece", "mean"),
            heldout_ece_sd=("heldout_ece", "std"),
        )
        .reset_index()
    )
    summary["fit"] = pd.Categorical(summary["fit"], categories=fit_order, ordered=True)
    return summary.sort_values("fit").assign(fit=lambda x: x["fit"].astype(str)).reset_index(drop=True)


def add_aic_bic_to_summary(summary: pd.DataFrame, information_criteria: pd.DataFrame) -> pd.DataFrame:
    ic = information_criteria[["fit", "aic", "bic"]]
    return summary.merge(ic, on="fit", how="left")


def parse_args():
    parser = argparse.ArgumentParser(description="Fit IRT models for benchmark response matrices.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--matrix",
        choices=sorted(RESPONSE_MATRICES),
        default="solver",
        help="Which response matrix to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved tables. Defaults to IRT/results/{matrix}.",
    )
    parser.add_argument("--heldout-repeats", type=int, default=5)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--run-item-bootstrap", action="store_true")
    parser.add_argument("--n-boot", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR / args.matrix

    warnings.filterwarnings("ignore")
    set_seed(args.seed)
    print(f"Using device: {args.device}")
    print(f"Using seed: {args.seed}")
    print(f"Using matrix: {args.matrix}")

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

    theta_2pl_item_marginal, history_2pl_item_marginal = fit_abilities_item_marginalized_pl(
        rm.data,
        pl=2,
        n_samples=2048,
        max_epochs=1000,
        lr=0.03,
        b_sd=1.5,
        device=args.device,
        seed=args.seed,
    )
    theta_2pl_item_marginal = theta_2pl_item_marginal - theta_2pl_item_marginal.mean()

    theta_3pl_item_marginal, history_3pl_item_marginal = fit_abilities_item_marginalized_pl(
        rm.data,
        pl=3,
        n_samples=4096,
        max_epochs=1000,
        lr=0.03,
        b_sd=1.5,
        c_alpha=1.0,
        c_beta=9.0,
        device=args.device,
        seed=args.seed,
    )
    theta_3pl_item_marginal = theta_3pl_item_marginal - theta_3pl_item_marginal.mean()

    capability_df = build_capability_table(
        df,
        rm,
        regular_fits,
        theta_1pl_item_marginal,
        theta_2pl_item_marginal,
        theta_3pl_item_marginal,
    )
    save_table(capability_df, args.output_dir / "capability_scores")

    capability_with_se_df = add_regular_se_columns(capability_df, regular_fits, rm.data)
    capability_with_se_df = maybe_add_item_bootstrap(
        capability_with_se_df,
        rm,
        args.device,
        run_item_bootstrap=args.run_item_bootstrap,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    save_table(capability_with_se_df, args.output_dir / "capability_scores_with_uncertainty")

    item_marginal_fits = {
        "1pl_item_marginal_mmle": {
            "theta": theta_1pl_item_marginal,
            "model": model_1pl_item_marginal,
            "history": history_1pl_item_marginal,
        },
        "2pl_item_marginal_mmle": {
            "theta": theta_2pl_item_marginal,
            "history": history_2pl_item_marginal,
        },
        "3pl_item_marginal_mmle": {
            "theta": theta_3pl_item_marginal,
            "history": history_3pl_item_marginal,
        },
    }
    information_criteria_df = build_information_criteria_table(
        rm,
        regular_fits,
        item_marginal_fits,
        device=args.device,
        seed=args.seed,
    )

    item_marginal_specs = {
        "1pl_item_marginal_mmle": {
            "kind": "flipped_1pl",
            "fit_kwargs": {"max_epochs": 500, "lr": 0.01, "n_quadrature": 31},
        },
        "2pl_item_marginal_mmle": {
            "kind": "prior_marginal",
            "fit_kwargs": {
                "pl": 2,
                "n_samples": 2048,
                "max_epochs": 1000,
                "lr": 0.03,
                "b_sd": 1.5,
                "device": args.device,
            },
            "predict_kwargs": {"pl": 2, "n_samples": 8192, "b_sd": 1.5},
        },
        "3pl_item_marginal_mmle": {
            "kind": "prior_marginal",
            "fit_kwargs": {
                "pl": 3,
                "n_samples": 4096,
                "max_epochs": 1000,
                "lr": 0.03,
                "b_sd": 1.5,
                "c_alpha": 1.0,
                "c_beta": 9.0,
                "device": args.device,
            },
            "predict_kwargs": {"pl": 3, "n_samples": 8192, "b_sd": 1.5, "c_alpha": 1.0, "c_beta": 9.0},
        },
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

    heldout_eval_raw["fit"] = pd.Categorical(heldout_eval_raw["fit"], categories=ABILITY_COLS, ordered=True)
    heldout_eval_raw = (
        heldout_eval_raw.sort_values(["rep", "fit"])
        .assign(fit=lambda x: x["fit"].astype(str))
        .reset_index(drop=True)
    )
    heldout_eval_summary = summarize_heldout_eval(heldout_eval_raw, ABILITY_COLS)
    heldout_eval_summary = add_aic_bic_to_summary(heldout_eval_summary, information_criteria_df)
    save_table(heldout_eval_summary, args.output_dir / "heldout_eval_summary")

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["matrix"] = args.matrix
    config["benchmark_id"] = rm.info.get("benchmark_id", "")
    config["n_subjects"] = rm.n_subjects
    config["n_items"] = rm.n_items
    config["density"] = rm.density
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved {args.output_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
