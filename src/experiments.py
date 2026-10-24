"""Experiment execution and batch runner for multi-seed data poisoning studies (Stage 3 & Stage 4)."""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.data import load_spambase, split_data
from src.attacks import random_label_flip, targeted_spam_to_ham
from src.models import get_models
from src.metrics import evaluate_predictions


def run_single_condition(
    df: pd.DataFrame,
    seed: int,
    attack: str,
    poison_rate: float
) -> List[Dict[str, Any]]:
    """Execute a single experimental condition across all three classifiers.

    Ensures that for a given (seed, attack, poison_rate) condition, all three models:
    1. Receive the exact same train/test split.
    2. Receive the exact same poisoned training labels.
    3. Are evaluated on the strictly clean, isolated test set.

    Parameters
    ----------
    df : pd.DataFrame
        Clean Spambase dataset.
    seed : int
        Experiment random seed (used for split, attack, and model initialization).
    attack : str
        Attack type: 'clean', 'random', or 'targeted'.
    poison_rate : float
        Fraction of training samples to poison (0.0 for clean baseline).

    Returns
    -------
    List[Dict[str, Any]]
        Three result dictionaries (one per model) containing performance metrics
        and confusion matrix counts (tn, fp, fn, tp).
    """
    # 1. Stratified 80/20 train-test split seeded with the experiment seed
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=seed)

    # 2. Apply poisoning attack to training labels
    if poison_rate == 0.0 or attack == "clean":
        attack_name = "clean"
        y_train_poisoned = y_train.copy()
        n_poisoned = 0
    elif attack == "random":
        attack_name = "random"
        y_train_poisoned, flipped_mask = random_label_flip(
            y_train, poison_rate=poison_rate, random_state=seed
        )
        n_poisoned = int(flipped_mask.sum())
    elif attack == "targeted":
        attack_name = "targeted"
        y_train_poisoned, flipped_mask = targeted_spam_to_ham(
            y_train, poison_rate=poison_rate, random_state=seed
        )
        n_poisoned = int(flipped_mask.sum())
    else:
        raise ValueError(f"Unknown attack type: '{attack}'. Supported: 'clean', 'random', 'targeted'.")

    # 3. Instantiate the three models using the exact same experiment seed
    models = get_models(random_state=seed)
    condition_results = []

    # 4. Train each model on the identical poisoned training set and evaluate on clean test set
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train_poisoned)
            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test.to_numpy(), y_pred, model_name=model_name)

            row = {
                "seed": seed,
                "attack": attack_name,
                "poison_rate": poison_rate,
                "model": model_name,
                "n_poisoned": n_poisoned,
                "accuracy": metrics["accuracy"],
                "spam_precision": metrics["spam_precision"],
                "spam_recall": metrics["spam_recall"],
                "spam_f1": metrics["spam_f1"],
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "tp": int(metrics["tp"]),
                "status": "success",
                "error_message": ""
            }
        except Exception as e:
            # Explicitly log failures instead of silently deleting them
            row = {
                "seed": seed,
                "attack": attack_name,
                "poison_rate": poison_rate,
                "model": model_name,
                "n_poisoned": n_poisoned,
                "accuracy": np.nan,
                "spam_precision": np.nan,
                "spam_recall": np.nan,
                "spam_f1": np.nan,
                "tn": np.nan,
                "fp": np.nan,
                "fn": np.nan,
                "tp": np.nan,
                "status": "failed",
                "error_message": str(e)
            }
        condition_results.append(row)

    return condition_results


def summarize_pilot_results(
    csv_path: str = "results/stage3_pilot_results.csv"
) -> pd.DataFrame:
    """Programmatically compute the Stage 3 summary table directly from the CSV source of truth."""
    df = pd.read_csv(csv_path)
    model_order = ["Logistic Regression", "Decision Tree", "Random Forest"]
    summary_rows = []

    for model_name in model_order:
        clean_sub = df[(df["attack"] == "clean") & (df["model"] == model_name)]
        rand20_sub = df[
            (df["attack"] == "random")
            & (np.isclose(df["poison_rate"], 0.20))
            & (df["model"] == model_name)
        ]
        targ20_sub = df[
            (df["attack"] == "targeted")
            & (np.isclose(df["poison_rate"], 0.20))
            & (df["model"] == model_name)
        ]

        c_rec_mean, c_rec_std = clean_sub["spam_recall"].mean(), clean_sub["spam_recall"].std(ddof=1)
        c_f1_mean, c_f1_std = clean_sub["spam_f1"].mean(), clean_sub["spam_f1"].std(ddof=1)
        r_rec_mean, r_rec_std = rand20_sub["spam_recall"].mean(), rand20_sub["spam_recall"].std(ddof=1)
        t_rec_mean, t_rec_std = targ20_sub["spam_recall"].mean(), targ20_sub["spam_recall"].std(ddof=1)
        t_prec_mean, t_prec_std = targ20_sub["spam_precision"].mean(), targ20_sub["spam_precision"].std(ddof=1)

        summary_rows.append({
            "Model": model_name,
            "Clean Recall": f"{c_rec_mean:.4f} ± {c_rec_std:.4f}",
            "Clean F1": f"{c_f1_mean:.4f} ± {c_f1_std:.4f}",
            "Random (20%) Recall": f"{r_rec_mean:.4f} ± {r_rec_std:.4f} ({(r_rec_mean - c_rec_mean)*100:+.2f}%)",
            "Targeted (20%) Recall": f"{t_rec_mean:.4f} ± {t_rec_std:.4f} ({(t_rec_mean - c_rec_mean)*100:+.2f}%)",
            "Targeted (20%) Precision": f"{t_prec_mean:.4f} ± {t_prec_std:.4f}",
            "clean_recall_mean": c_rec_mean,
            "clean_recall_std": c_rec_std,
            "clean_f1_mean": c_f1_mean,
            "clean_f1_std": c_f1_std,
            "random20_recall_mean": r_rec_mean,
            "random20_recall_std": r_rec_std,
            "targeted20_recall_mean": t_rec_mean,
            "targeted20_recall_std": t_rec_std,
            "targeted20_precision_mean": t_prec_mean,
            "targeted20_precision_std": t_prec_std,
        })

    return pd.DataFrame(summary_rows)


def aggregate_attack_results(
    csv_path: str = "results/attack_results.csv"
) -> pd.DataFrame:
    """Aggregate all conditions in attack_results.csv by (attack, poison_rate, model) using Mean and Std.

    Parameters
    ----------
    csv_path : str
        Path to saved attack_results.csv.

    Returns
    -------
    pd.DataFrame
        Aggregated table across seeds with mean, std, and formatted 'Mean ± Std' strings.
    """
    df = pd.read_csv(csv_path)
    model_order = ["Logistic Regression", "Decision Tree", "Random Forest"]
    attack_order = ["clean", "random", "targeted"]
    rate_order = [0.0, 0.01, 0.05, 0.10, 0.20]

    clean_means = (
        df[df["attack"] == "clean"]
        .groupby("model")[["accuracy", "spam_precision", "spam_recall", "spam_f1"]]
        .mean()
    )

    rows = []
    for attack in attack_order:
        rates = [0.0] if attack == "clean" else [r for r in rate_order if r > 0.0]
        for rate in rates:
            for model_name in model_order:
                sub = df[
                    (df["attack"] == attack)
                    & (np.isclose(df["poison_rate"], rate))
                    & (df["model"] == model_name)
                ]
                if sub.empty:
                    continue
                n_seeds = len(sub)
                n_poisoned = int(sub["n_poisoned"].iloc[0])

                acc_m, acc_s = sub["accuracy"].mean(), sub["accuracy"].std(ddof=1)
                prec_m, prec_s = sub["spam_precision"].mean(), sub["spam_precision"].std(ddof=1)
                rec_m, rec_s = sub["spam_recall"].mean(), sub["spam_recall"].std(ddof=1)
                f1_m, f1_s = sub["spam_f1"].mean(), sub["spam_f1"].std(ddof=1)

                c_rec = clean_means.loc[model_name, "spam_recall"]
                c_f1 = clean_means.loc[model_name, "spam_f1"]

                row = {
                    "attack": attack,
                    "poison_rate": rate,
                    "model": model_name,
                    "n_seeds": n_seeds,
                    "n_poisoned": n_poisoned,
                    "accuracy_mean": acc_m,
                    "accuracy_std": acc_s,
                    "spam_precision_mean": prec_m,
                    "spam_precision_std": prec_s,
                    "spam_recall_mean": rec_m,
                    "spam_recall_std": rec_s,
                    "spam_f1_mean": f1_m,
                    "spam_f1_std": f1_s,
                    "recall_drop": rec_m - c_rec,
                    "f1_drop": f1_m - c_f1,
                    "Accuracy": f"{acc_m:.4f} ± {acc_s:.4f}",
                    "Spam Precision": f"{prec_m:.4f} ± {prec_s:.4f}",
                    "Spam Recall": f"{rec_m:.4f} ± {rec_s:.4f}",
                    "Spam F1": f"{f1_m:.4f} ± {f1_s:.4f}",
                }
                if "tn" in sub.columns:
                    row["tn_mean"] = sub["tn"].mean()
                    row["fp_mean"] = sub["fp"].mean()
                    row["fn_mean"] = sub["fn"].mean()
                    row["tp_mean"] = sub["tp"].mean()
                rows.append(row)

    return pd.DataFrame(rows)


def run_pilot_study(
    data_path: str = "data/spambase.data",
    names_path: str = "data/spambase.names",
    seeds: Optional[List[int]] = None,
    poison_rates: Optional[List[float]] = None,
    attacks: Optional[List[str]] = None,
    output_csv: Optional[str] = "results/stage3_pilot_results.csv"
) -> pd.DataFrame:
    """Run the three-seed Stage 3 pilot study generating 81 result rows."""
    if seeds is None:
        seeds = [0, 1, 2]
    if poison_rates is None:
        poison_rates = [0.0, 0.01, 0.05, 0.10, 0.20]
    if attacks is None:
        attacks = ["random", "targeted"]

    df = load_spambase(data_path=data_path, names_path=names_path)
    all_rows = []

    print(f"Starting Stage 3 Pilot Study across seeds={seeds}, attacks={attacks}, rates={poison_rates}...")

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        clean_rows = run_single_condition(df, seed=seed, attack="clean", poison_rate=0.0)
        all_rows.extend(clean_rows)
        print(f"  [Seed {seed}] Clean Baseline (0%): 3 models recorded.")

        non_zero_rates = [r for r in poison_rates if r > 0.0]
        for attack in attacks:
            for rate in non_zero_rates:
                rows = run_single_condition(df, seed=seed, attack=attack, poison_rate=rate)
                all_rows.extend(rows)
                print(f"  [Seed {seed}] Attack '{attack}' @ {rate*100:.0f}%: 3 models recorded.")

    results_df = pd.DataFrame(all_rows)

    n_clean = len(results_df[results_df["attack"] == "clean"])
    n_poisoned = len(results_df[results_df["attack"] != "clean"])
    total_rows = len(results_df)
    expected_clean = len(seeds) * 3
    expected_poisoned = len(seeds) * len(attacks) * len([r for r in poison_rates if r > 0.0]) * 3
    expected_total = expected_clean + expected_poisoned

    print("\n=== Execution Verification ===")
    print(f"Clean-baseline rows:     {n_clean} (expected: {expected_clean})")
    print(f"Poisoned-condition rows: {n_poisoned} (expected: {expected_poisoned})")
    print(f"Total results rows:      {total_rows} (expected: {expected_total})")

    assert total_rows == expected_total, f"Expected {expected_total} rows, got {total_rows}"
    assert n_clean == expected_clean, f"Expected {expected_clean} clean rows, got {n_clean}"
    assert n_poisoned == expected_poisoned, f"Expected {expected_poisoned} poisoned rows, got {n_poisoned}"
    assert results_df["accuracy"].notnull().all(), "Found failed condition or NaN in metrics!"

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        export_cols = [
            "seed", "attack", "poison_rate", "model",
            "n_poisoned", "accuracy", "spam_precision", "spam_recall", "spam_f1"
        ]
        results_df[export_cols].to_csv(output_csv, index=False)
        print(f"\nSaved verified results to: {output_csv}")

    return results_df


def run_full_attack_study(
    data_path: str = "data/spambase.data",
    names_path: str = "data/spambase.names",
    seeds: Optional[List[int]] = None,
    poison_rates: Optional[List[float]] = None,
    attacks: Optional[List[str]] = None,
    output_csv: Optional[str] = "results/attack_results.csv"
) -> pd.DataFrame:
    """Run the complete 10-seed Stage 4 attack study generating exactly 270 result rows.

    Structure:
    - 30 clean-baseline rows: 10 seeds (0..9) x 1 rate (0.0) x 3 models
    - 240 poisoned-condition rows: 10 seeds (0..9) x 2 attacks x 4 non-zero rates x 3 models
    - Total: 270 rows saved to results/attack_results.csv (including confusion matrix counts tn, fp, fn, tp).
    """
    if seeds is None:
        seeds = list(range(10))
    if poison_rates is None:
        poison_rates = [0.0, 0.01, 0.05, 0.10, 0.20]
    if attacks is None:
        attacks = ["random", "targeted"]

    df = load_spambase(data_path=data_path, names_path=names_path)
    all_rows = []

    print(f"Starting Stage 4 Full Attack Study across seeds={seeds}, attacks={attacks}, rates={poison_rates}...")

    for seed in seeds:
        clean_rows = run_single_condition(df, seed=seed, attack="clean", poison_rate=0.0)
        all_rows.extend(clean_rows)

        non_zero_rates = [r for r in poison_rates if r > 0.0]
        for attack in attacks:
            for rate in non_zero_rates:
                rows = run_single_condition(df, seed=seed, attack=attack, poison_rate=rate)
                all_rows.extend(rows)
        print(f"  [Seed {seed}] Completed 27 runs (3 clean + 24 poisoned).")

    results_df = pd.DataFrame(all_rows)

    n_clean = len(results_df[results_df["attack"] == "clean"])
    n_poisoned = len(results_df[results_df["attack"] != "clean"])
    total_rows = len(results_df)

    print("\n=== Stage 4 Full Attack Study Verification ===")
    print(f"Clean-baseline rows:     {n_clean} (expected: 30)")
    print(f"Poisoned-condition rows: {n_poisoned} (expected: 240)")
    print(f"Total results rows:      {total_rows} (expected: 270)")

    assert total_rows == 270, f"Expected 270 rows, got {total_rows}"
    assert n_clean == 30, f"Expected 30 clean rows, got {n_clean}"
    assert n_poisoned == 240, f"Expected 240 poisoned rows, got {n_poisoned}"
    assert results_df["accuracy"].notnull().all(), "Found failed condition or NaN in metrics!"

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        export_cols = [
            "seed", "attack", "poison_rate", "model",
            "n_poisoned", "accuracy", "spam_precision", "spam_recall", "spam_f1",
            "tn", "fp", "fn", "tp"
        ]
        results_df[export_cols].to_csv(output_csv, index=False)
        print(f"\nSaved verified 270-row full attack results to: {output_csv}")

    return results_df
