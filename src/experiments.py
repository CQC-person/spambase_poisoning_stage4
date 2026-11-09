"""Experiment execution and batch runner for multi-seed data poisoning & defense studies (Stage 3 & Stage 4)."""

import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.data import load_spambase, split_data
from src.attacks import random_label_flip, targeted_spam_to_ham
from src.models import get_models
from src.metrics import evaluate_predictions
from src.defense import (
    knn_label_audit,
    oof_logreg_label_audit,
    evaluate_defense_auditing,
    compute_recovery_ratio,
)


def run_single_condition(
    df: pd.DataFrame,
    seed: int,
    attack: str,
    poison_rate: float
) -> List[Dict[str, Any]]:
    """Execute a single experimental condition across all three classifiers."""
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=seed)

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

    models = get_models(random_state=seed)
    condition_results = []

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
    """Aggregate all conditions in attack_results.csv by (attack, poison_rate, model) using Mean and Std."""
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
    """Run the complete 10-seed Stage 4 attack study generating exactly 270 result rows."""
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


def run_defense_condition(
    df: pd.DataFrame,
    seed: int,
    attack: str,
    poison_rate: float,
    defense_type: str = "knn",
    clean_metrics_lookup: Optional[Dict[str, Dict[str, float]]] = None,
    poisoned_metrics_lookup: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """Execute a single defense condition (audit + filter + retrain all 3 models) and record all metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Clean Spambase dataset.
    seed : int
        Random seed (0..9).
    attack : str
        'random' or 'targeted'.
    poison_rate : float
        Nonzero poisoning rate (e.g., 0.01, 0.05, 0.10, 0.20).
    defense_type : str
        'knn' (k=10, threshold=0.5) or 'oof_logreg' (n_splits=5, threshold=0.8).
    """
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=seed)

    if attack == "random":
        y_train_poisoned, flipped_mask = random_label_flip(
            y_train, poison_rate=poison_rate, random_state=seed
        )
    elif attack == "targeted":
        y_train_poisoned, flipped_mask = targeted_spam_to_ham(
            y_train, poison_rate=poison_rate, random_state=seed
        )
    else:
        raise ValueError(f"Unsupported attack for defense evaluation: {attack}")

    # Compute clean and poisoned reference metrics if not pre-supplied
    if clean_metrics_lookup is None:
        clean_rows = run_single_condition(df, seed=seed, attack="clean", poison_rate=0.0)
        clean_metrics_lookup = {r["model"]: r for r in clean_rows}
    if poisoned_metrics_lookup is None:
        poisoned_rows = run_single_condition(df, seed=seed, attack=attack, poison_rate=poison_rate)
        poisoned_metrics_lookup = {r["model"]: r for r in poisoned_rows}

    # 1. Run label auditing strictly on (X_train, y_train_poisoned) without clean labels or test set
    if defense_type == "knn":
        audit_res = knn_label_audit(X_train, y_train_poisoned, k=10, threshold=0.5)
    elif defense_type == "oof_logreg":
        audit_res = oof_logreg_label_audit(
            X_train, y_train_poisoned, n_splits=5, threshold=0.8, random_state=seed
        )
    else:
        raise ValueError(f"Unknown defense_type: '{defense_type}'. Expected 'knn' or 'oof_logreg'.")

    # 2. Evaluate detection accuracy strictly AFTER flagging
    det_metrics = evaluate_defense_auditing(audit_res["flagged_mask"], flipped_mask)
    X_defended = audit_res["X_train_defended"]
    y_defended = audit_res["y_train_defended"]
    audit_runtime_sec = float(audit_res["audit_runtime_sec"])

    # 3. Retrain all three models on the sanitized training set and evaluate on clean test set
    models = get_models(random_state=seed)
    rows = []

    for model_name, model in models.items():
        t_fit0 = time.perf_counter()
        model.fit(X_defended, y_defended)
        y_pred = model.predict(X_test)
        retrain_runtime_sec = time.perf_counter() - t_fit0
        total_runtime_sec = audit_runtime_sec + retrain_runtime_sec

        def_m = evaluate_predictions(y_test.to_numpy(), y_pred, model_name=model_name)
        cln_m = clean_metrics_lookup[model_name]
        psn_m = poisoned_metrics_lookup[model_name]

        rec_recovery = compute_recovery_ratio(
            def_m["spam_recall"], psn_m["spam_recall"], cln_m["spam_recall"]
        )
        f1_recovery = compute_recovery_ratio(
            def_m["spam_f1"], psn_m["spam_f1"], cln_m["spam_f1"]
        )

        rows.append({
            "defense": defense_type,
            "seed": seed,
            "attack": attack,
            "poison_rate": poison_rate,
            "model": model_name,
            "n_train": det_metrics["n_train"],
            "n_poisoned": det_metrics["n_poisoned"],
            "n_flagged": det_metrics["n_flagged"],
            "n_true_detected": det_metrics["n_true_detected"],
            "n_clean_removed": det_metrics["n_clean_removed"],
            "n_missed_poison": det_metrics["n_missed_poison"],
            "frac_clean_removed": det_metrics["frac_clean_removed"],
            "frac_train_removed": det_metrics["frac_train_removed"],
            "detection_precision": det_metrics["detection_precision"],
            "detection_recall": det_metrics["detection_recall"],
            "detection_f1": det_metrics["detection_f1"],
            "clean_accuracy": cln_m["accuracy"],
            "clean_spam_precision": cln_m["spam_precision"],
            "clean_spam_recall": cln_m["spam_recall"],
            "clean_spam_f1": cln_m["spam_f1"],
            "poisoned_accuracy": psn_m["accuracy"],
            "poisoned_spam_precision": psn_m["spam_precision"],
            "poisoned_spam_recall": psn_m["spam_recall"],
            "poisoned_spam_f1": psn_m["spam_f1"],
            "accuracy": def_m["accuracy"],
            "spam_precision": def_m["spam_precision"],
            "spam_recall": def_m["spam_recall"],
            "spam_f1": def_m["spam_f1"],
            "recall_gain": def_m["spam_recall"] - psn_m["spam_recall"],
            "f1_gain": def_m["spam_f1"] - psn_m["spam_f1"],
            "recovery_spam_recall": rec_recovery,
            "recovery_spam_f1": f1_recovery,
            "tn": int(def_m["tn"]),
            "fp": int(def_m["fp"]),
            "fn": int(def_m["fn"]),
            "tp": int(def_m["tp"]),
            "audit_runtime_sec": audit_runtime_sec,
            "retrain_runtime_sec": retrain_runtime_sec,
            "runtime_sec": total_runtime_sec,
        })

    return rows


def run_knn_defense_study(
    data_path: str = "data/spambase.data",
    names_path: str = "data/spambase.names",
    attack_results_csv: str = "results/attack_results.csv",
    seeds: Optional[List[int]] = None,
    poison_rates: Optional[List[float]] = None,
    attacks: Optional[List[str]] = None,
    output_csv: Optional[str] = "results/knn_defense_results.csv",
) -> pd.DataFrame:
    """Run Week 2 primary k-NN label-auditing defense across all nonzero rates and 10 seeds (240 rows)."""
    if seeds is None:
        seeds = list(range(10))
    if poison_rates is None:
        poison_rates = [0.01, 0.05, 0.10, 0.20]
    if attacks is None:
        attacks = ["random", "targeted"]

    df = load_spambase(data_path=data_path, names_path=names_path)
    attack_df = pd.read_csv(attack_results_csv) if os.path.exists(attack_results_csv) else None

    all_rows = []
    print(f"Starting Stage 4 Week 2 k-NN Defense Study (seeds={seeds}, rates={poison_rates})...")

    for seed in seeds:
        clean_lookup = None
        if attack_df is not None:
            c_sub = attack_df[(attack_df["seed"] == seed) & (attack_df["attack"] == "clean")]
            if len(c_sub) == 3:
                clean_lookup = {r["model"]: r for _, r in c_sub.iterrows()}

        for attack in attacks:
            for rate in poison_rates:
                poison_lookup = None
                if attack_df is not None:
                    p_sub = attack_df[
                        (attack_df["seed"] == seed)
                        & (attack_df["attack"] == attack)
                        & (np.isclose(attack_df["poison_rate"], rate))
                    ]
                    if len(p_sub) == 3:
                        poison_lookup = {r["model"]: r for _, r in p_sub.iterrows()}

                rows = run_defense_condition(
                    df,
                    seed=seed,
                    attack=attack,
                    poison_rate=rate,
                    defense_type="knn",
                    clean_metrics_lookup=clean_lookup,
                    poisoned_metrics_lookup=poison_lookup,
                )
                all_rows.extend(rows)
        print(f"  [Seed {seed}] Completed 24 k-NN defense runs (2 attacks x 4 rates x 3 models).")

    results_df = pd.DataFrame(all_rows)
    expected_rows = len(seeds) * len(attacks) * len(poison_rates) * 3
    assert len(results_df) == expected_rows, f"Expected {expected_rows} rows, got {len(results_df)}"

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"\nSaved verified {len(results_df)}-row k-NN defense results to: {output_csv}")

    return results_df


def run_oof_defense_study(
    data_path: str = "data/spambase.data",
    names_path: str = "data/spambase.names",
    attack_results_csv: str = "results/attack_results.csv",
    seeds: Optional[List[int]] = None,
    poison_rates: Optional[List[float]] = None,
    attacks: Optional[List[str]] = None,
    output_csv: Optional[str] = "results/oof_defense_results.csv",
) -> pd.DataFrame:
    """Run Week 3 extended 5-fold out-of-fold logistic-regression defense at 10% poisoning across 10 seeds (60 rows)."""
    if seeds is None:
        seeds = list(range(10))
    if poison_rates is None:
        poison_rates = [0.10]
    if attacks is None:
        attacks = ["random", "targeted"]

    df = load_spambase(data_path=data_path, names_path=names_path)
    attack_df = pd.read_csv(attack_results_csv) if os.path.exists(attack_results_csv) else None

    all_rows = []
    print(f"Starting Stage 4 Week 3 5-Fold OOF Defense Study (seeds={seeds}, rates={poison_rates})...")

    for seed in seeds:
        clean_lookup = None
        if attack_df is not None:
            c_sub = attack_df[(attack_df["seed"] == seed) & (attack_df["attack"] == "clean")]
            if len(c_sub) == 3:
                clean_lookup = {r["model"]: r for _, r in c_sub.iterrows()}

        for attack in attacks:
            for rate in poison_rates:
                poison_lookup = None
                if attack_df is not None:
                    p_sub = attack_df[
                        (attack_df["seed"] == seed)
                        & (attack_df["attack"] == attack)
                        & (np.isclose(attack_df["poison_rate"], rate))
                    ]
                    if len(p_sub) == 3:
                        poison_lookup = {r["model"]: r for _, r in p_sub.iterrows()}

                rows = run_defense_condition(
                    df,
                    seed=seed,
                    attack=attack,
                    poison_rate=rate,
                    defense_type="oof_logreg",
                    clean_metrics_lookup=clean_lookup,
                    poisoned_metrics_lookup=poison_lookup,
                )
                all_rows.extend(rows)
        print(f"  [Seed {seed}] Completed {len(attacks)*len(poison_rates)*3} OOF defense runs.")

    results_df = pd.DataFrame(all_rows)
    expected_rows = len(seeds) * len(attacks) * len(poison_rates) * 3
    assert len(results_df) == expected_rows, f"Expected {expected_rows} rows, got {len(results_df)}"

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"\nSaved verified {len(results_df)}-row OOF defense results to: {output_csv}")

    return results_df
