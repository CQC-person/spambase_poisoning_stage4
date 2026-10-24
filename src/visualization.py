"""Visualization utilities for exploratory data analysis and model performance comparisons."""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_class_distribution(y: pd.Series, title: str = "Spambase Class Distribution", save_path: Optional[str] = None):
    """Plot bar chart showing class counts and percentages."""
    counts = y.value_counts().sort_index()
    labels = ["Non-Spam (0)", "Spam (1)"]
    percentages = (counts / len(y)) * 100

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, counts.values, color=["#2b5c8f", "#d95f02"], edgecolor="black", width=0.5)
    
    for bar, pct in zip(bars, percentages):
        height = int(bar.get_height())
        label_text = f"{height:,}\n({pct:.1f}%)"
        ax.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_feature_comparison_box(df: pd.DataFrame, feature_name: str, title: Optional[str] = None, save_path: Optional[str] = None):
    """Plot boxplot comparing a numerical feature between Spam and Non-Spam."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_df = df.copy()
    plot_df["Email Class"] = plot_df["is_spam"].map({0: "Non-Spam (0)", 1: "Spam (1)"})
    sns.boxplot(
        x="Email Class",
        y=feature_name,
        hue="Email Class",
        data=plot_df,
        palette={"Non-Spam (0)": "#2b5c8f", "Spam (1)": "#d95f02"},
        ax=ax,
        showfliers=False,
        legend=False
    )
    ax.set_xlabel("Email Class", fontsize=11)
    ax.set_ylabel(f"{feature_name} (%)", fontsize=11)
    ax.set_title(title or f"Comparison of '{feature_name}' (Outliers Excluded)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_feature_comparison_hist(df: pd.DataFrame, feature_name: str, log_scale: bool = True, title: Optional[str] = None, save_path: Optional[str] = None):
    """Plot log-scale distribution or histogram of a feature comparing Spam and Non-Spam."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    spam_data = df[df["is_spam"] == 1][feature_name]
    ham_data = df[df["is_spam"] == 0][feature_name]

    if log_scale:
        spam_data = np.log1p(spam_data)
        ham_data = np.log1p(ham_data)
        xlabel = f"log(1 + {feature_name})"
    else:
        xlabel = feature_name

    sns.kdeplot(ham_data, label="Non-Spam (0)", color="#2b5c8f", fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(spam_data, label="Spam (1)", color="#d95f02", fill=True, alpha=0.3, ax=ax)

    ax.set_title(title or f"Density Distribution of '{feature_name}'", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_confusion_matrices(cms: Dict[str, np.ndarray], title_prefix: str = "Clean Baseline", save_path: Optional[str] = None):
    """Plot 1x3 confusion matrices for Logistic Regression, Decision Tree, and Random Forest."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    model_names = list(cms.keys())

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm = cms[name]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Ham (0)", "Pred Spam (1)"],
            yticklabels=["True Ham (0)", "True Spam (1)"],
            annot_kws={"size": 12, "weight": "bold"}
        )
        title_text = f"{name}\n({title_prefix})"
        ax.set_title(title_text, fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_clean_vs_poisoned_metrics(clean_df: pd.DataFrame, poisoned_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot 4 figures comparing Accuracy, Spam Precision, Spam Recall, and Spam F1 between Clean and 5% Poisoned."""
    metrics = [
        ("Accuracy", "Overall Accuracy"),
        ("Spam Precision", "Spam Precision (Class 1)"),
        ("Spam Recall", "Spam Recall (Class 1)"),
        ("Spam F1", "Spam F1-Score (Class 1)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    models = clean_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    for idx, (col_name, display_title) in enumerate(metrics):
        ax = axes[idx]
        clean_vals = clean_df[col_name].values
        poison_vals = poisoned_df[col_name].values
        deltas = poison_vals - clean_vals

        bars1 = ax.bar(x - width/2, clean_vals, width, label="Clean Baseline (0%)", color="#2b5c8f", edgecolor="black")
        bars2 = ax.bar(x + width/2, poison_vals, width, label="Poisoned (5% Random Flip)", color="#d95f02", edgecolor="black")

        for i, (c_bar, p_bar, delta) in enumerate(zip(bars1, bars2, deltas)):
            sign = "+" if delta >= 0 else ""
            delta_str = f"Δ: {sign}{delta:.3f}"
            ax.annotate(delta_str,
                        xy=(x[i], max(c_bar.get_height(), p_bar.get_height())),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                        color="#a50f15" if delta < 0 else "#006d2c")

        ax.set_title(display_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9.5)
        ax.set_ylim(0.70, 1.05)
        ax.set_ylabel("Score", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(loc="lower right", frameon=True, fontsize=9)

    plt.suptitle("Clean Baseline vs. 5% Random Label Flipping Attack Performance Comparison", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_attack_class_distributions(
    random_stats_df: pd.DataFrame,
    targeted_stats_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """Plot class distribution shifts across poisoning rates for Random vs Targeted attacks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rates = random_stats_df["Poison Rate"].tolist()
    x = np.arange(len(rates))
    width = 0.35

    ax1.plot(
        rates, random_stats_df["Spam Pct"],
        marker="o", linewidth=2.5, markersize=8, color="#2b5c8f",
        label="Random Flipping (Indiscriminate / Availability)"
    )
    ax1.plot(
        rates, targeted_stats_df["Spam Pct"],
        marker="s", linewidth=2.5, markersize=8, color="#d95f02",
        label="Targeted Spam->Ham (Evasion / Integrity)"
    )

    for i, rate in enumerate(rates):
        r_pct = random_stats_df.loc[i, "Spam Pct"]
        t_pct = targeted_stats_df.loc[i, "Spam Pct"]
        ax1.annotate(f"{r_pct:.1f}%", (rate, r_pct), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold", color="#2b5c8f")
        ax1.annotate(f"{t_pct:.1f}%", (rate, t_pct), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, fontweight="bold", color="#d95f02")

    ax1.set_title("Training Set Apparent Spam Proportion vs. Poison Rate", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Poisoning Rate", fontsize=11)
    ax1.set_ylabel("Apparent Spam Percentage (%)", fontsize=11)
    ax1.set_ylim(10, 50)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, loc="upper right")

    ax2.bar(x - width/2, targeted_stats_df["Ham Count"], width, label="Ham (Non-Spam)", color="#2b5c8f", edgecolor="black")
    ax2.bar(x + width/2, targeted_stats_df["Spam Count"], width, label="Spam (Remaining)", color="#d95f02", edgecolor="black")

    for i in range(len(rates)):
        h_cnt = targeted_stats_df.loc[i, "Ham Count"]
        s_cnt = targeted_stats_df.loc[i, "Spam Count"]
        ax2.annotate(f"{h_cnt}", (x[i] - width/2, h_cnt), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5)
        ax2.annotate(f"{s_cnt}", (x[i] + width/2, s_cnt), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, fontweight="bold", color="#d95f02")

    ax2.set_title("Targeted Attack: Class Counts Breakdown Across Poison Rates", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(rates, fontsize=10)
    ax2.set_xlabel("Poisoning Rate", fontsize=11)
    ax2.set_ylabel("Number of Samples in Training Set", fontsize=11)
    ax2.set_ylim(0, 3500)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)


def plot_pilot_learning_curves(
    results_df: pd.DataFrame,
    metric: str = "spam_recall",
    metric_display_name: str = "Spam Recall",
    save_path: Optional[str] = None,
    study_label: Optional[str] = None,
):
    """Plot multi-seed mean and standard deviation curves comparing Random vs Targeted attacks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    clean_sub = results_df[results_df["attack"] == "clean"].copy()

    random_sub = results_df[results_df["attack"] == "random"].copy()
    clean_for_random = clean_sub.copy()
    clean_for_random["attack"] = "random"
    random_full = pd.concat([clean_for_random, random_sub], ignore_index=True)

    targeted_sub = results_df[results_df["attack"] == "targeted"].copy()
    clean_for_targeted = clean_sub.copy()
    clean_for_targeted["attack"] = "targeted"
    targeted_full = pd.concat([clean_for_targeted, targeted_sub], ignore_index=True)

    model_styles = {
        "Logistic Regression": {"color": "#2b5c8f", "marker": "o", "ls": "-"},
        "Decision Tree": {"color": "#7570b3", "marker": "^", "ls": "--"},
        "Random Forest": {"color": "#1b9e77", "marker": "s", "ls": "-."}
    }

    rates = [0.0, 0.01, 0.05, 0.10, 0.20]
    rate_labels = ["0%", "1%", "5%", "10%", "20%"]

    for ax, data_subset, attack_title in [
        (ax1, random_full, "Random Label Flipping (Availability Threat)"),
        (ax2, targeted_full, "Targeted Spam->Ham Flipping (Integrity Threat)")
    ]:
        for model_name, style in model_styles.items():
            m_data = data_subset[data_subset["model"] == model_name]
            stats = m_data.groupby("poison_rate")[metric].agg(["mean", "std"]).reindex(rates)

            means = stats["mean"].values
            stds = stats["std"].fillna(0.0).values

            ax.plot(
                rates, means,
                label=model_name,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["ls"],
                linewidth=2.2,
                markersize=7
            )
            ax.fill_between(
                rates,
                np.maximum(0, means - stds),
                np.minimum(1.0, means + stds),
                color=style["color"],
                alpha=0.15
            )

        ax.set_title(attack_title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(rates)
        ax.set_xticklabels(rate_labels, fontsize=10)
        ax.set_xlabel("Poisoning Rate", fontsize=11)
        ax.set_ylabel(metric_display_name, fontsize=11)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, loc="lower left" if metric == "spam_recall" else "best")

    seeds = sorted(results_df["seed"].unique())
    if study_label is None:
        if len(seeds) == 10:
            study_label = "Full Attack Study (Seeds 0–9)"
        else:
            study_label = f"Pilot Study (Seeds {', '.join(map(str, seeds))})"
    plt.suptitle(f"{study_label}: {metric_display_name} Mean ± 1 Std across Poisoning Rates", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)


def plot_pilot_multi_metric_grid(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    study_label: Optional[str] = None,
):
    """Plot 2x2 grid of performance metrics (Spam Recall, Spam F1, Spam Precision, Accuracy)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    metrics_info = [
        ("spam_recall", "Spam Recall (Primary Metric)", axes[0, 0]),
        ("spam_f1", "Spam F1-Score (Primary Metric)", axes[0, 1]),
        ("spam_precision", "Spam Precision (Supporting Metric)", axes[1, 0]),
        ("accuracy", "Overall Accuracy (Supporting Metric)", axes[1, 1])
    ]

    clean_sub = results_df[results_df["attack"] == "clean"].copy()
    targeted_sub = results_df[results_df["attack"] == "targeted"].copy()
    clean_for_targeted = clean_sub.copy()
    clean_for_targeted["attack"] = "targeted"
    targeted_full = pd.concat([clean_for_targeted, targeted_sub], ignore_index=True)

    random_sub = results_df[results_df["attack"] == "random"].copy()
    clean_for_random = clean_sub.copy()
    clean_for_random["attack"] = "random"
    random_full = pd.concat([clean_for_random, random_sub], ignore_index=True)

    rates = [0.0, 0.01, 0.05, 0.10, 0.20]
    rate_labels = ["0%", "1%", "5%", "10%", "20%"]

    model_colors = {
        "Logistic Regression": "#2b5c8f",
        "Decision Tree": "#7570b3",
        "Random Forest": "#1b9e77"
    }

    for metric_col, title, ax in metrics_info:
        for model_name, color in model_colors.items():
            t_data = targeted_full[targeted_full["model"] == model_name]
            t_stats = t_data.groupby("poison_rate")[metric_col].agg(["mean", "std"]).reindex(rates)
            t_means = t_stats["mean"].values
            t_stds = t_stats["std"].fillna(0.0).values
            ax.plot(
                rates, t_means,
                color=color, linestyle="-", marker="o", linewidth=2.0, markersize=6,
                label=f"{model_name} (Targeted)"
            )
            ax.fill_between(
                rates,
                np.maximum(0.0, t_means - t_stds),
                np.minimum(1.0, t_means + t_stds),
                color=color, alpha=0.12
            )

            r_data = random_full[random_full["model"] == model_name]
            r_stats = r_data.groupby("poison_rate")[metric_col].agg(["mean"]).reindex(rates)
            ax.plot(
                rates, r_stats["mean"].values,
                color=color, linestyle=":", marker="x", linewidth=1.6, markersize=5, alpha=0.80,
                label=f"{model_name} (Random)"
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(rates)
        ax.set_xticklabels(rate_labels, fontsize=10)
        ax.set_xlabel("Poisoning Rate", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=8, loc="lower left" if "recall" in metric_col else "best")

    seeds = sorted(results_df["seed"].unique())
    if study_label is None:
        study_label = f"Seeds 0–{max(seeds)} Mean ± 1 Std" if len(seeds) == 10 else f"Seeds {', '.join(map(str, seeds))} Mean"
    plt.suptitle(f"Model-Family Multi-Metric Comparison: Targeted (Solid ± 1 Std) vs Random (Dotted) ({study_label})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_random_vs_targeted_degradation(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """Plot side-by-side comparison of Spam Recall and Spam F1 degradation under Random vs Targeted poisoning."""
    clean_means = (
        results_df[results_df["attack"] == "clean"]
        .groupby("model")[["spam_recall", "spam_f1"]]
        .mean()
    )

    models = ["Logistic Regression", "Decision Tree", "Random Forest"]
    rates = [0.05, 0.10, 0.20]
    rate_labels = ["5% Poisoning", "10% Poisoning", "20% Poisoning"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey="row")

    for row_idx, (metric_col, metric_name) in enumerate([
        ("spam_recall", "Spam Recall Drop (Percentage Points)"),
        ("spam_f1", "Spam F1 Drop (Percentage Points)")
    ]):
        for col_idx, (rate, r_label) in enumerate(zip(rates, rate_labels)):
            ax = axes[row_idx, col_idx]
            r_drops, r_errs, t_drops, t_errs = [], [], [], []

            for m in models:
                c_val = clean_means.loc[m, metric_col]
                r_sub = results_df[(results_df["attack"] == "random") & (np.isclose(results_df["poison_rate"], rate)) & (results_df["model"] == m)][metric_col]
                t_sub = results_df[(results_df["attack"] == "targeted") & (np.isclose(results_df["poison_rate"], rate)) & (results_df["model"] == m)][metric_col]

                r_drops.append((c_val - r_sub.mean()) * 100)
                r_errs.append(r_sub.std(ddof=1) * 100)
                t_drops.append((c_val - t_sub.mean()) * 100)
                t_errs.append(t_sub.std(ddof=1) * 100)

            x = np.arange(len(models))
            width = 0.36
            b1 = ax.bar(x - width/2, r_drops, width, yerr=r_errs, capsize=4, label="Random Flip (Availability)", color="#2b5c8f", edgecolor="black")
            b2 = ax.bar(x + width/2, t_drops, width, yerr=t_errs, capsize=4, label="Targeted Spam->Ham (Integrity)", color="#d95f02", edgecolor="black")

            for bar in list(b1) + list(b2):
                h = bar.get_height()
                ax.annotate(f"-{h:.1f}%", (bar.get_x() + bar.get_width()/2, max(h, 0)),
                            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

            if row_idx == 0:
                ax.set_title(r_label, fontsize=12, fontweight="bold", pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(["LogReg", "DecTree", "RandForest"], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(metric_name, fontsize=10.5, fontweight="bold")
            ax.set_ylim(0, 82 if metric_col == "spam_recall" else 68)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=True, fontsize=9, loc="upper left")

    plt.suptitle("Random vs. Targeted Poisoning Impact Across Models and Rates (10-Seed Mean ± 1 Std Loss)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_representative_confusion_matrices_grid(
    results_df: pd.DataFrame,
    seed: int = 0,
    attack: str = "targeted",
    save_path: Optional[str] = None
):
    """Plot 3x3 grid of representative confusion matrices at 0%, 10%, and 20% poisoning from saved CSV counts."""
    models = ["Logistic Regression", "Decision Tree", "Random Forest"]
    conditions = [
        ("clean", 0.0, "0% Clean Baseline"),
        (attack, 0.10, f"10% {attack.capitalize()} Poisoning"),
        (attack, 0.20, f"20% {attack.capitalize()} Poisoning"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.5))

    for row_idx, (att_name, rate, cond_label) in enumerate(conditions):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            sub = results_df[
                (results_df["seed"] == seed)
                & (results_df["attack"] == att_name)
                & (np.isclose(results_df["poison_rate"], rate))
                & (results_df["model"] == model_name)
            ].iloc[0]

            cm = np.array([
                [int(sub["tn"]), int(sub["fp"])],
                [int(sub["fn"]), int(sub["tp"])]
            ])

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues" if att_name == "clean" else ("Oranges" if attack == "targeted" else "Purples"),
                cbar=False,
                ax=ax,
                xticklabels=["Pred Ham (0)", "Pred Spam (1)"],
                yticklabels=["True Ham (0)", "True Spam (1)"],
                annot_kws={"size": 12, "weight": "bold"}
            )
            rec_pct = sub["spam_recall"] * 100
            f1_pct = sub["spam_f1"] * 100
            ax.set_title(
                f"{model_name} — {cond_label}\n(Recall: {rec_pct:.1f}%, F1: {f1_pct:.1f}%)",
                fontsize=10.5, fontweight="bold"
            )
            ax.set_ylabel("Actual Label" if col_idx == 0 else "", fontsize=10)
            ax.set_xlabel("Predicted Label" if row_idx == 2 else "", fontsize=10)

    plt.suptitle(
        f"Representative Test-Set Confusion Matrices (Seed {seed}, N_test = 921: 558 Ham, 363 Spam) — {attack.capitalize()} Attack at 0%, 10%, 20%",
        fontsize=13, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_seed_crossover_distributions(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """Plot 10-seed distribution (box + strip plot) at 10% and 20% targeted poisoning illustrating model crossover."""
    sub = results_df[
        (results_df["attack"] == "targeted")
        & (results_df["poison_rate"].isin([0.10, 0.20]))
    ].copy()
    sub["Poison Rate"] = sub["poison_rate"].map({0.10: "10% Targeted Poisoning", 0.20: "20% Targeted Poisoning"})

    palette = {
        "Logistic Regression": "#2b5c8f",
        "Decision Tree": "#7570b3",
        "Random Forest": "#1b9e77"
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    for ax, rate_str in [(ax1, "10% Targeted Poisoning"), (ax2, "20% Targeted Poisoning")]:
        rate_df = sub[sub["Poison Rate"] == rate_str]
        sns.boxplot(
            data=rate_df, x="model", y="spam_recall", hue="model",
            order=["Logistic Regression", "Decision Tree", "Random Forest"],
            palette=palette, width=0.45, boxprops=dict(alpha=0.35),
            showfliers=False, legend=False, ax=ax
        )
        sns.stripplot(
            data=rate_df, x="model", y="spam_recall", hue="model",
            order=["Logistic Regression", "Decision Tree", "Random Forest"],
            palette=palette, size=8, jitter=0.12, edgecolor="black", linewidth=1,
            legend=False, ax=ax
        )
        means = rate_df.groupby("model")["spam_recall"].mean()
        stds = rate_df.groupby("model")["spam_recall"].std(ddof=1)
        for idx, m in enumerate(["Logistic Regression", "Decision Tree", "Random Forest"]):
            ax.annotate(
                f"Mean: {means[m]*100:.2f}%\n(±{stds[m]*100:.2f}%)",
                xy=(idx, means[m]), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=palette[m], alpha=0.85)
            )
        ax.set_title(f"{rate_str} (10 Seeds)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Classifier Family", fontsize=11)
        ax.set_ylabel("Spam Recall on Clean Test Set", fontsize=11)
        ax.set_ylim(0.05, 0.95)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Model-Family Crossover & Cross-Seed Variance Under Targeted Poisoning (Seeds 0–9)", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)
