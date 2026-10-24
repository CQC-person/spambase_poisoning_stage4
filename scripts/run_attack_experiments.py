#!/usr/bin/env python3
"""CLI script to run multi-seed data poisoning attack experiments (Stage 3 Pilot & Stage 4 Full Study)."""

import os
import sys
import argparse

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments import (
    run_pilot_study,
    run_full_attack_study,
    summarize_pilot_results,
    aggregate_attack_results,
)


def main():
    parser = argparse.ArgumentParser(description="Run Spambase Data Poisoning Attack Experiments")
    parser.add_argument("--stage", choices=["stage3", "stage4"], default="stage4", help="Experiment stage to run (default: stage4 10-seed study)")
    parser.add_argument("--data-path", default="data/spambase.data", help="Path to spambase.data")
    parser.add_argument("--names-path", default="data/spambase.names", help="Path to spambase.names")
    parser.add_argument("--output-csv", default=None, help="Path to output results CSV")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds")
    parser.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.10, 0.20], help="Poisoning rates")

    args = parser.parse_args()

    if args.stage == "stage3":
        seeds = args.seeds if args.seeds is not None else [0, 1, 2]
        output_csv = args.output_csv or "results/stage3_pilot_results.csv"
        print("================================================================")
        print("      STAGE 3: REPRODUCIBLE DATA POISONING PILOT RUNNER         ")
        print("================================================================")
        results_df = run_pilot_study(
            data_path=args.data_path,
            names_path=args.names_path,
            seeds=seeds,
            poison_rates=args.rates,
            output_csv=output_csv,
        )
        display_cols = [
            "Model",
            "Clean Recall",
            "Clean F1",
            "Random (20%) Recall",
            "Targeted (20%) Recall",
            "Targeted (20%) Precision",
        ]
        summary_df = summarize_pilot_results(csv_path=output_csv)
        print("\n=== Programmatic Pilot Summary Table (Mean ± Sample Std from CSV) ===")
        print(summary_df[display_cols].to_string(index=False))
    else:
        seeds = args.seeds if args.seeds is not None else list(range(10))
        output_csv = args.output_csv or "results/attack_results.csv"
        print("================================================================")
        print("   STAGE 4 (WEEK 1): 10-SEED FULL ATTACK EXPERIMENT RUNNER      ")
        print("================================================================")
        print(f"Seeds:        {seeds}")
        print(f"Rates:        {args.rates}")
        print(f"Output File:  {output_csv}")
        print("================================================================\n")

        results_df = run_full_attack_study(
            data_path=args.data_path,
            names_path=args.names_path,
            seeds=seeds,
            poison_rates=args.rates,
            output_csv=output_csv,
        )

        agg_df = aggregate_attack_results(csv_path=output_csv)
        display_cols = [
            "attack",
            "poison_rate",
            "model",
            "n_poisoned",
            "Accuracy",
            "Spam Precision",
            "Spam Recall",
            "Spam F1",
        ]
        print("\n=== Stage 4 Full Attack Summary Table (Mean ± Std Across 10 Seeds) ===")
        print(agg_df[display_cols].to_string(index=False))

    print("\n[SUCCESS] Attack study completed and verified.")


if __name__ == "__main__":
    main()
