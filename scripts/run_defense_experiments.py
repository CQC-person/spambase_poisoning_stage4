#!/usr/bin/env python3
"""CLI script to run Stage 4 Week 2 (k-NN defense) and Week 3 (5-fold OOF defense) experiments."""

import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments import run_knn_defense_study, run_oof_defense_study


def main():
    parser = argparse.ArgumentParser(description="Run Stage 4 Label-Auditing Defense Experiments")
    parser.add_argument(
        "--defense",
        choices=["knn", "oof", "both"],
        default="both",
        help="Defense experiment to run: 'knn' (Week 2), 'oof' (Week 3), or 'both'",
    )
    parser.add_argument("--data-path", default="data/spambase.data", help="Path to spambase.data")
    parser.add_argument("--names-path", default="data/spambase.names", help="Path to spambase.names")
    parser.add_argument("--attack-csv", default="results/attack_results.csv", help="Path to attack_results.csv")
    parser.add_argument("--knn-output", default="results/knn_defense_results.csv", help="Path to k-NN defense CSV")
    parser.add_argument("--oof-output", default="results/oof_defense_results.csv", help="Path to OOF defense CSV")
    args = parser.parse_args()

    if args.defense in ("knn", "both"):
        print("================================================================")
        print("  STAGE 4 (WEEK 2): PRIMARY k-NN LABEL-AUDITING DEFENSE RUNNER  ")
        print("================================================================")
        run_knn_defense_study(
            data_path=args.data_path,
            names_path=args.names_path,
            attack_results_csv=args.attack_csv,
            output_csv=args.knn_output,
        )

    if args.defense in ("oof", "both"):
        print("\n================================================================")
        print("  STAGE 4 (WEEK 3): 5-FOLD OOF LOGISTIC AUDITING DEFENSE RUNNER ")
        print("================================================================")
        run_oof_defense_study(
            data_path=args.data_path,
            names_path=args.names_path,
            attack_results_csv=args.attack_csv,
            output_csv=args.oof_output,
        )

    print("\n[SUCCESS] Defense experiments completed and saved.")


if __name__ == "__main__":
    main()
