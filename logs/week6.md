# Research Log: Week 6 (Stage 4 Week 2) - Primary k-NN Defense Evaluation

## What I Learned
During Week 2 of Stage 4, I refactored our neighborhood-disagreement analysis into reusable functions in `src/defense.py` (`compute_knn_disagreement_scores` and `knn_label_audit`) and verified them with unit tests in `tests/test_defense.py` covering self-neighbor exclusion (querying 11 neighbors and removing index $i$), exact disagreement calculation, reproducibility, training-only `StandardScaler` fitting, and strict isolation from test data and clean labels. Executing the 10-seed defense benchmark (`results/knn_defense_results.csv`, 240 rows across seeds 0–9, both attacks, and nonzero rates 1%, 5%, 10%, and 20%) showed that $k$-NN ($k=10, \tau=0.5$) achieves $90.60\%$–$92.04\%$ detection recall under random flipping and $75.92\%$–$81.47\%$ under 1%–10% targeted flipping. For Logistic Regression, $k$-NN filtering recovers $73.2\%$ of lost spam recall at 10% random poisoning ($83.03\% \to 87.02\% \pm 1.76\%$) and $40.2\%$ of lost spam recall ($44.6\%$ of lost F1) at 10% targeted poisoning ($65.73\% \to 74.88\% \pm 3.11\%$).

## What Failed & Unexpected Findings
Two negative findings in `notebooks/06_knn_defense_evaluation.ipynb` provided key insight into fixed-threshold geometric auditing:
1. **False-Removal Cost at Low Poisoning Damage:** Across 1%–10% poisoning, $\tau=0.5$ removes $409.4$–$484.0$ clean boundary training samples ($11.24\%$–$14.61\%$ of clean training data). When poisoning damage is minimal (such as at 1% poisoning or for Random Forest under random noise), discarding clean boundary examples lowers tree recall, yielding negative recovery ratios.
2. **Neighborhood Inversion at 20% Targeted Poisoning:** When $50.8\%$ of training spam ($736/1,450$) is flipped to ham, poisoned spam points form local majorities (dropping $k$-NN detection recall to $39.78\%$) and cause surviving clean spam points to be falsely removed ($587.6$ clean samples removed, $19.96\%$), reducing recall across all models.

## Next Steps
In Week 3, I will implement 5-fold out-of-fold logistic-regression auditing ($\tau_{\text{OOF}}=0.8$) and compare both defenses at 10% poisoning.
