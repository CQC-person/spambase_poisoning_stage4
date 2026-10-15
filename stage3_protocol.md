# Stage 3 Frozen Protocol & Blueprint for Stage 4

## 1. Executive Summary & Purpose
This document establishes the frozen experimental protocol resulting from Stage 3 data-poisoning foundations on the UCI Spambase dataset (4,601 instances, 57 continuous numerical features). By synthesizing empirical insights from the 3-seed pilot study (seeds 0, 1, 2; 81 total experimental runs), this protocol formalizes:
1. Empirical answers to the core research questions regarding poisoning strategy, rate, and model robustness.
2. The exact pipeline placement, algorithmic specification, and evaluation framework for the manually implemented label-auditing defense in Stage 4.
3. The expanded 10-seed experimental design, artifact-saving rules, and statistical validation requirements.

---

## 2. Synthesis of Pilot Results & Research Question Answers

### Core Research Question:
> *How do poisoning strategy and poisoning rate affect the reliability of classical spam classifiers, and how much performance can a manually implemented label-auditing defense recover?*

### Empirical Answers from Stage 3 Pilot (Seeds 0, 1, 2):
1. **Impact of Poisoning Strategy (Availability vs. Integrity)**:
   - **Random Label Flipping (Availability / Denial of Service)**:
     - Inverts labels bidirectionally across the feature space ($0 \leftrightarrow 1$).
     - Preserves overall class balance near baseline (~39.4% to 43.6% spam).
     - Induces gradual, symmetric degradation across all metrics: accuracy drops by 5–10%, while spam recall decreases moderately (Random Forest drops from 91.37% to 83.25%; Logistic Regression from 89.72% to 80.12% at 20% poisoning).
   - **Targeted Spam-to-Ham Flipping (Integrity / Evasion)**:
     - Relabels only spam instances as legitimate non-spam ($1 \to 0$).
     - Distorts the training class balance from 39.40% down to 19.40% at 20% poisoning (50.8% of all training spam mislabeled as benign).
     - Induces catastrophic, stealthy collapse specifically in **Spam Recall**:
       - **Logistic Regression**: Collapses from 89.72% to **19.22%** (-70.50% drop).
       - **Random Forest**: Collapses from 91.37% to **30.54%** (-60.83% drop).
       - **Decision Tree**: Drops to 51.10% with high seed variance ($\pm 4.3\%$).
     - Crucially, **Spam Precision remains artificially high (~95%+)**, demonstrating that targeted evasion allows spam to evade detection without generating conspicuous false positive alarms on legitimate ham.

2. **Model Family Robustness Hierarchy**:
   - **Random Forest (Ensemble / Bagging)**: Demonstrates highest baseline accuracy (95.08%) and strong resilience up to 10% poisoning. Subspace feature randomization and multi-tree voting mitigate individual poisoned samples.
   - **Logistic Regression (Linear Hyperplane)**: Highly susceptible to targeted label corruption. Mislabeled spam samples pull the global decision hyperplane toward the spam feature cluster, causing severe under-detection.
   - **Decision Tree (Single Estimator)**: Exhibits high variance across seeds ($\pm 0.02 - 0.04$), as localized greedy recursive partitioning is vulnerable to perturbations near decision boundaries.

---

## 3. Stage 4 Defense Integration Blueprint: Label-Auditing

### 3.1 Defense Pipeline Placement
The manually implemented label-auditing defense operates under the **Filter-then-Train** paradigm. It sits strictly between the data collection/poisoning phase and final model induction:

```
[Clean Training Data] 
         │
         ▼
[Adversarial Poisoning Attack] ──► [Contaminated Training Labels y_train_poisoned]
                                                      │
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │   STAGE 4 DEFENSE MODULE    │
                                       │   (Label-Auditing Filter)   │
                                       └─────────────────────────────┘
                                                      │
                                                      ▼
                                       [Sanitized Training Labels y_train_clean]
                                                      │
                                                      ▼
                                       [Model Training (LR, DT, RF)]
                                                      │
                                                      ▼
                                       [Evaluation on Isolated Test Set]
```

### 3.2 Algorithmic Specification: Neighborhood Discrepancy Auditing (k-NN / LOF)
Without using external adversarial libraries, Stage 4 will implement a custom label auditor:
1. **Feature Distance Metric**: Standardize training features ($X_{\text{train}}$) using standard Z-score scaling fit strictly on training features.
2. **Neighborhood Label Consistency**: For each training sample $i$, find its $k$ nearest neighbors in feature space ($k \in \{5, 10, 15\}$).
3. **Suspicion Score**: Compute the label disagreement ratio:
   $$S(x_i, y_i) = \frac{1}{k} \sum_{j \in \mathcal{N}_k(x_i)} \mathbb{I}(y_j \neq y_i)$$
4. **Audit Action**:
   - Samples with $S(x_i, y_i) > \tau$ (e.g., threshold $\tau = 0.60$) are flagged as suspicious.
   - **Sanitization Strategy**: Either flip flagged labels to the local majority vote ($y_i \leftarrow \text{mode}(\mathcal{N}_k(x_i))$) or prune flagged instances from the training set.

### 3.3 Defense Evaluation Metrics
1. **Auditing Detection Performance** (evaluated against the ground-truth `flipped_mask` preserved from the attack):
   - **Detection Precision**: Fraction of flagged samples that were truly poisoned.
   - **Detection Recall (TPR)**: Fraction of truly poisoned samples successfully flagged.
   - **False Positive Rate (FPR)**: Fraction of clean samples falsely flagged.
2. **Performance Recovery Score**:
   $$R_{\text{recovery}}(M) = \frac{M_{\text{defended}} - M_{\text{poisoned}}}{M_{\text{clean}} - M_{\text{poisoned}}}$$
   Where $M \in \{\text{Spam Recall}, \text{Spam F1}\}$. $R=1.0$ represents 100% complete recovery of clean baseline performance.

---

## 4. Full Stage 4 Experimental Grid & Protocol

### 4.1 Ten Random Seeds
The complete investigation expands from 3 seeds (0, 1, 2) to **ten random seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9**.

### 4.2 Matrix of Experimental Conditions
- **Dataset**: UCI Spambase (4,601 rows, 57 features, stratified 80/20 train/test split per seed).
- **Poisoning Rates**: 0%, 1%, 5%, 10%, 20%.
- **Attacks**: Clean Baseline, Random Label Flipping, Targeted Spam-to-Ham.
- **Models**:
  - Standardized Logistic Regression (`solver="liblinear"`, `max_iter=2000`, `random_state=seed`)
  - Decision Tree (`max_depth=8`, `min_samples_leaf=5`, `random_state=seed`)
  - Random Forest (`n_estimators=200`, `min_samples_leaf=2`, `random_state=seed`)
- **Total Experimental Runs**:
  - Baseline: 10 seeds $\times$ 3 models = 30 rows
  - Undefended Poisoned: 10 seeds $\times$ 2 attacks $\times$ 4 rates $\times$ 3 models = 240 rows
  - Defended Pipeline: 10 seeds $\times$ 2 attacks $\times$ 4 rates $\times$ 3 models = 240 rows
  - **Total Stage 4 Grid: 510 evaluations**.

### 4.3 Expanded Metric Set
Every condition records:
1. `seed` (0–9)
2. `attack` (`clean`, `random`, `targeted`)
3. `poison_rate` (0.0, 0.01, 0.05, 0.10, 0.20)
4. `defense_applied` (`none`, `knn_audit`, `prune_audit`)
5. `model` (LR, DT, RF)
6. `n_poisoned` (exact count)
7. `n_detected` (auditor flags)
8. `audit_precision`, `audit_recall`, `audit_fpr`
9. `accuracy`, `spam_precision`, `spam_recall`, `spam_f1`
10. `recovery_recall`, `recovery_f1`
11. Confusion matrix entries (`tn`, `fp`, `fn`, `tp`)

---

## 5. Artifact Saving & Reproducibility Rules

1. **Repository Structure**:
   - `results/stage3_pilot_results.csv`: Frozen 81-row pilot benchmark.
   - `results/stage4_full_results.csv`: Complete 10-seed undefended and defended benchmark.
   - `figures/`: High-resolution vector figures (PDF/PNG at 300 DPI) for conference inclusion.
2. **Invariants & Testing**:
   - All tests in `tests/test_attacks.py` and `tests/test_defense.py` must achieve 100% pass rate before experiment execution.
   - Every script must execute from clean CLI: `python scripts/run_attack_experiments.py`.
   - Git commit hash must be embedded into the header of generated result tables.
