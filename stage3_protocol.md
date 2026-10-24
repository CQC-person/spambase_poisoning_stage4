# Stage 3 Frozen Experimental Protocol & Stage 4 Blueprint

## 1. Purpose & Benchmarking Principles
Following the benchmarking standards established in *"Just How Toxic Is Data Poisoning?"* (Schwarzschild et al., ICML 2021), data poisoning evaluations are highly sensitive to variations in dataset partitioning, hyperparameter selection, and preprocessing pipelines. This document formally **freezes** the experimental design verified during the Stage 3 pilot study ($81$ runs across seeds $0, 1, 2$) before scaling to the full 10-seed investigation and defense evaluation in Stage 4.

Any subsequent modification to the parameters defined below must be explicitly recorded in the **Protocol Change Log** (Section 4).

---

## 2. Frozen Experimental Specifications

### 2.1 Dataset & Split Procedure
* **Dataset:** UCI Spambase (`data/spambase.data`, `data/spambase.names`), comprising $4,601$ email instances ($2,788$ non-spam $[60.60\%]$, $1,813$ spam $[39.40\%]$) and $57$ continuous numerical features.
* **Partitioning Function:** `src.data.split_data(df, test_size=0.2, random_state=seed)`
* **Partition Ratio:** Stratified $80/20$ train-test split ($3,680$ training instances, $921$ test instances).
* **Isolation Invariant:** The test set ($X_{\text{test}}, y_{\text{test}}$) is partitioned *before* any poisoning, defense auditing, or feature scaling operation and remains bit-for-bit unchanged across all attack and defense evaluations. Clean labels and ground-truth poisoning masks (`flipped_mask`) are never used during defense flagging and are consulted strictly post-hoc for auditing evaluation.

### 2.2 Random Seeds
* **Stage 3 Pilot Seeds:** `seeds = [0, 1, 2]` ($3$ seeds).
* **Stage 4 Full Experiment Seeds:** `seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` ($10$ seeds).
* **Seed Propagation Rule:** Within each condition, the active `seed` is passed identically to:
  1. `train_test_split(..., random_state=seed)`
  2. `random_label_flip(..., random_state=seed)` / `targeted_spam_to_ham(..., random_state=seed)`
  3. Out-of-fold cross-validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`) and auditing estimators
  4. All three downstream classifier constructors (`random_state=seed`).

### 2.3 Poisoning Strategies & Rates
* **Poisoning Rate Definition:**
  $$\alpha = \frac{N_{\text{poison}}}{N_{\text{train}}}, \quad N_{\text{poison}} = \text{round}(3680 \times \alpha)$$
* **Frozen Rate Grid:** `poison_rates = [0.0, 0.01, 0.05, 0.10, 0.20]` corresponding to exactly $\{0, 37, 184, 368, 736\}$ modified training labels.
* **Attack Modalities:**
  1. `clean` ($\alpha = 0.0$): Unmodified training labels.
  2. `random` (`src.attacks.random_label_flip`): Uniform selection without replacement across all $3,680$ training samples; bidirectional inversion ($y_i \leftarrow 1 - y_i$).
  3. `targeted` (`src.attacks.targeted_spam_to_ham`): Uniform selection without replacement restricted strictly to training spam samples ($y_i = 1$); unidirectional inversion ($1 \to 0$).

### 2.4 Models & Fixed Hyperparameters
All models are instantiated via `src.models.get_models(random_state=seed)` with frozen hyperparameters:
1. **Logistic Regression:** `Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(solver='liblinear', max_iter=2000, random_state=seed))])` — `StandardScaler` is fit strictly on training features to prevent preprocessing leakage.
2. **Decision Tree:** `DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=seed)`
3. **Random Forest:** `RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed)`

### 2.5 Evaluation Metrics
* **Primary Classification Metrics (on Clean Test Set):**
  * `spam_recall` (Class 1 Recall / True Positive Rate): $\frac{\text{TP}}{\text{TP} + \text{FN}}$
  * `spam_f1` (Class 1 F1-Score): $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
* **Supporting Classification Metrics:**
  * `spam_precision` (Class 1 Precision): $\frac{\text{TP}}{\text{TP} + \text{FP}}$
  * `accuracy` (Overall Accuracy): $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
  * Confusion Matrix counts (`tn`, `fp`, `fn`, `tp`)
* **Stage 4 Defense & Comparison Metrics:**
  * **Detection Precision (`detection_precision`):** Fraction of flagged training instances that are truly poisoned ($\frac{|\text{Flagged} \cap \text{Poisoned}|}{|\text{Flagged}|}$).
  * **Detection Recall (`detection_recall`):** Fraction of truly poisoned training instances successfully flagged ($\frac{|\text{Flagged} \cap \text{Poisoned}|}{|\text{Poisoned}|}$).
  * **Detection F1 (`detection_f1`):** Harmonic mean of `detection_precision` and `detection_recall`.
  * **Number Flagged (`n_flagged`):** Total number of training samples flagged and removed by the defense.
  * **Clean Samples Incorrectly Removed (`n_clean_removed` and `frac_clean_removed`):** Count $|\text{Flagged} \cap \text{Clean}|$ and fraction $\frac{|\text{Flagged} \cap \text{Clean}|}{|\text{Clean}|}$ of unpoisoned training instances erroneously discarded (false-removal cost).
  * **Runtime (`runtime_sec`):** Wall-clock execution time (in seconds) required to run the label-auditing defense and retrain the downstream model per condition.
  * **Performance Recovery Ratio (`recovery_spam_recall`, `recovery_spam_f1`):**
    $$R_{\text{recovery}}(M) = \frac{M_{\text{defended}} - M_{\text{poisoned}}}{M_{\text{clean}} - M_{\text{poisoned}}}, \quad M \in \{\text{Spam Recall}, \text{Spam F1}\}$$
    *(Negative or undefined values when $M_{\text{clean}} \approx M_{\text{poisoned}}$ are retained and explicitly analyzed rather than discarded.)*

### 2.6 Standardized Filenames & Directory Structure
* `src/data.py`, `src/attacks.py`, `src/models.py`, `src/metrics.py`, `src/experiments.py`, `src/visualization.py`, `src/defense.py`
* `tests/test_attacks.py`, `tests/test_defense.py`
* `scripts/run_attack_experiments.py`
* `results/stage3_pilot_results.csv` (Frozen 81-row Stage 3 benchmark; single source of truth for Stage 3 summaries)
* `results/attack_results.csv` (270-row Stage 4 10-seed full attack benchmark: 30 clean + 240 poisoned rows)
* `results/knn_defense_results.csv` (Stage 4 primary $k$-NN label-auditing defense evaluation across nonzero poisoning rates and 10 seeds)
* `results/oof_defense_results.csv` (Stage 4 extended 5-fold out-of-fold logistic-regression defense comparison at 10% poisoning across 10 seeds)
* `notebooks/01_baseline_and_first_attack.ipynb` through `notebooks/04_pilot_analysis.ipynb` (Stage 3 notebooks)
* `notebooks/05_full_attack_analysis.ipynb`, `notebooks/06_knn_defense_evaluation.ipynb`, `notebooks/07_defense_comparison.ipynb` (Stage 4 notebooks)
* `paper/main.tex` and `paper/references.bib`

### 2.7 Plotting & Visualization Rules
* **Resolution:** Minimum `120 DPI` inline, `300 DPI` for exported publication figures.
* **Color Palette (Colorblind-Safe):**
  * Logistic Regression / Ham Class: `#2b5c8f` (Deep Blue, solid line `-`, circle marker `o`)
  * Decision Tree: `#7570b3` (Purple, dashed line `--`, triangle marker `^`)
  * Random Forest: `#1b9e77` (Teal Green, dash-dot line `-.`, square marker `s`)
  * Poisoned / Spam Class Highlight: `#d95f02` (Burnt Orange)
* **Uncertainty Representation:** All multi-seed line plots must display the cross-seed **Mean** curve accompanied by a shaded **$\pm 1\ \text{Std}$** error band (`alpha=0.15`).

---

## 3. Stage 4 Defense Integration Plan (Primary $k$-NN Auditing & Extended Out-of-Fold Comparison)

Both label-auditing defenses are implemented manually in `src/defense.py` using only NumPy, Pandas, and scikit-learn primitives (without external label-cleaning libraries) and operate in a strict **Filter-then-Train** pipeline between label poisoning and downstream classifier fitting. Neither defense ever accesses test-set instances ($X_{\text{test}}, y_{\text{test}}$), clean training labels ($y_{\text{train}}^{\text{clean}}$), or ground-truth `flipped_mask` during scoring or flagging.

### 3.1 Primary Defense: $k$-NN Neighborhood-Disagreement Auditing
1. **Training-Only Feature Standardization:** Fit `StandardScaler` strictly on the poisoned training feature matrix $X_{\text{train}}$ and transform $X_{\text{train}}$ into standardized Euclidean space.
2. **Self-Excluded $k=10$ Neighbor Search:** For each training instance $x_i$, query $k + 1 = 11$ nearest neighbors in $X_{\text{train}}^{\text{scaled}}$ and explicitly exclude $x_i$ itself so that exactly $k = 10$ other training examples form its neighborhood $\mathcal{N}_k(i)$.
3. **Disagreement Suspicion Score & Fixed Threshold ($\tau = 0.5$):** Define the $k$-NN suspicion score as the fraction of the $10$ neighbors whose recorded (poisoned) labels disagree with $y_i^{\text{recorded}}$:
   $$S_{\text{kNN}}(x_i, y_i^{\text{recorded}}) = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} \mathbb{I}\!\left(y_j^{\text{recorded}} \neq y_i^{\text{recorded}}\right)$$
   Flag any training example with $S_{\text{kNN}}(x_i, y_i^{\text{recorded}}) \ge 0.5$.
4. **Sanitization & Retraining Scope:** Remove all flagged examples from $(X_{\text{train}}, y_{\text{train}}^{\text{recorded}})$ and retrain all three classifiers (`Logistic Regression`, `Decision Tree`, `Random Forest`) across both attacks (`random`, `targeted`), all nonzero poisoning rates ($\alpha \in \{1\%, 5\%, 10\%, 20\%\}$), and all $10$ seeds ($0$–$9$).

### 3.2 Extended Defense: Five-Fold Out-of-Fold (OOF) Logistic-Regression Auditing
1. **Cross-Validated Out-of-Fold Probability Estimation:** Partition $(X_{\text{train}}, y_{\text{train}}^{\text{recorded}})$ using `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`. Within each fold, fit a fresh `StandardScaler` and `LogisticRegression(solver='liblinear', max_iter=2000, random_state=seed)` strictly on the four training folds and predict class probabilities on the held-out fifth fold. Repeat across all five folds so that every training instance $x_i$ receives exactly one out-of-fold probability vector $\hat{p}_{\text{OOF}}(\cdot \mid x_i)$ without in-sample overfitting leakage.
2. **OOF Suspicion Score & Fixed Threshold ($\tau_{\text{OOF}} = 0.8$):** Define the model-based suspicion score as:
   $$S_{\text{OOF}}(x_i, y_i^{\text{recorded}}) = 1 - \hat{p}_{\text{OOF}}\!\left(y_i^{\text{recorded}} \mid x_i\right)$$
   Flag any training example with $S_{\text{OOF}}(x_i, y_i^{\text{recorded}}) \ge 0.8$.
3. **Sanitization & Retraining Scope:** Remove all flagged training examples and retrain all three downstream classifiers on the remaining sanitized training subset.

### 3.3 Head-to-Head Defense Comparison Protocol at 10% Poisoning
To compare geometry-based ($k$-NN disagreement) and model-based (5-fold OOF logistic regression) label auditing under identical experimental conditions:
* **Shared Evaluation Grid:** Both defenses are evaluated at $\alpha = 10\%$ poisoning ($N_{\text{poison}} = 368$) across both attack types (`random` and `targeted`), all three classifiers (`Logistic Regression`, `Decision Tree`, `Random Forest`), and all $10$ seeds ($s \in \{0, \dots, 9\}$), using the exact same train-test splits and poisoned sample indices.
* **Comparison Dimensions:**
  1. `detection_precision`, `detection_recall`, and `detection_f1` against ground-truth `flipped_mask`
  2. Total number of flagged samples (`n_flagged`) and number/fraction of clean training samples incorrectly removed (`n_clean_removed`, `frac_clean_removed`)
  3. Defense execution and retraining `runtime_sec`
  4. Downstream test-set recovery of `spam_recall` and `spam_f1` ($R_{\text{recovery}}$), alongside qualitative inspection of representative false positives, missed poisoned samples, and test-set errors on Seed $0$.

---

## 4. Protocol Change Log

| Date | Version | Component Modified | Reason for Change | Approved By |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-17 | `v1.0-frozen` | Initial Protocol Freeze | Completed 5-point Stage 3 pilot audit (81/81 rows verified) | Quancheng |
| 2026-10-24 | `v1.1-stage4-plan` | Section 2.5 (Defense Metrics) & Section 3 (Stage 4 Defense Integration Plan) | Updated Stage 4 defense design per advisor feedback and Stage 4 research plan: specified $k$-NN ($k=10, \tau=0.5$, self-excluded pruning) as primary defense across nonzero poisoning rates, added 5-fold out-of-fold logistic-regression auditing ($\tau_{\text{OOF}}=0.8$) as extended model-based defense, and defined the head-to-head comparison protocol at 10% poisoning (detection precision/recall/F1, clean samples incorrectly removed, runtime, and spam recall/F1 recovery). Also reconciled Stage 3 written summary tables programmatically from `results/stage3_pilot_results.csv`. | Yuanyi & Quancheng |
