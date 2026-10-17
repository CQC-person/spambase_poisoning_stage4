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
* **Isolation Invariant:** The test set ($X_{\text{test}}, y_{\text{test}}$) is partitioned *before* any poisoning or feature scaling operation and remains bit-for-bit unchanged across all attack and defense evaluations.

### 2.2 Random Seeds
* **Stage 3 Pilot Seeds:** `seeds = [0, 1, 2]` ($3$ seeds).
* **Stage 4 Full Experiment Seeds:** `seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` ($10$ seeds).
* **Seed Propagation Rule:** Within each condition, the active `seed` is passed identically to:
  1. `train_test_split(..., random_state=seed)`
  2. `random_label_flip(..., random_state=seed)` / `targeted_spam_to_ham(..., random_state=seed)`
  3. All three model constructors (`random_state=seed`).

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
1. **Logistic Regression:** `Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(solver='liblinear', max_iter=2000, random_state=seed))])` — `StandardScaler` is fit strictly on `X_train` to prevent preprocessing leakage.
2. **Decision Tree:** `DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=seed)`
3. **Random Forest:** `RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed)`

### 2.5 Evaluation Metrics
* **Primary Metrics:**
  * `spam_recall` (Class 1 Recall / True Positive Rate): $\frac{\text{TP}}{\text{TP} + \text{FN}}$
  * `spam_f1` (Class 1 F1-Score): $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
* **Supporting Classification Metrics:**
  * `spam_precision` (Class 1 Precision): $\frac{\text{TP}}{\text{TP} + \text{FP}}$
  * `accuracy` (Overall Accuracy): $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
  * Confusion Matrix counts (`tn`, `fp`, `fn`, `tp`)
* **Stage 4 Defense Evaluation Metrics:**
  * **Auditing Detection Precision & Recall:** Evaluated by comparing the defense's flagged sample mask against the ground-truth `flipped_mask`.
  * **Performance Recovery Ratio:**
    $$R_{\text{recovery}}(M) = \frac{M_{\text{defended}} - M_{\text{poisoned}}}{M_{\text{clean}} - M_{\text{poisoned}}}, \quad M \in \{\text{Spam Recall}, \text{Spam F1}\}$$

### 2.6 Standardized Filenames & Directory Structure
* `src/data.py`, `src/attacks.py`, `src/models.py`, `src/metrics.py`, `src/experiments.py`, `src/visualization.py`
* `tests/test_attacks.py`
* `scripts/run_attack_experiments.py`
* `results/stage3_pilot_results.csv` (Frozen 81-row Stage 3 benchmark)
* `results/stage4_full_results.csv` (Reserved for 10-seed Stage 4 benchmark)
* `notebooks/01_baseline_and_first_attack.ipynb`
* `notebooks/02_attack_validation.ipynb`
* `notebooks/03_pilot_experiment.ipynb`
* `notebooks/04_pilot_analysis.ipynb`
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

## 3. Stage 4 Defense Integration Plan (Label Auditing)
In Stage 4, the manually implemented $k$-NN neighborhood label-auditing defense will be inserted strictly between the poisoning step and model fitting (**Filter-then-Train**):
1. Standardize `X_train` using `StandardScaler` fit only on `X_train`.
2. For each training instance $x_i$, query its $k$ nearest neighbors ($k=10$) in Euclidean feature space and compute the neighbor label disagreement ratio $S(x_i, y_i)$.
3. Flag instances with $S(x_i, y_i) > \tau$ and sanitize `y_train_poisoned` (comparing majority-vote relabeling vs. sample pruning) before fitting the three classifiers.

---

## 4. Protocol Change Log

| Date | Version | Component Modified | Reason for Change | Approved By |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-17 | `v1.0-frozen` | Initial Protocol Freeze | Completed 5-point Stage 3 pilot audit (81/81 rows verified) | Quancheng |
