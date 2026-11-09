# Adversarial Data Poisoning & Defense on Classical Spam Classifiers (Stage 3 & Stage 4 Weeks 1–3)

This repository contains the complete codebase, unit test suite (25 tests passing), experimental pipelines, audited Stage 3 pilot results (81 runs), Stage 4 Week 1 full 10-seed attack results (270 runs), Stage 4 Week 2 primary $k$-NN defense results (240 runs), Stage 4 Week 3 extended 5-fold out-of-fold logistic-regression defense comparison results (60 runs), publication-ready 300-DPI figures (`fig1`–`fig10`), frozen protocol, and LaTeX manuscript on the **UCI Spambase** dataset.

---

## 🔬 Core Research Question
> **How do poisoning strategy and poisoning rate affect the reliability of classical spam classifiers, and how effectively can two manually implemented label-auditing methods recover lost performance?**

---

## 📁 Repository Architecture

```text
spambase_project/
├── data/                                    # Official UCI Spambase dataset & documentation
│   ├── spambase.data                        # 4,601 email instances (57 features + binary label)
│   ├── spambase.names                       # Feature definitions & class statistics
│   └── spambase.DOCUMENTATION               # HP Labs dataset technical documentation
├── src/                                     # Modular, reusable Python source files
│   ├── __init__.py
│   ├── data.py                              # Dataset loading, summary checks, stratified 80/20 split
│   ├── attacks.py                           # random_label_flip and targeted_spam_to_ham
│   ├── defense.py                           # Primary k-NN (k=10, τ=0.5) & Extended 5-fold OOF LogReg (τ=0.8) auditing
│   ├── models.py                            # Standardized LogReg, Decision Tree, Random Forest
│   ├── metrics.py                           # Precision, recall, F1, accuracy, confusion matrices
│   ├── visualization.py                     # EDA, confusion matrices, error-band curves, 2x2 grid, crossover plots
│   └── experiments.py                       # Multi-seed attack & defense batch runners & programmatic table aggregators
├── tests/                                   # Automated unit test suite (25 unit tests passing)
│   ├── test_attacks.py                      # 17 unit tests verifying exact counts, directions, isolation
│   └── test_defense.py                      # 8 unit tests verifying self-exclusion, exact scores, 1 OOF/sample, isolation
├── notebooks/                               # Interactive research notebooks (all cells executed & verified)
│   ├── 01_baseline_and_first_attack.ipynb   # Stage 3 Week 1: EDA, threat model, baseline & 5% random flip
│   ├── 02_attack_validation.ipynb           # Stage 3 Week 2: Attack validation tables & class distribution shifts
│   ├── 03_pilot_experiment.ipynb            # Stage 3 Week 3: 3-seed pilot study, Mean ± 1 Std error-band curves
│   ├── 04_pilot_analysis.ipynb              # Stage 3 Week 4: 5-point pilot audit, 3 hypotheses & crossover analysis
│   ├── 05_full_attack_analysis.ipynb        # Stage 4 Week 1: 10-seed full attack analysis (270 runs), fig1–fig6
│   ├── 06_knn_defense_evaluation.ipynb      # Stage 4 Week 2: Primary k-NN defense evaluation (240 runs), fig7–fig8
│   └── 07_defense_comparison.ipynb          # Stage 4 Week 3: k-NN vs. 5-fold OOF comparison at 10% (120 runs), fig9–fig10
├── scripts/                                 # Standalone CLI experiment runners
│   ├── run_attack_experiments.py            # Attack batch runner (--stage stage4 for 270 rows, --stage stage3 for 81 rows)
│   └── run_defense_experiments.py           # Defense batch runner (--defense knn, --defense oof, or --defense both)
├── results/                                 # Audited experimental datasets, tables & publication figures
│   ├── stage3_pilot_results.csv             # Verified 81-row Stage 3 pilot dataset (Seeds 0–2)
│   ├── attack_results.csv                   # Verified 270-row Stage 4 full attack dataset (Seeds 0–9: 30 clean + 240 poisoned)
│   ├── knn_defense_results.csv              # Verified 240-row Stage 4 Week 2 k-NN defense dataset (Seeds 0–9, 1%–20%)
│   ├── oof_defense_results.csv              # Verified 60-row Stage 4 Week 3 5-fold OOF defense dataset (Seeds 0–9, 10%)
│   ├── tables/                              # Programmatically generated summary, detection, recovery & error analysis CSVs
│   └── figures/                             # 11 publication-ready 300-DPI figures (fig1–fig10)
├── paper/                                   # Academic conference paper structure (IEEEtran)
│   ├── main.tex                             # Drafted Intro, Background, Threat Model, Methods, Attack & Defense Results, Discussion, Ethics
│   ├── figures/                             # Publication figures linked in main.tex
│   └── references.bib                       # Formatted BibTeX references
├── logs/                                    # Weekly research logs (200–300 words each)
│   ├── week1.md                             # Stage 3 Week 1 log
│   ├── week2.md                             # Stage 3 Week 2 log
│   ├── week3.md                             # Stage 3 Week 3 log
│   ├── week4.md                             # Stage 3 Week 4 log & post-audit numerical correction note
│   ├── week5.md                             # Stage 4 Week 1 log (10-seed full attack study)
│   ├── week6.md                             # Stage 4 Week 2 log (Primary k-NN defense implementation & evaluation)
│   └── week7.md                             # Stage 4 Week 3 log (5-fold OOF extension & two-defense comparison)
├── stage3_protocol.md                       # Frozen experimental protocol & Stage 4 two-defense comparison blueprint
├── pytest.ini                               # Pytest path configuration
└── README.md                                # Project overview and reproduction guide
```

---

## 📊 Stage 4 Week 1: Full 10-Seed Attack Benchmark Summary (Seeds 0–9 — 270 Runs)

*Programmatically generated directly from `results/attack_results.csv` (Mean $\pm$ Sample Std across seeds 0–9):*

| Model | Clean (0%) Recall | Clean (0%) F1 | Random (20%) Recall | Targeted (10%) Recall | Targeted (20%) Recall | Targeted (20%) Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Std)** | $0.8848 \pm 0.0174$ | $0.9014 \pm 0.0109$ | $0.7989 \pm 0.0268$ (-8.60%) | $0.6573 \pm 0.0296$ (-22.75%) | **$0.2154 \pm 0.0364$** (-66.94%) | $0.9563 \pm 0.0254$ |
| **Decision Tree** | $0.8708 \pm 0.0253$ | $0.8899 \pm 0.0129$ | $0.8091 \pm 0.0231$ (-6.17%) | **$0.7760 \pm 0.0347$** (-9.48%) | $0.4507 \pm 0.0967$ (-42.01%) | $0.9435 \pm 0.0247$ |
| **Random Forest (200 trees)** | **$0.9198 \pm 0.0144$** | **$0.9366 \pm 0.0067$** | **$0.8769 \pm 0.0208$** (-4.30%) | $0.7606 \pm 0.0156$ (-15.92%) | $0.3146 \pm 0.0250$ (-60.52%) | **$0.9881 \pm 0.0103$** |

---

## 🛡️ Stage 4 Weeks 2–3: Two-Defense Head-to-Head Comparison at 10% Poisoning (Seeds 0–9)

*Programmatically generated from `results/knn_defense_results.csv` and `results/oof_defense_results.csv`:*

### 1. Label-Auditing Detection Performance & False-Removal Cost ($N_{\text{poison}} = 368$)

| Attack (10%) | Defense Method | Flagged Samples | Clean Samples Removed | Detection Precision | Detection Recall | Detection F1 | Audit Runtime (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random (10%)** | **$k$-NN ($k=10, \tau=0.5$)** | $773.4 \pm 18.3$ | $434.7 \pm 17.3$ (13.12%) | $0.4381 \pm 0.0102$ | **$0.9204 \pm 0.0136$** | $0.5936 \pm 0.0106$ | **$0.0549 \pm 0.0005$** |
| **Random (10%)** | **5-Fold OOF LogReg ($\tau=0.8$)** | $216.4 \pm 8.7$ | **$36.6 \pm 3.9$ (1.11%)** | **$0.8309 \pm 0.0168$** | $0.4886 \pm 0.0217$ | **$0.6151 \pm 0.0196$** | $0.0562 \pm 0.0043$ |
| **Targeted (10%)** | **$k$-NN ($k=10, \tau=0.5$)** | $763.4 \pm 24.5$ | $484.0 \pm 18.5$ (14.61%) | $0.3661 \pm 0.0100$ | **$0.7592 \pm 0.0287$** | **$0.4939 \pm 0.0135$** | **$0.0545 \pm 0.0005$** |
| **Targeted (10%)** | **5-Fold OOF LogReg ($\tau=0.8$)** | $124.2 \pm 6.7$ | **$53.0 \pm 4.0$ (1.60%)** | **$0.5729 \pm 0.0290$** | $0.1935 \pm 0.0162$ | $0.2891 \pm 0.0213$ | $0.0970 \pm 0.0030$ |

### 2. Downstream Classifier Recovery at 10% Poisoning (Seeds 0–9)

| Attack (10%) | Classifier | Defense Method | Clean Recall | Poisoned Recall | Defended Recall | Recall Recovery | Poisoned F1 | Defended F1 | F1 Recovery |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random (10%)** | Logistic Regression | $k$-NN ($\tau=0.5$) | $0.8848$ | $0.8303$ | $0.8702 \pm 0.0176$ | $+73.2\%$ | $0.8702$ | $0.8907 \pm 0.0102$ | $+65.6\%$ |
| **Random (10%)** | Logistic Regression | 5-Fold OOF ($\tau=0.8$) | $0.8848$ | $0.8303$ | **$0.8934 \pm 0.0185$** | **$+115.7\%$** | $0.8702$ | **$0.9023 \pm 0.0105$** | **$+102.7\%$** |
| **Random (10%)** | Decision Tree | $k$-NN ($\tau=0.5$) | $0.8708$ | $0.8468$ | $0.8314 \pm 0.0290$ | $-64.4\%$ | $0.8740$ | $0.8701 \pm 0.0173$ | $-24.2\%$ |
| **Random (10%)** | Decision Tree | 5-Fold OOF ($\tau=0.8$) | $0.8708$ | $0.8468$ | **$0.8592 \pm 0.0247$** | **$+51.7\%$** | $0.8740$ | **$0.8774 \pm 0.0162$** | **$+21.7\%$** |
| **Random (10%)** | Random Forest | $k$-NN ($\tau=0.5$) | $0.9198$ | $0.9102$ | $0.8815 \pm 0.0174$ | $-297.1\%$ | $0.9279$ | $0.9174 \pm 0.0090$ | $-121.5\%$ |
| **Random (10%)** | Random Forest | 5-Fold OOF ($\tau=0.8$) | $0.9198$ | $0.9102$ | **$0.9157 \pm 0.0158$** | **$+57.1\%$** | $0.9279$ | **$0.9300 \pm 0.0097$** | **$+24.3\%$** |
| **Targeted (10%)** | Logistic Regression | $k$-NN ($\tau=0.5$) | $0.8848$ | $0.6573$ | **$0.7488 \pm 0.0311$** | **$+40.2\%$** | $0.7747$ | **$0.8313 \pm 0.0182$** | **$+44.6\%$** |
| **Targeted (10%)** | Logistic Regression | 5-Fold OOF ($\tau=0.8$) | $0.8848$ | $0.6573$ | $0.7198 \pm 0.0326$ | $+27.5\%$ | $0.7747$ | $0.8183 \pm 0.0206$ | $+34.4\%$ |
| **Targeted (10%)** | Decision Tree | $k$-NN ($\tau=0.5$) | $0.8708$ | $0.7760$ | $0.7408 \pm 0.0290$ | $-37.2\%$ | $0.8426$ | $0.8279 \pm 0.0189$ | $-31.1\%$ |
| **Targeted (10%)** | Decision Tree | 5-Fold OOF ($\tau=0.8$) | $0.8708$ | $0.7760$ | $0.7518 \pm 0.0351$ | $-25.6\%$ | $0.8426$ | $0.8301 \pm 0.0217$ | $-26.5\%$ |
| **Targeted (10%)** | Random Forest | $k$-NN ($\tau=0.5$) | $0.9198$ | $0.7606$ | $0.7672 \pm 0.0211$ | $+4.2\%$ | $0.8547$ | $0.8575 \pm 0.0141$ | $+3.3\%$ |
| **Targeted (10%)** | Random Forest | 5-Fold OOF ($\tau=0.8$) | $0.9198$ | $0.7606$ | **$0.7796 \pm 0.0196$** | **$+11.9\%$** | $0.8547$ | **$0.8678 \pm 0.0117$** | **$+16.0\%$** |

---

## 📊 Audited Stage 3 Pilot Benchmark Summary (Seeds 0, 1, 2 — 81 Runs)

*Programmatically generated directly from `results/stage3_pilot_results.csv` (Mean $\pm$ Sample Std across seeds 0, 1, 2):*

| Model | Clean (0%) Recall | Clean (0%) F1 | Random (20%) Recall | Targeted (20%) Recall | Targeted (20%) Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Std)** | $0.8972 \pm 0.0111$ | $0.9114 \pm 0.0050$ | $0.8191 \pm 0.0214$ (-7.81%) | **$0.1928 \pm 0.0208$** (-70.43%) | $0.9365 \pm 0.0205$ |
| **Decision Tree** | $0.8861 \pm 0.0016$ | $0.9010 \pm 0.0028$ | $0.8017 \pm 0.0225$ (-8.45%) | $0.5106 \pm 0.0777$ (-37.56%) | $0.9590 \pm 0.0236$ |
| **Random Forest (200 trees)** | **$0.9137 \pm 0.0080$** | **$0.9360 \pm 0.0024$** | **$0.8770 \pm 0.0177$** (-3.67%) | $0.3039 \pm 0.0396$ (-60.97%) | **$0.9847 \pm 0.0065$** |

---

## 🚀 Reproduction Guide

```bash
# 1. Run all 25 unit tests (17 attack tests + 8 defense tests)
pytest -v

# 2. Run the Stage 4 10-seed full attack study (270 rows in results/attack_results.csv)
python scripts/run_attack_experiments.py --stage stage4

# 3. Run the Stage 4 Week 2 k-NN defense (240 rows) & Week 3 5-fold OOF defense (60 rows)
python scripts/run_defense_experiments.py --defense both

# 4. Open interactive notebooks
jupyter notebook
```
