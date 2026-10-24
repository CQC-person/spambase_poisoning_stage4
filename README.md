# Adversarial Data Poisoning & Defense on Classical Spam Classifiers (Stage 3 & Stage 4 Week 1)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests: pytest](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Stage 3 Pilot: 81/81 Verified](https://img.shields.io/badge/pilot-81%20runs%20verified-success.svg)](results/stage3_pilot_results.csv)
[![Stage 4 Attack Study: 270/270 Verified](https://img.shields.io/badge/stage4%20attacks-270%20runs%20verified-success.svg)](results/attack_results.csv)

This repository contains the complete codebase, unit test suite, experimental pipeline, audited Stage 3 pilot results (81 runs), Stage 4 Week 1 full 10-seed attack results (270 runs), publication-ready figures, frozen protocol, and LaTeX manuscript draft on the **UCI Spambase** dataset.

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
│   ├── models.py                            # Standardized LogReg, Decision Tree, Random Forest
│   ├── metrics.py                           # Precision, recall, F1, accuracy, confusion matrices
│   ├── visualization.py                     # EDA, confusion matrices, error-band curves, 2x2 grid, crossover plots
│   └── experiments.py                       # Multi-seed batch runners (Stage 3 81-row & Stage 4 270-row) & table aggregators
├── tests/                                   # Automated unit test suite
│   └── test_attacks.py                      # 17 unit tests verifying exact counts, directions, isolation
├── notebooks/                               # Interactive research notebooks (all cells executed & verified)
│   ├── 01_baseline_and_first_attack.ipynb   # Stage 3 Week 1: EDA, threat model, baseline & 5% random flip
│   ├── 02_attack_validation.ipynb           # Stage 3 Week 2: Attack validation tables & class distribution shifts
│   ├── 03_pilot_experiment.ipynb            # Stage 3 Week 3: 3-seed pilot study, Mean ± 1 Std error-band curves
│   ├── 04_pilot_analysis.ipynb              # Stage 3 Week 4: 5-point pilot audit, 3 hypotheses & crossover analysis
│   └── 05_full_attack_analysis.ipynb        # Stage 4 Week 1: 10-seed full attack analysis (270 runs), 6 figures & confusion matrices
├── scripts/                                 # Standalone CLI experiment runners
│   └── run_attack_experiments.py            # Batch runner (--stage stage4 for 270 rows, --stage stage3 for 81 rows)
├── results/                                 # Audited experimental datasets, tables & publication figures
│   ├── stage3_pilot_results.csv             # Verified 81-row Stage 3 pilot dataset (Seeds 0–2)
│   ├── attack_results.csv                   # Verified 270-row Stage 4 full attack dataset (Seeds 0–9: 30 clean + 240 poisoned)
│   ├── tables/                              # Programmatically generated summary & confusion matrix CSV tables
│   └── figures/                             # 300-DPI publication-ready figures (fig1–fig6)
├── paper/                                   # Academic conference paper structure (IEEEtran)
│   ├── main.tex                             # Drafted Intro, Background, Threat Model, Methods, and 10-Seed Attack Results
│   ├── figures/                             # Publication figures linked in main.tex
│   └── references.bib                       # Formatted BibTeX references (Biggio 2012, Schwarzschild 2021, NIST 2024, etc.)
├── logs/                                    # Weekly research logs (200–300 words each)
│   ├── week1.md                             # Stage 3 Week 1 log
│   ├── week2.md                             # Stage 3 Week 2 log
│   ├── week3.md                             # Stage 3 Week 3 log
│   ├── week4.md                             # Stage 3 Week 4 log & post-audit numerical correction note
│   └── week5.md                             # Stage 4 Week 1 log (10-seed full attack study & analysis)
├── stage3_protocol.md                       # Frozen experimental protocol & Stage 4 two-defense comparison blueprint
├── pytest.ini                               # Pytest path configuration
└── README.md                                # Project overview and reproduction guide
```

---

## 📊 Stage 4 Full 10-Seed Attack Benchmark Summary (Seeds 0–9 — 270 Runs)

*Programmatically generated directly from `results/attack_results.csv` (Mean $\pm$ Sample Std across seeds 0–9):*

| Model | Clean (0%) Recall | Clean (0%) F1 | Random (20%) Recall | Targeted (10%) Recall | Targeted (20%) Recall | Targeted (20%) Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Std)** | $0.8848 \pm 0.0174$ | $0.9014 \pm 0.0109$ | $0.7989 \pm 0.0268$ (-8.60%) | $0.6573 \pm 0.0296$ (-22.75%) | **$0.2154 \pm 0.0364$** (-66.94%) | $0.9563 \pm 0.0254$ |
| **Decision Tree** | $0.8708 \pm 0.0253$ | $0.8899 \pm 0.0129$ | $0.8091 \pm 0.0231$ (-6.17%) | **$0.7760 \pm 0.0347$** (-9.48%) | $0.4507 \pm 0.0967$ (-42.01%) | $0.9435 \pm 0.0247$ |
| **Random Forest (200 trees)** | **$0.9198 \pm 0.0144$** | **$0.9366 \pm 0.0067$** | **$0.8769 \pm 0.0208$** (-4.30%) | $0.7606 \pm 0.0156$ (-15.92%) | $0.3146 \pm 0.0250$ (-60.52%) | **$0.9881 \pm 0.0103$** |

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
# 1. Run all 17 unit tests
pytest -v

# 2. Run the Stage 4 10-seed full attack study (generates 270 rows in results/attack_results.csv)
python scripts/run_attack_experiments.py --stage stage4

# 3. Re-run the Stage 3 3-seed pilot study (generates 81 rows in results/stage3_pilot_results.csv)
python scripts/run_attack_experiments.py --stage stage3

# 4. Open interactive notebooks
jupyter notebook
```
