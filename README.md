# Adversarial Data Poisoning & Defense on Classical Spam Classifiers (Stage 3)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests: pytest](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Pilot: 81/81 Verified](https://img.shields.io/badge/pilot-81%20runs%20verified-success.svg)](results/stage3_pilot_results.csv)

This repository contains the complete codebase, unit test suite, experimental pipeline, audited pilot results, frozen protocol, and LaTeX manuscript draft for **Stage 3: Data Poisoning Foundations** on the **UCI Spambase** dataset.

---

## 🔬 Core Research Question
> **How do poisoning strategy and poisoning rate affect the reliability of classical spam classifiers, and how much performance can a manually implemented label-auditing defense recover?**

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
│   ├── visualization.py                     # EDA, confusion matrices, error-band curves, 2x2 grid
│   └── experiments.py                       # Multi-seed batch runner (run_single_condition, run_pilot_study)
├── tests/                                   # Automated unit test suite
│   └── test_attacks.py                      # 17 unit tests verifying exact counts, directions, isolation
├── notebooks/                               # Interactive research notebooks (all cells executed & verified)
│   ├── 01_baseline_and_first_attack.ipynb   # Week 1: EDA, threat model, baseline & 5% random flip
│   ├── 02_attack_validation.ipynb           # Week 2: Attack validation tables & class distribution shifts
│   ├── 03_pilot_experiment.ipynb            # Week 3: 3-seed pilot study, Mean ± 1 Std error-band curves
│   └── 04_pilot_analysis.ipynb              # Week 4: 5-point pilot audit, 3 hypotheses & crossover analysis
├── scripts/                                 # Standalone CLI experiment runners
│   └── run_attack_experiments.py            # Batch runner generating results/stage3_pilot_results.csv
├── results/                                 # Audited experimental datasets
│   └── stage3_pilot_results.csv             # Verified 81-row pilot dataset (9 clean + 72 poisoned)
├── paper/                                   # Academic conference paper structure (IEEEtran)
│   ├── main.tex                             # Drafted Introduction, Background, Threat Model, and Methods
│   └── references.bib                       # 8 formatted BibTeX references (including Biggio 2012, Schwarzschild 2021)
├── logs/                                    # Weekly research logs (200–300 words each)
│   ├── week1.md                             # Week 1 log
│   ├── week2.md                             # Week 2 log
│   ├── week3.md                             # Week 3 log
│   └── week4.md                             # Week 4 log & Stage 3 reflection
├── stage3_protocol.md                       # Frozen experimental protocol & Stage 4 defense blueprint
├── pytest.ini                               # Pytest path configuration
└── README.md                                # Project overview and reproduction guide
```

---

## 📊 Audited Pilot Benchmark Summary (Seeds 0, 1, 2 — 81 Runs)

| Model | Clean (0%) Recall | Clean (0%) F1 | Random (20%) Recall | Targeted (20%) Recall | Targeted (20%) Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Std)** | $0.8972 \pm 0.011$ | $0.9114 \pm 0.005$ | $0.8062 \pm 0.012$ | **$0.1922 \pm 0.015$** (-70.50\%) | $0.9365 \pm 0.008$ |
| **Decision Tree** | $0.8861 \pm 0.002$ | $0.9010 \pm 0.003$ | $0.8246 \pm 0.021$ | $0.5110 \pm 0.043$ (-37.51\%) | $0.9590 \pm 0.012$ |
| **Random Forest (200 trees)** | **$0.9137 \pm 0.008$** | **$0.9360 \pm 0.002$** | **$0.8329 \pm 0.009$** | $0.3054 \pm 0.018$ (-60.83\%) | **$0.9847 \pm 0.004$** |

---

## 🚀 Reproduction Guide

```bash
# 1. Run all 17 unit tests
pytest -v

# 2. Re-run the 3-seed pilot study (generates 81 rows in results/stage3_pilot_results.csv)
python scripts/run_attack_experiments.py

# 3. Open interactive notebooks
jupyter notebook
```
