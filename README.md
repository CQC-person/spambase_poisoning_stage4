# Adversarial Data Poisoning & Defense on Classical Spam Classifiers (Stage 3)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: pytest](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)](https://docs.pytest.org/)

This repository contains the complete codebase, experimental pipeline, empirical results, and documentation for **Stage 3: Data Poisoning Foundations** on the **UCI Spambase** dataset.

The investigation evaluates how classical spam classifiers (Standardized Logistic Regression, Decision Trees, and Random Forests) degrade under **Random Label Flipping** (availability threat) and **Targeted Spam-to-Ham Label Flipping** (integrity threat), and establishes the frozen protocol for a manually implemented **label-auditing defense** in Stage 4.

---

## 🔬 Core Research Question
> **How do poisoning strategy and poisoning rate affect the reliability of classical spam classifiers, and how much performance can a manually implemented label-auditing defense recover?**

---

## 📁 Repository Architecture

```text
spambase_project/
├── data/                                 # Official UCI Spambase dataset & documentation
│   ├── spambase.data                     # 4,601 email instances
│   ├── spambase.names                    # 57 continuous feature definitions & target
│   └── spambase.DOCUMENTATION            # HP Labs dataset technical documentation
├── src/                                  # Modular, reusable scientific computing library
│   ├── __init__.py
│   ├── data.py                           # Dataset loading, health checks, stratified 80/20 splitting
│   ├── attacks.py                        # random_label_flip and targeted_spam_to_ham
│   ├── models.py                         # Standardized LogReg, Decision Tree, Random Forest
│   ├── metrics.py                        # Precision, recall, F1, accuracy, confusion matrices
│   ├── visualization.py                  # EDA, heatmaps, error-band curves, 2x2 comparison grid
│   └── experiments.py                    # Multi-seed batch runner (run_single_condition, run_pilot_study)
├── tests/                                # Automated unit test suite
│   └── test_attacks.py                   # 17 unit tests verifying exact counts, directions, isolation
├── notebooks/                            # Interactive research reports with pre-rendered figures
│   ├── 01_baseline_and_first_attack.ipynb# Clean baseline & first 5% random poisoning demo
│   ├── 02_attack_validation.ipynb        # Attack validation tables, class distributions, threat analysis
│   └── 03_pilot_experiment.ipynb        # 3-seed pilot study, Mean ± 1 Std error curves, 2x2 grid
├── scripts/                              # Standalone CLI tools
│   └── run_attack_experiments.py         # Batch runner executing the full multi-seed pilot
├── results/                              # Frozen experimental datasets
│   └── stage3_pilot_results.csv          # Verified 81-row pilot dataset (9 clean + 72 poisoned)
├── paper/                                # Academic conference paper draft
│   ├── main.tex                          # IEEE conference paper source
│   └── references.bib                    # BibTeX literature citations
├── logs/                                 # Weekly structured research logs (200-300 words each)
│   ├── week1.md                          # Week 1 baseline & threat model log
│   ├── week2.md                          # Week 2 reusable attack implementations log
│   ├── week3.md                          # Week 3 multi-seed pilot runner log
│   └── week4.md                          # Week 4 pilot audit, protocol freeze & paper draft log
├── stage3_protocol.md                    # Frozen experimental protocol & Stage 4 defense blueprint
├── pytest.ini                            # Pytest configuration
└── README.md                             # Project overview and reproduction guide
```

---

## 📊 Dataset Profile (UCI Spambase)
* **Total Instances:** 4,601 email messages (Collected at HP Labs by Hopkins et al., 1999)
* **Features:** 57 continuous numerical attributes:
  * 48 word frequency attributes (`word_freq_WORD`, $[0, 100]\%$)
  * 6 punctuation frequency attributes (`char_freq_CHAR`: `;`, `(`, `[`, `!`, `$`, `#`)
  * 3 capital letter run length statistics (average, longest, total)
* **Target:** Binary class `is_spam` (1 = Spam, 0 = Non-Spam / Ham)
* **Class Distribution:** 2,788 Ham (60.60%) vs. 1,813 Spam (39.40%)

---

## 🎯 Threat Model Summary (NIST AML Taxonomy)
* **Protected System:** Production email filtering gateway deploying classical ML classifiers.
* **Attacker Profile:** External spammer attempting unauthorized inbox delivery.
* **Attacker Access:** Training-time label control bounded by poisoning rate $\alpha \in [0.0, 0.20]$. Features $X$ and test set $D_{\text{test}}$ remain strictly clean and immutable.
* **Attack Strategies:**
  1. **Random Label Flipping (Availability Threat):** Indiscriminate bidirectional noise ($0 \leftrightarrow 1$). Degrades overall accuracy and classifier confidence (Denial of Service).
  2. **Targeted Spam-to-Ham Flipping (Integrity Threat):** Unidirectional evasion attack ($1 \to 0$). Specifically pulls decision boundaries into spam feature clusters, causing catastrophic drops in spam recall while keeping spam precision deceptively high ($>95\%$).

---

## 📈 Key Empirical Findings (Stage 3 Pilot Benchmark)

Evaluated across **Seeds 0, 1, and 2** (81 total runs, 0 failures):

| Model | Clean Baseline (0%) F1 | Random Flip (20%) Recall | Targeted Flip (20%) Recall | Targeted Flip (20%) Precision |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Std)** | $0.9114 \pm 0.005$ | $0.8012 \pm 0.012$ | **$0.1922 \pm 0.015$** (-70.50\%) | **$0.9542 \pm 0.008$** (Stealthy!) |
| **Decision Tree** | $0.9010 \pm 0.003$ | $0.8245 \pm 0.021$ | $0.5110 \pm 0.043$ (-37.51\%) | $0.9412 \pm 0.012$ |
| **Random Forest (200 trees)** | **$0.9360 \pm 0.002$** | **$0.8325 \pm 0.009$** | $0.3054 \pm 0.018$ (-60.83\%) | **$0.9780 \pm 0.004$** |

### Core Takeaways:
1. **Linear Vulnerability:** Logistic Regression experiences total recall collapse under targeted poisoning, allowing $>80\%$ of spam to penetrate the filter.
2. **Ensemble Robustness:** Random Forest maintains superior accuracy and noise resilience up to 10\% poisoning due to bootstrap aggregating and feature subspace projection.
3. **Evasion Stealth:** Under targeted attack, spam precision stays $>95\%$, hiding the attack from administrators monitoring false alarm rates.

---

## 🚀 Quickstart & Reproduction Guide

### 1. Run Automated Unit Tests (17/17 Tests)
```bash
pytest -v
```

### 2. Run the Multi-Seed Pilot Experiment via CLI
```bash
python scripts/run_attack_experiments.py --seeds 0 1 2 --rates 0.0 0.01 0.05 0.10 0.20
```

### 3. Launch Interactive Research Notebooks
```bash
jupyter notebook
```
Navigate to `notebooks/` and run:
* `01_baseline_and_first_attack.ipynb`
* `02_attack_validation.ipynb`
* `03_pilot_experiment.ipynb`

---

## 🛡️ Stage 4 Roadmap (Defense Blueprint)
As detailed in [`stage3_protocol.md`](stage3_protocol.md), Stage 4 expands the pipeline to:
* **Ten Random Seeds (0–9)** (510 total evaluations)
* **Manually Implemented k-NN Label-Auditing Defense** operating under a Filter-then-Train paradigm
* **Performance Recovery Score Evaluation**:
  $$R_{\text{recovery}} = \frac{M_{\text{defended}} - M_{\text{poisoned}}}{M_{\text{clean}} - M_{\text{poisoned}}}$$
