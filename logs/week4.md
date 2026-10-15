# Research Log: Week 4 - Pilot Analysis, Protocol Freeze, and Paper Draft

## What I Learned
During Week 4, I completed the formal audit of our Stage 3 pilot study results (verifying all 81 conditions completed with 0 errors or missing values) and synthesized our findings into the frozen protocol (`stage3_protocol.md`) and draft conference paper (`paper/main.tex`). 

The pilot data definitively answered our first research question: targeted spam-to-ham label flipping acts as an acute integrity threat, causing severe evasion where Logistic Regression spam recall drops from 89.72% down to 19.22% (and Random Forest to 30.54%) at 20% poisoning, while spam precision remains deceptively high (>95%). Conversely, random flipping acts as an availability threat, inducing symmetric, moderate degradation. To answer the second research question in Stage 4, I designed the defense blueprint: a manually implemented k-NN neighborhood label auditor placed in a filter-then-train pipeline. Defense performance will be evaluated through detection precision/recall against ground-truth flipped masks and an explicit recovery metric ($R_{\text{recovery}}$).

## What Failed
Designing the defense evaluation revealed an important trade-off between instance pruning and label correction. Pruning removes flagged samples but reduces training set size, whereas relabeling by neighborhood consensus risks miscorrection if local clusters are densely poisoned. Formalizing both strategies in `stage3_protocol.md` ensures a rigorous comparative study in Stage 4.

## Next Steps
With Stage 3 fully validated and frozen, I will transition into Stage 4. I will implement the custom k-NN label-auditing module in `src/defense.py`, expand the execution grid to ten random seeds (0–9), and execute the full 510-condition evaluation to measure defense recovery.
