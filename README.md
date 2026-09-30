# AD-IRT: Adaptive Dimensionality Item Response Theory

**Adaptive Dimensionality IRT for Sparse Cross-Benchmark Evaluation**

## Results (Evo-SOTA.io VLA, nested 5-fold CV, mean ± SEM)

| Method | Rank ρ | MAE |
|--------|--------|-----|
| Simple Averaging | 0.799 ± 0.029 | **0.106 ± 0.002** |
| IRT 1PL | 0.839 ± 0.021 | 0.205 ± 0.006 |
| MIRT K=2 | 0.775 ± 0.035 | 0.287 ± 0.006 |
| **AD-IRT** | **0.851 ± 0.023** | 0.244 ± 0.006 |

AD-IRT wins 4/5 folds (Wilcoxon one-sided p = 0.0625). w is tuned on inner 3-fold CV; in this default run the tuning still uses f_u computed from the full mask, so it is not free of test-fold information (see the caveat below).

**Leakage caveat.** The table above is *not* fully leak-free. `src/experiment.py` computes each model's family count f_u (which sets the blend weight α_u) from the **full observation mask** (`model_fc`), including outer-test observations, and uses it both in inner-CV tuning and on the outer test fold. The paper treats reporting breadth as known metadata; under a strict protocol it is a quantity that should come from training data only.

**Leak-free variant** (f_u from training observations only; `python scripts/strict_no_leak.py`, which runs `src/experiment.py --train-only-fc` unchanged):

| AD-IRT | Rank ρ | MAE | Folds beating IRT | Wilcoxon p (one-sided) |
|--------|--------|-----|------|------|
| Default (f_u from full mask) | 0.851 ± 0.023 | 0.244 ± 0.006 | 4/5 | 0.0625 |
| **Leak-free (f_u from train only)** | 0.845 ± 0.021 | 0.244 ± 0.006 | 3/5 | 0.3125 |

Avg / IRT / MIRT rows are unaffected (they do not use f_u). Under the leak-free protocol AD-IRT's mean ρ is still above IRT (0.839) but the fold-level advantage is not significant. The re-run output is identical to the committed `cv_results_train_only_fc.json`.

## Reproduce

```bash
git clone https://github.com/MINT-SJTU/Evo-SOTA.io.git evo_sota
pip install numpy scipy matplotlib scikit-learn
python src/parse_evosota.py --evo_dir evo_sota/public/data --out_dir data
python src/experiment.py
python scripts/strict_no_leak.py   # leak-free variant
```

## Author

Jung Min Kang · Independent Researcher, Seoul · ORCID: 0009-0007-9599-2792
