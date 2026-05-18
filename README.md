# AD-IRT: Adaptive Dimensionality Item Response Theory

**Adaptive Dimensionality IRT for Sparse Cross-Benchmark Evaluation**

## Results (Evo-SOTA.io VLA, nested 5-fold CV, mean ± SEM)

| Method | Rank ρ | MAE |
|--------|--------|-----|
| Simple Averaging | 0.799 ± 0.029 | **0.106 ± 0.002** |
| IRT 1PL | 0.839 ± 0.021 | 0.205 ± 0.006 |
| MIRT K=2 | 0.775 ± 0.035 | 0.287 ± 0.006 |
| **AD-IRT** | **0.851 ± 0.023** | 0.244 ± 0.006 |

AD-IRT wins 4/5 folds. w tuned on inner 3-fold CV (no test leakage).

## Reproduce

```bash
git clone https://github.com/MINT-SJTU/Evo-SOTA.io.git evo_sota
pip install numpy scipy matplotlib scikit-learn
python src/parse_evosota.py --evo_dir evo_sota/public/data --out_dir data
python src/experiment.py
```

## Author

Jung Min Kang · Independent Researcher, Seoul · ORCID: 0009-0007-9599-2792
