#!/usr/bin/env python3
"""
Leak-free (strict) variant of the AD-IRT nested CV.

In the default run, src/experiment.py computes each model's family count f_u
from the FULL observation mask (model_fc), so outer-test observations can
change f_u and therefore the blend weights alpha_u used on that test fold.
The strict variant computes f_u from the training split only (inner-train for
tuning w, outer-train for evaluation). src/experiment.py already implements
this behind --train-only-fc; this script runs it WITHOUT modifying
experiment.py, writes to a separate suffix, and compares against the
committed default and committed strict outputs.

Usage: python scripts/strict_no_leak.py
Writes: cv_results_strict_rerun.json, data/fold_results_strict_rerun.csv,
        data/fold_assignments_strict_rerun.csv
"""
import csv, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIX = "_strict_rerun"

subprocess.run([sys.executable, str(ROOT / "src" / "experiment.py"),
                "--train-only-fc", "--output-suffix", SUFFIX], check=True, cwd=ROOT)

new = json.load(open(ROOT / f"cv_results{SUFFIX}.json"))
default = json.load(open(ROOT / "cv_results.json"))
committed_strict_path = ROOT / "cv_results_train_only_fc.json"
committed_strict = json.load(open(committed_strict_path)) if committed_strict_path.exists() else None

print("\n=== AD-IRT rank rho: default (f_u from full mask) vs strict (f_u from train only) ===")
print(f"default (committed cv_results.json): {default['AD-IRT']['rho_mean']} +/- {default['AD-IRT']['rho_se']}")
print(f"strict  (this re-run):               {new['AD-IRT']['rho_mean']} +/- {new['AD-IRT']['rho_se']}")
if committed_strict:
    print(f"strict  (committed cv_results_train_only_fc.json): "
          f"{committed_strict['AD-IRT']['rho_mean']} +/- {committed_strict['AD-IRT']['rho_se']}")
    print("re-run == committed strict:", new == committed_strict)

rows = list(csv.DictReader(open(ROOT / "data" / f"fold_results{SUFFIX}.csv")))
wins = sum(float(r["rho_adirt"]) > float(r["rho_irt"]) for r in rows)
print(f"strict: AD-IRT beats IRT in {wins}/{len(rows)} folds")
