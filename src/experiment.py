"""
AD-IRT Experiment — Nested 5-fold cross-validation.

Usage:
  # Option A: parse from Evo-SOTA clone
  python src/parse_evosota.py --evo_dir evo_sota/public/data --out_dir data

  # Then run experiment
  python src/experiment.py

Reads:  data/score_matrix.npy, data/metadata.json
Writes: data/fold_assignments.csv, data/fold_results.csv, cv_results.json
"""
import numpy as np, json, csv, os, sys
from collections import defaultdict
from scipy.special import expit as sigmoid
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import KFold

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train-only-fc', action='store_true',
                    help='Compute family count from train split only (strict protocol)')
parser.add_argument('--output-suffix', type=str, default='',
                    help='Suffix for output files (e.g., _train_only_fc)')
parser.add_argument('--verify-cache', action='store_true',
                    help='Verify frozen outputs without recomputing')
parser.add_argument('--fast-check', action='store_true', help='Run 1 fold only')
parser.add_argument('--max-iter-irt', type=int, default=60)
parser.add_argument('--max-iter-mirt', type=int, default=50)
cli_args, _ = parser.parse_known_args()

np.random.seed(42)


# ── Load data ──
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR_EARLY = os.path.join(os.path.dirname(__file__), '..')

if cli_args.verify_cache:
    import csv as csv_mod
    print("Verifying cached outputs...")
    try:
        with open(os.path.join(DATA_DIR, 'fold_results.csv')) as f:
            rows = list(csv_mod.DictReader(f))
        assert len(rows) == 5, f"Expected 5 folds, got {len(rows)}"
        with open(os.path.join(OUT_DIR_EARLY, 'cv_results.json')) as f:
            cv = json.load(f)
        for m in ['Avg', 'IRT', 'MIRT', 'AD-IRT']:
            assert m in cv, f"Missing {m}"
        assert os.path.exists(os.path.join(DATA_DIR, 'fold_assignments.csv'))
        print("All cached outputs verified. OK")
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)
    sys.exit(0)
matrix = np.load(os.path.join(DATA_DIR, 'score_matrix.npy'))
with open(os.path.join(DATA_DIR, 'metadata.json')) as f:
    meta = json.load(f)
models, items = meta['models'], meta['items']
N, M = matrix.shape

# ── Benchmark families ──
families_def = {
    'LIBERO': [j for j,it in enumerate(items) if it.startswith('LIBERO-')],
    'LIBERO-Plus': [j for j,it in enumerate(items) if 'Plus' in it],
    'CALVIN': [j for j,it in enumerate(items) if it.startswith('CALV')],
    'Meta-World': [j for j,it in enumerate(items) if it.startswith('MW-')],
    'Dexterous': [j for j,it in enumerate(items) if it.startswith('Dex-')],
    'RoboTwin': [j for j,it in enumerate(items) if it.startswith('RTwin')],
    'Other': [j for j,it in enumerate(items) if it.startswith('Robo')],
}
model_fc = np.zeros(N)
for i in range(N):
    for fj in families_def.values():
        if any(not np.isnan(matrix[i, j]) for j in fj):
            model_fc[i] += 1

def compute_fc_from_obs(observations, N, families):
    """Compute family count from observations only (no leakage)."""
    obs_items = defaultdict(set)
    for u, j, _ in observations:
        obs_items[int(u)].add(int(j))
    fc = np.zeros(N)
    for u in range(N):
        for fj in families.values():
            if any(j in obs_items[u] for j in fj):
                fc[u] += 1
    return fc

obs = np.array([(i, j, matrix[i, j])
                for i in range(N) for j in range(M) if not np.isnan(matrix[i, j])])
print(f"Data: {N} models x {M} items, {len(obs)} obs, "
      f"{len(obs)/(N*M):.1%} coverage")

# ── Helpers ──
def build_maps(data):
    ui, iu = defaultdict(list), defaultdict(list)
    for r in data:
        u, j, s = int(r[0]), int(r[1]), r[2]
        ui[u].append((j, s)); iu[j].append((u, s))
    return ui, iu

def fit_irt(train, N, M, n_iter=None):
    if n_iter is None: n_iter = cli_args.max_iter_irt
    th, b = np.zeros(N), np.zeros(M)
    ui, iu = build_maps(train)
    for _ in range(n_iter):
        for u in range(N):
            if not ui[u]: continue
            g = sum(s - sigmoid(th[u] - b[j]) for j, s in ui[u])
            th[u] += 0.05 * g / len(ui[u])
        for j in range(M):
            if not iu[j]: continue
            g = sum(-(s - sigmoid(th[u] - b[j])) for u, s in iu[j])
            b[j] += 0.05 * g / len(iu[j])
        th -= th.mean()
    return th, b

def fit_mirt(train, N, M, K=2, n_iter=None):
    if n_iter is None: n_iter = cli_args.max_iter_mirt
    rng = np.random.RandomState(42)
    Th = rng.randn(N, K) * 0.05
    A = rng.randn(M, K) * 0.05 + 0.3
    d = np.zeros(M)
    ui, iu = build_maps(train)
    for _ in range(n_iter):
        for u in range(N):
            if not ui[u]: continue
            g = np.zeros(K)
            for j, s in ui[u]:
                g += A[j] * (s - sigmoid(A[j] @ Th[u] + d[j]))
            Th[u] += 0.02 * (g - 0.01 * Th[u]) / len(ui[u])
        for j in range(M):
            if not iu[j]: continue
            ga, gd = np.zeros(K), 0.0
            for u, s in iu[j]:
                r = s - sigmoid(A[j] @ Th[u] + d[j])
                ga += Th[u] * r; gd += r
            A[j] += 0.02 * (ga - 0.01 * A[j]) / len(iu[j])
            d[j] += 0.02 * gd / len(iu[j])
    return Th, A, d

def pred_irt(th, b, test):
    return np.array([np.clip(sigmoid(th[int(u)] - b[int(j)]), 1e-3, 1-1e-3)
                     for u, j, _ in test])

def pred_mirt(Th, A, d, test):
    return np.array([np.clip(sigmoid(A[int(j)] @ Th[int(u)] + d[int(j)]), 1e-3, 1-1e-3)
                     for u, j, _ in test])

def pred_adirt(th, b, Th, A, d, alphas, test):
    return np.array([
        np.clip((1 - alphas[int(u)]) * sigmoid(th[int(u)] - b[int(j)])
                + alphas[int(u)] * sigmoid(A[int(j)] @ Th[int(u)] + d[int(j)]),
                1e-3, 1-1e-3)
        for u, j, _ in test])

def eval_rho(preds, test):
    mp, ma = defaultdict(list), defaultdict(list)
    for idx in range(len(test)):
        u = int(test[idx, 0])
        mp[u].append(preds[idx]); ma[u].append(test[idx, 2])
    shared = [u for u in mp if len(mp[u]) >= 2]
    if len(shared) < 5:
        return float('nan')
    return spearmanr([np.mean(mp[u]) for u in shared],
                     [np.mean(ma[u]) for u in shared])[0]

def eval_mae(preds, test):
    return float(np.mean(np.abs(preds - test[:, 2])))

# ── Nested 5-fold CV ──
print("\n=== Nested 5-fold CV ===")
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
W_GRID = np.arange(-3, 4, 0.5)

results = {m: {'rho': [], 'mae': []} for m in ['Avg', 'IRT', 'MIRT', 'AD-IRT']}
w_per_fold = []
fold_assignments = []  # (obs_idx, outer_fold)

max_outer_folds = 1 if cli_args.fast_check else 5
for fold, (tr_idx, te_idx) in enumerate(outer_kf.split(obs)):
    if fold >= max_outer_folds:
        break
    train_outer, test_outer = obs[tr_idx], obs[te_idx]
    for idx in tr_idx:
        fold_assignments.append((int(idx), fold + 1, 'train'))
    for idx in te_idx:
        fold_assignments.append((int(idx), fold + 1, 'test'))

    # ── Inner CV: tune w ──
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=fold)
    best_w, best_inner_rho = 0.0, -1.0
    for w_cand in W_GRID:
        inner_rhos = []
        for itr, iva in inner_kf.split(train_outer):
            i_train, i_val = train_outer[itr], train_outer[iva]
            th_i, b_i = fit_irt(i_train, N, M, n_iter=30)
            Th_i, A_i, d_i = fit_mirt(i_train, N, M, K=2, n_iter=25)
            fc_i = compute_fc_from_obs(i_train, N, families_def) if cli_args.train_only_fc else model_fc
            alphas_i = sigmoid(w_cand * (fc_i - 1.5))
            p_i = pred_adirt(th_i, b_i, Th_i, A_i, d_i, alphas_i, i_val)
            r = eval_rho(p_i, i_val)
            if not np.isnan(r):
                inner_rhos.append(r)
        if inner_rhos and np.mean(inner_rhos) > best_inner_rho:
            best_w, best_inner_rho = w_cand, np.mean(inner_rhos)
    w_per_fold.append(float(best_w))

    # ── Outer: fit on full train, evaluate on test with frozen w ──
    fc_outer = compute_fc_from_obs(train_outer, N, families_def) if cli_args.train_only_fc else model_fc
    # Simple averaging
    ms, mc = np.zeros(N), np.zeros(N)
    isums, ic = np.zeros(M), np.zeros(M)
    for u, j, s in train_outer:
        u, j = int(u), int(j)
        ms[u] += s; mc[u] += 1; isums[j] += s; ic[j] += 1
    mm = np.divide(ms, mc, out=np.full(N, 0.5), where=mc > 0)
    im = np.divide(isums, ic, out=np.full(M, 0.5), where=ic > 0)
    gm = train_outer[:, 2].mean()
    p_avg = np.array([np.clip(mm[int(u)] + im[int(j)] - gm, 1e-3, 1-1e-3)
                      for u, j, _ in test_outer])
    results['Avg']['rho'].append(eval_rho(p_avg, test_outer))
    results['Avg']['mae'].append(eval_mae(p_avg, test_outer))

    # IRT
    th, b = fit_irt(train_outer, N, M)
    p_irt = pred_irt(th, b, test_outer)
    results['IRT']['rho'].append(eval_rho(p_irt, test_outer))
    results['IRT']['mae'].append(eval_mae(p_irt, test_outer))

    # MIRT
    Th, A, d = fit_mirt(train_outer, N, M, K=2)
    p_mirt = pred_mirt(Th, A, d, test_outer)
    results['MIRT']['rho'].append(eval_rho(p_mirt, test_outer))
    results['MIRT']['mae'].append(eval_mae(p_mirt, test_outer))

    # AD-IRT (frozen w from inner CV)
    alphas = sigmoid(best_w * (fc_outer - 1.5))
    p_ad = pred_adirt(th, b, Th, A, d, alphas, test_outer)
    results['AD-IRT']['rho'].append(eval_rho(p_ad, test_outer))
    results['AD-IRT']['mae'].append(eval_mae(p_ad, test_outer))

    print(f"  Fold {fold+1}: w*={best_w:+.1f}  "
          f"Avg={results['Avg']['rho'][-1]:.3f}  "
          f"IRT={results['IRT']['rho'][-1]:.3f}  "
          f"MIRT={results['MIRT']['rho'][-1]:.3f}  "
          f"AD-IRT={results['AD-IRT']['rho'][-1]:.3f}")

# ── Write fold_results.csv ──
suffix = cli_args.output_suffix
if cli_args.fast_check and not suffix:
    suffix = '_fast'
with open(os.path.join(DATA_DIR, f'fold_results{suffix}.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['fold', 'w_inner', 'rho_avg', 'rho_irt', 'rho_mirt', 'rho_adirt',
                'mae_avg', 'mae_irt', 'mae_mirt', 'mae_adirt'])
    for i in range(min(5, len(w_per_fold))):
        w.writerow([i + 1, w_per_fold[i],
                    round(results['Avg']['rho'][i], 6),
                    round(results['IRT']['rho'][i], 6),
                    round(results['MIRT']['rho'][i], 6),
                    round(results['AD-IRT']['rho'][i], 6),
                    round(results['Avg']['mae'][i], 6),
                    round(results['IRT']['mae'][i], 6),
                    round(results['MIRT']['mae'][i], 6),
                    round(results['AD-IRT']['mae'][i], 6)])

# ── Write fold_assignments.csv ──
with open(os.path.join(DATA_DIR, f'fold_assignments{suffix}.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['obs_idx', 'model_idx', 'item_idx', 'score', 'outer_fold', 'split'])
    for obs_idx, fold_num, split in fold_assignments:
        u, j, s = int(obs[obs_idx, 0]), int(obs[obs_idx, 1]), obs[obs_idx, 2]
        w.writerow([obs_idx, u, j, round(s, 6), fold_num, split])

# ── Write cv_results.json ──
cv_out = {}
for m in results:
    rvals = [v for v in results[m]['rho'] if not np.isnan(v)]
    # sample SEM (ddof=1)
    se = float(np.std(rvals, ddof=1) / np.sqrt(len(rvals))) if len(rvals) > 1 else 0.0
    mae_vals = results[m]['mae'][:len(rvals)]
    mae_se = float(np.std(mae_vals, ddof=1) / np.sqrt(len(mae_vals))) if len(mae_vals) > 1 else 0.0
    cv_out[m] = {
        'rho_mean': round(float(np.mean(rvals)), 4),
        'rho_se': round(se, 4),
        'mae_mean': round(float(np.mean(results[m]['mae'])), 4),
        'mae_se': round(mae_se, 4),
    }
OUT_DIR = os.path.join(os.path.dirname(__file__), '..')
with open(os.path.join(OUT_DIR, f'cv_results{suffix}.json'), 'w') as f:
    json.dump(cv_out, f, indent=2)

# ── Summary ──
print(f"\n{'='*60}")
print(f"{'Method':<12s} {'Rank rho':>16s} {'MAE':>16s}")
print(f"{'─'*12} {'─'*16} {'─'*16}")
for m in results:
    r = cv_out[m]
    print(f"{m:<12s} {r['rho_mean']:.4f} +/- {r['rho_se']:.4f}  "
          f"{r['mae_mean']:.4f} +/- {r['mae_se']:.4f}")

n_completed = len(w_per_fold)
ad_rhos = results['AD-IRT']['rho'][:n_completed]
irt_rhos = results['IRT']['rho'][:n_completed]
diffs = [a - i for a, i in zip(ad_rhos, irt_rhos)]
print(f"\nAD-IRT wins {sum(1 for d in diffs if d>0)}/5 folds")
print(f"Mean delta(AD-IRT - IRT) = {np.mean(diffs):+.4f}")
try:
    _, pval = wilcoxon(ad_rhos, irt_rhos, alternative='greater')
    print(f"Wilcoxon p = {pval:.4f}")
except Exception:
    pass
print(f"w per fold: {w_per_fold}")

print(f"\nOutputs written to {DATA_DIR}/")
