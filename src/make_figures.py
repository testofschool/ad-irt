"""
Generate all paper figures from frozen results.
Usage: python src/make_figures.py
Reads:  data/score_matrix.npy, data/metadata.json, data/fold_results.csv
Writes: figures/fig1_coverage_difficulty.pdf, figures/fig2_mechanism.pdf
"""
import numpy as np, json, csv, os
from collections import defaultdict
from scipy.special import expit as sigmoid
from scipy.stats import spearmanr
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.dpi": 300, "savefig.bbox": "tight"})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')
DATA = os.path.join(ROOT, 'data')
FIGS = os.path.join(ROOT, 'figures')
os.makedirs(FIGS, exist_ok=True)

mat = np.load(os.path.join(DATA, 'score_matrix.npy'))
with open(os.path.join(DATA, 'metadata.json')) as f: meta = json.load(f)
items = meta['items']; N, M = mat.shape

# Full-data IRT fit for item parameters
obs = np.array([(i,j,mat[i,j]) for i in range(N) for j in range(M) if not np.isnan(mat[i,j])])
ui, iu = defaultdict(list), defaultdict(list)
for r in obs:
    u,j,s = int(r[0]),int(r[1]),r[2]; ui[u].append((j,s)); iu[j].append((u,s))
th, b = np.zeros(N), np.zeros(M)
for _ in range(60):
    for u in range(N):
        if not ui[u]: continue
        g = sum(s-sigmoid(th[u]-b[j]) for j,s in ui[u])
        th[u] += 0.05*g/len(ui[u])
    for j in range(M):
        if not iu[j]: continue
        g = sum(-(s-sigmoid(th[u]-b[j])) for u,s in iu[j])
        b[j] += 0.05*g/len(iu[j])
    th -= th.mean()

covs = [np.sum(~np.isnan(mat[:,j]))/N for j in range(M)]
rho_cd = spearmanr(covs, b)[0]

# Fig 1
fig, ax = plt.subplots(figsize=(5, 3.8))
cmap = {'LIBERO-':'#534AB7','Plus':'#D85A30','CALV':'#1D9E75','MW-':'#E24B4A','Dex':'#BA7517'}
colors = []
for j in range(M):
    c = '#888780'
    for pf, cl in cmap.items():
        if pf in items[j]: c = cl; break
    colors.append(c)
ax.scatter([c*100 for c in covs], b, c=colors, s=50, zorder=3, edgecolors='white', linewidth=0.5)
ax.set_title(f'Coverage vs. difficulty (Spearman $\\rho$ = {rho_cd:.3f})')
ax.set_xlabel('Reporting coverage (%)'); ax.set_ylabel('IRT difficulty ($b$)')
legend_els = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#534AB7',markersize=7,label='LIBERO'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#D85A30',markersize=7,label='LIBERO-Plus'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#1D9E75',markersize=7,label='CALVIN'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#E24B4A',markersize=7,label='Meta-World'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#BA7517',markersize=7,label='Dexterous'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#888780',markersize=7,label='Other'),
]
ax.legend(handles=legend_els, fontsize=7, loc='upper right'); ax.grid(alpha=0.2)
plt.savefig(os.path.join(FIGS, 'fig1_coverage_difficulty.pdf')); plt.close()

# Fig 2
folds = []
with open(os.path.join(DATA, 'fold_results.csv')) as f:
    for row in csv.DictReader(f): folds.append(row)
mean_w = np.mean([float(f['w_inner']) for f in folds])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))
fc_range = np.linspace(0.5, 5.5, 100)
alpha_vals = sigmoid(mean_w * (fc_range - 1.5))
ax1.plot(fc_range, alpha_vals, color='#534AB7', linewidth=2)
ax1.fill_between(fc_range, alpha_vals, alpha=0.1, color='#534AB7')
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax1.set_xlabel('Benchmark families reported'); ax1.set_ylabel('$\\alpha$ (MIRT weight)')
ax1.set_title(f'Blending function ($\\bar{{w}}$ = {mean_w:.1f})', fontsize=9)
ax1.annotate('More MIRT\n(item calibration)', xy=(1.2, 0.56), fontsize=7, color='#534AB7')
ax1.annotate('More IRT\n(stable $\\theta$)', xy=(3.8, 0.30), fontsize=7, color='#5DCAA5')
ax1.set_ylim(0.15, 0.72); ax1.grid(alpha=0.2)

x = np.arange(5); w_ = 0.22
ax2.bar(x-w_, [float(f['rho_irt']) for f in folds], w_, label='IRT', color='#5DCAA5', edgecolor='white')
ax2.bar(x, [float(f['rho_adirt']) for f in folds], w_, label='AD-IRT', color='#534AB7', edgecolor='white')
ax2.bar(x+w_, [float(f['rho_mirt']) for f in folds], w_, label='MIRT', color='#85B7EB', edgecolor='white')
ax2.set_xlabel('Fold'); ax2.set_ylabel('Rank $\\rho$')
ax2.set_title('Per-fold accuracy (nested CV)', fontsize=9)
ax2.set_xticks(x); ax2.set_xticklabels([f'{i+1}' for i in range(5)])
ax2.legend(fontsize=7, loc='upper center', ncol=3, framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
ax2.set_ylim(0.55, 0.98); ax2.grid(axis='y', alpha=0.2)
plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(FIGS, 'fig2_mechanism.pdf')); plt.close()

print(f"Figures saved. Fig1 rho={rho_cd:.3f}, Fig2 w={mean_w:.1f}")
