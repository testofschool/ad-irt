"""
Parse Evo-SOTA.io JSON files into a frozen score matrix.
Usage: git clone https://github.com/MINT-SJTU/Evo-SOTA.io.git evo_sota
       python src/parse_evosota.py --evo_dir evo_sota/public/data --out_dir data
"""
import json, re, csv, argparse, os
import numpy as np

def norm(name):
    return re.sub(r'\s*\([^)]*\)\s*', '', name.lower().strip())

def flat_entries(data):
    out = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                out.extend(v)
            elif isinstance(v, dict):
                for v2 in v.values():
                    if isinstance(v2, list): out.extend(v2)
    return [e for e in out if isinstance(e, dict) and 'name' in e]

def parse(evo_dir):
    scores = []
    def add(m, it, v, mx=100):
        try:
            val = float(v);
            if mx > 1: val /= mx
            scores.append((m.strip(), it, min(max(val, 0), 1)))
        except: pass

    # LIBERO
    with open(os.path.join(evo_dir, 'libero.json')) as f: d = json.load(f)
    for e in flat_entries(d):
        for k in ['spatial','object','goal','long']:
            if e.get(k) is not None: add(e['name'], f'LIBERO-{k.capitalize()}', e[k])
    # LIBERO-Plus
    with open(os.path.join(evo_dir, 'liberoPlus.json')) as f: d = json.load(f)
    for e in flat_entries(d):
        for k in ['background','camera','language','layout','light','noise','robot']:
            if e.get(k) is not None: add(e['name'], f'LPlus-{k[:5].capitalize()}', e[k])
    # CALVIN
    with open(os.path.join(evo_dir, 'calvin.json')) as f: d = json.load(f)
    smap = {'abcd_d':'CALV-ABCD','abc_d':'CALV-ABC','d_d':'CALV-D'}
    for sn, sd in d.items():
        it = smap.get(sn, f'CALV-{sn}')
        for e in flat_entries(sd):
            if e.get('avg_len') is not None: add(e['name'], it, e['avg_len'], 5)
    # Meta-World
    with open(os.path.join(evo_dir, 'metaworld.json')) as f: d = json.load(f)
    for e in flat_entries(d):
        for k in ['easy','medium','hard','very_hard']:
            if e.get(k) is not None: add(e['name'], f'MW-{k.replace("_","-").title()}', e[k])
    # RoboChallenge
    with open(os.path.join(evo_dir, 'robochallenge.json')) as f: d = json.load(f)
    for e in flat_entries(d):
        if e.get('success_rate') is not None:
            v = float(e['success_rate']); add(e['name'], 'RoboChall', v, 100 if v > 1 else 1)
    # RoboCasa
    with open(os.path.join(evo_dir, 'robocasa_gr1_tabletop.json')) as f: d = json.load(f)
    for e in flat_entries(d):
        if e.get('avg_success_rate') is not None:
            v = float(e['avg_success_rate']); add(e['name'], 'RoboCasa', v, 100 if v > 1 else 1)
    # RoboTwin
    with open(os.path.join(evo_dir, 'robotwin2.json')) as f: d = json.load(f)
    for e in flat_entries(d):
        for k in ['easy','hard']:
            if e.get(k) is not None: add(e['name'], f'RTwin-{k.capitalize()}', e[k])
    # Dexterous
    public_dir = os.path.dirname(os.path.normpath(evo_dir))
    dex_path = os.path.join(public_dir, 'dex', 'data', 'leaderboard.json')
    with open(dex_path) as f: dex = json.load(f)
    skip = {'setting','source','proof','meanSucc'}
    for m in dex.get('methods', []):
        nm = m.get('shortName', m.get('title', ''))
        if not nm: continue
        for bid, bd in m.get('benchmarks', {}).items():
            if not isinstance(bd, dict): continue
            for task, sc in bd.get('values', {}).items():
                if task in skip or sc is None: continue
                try:
                    v = float(sc); add(nm, f'Dex-{bid[:6]}-{task}', v, 100 if v > 1 else 1)
                except: pass
    return scores

def build_matrix(scores, min_items_per_model=3, min_models_per_item=5):
    best = {}
    for m, it, s in scores:
        key = (norm(m), it)
        if key not in best or s > best[key][2]: best[key] = (m, it, s)
    mdls = sorted(set(k[0] for k in best))
    itms = sorted(set(k[1] for k in best))
    mi = {m:i for i,m in enumerate(mdls)}
    ji = {t:j for j,t in enumerate(itms)}
    mat = np.full((len(mdls), len(itms)), np.nan)
    for (m,t),(_,_,s) in best.items(): mat[mi[m], ji[t]] = s
    # Filter
    gj = np.where(np.sum(~np.isnan(mat),axis=0)>=min_models_per_item)[0]
    mat = mat[:,gj]; itms = [itms[j] for j in gj]
    gi = np.where(np.sum(~np.isnan(mat),axis=1)>=min_items_per_model)[0]
    mat = mat[gi,:]; mdls = [mdls[i] for i in gi]
    return mat, mdls, itms

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--evo_dir', default='evo_sota/public/data')
    p.add_argument('--out_dir', default='data')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scores = parse(args.evo_dir)
    mat, mdls, itms = build_matrix(scores)
    # Save CSV matrix
    with open(os.path.join(args.out_dir, 'score_matrix.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model'] + itms)
        for i, m in enumerate(mdls):
            row = [m] + [f'{mat[i,j]:.4f}' if not np.isnan(mat[i,j]) else '' for j in range(len(itms))]
            w.writerow(row)
    # Save metadata
    with open(os.path.join(args.out_dir, 'metadata.json'), 'w') as f:
        json.dump({'models': mdls, 'items': itms, 'n': len(mdls), 'm': len(itms),
                   'obs': int(np.sum(~np.isnan(mat))), 'date': '2026-05-19'}, f, indent=2)
    np.save(os.path.join(args.out_dir, 'score_matrix.npy'), mat)
    print(f"Saved: {len(mdls)} models x {len(itms)} items, {int(np.sum(~np.isnan(mat)))} obs")
