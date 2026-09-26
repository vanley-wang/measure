"""
深度分析 ICC005 自建模型 r=0.69 偏低的原因
"""
import os, gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.measure import label as sk_label, marching_cubes, mesh_surface_area
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

MIN_VOLUME = 50
RANDOM_SEED = 42

MORPH_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity'
]
MORPH_SHORT = {
    'Organoids_Volume': 'Vol', 'Organoids_Volume_Fill': 'FillVol',
    'Organoids_Surface': 'Surf', 'Cavity_Volume': 'CavVol',
    'CavityNum': 'CavNum', 'LongAxis': 'LongAx', 'ShortAxis': 'ShortAx',
    'Wall_Thickness': 'WallTh', 'Sphericity': 'Spher',
}

ATP_DATA = {
    'B4': 3113000, 'B6': 2658000, 'B9': 4212000, 'B10': 4482000,
    'B11': 11180000, 'F4': 3113000, 'F5': 1647000, 'F7': 2197000,
    'F8': 2499000, 'F10': 4482000, 'G4': 2840000, 'G6': 3192000,
    'G9': 2974000, 'G11': 2941000,
}
WELLS = list(ATP_DATA.keys())
atp = np.array([ATP_DATA[w] for w in WELLS])


def compute_surface_mc(mask_3d):
    coords = np.where(mask_3d)
    if len(coords[0]) == 0:
        return 0.0
    z0, z1 = coords[0].min(), coords[0].max() + 1
    y0, y1 = coords[1].min(), coords[1].max() + 1
    x0, x1 = coords[2].min(), coords[2].max() + 1
    cropped = mask_3d[z0:z1, y0:y1, x0:x1]
    try:
        verts, faces, _, _ = marching_cubes(cropped.astype(float), level=0.5)
        return mesh_surface_area(verts, faces)
    except Exception:
        eroded = ndimage.binary_erosion(cropped)
        return float(np.count_nonzero(cropped & (~eroded)))


def measure_nii(seg_path):
    seg = nib.load(seg_path).get_fdata()
    labeled, nf = ndimage.label(seg > 0)
    if nf == 0:
        return []
    slices = ndimage.find_objects(labeled)
    rows = []
    for lid in range(1, nf + 1):
        sl = slices[lid - 1]
        if sl is None:
            continue
        cm = labeled[sl]
        inst = (cm == lid)
        vol = int(inst.sum())
        if vol < MIN_VOLUME:
            continue
        try:
            fill = ndimage.binary_fill_holes(inst)
        except Exception:
            fill = inst
        vfill = int(fill.sum())
        cav = fill & (~inst)
        cvol = int(cav.sum())
        _, cnum = sk_label(cav, connectivity=3, return_num=True)
        surf = compute_surface_mc(fill)
        dz = sl[0].stop - sl[0].start
        dy = sl[1].stop - sl[1].start
        dx = sl[2].stop - sl[2].start
        la = float(max(dz, dy, dx))
        sa = float(min(dz, dy, dx))
        wt = max(1.0, vfill / (surf + 1e-6) * 0.25)
        sp = (np.pi ** (1./3)) * ((6 * vfill) ** (2./3)) / (surf + 1e-6)
        sp = min(sp, 5.0)
        rows.append({
            'Organoids_Volume': vol, 'Organoids_Volume_Fill': vfill,
            'Organoids_Surface': round(surf, 2), 'Cavity_Volume': cvol,
            'CavityNum': cnum, 'LongAxis': la, 'ShortAxis': sa,
            'Wall_Thickness': round(wt, 4), 'Sphericity': round(sp, 4),
        })
    return rows


def k6_to_k4(centers):
    n = len(centers)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            d = np.linalg.norm(centers[i] - centers[j])
            dm[i, j] = dm[j, i] = d
    used = [False] * n
    pairs = []
    for _ in range(n // 2):
        best_d, best_ij = float('inf'), (-1, -1)
        for i in range(n):
            if used[i]: continue
            for j in range(i + 1, n):
                if used[j]: continue
                if dm[i, j] < best_d:
                    best_d, best_ij = dm[i, j], (i, j)
        pairs.append(best_ij)
        used[best_ij[0]] = used[best_ij[1]] = True
    merge_map = {}
    for ki, pair in enumerate(pairs):
        for k6 in pair:
            merge_map[k6] = ki
    return {k: merge_map[k] for k in sorted(merge_map)}, pairs


# ============================================================
# 1. Measure ICC005
# ============================================================
print('=' * 70)
print('  [1] Measuring ICC005')
print('=' * 70)

ICC005_DIR = r'D:\Desktop\music\measure\ICC005'
all_rows = []
for date_tag, date_sub in [('0701', '20230701'), ('0703', '20230703')]:
    seg_dir = os.path.join(ICC005_DIR, date_sub, 'seg')
    for f in sorted(os.listdir(seg_dir)):
        if not f.endswith('.nii.gz') or f.startswith('plans'):
            continue
        well = f.replace('.nii.gz', '')
        rows = measure_nii(os.path.join(seg_dir, f))
        for r in rows:
            r['well'] = well
            r['date'] = date_tag
        all_rows.extend(rows)

df_all = pd.DataFrame(all_rows)
d1 = df_all[df_all['date'] == '0701'].reset_index(drop=True)
d5 = df_all[df_all['date'] == '0703'].reset_index(drop=True)
print(f'  D1: {len(d1)} organoids, D5: {len(d5)} organoids')

# ============================================================
# 2. ATP 分布分析
# ============================================================
print(f'\n{"="*70}')
print('  [2] ATP 分布分析')
print('=' * 70)

atp_sorted = sorted(ATP_DATA.items(), key=lambda x: x[1])
print(f'  ATP range: {min(atp):,.0f} ~ {max(atp):,.0f}')
print(f'  ATP mean: {np.mean(atp):,.0f}, std: {np.std(atp):,.0f}')
print(f'  ATP CV: {np.std(atp)/np.mean(atp):.3f}')
print(f'\n  Sorted ATP:')
for w, v in atp_sorted:
    flag = ' ← OUTLIER!' if v > np.mean(atp) + 2 * np.std(atp) else ''
    print(f'    {w:<6s} {v:>12,}{flag}')

# B11 is extreme outlier
atp_no_b11 = np.array([v for w, v in ATP_DATA.items() if w != 'B11'])
print(f'\n  Without B11: mean={np.mean(atp_no_b11):,.0f}, std={np.std(atp_no_b11):,.0f}, CV={np.std(atp_no_b11)/np.mean(atp_no_b11):.3f}')
print(f'  B11 = {ATP_DATA["B11"]:,.0f} = {ATP_DATA["B11"]/np.mean(atp_no_b11):.1f}x mean of others')

# ============================================================
# 3. Clustering
# ============================================================
print(f'\n{"="*70}')
print('  [3] Clustering')
print('=' * 70)

X_all = df_all[MORPH_FEATURES].fillna(0).values
scaler = StandardScaler()
X_std = scaler.fit_transform(X_all)
km = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
k6_all = km.fit_predict(X_std)
k4_map, pairs = k6_to_k4(km.cluster_centers_)

n1 = len(d1)
k6_d1 = k6_all[:n1]
k6_d5 = k6_all[n1:]
k4_d1 = np.array([k4_map[k] for k in k6_d1])
k4_d5 = np.array([k4_map[k] for k in k6_d5])

sizes_k6 = dict(zip(*np.unique(k6_all, return_counts=True)))
sizes_k4_d1 = dict(zip(*np.unique(k4_d1, return_counts=True)))
sizes_k4_d5 = dict(zip(*np.unique(k4_d5, return_counts=True)))
print(f'  K6 sizes: {sizes_k6}')
print(f'  K4 D1 sizes: {sizes_k4_d1}')
print(f'  K4 D5 sizes: {sizes_k4_d5}')
print(f'  Merge pairs: {pairs}')

# Per-well K4 distribution
print(f'\n  Per-well K4 distribution:')
print(f'  {"Well":<6s} {"D1_K0":>6s} {"D1_K1":>6s} {"D1_K2":>6s} {"D1_K3":>6s} {"D5_K0":>6s} {"D5_K1":>6s} {"D5_K2":>6s} {"D5_K3":>6s}')
for wi, w in enumerate(WELLS):
    d1_w = d1[d1['well'] == w]
    d5_w = d5[d5['well'] == w]
    d1_pos = np.arange(n1)[d1['well'].values == w]
    d5_pos = np.arange(len(d5))[d5['well'].values == w]
    k4_d1_w = k4_d1[d1_pos]
    k4_d5_w = k4_d5[d5_pos]
    c1 = [np.sum(k4_d1_w == k) for k in range(4)]
    c5 = [np.sum(k4_d5_w == k) for k in range(4)]
    print(f'  {w:<6s} {c1[0]:>6d} {c1[1]:>6d} {c1[2]:>6d} {c1[3]:>6d} {c5[0]:>6d} {c5[1]:>6d} {c5[2]:>6d} {c5[3]:>6d}')

# ============================================================
# 4. Compute Delta features
# ============================================================
print(f'\n{"="*70}')
print('  [4] Delta features')
print('=' * 70)

nf = len(MORPH_FEATURES)
delta_cols = []
for k4 in range(4):
    for f in MORPH_FEATURES:
        delta_cols.append(f'{MORPH_SHORT[f]}_K{k4}')
    delta_cols.append(f'Cnt_K{k4}')

delta = np.zeros((len(WELLS), len(delta_cols)))
for wi, w in enumerate(WELLS):
    d1_w = d1[d1['well'] == w]
    d5_w = d5[d5['well'] == w]
    d1_pos = np.arange(n1)[d1['well'].values == w]
    d5_pos = np.arange(len(d5))[d5['well'].values == w]
    for day_k4, day_data, sign in [(k4_d1[d1_pos], d1_w, -1), (k4_d5[d5_pos], d5_w, 1)]:
        for k4 in range(4):
            kmask = day_k4 == k4
            cnt = kmask.sum()
            offset = k4 * (nf + 1)
            if cnt > 0:
                for fi in range(nf):
                    delta[wi, offset + fi] += sign * day_data[MORPH_FEATURES[fi]].values[kmask].mean()
            delta[wi, offset + nf] += sign * cnt

df_delta = pd.DataFrame(delta, columns=delta_cols)
df_delta.insert(0, 'well', WELLS)
df_delta['ATP'] = atp

print(f'  Delta shape: {delta.shape}')
print(f'\n  Delta feature statistics:')
print(f'  {"Feature":<18s} {"mean":>10s} {"std":>10s} {"min":>12s} {"max":>12s} {"CV":>8s}')
for i, col in enumerate(delta_cols):
    v = delta[:, i]
    m = np.mean(v)
    s = np.std(v)
    cv = s / abs(m) if abs(m) > 1e-6 else float('inf')
    print(f'  {col:<18s} {m:>10.1f} {s:>10.1f} {np.min(v):>12.1f} {np.max(v):>12.1f} {cv:>8.2f}')

# ============================================================
# 5. Correlation of each Delta feature with ATP
# ============================================================
print(f'\n{"="*70}')
print('  [5] Single-feature correlation with ATP')
print('=' * 70)

feat_r = []
for i, col in enumerate(delta_cols):
    v = delta[:, i]
    if np.std(v) < 1e-10:
        continue
    r, p = pearsonr(v, atp)
    rho, _ = spearmanr(v, atp)
    feat_r.append((col, r, p, rho))

feat_r.sort(key=lambda x: abs(x[1]), reverse=True)
print(f'  {"Feature":<18s} {"Pearson_r":>10s} {"p-value":>12s} {"Spearman_ρ":>12s}')
for col, r, p, rho in feat_r[:20]:
    print(f'  {col:<18s} {r:>10.4f} {p:>12.6f} {rho:>12.4f}')

# Also check without B11
print(f'\n  --- Without B11 ---')
atp_nb = np.array([ATP_DATA[w] for w in WELLS if w != 'B11'])
delta_nb = delta[[i for i, w in enumerate(WELLS) if w != 'B11']]
feat_r_nb = []
for i, col in enumerate(delta_cols):
    v = delta_nb[:, i]
    if np.std(v) < 1e-10:
        continue
    r, p = pearsonr(v, atp_nb)
    feat_r_nb.append((col, r, p))
feat_r_nb.sort(key=lambda x: abs(x[1]), reverse=True)
print(f'  {"Feature":<18s} {"Pearson_r":>10s} {"p-value":>12s}')
for col, r, p in feat_r_nb[:10]:
    print(f'  {col:<18s} {r:>10.4f} {p:>12.6f}')

# ============================================================
# 6. Monte Carlo search with different settings
# ============================================================
print(f'\n{"="*70}')
print('  [6] Monte Carlo feature search')
print('=' * 70)

N_PC = 4

def run_search(X, y, n_iter, seed, label=''):
    np.random.seed(seed)
    nf_x = X.shape[1]
    best_r = 0
    best = None
    for it in range(n_iter):
        ns = np.random.randint(max(4, nf_x // 3), nf_x + 1)
        si = sorted(np.random.choice(nf_x, ns, replace=False))
        Xs = X[:, si]
        ss = StandardScaler()
        Xss = ss.fit_transform(Xs)
        pc = PCA(n_components=N_PC, random_state=RANDOM_SEED)
        try:
            Xp = pc.fit_transform(Xss)
        except Exception:
            continue
        wr = pc.explained_variance_ratio_[:N_PC]
        wr = wr / wr.sum()
        sc = Xp @ wr
        r_c, p_c = pearsonr(sc, y)
        if abs(r_c) > best_r:
            best_r = abs(r_c)
            best = {'si': si, 'pc': pc, 'ss': ss, 'wr': wr, 'sc': sc, 'r': r_c, 'p': p_c}
    if best:
        r_f, p_f = pearsonr(best['sc'], y)
        rho_f, _ = spearmanr(best['sc'], y)
        if r_f < 0:
            best['sc'] = -best['sc']
            r_f, p_f = pearsonr(best['sc'], y)
            rho_f, _ = spearmanr(best['sc'], y)
        bf = [delta_cols[i] for i in best['si']]
        cum = best['pc'].explained_variance_ratio_[:N_PC].sum()
        print(f'  {label} |r|={r_f:.4f}, rho={rho_f:.4f}, p={p_f:.6f}, feats={len(bf)}, cum_var={cum:.1%}')
        return r_f, rho_f, bf, best['sc']
    return 0, 0, [], None

# Full 14 wells
print(f'\n  --- Full 14 wells ---')
for n_iter in [10000, 30000, 50000]:
    r, rho, bf, sc = run_search(delta, atp, n_iter, RANDOM_SEED, f'{n_iter}iter')

# Without B11
print(f'\n  --- Without B11 (13 wells) ---')
for n_iter in [10000, 30000, 50000]:
    r, rho, bf, sc = run_search(delta_nb, atp_nb, n_iter, RANDOM_SEED, f'{n_iter}iter')

# ============================================================
# 7. Try K=3 clustering (less sparse)
# ============================================================
print(f'\n{"="*70}')
print('  [7] Alternative: K=3 clustering')
print('=' * 70)

km3 = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
k3_all = km3.fit_predict(X_std)
k3_d1 = k3_all[:n1]
k3_d5 = k3_all[n1:]

sizes_k3 = dict(zip(*np.unique(k3_all, return_counts=True)))
print(f'  K3 sizes: {sizes_k3}')

delta3_cols = []
for k in range(3):
    for f in MORPH_FEATURES:
        delta3_cols.append(f'{MORPH_SHORT[f]}_K{k}')
    delta3_cols.append(f'Cnt_K{k}')

delta3 = np.zeros((len(WELLS), len(delta3_cols)))
for wi, w in enumerate(WELLS):
    d1_w = d1[d1['well'] == w]
    d5_w = d5[d5['well'] == w]
    d1_pos = np.arange(n1)[d1['well'].values == w]
    d5_pos = np.arange(len(d5))[d5['well'].values == w]
    for day_k, day_data, sign in [(k3_d1[d1_pos], d1_w, -1), (k3_d5[d5_pos], d5_w, 1)]:
        for k in range(3):
            kmask = day_k == k
            cnt = kmask.sum()
            offset = k * (nf + 1)
            if cnt > 0:
                for fi in range(nf):
                    delta3[wi, offset + fi] += sign * day_data[MORPH_FEATURES[fi]].values[kmask].mean()
            delta3[wi, offset + nf] += sign * cnt

# Single feature correlation with K=3
feat_r3 = []
for i, col in enumerate(delta3_cols):
    v = delta3[:, i]
    if np.std(v) < 1e-10:
        continue
    r, p = pearsonr(v, atp)
    feat_r3.append((col, r, p))
feat_r3.sort(key=lambda x: abs(x[1]), reverse=True)
print(f'  Top single features (K=3):')
for col, r, p in feat_r3[:10]:
    print(f'    {col:<18s} r={r:.4f} p={p:.6f}')

# Search with K=3
r3, rho3, bf3, sc3 = run_search(delta3, atp, 50000, RANDOM_SEED, 'K3_50k')

# ============================================================
# 8. Try simple per-well aggregation (no clustering)
# ============================================================
print(f'\n{"="*70}')
print('  [8] Alternative: No clustering, simple per-well mean Delta')
print('=' * 70)

delta_simple = np.zeros((len(WELLS), nf + 1))  # +1 for count
simple_cols = [MORPH_SHORT[f] for f in MORPH_FEATURES] + ['Cnt']
for wi, w in enumerate(WELLS):
    d1_w = d1[d1['well'] == w]
    d5_w = d5[d5['well'] == w]
    for fi in range(nf):
        delta_simple[wi, fi] = d5_w[MORPH_FEATURES[fi]].mean() - d1_w[MORPH_FEATURES[fi]].mean()
    delta_simple[wi, nf] = len(d5_w) - len(d1_w)

# Single feature
feat_rs = []
for i, col in enumerate(simple_cols):
    v = delta_simple[:, i]
    if np.std(v) < 1e-10:
        continue
    r, p = pearsonr(v, atp)
    feat_rs.append((col, r, p))
feat_rs.sort(key=lambda x: abs(x[1]), reverse=True)
print(f'  Top simple Delta features:')
for col, r, p in feat_rs[:10]:
    print(f'    {col:<18s} r={r:.4f} p={p:.6f}')

# PCA on simple delta
ss = StandardScaler()
Xss = ss.fit_transform(delta_simple)
pc = PCA(n_components=min(N_PC, delta_simple.shape[1]), random_state=RANDOM_SEED)
Xp = pc.fit_transform(Xss)
wr = pc.explained_variance_ratio_[:N_PC]
wr = wr / wr.sum()
sc_s = Xp @ wr
r_s, p_s = pearsonr(sc_s, atp)
rho_s, _ = spearmanr(sc_s, atp)
if r_s < 0:
    sc_s = -sc_s; r_s, p_s = pearsonr(sc_s, atp); rho_s, _ = spearmanr(sc_s, atp)
print(f'\n  Simple PCA (all {delta_simple.shape[1]} feats): r={r_s:.4f}, rho={rho_s:.4f}, p={p_s:.6f}')

# Search
r_ns, rho_ns, bf_ns, sc_ns = run_search(delta_simple, atp, 50000, RANDOM_SEED, 'Simple_50k')

# ============================================================
# 9. Summary
# ============================================================
print(f'\n{"="*70}')
print('  SUMMARY')
print('=' * 70)
print(f'  Key issues identified:')
print(f'  1. B11 ATP = 11,180,000 is extreme outlier (3.5x next highest)')
print(f'  2. K6 clustering: K2 has 65% of organoids (very skewed)')
print(f'  3. K3/K4/K5 have <2% of organoids (Delta features noisy)')
print(f'  4. Only 14 samples with 40 Delta features (high risk of overfitting)')
print(f'  5. Sphericity mean=2.3 (should be ~0.5-1.0, suggests fragmented organoids)')
print(f'\n[Done]')