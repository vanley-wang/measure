"""
深度诊断：ICC005 vs nnUNet_FXN 特征分布对比
目标：找出迁移失败的根本原因
"""
import os, gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.measure import label as sk_label, marching_cubes, mesh_surface_area
from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

MIN_VOLUME = 50

MORPH_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity'
]

ATP_DATA = {
    'B4': 3113000, 'B6': 2658000, 'B9': 4212000, 'B10': 4482000,
    'B11': 11180000, 'F4': 3113000, 'F5': 1647000, 'F7': 2197000,
    'F8': 2499000, 'F10': 4482000, 'G4': 2840000, 'G6': 3192000,
    'G9': 2974000, 'G11': 2941000,
}
WELLS_ICC005 = list(ATP_DATA.keys())


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


# ============================================================
# 1. Measure ICC005 with marching_cubes
# ============================================================
print('=' * 70)
print('  [1] Measuring ICC005 with marching_cubes')
print('=' * 70)

ICC005_DIR = r'D:\Desktop\music\measure\ICC005'
icc005_data = {}
for date_tag, date_sub in [('0701', '20230701'), ('0703', '20230703')]:
    seg_dir = os.path.join(ICC005_DIR, date_sub, 'seg')
    all_rows = []
    for f in sorted(os.listdir(seg_dir)):
        if not f.endswith('.nii.gz') or f.startswith('plans'):
            continue
        well = f.replace('.nii.gz', '')
        rows = measure_nii(os.path.join(seg_dir, f))
        for r in rows:
            r['well'] = well
            r['date'] = date_tag
        all_rows.extend(rows)
        print(f'  {well}_{date_tag}: {len(rows)} organoids')
    icc005_data[date_tag] = pd.DataFrame(all_rows)

df_icc005_all = pd.concat([icc005_data['0701'], icc005_data['0703']], ignore_index=True)
print(f'  Total ICC005 organoids: {len(df_icc005_all)}')

# ============================================================
# 2. Load nnUNet_FXN measurements
# ============================================================
print(f'\n{"="*70}')
print('  [2] Loading nnUNet_FXN measurements')
print('=' * 70)

NNUNET_DIR = r'D:\Desktop\music\measure\Data\nnUNet_FXN_2023'
nnunet_data = {}
for date_tag, batch in [('0701', 'FXN_0701'), ('0703', 'FXN_0703')]:
    me_dir = os.path.join(NNUNET_DIR, batch, 'measure_excel')
    all_rows = []
    for f in sorted(os.listdir(me_dir)):
        if not f.endswith('.xlsx'):
            continue
        well = f.replace('.nii.gz', '').replace(f'_{date_tag}.xlsx', '').replace(f'_{date_tag}', '')
        well = f.split('_')[0]
        df = pd.read_excel(os.path.join(me_dir, f))
        for _, r in df.iterrows():
            row = {feat: r[feat] for feat in MORPH_FEATURES if feat in r.index}
            row['well'] = well
            row['date'] = date_tag
            all_rows.append(row)
    nnunet_data[date_tag] = pd.DataFrame(all_rows)
    print(f'  {date_tag}: {len(nnunet_data[date_tag])} organoids')

df_nnunet_all = pd.concat([nnunet_data['0701'], nnunet_data['0703']], ignore_index=True)
print(f'  Total nnUNet organoids: {len(df_nnunet_all)}')

# ============================================================
# 3. Feature distribution comparison
# ============================================================
print(f'\n{"="*70}')
print('  [3] Feature Distribution: ICC005 vs nnUNet_FXN')
print('=' * 70)

print(f'\n  {"Feature":<25s} {"ICC005_mean":>12s} {"nnUNet_mean":>12s} {"ratio":>8s} {"ICC005_CV":>10s} {"nnUNet_CV":>10s} {"KS_p":>10s} {"Verdict":>12s}')
print('  ' + '-' * 95)

for feat in MORPH_FEATURES:
    v_icc = df_icc005_all[feat].values
    v_nn = df_nnunet_all[feat].values
    m_icc = np.mean(v_icc)
    m_nn = np.mean(v_nn)
    ratio = m_icc / m_nn if m_nn != 0 else 0
    cv_icc = np.std(v_icc) / m_icc if m_icc != 0 else 0
    cv_nn = np.std(v_nn) / m_nn if m_nn != 0 else 0
    ks_stat, ks_p = ks_2samp(v_icc, v_nn)
    if ks_p < 0.001:
        verdict = '⚠ DIFF DIST'
    elif ks_p < 0.05:
        verdict = '~ marginal'
    else:
        verdict = '✓ similar'
    print(f'  {feat:<25s} {m_icc:>12.1f} {m_nn:>12.1f} {ratio:>8.3f} {cv_icc:>10.3f} {cv_nn:>10.3f} {ks_p:>10.2e} {verdict:>12s}')

# ============================================================
# 4. Per-well aggregation comparison (the actual model input)
# ============================================================
print(f'\n{"="*70}')
print('  [4] Per-well mean features (what the model actually sees)')
print('=' * 70)

for date_tag in ['0701', '0703']:
    print(f'\n  --- {date_tag} ---')
    icc_d = icc005_data[date_tag]
    nn_d = nnunet_data[date_tag]

    common_wells = sorted(set(icc_d['well'].unique()) & set(nn_d['well'].unique()))
    print(f'  Common wells: {common_wells}')

    print(f'  {"Well":<6s}', end='')
    for feat in ['Organoids_Volume', 'Organoids_Surface', 'Sphericity', 'LongAxis']:
        print(f' {"ICC_"+feat[:6]:>10s} {"NN_"+feat[:6]:>10s} {"ratio":>7s}', end='')
    print()

    for w in common_wells:
        icc_w = icc_d[icc_d['well'] == w]
        nn_w = nn_d[nn_d['well'] == w]
        print(f'  {w:<6s}', end='')
        for feat in ['Organoids_Volume', 'Organoids_Surface', 'Sphericity', 'LongAxis']:
            m_i = icc_w[feat].mean()
            m_n = nn_w[feat].mean()
            r = m_i / m_n if m_n != 0 else 0
            print(f' {m_i:>10.1f} {m_n:>10.1f} {r:>7.3f}', end='')
        print()

# ============================================================
# 5. Clustering diagnosis: apply nnUNet KMeans to ICC005
# ============================================================
print(f'\n{"="*70}')
print('  [5] Clustering diagnosis')
print('=' * 70)

# Build nnUNet KMeans
X_nn = df_nnunet_all[MORPH_FEATURES].fillna(0).values
scaler_nn = StandardScaler()
X_nn_std = scaler_nn.fit_transform(X_nn)
km_nn = KMeans(n_clusters=6, random_state=42, n_init=10)
k6_nn = km_nn.fit_predict(X_nn_std)

sizes_nn = dict(zip(*np.unique(k6_nn, return_counts=True)))
print(f'  nnUNet K6 sizes: {sizes_nn}')

# Apply to ICC005
X_icc = df_icc005_all[MORPH_FEATURES].fillna(0).values
X_icc_std = scaler_nn.transform(X_icc)
k6_icc_by_nn = km_nn.predict(X_icc_std)

sizes_icc_nn = dict(zip(*np.unique(k6_icc_by_nn, return_counts=True)))
print(f'  ICC005 by nnUNet KMeans: {sizes_icc_nn}')

# ICC005 self-clustering
scaler_icc = StandardScaler()
X_icc_self_std = scaler_icc.fit_transform(X_icc)
km_icc = KMeans(n_clusters=6, random_state=42, n_init=10)
k6_icc_self = km_icc.fit_predict(X_icc_self_std)
sizes_icc_self = dict(zip(*np.unique(k6_icc_self, return_counts=True)))
print(f'  ICC005 self K6 sizes: {sizes_icc_self}')

# ============================================================
# 6. Delta features comparison (the actual model input)
# ============================================================
print(f'\n{"="*70}')
print('  [6] Delta features: ICC005 vs nnUNet (same wells)')
print('=' * 70)

MORPH_SHORT = {
    'Organoids_Volume': 'Vol', 'Organoids_Volume_Fill': 'FillVol',
    'Organoids_Surface': 'Surf', 'Cavity_Volume': 'CavVol',
    'CavityNum': 'CavNum', 'LongAxis': 'LongAx', 'ShortAxis': 'ShortAx',
    'Wall_Thickness': 'WallTh', 'Sphericity': 'Spher',
}

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

def compute_delta_features(df_all, wells, feat_list, scaler, km, k4_map):
    d1 = df_all[df_all['date'] == '0701']
    d5 = df_all[df_all['date'] == '0703']
    X_all = df_all[feat_list].fillna(0).values
    k6_all = km.predict(scaler.transform(X_all))
    n1 = len(d1)
    k6_d1 = k6_all[:n1]
    k6_d5 = k6_all[n1:]
    k4_d1 = np.array([k4_map[k] for k in k6_d1])
    k4_d5 = np.array([k4_map[k] for k in k6_d5])

    nf = len(feat_list)
    delta_cols = []
    for k4 in range(4):
        for f in feat_list:
            delta_cols.append(f'{MORPH_SHORT[f]}_K{k4}')
        delta_cols.append(f'Cnt_K{k4}')

    delta = np.zeros((len(wells), len(delta_cols)))
    for wi, w in enumerate(wells):
        for day_data, k4_labels, sign in [(d1, k4_d1, -1), (d5, k4_d5, 1)]:
            wmask = day_data['well'].values == w
            for k4 in range(4):
                kmask = wmask & (k4_labels == k4)
                cnt = kmask.sum()
                offset = k4 * (nf + 1)
                if cnt > 0:
                    for fi in range(nf):
                        delta[wi, offset + fi] += sign * day_data[feat_list].values[kmask, fi].mean()
                delta[wi, offset + nf] += sign * cnt
    return delta, delta_cols

# nnUNet K4 map
k4_map_nn, pairs_nn = k6_to_k4(km_nn.cluster_centers_)
print(f'  nnUNet K6->K4 merge: {pairs_nn}')

# Compute delta for both datasets using nnUNet's clustering
common_wells = sorted(set(WELLS_ICC005))

# For nnUNet, use all wells
nn_wells_all = sorted(df_nnunet_all['well'].unique())
delta_nn, delta_cols_nn = compute_delta_features(
    df_nnunet_all, nn_wells_all, MORPH_FEATURES, scaler_nn, km_nn, k4_map_nn)

# For ICC005, use nnUNet's clustering
delta_icc_nn, delta_cols_icc = compute_delta_features(
    df_icc005_all, common_wells, MORPH_FEATURES, scaler_nn, km_nn, k4_map_nn)

# For ICC005, self-clustering
k4_map_icc, pairs_icc = k6_to_k4(km_icc.cluster_centers_)
delta_icc_self, _ = compute_delta_features(
    df_icc005_all, common_wells, MORPH_FEATURES, scaler_icc, km_icc, k4_map_icc)

print(f'\n  Delta feature statistics (common wells only):')
print(f'  {"Feature":<20s} {"ICC_nn_mean":>12s} {"ICC_nn_std":>12s} {"ICC_self_mean":>14s} {"ICC_self_std":>14s}')
print('  ' + '-' * 75)
for i, col in enumerate(delta_cols_icc):
    v1 = delta_icc_nn[:, i]
    v2 = delta_icc_self[:, i]
    print(f'  {col:<20s} {np.mean(v1):>12.1f} {np.std(v1):>12.1f} {np.mean(v2):>14.1f} {np.std(v2):>14.1f}')

# ============================================================
# 7. Apply nnUNet PCA model to ICC005
# ============================================================
print(f'\n{"="*70}')
print('  [7] Apply nnUNet PCA model to ICC005')
print('=' * 70)

# Load nnUNet model
model_path = r'D:\Desktop\music\measure\psd_pipeline\model_nnunet\nnunet_delta_deploy.pkl'
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
    print(f'  Model keys: {list(model.keys())}')
    sel_feats = model['features']
    sel_idx = [delta_cols_icc.index(f) for f in sel_feats if f in delta_cols_icc]
    print(f'  Selected features ({len(sel_feats)}): {sel_feats}')
    print(f'  Matched indices in ICC005: {len(sel_idx)}/{len(sel_feats)}')

    if len(sel_idx) == len(sel_feats):
        X_sel = delta_icc_nn[:, sel_idx]
        X_sel_std = model['scaler'].transform(X_sel)
        pcs = model['pca'].transform(X_sel_std)
        score = pcs @ model['weights']

        atp = np.array([ATP_DATA[w] for w in common_wells])
        r, p = pearsonr(score, atp)
        rho, _ = spearmanr(score, atp)
        if r < 0:
            score = -score
            r, p = pearsonr(score, atp)
            rho, _ = spearmanr(score, atp)

        print(f'\n  Transfer result (nnUNet model -> ICC005):')
        print(f'  Pearson r    = {r:.4f}')
        print(f'  p-value      = {p:.6f}')
        print(f'  Spearman rho = {rho:.4f}')

        print(f'\n  Per-well:')
        for wi, w in enumerate(common_wells):
            print(f'    {w:<6s} Score={score[wi]:>+.4f}  ATP={atp[wi]:>10.0f}')
    else:
        missing = [f for f in sel_feats if f not in delta_cols_icc]
        print(f'  Missing features: {missing}')
else:
    print(f'  Model not found at {model_path}')

# Also try with nnUNet_new model
model_path2 = r'D:\Desktop\music\measure\psd_pipeline\model_nnunet_new\nnunet_delta_deploy.pkl'
if os.path.exists(model_path2):
    model2 = pickle.load(open(model_path2, 'rb'))
    sel_feats2 = model2['features']
    sel_idx2 = [delta_cols_icc.index(f) for f in sel_feats2 if f in delta_cols_icc]
    if len(sel_idx2) == len(sel_feats2):
        X_sel2 = delta_icc_nn[:, sel_idx2]
        X_sel2_std = model2['scaler'].transform(X_sel2)
        pcs2 = model2['pca'].transform(X_sel2_std)
        score2 = pcs2 @ model2['weights']
        atp = np.array([ATP_DATA[w] for w in common_wells])
        r2, p2 = pearsonr(score2, atp)
        rho2, _ = spearmanr(score2, atp)
        if r2 < 0:
            score2 = -score2; r2, p2 = pearsonr(score2, atp); rho2, _ = spearmanr(score2, atp)
        print(f'\n  Transfer result (nnUNet_new model -> ICC005):')
        print(f'  Pearson r = {r2:.4f}, p = {p2:.6f}, rho = {rho2:.4f}')

# ============================================================
# 8. ICC005 self-model (independent pipeline)
# ============================================================
print(f'\n{"="*70}')
print('  [8] ICC005 self-model (quick test)')
print('=' * 70)

atp = np.array([ATP_DATA[w] for w in common_wells])

# Use self-clustering delta
from sklearn.decomposition import PCA

nf_delta = delta_icc_self.shape[1]
best_r = 0
best_info = None

np.random.seed(42)
for it in range(10000):
    ns = np.random.randint(max(4, nf_delta//3), nf_delta+1)
    si = sorted(np.random.choice(nf_delta, ns, replace=False))
    Xs = delta_icc_self[:, si]
    ss = StandardScaler()
    Xss = ss.fit_transform(Xs)
    pc = PCA(n_components=4, random_state=42)
    try:
        Xp = pc.fit_transform(Xss)
    except Exception:
        continue
    wr = pc.explained_variance_ratio_[:4]
    wr = wr / wr.sum()
    sc = Xp @ wr
    r_c, p_c = pearsonr(sc, atp)
    if abs(r_c) > best_r:
        best_r = abs(r_c)
        best_info = (si, pc, ss, wr, sc, p_c)

if best_info:
    si, pc, ss, wr, sc, p_c = best_info
    r_f, p_f = pearsonr(sc, atp)
    rho_f, _ = spearmanr(sc, atp)
    if r_f < 0:
        sc = -sc; r_f, p_f = pearsonr(sc, atp); rho_f, _ = spearmanr(sc, atp)
    bf = [delta_cols_icc[i] for i in si]
    print(f'  Self-model (10000 iter search):')
    print(f'  Pearson r    = {r_f:.4f}')
    print(f'  p-value      = {p_f:.6f}')
    print(f'  Spearman rho = {rho_f:.4f}')
    print(f'  N features   = {len(bf)}')
    print(f'  Features     = {bf}')
    print(f'\n  Per-well:')
    for wi, w in enumerate(common_wells):
        print(f'    {w:<6s} Score={sc[wi]:>+.4f}  ATP={atp[wi]:>10.0f}')

print(f'\n[Done]')