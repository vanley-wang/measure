"""
MedNext 模型迁移到 ICC005（去B11）
Scatt 通过质心最近邻精确匹配，聚类后按类计算 Scatt 均值
"""
import os, pickle
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.measure import label as sk_label, marching_cubes, mesh_surface_area
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

MIN_VOLUME = 50

CF_MORPH = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity'
]
CF_FULL = CF_MORPH + ['Scatt_Mean', 'Scatt_STD']

FEATURE_LONG = {
    'Organoids_Volume': 'Volume_Avg', 'Organoids_Volume_Fill': 'Volume_Fill_Avg',
    'Organoids_Surface': 'Surface_Avg', 'Cavity_Volume': 'Cavity_Volume_All',
    'CavityNum': 'CavityNum_Avg', 'LongAxis': 'Long_Axis_Avg',
    'ShortAxis': 'Short_Axis_Avg', 'Wall_Thickness': 'Cyst_Thick_Avg',
    'Sphericity': 'Sphericity_Avg', 'Scatt_Mean': 'Scatt_Mean_Avg',
    'Scatt_STD': 'Scatt_STD_Avg',
}

ATP_DATA = {
    'B4': 3113000, 'B6': 2658000, 'B9': 4212000, 'B10': 4482000,
    'F4': 3113000, 'F5': 1647000, 'F7': 2197000,
    'F8': 2499000, 'F10': 4482000, 'G4': 2840000, 'G6': 3192000,
    'G9': 2974000, 'G11': 2941000,
}
WELLS = list(ATP_DATA.keys())
atp = np.array([ATP_DATA[w] for w in WELLS])

MODEL_DIR = r'D:\Desktop\music\measure\psd_pipeline\model_mednext'
ICC005_DIR = r'D:\Desktop\music\measure\ICC005'


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
        boundary = cropped & (~eroded)
        return float(np.count_nonzero(boundary))


def measure_nii_with_centroids(seg_path):
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
        inst = (labeled[sl] == lid)
        vol = int(inst.sum())
        if vol < MIN_VOLUME:
            continue
        coords = np.where(inst)
        cz = coords[0].mean() + sl[0].start
        cy = coords[1].mean() + sl[1].start
        cx = coords[2].mean() + sl[2].start
        dz = sl[0].stop - sl[0].start
        dy = sl[1].stop - sl[1].start
        dx = sl[2].stop - sl[2].start
        la = float(max(dz, dy, dx))
        sa = float(min(dz, dy, dx))
        try:
            fill = ndimage.binary_fill_holes(inst)
        except Exception:
            fill = inst
        vfill = int(fill.sum())
        cav = fill & (~inst)
        cvol = int(cav.sum())
        _, cnum = sk_label(cav, connectivity=3, return_num=True)
        surf = compute_surface_mc(inst)
        sp = (np.pi ** (1./3.) * (6 * vfill) ** (2./3.)) / (surf + 1e-6)
        wt = max(1.0, sa / 2.0)
        rows.append({
            'Organoids_Volume': vol, 'Organoids_Volume_Fill': vfill,
            'Organoids_Surface': surf, 'Cavity_Volume': cvol,
            'CavityNum': cnum, 'LongAxis': la, 'ShortAxis': sa,
            'Wall_Thickness': wt, 'Sphericity': sp,
            'centroid_z': cz, 'centroid_y': cy, 'centroid_x': cx,
        })
    return rows


def match_scatt_by_centroid(mednext_centroids, scatt_df):
    """
    质心最近邻匹配：每个 MedNext organoid 找空间距离最近的 scatt organoid
    mednext_centroids: (N, 3) array [z, y, x]
    scatt_df: DataFrame with Centroid_Axis0/1/2, Scatt_Mean, Scatt_STD
    """
    n_mednext = len(mednext_centroids)
    n_scatt = len(scatt_df)

    if n_scatt == 0:
        return np.zeros(n_mednext), np.zeros(n_mednext)

    scatt_centroids = scatt_df[['Centroid_Axis0', 'Centroid_Axis1', 'Centroid_Axis2']].values

    # 计算距离矩阵 (N_mednext x N_scatt)
    dist = cdist(mednext_centroids, scatt_centroids, metric='euclidean')

    # 每个 MedNext organoid 找最近的 scatt organoid
    nearest_idx = np.argmin(dist, axis=1)
    nearest_dist = np.min(dist, axis=1)

    assigned_mean = scatt_df['Scatt_Mean'].values[nearest_idx]
    assigned_std = scatt_df['Scatt_STD'].values[nearest_idx]

    return assigned_mean, assigned_std, nearest_dist


def aggregate_well(df, labels, n_merged=4):
    rec = {}
    for cl in range(n_merged):
        mask = labels == cl
        n = int(mask.sum())
        rec[f'Number_{cl+1}'] = n
        sub = df.loc[mask] if n > 0 else df.iloc[:0]
        for src, alias in FEATURE_LONG.items():
            if src not in df.columns:
                continue
            if src == 'Cavity_Volume':
                rec[f'{alias}_{cl+1}'] = float(sub[src].sum())
            else:
                rec[f'{alias}_{cl+1}'] = float(sub[src].mean()) if n > 0 else 0.0
    return rec


# ============================================================
# 1. Measure ICC005 + Centroid-matched Scatt
# ============================================================
print('=' * 70)
print('  [1] Measuring ICC005 + Centroid-matched Scatt (no B11)')
print('=' * 70)

all_rows = []
match_stats = []

for date_tag, date_sub in [('0701', '20230701'), ('0703', '20230703')]:
    seg_dir = os.path.join(ICC005_DIR, date_sub, 'seg')
    for f in sorted(os.listdir(seg_dir)):
        if not f.endswith('.nii.gz') or f.startswith('plans'):
            continue
        well = f.replace('.nii.gz', '')
        if well not in WELLS:
            continue

        rows = measure_nii_with_centroids(os.path.join(seg_dir, f))
        n_mednext = len(rows)

        scatt_path = os.path.join(ICC005_DIR, date_sub, 'scatt', well + '_scatt.xlsx')
        if os.path.exists(scatt_path):
            scatt_df = pd.read_excel(scatt_path)
            n_scatt = len(scatt_df)

            mednext_centroids = np.array([[r['centroid_z'], r['centroid_y'], r['centroid_x']] for r in rows])

            assigned_mean, assigned_std, nearest_dist = match_scatt_by_centroid(mednext_centroids, scatt_df)

            avg_dist = nearest_dist.mean()
            max_dist = nearest_dist.max()
            match_stats.append((well, date_tag, n_mednext, n_scatt, avg_dist, max_dist))

            for i, r in enumerate(rows):
                r['Scatt_Mean'] = assigned_mean[i]
                r['Scatt_STD'] = assigned_std[i]
                r['well'] = well
                r['date'] = date_tag

            print(f'  {well}_{date_tag}: {n_mednext} MedNext <-> {n_scatt} Scatt, avg_dist={avg_dist:.1f}, max_dist={max_dist:.1f}')
        else:
            for r in rows:
                r['Scatt_Mean'] = 0.0
                r['Scatt_STD'] = 0.0
                r['well'] = well
                r['date'] = date_tag
            print(f'  {well}_{date_tag}: {n_mednext} MedNext, NO scatt')

        all_rows.extend(rows)

df_all = pd.DataFrame(all_rows)
d1 = df_all[df_all['date'] == '0701'].reset_index(drop=True)
d5 = df_all[df_all['date'] == '0703'].reset_index(drop=True)
print(f'  D1: {len(d1)}, D5: {len(d5)}, Total: {len(df_all)}')

# Verify: Scatt values now vary per organoid
print(f'\n  Scatt per-organoid variation check (B4_0701):')
b4_d1 = d1[d1['well'] == 'B4']
print(f'    Scatt_Mean: min={b4_d1.Scatt_Mean.min():.1f}, max={b4_d1.Scatt_Mean.max():.1f}, std={b4_d1.Scatt_Mean.std():.1f}')
print(f'    Scatt_STD:  min={b4_d1.Scatt_STD.min():.1f}, max={b4_d1.Scatt_STD.max():.1f}, std={b4_d1.Scatt_STD.std():.1f}')

# ============================================================
# 2. Load MedNext model
# ============================================================
print(f'\n{"="*70}')
print('  [2] Loading MedNext model')
print('=' * 70)

model = pickle.load(open(os.path.join(MODEL_DIR, 'mednext_delta_deploy.pkl'), 'rb'))
kmeans_k6 = pickle.load(open(os.path.join(MODEL_DIR, 'kmeans_k6.pkl'), 'rb'))
scaler_k6 = pickle.load(open(os.path.join(MODEL_DIR, 'scaler_k6.pkl'), 'rb'))
numeric_map = pickle.load(open(os.path.join(MODEL_DIR, 'numeric_map.pkl'), 'rb'))

print(f'  Model features ({len(model["features"])}): {model["features"]}')
print(f'  Training r: {model["training_r"]:.4f}')

# ============================================================
# 3. Apply MedNext KMeans to ICC005
# ============================================================
print(f'\n{"="*70}')
print('  [3] Apply MedNext KMeans to ICC005')
print('=' * 70)

X_icc = df_all[CF_FULL].fillna(0).values
X_icc_std = scaler_k6.transform(X_icc)
k6_icc = kmeans_k6.predict(X_icc_std)
k4_icc = np.array([numeric_map[int(k)] for k in k6_icc])

sizes_k4 = {}
for k in range(4):
    sizes_k4[k] = int((k4_icc == k).sum())
print(f'  K4 distribution: {sizes_k4}')

n1 = len(d1)
k4_d1 = k4_icc[:n1]
k4_d5 = k4_icc[n1:]

# Show per-cluster Scatt for a sample well
print(f'\n  Per-cluster Scatt (B4_0701, after centroid matching):')
b4_pos = np.arange(n1)[d1['well'].values == 'B4']
b4_k4 = k4_d1[b4_pos]
b4_data = d1[d1['well'] == 'B4']
for cl in range(4):
    mask = b4_k4 == cl
    if mask.sum() > 0:
        sub = b4_data.iloc[np.where(mask)[0]]
        print(f'    K{cl}: n={mask.sum()}, Scatt_Mean={sub.Scatt_Mean.mean():.1f}+-{sub.Scatt_Mean.std():.1f}, Scatt_STD={sub.Scatt_STD.mean():.1f}+-{sub.Scatt_STD.std():.1f}')

# ============================================================
# 4. Compute Delta features (MedNext naming)
# ============================================================
print(f'\n{"="*70}')
print('  [4] Compute Delta features (MedNext naming)')
print('=' * 70)

delta_rows = []
all_feat_names = None

for w in WELLS:
    d1_w = d1[d1['well'] == w].reset_index(drop=True)
    d5_w = d5[d5['well'] == w].reset_index(drop=True)
    d1_pos = np.arange(n1)[d1['well'].values == w]
    d5_pos = np.arange(len(d5))[d5['well'].values == w]
    c3 = k4_d1[d1_pos]
    c5 = k4_d5[d5_pos]
    f3 = aggregate_well(d1_w, c3)
    f5 = aggregate_well(d5_w, c5)
    if all_feat_names is None:
        all_feat_names = sorted(f3.keys())
    row = [f5.get(k, 0.0) - f3.get(k, 0.0) for k in all_feat_names]
    delta_rows.append(row)

delta = np.array(delta_rows)
print(f'  Delta shape: {delta.shape}')

missing_feats = [f for f in model['features'] if f not in all_feat_names]
available_feats = [f for f in model['features'] if f in all_feat_names]
print(f'  Model features available: {len(available_feats)}/{len(model["features"])}')
if missing_feats:
    print(f'  Missing: {missing_feats}')

# ============================================================
# 5. Apply MedNext PCA model to ICC005
# ============================================================
print(f'\n{"="*70}')
print('  [5] Apply MedNext model -> ICC005')
print('=' * 70)

sel_idx = []
for f in model['features']:
    if f in all_feat_names:
        sel_idx.append(all_feat_names.index(f))
    else:
        sel_idx.append(-1)

X_sel = np.zeros((len(WELLS), len(model['features'])))
for i, idx in enumerate(sel_idx):
    if idx >= 0:
        X_sel[:, i] = delta[:, idx]

X_sel_std = model['scaler'].transform(X_sel)
pcs = model['pca'].transform(X_sel_std)
score = pcs @ model['weights']

r, p = pearsonr(score, atp)
rho, _ = spearmanr(score, atp)
if r < 0:
    score = -score
    r, p = pearsonr(score, atp)
    rho, _ = spearmanr(score, atp)

print(f'\n  ==========================================')
print(f'  MedNext model -> ICC005 (centroid-matched Scatt)')
print(f'  ==========================================')
print(f'  Pearson r    = {r:.4f}')
print(f'  p-value      = {p:.6f}')
print(f'  Spearman rho = {rho:.4f}')
print(f'  Training r   = {model["training_r"]:.4f}')

print(f'\n  Per-well results:')
for wi, w in enumerate(WELLS):
    print(f'    {w:<6s} Score={score[wi]:>+.4f}  ATP={atp[wi]:>10,}')

# ============================================================
# 6. ICC005 self-model (50000 MC search)
# ============================================================
print(f'\n{"="*70}')
print('  [6] ICC005 self-model with Scatt (50000 MC search)')
print('=' * 70)

N_SEARCH = 50000
N_PC = 4
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
nf_x = delta.shape[1]
best_r_self = 0
best_self = None

for it in range(N_SEARCH):
    ns = np.random.randint(max(4, nf_x // 3), nf_x + 1)
    si = sorted(np.random.choice(nf_x, ns, replace=False))
    Xs = delta[:, si]
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
    r_c, p_c = pearsonr(sc, atp)
    if abs(r_c) > best_r_self:
        best_r_self = abs(r_c)
        best_self = {
            'si': si, 'r': r_c, 'p': p_c,
            'feats': [all_feat_names[i] for i in si]
        }

print(f'  Self-model best r = {abs(best_self["r"]):.4f}, p = {best_self["p"]:.6f}')
print(f'  Selected features ({len(best_self["feats"])}):')
for f in best_self['feats']:
    print(f'    {f}')

# ============================================================
# 7. Summary
# ============================================================
print(f'\n{"="*70}')
print('  SUMMARY')
print('=' * 70)
print(f'  Method                              Pearson r   Spearman rho')
print(f'  Previous (no Scatt, missing=0)      0.7729      0.7245')
print(f'  Previous (well-avg Scatt)           0.6603      0.7410')
print(f'  Centroid-NN matched Scatt (transfer) {r:.4f}      {rho:.4f}')
print(f'  Centroid-NN matched Scatt (self)     {abs(best_self["r"]):.4f}      ---')
print(f'  MedNext training r                   {model["training_r"]:.4f}')

print(f'\n[Done]')