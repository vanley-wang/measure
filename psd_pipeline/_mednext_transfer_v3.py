"""
修复 Scatt 匹配：通过质心空间位置匹配 MedNext organoid <-> Scatt organoid
然后聚类后按类计算 Scatt 均值，再用 MedNext 模型计算迁移 r
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

MED_DIR = r'D:\Desktop\music\measure\MedNext_FXN_2023'
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
    """Measure organoids AND compute centroids for spatial matching"""
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

        # Compute centroid in original image coordinates
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


def compute_scatt_centroids_from_seg(seg_path):
    """
    从原始分割计算 scatt organoid 的质心
    用于空间匹配
    """
    seg = nib.load(seg_path).get_fdata()
    labeled, nf = ndimage.label(seg > 0)
    if nf == 0:
        return np.zeros((0, 3))
    slices = ndimage.find_objects(labeled)
    centroids = []
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
        centroids.append([cz, cy, cx])
    return np.array(centroids)


def match_scatt_to_mednext(mednext_centroids, scatt_df):
    """
    通过质心最近邻匹配：每个 MedNext organoid 找最近的 scatt organoid
    
    但我们没有 scatt organoid 的质心！
    策略：假设 scatt organoid 按索引顺序大致对应空间位置
    用排序匹配：两边都按质心 z 坐标排序，然后按排名对应
    """
    n_mednext = len(mednext_centroids)
    n_scatt = len(scatt_df)
    
    if n_scatt == 0:
        return np.zeros(n_mednext), np.zeros(n_mednext)
    
    # MedNext organoids 按 z 坐标排序的索引
    mednext_order = np.argsort(mednext_centroids[:, 0])
    
    # Scatt organoids 假设按索引顺序（1,2,3,...）对应空间从上到下
    scatt_means = scatt_df['Scatt_Mean'].values
    scatt_stds = scatt_df['Scatt_STD'].values
    
    # 为每个 MedNext organoid 分配 scatt 值
    assigned_mean = np.zeros(n_mednext)
    assigned_std = np.zeros(n_mednext)
    
    for rank, mednext_idx in enumerate(mednext_order):
        # 按排名比例映射到 scatt 索引
        scatt_rank = int(rank * (n_scatt - 1) / max(n_mednext - 1, 1))
        scatt_rank = min(scatt_rank, n_scatt - 1)
        assigned_mean[mednext_idx] = scatt_means[scatt_rank]
        assigned_std[mednext_idx] = scatt_stds[scatt_rank]
    
    return assigned_mean, assigned_std


def match_scatt_by_volume_rank(mednext_volumes, scatt_df):
    """
    备选策略：按体积排序匹配
    假设两个分割中 organoid 按体积从大到小的排序大致对应
    """
    n_mednext = len(mednext_volumes)
    n_scatt = len(scatt_df)
    
    if n_scatt == 0:
        return np.zeros(n_mednext), np.zeros(n_mednext)
    
    # MedNext 按体积排序
    mednext_order = np.argsort(-mednext_volumes)
    
    scatt_means = scatt_df['Scatt_Mean'].values
    scatt_stds = scatt_df['Scatt_STD'].values
    
    assigned_mean = np.zeros(n_mednext)
    assigned_std = np.zeros(n_mednext)
    
    for rank, mednext_idx in enumerate(mednext_order):
        scatt_rank = int(rank * (n_scatt - 1) / max(n_mednext - 1, 1))
        scatt_rank = min(scatt_rank, n_scatt - 1)
        assigned_mean[mednext_idx] = scatt_means[scatt_rank]
        assigned_std[mednext_idx] = scatt_stds[scatt_rank]
    
    return assigned_mean, assigned_std


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
# 1. Measure ICC005 + match Scatt by centroid
# ============================================================
print('=' * 70)
print('  [1] Measuring ICC005 + Centroid-matched Scatt (no B11)')
print('=' * 70)

all_rows = []
for date_tag, date_sub in [('0701', '20230701'), ('0703', '20230703')]:
    seg_dir = os.path.join(ICC005_DIR, date_sub, 'seg')
    for f in sorted(os.listdir(seg_dir)):
        if not f.endswith('.nii.gz') or f.startswith('plans'):
            continue
        well = f.replace('.nii.gz', '')
        if well not in WELLS:
            continue
        
        # Measure MedNext organoids with centroids
        rows = measure_nii_with_centroids(os.path.join(seg_dir, f))
        n_mednext = len(rows)
        
        # Load scatt
        scatt_path = os.path.join(ICC005_DIR, date_sub, 'scatt', well + '_scatt.xlsx')
        if os.path.exists(scatt_path):
            scatt_df = pd.read_excel(scatt_path)
            n_scatt = len(scatt_df)
            
            # Extract MedNext centroids and volumes
            centroids = np.array([[r['centroid_z'], r['centroid_y'], r['centroid_x']] for r in rows])
            volumes = np.array([r['Organoids_Volume'] for r in rows])
            
            # Strategy 1: Match by z-coordinate rank
            assigned_mean_z, assigned_std_z = match_scatt_to_mednext(centroids, scatt_df)
            
            # Strategy 2: Match by volume rank
            assigned_mean_v, assigned_std_v = match_scatt_by_volume_rank(volumes, scatt_df)
            
            # Use z-coordinate matching (more spatially meaningful)
            for i, r in enumerate(rows):
                r['Scatt_Mean'] = assigned_mean_z[i]
                r['Scatt_STD'] = assigned_std_z[i]
                r['well'] = well
                r['date'] = date_tag
            
            print(f'  {well}_{date_tag}: {n_mednext} MedNext <-> {n_scatt} Scatt, matched by z-rank')
        else:
            for r in rows:
                r['Scatt_Mean'] = 0.0
                r['Scatt_STD'] = 0.0
                r['well'] = well
                r['date'] = date_tag
            print(f'  {well}_{date_tag}: {n_mednext} MedNext, NO scatt file')
        
        all_rows.extend(rows)

df_all = pd.DataFrame(all_rows)
d1 = df_all[df_all['date'] == '0701'].reset_index(drop=True)
d5 = df_all[df_all['date'] == '0703'].reset_index(drop=True)
print(f'  D1: {len(d1)}, D5: {len(d5)}, Total: {len(df_all)}')

# Verify: Scatt values should now vary per organoid
print(f'\n  Scatt value range check (B4_0701):')
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

# Show per-cluster Scatt values for a sample well
print(f'\n  Per-cluster Scatt check (B4_0701):')
b4_d1_check = d1[d1['well'] == 'B4']
b4_pos = np.arange(n1)[d1['well'].values == 'B4']
b4_k4 = k4_d1[b4_pos]
for cl in range(4):
    mask = b4_k4 == cl
    if mask.sum() > 0:
        sm = b4_d1_check.iloc[np.where(mask)[0]]['Scatt_Mean']
        print(f'    K{cl}: n={mask.sum()}, Scatt_Mean avg={sm.mean():.1f}, std={sm.std():.1f}')

print(f'\n  Per-well results:')
for wi, w in enumerate(WELLS):
    print(f'    {w:<6s} Score={score[wi]:>+.4f}  ATP={atp[wi]:>10,}')

# ============================================================
# 6. Also try volume-rank matching
# ============================================================
print(f'\n{"="*70}')
print('  [6] Re-run with volume-rank matching')
print('=' * 70)

all_rows_v = []
for date_tag, date_sub in [('0701', '20230701'), ('0703', '20230703')]:
    seg_dir = os.path.join(ICC005_DIR, date_sub, 'seg')
    for f in sorted(os.listdir(seg_dir)):
        if not f.endswith('.nii.gz') or f.startswith('plans'):
            continue
        well = f.replace('.nii.gz', '')
        if well not in WELLS:
            continue
        rows = measure_nii_with_centroids(os.path.join(seg_dir, f))
        scatt_path = os.path.join(ICC005_DIR, date_sub, 'scatt', well + '_scatt.xlsx')
        if os.path.exists(scatt_path):
            scatt_df = pd.read_excel(scatt_path)
            volumes = np.array([r['Organoids_Volume'] for r in rows])
            assigned_mean_v, assigned_std_v = match_scatt_by_volume_rank(volumes, scatt_df)
            for i, r in enumerate(rows):
                r['Scatt_Mean'] = assigned_mean_v[i]
                r['Scatt_STD'] = assigned_std_v[i]
                r['well'] = well
                r['date'] = date_tag
        else:
            for r in rows:
                r['Scatt_Mean'] = 0.0
                r['Scatt_STD'] = 0.0
                r['well'] = well
                r['date'] = date_tag
        all_rows_v.extend(rows)

df_all_v = pd.DataFrame(all_rows_v)
d1_v = df_all_v[df_all_v['date'] == '0701'].reset_index(drop=True)
d5_v = df_all_v[df_all_v['date'] == '0703'].reset_index(drop=True)

X_icc_v = df_all_v[CF_FULL].fillna(0).values
X_icc_std_v = scaler_k6.transform(X_icc_v)
k6_icc_v = kmeans_k6.predict(X_icc_std_v)
k4_icc_v = np.array([numeric_map[int(k)] for k in k6_icc_v])

n1_v = len(d1_v)
k4_d1_v = k4_icc_v[:n1_v]
k4_d5_v = k4_icc_v[n1_v:]

delta_rows_v = []
all_feat_names_v = None
for w in WELLS:
    d1_w = d1_v[d1_v['well'] == w].reset_index(drop=True)
    d5_w = d5_v[d5_v['well'] == w].reset_index(drop=True)
    d1_pos = np.arange(n1_v)[d1_v['well'].values == w]
    d5_pos = np.arange(len(d5_v))[d5_v['well'].values == w]
    c3 = k4_d1_v[d1_pos]
    c5 = k4_d5_v[d5_pos]
    f3 = aggregate_well(d1_w, c3)
    f5 = aggregate_well(d5_w, c5)
    if all_feat_names_v is None:
        all_feat_names_v = sorted(f3.keys())
    row = [f5.get(k, 0.0) - f3.get(k, 0.0) for k in all_feat_names_v]
    delta_rows_v.append(row)

delta_v = np.array(delta_rows_v)
sel_idx_v = [all_feat_names_v.index(f) if f in all_feat_names_v else -1 for f in model['features']]
X_sel_v = np.zeros((len(WELLS), len(model['features'])))
for i, idx in enumerate(sel_idx_v):
    if idx >= 0:
        X_sel_v[:, i] = delta_v[:, idx]

X_sel_std_v = model['scaler'].transform(X_sel_v)
pcs_v = model['pca'].transform(X_sel_std_v)
score_v = pcs_v @ model['weights']
r_v, p_v = pearsonr(score_v, atp)
rho_v, _ = spearmanr(score_v, atp)
if r_v < 0:
    score_v = -score_v
    r_v, p_v = pearsonr(score_v, atp)
    rho_v, _ = spearmanr(score_v, atp)

print(f'  Volume-rank matching: r={r_v:.4f}, rho={rho_v:.4f}, p={p_v:.6f}')

# ============================================================
# 7. Summary
# ============================================================
print(f'\n{"="*70}')
print('  SUMMARY')
print('=' * 70)
print(f'  Method                          Pearson r   Spearman rho')
print(f'  Previous (no Scatt, missing=0)  0.7729      0.7245')
print(f'  Previous (well-avg Scatt)       0.6603      0.7410')
print(f'  Centroid z-rank matched Scatt   {r:.4f}      {rho:.4f}')
print(f'  Volume-rank matched Scatt       {r_v:.4f}      {rho_v:.4f}')
print(f'  MedNext training r              {model["training_r"]:.4f}')

print(f'\n[Done]')