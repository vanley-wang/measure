"""
所有模型迁移到 ICC005（去B11，质心匹配 Scatt）的对比
模型: model_mednext, model_nnunet, model_nnunet_new, model (paper)
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

PSD_DIR = r'D:\Desktop\music\measure\psd_pipeline'
ICC005_DIR = r'D:\Desktop\music\measure\ICC005'

MODELS = {
    'MedNext_FXN': {
        'dir': os.path.join(PSD_DIR, 'model_mednext'),
        'file': 'mednext_delta_deploy.pkl',
    },
    'nnUNet_FXN': {
        'dir': os.path.join(PSD_DIR, 'model_nnunet'),
        'file': 'nnunet_delta_deploy.pkl',
    },
    'nnUNet_FXN_new': {
        'dir': os.path.join(PSD_DIR, 'model_nnunet_new'),
        'file': 'nnunet_delta_deploy.pkl',
    },
    'FXN_paper': {
        'dir': os.path.join(PSD_DIR, 'model'),
        'file': 'paper_delta_deploy.pkl',
    },
}


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
    n_mednext = len(mednext_centroids)
    n_scatt = len(scatt_df)
    if n_scatt == 0:
        return np.zeros(n_mednext), np.zeros(n_mednext), np.zeros(n_mednext)
    scatt_centroids = scatt_df[['Centroid_Axis0', 'Centroid_Axis1', 'Centroid_Axis2']].values
    dist = cdist(mednext_centroids, scatt_centroids, metric='euclidean')
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
            for i, r in enumerate(rows):
                r['Scatt_Mean'] = assigned_mean[i]
                r['Scatt_STD'] = assigned_std[i]
                r['well'] = well
                r['date'] = date_tag
            print(f'  {well}_{date_tag}: {n_mednext} MedNext <-> {n_scatt} Scatt, avg_dist={nearest_dist.mean():.1f}')
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
n1 = len(d1)
print(f'  D1: {n1}, D5: {len(d5)}, Total: {len(df_all)}')

# ============================================================
# 2. Test each model
# ============================================================
results = []

for model_name, model_info in MODELS.items():
    print(f'\n{"="*70}')
    print(f'  [{model_name}] Transfer to ICC005')
    print('=' * 70)

    model_path = os.path.join(model_info['dir'], model_info['file'])
    if not os.path.exists(model_path):
        print(f'  SKIP: {model_path} not found')
        continue

    m = pickle.load(open(model_path, 'rb'))
    model_feats = m['features']
    train_r = m.get('training_r', m.get('r', 0))

    print(f'  Features ({len(model_feats)}): {model_feats}')
    print(f'  Training r: {train_r:.4f}')

    # Load KMeans and scaler for this model
    kmeans_path = os.path.join(model_info['dir'], 'kmeans_k6.pkl')
    scaler_path = os.path.join(model_info['dir'], 'scaler_k6.pkl')

    if os.path.exists(kmeans_path) and os.path.exists(scaler_path):
        kmeans_k6 = pickle.load(open(kmeans_path, 'rb'))
        scaler_k6 = pickle.load(open(scaler_path, 'rb'))
    else:
        print(f'  SKIP: kmeans_k6.pkl or scaler_k6.pkl not found')
        continue

    # Get numeric_map
    numeric_map = m.get('numeric_map', None)
    if numeric_map is None:
        nmap_path = os.path.join(model_info['dir'], 'numeric_map.pkl')
        if os.path.exists(nmap_path):
            numeric_map = pickle.load(open(nmap_path, 'rb'))
        else:
            numeric_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2, 5: 3}
            print(f'  WARNING: numeric_map not found, using default')

    # Apply KMeans to ICC005
    X_icc = df_all[CF_FULL].fillna(0).values
    X_icc_std = scaler_k6.transform(X_icc)
    k6_icc = kmeans_k6.predict(X_icc_std)
    k4_icc = np.array([numeric_map[int(k)] for k in k6_icc])

    sizes_k4 = {k: int((k4_icc == k).sum()) for k in range(4)}
    print(f'  K4 distribution: {sizes_k4}')

    k4_d1 = k4_icc[:n1]
    k4_d5 = k4_icc[n1:]

    # Compute Delta features
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

    missing = [f for f in model_feats if f not in all_feat_names]
    available = [f for f in model_feats if f in all_feat_names]
    print(f'  Available features: {len(available)}/{len(model_feats)}')
    if missing:
        print(f'  Missing: {missing}')

    # Build feature vector for model
    sel_idx = []
    for f in model_feats:
        if f in all_feat_names:
            sel_idx.append(all_feat_names.index(f))
        else:
            sel_idx.append(-1)

    X_sel = np.zeros((len(WELLS), len(model_feats)))
    for i, idx in enumerate(sel_idx):
        if idx >= 0:
            X_sel[:, i] = delta[:, idx]

    # Apply model
    X_sel_std = m['scaler'].transform(X_sel)
    pcs = m['pca'].transform(X_sel_std)
    score = pcs @ m['weights']

    r, p = pearsonr(score, atp)
    rho, _ = spearmanr(score, atp)
    if r < 0:
        score = -score
        r, p = pearsonr(score, atp)
        rho, _ = spearmanr(score, atp)

    print(f'\n  >>> Transfer Pearson r    = {r:.4f}')
    print(f'  >>> Transfer p-value      = {p:.6f}')
    print(f'  >>> Transfer Spearman rho = {rho:.4f}')
    print(f'  >>> Training r            = {abs(train_r):.4f}')
    print(f'  >>> Delta r               = {r - abs(train_r):.4f}')

    results.append({
        'model': model_name,
        'train_r': abs(train_r),
        'transfer_r': r,
        'transfer_rho': rho,
        'transfer_p': p,
        'delta_r': r - abs(train_r),
        'k4_sizes': sizes_k4,
        'n_missing': len(missing),
        'missing': missing,
    })

# ============================================================
# 3. Summary
# ============================================================
print(f'\n{"="*70}')
print('  SUMMARY: All Models Transfer to ICC005 (no B11, centroid-matched Scatt)')
print('=' * 70)
print(f'  {"Model":<20s} {"Train_r":>8s} {"Trans_r":>8s} {"Trans_rho":>10s} {"Delta_r":>8s} {"p-value":>10s} {"Missing":>8s}')
print(f'  {"-"*20} {"-"*8} {"-"*8} {"-"*10} {"-"*8} {"-"*10} {"-"*8}')
for r in results:
    print(f'  {r["model"]:<20s} {r["train_r"]:>8.4f} {r["transfer_r"]:>8.4f} {r["transfer_rho"]:>10.4f} {r["delta_r"]:>8.4f} {r["transfer_p"]:>10.6f} {r["n_missing"]:>8d}')
    print(f'    K4: {r["k4_sizes"]}')
    if r["missing"]:
        print(f'    Missing: {r["missing"]}')

print(f'\n[Done]')