"""
用 MedNext 模型迁移到 ICC005（去掉 B11）
"""
import os, pickle, gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.measure import label as sk_label, marching_cubes, mesh_surface_area
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

MIN_VOLUME = 50

MORPH_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity'
]

FEATURE_LONG = {
    'Organoids_Volume': 'Volume_Avg', 'Organoids_Volume_Fill': 'Volume_Fill_Avg',
    'Organoids_Surface': 'Surface_Avg', 'Cavity_Volume': 'Cavity_Volume_All',
    'CavityNum': 'CavityNum_Avg', 'LongAxis': 'Long_Axis_Avg',
    'ShortAxis': 'Short_Axis_Avg', 'Wall_Thickness': 'Cyst_Thick_Avg',
    'Sphericity': 'Sphericity_Avg',
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
# 1. Measure ICC005
# ============================================================
print('=' * 70)
print('  [1] Measuring ICC005 (marching_cubes, no B11)')
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
        rows = measure_nii(os.path.join(seg_dir, f))
        for r in rows:
            r['well'] = well
            r['date'] = date_tag
        all_rows.extend(rows)
        print(f'  {well}_{date_tag}: {len(rows)} organoids')

df_all = pd.DataFrame(all_rows)
d1 = df_all[df_all['date'] == '0701'].reset_index(drop=True)
d5 = df_all[df_all['date'] == '0703'].reset_index(drop=True)
print(f'  D1: {len(d1)}, D5: {len(d5)}, Total: {len(df_all)}')

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
print(f'  K6->K4 numeric_map: {numeric_map}')

# MedNext uses 11 features (morph + scatt), but ICC005 only has 9 morph
# Need to check which features the model uses
CF_MORPH = MORPH_FEATURES  # 9 features
CF_FULL = MORPH_FEATURES + ['Scatt_Mean', 'Scatt_STD']  # 11 features (MedNext training)

# Check: MedNext KMeans was trained on 11 features
print(f'  KMeans n_features: {kmeans_k6.cluster_centers_.shape[1]}')
print(f'  Scaler n_features: {len(scaler_k6.mean_)}')

# ICC005 has no Scatt features - need to add zeros
# Add Scatt_Mean=0, Scatt_STD=0 to ICC005 measurements
for df in [d1, d5, df_all]:
    df['Scatt_Mean'] = 0.0
    df['Scatt_STD'] = 0.0

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

sizes_k6 = dict(zip(*np.unique(k6_icc, return_counts=True)))
sizes_k4 = dict(zip(*np.unique(k4_icc, return_counts=True)))
print(f'  K6 distribution: {sizes_k6}')
print(f'  K4 distribution: {sizes_k4}')

n1 = len(d1)
k4_d1 = k4_icc[:n1]
k4_d5 = k4_icc[n1:]

# ============================================================
# 4. Compute Delta features (MedNext naming convention)
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
df_delta = pd.DataFrame(delta, columns=all_feat_names)
df_delta.insert(0, 'well', WELLS)

print(f'  Delta shape: {delta.shape}')
print(f'  Feature names: {all_feat_names}')

# Check which model features are available
missing_feats = [f for f in model['features'] if f not in all_feat_names]
available_feats = [f for f in model['features'] if f in all_feat_names]
print(f'\n  Model features available: {len(available_feats)}/{len(model["features"])}')
if missing_feats:
    print(f'  Missing features: {missing_feats}')

# ============================================================
# 5. Apply MedNext PCA model to ICC005
# ============================================================
print(f'\n{"="*70}')
print('  [5] Apply MedNext PCA model to ICC005')
print('=' * 70)

if len(missing_feats) == 0:
    sel_idx = [all_feat_names.index(f) for f in model['features']]
    X_sel = delta[:, sel_idx]
    X_sel_std = model['scaler'].transform(X_sel)
    pcs = model['pca'].transform(X_sel_std)
    score = pcs @ model['weights']

    r, p = pearsonr(score, atp)
    rho, _ = spearmanr(score, atp)
    if r < 0:
        score = -score
        r, p = pearsonr(score, atp)
        rho, _ = spearmanr(score, atp)

    print(f'  Transfer result (MedNext model -> ICC005, no B11):')
    print(f'  Pearson r    = {r:.4f}')
    print(f'  p-value      = {p:.6f}')
    print(f'  Spearman rho = {rho:.4f}')
    print(f'  Training r   = {model["training_r"]:.4f}')
    print(f'  Delta r      = {r - model["training_r"]:.4f}')

    print(f'\n  Per-well results:')
    for wi, w in enumerate(WELLS):
        print(f'    {w:<6s} Score={score[wi]:>+.4f}  ATP={atp[wi]:>10,}')
else:
    print(f'  Cannot apply model - missing features: {missing_feats}')
    print(f'  Filling missing features with 0...')

    # Create full feature vector with zeros for missing
    sel_idx = []
    for f in model['features']:
        if f in all_feat_names:
            sel_idx.append(all_feat_names.index(f))
        else:
            sel_idx.append(-1)  # placeholder

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

    print(f'\n  Transfer result (MedNext model -> ICC005, no B11, missing=0):')
    print(f'  Pearson r    = {r:.4f}')
    print(f'  p-value      = {p:.6f}')
    print(f'  Spearman rho = {rho:.4f}')
    print(f'  Training r   = {model["training_r"]:.4f}')

    print(f'\n  Per-well results:')
    for wi, w in enumerate(WELLS):
        print(f'    {w:<6s} Score={score[wi]:>+.4f}  ATP={atp[wi]:>10,}')

# ============================================================
# 6. Also compute MedNext self-training r for reference
# ============================================================
print(f'\n{"="*70}')
print('  [6] MedNext self-training reference')
print('=' * 70)

# Load MedNext Day3/Day5 data and recompute delta
d3_dir = os.path.join(MED_DIR, 'FXN_0701', 'measure_excel')
d5_dir = os.path.join(MED_DIR, 'FXN_0703', 'measure_excel')

CF_MED = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
          'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
          'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']

ATP_MED = {
    'B10': 601300, 'B11': 11180000, 'B2': 5391000, 'B3': 6538000,
    'B4': 7103000, 'B5': 511900, 'B6': 404500, 'B7': 403900,
    'B8': 312700, 'B9': 140300, 'C10': 211800, 'C11': 13930000,
    'C2': 6336000, 'C3': 8336000, 'C4': 6800000, 'C5': 330900,
    'C6': 238900, 'C7': 682100, 'C8': 211300, 'C9': 465900,
    'D11': 11240000, 'E11': 14700000, 'F10': 21910000, 'F11': 11180000,
    'F2': 18240000, 'F3': 14110000, 'F4': 13740000, 'F5': 17250000,
    'F6': 20320000, 'F7': 20000000, 'F8': 17170000, 'F9': 15830000,
}

d3_files = {f.replace('.xlsx','').split('_')[0]: f for f in os.listdir(d3_dir) if f.endswith('.xlsx')}
d5_files = {f.replace('.xlsx','').split('_')[0]: f for f in os.listdir(d5_dir) if f.endswith('.xlsx')}
common_med = sorted(set(d3_files) & set(d5_files))

med_delta_rows = []
med_well_ids = []
med_atp_vals = []
med_feat_names = None

for wid in common_med:
    df3 = pd.read_excel(os.path.join(d3_dir, d3_files[wid]))
    df5 = pd.read_excel(os.path.join(d5_dir, d5_files[wid]))
    X3 = df3[CF_MED].fillna(0).values
    X5 = df5[CF_MED].fillna(0).values
    c3 = np.array([numeric_map[int(l)] for l in kmeans_k6.predict(scaler_k6.transform(X3))])
    c5 = np.array([numeric_map[int(l)] for l in kmeans_k6.predict(scaler_k6.transform(X5))])
    f3 = aggregate_well(df3, c3)
    f5 = aggregate_well(df5, c5)
    if med_feat_names is None:
        med_feat_names = sorted(f3.keys())
    row = [f5.get(k, 0.0) - f3.get(k, 0.0) for k in med_feat_names]
    med_delta_rows.append(row)
    med_well_ids.append(wid)
    med_atp_vals.append(ATP_MED.get(wid, np.nan))

med_delta = np.array(med_delta_rows)
med_atp = np.array(med_atp_vals)

# Apply model
sel_idx_med = [med_feat_names.index(f) for f in model['features']]
X_med_sel = med_delta[:, sel_idx_med]
X_med_std = model['scaler'].transform(X_med_sel)
pcs_med = model['pca'].transform(X_med_std)
score_med = pcs_med @ model['weights']

r_med, p_med = pearsonr(score_med, med_atp)
rho_med, _ = spearmanr(score_med, med_atp)
if r_med < 0:
    score_med = -score_med
    r_med, p_med = pearsonr(score_med, med_atp)
    rho_med, _ = spearmanr(score_med, med_atp)

print(f'  MedNext self r  = {r_med:.4f}')
print(f'  MedNext self rho= {rho_med:.4f}')
print(f'  N wells         = {len(common_med)}')

print(f'\n[Done]')