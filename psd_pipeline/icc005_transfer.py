import os, sys, pickle, gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.measure import label as sk_label
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

ICC005_DIR = r'D:\Desktop\music\measure\ICC005'
MODEL_PATH = r'D:\Desktop\music\measure\psd_pipeline\model_icc_morph\icc_morph_deploy.pkl'
OUT_DIR = r'D:\Desktop\music\measure\psd_pipeline\icc005_results'

ATP_DATA = {
    'B4':  3113000,
    'B6':  2658000,
    'B9':  4212000,
    'B10': 4482000,
    'B11': 2923000,
    'F4':  3113000,
    'F5':  1647000,
    'F7':  2197000,
    'F8':  2499000,
    'F10': 4482000,
    'G4':  2840000,
    'G6':  3192000,
    'G9':  2974000,
    'G11': 2941000,
}
os.makedirs(OUT_DIR, exist_ok=True)

MIN_VOLUME = 50
MORPH_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity'
]
MORPH_SHORT = {
    'Organoids_Volume': 'Vol',
    'Organoids_Volume_Fill': 'FillVol',
    'Organoids_Surface': 'Surf',
    'Cavity_Volume': 'CavVol',
    'CavityNum': 'CavNum',
    'LongAxis': 'LongAx',
    'ShortAxis': 'ShortAx',
    'Wall_Thickness': 'WallTh',
    'Sphericity': 'Spher',
}


def compute_surface_area(mask_3d):
    coords = np.where(mask_3d)
    if len(coords[0]) == 0:
        return 0.0
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1
    cropped = mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]
    eroded = ndimage.binary_erosion(cropped)
    boundary = cropped & (~eroded)
    return float(np.count_nonzero(boundary))


def measure_one_mask(seg_path, well_id):
    seg = nib.load(seg_path).get_fdata()
    labeled, num_features = ndimage.label(seg > 0)
    if num_features == 0:
        return []

    slices = ndimage.find_objects(labeled)
    rows = []

    for label_id in range(1, num_features + 1):
        sl = slices[label_id - 1]
        if sl is None:
            continue

        crop_label = labeled[sl]
        inst_mask = (crop_label == label_id)
        volume_raw = int(np.count_nonzero(inst_mask))
        if volume_raw < MIN_VOLUME:
            continue

        try:
            fill_mask = ndimage.binary_fill_holes(inst_mask)
        except Exception:
            fill_mask = inst_mask
        volume_fill = int(np.count_nonzero(fill_mask))

        cavity_mask = fill_mask & (~inst_mask)
        cavity_volume = int(np.count_nonzero(cavity_mask))
        _, cavity_num = sk_label(cavity_mask, connectivity=3, return_num=True)

        surface = compute_surface_area(fill_mask)

        dz = sl[0].stop - sl[0].start
        dy = sl[1].stop - sl[1].start
        dx = sl[2].stop - sl[2].start
        long_axis = float(max(dz, dy, dx))
        short_axis = float(min(dz, dy, dx))

        wall_thickness = max(1.0, volume_fill / (surface + 1e-6) * 0.25)

        sphericity = (np.pi ** (1.0 / 3)) * ((6.0 * volume_fill) ** (2.0 / 3)) / (surface + 1e-6)
        sphericity = min(sphericity, 5.0)

        rows.append({
            'Index': f"{well_id}_{label_id}",
            'Organoids_Volume': volume_raw,
            'Organoids_Volume_Fill': volume_fill,
            'Organoids_Surface': round(surface, 2),
            'Cavity_Volume': cavity_volume,
            'CavityNum': cavity_num,
            'LongAxis': long_axis,
            'ShortAxis': short_axis,
            'Wall_Thickness': round(wall_thickness, 4),
            'Sphericity': round(sphericity, 4),
        })

    return rows


print('=' * 60)
print('  ICC005 Transfer Validation')
print('=' * 60)

model = pickle.load(open(MODEL_PATH, 'rb'))
scaler_k6 = model['scaler_k6']
kmeans_k6 = model['kmeans_k6']
k4_map = model['k4_map']
selected_features = model['features']
scaler_pca = model['scaler_pca']
pca = model['pca']
weights = model['weights']
beta = model['beta']
intercept = model['intercept']

print(f'\n[Model] Training r={model["r"]:.4f}, N={model["n_wells"]}')
print(f'  PCA features ({len(selected_features)}): {selected_features}')
print(f'  K4 merge map: {k4_map}')
print(f'  Weights: {weights}')
print(f'  Intercept: {intercept:.6f}')

print('\n[Measuring] ICC005 masks...')
all_measurements = {}

for date_tag, date_dir in [('0701', '20230701'), ('0703', '20230703')]:
    seg_subdir = os.path.join(ICC005_DIR, date_dir, 'seg')
    seg_files = [f for f in sorted(os.listdir(seg_subdir))
                 if f.endswith('.nii.gz') and not f.startswith('plans')]
    print(f'  [{date_tag}] {len(seg_files)} wells')
    for seg_file in tqdm(seg_files, desc=f'  {date_tag}'):
        well_id = seg_file.replace('.nii.gz', '')
        seg_path = os.path.join(seg_subdir, seg_file)
        rows = measure_one_mask(seg_path, well_id)
        if rows:
            df = pd.DataFrame(rows)
            if well_id not in all_measurements:
                all_measurements[well_id] = {}
            all_measurements[well_id][date_tag] = df
        gc.collect()

wells_with_both = [w for w, dates in all_measurements.items() if '0701' in dates and '0703' in dates]
print(f'\n  Paired wells: {len(wells_with_both)} -> {wells_with_both}')

all_morph = []
for well_id in wells_with_both:
    for date_tag in ['0701', '0703']:
        df = all_measurements[well_id][date_tag]
        X = df[MORPH_FEATURES].fillna(0).values
        all_morph.append(X)
all_morph = np.vstack(all_morph)

X_std = scaler_k6.transform(all_morph)
k6_all = kmeans_k6.predict(X_std)
k4_all = np.array([k4_map[k] for k in k6_all])

all_morph_k4 = []
idx = 0
for well_id in wells_with_both:
    for date_tag in ['0701', '0703']:
        df = all_measurements[well_id][date_tag]
        n = len(df)
        k4_labels = k4_all[idx:idx + n]
        idx += n
        all_morph_k4.append((well_id, date_tag, k4_labels, df))

results = {}
for well_id, date_tag, k4_labels, df in all_morph_k4:
    if well_id not in results:
        results[well_id] = {}
    results[well_id][date_tag] = {}
    for k4 in range(4):
        mask = k4_labels == k4
        cnt = int(mask.sum())
        if cnt > 0:
            means = {f: df[f].values[mask].mean() for f in MORPH_FEATURES}
            means['Cnt'] = cnt
        else:
            means = {f: 0.0 for f in MORPH_FEATURES}
            means['Cnt'] = 0
        results[well_id][date_tag][k4] = means

delta_records = []
for well_id in wells_with_both:
    d1 = results[well_id]['0701']
    d5 = results[well_id]['0703']
    feat_dict = {}
    for k4 in range(4):
        for f in MORPH_FEATURES:
            sf = MORPH_SHORT[f]
            feat_name = f'{sf}_K{k4}'
            feat_dict[feat_name] = d5[k4][f] - d1[k4][f]
        feat_dict[f'Cnt_K{k4}'] = d5[k4]['Cnt'] - d1[k4]['Cnt']
    delta_records.append({'well_id': well_id, **feat_dict})

df_delta = pd.DataFrame(delta_records).set_index('well_id')
print(f'  Delta matrix: {df_delta.shape[0]} wells x {df_delta.shape[1]} features')

X_sel = df_delta[selected_features].fillna(0).values
X_std_sel = scaler_pca.transform(X_sel)
X_pca = pca.transform(X_std_sel)
scores = X_pca @ weights
beta_scores = intercept + X_sel @ beta

df_atp = pd.DataFrame([{'well_id': k, 'ATP': v} for k, v in ATP_DATA.items()])
df_score = pd.DataFrame({'well_id': wells_with_both, 'score': scores})
df_merged = df_score.merge(df_atp, on='well_id', how='inner')

print(f'  Matched: {len(df_merged)} wells')

r, p = pearsonr(df_merged['score'], df_merged['ATP'])
rho, _ = spearmanr(df_merged['score'], df_merged['ATP'])

if r < 0:
    df_merged['score'] = -df_merged['score']
    r, p = pearsonr(df_merged['score'], df_merged['ATP'])
    rho, _ = spearmanr(df_merged['score'], df_merged['ATP'])
    print(f'  (Scores flipped for positive correlation)')

print(f'\n{"="*60}')
print(f'  RESULTS')
print(f'{"="*60}')
print(f'  Pearson r  = {r:.4f}')
print(f'  p-value    = {p:.6f}')
print(f'  Spearman   = {rho:.4f}')
print(f'  N          = {len(df_merged)}')
print(f'  Model r (training) = {model["r"]:.4f}')
print(f'  Delta r   = {r - model["r"]:+.4f}')

print(f'\n  Per-well:')
for _, row in df_merged.iterrows():
    print(f'    {row["well_id"]:>6s}  Score={row["score"]:+.4f}  ATP={row["ATP"]:>10.0f}')

# =========== DIAGNOSTICS ===========
print(f'\n{"="*60}')
print(f'  DIAGNOSTICS')
print(f'{"="*60}')

print(f'\n[Cluster distribution]')
k6_dist, _ = np.histogram(k6_all, bins=range(7))
k4_dist = np.zeros(4, dtype=int)
for k6 in range(6):
    k4_dist[k4_map[k6]] += k6_dist[k6]
print(f'  K6: {dict(zip(range(6), k6_dist))}')
print(f'  K4: {dict(zip(range(4), k4_dist))}')
k4_pct = k4_dist / k4_dist.sum() * 100
for k4 in range(4):
    print(f'    K{k4}: {k4_dist[k4]:>6d} ({k4_pct[k4]:.1f}%)')

print(f'\n[Feature stats ICC005 vs model training]')
global_stats = {}
for f in MORPH_FEATURES:
    vals = np.concatenate([all_measurements[w][d][f].values for w in wells_with_both for d in ['0701','0703']])
    global_stats[f] = {'mean': vals.mean(), 'std': vals.std(), 'min': vals.min(), 'max': vals.max()}

print(f'  {"Feature":<22s} {"ICC005_mean":>12s} {"Train_mean":>12s} {"ICC005_std":>12s} {"Train_std":>12s}')
for i, f in enumerate(MORPH_FEATURES):
    gs = global_stats[f]
    print(f'  {f:<22s} {gs["mean"]:>12.1f} {scaler_k6.mean_[i]:>12.1f} {gs["std"]:>12.1f} {scaler_k6.scale_[i]:>12.1f}')

print(f'\n[Selected feature deltas]')
for f in selected_features:
    vals = df_delta[f].values
    print(f'  {f:<18s} mean={vals.mean():>+10.4f}  std={vals.std():>.4f}  min/max={vals.min():>+.4f}/{vals.max():>+.4f}')

print(f'\n[PCA analysis]')
pc_scores = X_pca
print(f'  PC explained var ratio: {pca.explained_variance_ratio_[:4]}')
for i in range(4):
    vals = pc_scores[:, i]
    print(f'  PC{i+1}: mean={vals.mean():>.4f}, std={vals.std():>.4f}, range=[{vals.min():>.4f}, {vals.max():>.4f}]')

# Save
out_df = df_merged[['well_id', 'score', 'ATP']].copy()
out_df.to_excel(os.path.join(OUT_DIR, 'icc005_transfer_results.xlsx'), index=False)
df_delta_out = df_delta.copy()
df_delta_out.insert(0, 'well_id', df_delta_out.index)
df_delta_out.to_excel(os.path.join(OUT_DIR, 'icc005_delta_features.xlsx'), index=False)

print(f'\n  Results saved to: {OUT_DIR}')
print(f'[Done]')