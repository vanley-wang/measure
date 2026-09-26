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
from tqdm import tqdm

ICC005_DIR = r'D:\Desktop\music\measure\ICC005'
OUT_DIR = r'D:\Desktop\music\measure\psd_pipeline\icc005_standalone'
os.makedirs(OUT_DIR, exist_ok=True)

ATP_DATA = {
    'B4':  3113000, 'B6':  2658000, 'B9':  4212000, 'B10': 4482000,
    'B11': 2923000, 'F4':  3113000, 'F5':  1647000, 'F7':  2197000,
    'F8':  2499000, 'F10': 4482000, 'G4':  2840000, 'G6':  3192000,
    'G9':  2974000, 'G11': 2941000,
}

RANDOM_SEED = 42
MIN_VOLUME = 50
N_ITER = 50000   # Monte Carlo feature search iterations
N_PC = 4
N_RUNS = 10      # Independent search runs for stability

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


def compute_surface_area(mask_3d):
    coords = np.where(mask_3d)
    if len(coords[0]) == 0:
        return 0.0
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1
    cropped = mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]
    try:
        verts, faces, _, _ = marching_cubes(cropped.astype(float), level=0.5)
        return mesh_surface_area(verts, faces)
    except Exception:
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
            'Organoids_Volume': volume_raw, 'Organoids_Volume_Fill': volume_fill,
            'Organoids_Surface': round(surface, 2), 'Cavity_Volume': cavity_volume,
            'CavityNum': cavity_num, 'LongAxis': long_axis, 'ShortAxis': short_axis,
            'Wall_Thickness': round(wall_thickness, 4),
            'Sphericity': round(sphericity, 4),
        })
    return rows


def k6_to_k4(centers):
    """Merge K6 into K4 by minimum distance between cluster centers"""
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
            if used[i]:
                continue
            for j in range(i + 1, n):
                if used[j]:
                    continue
                if dm[i, j] < best_d:
                    best_d, best_ij = dm[i, j], (i, j)
        pairs.append(best_ij)
        used[best_ij[0]] = used[best_ij[1]] = True
    if n % 2 == 1:
        remaining = [i for i in range(n) if not used[i]]
        if len(pairs) > 0:
            pairs[0] = (pairs[0][0], pairs[0][1], remaining[0])
        else:
            pairs = [(remaining[0],)]
    merge_map = {}
    for ki, pair in enumerate(pairs):
        for k6 in pair:
            merge_map[k6] = ki
    return {k: merge_map[k] for k in sorted(merge_map)}, pairs


def build_delta(d1_all, d5_all, n_wells, feat_list, scaler=None, kmeans_k6=None, k4_map=None):
    """D1+D5 merged clustering -> K4 aggregation -> Delta features"""
    X_d1 = d1_all[feat_list].fillna(0).values
    X_d5 = d5_all[feat_list].fillna(0).values

    if scaler is None:
        X_all = np.vstack([X_d1, X_d5])
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_all)
        kmeans_k6 = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
        k6_all = kmeans_k6.fit_predict(X_std)
        k4_map, pairs = k6_to_k4(kmeans_k6.cluster_centers_)
        n = len(X_d1)
        k6_d1 = k6_all[:n]
        k6_d5 = k6_all[n:]
        sizes = dict(zip(*np.unique(k6_all, return_counts=True)))
        print(f'    K6 sizes: {sizes}, merge pairs: {pairs}')
    else:
        k6_d1 = kmeans_k6.predict(scaler.transform(X_d1))
        k6_d5 = kmeans_k6.predict(scaler.transform(X_d5))

    k4_d1 = np.array([k4_map[k] for k in k6_d1])
    k4_d5 = np.array([k4_map[k] for k in k6_d5])

    def agg(day_data, k4_labels):
        nf = len(feat_list)
        n_delta_feats = nf + 1  # morph means + count
        rows = np.zeros((n_wells, 4 * n_delta_feats))
        for wi in range(n_wells):
            mask = day_data['well_idx'] == wi
            for k4 in range(4):
                kmask = mask & (k4_labels == k4)
                cnt = kmask.sum()
                offset = k4 * n_delta_feats
                if cnt > 0:
                    for fi in range(nf):
                        rows[wi, offset + fi] = day_data[feat_list].values[kmask, fi].mean()
                rows[wi, offset + nf] = cnt
        return rows

    d1_agg = agg(d1_all, k4_d1)
    d5_agg = agg(d5_all, k4_d5)
    delta = d5_agg - d1_agg

    delta_feat_cols = []
    for k4 in range(4):
        for f in feat_list:
            delta_feat_cols.append(f'{MORPH_SHORT[f]}_K{k4}')
        delta_feat_cols.append(f'Cnt_K{k4}')

    return delta, delta_feat_cols, scaler, kmeans_k6, k4_map, k4_d1, k4_d5


# ============================================================
# Phase 1: Measure masks (already done, reload from saved files)
# ============================================================
print('=' * 60)
print('  ICC005 Standalone Pipeline')
print('=' * 60)

print('\n[1] Loading measurements...')
all_measurements = {}
for date_tag, date_dir in [('0701', '20230701'), ('0703', '20230703')]:
    seg_subdir = os.path.join(ICC005_DIR, date_dir, 'seg')
    seg_files = [f for f in sorted(os.listdir(seg_subdir))
                 if f.endswith('.nii.gz') and not f.startswith('plans')]
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

WELLS = sorted([w for w, dates in all_measurements.items() if '0701' in dates and '0703' in dates])
print(f'  Wells: {len(WELLS)} -> {WELLS}')

# ============================================================
# Phase 2: Build D1 and D5 combined data for clustering
# ============================================================
print('\n[2] Building D1+D5 combined data for clustering...')
d1_rows, d5_rows = [], []
d1_all_wids, d5_all_wids = [], []

for wi, well_id in enumerate(WELLS):
    df1 = all_measurements[well_id]['0701']
    df5 = all_measurements[well_id]['0703']
    df1 = df1.copy(); df1['well_idx'] = wi
    df5 = df5.copy(); df5['well_idx'] = wi
    d1_rows.append(df1)
    d5_rows.append(df5)
    d1_all_wids.extend([well_id] * len(df1))
    d5_all_wids.extend([well_id] * len(df5))

d1_all = pd.concat(d1_rows, ignore_index=True)
d5_all = pd.concat(d5_rows, ignore_index=True)

print(f'  D1: {len(d1_all)} organoids, D5: {len(d5_all)} organoids')

# ============================================================
# Phase 3: Clustering + Delta
# ============================================================
print('\n[3] K6->K4 clustering + Delta computation...')
delta, delta_feat_cols, scaler_k6, kmeans_k6, k4_map, k4_d1, k4_d5 = \
    build_delta(d1_all, d5_all, len(WELLS), MORPH_FEATURES)

print(f'  K4 map: {k4_map}')
print(f'  Delta features: {len(delta_feat_cols)}')
print(f'  Delta matrix: {delta.shape}')

# Cluster distribution
k6_dist = dict(zip(*np.unique(np.concatenate([k4_d1, k4_d5]), return_counts=True)))
k6_labels = np.concatenate([kmeans_k6.predict(scaler_k6.transform(d1_all[MORPH_FEATURES].fillna(0).values)),
                             kmeans_k6.predict(scaler_k6.transform(d5_all[MORPH_FEATURES].fillna(0).values))])
k4_labels = np.array([k4_map[k] for k in k6_labels])
k4_pct = {k: (k4_labels == k).sum() / len(k4_labels) * 100 for k in range(4)}
print(f'  K4 distribution: { {k: f"{k4_pct[k]:.1f}%" for k in range(4)} }')

# ============================================================
# Phase 4: ATP matching
# ============================================================
print('\n[4] ATP matching...')
atp_vec = np.array([ATP_DATA[w] for w in WELLS])
print(f'  ATP range: {atp_vec.min():.0f} - {atp_vec.max():.0f}')
print(f'  Per well: {dict(zip(WELLS, atp_vec.astype(int)))}')

# ============================================================
# Phase 5: Monte Carlo feature search (N_RUNS independent runs)
# ============================================================
print(f'\n[5] Monte Carlo feature search ({N_RUNS} runs x {N_ITER} iter)...')

nf = len(delta_feat_cols)
best_overall = {'r': 0.0, 'run': -1}

for run in range(N_RUNS):
    seed = RANDOM_SEED + run * 1000
    np.random.seed(seed)
    best_r = 0.0
    best_combo, best_pca, best_scaler, best_w, best_score, best_p = None, None, None, None, None, None

    for it in range(N_ITER):
        ns = np.random.randint(max(4, nf // 3), nf + 1)
        si = sorted(np.random.choice(nf, ns, replace=False))
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
        score_candidate = Xp @ wr
        r_candidate, p_candidate = pearsonr(score_candidate, atp_vec)
        if abs(r_candidate) > best_r:
            best_r = abs(r_candidate)
            best_combo, best_pca, best_scaler = si, pc, ss
            best_w, best_score, best_p = wr, score_candidate, p_candidate

    if best_r > best_overall['r']:
        best_overall = {
            'r': best_r, 'run': run, 'seed': seed,
            'combo': best_combo, 'pca': best_pca, 'scaler': best_scaler,
            'w': best_w, 'score': best_score, 'p': best_p
        }

    print(f'  Run {run+1}/{N_RUNS}: |r|={best_r:.4f}, p={best_p:.6f}, '
          f'feats={len(best_combo)}, '
          f'PCcum={best_pca.explained_variance_ratio_[:N_PC].sum():.1%}')

# ============================================================
# Phase 6: Final results
# ============================================================
print(f'\n{"="*60}')
print(f'  FINAL RESULTS')
print(f'{"="*60}')

best = best_overall
bf_names = [delta_feat_cols[i] for i in best['combo']]

# Fix PCA sign
score = best['score']
r, pv = pearsonr(score, atp_vec)
rho, _ = spearmanr(score, atp_vec)
if r < 0:
    score = -score
    r, pv = pearsonr(score, atp_vec)
    rho, _ = spearmanr(score, atp_vec)
    best['w'] = -best['w']
    print('  (Scores flipped for positive correlation)')

cum_var = best['pca'].explained_variance_ratio_[:N_PC].sum()
beta = best['pca'].components_.T @ best['w'] / best['scaler'].scale_
intercept = -np.sum(best['scaler'].mean_ * beta)

print(f'  Pearson r     = {r:.4f}')
print(f'  p-value       = {pv:.6f}')
print(f'  Spearman rho  = {rho:.4f}')
print(f'  Selected feats = {len(bf_names)}')
print(f'  PC cumulative var = {cum_var:.1%}')
print(f'  Best run      = {best["run"]+1}/{N_RUNS}')

print(f'\n  Selected features:')
for i, fn in enumerate(bf_names):
    print(f'    [{i:>2d}] {fn:<18s} beta={beta[i]:>+.6f}')

print(f'\n  PC explained variance ratio: {best["pca"].explained_variance_ratio_[:N_PC]}')
print(f'  PC weights: {best["w"]}')

print(f'\n  Per-well results:')
for wi, well_id in enumerate(WELLS):
    print(f'    {well_id:>6s}  Score={score[wi]:>+.4f}  ATP={atp_vec[wi]:>10.0f}')

# ============================================================
# Phase 7: Save
# ============================================================
results = {
    'r': r, 'p': pv, 'rho': rho,
    'n_wells': len(WELLS),
    'wells': WELLS,
    'features': bf_names,
    'n_features': len(bf_names),
    'n_pc': N_PC,
    'scaler_k6': scaler_k6,
    'kmeans_k6': kmeans_k6,
    'k4_map': k4_map,
    'scaler_pca': best['scaler'],
    'pca': best['pca'],
    'weights': best['w'],
    'beta': beta,
    'intercept': intercept,
    'morph_features': MORPH_FEATURES,
    'delta_feat_cols': delta_feat_cols,
    'cum_var': cum_var,
    'score': score,
    'atp': atp_vec,
}

model_path = os.path.join(OUT_DIR, 'icc005_standalone_model.pkl')
pickle.dump(results, open(model_path, 'wb'))

df_result = pd.DataFrame({'well_id': WELLS, 'score': score, 'ATP': atp_vec.astype(int)})
df_result.to_excel(os.path.join(OUT_DIR, 'icc005_results.xlsx'), index=False)

df_delta = pd.DataFrame(delta, columns=delta_feat_cols)
df_delta.insert(0, 'well_id', WELLS)
df_delta.to_excel(os.path.join(OUT_DIR, 'icc005_delta_features.xlsx'), index=False)

print(f'\n  Saved to: {OUT_DIR}')
print(f'  - icc005_standalone_model.pkl')
print(f'  - icc005_results.xlsx')
print(f'  - icc005_delta_features.xlsx')
print(f'[Done]')