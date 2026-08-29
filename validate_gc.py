import os, sys, glob, pickle, warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'step7_pca_atp_v2_stratified')))
from step7_pca_atp_v2_stratified.modeling import stratified_median_aggregation
from step7_pca_atp_v2_stratified.config import ATP_DATABASE, HEALTHY_CLUSTERS
from cluster_utils import RAW_FEATURES, load_model_package

warnings.filterwarnings('ignore')

GC_DIR = os.path.join('Data', 'FXN_2023_new（GC）')
GC_MODEL = os.path.join('model', 'Kmeans-scatt-gc.pickle')
ICC_PCA = os.path.join('model', 'PCA_ATP_v2_extended.pkl')

print('=' * 70)
print('  GC CROSS-CANCER VALIDATION')
print('  KMeans: GC-trained  |  PCA weights: ICC-trained')
print('=' * 70)

# ============================================================
# Step 1: Load ICC PCA model
# ============================================================
with open(ICC_PCA, 'rb') as f:
    icc_pkg = pickle.load(f)

icc_scaler = icc_pkg['scaler']
icc_pca = icc_pkg['pca']
icc_wts = icc_pkg['weights']
icc_sel_feats = icc_pkg['selected_features']

print(f'\nICC PCA model loaded:')
print(f'  Selected features: {len(icc_sel_feats)}')
for i, f in enumerate(icc_sel_feats):
    print(f'    {i+1}. {f}')
print(f'  PCA components: {icc_pca.n_components_}')
print(f'  PC weights: {icc_wts}')

# ============================================================
# Step 2: Load GC data with GC KMeans clustering
# ============================================================
pkg = load_model_package(GC_MODEL)
gc_model = pkg['kmeans']
gc_scaler = pkg['scaler']
gc_r2f = pkg['raw_to_final']

feats_list = list(RAW_FEATURES)

all_dfs = []
wells_seen = set()
for day_suffix, day_val in [('0701', '0701'), ('0703', '0703')]:
    day_dir = os.path.join(GC_DIR, f'FXN_2023{day_suffix}', 'measure_excel')
    if not os.path.exists(day_dir):
        continue
    for fpath in sorted(glob.glob(os.path.join(day_dir, '*.xlsx'))):
        fname = os.path.basename(fpath)
        wid = fname.replace('_0701.xlsx', '').replace('_0703.xlsx', '').replace('.xlsx', '')
        if not wid:
            continue
        key = (wid, day_val)
        if key in wells_seen:
            continue
        tmp = pd.read_excel(fpath)
        if 'Index' in tmp.columns and 'Object_Id' not in tmp.columns:
            tmp = tmp.rename(columns={'Index': 'Object_Id'})
        if 'Cavity_Ratio' not in tmp.columns:
            tmp['Cavity_Ratio'] = tmp['Cavity_Volume'] / (tmp['Organoids_Volume_Fill'] + 1e-10)
        tmp = tmp.dropna(subset=feats_list)
        if len(tmp) == 0:
            continue
        tmp['_well'] = fname.replace('.xlsx', '')
        tmp['_well_id'] = wid
        tmp['_day'] = day_val
        all_dfs.append(tmp)
        wells_seen.add(key)

df_gc = pd.concat(all_dfs, ignore_index=True)
wells_gc = sorted(set(w for w, _ in wells_seen))
print(f'\nGC data loaded: {len(wells_gc)} wells, {len(df_gc)} objects')

X_gc = gc_scaler.transform(df_gc[feats_list])
labels_gc = np.array([gc_r2f.get(l, l) for l in gc_model.predict(X_gc)])
df_gc['Cluster'] = labels_gc

print(f'GC KMeans cluster distribution:')
for c in range(4):
    cnt = (labels_gc == c).sum()
    names = {0: 'Red(H)', 1: 'Yellow(H)', 2: 'Green(I)', 3: 'Blue(D)'}
    print(f'  {names[c]:12s}: {cnt:>8,} ({cnt/len(labels_gc)*100:>5.1f}%)')

# ============================================================
# Step 3: Build well statistics (ws)
# ============================================================
stats_list = []
for (_w, _d), g in df_gc.groupby(['_well', '_day']):
    tot = len(g)
    vc = g['Cluster'].value_counts()
    r, y, gr, b = [vc.get(i, 0) for i in range(4)]
    h = r + y
    stats_list.append({
        '_well': _w, '_day': _d,
        '_well_id': _w.split('_')[0] if '_' in _w else _w,
        'Total': tot, 'Red': r, 'Yellow': y, 'Green': gr, 'Blue': b, 'Healthy': h,
        'Red_Fraction': r / tot, 'Yellow_Fraction': y / tot,
        'Healthy_Fraction': h / tot, 'Blue_Fraction': b / tot,
    })
ws_gc = pd.DataFrame(stats_list)
print(f'\nWell-day stats: {len(ws_gc)} rows')

# ============================================================
# Step 4: Feature extraction (same pipeline as ICC)
# ============================================================
fm_gc, extended_sel, d3_feats, d5_feats = stratified_median_aggregation(
    df_gc, ws_gc, feats_list, wells_gc
)
fm_gc['ATP'] = fm_gc['Well_ID'].map(ATP_DATABASE)
print(f'\nFeature matrix: {fm_gc.shape}')
print(f'Extended features: {len(extended_sel)}')

# ============================================================
# Step 5: Apply ICC PCA weights
# ============================================================
common_feats = [f for f in icc_sel_feats if f in fm_gc.columns]
missing = set(icc_sel_feats) - set(common_feats)
if missing:
    print(f'\nWARNING: Missing features: {missing}')

X_gc_feat = fm_gc[common_feats].values
valid = ~np.isnan(X_gc_feat).any(axis=1) & fm_gc['ATP'].notna()
print(f'\nComplete wells: {valid.sum()}/{len(fm_gc)}')

X_valid = X_gc_feat[valid]
atp_valid = fm_gc.loc[valid, 'ATP'].values
ids_valid = fm_gc.loc[valid, 'Well_ID'].values

X_scaled = icc_scaler.transform(X_valid)
X_pcs = icc_pca.transform(X_scaled)
score_gc = np.dot(X_pcs, icc_wts)

r_gc, p_gc = pearsonr(score_gc, atp_valid)
sp_gc, sp_p_gc = spearmanr(score_gc, atp_valid)

# ============================================================
# Step 6: Results
# ============================================================
print('\n' + '=' * 50)
print('GC CROSS-CANCER VALIDATION RESULTS')
print('=' * 50)
print(f'Test samples: {valid.sum()} wells')
print(f'Features: {len(common_feats)}/{len(icc_sel_feats)}')
print(f'Pearson r  = {r_gc:.4f} (p={p_gc:.2e})')
print(f'Spearman rho = {sp_gc:.4f} (p={sp_p_gc:.2e})')

print(f'\n┌─────────────────────────────────────────────────┐')
print(f'│  ICC (胆管癌) training:  r = 0.9311             │')
print(f'│  GC  (胃癌)   test:      r = {r_gc:.4f}             │')
print(f'│  Delta r = {0.9311 - r_gc:+.4f}                          │')
print(f'└─────────────────────────────────────────────────┘')

print('\nScore vs ATP:')
for i, (sid, s, a) in enumerate(zip(ids_valid, score_gc, atp_valid)):
    print(f'  {sid}: Score={s:+.4f}, ATP={a:.0f}')

print('\nDONE')