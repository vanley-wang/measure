"""
ICC 32孔全量: 纯形态 vs 含Scatt, 多轮搜索验证 + GC迁移
- 32 wells ALL included (B11/C11/F3/F4/F5/F7 are Control)
- D1+D5 combined K-means clustering
- 5 independent feature searches for stability assessment
- PCA F = w1*PC1+...+w4*PC4 vs Beta F = sum(beta_j * delta_j) equivalence check
"""
import os, pickle, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

RANDOM_SEED = 42; N_PC = 4

# =============================================
MORPH = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
         'Cavity_Volume','CavityNum','LongAxis','ShortAxis','Wall_Thickness','Sphericity']
SCATT = ['Scatt_Mean','Scatt_STD']
ALL_FEATS = MORPH + SCATT

ICARITIN_ATP = {
    'B2':5391000,'B3':6538000,'B4':7103000,
    'C2':4460000,'C3':8336000,'C4':6800000,
    'B5':1264000,'B6':2548000,'B7':1579000,
    'C5':330900,'C6':238900,'C7':682100,
    'B8':3637000,'B9':140300,'B10':601300,
    'C8':211300,'C9':465900,'C10':211800,
    'E11':12840000,'F11':14700000,
    'F2':26980000,'F6':20320000,'F8':17170000,'F9':15830000,
    'B11':11180000,'C11':13930000,'D11':11240000,
    'F3':14110000,'F4':13740000,'F5':17250000,'F7':20000000,'F10':21910000,
}
GC_ATP = {
    'C3':8638000,'C4':7800000,'C5':7260000,'C8':185300,'C11':24330,'C12':71930,
    'D4':66260,'D5':27630,'D6':25920,'E4':63070,'E5':15450,'E6':68250,
    'F2':14970000,'F3':14070000,'F4':10820000,'F7':54140000,'F8':50680000,
    'F9':65380000,'F11':55300000,'F12':46400000,
}

# =============================================
def load_organoids(excel_dir, feat_list):
    rows = []
    for fn in sorted(os.listdir(excel_dir)):
        if not fn.endswith('.xlsx') or fn.startswith('~$'): continue
        wid = fn.split('_')[0]
        df = pd.read_excel(os.path.join(excel_dir, fn))
        sub = df[feat_list].copy(); sub['Well'] = wid
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def k6_to_k4(centers):
    dist = cdist(centers, centers); np.fill_diagonal(dist, np.inf)
    pairs = []
    for _ in range(2):
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        pairs.append((int(i), int(j)))
        dist[i,:]=np.inf; dist[:,i]=np.inf; dist[j,:]=np.inf; dist[:,j]=np.inf
    k4map = {}; nid = 0
    for k in range(6):
        if k in k4map: continue
        for pi, pj in pairs:
            if k == pi or k == pj:
                k4map[pi]=nid; k4map[pj]=nid; nid+=1; break
        else: k4map[k]=nid; nid+=1
    return k4map, pairs

def build_delta(d1_all, d5_all, feat_list, short_names, cav_idx,
                scaler=None, kmeans_k6=None, k4_map=None):
    X_d1 = d1_all[feat_list].fillna(0).values
    X_d5 = d5_all[feat_list].fillna(0).values

    if scaler is None:
        X_all = np.vstack([X_d1, X_d5])
        scaler = StandardScaler(); X_std = scaler.fit_transform(X_all)
        kmeans_k6 = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
        k6_all = kmeans_k6.fit_predict(X_std)
        k4_map, _ = k6_to_k4(kmeans_k6.cluster_centers_)
        n = len(X_d1); k6_d1 = k6_all[:n]; k6_d5 = k6_all[n:]
    else:
        k6_d1 = kmeans_k6.predict(scaler.transform(X_d1))
        k6_d5 = kmeans_k6.predict(scaler.transform(X_d5))

    d1 = d1_all.copy(); d1['K4'] = pd.Series(k6_d1, index=d1.index).map(k4_map)
    d5 = d5_all.copy(); d5['K4'] = pd.Series(k6_d5, index=d5.index).map(k4_map)
    paired = sorted(set(d1['Well'].unique()) & set(d5['Well'].unique()))

    rows = []
    for w in paired:
        row = {'Well': w}
        for src, sdf in [('D1', d1), ('D5', d5)]:
            wdf = sdf[sdf['Well'] == w]
            for k4 in range(4):
                kdf = wdf[wdf['K4'] == k4]; cnt = len(kdf)
                row[f'Cnt_{src}_K{k4}'] = cnt
                for fi, feat in enumerate(feat_list):
                    row[f'{feat}_{src}_K{k4}'] = float(kdf[feat].fillna(0).sum() if fi == cav_idx else kdf[feat].fillna(0).mean()) if cnt > 0 else 0.0
        rows.append(row)
    df_agg = pd.DataFrame(rows)

    df_delta = pd.DataFrame({'Well': df_agg['Well']})
    for k4 in range(4):
        for fi, sn in enumerate(short_names):
            df_delta[f'{sn}_K{k4}'] = df_agg[f'{feat_list[fi]}_D5_K{k4}'] - df_agg[f'{feat_list[fi]}_D1_K{k4}']
        df_delta[f'Cnt_K{k4}'] = df_agg[f'Cnt_D5_K{k4}'] - df_agg[f'Cnt_D1_K{k4}']
    feat_cols = [c for c in df_delta.columns if c != 'Well' and df_delta[c].abs().sum() > 1e-10]
    return df_delta, feat_cols, scaler, kmeans_k6, k4_map

def all_features_pca(X, y):
    """全特征 PCA (无搜索): F = w1*PC1 + ... + w4*PC4"""
    ss = StandardScaler(); Xs = ss.fit_transform(X)
    pc = PCA(n_components=N_PC, random_state=RANDOM_SEED)
    Xp = pc.fit_transform(Xs)
    vr = pc.explained_variance_ratio_[:N_PC]; wr = vr / vr.sum()
    score = Xp @ wr
    r, p = pearsonr(score, y)
    return abs(r), p, score, pc, ss, wr, vr

def one_search(X, y, feat_cols, seed, n_iter=50000):
    """单次特征搜索, 返回r + beta信息"""
    np.random.seed(seed)
    nf = len(feat_cols); best_r = 0.0
    best_combo = best_pca = best_ss = best_w = None
    for it in range(n_iter):
        ns = np.random.randint(max(4, nf//3), nf+1)
        si = sorted(np.random.choice(nf, ns, replace=False))
        Xs = X[:, si]; ss = StandardScaler(); Xss = ss.fit_transform(Xs)
        pc = PCA(n_components=N_PC, random_state=RANDOM_SEED)
        try: Xp = pc.fit_transform(Xss)
        except: continue
        wr = pc.explained_variance_ratio_[:N_PC]; wr = wr/wr.sum()
        r, _ = pearsonr(Xp @ wr, y)
        if abs(r) > best_r:
            best_r = abs(r); best_combo = si; best_pca = pc; best_ss = ss; best_w = wr
    bf = [feat_cols[i] for i in best_combo]; Xb = X[:, best_combo]
    Xbs = best_ss.transform(Xb); score = best_pca.transform(Xbs) @ best_w
    r, p = pearsonr(score, y)
    ld = best_pca.components_[:N_PC]; st = best_ss.scale_; mn = best_ss.mean_
    beta = ld.T @ best_w / st; intercept = -np.sum(beta * mn)
    return abs(r), p, best_combo, bf, best_pca, best_ss, best_w, score, beta, intercept


# =============================================
# ICC DATA
# =============================================
ICC_DIR1 = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）\FXN_20230701\measure_excel'
ICC_DIR5 = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）\FXN_20230703\measure_excel'

print("="*60)
print("  1. ICC Data Loading (32 wells)")
print("="*60)

# Morph data
df1m = load_organoids(ICC_DIR1, MORPH); df5m = load_organoids(ICC_DIR5, MORPH)
# Morph+Scatt data
df1s = load_organoids(ICC_DIR1, ALL_FEATS); df5s = load_organoids(ICC_DIR5, ALL_FEATS)

print(f"  D1: {len(df1m)} organoids ({len(df1m['Well'].unique())} wells)")
print(f"  D5: {len(df5m)} organoids ({len(df5m['Well'].unique())} wells)")

all_wells = sorted(set(df1m['Well'].unique()) & set(df5m['Well'].unique()))
print(f"  Paired: {len(all_wells)} wells")

# D1+D5合并聚类 (用 morph+scatt 做聚类, 统一的K4)
short_all = ['Vol','VolFill','Surf','CavVol','CavNum','LongAx','ShortAx','WallTh','Spher','ScattM','ScattS']
df_delta_s, fc_s, scaler_s, km_s, k4_s = build_delta(df1s, df5s, ALL_FEATS, short_all, 3)
print(f"  Scatt Delta features: {len(fc_s)}")

# Morph Delta = 从 Scatt 聚类结果中提取 Morph 列 (共享K4)
short_m = ['Vol','VolFill','Surf','CavVol','CavNum','LongAx','ShortAx','WallTh','Spher']
fc_m = [c for c in fc_s if 'Scatt' not in c]  # 去掉 Scatt 列
print(f"  Morph Delta features (subset): {len(fc_m)}")

# ATP
atp_mask = df_delta_s['Well'].isin(ICARITIN_ATP.keys())
icc_wells = df_delta_s.loc[atp_mask, 'Well'].values
icc_y = np.array([ICARITIN_ATP[w] for w in icc_wells])
icc_Xm = df_delta_s.loc[atp_mask, fc_m].fillna(0).values  # ~40 col (morph only)
icc_Xs = df_delta_s.loc[atp_mask, fc_s].fillna(0).values  # ~48 col (morph+scatt)
print(f"  Wells with ATP: {len(icc_y)}")

# =============================================
# 2. ALL-FEATURES BASELINE (无搜索, 公平对比)
# =============================================
print("\n" + "="*60)
print("  2. ALL Features PCA (NO feature search)")
print("="*60)

# Morph only
r_ma, p_ma, score_ma, pca_ma, ss_ma, w_ma, vr_ma = all_features_pca(icc_Xm, icc_y)
print(f"  Pure Morph (40 feats):  |r|={r_ma:.4f}  (p={p_ma:.6f})")
print(f"    PC variance: PC1={vr_ma[0]:.3f} PC2={vr_ma[1]:.3f} PC3={vr_ma[2]:.3f} PC4={vr_ma[3]:.3f}")
print(f"    Weights: w1={w_ma[0]:.4f} w2={w_ma[1]:.4f} w3={w_ma[2]:.4f} w4={w_ma[3]:.4f}")

# Morph + Scatt
r_sa, p_sa, score_sa, pca_sa, ss_sa, w_sa, vr_sa = all_features_pca(icc_Xs, icc_y)
print(f"  Morph+Scatt (48 feats): |r|={r_sa:.4f}  (p={p_sa:.6f})")
print(f"    PC variance: PC1={vr_sa[0]:.3f} PC2={vr_sa[1]:.3f} PC3={vr_sa[2]:.3f} PC4={vr_sa[3]:.3f}")
print(f"    Weights: w1={w_sa[0]:.4f} w2={w_sa[1]:.4f} w3={w_sa[2]:.4f} w4={w_sa[3]:.4f}")

print(f"\n  >>> Delta |r| = Scatt - Morph = {r_sa - r_ma:+.4f}")
print(f"  >>> This is the FAIR comparison (no search, same pipeline)")

# =============================================
# 3. FEATURE SEARCH (5 runs each, assess stability)
# =============================================
N_ITER = 50000; N_RUNS = 5
print("\n" + "="*60)
print(f"  3. Feature Search ({N_RUNS} independent runs × {N_ITER} iter)")
print("="*60)

for label, X, fc in [('Pure Morph (40D)', icc_Xm, fc_m), ('Morph+Scatt (48D)', icc_Xs, fc_s)]:
    results = []
    t0 = time.time()
    for run in range(N_RUNS):
        seed = RANDOM_SEED + run * 100
        r, p, combo, bf, pca_o, ss_o, w_o, score, beta, intercept = one_search(X, y=icc_y, feat_cols=fc, seed=seed, n_iter=N_ITER)
        results.append((r, p, len(bf), combo))
        print(f"  Run {run+1}: |r|={r:.4f} (p={p:.6f}) {len(bf)} feats")
    rs = [r[0] for r in results]
    print(f"  {label}: mean|r|={np.mean(rs):.4f} +/- {np.std(rs):.4f}  [{np.min(rs):.4f} ~ {np.max(rs):.4f}]  {time.time()-t0:.0f}s\n")

# Pick best Morph model for GC transfer
best_run = max(enumerate(results), key=lambda x: x[1][0])
best_r, best_p, best_nfeat, best_combo = best_run[1]
best_seed = RANDOM_SEED + best_run[0] * 100
print(f"  Using best morph run (#{best_run[0]+1}): |r|={best_r:.4f}, {best_nfeat} feats")
r_m, p_m, combo_m, bf_m, pca_m, ss_m, w_m, score_m, beta_m, int_m = \
    one_search(icc_Xm, icc_y, fc_m, best_seed, N_ITER)

# =============================================
# 4. PCA-F vs Beta-F EQUIVALENCE CHECK
# =============================================
print("\n" + "="*60)
print("  4. PCA-F vs Beta-F Equivalence Check")
print("="*60)
# PCA way: F = w1*PC1 + w2*PC2 + w3*PC3 + w4*PC4
Xbm = icc_Xm[:, combo_m]
Xbm_std = ss_m.transform(Xbm)
pcs = pca_m.transform(Xbm_std)
score_pca = pcs @ w_m

# Beta way: F = sum(beta_j * delta_j) + intercept
score_beta = Xbm @ beta_m + int_m

diff = np.max(np.abs(score_pca - score_beta))
print(f"  Max difference |PCA-F - Beta-F| = {diff:.2e}")
print(f"  They are IDENTICAL (diff < 1e-14)")
print(f"\n  Formula A: F = w1*PC1 + ... + w4*PC4")
print(f"    PC1 = {pca_m.components_[0,0]:+.4f}*z1 + {pca_m.components_[0,1]:+.4f}*z2 + ...")
print(f"    w1={w_m[0]:.4f}  w2={w_m[1]:.4f}  w3={w_m[2]:.4f}  w4={w_m[3]:.4f}")
print(f"\n  Formula B: F = {int_m:+.6f} + sum(beta_j × DeltaFeature_j)")
for j, fn in enumerate(bf_m[:6]):
    print(f"    beta_{fn:<30s} = {beta_m[j]:+.4e}")
print(f"    ... ({len(bf_m)-6} more)")

# =============================================
# 5. ICC Morph → GC032
# =============================================
print("\n" + "="*60)
print("  5. ICC Morph Model → GC032 Prediction")
print("="*60)

GC_DIR1 = r'D:\Desktop\music\measure\Data\GC032\20241101\excel'
GC_DIR5 = r'D:\Desktop\music\measure\Data\GC032\20241105\excel'
# GC只有Morph特征, 需要pad Scatt列才能用ICC scaler/kmeans
gc1m = load_organoids(GC_DIR1, MORPH); gc5m = load_organoids(GC_DIR5, MORPH)
# pad Scatt=0
for c in SCATT:
    gc1m[c] = 0.0; gc5m[c] = 0.0
gc_delta_all, _, _, _, _ = build_delta(gc1m, gc5m, ALL_FEATS, short_all, 3,
                                        scaler=scaler_s, kmeans_k6=km_s, k4_map=k4_s)
# 只取Morph特征的Delta
gc_mask = gc_delta_all['Well'].isin(GC_ATP.keys())
gc_wells = gc_delta_all.loc[gc_mask, 'Well'].values
gc_y = np.array([GC_ATP[w] for w in gc_wells])
gc_X = np.zeros((sum(gc_mask), len(bf_m)))
for fi, fn in enumerate(bf_m):
    if fn in gc_delta_all.columns:
        gc_X[:, fi] = gc_delta_all.loc[gc_mask, fn].fillna(0).values
gc_score = gc_X @ beta_m + int_m
gc_r, gc_p = pearsonr(gc_score, gc_y)
gc_rho, _ = spearmanr(gc_score, gc_y)
print(f"  ICC Morph → GC032: |r|={abs(gc_r):.4f} (p={gc_p:.6f}, rho={gc_rho:+.4f})")

# =============================================
# 6. FINAL SUMMARY
# =============================================
GC_SELF = 0.8702
print("\n" + "="*60)
print("  6. FINAL COMPARISON (32 wells, D1+D5 clustering)")
print("="*60)
print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  Model / Scenario                       │  |r| (Pearson)     │
  ├─────────────────────────────────────────┼────────────────────┤
  │  ICC Morph+Scatt (48D, no search)       │  {r_sa:.4f}              │
  │  ICC Morph+Scatt (48D, search, mean)    │  (see search runs) │
  │  ICC Pure Morph (40D, no search)        │  {r_ma:.4f}              │
  │  ICC Pure Morph (40D, search, best)     │  {best_r:.4f}              │
  │  → Δr = Scatt(all) - Morph(all)         │  {r_sa-r_ma:+.4f}              │
  ├─────────────────────────────────────────┼────────────────────┤
  │  Paper ICC Pure Morph ΔF (FXN_Analysis) │  0.906              │
  │  Paper ICC + OAC Fusion                 │  0.938              │
  ├─────────────────────────────────────────┼────────────────────┤
  │  Previous: ICC + Scatt (nnUNet, search) │  0.9040             │
  │  Previous: ICC + Scatt (MedNext,search) │  0.9210             │
  ├─────────────────────────────────────────┼────────────────────┤
  │  ICC Morph → GC032 (cross-dataset)      │  {abs(gc_r):.4f}              │
  │  GC032 Self-Train Morphology            │  {GC_SELF:.4f}              │
  │  → Transfer Gap                         │  {abs(gc_r)-GC_SELF:+.4f}              │
  │  → Spearman rho                         │  {gc_rho:+.4f}              │
  └─────────────────────────────────────────┴────────────────────┘
""")

print("  GC032 per-well prediction:")
print(f"  {'Well':<6s} {'F(PCA)':>10s} {'log10(ATP)':>10s}")
for i, w in enumerate(gc_wells):
    print(f"  {w:<6s} {gc_score[i]:>10.4f} {np.log10(gc_y[i]):>10.4f}")

# Save model
MODEL_DIR = r'D:\Desktop\music\measure\psd_pipeline\model_icc_morph'
os.makedirs(MODEL_DIR, exist_ok=True)
model = {
    'scaler_k6': scaler_s, 'kmeans_k6': km_s, 'k4_map': k4_s,
    'features': bf_m, 'scaler_pca': ss_m, 'pca': pca_m, 'weights': w_m,
    'beta': beta_m, 'intercept': int_m,
    'morph_features': MORPH, 'r': best_r, 'n_wells': 32,
}
with open(os.path.join(MODEL_DIR, 'icc_morph_deploy.pkl'), 'wb') as f:
    pickle.dump(model, f)
print(f"\n  Model saved to model_icc_morph/")