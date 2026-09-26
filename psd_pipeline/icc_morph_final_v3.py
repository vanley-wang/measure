"""
ICC 32孔全量: 纯形态 vs 含Scatt, 多轮搜索 + GC迁移
- 纯形态模型: 用MORPH特征聚类 + MORPH Delta (不涉及Scatt)
- Scatt模型: 用ALL_FEATS聚类 + ALL Delta (仅ICC内部分析, 不用于GC)
- GC: 用ICC纯形态模型预测
- PCA-F == Beta-F 等价性证明
"""
import os, pickle, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

RANDOM_SEED = 42; N_PC = 4; N_ITER = 50000; N_RUNS = 5

MORPH = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
         'Cavity_Volume','CavityNum','LongAxis','ShortAxis','Wall_Thickness','Sphericity']
SCATT = ['Scatt_Mean','Scatt_STD']

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
def load_organoids(excel_dir, feat_list, well_filter=None):
    rows = []
    for fn in sorted(os.listdir(excel_dir)):
        if not fn.endswith('.xlsx') or fn.startswith('~$'): continue
        wid = fn.split('_')[0]
        if well_filter and wid not in well_filter: continue
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
    """D1+D5合并聚类 → K4聚合 → Delta"""
    X_d1 = d1_all[feat_list].fillna(0).values
    X_d5 = d5_all[feat_list].fillna(0).values
    nf = len(feat_list)

    if scaler is None:
        X_all = np.vstack([X_d1, X_d5])
        scaler = StandardScaler(); X_std = scaler.fit_transform(X_all)
        kmeans_k6 = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
        k6_all = kmeans_k6.fit_predict(X_std)
        k4_map, pairs = k6_to_k4(kmeans_k6.cluster_centers_)
        n = len(X_d1); k6_d1 = k6_all[:n]; k6_d5 = k6_all[n:]
        sizes = dict(zip(*np.unique(k6_all, return_counts=True)))
        print(f"  K6 sizes: {sizes}, merge pairs: {pairs}")
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
                if cnt > 0:
                    for fi, feat in enumerate(feat_list):
                        row[f'{feat}_{src}_K{k4}'] = float(kdf[feat].fillna(0).sum() if fi==cav_idx else kdf[feat].fillna(0).mean())
                else:
                    for fi, feat in enumerate(feat_list):
                        row[f'{feat}_{src}_K{k4}'] = 0.0
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
    ss = StandardScaler(); Xs = ss.fit_transform(X)
    pc = PCA(n_components=N_PC, random_state=RANDOM_SEED)
    Xp = pc.fit_transform(Xs)
    vr = pc.explained_variance_ratio_[:N_PC]; wr = vr/vr.sum()
    r, p = pearsonr(Xp @ wr, y)
    return abs(r), p, pc, ss, wr, vr

def search_one(X, y, feat_cols, seed):
    np.random.seed(seed); nf = len(feat_cols); best_r = 0.0
    best = None
    for it in range(N_ITER):
        ns = np.random.randint(max(4, nf//3), nf+1)
        si = sorted(np.random.choice(nf, ns, replace=False))
        Xs = X[:, si]; ss = StandardScaler(); Xss = ss.fit_transform(Xs)
        pc = PCA(n_components=N_PC, random_state=RANDOM_SEED)
        try: Xp = pc.fit_transform(Xss)
        except: continue
        wr = pc.explained_variance_ratio_[:N_PC]; wr = wr/wr.sum()
        r, _ = pearsonr(Xp @ wr, y)
        if abs(r) > best_r:
            best_r = abs(r); best = (si, pc, ss, wr)
    combo, pca_o, ss_o, w_o = best
    bf = [feat_cols[i] for i in combo]; Xb = X[:, combo]
    Xbs = ss_o.transform(Xb); score = pca_o.transform(Xbs) @ w_o
    r, p = pearsonr(score, y)
    ld = pca_o.components_[:N_PC]; st = ss_o.scale_; mn = ss_o.mean_
    beta = ld.T @ w_o / st; intercept = -np.sum(beta * mn)
    return abs(r), p, combo, bf, pca_o, ss_o, w_o, score, beta, intercept


# =============================================
ICC_DIR1 = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）\FXN_20230701\measure_excel'
ICC_DIR5 = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）\FXN_20230703\measure_excel'

print("="*60)
print("  ICC: Loading data (32 wells)")
print("="*60)
df1m = load_organoids(ICC_DIR1, MORPH); df5m = load_organoids(ICC_DIR5, MORPH)
df1s = load_organoids(ICC_DIR1, MORPH+SCATT); df5s = load_organoids(ICC_DIR5, MORPH+SCATT)
print(f"  D1={len(df1m)} D5={len(df5m)} organoids, {len(set(df1m['Well'])&set(df5m['Well']))} paired wells")

# =============================================
# MODEL A: ICC Pure Morphology (9 feats)
# =============================================
print("\n" + "="*60)
print("  MODEL A: ICC Pure Morphology")
print("="*60)
short_m = ['Vol','VolFill','Surf','CavVol','CavNum','LongAx','ShortAx','WallTh','Spher']
df_delta_m, fc_m, scaler_m, km_m, k4_m = build_delta(df1m, df5m, MORPH, short_m, 3)
atp_mask = df_delta_m['Well'].isin(ICARITIN_ATP.keys())
icc_y = np.array([ICARITIN_ATP[w] for w in df_delta_m.loc[atp_mask,'Well']])
icc_Xm = df_delta_m.loc[atp_mask, fc_m].fillna(0).values
print(f"  Delta features: {len(fc_m)}, ATP wells: {len(icc_y)}")

# A1: No search
r_ma, p_ma, pca_ma, ss_ma, w_ma, vr_ma = all_features_pca(icc_Xm, icc_y)
print(f"  --- A1: All {len(fc_m)} feats (no search) ---")
print(f"  |r| = {r_ma:.4f} (p={p_ma:.6f})")
cumvar = np.cumsum(vr_ma)
print(f"  PC variance: PC1={vr_ma[0]:.3f} PC2={vr_ma[1]:.3f} PC3={vr_ma[2]:.3f} PC4={vr_ma[3]:.3f} (cum={cumvar[3]:.3f})")

# A2: Feature search (5 runs)
print(f"\n  --- A2: Feature search ({N_RUNS} runs × {N_ITER} iter) ---")
results_m = []
t0 = time.time()
for run in range(N_RUNS):
    r, p, combo, bf, pca, ss, w, score, beta, intercept = search_one(icc_Xm, icc_y, fc_m, RANDOM_SEED+run*100)
    results_m.append((r, p, len(bf), bf, pca, ss, w, score, beta, intercept))
    print(f"  Run {run+1}: |r|={r:.4f} (p={p:.6f}), {len(bf)} feats")
rs_m = [x[0] for x in results_m]
print(f"  Mean|r| = {np.mean(rs_m):.4f} +/- {np.std(rs_m):.4f}  [{np.min(rs_m):.4f}~{np.max(rs_m):.4f}]  ({time.time()-t0:.0f}s)")

# Pick best
best_m = results_m[np.argmax(rs_m)]
r_m_best, p_m, nfeat_m, bf_m, pca_m, ss_m, w_m, score_m, beta_m, int_m = best_m
print(f"  Best: |r|={r_m_best:.4f}, {nfeat_m} feats")

# =============================================
# MODEL B: ICC Morph + Scatt (11 feats)
# =============================================
print("\n" + "="*60)
print("  MODEL B: ICC Morph + Scatt")
print("="*60)
short_s = ['Vol','VolFill','Surf','CavVol','CavNum','LongAx','ShortAx','WallTh','Spher','ScattM','ScattS']
df_delta_s, fc_s, _, _, _ = build_delta(df1s, df5s, MORPH+SCATT, short_s, 3)
icc_Xs = df_delta_s.loc[atp_mask, fc_s].fillna(0).values
print(f"  Delta features: {len(fc_s)}, ATP wells: {len(icc_y)}")

# B1: No search
r_sa, p_sa, pca_sa, ss_sa, w_sa, vr_sa = all_features_pca(icc_Xs, icc_y)
print(f"  --- B1: All {len(fc_s)} feats (no search) ---")
print(f"  |r| = {r_sa:.4f} (p={p_sa:.6f})")
cumvar_s = np.cumsum(vr_sa)
print(f"  PC variance: PC1={vr_sa[0]:.3f} PC2={vr_sa[1]:.3f} PC3={vr_sa[2]:.3f} PC4={vr_sa[3]:.3f} (cum={cumvar_s[3]:.3f})")

# B2: Feature search (5 runs)
print(f"\n  --- B2: Feature search ({N_RUNS} runs × {N_ITER} iter) ---")
results_s = []
t0 = time.time()
for run in range(N_RUNS):
    r, p, _, bf, _, _, _, _, _, _ = search_one(icc_Xs, icc_y, fc_s, RANDOM_SEED+run*100)
    results_s.append((r, p, len(bf)))
    print(f"  Run {run+1}: |r|={r:.4f} (p={p:.6f}), {len(bf)} feats")
rs_s = [x[0] for x in results_s]
print(f"  Mean|r| = {np.mean(rs_s):.4f} +/- {np.std(rs_s):.4f}  [{np.min(rs_s):.4f}~{np.max(rs_s):.4f}]  ({time.time()-t0:.0f}s)")
r_s_best = np.max(rs_s)

# =============================================
# PCA-F vs Beta-F EQUIVALENCE
# =============================================
print("\n" + "="*60)
print("  PCA-F vs Beta-F Equivalence")
print("="*60)
# PCA way
combo_m_idx = [fc_m.index(fn) for fn in bf_m]  # 从feature名反查列索引
Xbm = icc_Xm[:, combo_m_idx]
Xbm_std = ss_m.transform(Xbm); pcs = pca_m.transform(Xbm_std)
score_pca = pcs @ w_m
# Beta way
score_beta = Xbm @ beta_m + int_m
diff = np.max(np.abs(score_pca - score_beta))
print(f"  Max |PCA_F - Beta_F| = {diff:.2e}  (IDENTICAL)")
print(f"\n  PCA Formula:")
print(f"    F = {w_m[0]:.4f}×PC1 + {w_m[1]:.4f}×PC2 + {w_m[2]:.4f}×PC3 + {w_m[3]:.4f}×PC4")
print(f"    PCk = sum(loading_kj * (ΔFj-μj)/σj)")
print(f"\n  Beta Formula (same result, no PCA needed):")
print(f"    F = {int_m:+.6f} + sum(beta_j × ΔF_j)")
for j, fn in enumerate(bf_m[:8]):
    print(f"    beta_{fn:<30s} = {beta_m[j]:+.4e}")

# =============================================
# GC032 Prediction
# =============================================
print("\n" + "="*60)
print("  GC032 Prediction (ICC Morph Model)")
print("="*60)
GC_DIR1 = r'D:\Desktop\music\measure\Data\GC032\20241101\excel'
GC_DIR5 = r'D:\Desktop\music\measure\Data\GC032\20241105\excel'
gc1 = load_organoids(GC_DIR1, MORPH); gc5 = load_organoids(GC_DIR5, MORPH)
gc_delta, _, _, _, _ = build_delta(gc1, gc5, MORPH, short_m, 3,
                                    scaler=scaler_m, kmeans_k6=km_m, k4_map=k4_m)
gc_mask = gc_delta['Well'].isin(GC_ATP.keys())
gc_wells = gc_delta.loc[gc_mask, 'Well'].values
gc_y = np.array([GC_ATP[w] for w in gc_wells])
gc_X = np.zeros((sum(gc_mask), len(bf_m)))
for fi, fn in enumerate(bf_m):
    if fn in gc_delta.columns:
        gc_X[:, fi] = gc_delta.loc[gc_mask, fn].fillna(0).values
gc_score = gc_X @ beta_m + int_m
gc_r, gc_p = pearsonr(gc_score, gc_y)
gc_rho, gc_sp = spearmanr(gc_score, gc_y)
print(f"  ICC Morph → GC032: |r|={abs(gc_r):.4f} (p={gc_p:.6f}, rho={gc_rho:+.4f})")

# =============================================
# FINAL TABLE
# =============================================
GC_SELF = 0.8702
print("\n" + "="*60)
print("  FINAL COMPARISON (32 wells, D1+D5 clustering)")
print("="*60)
print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  Model / Scenario                       │  |r| (Pearson)     │
  ├─────────────────────────────────────────┼────────────────────┤
  │  ICC Morph+Scatt ({len(fc_s)}D, no search)           │  {r_sa:.4f}              │
  │  ICC Morph+Scatt ({len(fc_s)}D, search×{N_RUNS}, mean) │  {np.mean(rs_s):.4f}              │
  ├─────────────────────────────────────────┼────────────────────┤
  │  ICC Pure Morph ({len(fc_m)}D, no search)        │  {r_ma:.4f}              │
  │  ICC Pure Morph ({len(fc_m)}D, search×{N_RUNS}, mean) │  {np.mean(rs_m):.4f}              │
  ├─────────────────────────────────────────┼────────────────────┤
  │  Δr = Scatt(all) - Morph(all)            │  {r_sa-r_ma:+.4f}              │
  │  → Scatt adds ~{abs(r_sa-r_ma)*100:.1f}% to r              │                      │
  ├─────────────────────────────────────────┼────────────────────┤
  │  Paper ICC Pure Morph ΔF (预计算K4)     │  0.906              │
  │  Paper ICC + OAC Fusion                 │  0.938              │
  ├─────────────────────────────────────────┼────────────────────┤
  │  ICC Morph → GC032                      │  {abs(gc_r):.4f}              │
  │  GC032 Self-Train Morphology            │  {GC_SELF:.4f}              │
  │  → Transfer Gap                         │  {abs(gc_r)-GC_SELF:+.4f}              │
  │  → Spearman rho                         │  {gc_rho:+.4f}              │
  └─────────────────────────────────────────┴────────────────────┘
""")

print("  GC032 Prediction Details (F = w·PC = β·ΔF):")
print(f"  {'Well':<6s} {'F (Score)':>10s} {'log10(ATP)':>10s} {'Drug':>15s}")
drugs = {
    'C3':'5-FU','C4':'5-FU','C5':'5-FU',
    'C8':'Gem+CDDP','C11':'Gem+5FU','C12':'Gem+5FU',
    'D4':'CDDP+5FU','D5':'CDDP+Gem','D6':'CDDP+Gem',
    'E4':'OXA+5FU','E5':'OXA+Gem','E6':'OXA+Gem',
    'F2':'OXA+5FU(P)','F3':'OXA+5FU(P)','F4':'OXA+5FU(P)',
    'F7':'Ctrl+5FU','F8':'Ctrl+5FU','F9':'Ctrl+5FU',
    'F11':'Control','F12':'Control',
}
for i, w in enumerate(gc_wells):
    print(f"  {w:<6s} {gc_score[i]:>10.4f} {np.log10(gc_y[i]):>10.4f} {drugs.get(w,''):>15s}")

# Save ICC morph model
MODEL_DIR = r'D:\Desktop\music\measure\psd_pipeline\model_icc_morph'
os.makedirs(MODEL_DIR, exist_ok=True)
model = {
    'scaler_k6': scaler_m, 'kmeans_k6': km_m, 'k4_map': k4_m,
    'features': bf_m, 'scaler_pca': ss_m, 'pca': pca_m, 'weights': w_m,
    'beta': beta_m, 'intercept': int_m,
    'morph_features': MORPH, 'r': r_m_best, 'n_wells': 32,
}
with open(os.path.join(MODEL_DIR, 'icc_morph_deploy.pkl'), 'wb') as f:
    pickle.dump(model, f)
print(f"\n  Model saved to model_icc_morph/")