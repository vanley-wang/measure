"""
ICC Pure Morphology + Scatt comparison (D1+D5 combined clustering)
纯形态 vs 含Scatt, 特征搜索 vs 全特征, ICC→GC转移
"""
import os, pickle, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

RANDOM_SEED = 42; N_SEARCH = 50000; N_PC = 4

# =============================================
# 特征定义
# =============================================
MORPH = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
         'Cavity_Volume','CavityNum','LongAxis','ShortAxis','Wall_Thickness','Sphericity']
SCATT = ['Scatt_Mean', 'Scatt_STD']
ALL_FEATS = MORPH + SCATT  # 11 features

FEAT_SHORT_M = ['Vol','VolFill','Surf','CavVol','CavNum','LongAx','ShortAx','WallTh','Spher']
FEAT_SHORT_S = ['ScattM','ScattS']
FEAT_SHORT_ALL = FEAT_SHORT_M + FEAT_SHORT_S
CAV_IDX = 3
SCATT_IDX = [9, 10]

# ATP
ICARITIN_ATP = {
    'E11':12840000,'F2':26980000,'F6':20320000,'F8':17170000,'F9':15830000,'F11':14700000,
    'B2':5391000,'B3':6538000,'B4':7103000,'C2':4460000,'C3':8336000,'C4':6800000,
    'B5':1264000,'B6':2548000,'B7':1579000,'C5':330900,'C6':238900,'C7':682100,
    'B8':3637000,'B9':140300,'B10':601300,'C8':211300,'C9':465900,'C10':211800,
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

def build_pipeline(all_d1, all_d5, paired, feat_list, short_names,
                   cav_idx, scaler=None, kmeans_k6=None, k4_map=None):
    """
    Day1+Day5合并聚类 → K4聚合 → Delta特征
    """
    X_d1 = all_d1[feat_list].fillna(0).values
    X_d5 = all_d5[feat_list].fillna(0).values

    if scaler is None:
        X_all = np.vstack([X_d1, X_d5])
        scaler = StandardScaler(); X_std = scaler.fit_transform(X_all)
        kmeans_k6 = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
        k6_all = kmeans_k6.fit_predict(X_std)
        k4_map, pairs = k6_to_k4(kmeans_k6.cluster_centers_)
        n = len(X_d1); k6_d1 = k6_all[:n]; k6_d5 = k6_all[n:]
    else:
        k6_d1 = kmeans_k6.predict(scaler.transform(X_d1))
        k6_d5 = kmeans_k6.predict(scaler.transform(X_d5))

    d1 = all_d1.copy(); d1['K4'] = pd.Series(k6_d1, index=d1.index).map(k4_map)
    d5 = all_d5.copy(); d5['K4'] = pd.Series(k6_d5, index=d5.index).map(k4_map)

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

def pca_all_features(X, y):
    ss = StandardScaler(); Xs = ss.fit_transform(X)
    pc = PCA(n_components=N_PC, random_state=RANDOM_SEED)
    Xp = pc.fit_transform(Xs)
    wr = pc.explained_variance_ratio_[:N_PC]; wr = wr / wr.sum()
    r, p = pearsonr(Xp @ wr, y)
    return r, p

def search_pca(X, y, feat_cols):
    nf = len(feat_cols); best_r = 0.0; best = None
    t0 = time.time()
    for it in range(N_SEARCH):
        ns = np.random.randint(max(4, nf//3), nf+1)
        si = sorted(np.random.choice(nf, ns, replace=False))
        Xs = X[:, si]; ss = StandardScaler(); Xss = ss.fit_transform(Xs)
        pc = PCA(n_components=N_PC, random_state=RANDOM_SEED)
        try: Xp = pc.fit_transform(Xss)
        except: continue
        wr = pc.explained_variance_ratio_[:N_PC]; wr = wr / wr.sum()
        r, _ = pearsonr(Xp @ wr, y)
        if abs(r) > best_r:
            best_r = abs(r); best_combo = si; best_pca = pc; best_ss = ss; best_w = wr
        if (it+1) % 10000 == 0:
            print(f"    {it+1}/{N_SEARCH} best|r|={best_r:.4f} ({time.time()-t0:.0f}s)")
    bf = [feat_cols[i] for i in best_combo]; Xb = X[:, best_combo]
    Xbs = best_ss.transform(Xb); score = best_pca.transform(Xbs) @ best_w
    r, p = pearsonr(score, y)
    # beta
    ld = best_pca.components_[:N_PC]; st = best_ss.scale_; mn = best_ss.mean_
    beta = ld.T @ best_w / st; intercept = -np.sum(beta * mn)
    return r, p, best_combo, bf, best_pca, best_ss, best_w, score, beta, intercept

def apply_model(df_delta, feat_cols, best_feat, beta, intercept, y_dict):
    mask = df_delta['Well'].isin(y_dict.keys()).values
    wells = df_delta.loc[mask, 'Well'].values
    X = np.zeros((sum(mask), len(best_feat)))
    for fi, fn in enumerate(best_feat):
        if fn in df_delta.columns:
            X[:, fi] = df_delta.loc[mask, fn].fillna(0).values
    score = X @ beta + intercept
    y = np.array([y_dict[w] for w in wells])
    r, p = pearsonr(score, y)
    rho, _ = spearmanr(score, y)
    return abs(r), p, rho, score, y, wells


# =============================================
# MAIN
# =============================================
ICC_DIR1 = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）\FXN_20230701\measure_excel'
ICC_DIR5 = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）\FXN_20230703\measure_excel'

# === Model 1: ICC Morph + Scatt (11 feats, search) ===
print("="*60)
print("  MODEL 1: ICC Morph + Scatt (feature search)")
print("="*60)
df1 = load_organoids(ICC_DIR1, ALL_FEATS); df5 = load_organoids(ICC_DIR5, ALL_FEATS)
paired = sorted(set(df1['Well'].unique()) & set(df5['Well'].unique()))
df_delta, fc, scaler_s, km_s, k4_s = build_pipeline(df1, df5, paired, ALL_FEATS, FEAT_SHORT_ALL, CAV_IDX)
icc_mask = df_delta['Well'].isin(ICARITIN_ATP.keys()).values
icc_y = np.array([ICARITIN_ATP[w] for w in df_delta.loc[icc_mask,'Well']])
icc_X = df_delta.loc[icc_mask, fc].fillna(0).values
# all-features baseline
r_scatt_all, _ = pca_all_features(icc_X, icc_y)
print(f"  Scatt all-features: r={abs(r_scatt_all):.4f}")
# search
r_s, p_s, combo_s, bf_s, pca_s, ss_s, w_s, score_s, beta_s, int_s = search_pca(icc_X, icc_y, fc)
print(f"  Scatt + search: |r|={abs(r_s):.4f} (p={p_s:.6f})")

# === Model 2: ICC Pure Morph (9 feats, search) ===
print("\n" + "="*60)
print("  MODEL 2: ICC Pure Morphology (feature search)")
print("="*60)
df1m = load_organoids(ICC_DIR1, MORPH); df5m = load_organoids(ICC_DIR5, MORPH)
df_delta_m, fc_m, scaler_m, km_m, k4_m = build_pipeline(df1m, df5m, paired, MORPH, FEAT_SHORT_M, CAV_IDX)
icc_Xm = df_delta_m.loc[icc_mask, fc_m].fillna(0).values
# all-features baseline
r_morph_all, _ = pca_all_features(icc_Xm, icc_y)
print(f"  Morph all-features: r={abs(r_morph_all):.4f}")
# search
r_m, p_m, combo_m, bf_m, pca_m, ss_m, w_m, score_m, beta_m, int_m = search_pca(icc_Xm, icc_y, fc_m)
print(f"  Morph + search: |r|={abs(r_m):.4f} (p={p_m:.6f})")

# === Model 3: ICC Pure Morph → GC032 ===
print("\n" + "="*60)
print("  MODEL 3: ICC Morph Model → Predict GC032")
print("="*60)
GC_DIR1 = r'D:\Desktop\music\measure\Data\GC032\20241101\excel'
GC_DIR5 = r'D:\Desktop\music\measure\Data\GC032\20241105\excel'
gc1 = load_organoids(GC_DIR1, MORPH); gc5 = load_organoids(GC_DIR5, MORPH)
gc_paired = sorted(set(gc1['Well'].unique()) & set(gc5['Well'].unique()))
gc_delta, _, _, _, _ = build_pipeline(gc1, gc5, gc_paired, MORPH, FEAT_SHORT_M, CAV_IDX,
                                       scaler=scaler_m, kmeans_k6=km_m, k4_map=k4_m)
r_gc, p_gc, rho_gc, score_gc, y_gc, w_gc = apply_model(gc_delta, fc_m, bf_m, beta_m, int_m, GC_ATP)
print(f"  ICC Morph → GC032: |r|={r_gc:.4f} (p={p_gc:.6f}, rho={rho_gc:.4f})")

# === Model 4: GC Self-Train Morph (already computed: 0.8702) ===
GC_SELF = 0.8702

# =============================================
# FINAL TABLE
# =============================================
print("\n" + "="*60)
print("  FINAL COMPARISON TABLE")
print("="*60)
print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │             ICC Pure Morphology Model Analysis              │
  ├──────────────────────────────────────┬──────────────────────┤
  │  Model / Scenario                    │  |r| (Pearson)       │
  ├──────────────────────────────────────┼──────────────────────┤
  │  ICC + Scatt (all 44 feats, no sel)  │  {abs(r_scatt_all):.4f}                │
  │  ICC + Scatt (feature search)        │  {abs(r_s):.4f}                │
  ├──────────────────────────────────────┼──────────────────────┤
  │  ICC Pure Morph (all 40 feats, no sel)│  {abs(r_morph_all):.4f}                │
  │  ICC Pure Morph (feature search)     │  {abs(r_m):.4f}                │
  ├──────────────────────────────────────┼──────────────────────┤
  │  Δr = Scatt(all) - Morph(all)        │  {abs(r_scatt_all)-abs(r_morph_all):+.4f}                │
  │  → Scatt contribution                │  ~{abs(abs(r_scatt_all)-abs(r_morph_all)):.4f}              │
  ├──────────────────────────────────────┼──────────────────────┤
  │  Paper ICC Pure Morph (K4 pre-comp)  │  0.9060               │
  │  Paper ICC + OAC Fusion              │  0.9380               │
  ├──────────────────────────────────────┼──────────────────────┤
  │  ICC Morph → GC032 (cross-dataset)   │  {r_gc:.4f}                │
  │  GC032 Self-Train Morphology         │  {GC_SELF:.4f}                │
  │  → Transfer Gap                      │  {r_gc-GC_SELF:+.4f}                │
  │  → Spearman rho (transfer)           │  {rho_gc:.4f}                │
  └──────────────────────────────────────┴──────────────────────┘
""")

# =============================================
# 回答: beta 是什么
# =============================================
print("="*60)
print("  Beta Explanation")
print("="*60)
print(f"""
  Beta是"直接特征权重"——对选中的{len(bf_m)}个Delta特征, Score = intercept + Σ(beta_j × ΔFeature_j)

  Beta的推导:
    PCA Score = w₁×PC1 + w₂×PC2 + w₃×PC3 + w₄×PC4
    PCk = Σ(loading_kj × (ΔFeature_j - mean_j) / std_j)
    → Score = Σ(w_k × Σ(loading_kj × (ΔF_j - mean_j)/std_j))
            = Σ[Σ(w_k × loading_kj / std_j)] × ΔF_j - Σ[Σ(w_k × loading_kj × mean_j/std_j)]
            = Σ beta_j × ΔF_j + intercept

  beta_j = Σ(w_k × loading_kj) / std_j         (对每个选中特征j)
  intercept = -Σ(beta_j × mean_j)

  对新数据预测: 直接计算 DeltaFeatures → × beta → + intercept → Score
  不需要重新跑PCA, 不需要StandardScaler, 纯线性加权即可

  选中特征 (ICC Pure Morph):
""")
for j, fn in enumerate(bf_m):
    print(f"    {fn:<35s} beta = {beta_m[j]:>+15.6e}")
print(f"    {'intercept':<35s}      = {int_m:>+15.6e}")

print(f"\n  GC032 Prediction Details:")
print(f"  {'Well':<6s} {'Score':>10s} {'log10(ATP)':>10s} {'ATP':>12s}")
for i, w in enumerate(w_gc):
    print(f"  {w:<6s} {score_gc[i]:>10.4f} {np.log10(y_gc[i]):>10.4f} {y_gc[i]:>12.0f}")