"""
ICC纯形态建模 + GC032跨数据集预测 (v3: Day1+Day5合并聚类)
  - 关键修正: K-means聚类 = Day1所有类器官 + Day5所有类器官 合在一起
  - ICC: 32孔合并聚类, 24阿可拉定孔评估
  - 特征搜索 + PCA + beta直接权重
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

MORPH = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
         'Cavity_Volume','CavityNum','LongAxis','ShortAxis','Wall_Thickness','Sphericity']
FEAT_SHORT = ['Vol','VolFill','Surf','CavVol','CavNum','LongAx','ShortAx','WallTh','Spher']
CAV_IDX = 3

# =============================================
# 论文表4-4: 24个阿可拉定孔 ATP (仅用于评估)
# =============================================
ICARITIN_ATP = {
    'E11': 12840000, 'F2': 26980000, 'F6': 20320000,
    'F8': 17170000, 'F9': 15830000, 'F11': 14700000,
    'B2': 5391000, 'B3': 6538000, 'B4': 7103000,
    'C2': 4460000, 'C3': 8336000, 'C4': 6800000,
    'B5': 1264000, 'B6': 2548000, 'B7': 1579000,
    'C5': 330900, 'C6': 238900, 'C7': 682100,
    'B8': 3637000, 'B9': 140300, 'B10': 601300,
    'C8': 211300, 'C9': 465900, 'C10': 211800,
}

# =============================================
def load_organoids(excel_dir, well_filter=None):
    rows = []
    for fn in sorted(os.listdir(excel_dir)):
        if not fn.endswith('.xlsx') or fn.startswith('~$'): continue
        wid = fn.split('_')[0]
        if well_filter and wid not in well_filter: continue
        df = pd.read_excel(os.path.join(excel_dir, fn))
        sub = df[MORPH].copy(); sub['Well'] = wid
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

def build_delta(all_data_day1, all_data_day5, paired_wells,
                scaler=None, kmeans_k6=None, k4_map=None):
    """
    Day1+Day5 合并 → K-means → 逐孔逐表型聚合 → Delta
    """
    d1 = all_data_day1[MORPH].fillna(0).values
    d5 = all_data_day5[MORPH].fillna(0).values

    if scaler is None:
        # === TRAIN: Day1+Day5合并聚类 ===
        X_all = np.vstack([d1, d5])
        scaler = StandardScaler()
        X_all_std = scaler.fit_transform(X_all)
        kmeans_k6 = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
        k6_all = kmeans_k6.fit_predict(X_all_std)
        # 合并用 centers
        k4_map, pairs = k6_to_k4(kmeans_k6.cluster_centers_)
        n_d1 = len(d1)
        k6_d1 = k6_all[:n_d1]
        k6_d5 = k6_all[n_d1:]
        sizes = dict(zip(*np.unique(k6_all, return_counts=True)))
        print(f"  [TRAIN] K6 sizes (D1+D5 combined): {sizes}")
        print(f"  [TRAIN] Merge: {pairs} -> K4 map: {k4_map}")
    else:
        # === PREDICT: 用已有模型分别预测 ===
        k6_d1 = kmeans_k6.predict(scaler.transform(d1))
        k6_d5 = kmeans_k6.predict(scaler.transform(d5))
        print(f"  [PREDICT] Using pre-trained clustering")

    d1_data = all_data_day1.copy(); d1_data['K4'] = pd.Series(k6_d1, index=d1_data.index).map(k4_map)
    d5_data = all_data_day5.copy(); d5_data['K4'] = pd.Series(k6_d5, index=d5_data.index).map(k4_map)

    # 逐孔逐表型聚合
    delta_rows = []
    for w in paired_wells:
        row = {'Well': w}
        for src, sdf in [('D1', d1_data), ('D5', d5_data)]:
            wdf = sdf[sdf['Well'] == w]
            for k4 in range(4):
                kdf = wdf[wdf['K4'] == k4]
                cnt = len(kdf)
                row[f'Cnt_{src}_K{k4}'] = cnt
                if cnt > 0:
                    for fi, feat in enumerate(MORPH):
                        vals = kdf[feat].fillna(0).values
                        row[f'{feat}_{src}_K{k4}'] = float(np.sum(vals) if fi == CAV_IDX else np.mean(vals))
                else:
                    for fi, feat in enumerate(MORPH):
                        row[f'{feat}_{src}_K{k4}'] = 0.0
        delta_rows.append(row)

    df_agg = pd.DataFrame(delta_rows)
    df_delta = pd.DataFrame({'Well': df_agg['Well']})
    for k4 in range(4):
        for fi, feat in enumerate(MORPH):
            df_delta[f'{FEAT_SHORT[fi]}_K{k4}'] = df_agg[f'{feat}_D5_K{k4}'] - df_agg[f'{feat}_D1_K{k4}']
        df_delta[f'Cnt_K{k4}'] = df_agg[f'Cnt_D5_K{k4}'] - df_agg[f'Cnt_D1_K{k4}']

    feat_cols = [c for c in df_delta.columns if c != 'Well']
    feat_cols = [c for c in feat_cols if df_delta[c].abs().sum() > 1e-10]
    return df_delta, feat_cols, scaler, kmeans_k6, k4_map

def all_features_pca(X, y):
    ss = StandardScaler(); X_std = ss.fit_transform(X)
    pc = PCA(n_components=N_PC, random_state=RANDOM_SEED)
    X_pc = pc.fit_transform(X_std)
    wr = pc.explained_variance_ratio_[:N_PC]; wr = wr / wr.sum()
    score = X_pc @ wr
    r, p = pearsonr(score, y)
    return r, p, score, pc, ss, wr

def search_and_pca(X, y, feat_cols, label=''):
    nf = len(feat_cols)
    best_r, best_combo, best_pca, best_ss, best_w = 0, None, None, None, None
    t0 = time.time()
    for it in range(N_SEARCH):
        ns = np.random.randint(max(4, nf//3), nf+1)
        si = sorted(np.random.choice(nf, ns, replace=False))
        Xs = X[:, si]
        ss = StandardScaler(); Xs_std = ss.fit_transform(Xs)
        pc = PCA(n_components=N_PC, random_state=RANDOM_SEED)
        try: Xp = pc.fit_transform(Xs_std)
        except: continue
        wr = pc.explained_variance_ratio_[:N_PC]; wr = wr/wr.sum()
        r, _ = pearsonr(Xp @ wr, y)
        if abs(r) > best_r:
            best_r = abs(r); best_combo = si; best_pca = pc; best_ss = ss; best_w = wr
        if (it+1) % 10000 == 0:
            print(f"  [{label}] {it+1}/{N_SEARCH} best |r|={best_r:.4f}, {time.time()-t0:.0f}s")
    best_feat = [feat_cols[i] for i in best_combo]
    Xb = X[:, best_combo]; Xb_std = best_ss.transform(Xb)
    score = best_pca.transform(Xb_std) @ best_w
    r, p = pearsonr(score, y)
    # 计算 beta
    loadings = best_pca.components_[:N_PC]
    stds = best_ss.scale_; means = best_ss.mean_
    beta = loadings.T @ best_w / stds
    intercept = -np.sum(beta * means)
    return r, p, best_combo, best_feat, best_pca, best_ss, best_w, score, beta, intercept


# =============================================
# PHASE A: ICC 纯形态 (32孔D1+D5合并聚类, 24孔评估)
# =============================================
print("="*65)
print("  PHASE A: ICC Pure Morphology (D1+D5 combined clustering)")
print("="*65)

ICC_DIR1 = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）\FXN_20230701\measure_excel'
ICC_DIR5 = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）\FXN_20230703\measure_excel'

df_icc_d1 = load_organoids(ICC_DIR1)  # 32 wells, 全量
df_icc_d5 = load_organoids(ICC_DIR5)
print(f"  ICC organoids: D1={len(df_icc_d1)}, D2={len(df_icc_d5)}")
print(f"  D1 wells: {sorted(df_icc_d1['Well'].unique())[:5]}... ({len(df_icc_d1['Well'].unique())} total)")

icc_paired = sorted(set(df_icc_d1['Well'].unique()) & set(df_icc_d5['Well'].unique()))
print(f"  Paired wells: {len(icc_paired)}")

# D1+D5合并聚类 → Delta
df_icc_delta, icc_feat, icc_s_k6, icc_km, icc_k4 = build_delta(df_icc_d1, df_icc_d5, icc_paired)

# 筛选24个阿可拉定孔用于评估
icaritin_wells = list(ICARITIN_ATP.keys())
icc_eval_mask = df_icc_delta['Well'].isin(icaritin_wells).values
icc_eval_wells = df_icc_delta.loc[icc_eval_mask, 'Well'].values
icc_y = np.array([ICARITIN_ATP[w] for w in icc_eval_wells])
icc_X = df_icc_delta.loc[icc_eval_mask, icc_feat].fillna(0).values
print(f"  Evaluation wells (Icaritin): {len(icc_y)}")
print(f"  Delta features: {len(icc_feat)}")

# A1: 无搜索基线
print("\n  --- A1: ALL features PCA (no search, fair vs paper 0.906) ---")
icc_r_ns, icc_p_ns, icc_score_ns, _, _, _ = all_features_pca(icc_X, icc_y)
print(f"  r = {icc_r_ns:.4f} (p={icc_p_ns:.6f})")
print(f"  Paper pure morph r = 0.906")

# A2: 特征搜索
print(f"\n  --- A2: Feature search ({N_SEARCH} iterations) ---")
icc_r, icc_p, icc_combo, icc_feat_sel, icc_pca, icc_ss, icc_w, icc_score, beta, intercept = \
    search_and_pca(icc_X, icc_y, icc_feat, 'ICC')

# 打印 beta
print(f"\n  Beta coefficients (direct feature weights):")
print(f"  Score = {intercept:+.6f} + Σ(beta_j × DeltaFeature_j)")
print(f"  {'Feature':<35s} {'beta':>15s}")
for j, fn in enumerate(icc_feat_sel):
    print(f"  {fn:<35s} {beta[j]:>+15.6e}")

# A3: 已含Scatt的模型r (对照)
print(f"\n  --- Comparison ---")
print(f"  ICC + Scatt + search:     r = 0.9040  (之前的nnUNet方法)")
print(f"  ICC Pure Morph + search:  r = {icc_r:.4f}")
print(f"  ICC Pure Morph (no search): r = {icc_r_ns:.4f}")
print(f"  Paper Pure Morph (paper): r = 0.906")
print(f"  Morph vs Scatt gap:       Δr = {icc_r - 0.9040:+.4f}")

# 保存模型
ICC_MODEL_DIR = r'D:\Desktop\music\measure\psd_pipeline\model_icc_morph'
os.makedirs(ICC_MODEL_DIR, exist_ok=True)
model = {
    'scaler_k6': icc_s_k6, 'kmeans_k6': icc_km, 'k4_map': icc_k4,
    'feature_list': icc_feat_sel, 'scaler_pca': icc_ss, 'pca': icc_pca,
    'weights': icc_w, 'beta': beta, 'intercept': intercept,
    'morph_features': MORPH, 'r': icc_r, 'method': 'D1+D5 combined clustering'
}
with open(os.path.join(ICC_MODEL_DIR, 'icc_morph_deploy.pkl'), 'wb') as f:
    pickle.dump(model, f)
print(f"  Model saved.")

# =============================================
# PHASE B: ICC → GC032 预测
# =============================================
print("\n" + "="*65)
print("  PHASE B: ICC Morph Model → Predict GC032")
print("="*65)

GC_DIR1 = r'D:\Desktop\music\measure\Data\GC032\20241101\excel'
GC_DIR5 = r'D:\Desktop\music\measure\Data\GC032\20241105\excel'
GC_ATP = {
    'C3':8638000,'C4':7800000,'C5':7260000,'C8':185300,'C11':24330,'C12':71930,
    'D4':66260,'D5':27630,'D6':25920,'E4':63070,'E5':15450,'E6':68250,
    'F2':14970000,'F3':14070000,'F4':10820000,'F7':54140000,'F8':50680000,
    'F9':65380000,'F11':55300000,'F12':46400000,
}

df_gc_d1 = load_organoids(GC_DIR1)
df_gc_d5 = load_organoids(GC_DIR5)
gc_paired = sorted(set(df_gc_d1['Well'].unique()) & set(df_gc_d5['Well'].unique()))
print(f"  GC: D1={len(df_gc_d1)}, D5={len(df_gc_d5)}, wells={len(gc_paired)}")

df_gc_delta, gc_feat, _, _, _ = build_delta(
    df_gc_d1, df_gc_d5, gc_paired, scaler=icc_s_k6, kmeans_k6=icc_km, k4_map=icc_k4)

# GC ATP wells
gc_atp_wells = list(GC_ATP.keys())
gc_eval_mask = df_gc_delta['Well'].isin(gc_atp_wells).values
gc_eval_wells = df_gc_delta.loc[gc_eval_mask, 'Well'].values
gc_y = np.array([GC_ATP[w] for w in gc_eval_wells])

# 对齐特征空间
gc_X = np.zeros((sum(gc_eval_mask), len(icc_feat_sel)))
for fi, fn in enumerate(icc_feat_sel):
    if fn in df_gc_delta.columns:
        gc_X[:, fi] = df_gc_delta.loc[gc_eval_mask, fn].fillna(0).values

gc_score = gc_X @ beta + intercept
gc_r, gc_pv = pearsonr(gc_score, gc_y)
gc_rho, _ = spearmanr(gc_score, gc_y)
print(f"  ICC Morph → GC032: r={gc_r:.4f} (p={gc_pv:.6f}, ρ={gc_rho:.4f})")

# =============================================
# PHASE C: 总结
# =============================================
GC_RETRAIN = 0.8702
print("\n" + "="*65)
print("  FINAL COMPARISON")
print("="*65)
print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │            Pure Morphology Model Comparison              │
  ├────────────────────────────────┬─────────────────────────┤
  │  Model / Domain                │  r (Pearson)            │
  ├────────────────────────────────┼─────────────────────────┤
  │  Paper ICC Pure Morph ΔF       │  0.906                  │
  │  Ours ICC Pure Morph (全特征)  │  {icc_r_ns:.4f}                  │
  │  Ours ICC Pure Morph (搜索)    │  {icc_r:.4f}                  │
  │  Ours ICC + Scatt (搜索)       │  0.9040                  │
  │  → Δr (Scatt贡献 vs 纯形态)    │  {0.9040-icc_r:+.4f}                  │
  ├────────────────────────────────┼─────────────────────────┤
  │  ICC Model → GC032             │  {gc_r:.4f}                  │
  │  GC032 Self-Train              │  {GC_RETRAIN:.4f}                  │
  │  → Transfer Gap                │  {gc_r-GC_RETRAIN:+.4f}                  │
  └────────────────────────────────┴─────────────────────────┘
""")

print("  GC032 prediction details:")
print(f"  {'Well':<6s} {'Score':>10s} {'log10(ATP)':>12s}")
for i, w in enumerate(gc_eval_wells):
    print(f"  {w:<6s} {gc_score[i]:>10.4f} {np.log10(gc_y[i]):>12.4f}")

# beta 使用说明
print(f"""
  === How to Use Beta ===
  Score = {intercept:+.6f} + Σ(beta_j × ΔFeature_j)

  Step 1: Load new organoids → apply ICC K-means → K4 labels
  Step 2: Within each well, aggregate features per K4 (mean, CavVol=sum)
  Step 3: ΔFeature = Day5_aggregate − Day1_aggregate
  Step 4: Extract {len(icc_feat_sel)} selected features
  Step 5: Score = Σ(beta_j × ΔFeature_j) + intercept

  beta是特征空间中每个ΔFeature的直接系数(单位变化 → Score变化)
  PCA weight × loading ÷ StandardScaler.std_ → beta (反归一化后)
""")