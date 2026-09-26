"""
ICC纯形态建模 + GC032跨数据集预测 + 三向对比
  Phase A: ICC纯形态模型训练 (K6→4 → Delta → 特征搜索 → PCA)
  Phase B: 用ICC模型预测GC032
  Phase C: 对比: ICC纯形态r vs 论文0.906, ICC→GC r vs GC自训练0.870
"""
import os, pickle, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

RANDOM_SEED = 42; N_SEARCH = 50000; N_PC = 4; MIN_ORG = 50

# =============================================
# 共享: 形态特征 (9个, 无Scatt)
# =============================================
MORPH = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
         'Cavity_Volume','CavityNum','LongAxis','ShortAxis','Wall_Thickness','Sphericity']
FEAT_SHORT = ['Vol','VolFill','Surf','CavVol','CavNum','LongAx','ShortAx','WallTh','Spher']
CAV_IDX = 3

# =============================================
# 共享: 数据处理工具函数
# =============================================
def load_organoids(excel_dir, well_filter=None):
    rows = []
    for fn in sorted(os.listdir(excel_dir)):
        if not fn.endswith('.xlsx') or fn.startswith('~$'): continue
        wid = fn.split('_')[0]
        if well_filter and wid not in well_filter: continue
        df = pd.read_excel(os.path.join(excel_dir, fn))
        sub = df[MORPH].copy()
        sub['Well'] = wid
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def k6_to_k4(X_std, k6_labels, centers):
    dist = cdist(centers, centers); np.fill_diagonal(dist, np.inf)
    pairs = []
    for _ in range(2):
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        pairs.append((int(i), int(j)))
        dist[i,:]=np.inf; dist[:,i]=np.inf; dist[j,:]=np.inf; dist[:,j]=np.inf
    k4map = {}; nid = 0
    for i in range(6):
        if i in k4map: continue
        for pi, pj in pairs:
            if i == pi or i == pj:
                k4map[pi]=nid; k4map[pj]=nid; nid+=1; break
        else: k4map[i]=nid; nid+=1
    return k4map, pairs

def build_delta(all_data, paired_wells, scaler_k6=None, kmeans_k6=None, k4_map=None):
    """构建 Delta 特征矩阵 (20 wells × features)"""
    d1_data = all_data[all_data['Src']=='D1'].copy()
    d5_data = all_data[all_data['Src']=='D5'].copy()

    X_d1 = d1_data[MORPH].fillna(0).values
    X_d5 = d5_data[MORPH].fillna(0).values

    if scaler_k6 is None:
        scaler_k6 = StandardScaler()
        X_d1_std = scaler_k6.fit_transform(X_d1)
        kmeans_k6 = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
        k6_d1 = kmeans_k6.fit_predict(X_d1_std)
        k4_map, pairs = k6_to_k4(X_d1_std, k6_d1, kmeans_k6.cluster_centers_)
        print(f"  [TRAIN] Clustering: K6 sizes={dict(zip(*np.unique(k6_d1,return_counts=1)))}")
        print(f"  [TRAIN] Merge pairs={pairs}, K4 map={k4_map}")
    else:
        X_d1_std = scaler_k6.transform(X_d1)
        k6_d1 = kmeans_k6.predict(X_d1_std)
        print(f"  [PREDICT] Using pre-trained clustering model")

    d1_data['K4'] = k6_d1
    d1_data['K4'] = d1_data['K4'].map(k4_map)

    X_d5_std = scaler_k6.transform(X_d5)
    k6_d5 = kmeans_k6.predict(X_d5_std)
    d5_data['K4'] = k6_d5
    d5_data['K4'] = d5_data['K4'].map(k4_map)

    # 聚合
    rows = []
    for w in paired_wells:
        row = {'Well': w}
        for src, sdf in [('D1', d1_data), ('D5', d5_data)]:
            wdf = sdf[sdf['Well'] == w]
            for k4 in range(4):
                kdf = wdf[wdf['K4'] == k4]
                cnt = len(kdf)
                row[f'Cnt_{src}_K{k4}'] = cnt
                for fi, feat in enumerate(MORPH):
                    if cnt == 0:
                        row[f'{feat}_{src}_K{k4}'] = 0.0
                    elif fi == CAV_IDX:
                        row[f'{feat}_{src}_K{k4}'] = float(kdf[feat].fillna(0).sum())
                    else:
                        row[f'{feat}_{src}_K{k4}'] = float(kdf[feat].fillna(0).mean())
        rows.append(row)
    df_agg = pd.DataFrame(rows)

    # Delta = D5 - D1
    df_delta = pd.DataFrame({'Well': df_agg['Well']})
    for k4 in range(4):
        for fi, feat in enumerate(MORPH):
            df_delta[f'{FEAT_SHORT[fi]}_K{k4}'] = df_agg[f'{feat}_D5_K{k4}'] - df_agg[f'{feat}_D1_K{k4}']
        df_delta[f'Cnt_K{k4}'] = df_agg[f'Cnt_D5_K{k4}'] - df_agg[f'Cnt_D1_K{k4}']

    feat_cols = [c for c in df_delta.columns if c != 'Well']
    feat_cols = [c for c in feat_cols if df_delta[c].abs().sum() > 1e-10]
    return df_delta, feat_cols, scaler_k6, kmeans_k6, k4_map

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
    r_fin, p_fin = pearsonr(score, y)
    return best_r, r_fin, p_fin, best_combo, best_feat, best_pca, best_ss, best_w, score


# =============================================
# PHASE A: ICC 纯形态建模
# =============================================
print("="*65)
print("  PHASE A: ICC Pure Morphology Model Training")
print("="*65)

ICC_DIR1 = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）\FXN_20230701\measure_excel'
ICC_DIR5 = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）\FXN_20230703\measure_excel'

ICC_ATP = {
    'B10':601300,'B11':11180000,'B2':5391000,'B3':6538000,'B4':7103000,
    'B5':1264000,'B6':2548000,'B7':1579000,'B8':3637000,'B9':140300,
    'C10':211800,'C11':13930000,'C2':4460000,'C3':8336000,'C4':6800000,
    'C5':330900,'C6':238900,'C7':682100,'C8':211300,'C9':465900,
    'D11':11240000,'E11':12840000,'F10':21910000,'F11':14700000,
    'F2':26980000,'F3':14110000,'F4':13740000,'F5':17250000,
    'F6':20320000,'F7':20000000,'F8':17170000,'F9':15830000,
}
icc_wells = set(ICC_ATP.keys())

df_icc_d1 = load_organoids(ICC_DIR1, icc_wells); df_icc_d1['Src']='D1'
df_icc_d5 = load_organoids(ICC_DIR5, icc_wells); df_icc_d5['Src']='D5'
df_icc_all = pd.concat([df_icc_d1, df_icc_d5], ignore_index=True)
icc_paired = sorted(set(df_icc_d1['Well'].unique()) & set(df_icc_d5['Well'].unique()))
print(f"  ICC organoids: D1={len(df_icc_d1)}, D5={len(df_icc_d5)}, paired={len(icc_paired)}")

df_icc_delta, icc_feat, icc_scaler_k6, icc_kmeans, icc_k4map = build_delta(df_icc_all, icc_paired)
icc_ATP_vals = np.array([ICC_ATP[w] for w in df_icc_delta['Well']])
icc_X = df_icc_delta[icc_feat].fillna(0).values
print(f"  ICC Delta features: {len(icc_feat)}")

icc_best_r, icc_r, icc_p, icc_combo, icc_best_feat, icc_pca, icc_ss, icc_w, icc_score = \
    search_and_pca(icc_X, icc_ATP_vals, icc_feat, 'ICC')

print(f"\n  >>> ICC Pure Morphology Result: r = {icc_r:.4f} (p={icc_p:.6f})")
print(f"  >>> Paper Pure Morphology ΔF:     r = 0.906")
print(f"  >>> Gap: {icc_r - 0.906:+.4f}")
print(f"  Selected features: {len(icc_best_feat)}/{len(icc_feat)}")

# 保存ICC模型
ICC_MODEL_DIR = r'D:\Desktop\music\measure\psd_pipeline\model_icc_morph'
os.makedirs(ICC_MODEL_DIR, exist_ok=True)
icc_model = {
    'scaler_k6': icc_scaler_k6, 'kmeans_k6': icc_kmeans, 'k4_map': icc_k4map,
    'feature_list': icc_best_feat, 'scaler_pca': icc_ss, 'pca': icc_pca,
    'weights': icc_w, 'morph_features': MORPH, 'r': icc_r
}
with open(os.path.join(ICC_MODEL_DIR, 'icc_morph_deploy.pkl'), 'wb') as f:
    pickle.dump(icc_model, f)
print(f"  Model saved to model_icc_morph/")

# =============================================
# PHASE B: ICC纯形态模型 → 预测 GC032
# =============================================
print("\n" + "="*65)
print("  PHASE B: ICC Model → Predict GC032")
print("="*65)

GC_DIR1 = r'D:\Desktop\music\measure\Data\GC032\20241101\excel'
GC_DIR5 = r'D:\Desktop\music\measure\Data\GC032\20241105\excel'
GC_ATP = {
    'C3':8638000,'C4':7800000,'C5':7260000,'C8':185300,'C11':24330,'C12':71930,
    'D4':66260,'D5':27630,'D6':25920,'E4':63070,'E5':15450,'E6':68250,
    'F2':14970000,'F3':14070000,'F4':10820000,'F7':54140000,'F8':50680000,
    'F9':65380000,'F11':55300000,'F12':46400000,
}
gc_wells = set(GC_ATP.keys())

df_gc_d1 = load_organoids(GC_DIR1, gc_wells); df_gc_d1['Src']='D1'
df_gc_d5 = load_organoids(GC_DIR5, gc_wells); df_gc_d5['Src']='D5'
df_gc_all = pd.concat([df_gc_d1, df_gc_d5], ignore_index=True)
gc_paired = sorted(set(df_gc_d1['Well'].unique()) & set(df_gc_d5['Well'].unique()))
print(f"  GC organoids: D1={len(df_gc_d1)}, D5={len(df_gc_d5)}, paired={len(gc_paired)}")

# 用ICC模型预测GC
df_gc_delta, gc_feat, _, _, _ = build_delta(
    df_gc_all, gc_paired, scaler_k6=icc_scaler_k6, kmeans_k6=icc_kmeans, k4_map=icc_k4map)

# 对齐特征: GC Delta → ICC feature list
gc_X_full = df_gc_delta[gc_feat].fillna(0).values
# 找到ICC特征在GC特征中的索引
icc_feat_in_gc = {}
for fi, fn in enumerate(icc_best_feat):
    if fn in gc_feat:
        icc_feat_in_gc[fi] = gc_feat.index(fn)

if len(icc_feat_in_gc) < len(icc_best_feat):
    missing_icc = [fn for fn in icc_best_feat if fn not in gc_feat]
    print(f"  WARNING: {len(missing_icc)} ICC features missing in GC: {missing_icc}")
    # 填充缺失特征为0
    gc_X_aligned = np.zeros((len(df_gc_delta), len(icc_best_feat)))
    for fi, fn in enumerate(icc_best_feat):
        if fn in gc_feat:
            gc_X_aligned[:, fi] = df_gc_delta[fn].fillna(0).values
else:
    gc_X_aligned = np.zeros((len(df_gc_delta), len(icc_best_feat)))
    for fi, fn in enumerate(icc_best_feat):
        gc_X_aligned[:, fi] = df_gc_delta[fn].fillna(0).values

# PCA预测
gc_X_std = icc_ss.transform(gc_X_aligned)
gc_pc = icc_pca.transform(gc_X_std)
gc_score = gc_pc @ icc_w

gc_y = np.array([GC_ATP[w] for w in df_gc_delta['Well']])
gc_r, gc_pv = pearsonr(gc_score, gc_y)
gc_rho, _ = spearmanr(gc_score, gc_y)
print(f"\n  >>> ICC Morph Model → GC032 Prediction: r={gc_r:.4f} (p={gc_pv:.6f}, rho={gc_rho:.4f})")

# =============================================
# PHASE C: 三向对比总结
# =============================================
print("\n" + "="*65)
print("  PHASE C: Three-way Comparison")
print("="*65)
print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │              Pure Morphology Model Comparison           │
  ├──────────────────────────┬──────────────────────────────┤
  │  Model / Scenario        │  r (Pearson)                 │
  ├──────────────────────────┼──────────────────────────────┤
  │  Paper: ICC纯形态ΔF       │  0.906                       │
  │  Ours: ICC纯形态ΔF        │  {icc_r:.4f}                       │
  │  → Gap vs Paper          │  {icc_r-0.906:+.4f}                       │
  ├──────────────────────────┼──────────────────────────────┤
  │  GC032 自训练 (5药)       │  0.8702                      │
  │  ICC模型 → GC032 预测    │  {gc_r:.4f}                       │
  │  → Transfer Gap          │  {gc_r-0.8702:+.4f}                       │
  ├──────────────────────────┼──────────────────────────────┤
  │  Paper: 含OAC ΔF         │  0.938                       │
  │  Ours: 含Scatt ΔF        │  0.9040                      │
  │  → Scatt贡献 (vs 纯形态) │  {0.9040-icc_r:+.4f}                       │
  └──────────────────────────┴──────────────────────────────┘
""")

# 逐孔详情
print("  GC032 Prediction Details:")
print(f"  {'Well':<6s} {'Score':>10s} {'ATP':>12s} {'log10(ATP)':>12s}")
for i, w in enumerate(df_gc_delta['Well']):
    print(f"  {w:<6s} {gc_score[i]:>10.4f} {gc_y[i]:>12.0f} {np.log10(gc_y[i]):>12.4f}")