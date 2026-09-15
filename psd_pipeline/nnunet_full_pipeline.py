# ============================================================
# nnUNet ICC 完整 Pipeline
# Step 1: K=6 聚类 → 合并为 4 表型
# Step 2: 逐孔逐表型均值聚合
# Step 3: Delta = D5 − D3 → 特征穷举搜索 → PCA → ATP
# ============================================================
import os, sys, pickle, time, warnings, random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NNUNET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'nnUNet_FXN')
D3_DIR = os.path.join(NNUNET_DIR, 'FXN_0701', 'measure_excel')
D5_DIR = os.path.join(NNUNET_DIR, 'FXN_0703', 'measure_excel')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_nnunet_new')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_nnunet_new')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CF = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
      'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
      'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']
CF_SHORT = ['Vol','FillVol','Surf','CavVol','CavNum',
            'LongAx','ShortAx','WallTh','Spher','ScattM','ScattS']
N_MERGED = 4

ATP_DB = {
    'B10': 601300, 'B11': 11180000, 'B2': 5391000, 'B3': 6538000,
    'B4': 7103000, 'B5': 511900, 'B6': 404500, 'B7': 403900,
    'B8': 312700, 'B9': 140300, 'C10': 211800, 'C11': 13930000,
    'C2': 6336000, 'C3': 8336000, 'C4': 6800000, 'C5': 330900,
    'C6': 238900, 'C7': 682100, 'C8': 211300, 'C9': 465900,
    'D11': 11240000, 'E11': 14700000, 'F10': 21910000, 'F11': 11180000,
    'F2': 18240000, 'F3': 14110000, 'F4': 13740000, 'F5': 17250000,
    'F6': 20320000, 'F7': 20000000, 'F8': 17170000, 'F9': 15830000,
}

N_SEARCH = 50000
N_JOBS = -1
N_PC = 4
RANDOM_SEED = 42

# ================================================================
# STEP 1: K=6 聚类 + 自动合并为 4 表型
# ================================================================
print('='*60)
print('  STEP 1: K=6 Clustering + Merge → 4 Phenotypes')
print('='*60)

dfs_d3, dfs_d5 = [], []
for f in sorted(os.listdir(D3_DIR)):
    if not f.endswith('.xlsx'): continue
    df = pd.read_excel(os.path.join(D3_DIR, f))
    dfs_d3.append(df[CF].fillna(0))
for f in sorted(os.listdir(D5_DIR)):
    if not f.endswith('.xlsx'): continue
    df = pd.read_excel(os.path.join(D5_DIR, f))
    dfs_d5.append(df[CF].fillna(0))
X3_only = pd.concat(dfs_d3, ignore_index=True).values
X5_only = pd.concat(dfs_d5, ignore_index=True).values
print(f'  Day3 organoids: {len(X3_only)}, Day5 organoids: {len(X5_only)}')
X_all = np.vstack([X3_only, X5_only])
print(f'  Total D3+D5 organoids: {len(X_all)}')

scaler_k6 = StandardScaler()
X_std = scaler_k6.fit_transform(X_all)
kmeans = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
k6_labels = kmeans.fit_predict(X_std)

print(f'  K=6 cluster sizes:')
for i in range(6):
    print(f'    C{i}: {(k6_labels==i).sum():>6,} ({(k6_labels==i).sum()/len(k6_labels)*100:>5.1f}%)')

# 自动合并：找最近的两个对
centers = kmeans.cluster_centers_
dist = cdist(centers, centers)
np.fill_diagonal(dist, np.inf)

# 找最接近的两对
pairs = []
for _ in range(2):
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    pairs.append((i, j))
    dist[i, :] = np.inf; dist[:, i] = np.inf
    dist[j, :] = np.inf; dist[:, j] = np.inf

print(f'\n  Merge pairs (closest centers): {pairs}')

# 构建合并映射
merge_map = {}
remaining = set(range(6))
for a, b in pairs:
    merge_map[a] = a  # 暂存
    merge_map[b] = a
    remaining -= {a, b}

# 剩余两个单独保留
for r in sorted(remaining):
    merge_map[r] = r

# 重新编号到 0-3，按空腔体积排序
temp_labels = np.array([merge_map[l] for l in k6_labels])
unique_temps = sorted(set(merge_map.values()))
cavity_means = {}
for u in unique_temps:
    mask = temp_labels == u
    cavity_means[u] = np.mean(X_all[mask, CF.index('Cavity_Volume')])

ordered = sorted(unique_temps, key=lambda u: cavity_means[u])  # 空腔小→大
numeric_map = {k6_label: ordered.index(merge_map[k6_label]) for k6_label in range(6)}

print(f'  K6→K4 mapping: {numeric_map}')

# 每个 K4 类别的描述
k4_labels = np.array([numeric_map[l] for l in k6_labels])
print(f'\n  K=4 merged clusters:')
names = ['小体积基准型','中等过渡型','高散射实心型','巨大囊泡型']
for i in range(N_MERGED):
    mask = k4_labels == i
    center = np.mean(X_std[mask], axis=0)
    print(f'  C{i} ({names[i]}): {(mask).sum():>6,} ({(mask).sum()/len(k4_labels)*100:>5.1f}%)', end='')
    top = np.argsort(-np.abs(center))[:3]
    for t in top:
        print(f'  {CF_SHORT[t]}={center[t]:+.2f}', end='')
    print()

# 保存
pickle.dump(kmeans, open(os.path.join(MODEL_DIR, 'kmeans_k6.pkl'), 'wb'))
pickle.dump(scaler_k6, open(os.path.join(MODEL_DIR, 'scaler_k6.pkl'), 'wb'))
pickle.dump(numeric_map, open(os.path.join(MODEL_DIR, 'numeric_map.pkl'), 'wb'))
print(f'\n  Models saved: model_nnunet/')

# ================================================================
# STEP 2: 逐孔逐表型均值聚合 + Delta
# ================================================================
print(f'\n{"="*60}')
print(f'  STEP 2: Per-well Per-phenotype Aggregation + Delta')
print(f'{"="*60}')

FEATURE_LONG = {
    'Organoids_Volume':'Volume_Avg','Organoids_Volume_Fill':'Volume_Fill_Avg',
    'Organoids_Surface':'Surface_Avg','Cavity_Volume':'Cavity_Volume_All',
    'CavityNum':'CavityNum_Avg','LongAxis':'Long_Axis_Avg',
    'ShortAxis':'Short_Axis_Avg','Wall_Thickness':'Cyst_Thick_Avg',
    'Sphericity':'Sphericity_Avg','Scatt_Mean':'Scatt_Mean_Avg',
    'Scatt_STD':'Scatt_STD_Avg',
}

def aggregate_well(df, labels):
    rec = {}
    for cl in range(N_MERGED):
        mask = labels == cl
        n = int(mask.sum())
        rec[f'Number_{cl+1}'] = n
        sub = df.loc[mask] if n > 0 else df.iloc[:0]
        for src, alias in FEATURE_LONG.items():
            if src not in df.columns: continue
            if src == 'Cavity_Volume':
                rec[f'{alias}_{cl+1}'] = float(sub[src].sum())
            else:
                rec[f'{alias}_{cl+1}'] = float(sub[src].mean()) if n > 0 else 0.0
    return rec

d3_files = {f.replace('.xlsx','').split('_')[0]: f for f in os.listdir(D3_DIR) if f.endswith('.xlsx')}
d5_files = {f.replace('.xlsx','').split('_')[0]: f for f in os.listdir(D5_DIR) if f.endswith('.xlsx')}
common = sorted(set(d3_files) & set(d5_files))
print(f'  Paired wells: {len(common)}')

delta_rows, well_ids, atp_vals, all_feat_names = [], [], [], None
for wid in common:
    d3 = pd.read_excel(os.path.join(D3_DIR, d3_files[wid]))
    d5 = pd.read_excel(os.path.join(D5_DIR, d5_files[wid]))
    X3 = d3[CF].fillna(0).values; X5 = d5[CF].fillna(0).values
    c3 = np.array([numeric_map[l] for l in kmeans.predict(scaler_k6.transform(X3))])
    c5 = np.array([numeric_map[l] for l in kmeans.predict(scaler_k6.transform(X5))])
    f3 = aggregate_well(d3, c3); f5 = aggregate_well(d5, c5)
    if all_feat_names is None:
        all_feat_names = sorted(f3.keys())
    row = [f5.get(k, 0.0) - f3.get(k, 0.0) for k in all_feat_names]
    delta_rows.append(row)
    well_ids.append(wid)
    atp_vals.append(ATP_DB.get(wid, np.nan))

df_delta = pd.DataFrame(delta_rows, columns=all_feat_names)
df_delta['Well_ID'] = well_ids
df_delta['ATP'] = atp_vals

# 移除常数列
valid_cols = []
for c in all_feat_names:
    v = df_delta[c].dropna()
    if len(v) > 0 and v.nunique() > 1:
        valid_cols.append(c)
print(f'  Valid Delta features: {len(valid_cols)}')
df_delta.to_excel(os.path.join(OUTPUT_DIR, 'nnunet_delta_table.xlsx'), index=False)

# ================================================================
# STEP 3: 特征穷举搜索 + PCA
# ================================================================
print(f'\n{"="*60}')
print(f'  STEP 3: Feature Search ({N_SEARCH} iterations) + PCA')
print(f'{"="*60}')

df_clean = df_delta.dropna(subset=['ATP']).copy()
y_atp = df_clean['ATP'].values
X_full = df_clean[valid_cols].values
n_features = len(valid_cols)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

best_r, best_combo, best_pca, best_scaler, best_weights = 0.0, None, None, None, None
search_results = []

t0 = time.time()
for it in range(N_SEARCH):
    # 随机选特征数 6 ~ n_features
    n_sel = np.random.randint(max(4, n_features//3), n_features+1)
    sel_idx = sorted(np.random.choice(n_features, n_sel, replace=False))

    X_sel = X_full[:, sel_idx]
    # 标准化
    ss = StandardScaler()
    X_std = ss.fit_transform(X_sel)
    # PCA
    pc = PCA(n_components=N_PC)
    try:
        X_pc = pc.fit_transform(X_std)
    except:
        continue
    # 方差加权
    wr = pc.explained_variance_ratio_[:N_PC]
    wr = wr / wr.sum()
    score = X_pc @ wr
    r, _ = pearsonr(score, y_atp)
    abs_r = abs(r)

    search_results.append(abs_r)
    if abs_r > best_r:
        best_r = abs_r
        best_combo = sel_idx
        best_pca = pc
        best_scaler = ss
        best_weights = wr

    if (it+1) % 10000 == 0:
        elapsed = time.time() - t0
        print(f'  [{it+1}/{N_SEARCH}] best |r|={best_r:.4f}, elapsed={elapsed:.0f}s')

elapsed = time.time() - t0
print(f'\n  Done in {elapsed:.0f}s')
print(f'  Best |r| = {best_r:.4f}')
print(f'  Selected features: {len(best_combo)}/{n_features}')
for idx in best_combo:
    print(f'    {valid_cols[idx]}')

# ================================================================
# STEP 4: 最终 PCA 分析 + 保存模型
# ================================================================
print(f'\n{"="*60}')
print(f'  STEP 4: Final PCA & Model Export')
print(f'{"="*60}')

X_best = X_full[:, best_combo]
best_feat_names = [valid_cols[i] for i in best_combo]

# 方差解释
for i, v in enumerate(best_pca.explained_variance_ratio_):
    print(f'  PC{i+1}: {v*100:>6.1f}% (cum {best_pca.explained_variance_ratio_.cumsum()[i]*100:.1f}%)')

# 综合得分
X_std_best = best_scaler.transform(X_best)
pc_scores = best_pca.transform(X_std_best)
final_score = pc_scores @ best_weights
r_final, p_final = pearsonr(final_score, y_atp)
rho_final, p_s = spearmanr(final_score, y_atp)
print(f'\n  Final Score vs ATP: r={r_final:.4f}, p={p_final:.6f}, rho={rho_final:.4f}')

# 每个 PC 的 Loading
print(f'\n  PC Loading Matrix:')
header = '  ' + ' '.join(f'{s:>10s}' for s in [fn.split('_')[-1][:8] for fn in best_feat_names])
print(header)
for i in range(N_PC):
    row_str = f'  PC{i+1}'
    for j in range(len(best_combo)):
        row_str += f'{best_pca.components_[i][j]:>+10.3f}'
    print(row_str)

# Beta 公式
beta = np.zeros(len(best_combo))
for i in range(N_PC):
    beta += best_weights[i] * best_pca.components_[i]
beta /= best_scaler.scale_
# 调整符号使 r>0
if np.corrcoef(X_std_best @ (best_pca.components_[:N_PC].T @ best_weights), y_atp)[0,1] < 0:
    beta = -beta
    best_weights = -best_weights

print(f'\n  Score = Σ(βⱼ × ΔFeatureⱼ)')
beta_tuples = list(zip(best_feat_names, beta, np.abs(beta)))
beta_tuples.sort(key=lambda x: -x[2])
for name, b, _ in beta_tuples:
    print(f'  β_{name:<35s} = {b:+.6e}')

# 保存部署包
deploy = {
    'scaler': best_scaler,
    'pca': best_pca,
    'weights': best_weights,
    'beta': beta,
    'features': best_feat_names,
    'training_r': r_final,
    'numeric_map': numeric_map,
    'valid_cols': valid_cols,
}
pickle.dump(deploy, open(os.path.join(MODEL_DIR, 'nnunet_delta_deploy.pkl'), 'wb'))

# 保存结果
df_result = pd.DataFrame({'Well_ID': well_ids})
df_result['ATP'] = df_result['Well_ID'].map(ATP_DB)
# 计算每个well的完整Score
X_all_std = best_scaler.transform(df_clean[best_feat_names].values)
df_result['Score'] = np.nan
for i, wid in enumerate(df_clean['Well_ID']):
    mask = df_result['Well_ID'] == wid
    df_result.loc[mask, 'Score'] = float(X_all_std[i] @ best_pca.components_[:N_PC].T @ best_weights)
df_result.to_excel(os.path.join(OUTPUT_DIR, 'nnunet_prediction.xlsx'), index=False)

print(f'\n  Model saved: model_nnunet/nnunet_delta_deploy.pkl')
print(f'  Results saved: output/nnunet_prediction.xlsx')
print(f'\n{"="*60}')
print(f'  Summary')
print(f'{"="*60}')
print(f'  nnUNet Pipeline |r| = {best_r:.4f}')
print(f'  Original ICC Pipeline |r| = 0.9040  (for reference)')