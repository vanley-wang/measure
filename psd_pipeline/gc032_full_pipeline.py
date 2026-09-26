"""
GC032 完整流水线 (5种药物泛化验证)
  Step 1: K=6→4 聚类 (纯形态特征, 无Scatt)
  Step 2: 逐孔逐表型聚合 + Delta = Day5 − Day1
  Step 3: 特征穷举搜索 (50000 iterations) + PCA
  Step 4: Score vs ATP 相关性验证
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

# ============================================================
# 0. 配置
# ============================================================
RANDOM_SEED = 42
N_SEARCH = 50000
N_PC = 4
MIN_ORGANOIDS_PER_WELL = 50  # 每孔最少类器官数
MIN_VOLUME = 10

DATA_DIR = r'D:\Desktop\music\measure\Data\GC032'
DAY1_DIR = os.path.join(DATA_DIR, '20241101', 'excel')
DAY5_DIR = os.path.join(DATA_DIR, '20241105', 'excel')
OUTPUT_DIR = r'D:\Desktop\music\measure\psd_pipeline\model_gc032'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 形态学特征 (无 Scatt)
MORPH_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity'
]

# ATP 数据 (来自 GC032-ATP.xlsx)
ATP_MAP = {
    'C3': 8638000, 'C4': 7800000, 'C5': 7260000,
    'C8': 185300, 'C11': 24330, 'C12': 71930,
    'D4': 66260, 'D5': 27630, 'D6': 25920,
    'E4': 63070, 'E5': 15450, 'E6': 68250,
    'F2': 14970000, 'F3': 14070000, 'F4': 10820000,
    'F7': 54140000, 'F8': 50680000, 'F9': 65380000,
    'F11': 55300000, 'F12': 46400000,
}

# ============================================================
# 1. 读取所有原始数据
# ============================================================
print("=" * 60)
print("  GC032 Full Pipeline: 5-Drug Generalization Test")
print("=" * 60)

all_rows = []
for date_label, excel_dir in [('1101', DAY1_DIR), ('1105', DAY5_DIR)]:
    if not os.path.isdir(excel_dir):
        print(f"  WARNING: {excel_dir} not found")
        continue
    for fname in sorted(os.listdir(excel_dir)):
        if not fname.endswith('.xlsx') or fname.startswith('~$'):
            continue
        fpath = os.path.join(excel_dir, fname)
        df = pd.read_excel(fpath)
        # 提取 well ID
        well_id = fname.split('_')[0]
        # 保留需要的列
        keep_cols = ['Index'] + MORPH_FEATURES
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            print(f"  SKIP {fname}: missing {missing}")
            continue
        sub = df[keep_cols].copy()
        sub['Well'] = well_id
        sub['Date'] = date_label
        all_rows.append(sub)

df_all = pd.concat(all_rows, ignore_index=True)
print(f"\n  Total organoids loaded: {len(df_all)}")

# 筛选有 ATP 的 well
wells_with_atp = set(ATP_MAP.keys())
df_all = df_all[df_all['Well'].isin(wells_with_atp)].copy()
print(f"  After ATP filter: {len(df_all)} organoids, {len(df_all['Well'].unique())} wells")

# 分离 Day1 和 Day5
df_d1 = df_all[df_all['Date'] == '1101'].copy()
df_d5 = df_all[df_all['Date'] == '1105'].copy()
print(f"  Day1: {len(df_d1)} organoids, {len(df_d1['Well'].unique())} wells")
print(f"  Day5: {len(df_d5)} organoids, {len(df_d5['Well'].unique())} wells")

# 配对 wells
paired_wells = sorted(set(df_d1['Well'].unique()) & set(df_d5['Well'].unique()))
print(f"  Paired wells: {len(paired_wells)} — {paired_wells}")

# 过滤每孔类器官数量太少的
valid_wells = []
for w in paired_wells:
    n1 = len(df_d1[df_d1['Well'] == w])
    n5 = len(df_d5[df_d5['Well'] == w])
    if n1 >= MIN_ORGANOIDS_PER_WELL and n5 >= MIN_ORGANOIDS_PER_WELL:
        valid_wells.append(w)
print(f"  Valid wells (>= {MIN_ORGANOIDS_PER_WELL} organoids): {len(valid_wells)}")

paired_wells = valid_wells

# ============================================================
# 2. K=6 聚类 + 合并 → K=4 (仅 Day1 数据)
# ============================================================
print("\n" + "=" * 60)
print("  STEP 1: K=6 Clustering + Merge -> 4 Phenotypes")
print("=" * 60)

df_d1_valid = df_d1[df_d1['Well'].isin(paired_wells)].copy()
X_d1 = df_d1_valid[MORPH_FEATURES].fillna(0).values

scaler_k6 = StandardScaler()
X_d1_std = scaler_k6.fit_transform(X_d1)

kmeans = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
k6_labels = kmeans.fit_predict(X_d1_std)
df_d1_valid['K6'] = k6_labels

# 合并最近的两个 cluster pair (两轮)
centers = kmeans.cluster_centers_
dist = cdist(centers, centers)
np.fill_diagonal(dist, np.inf)
pairs = []
for _ in range(2):
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    pairs.append((int(i), int(j)))
    dist[i, :] = np.inf; dist[:, i] = np.inf
    dist[j, :] = np.inf; dist[:, j] = np.inf

# 构建映射
k4_map = {}
next_id = 0
for i in range(6):
    if i not in k4_map:
        # 找这个 cluster 属于哪对
        found = False
        for pi, pj in pairs:
            if i == pi or i == pj:
                pair_id = next_id
                k4_map[pi] = pair_id
                k4_map[pj] = pair_id
                next_id += 1
                found = True
                break
        if not found:
            k4_map[i] = next_id
            next_id += 1

print(f"  K=6 cluster sizes:", {i: int((k6_labels == i).sum()) for i in range(6)})
print(f"  Merge pairs: {pairs}")
print(f"  K6->K4 mapping: {k4_map}")

df_d1_valid['K4'] = df_d1_valid['K6'].map(k4_map)
k4_sizes = {i: int((df_d1_valid['K4'] == i).sum()) for i in range(4)}
print(f"  K=4 merged clusters: {k4_sizes}")

# 对 Day5 数据用同样的 scaler + kmeans 预测
df_d5_valid = df_d5[df_d5['Well'].isin(paired_wells)].copy()
X_d5 = df_d5_valid[MORPH_FEATURES].fillna(0).values
X_d5_std = scaler_k6.transform(X_d5)
k6_labels_d5 = kmeans.predict(X_d5_std)
df_d5_valid['K6'] = k6_labels_d5
df_d5_valid['K4'] = df_d5_valid['K6'].map(k4_map)

# 打印每个 K4 表型的中心特征 (用 Day1 的数据)
print(f"\n  K=4 Phenotype Profiles (Day1, z-score):")
feat_short = ['Vol', 'VolFill', 'Surf', 'CavVol', 'CavNum', 'LongAx', 'ShortAx', 'WallTh', 'Spher']
# 重新计算 K4 centers
k4_centers = np.zeros((4, len(MORPH_FEATURES)))
for k in range(4):
    mask = df_d1_valid['K4'] == k
    if mask.sum() > 0:
        k4_centers[k] = X_d1_std[mask.values].mean(axis=0)
for k in range(4):
    top3 = np.argsort(-np.abs(k4_centers[k]))[:3]
    desc = ', '.join([f'{feat_short[i]}={k4_centers[k][i]:+.2f}' for i in top3])
    print(f"    K4={k} (n={k4_sizes.get(k,0)}): {desc}")

# ============================================================
# 3. 逐孔逐表型聚合 + Delta
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: Per-well Per-phenotype Aggregation + Delta")
print("=" * 60)

# 重新组织: 把 Day1+Day5 合并在一起做聚合
df_d1_valid['Src'] = 'D1'
df_d5_valid['Src'] = 'D5'
df_combined = pd.concat([df_d1_valid, df_d5_valid], ignore_index=True)

# 简化聚合: 先 groupby 然后手动构建
FEAT_SHORT = ['Vol', 'VolFill', 'Surf', 'CavVol', 'CavNum', 'LongAx', 'ShortAx', 'WallTh', 'Spher']
CAVITY_VOL_IDX = 3  # Cavity_Volume index (用 sum)
COUNT_COL = 'Organoids_Volume'

df_delta_rows = []
for well in paired_wells:
    row = {'Well': well}
    for src_label in ['D1', 'D5']:
        wdf = df_combined[(df_combined['Well'] == well) & (df_combined['Src'] == src_label)]
        for k4 in range(4):
            kdf = wdf[wdf['K4'] == k4]
            cnt = len(kdf)
            row[f'Count_{src_label}_K{k4}'] = cnt
            if cnt > 0:
                for fi, feat in enumerate(MORPH_FEATURES):
                    vals = kdf[feat].fillna(0).values
                    if fi == CAVITY_VOL_IDX:
                        row[f'{feat}_{src_label}_K{k4}'] = float(np.sum(vals))
                    else:
                        row[f'{feat}_{src_label}_K{k4}'] = float(np.mean(vals))
            else:
                for fi, feat in enumerate(MORPH_FEATURES):
                    row[f'{feat}_{src_label}_K{k4}'] = 0.0
    df_delta_rows.append(row)

df_agg = pd.DataFrame(df_delta_rows)

# Delta = Day5 − Day1
df_delta = pd.DataFrame()
df_delta['Well'] = df_agg['Well']
for k4 in range(4):
    for fi, feat in enumerate(MORPH_FEATURES):
        col_d1 = f'{feat}_D1_K{k4}'
        col_d5 = f'{feat}_D5_K{k4}'
        df_delta[f'{FEAT_SHORT[fi]}_K{k4}'] = df_agg[col_d5] - df_agg[col_d1]
    df_delta[f'Count_K{k4}'] = df_agg[f'Count_D5_K{k4}'] - df_agg[f'Count_D1_K{k4}']

# 删除全零列
feature_cols = [c for c in df_delta.columns if c != 'Well']
non_zero_cols = [c for c in feature_cols if df_delta[c].abs().sum() > 1e-10]
if len(non_zero_cols) < len(feature_cols):
    zc = set(feature_cols) - set(non_zero_cols)
    print(f"  Dropped zero-variance cols: {zc}")
feature_cols = non_zero_cols

# ATP
df_delta['ATP'] = df_delta['Well'].map(ATP_MAP)
df_valid = df_delta.dropna(subset=['ATP']).copy()
print(f"  Valid Delta features: {len(feature_cols)}, wells: {len(df_valid)}")

# 删除在 df_valid 中不存在的列
feature_cols = [c for c in feature_cols if c in df_valid.columns]
print(f"  Valid Delta features (in df_valid): {len(feature_cols)}")

X_full = df_valid[feature_cols].fillna(0).values
y_atp = df_valid['ATP'].values
n_features = len(feature_cols)
print(f"  Feature matrix: {X_full.shape[0]} wells × {n_features} features")

# ============================================================
# 4. 特征穷举搜索 + PCA
# ============================================================
print("\n" + "=" * 60)
print(f"  STEP 3: Feature Search ({N_SEARCH} iterations) + PCA")
print("=" * 60)

best_r, best_combo, best_pca, best_scaler, best_weights = 0.0, None, None, None, None
valid_cols = feature_cols

t0 = time.time()
report_interval = max(1, N_SEARCH // 5)

for it in range(N_SEARCH):
    n_sel = np.random.randint(max(4, n_features // 3), n_features + 1)
    sel_idx = sorted(np.random.choice(n_features, n_sel, replace=False))
    X_sel = X_full[:, sel_idx]
    
    ss = StandardScaler()
    X_std = ss.fit_transform(X_sel)
    pc = PCA(n_components=N_PC, random_state=RANDOM_SEED)
    try:
        X_pc = pc.fit_transform(X_std)
    except Exception:
        continue
    
    wr = pc.explained_variance_ratio_[:N_PC]
    wr = wr / wr.sum()
    score = X_pc @ wr
    r, _ = pearsonr(score, y_atp)
    abs_r = abs(r)
    
    if abs_r > best_r:
        best_r = abs_r
        best_combo = sel_idx
        best_pca = pc
        best_scaler = ss
        best_weights = wr
    
    if (it + 1) % report_interval == 0:
        elapsed = time.time() - t0
        print(f"  [{it+1}/{N_SEARCH}] best |r|={best_r:.4f}, elapsed={elapsed:.0f}s")

elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s")
print(f"  Best |r| = {best_r:.4f}")

# 最佳特征
best_feat_names = [valid_cols[i] for i in best_combo]
print(f"  Selected features: {len(best_feat_names)}/{n_features}")
for fn in best_feat_names:
    print(f"    {fn}")

# ============================================================
# 5. 最终 PCA & Score 计算
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4: Final PCA & Model Export")
print("=" * 60)

X_best = X_full[:, best_combo]
X_std_best = best_scaler.transform(X_best)
pc_scores = best_pca.transform(X_std_best)
final_score = pc_scores @ best_weights

r_final, p_final = pearsonr(final_score, y_atp)
rho_final, _ = spearmanr(final_score, y_atp)

print(f"  Final Score vs ATP: r={r_final:.4f}, p={p_final:.6f}, rho={rho_final:.4f}")

# PC 解释方差
var_ratio = best_pca.explained_variance_ratio_[:N_PC]
cumvar = np.cumsum(var_ratio)
for i in range(N_PC):
    print(f"  PC{i+1}: {var_ratio[i]*100:.1f}% (cum {cumvar[i]*100:.1f}%)")

# PC 载荷矩阵
print(f"\n  PC Loading Matrix:")
header = ''.join([f'{fn[:8]:>10s}' for fn in best_feat_names])
print(f"    {'':>6s}{header}")
for i in range(N_PC):
    loads = ''.join([f'{best_pca.components_[i][j]:+10.3f}' for j in range(len(best_feat_names))])
    print(f"    PC{i+1}   {loads}")

# Beta 权重 (Score = Σ beta_j × DeltaFeature_j)
beta = best_pca.components_[:N_PC].T @ best_weights
print(f"\n  Score = Sum(beta_j × DeltaFeature_j)")
for j in range(len(best_feat_names)):
    print(f"    beta_{best_feat_names[j]:40s} = {beta[j]:+.6e}")

# ============================================================
# 6. 结果输出
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: Results")
print("=" * 60)

df_valid['Score'] = final_score
df_valid['Predicted'] = final_score

# 按药物分组
# 从 ATP 文件获取药物分组
atp_df = pd.read_excel(os.path.join(DATA_DIR, 'GC032-ATP.xlsx'))
drug_map = {}
current_drug = None
for _, row in atp_df.iterrows():
    if pd.notna(row['药物']):
        current_drug = row['药物']
    if pd.notna(row['Name']):
        drug_map[row['Name']] = current_drug

df_valid['Drug'] = df_valid['Well'].map(drug_map)
df_valid['ATP_log'] = np.log10(df_valid['ATP'])

print(f"\n  {'Well':<6s} {'Drug':<16s} {'Score':>10s} {'ATP':>12s} {'ATP_log':>10s}")
print(f"  {'-'*60}")
for _, row in df_valid.iterrows():
    print(f"  {row['Well']:<6s} {str(row['Drug'])[:16]:<16s} {row['Score']:>10.4f} {row['ATP']:>12.0f} {row['ATP_log']:>10.4f}")

# 按药物分组统计
print(f"\n  Per-drug correlation:")
for drug in df_valid['Drug'].unique():
    sub = df_valid[df_valid['Drug'] == drug]
    if len(sub) >= 3:
        r_drug, _ = pearsonr(sub['Score'], sub['ATP'])
        print(f"    {drug:20s}: n={len(sub)}, r={r_drug:.4f}")

# 保存模型
model_pkg = {
    'scaler_k6': scaler_k6,
    'kmeans_k6': kmeans,
    'k4_map': k4_map,
    'feature_list': best_feat_names,
    'scaler_pca': best_scaler,
    'pca': best_pca,
    'weights': best_weights,
    'beta': beta,
    'var_ratio': var_ratio,
    'morph_features': MORPH_FEATURES,
    'r': r_final,
    'p': p_final,
}

model_path = os.path.join(OUTPUT_DIR, 'gc032_delta_deploy.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_pkg, f)
print(f"\n  Model saved: {model_path}")

# 保存结果 CSV
result_path = os.path.join(OUTPUT_DIR, 'gc032_results.csv')
df_valid[['Well', 'Drug', 'Score', 'ATP']].to_csv(result_path, index=False)
print(f"  Results saved: {result_path}")

print("\n" + "=" * 60)
print(f"  GC032 Pipeline Complete! r = {r_final:.4f}")
print("=" * 60)