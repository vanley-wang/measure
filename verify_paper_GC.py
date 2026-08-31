"""
论文模型验证：GC数据用GC专用KMeans + PCA，然后按论文公式计算F
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

print('=' * 70)
print('  论文模型验证: GC 联合用药实验 (GC专用KMeans+PCA)')
print('  验证目标: r = 0.887 (论文声称)')
print('=' * 70)

# ============================================================
# 1. 加载ATP数据
# ============================================================
atp_file = r'D:\Desktop\music\measure\Data\FXN_2023_new（GC）\ATP.xlsx'
atp_df = pd.read_excel(atp_file)

atp_dict = {}
if 'ATP' in atp_df.columns:
    for idx, row in atp_df.iterrows():
        name = str(row['Name']).strip() if pd.notna(row['Name']) else ''
        if not name: continue
        wid = name.split('_')[0] if '_' in name else name
        atp_val = row['ATP']
        if pd.notna(atp_val) and float(atp_val) > 0:
            atp_dict[wid] = float(atp_val)

print(f'解析到 {len(atp_dict)} 个ATP值')

# ============================================================
# 2. 加载measure_excel + scatt数据
# ============================================================
GC_BASE = r'D:\Desktop\music\measure\Data\FXN_2023_new（GC）'

def load_gc_full(base_dir, day_folder):
    measure_dir = os.path.join(base_dir, day_folder, 'measure_excel')
    scatt_dir = os.path.join(base_dir, day_folder, 'scatt')
    
    if not os.path.exists(measure_dir):
        return None
    
    dfs = []
    measure_files = sorted(glob.glob(os.path.join(measure_dir, '*.xlsx')))
    print(f'\n  [{day_folder}] 加载 {len(measure_files)} 个文件...')
    
    for fp in measure_files:
        fn = os.path.basename(fp)
        wid = fn.replace('.xlsx', '')
        parts = wid.split('_')
        if len(parts) < 2: continue
        wid = parts[0]
        day_suffix = parts[1]
        
        tmp = pd.read_excel(fp)
        if 'Index' in tmp.columns:
            tmp = tmp.rename(columns={'Index': 'Object_Id'})
        
        scatt_file = os.path.join(scatt_dir, f'{wid}_{day_suffix}_scatt.xlsx')
        if os.path.exists(scatt_file):
            scatt_tmp = pd.read_excel(scatt_file)
            if len(scatt_tmp) == len(tmp):
                for col in scatt_tmp.columns:
                    if col not in tmp.columns and col not in ['Index', 'Object_Id']:
                        tmp[col] = scatt_tmp[col].values
        
        tmp['_well'] = fn.replace('.xlsx', '')
        tmp['_well_id'] = wid
        tmp['_day'] = day_suffix
        dfs.append(tmp)
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)

df_d1 = load_gc_full(GC_BASE, 'FXN_20230701')
df_d2 = load_gc_full(GC_BASE, 'FXN_20230703')

print(f'\nDay1: {len(df_d1)} objects, Day2: {len(df_d2)} objects')

# ============================================================
# 3. GC专用KMeans (4类)
# ============================================================
cluster_feats = ['Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
                 'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
                 'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD']

df_combined = pd.concat([df_d1[cluster_feats], df_d2[cluster_feats]], ignore_index=True)
scaler = StandardScaler().fit(df_combined)
X_scaled = scaler.transform(df_combined)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
labels_all = kmeans.labels_

df_d1 = df_d1.copy()
df_d2 = df_d2.copy()
df_d1['Cluster'] = labels_all[:len(df_d1)]
df_d2['Cluster'] = labels_all[len(df_d1):]

# 聚类分布
for c in range(4):
    n1 = (df_d1['Cluster'] == c).sum()
    n2 = (df_d2['Cluster'] == c).sum()
    print(f'  Cluster {c}: Day1={n1}, Day2={n2}')

# 按Volume均值排序，0=最大(健康), 3=最小(损伤)
cluster_vol = []
for c in range(4):
    cv = df_combined.iloc[labels_all == c]['Organoids_Volume'].mean()
    cluster_vol.append((c, cv))
cluster_vol.sort(key=lambda x: x[1], reverse=True)
cluster_map = {old: new for new, (old, _) in enumerate(cluster_vol)}
print(f'  聚类映射 (按体积降序): {cluster_map}')

df_d1['Cluster'] = df_d1['Cluster'].map(cluster_map)
df_d2['Cluster'] = df_d2['Cluster'].map(cluster_map)

for c in range(4):
    n1 = (df_d1['Cluster'] == c).sum()
    n2 = (df_d2['Cluster'] == c).sum()
    names = {0: '大囊状(H)', 1: '大实心(H)', 2: '小实心(I)', 3: '极小(D)'}
    print(f'  Cluster {c} ({names[c]}): Day1={n1}, Day2={n2}')

df_all = pd.concat([df_d1, df_d2], ignore_index=True)

# ============================================================
# 4. 提取论文15维特征 (X11-X43)
# ============================================================
def extract_paper_features(df, well_id, day_val):
    wdf = df[(df['_well_id'] == well_id) & (df['_day'] == day_val)]
    f = {}
    
    c1 = wdf[wdf['Cluster'] == 0]; n1 = len(c1)
    f['X11'] = c1['Cavity_Volume'].mean() if n1 > 0 else 0
    f['X12'] = c1['Organoids_Surface'].mean() if n1 > 0 else 0
    f['X13'] = c1['Sphericity'].mean() if n1 > 0 else 0
    
    c2 = wdf[wdf['Cluster'] == 1]; n2 = len(c2)
    f['X21'] = n2
    f['X22'] = c2['ShortAxis'].mean() if n2 > 0 else 0
    f['X23'] = c2['Organoids_Surface'].mean() if n2 > 0 else 0
    f['X24'] = c2['Organoids_Volume_Fill'].mean() if n2 > 0 else 0
    
    c3 = wdf[wdf['Cluster'] == 2]; n3 = len(c3)
    cv3 = c3['Cavity_Volume'].mean() if n3 > 0 else 0
    sa3 = c3['Organoids_Surface'].mean() if n3 > 0 else 1
    f['X31'] = cv3 / sa3 if sa3 > 0 else 0
    f['X32'] = n3
    f['X33'] = c3['Scatt_Mean'].mean() if n3 > 0 else 0
    f['X34'] = c3['ShortAxis'].mean() if n3 > 0 else 0
    f['X35'] = c3['Organoids_Surface'].mean() if n3 > 0 else 0
    
    c4 = wdf[wdf['Cluster'] == 3]; n4 = len(c4)
    f['X41'] = c4['LongAxis'].mean() if n4 > 0 else 0
    f['X42'] = c4['Scatt_Mean'].mean() if n4 > 0 else 0
    f['X43'] = c4['ShortAxis'].mean() if n4 > 0 else 0
    
    return f

feature_rows = []
well_day = df_all.groupby(['_well_id', '_day']).size().reset_index()

for _, row in well_day.iterrows():
    wid = row['_well_id']; dv = row['_day']
    feat = extract_paper_features(df_all, wid, dv)
    feat['_well_id'] = wid; feat['_day'] = dv
    feature_rows.append(feat)

feat_df = pd.DataFrame(feature_rows)
print(f'\n提取了 {len(feat_df)} 条特征记录')

# ============================================================
# 5. 训练GC专用PCA + 计算P1-P4和F
# ============================================================
feat_names = ['X11','X12','X13','X21','X22','X23','X24','X31','X32','X33','X34','X35','X41','X42','X43']

# 标准化特征
X_feat = feat_df[feat_names].fillna(0).values
scaler_feat = StandardScaler().fit(X_feat)
X_scaled_feat = scaler_feat.transform(X_feat)

# PCA
pca = PCA(n_components=4)
pca.fit(X_scaled_feat)

print(f'\nPCA解释方差: {pca.explained_variance_ratio_}')
print(f'累计: {pca.explained_variance_ratio_.cumsum()}')

P_all = pca.transform(X_scaled_feat)

for i in range(4):
    feat_df[f'P{i+1}'] = P_all[:, i]

# 论文权重
w1, w2, w3, w4 = 0.468, 0.247, 0.165, 0.120
feat_df['F'] = w1*feat_df['P1'] + w2*feat_df['P2'] + w3*feat_df['P3'] + w4*feat_df['P4']

# ============================================================
# 6. 计算 ΔF + 匹配ATP
# ============================================================
d1 = feat_df[feat_df['_day'] == '0701'][['_well_id','F','P1','P2','P3','P4']].copy()
d1.columns = ['_well_id','F_D1','P1_D1','P2_D1','P3_D1','P4_D1']

d2 = feat_df[feat_df['_day'] == '0703'][['_well_id','F','P1','P2','P3','P4']].copy()
d2.columns = ['_well_id','F_D2','P1_D2','P2_D2','P3_D2','P4_D2']

merged = d1.merge(d2, on='_well_id')
merged['Delta_F'] = merged['F_D2'] - merged['F_D1']
merged['ATP'] = merged['_well_id'].map(atp_dict)

valid = merged['ATP'].notna()
print(f'\n总wells: {len(merged)}, 有ATP: {valid.sum()}')

# ============================================================
# 7. 相关性分析
# ============================================================
if valid.sum() >= 3:
    r_delta, p_delta = pearsonr(merged.loc[valid, 'Delta_F'], merged.loc[valid, 'ATP'])
    sp_delta, spp_delta = spearmanr(merged.loc[valid, 'Delta_F'], merged.loc[valid, 'ATP'])
    
    r_f2, p_f2 = pearsonr(merged.loc[valid, 'F_D2'], merged.loc[valid, 'ATP'])
    sp_f2, spp_f2 = spearmanr(merged.loc[valid, 'F_D2'], merged.loc[valid, 'ATP'])
    
    r_f1, p_f1 = pearsonr(merged.loc[valid, 'F_D1'], merged.loc[valid, 'ATP'])
    
    print(f'\n{"=" * 80}')
    print(f'  █ GC验证结果 (论文声称 r = 0.887)')
    print(f'{"=" * 80}')
    print(f'{"方法":30s} {"Pearson r":>10s} {"p-value":>12s} {"Spearman ρ":>12s}')
    print(f'{"-" * 70}')
    print(f'{"ΔF vs ATP":30s} {r_delta:>10.4f} {p_delta:>12.2e} {sp_delta:>12.4f}')
    print(f'{"F_D2 (终点得分)":30s} {r_f2:>10.4f} {p_f2:>12.2e} {sp_f2:>12.4f}')
    print(f'{"F_D1 (基线得分)":30s} {r_f1:>10.4f} {p_f1:>12.2e} {"-":>12s}')
    
    gap_delta = abs(r_delta - 0.887)
    gap_f2 = abs(r_f2 - 0.887)
    print(f'\n  ΔF与论文差距: |{r_delta:.4f} - 0.887| = {gap_delta:.4f}')
    print(f'  F_D2与论文差距: |{r_f2:.4f} - 0.887| = {gap_f2:.4f}')
    
    if gap_delta < 0.10:
        verdict = '✅ 论文结果可信！'
    elif gap_delta < 0.20:
        verdict = '⚠️ 论文结果基本可信，有轻微偏差'
    elif gap_delta < 0.30:
        verdict = '❌ 论文结果存疑，偏差较大'
    else:
        verdict = '❌ 论文结果不可信！偏差>0.30'
    
    print(f'\n  最终判定: {verdict}')

# ============================================================
# 8. 详细数据
# ============================================================
print(f'\n{"=" * 100}')
print(f'  每孔详细数据 ({valid.sum()} 有效样本)')
print(f'{"=" * 100}')
if valid.sum() > 0:
    detail = merged.loc[valid].copy()
    print(f'{"Well":>6s} {"F_D1":>10s} {"F_D2":>10s} {"ΔF":>10s} {"ATP":>12s}')
    print('-' * 50)
    for _, row in detail.sort_values('ATP').iterrows():
        print(f'{row["_well_id"]:>6s} {row["F_D1"]:>10.2f} {row["F_D2"]:>10.2f} {row["Delta_F"]:>10.2f} {row["ATP"]:>12.0f}')

output_file = r'D:\Desktop\music\measure\GC_paper_verification_v2.xlsx'
merged.to_excel(output_file, index=False)
print(f'\n完整结果: {output_file}')
print('DONE')