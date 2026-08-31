import os, sys, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

sys.path.insert(0, 'step7_pca_atp_v2_stratified')
from cluster_utils import RAW_FEATURES, load_model_package
from step7_pca_atp_v2_stratified.modeling import apply_clustering

print('='*70)
print('  论文模型验证: ICC005 联合用药数据')
print('  验证论文声称: r = 0.887 (泛化能力)')
print('='*70)

# ============================================================
# 1. 加载ICC005两天的数据
# ============================================================
def load_icc005_data(base_dir, day_val):
    feats = list(RAW_FEATURES)
    measure_dir = os.path.join(base_dir, 'measure_excel')
    if not os.path.exists(measure_dir):
        print(f'  ERROR: {measure_dir} not found')
        return None, None, []
    
    dfs = []; wells = []
    for fp in sorted(glob.glob(os.path.join(measure_dir, '*.xlsx'))):
        fn = os.path.basename(fp)
        wid = fn.replace(f'_0424.xlsx','').replace(f'_0426.xlsx','').replace('.xlsx','')
        if not wid: continue
        
        tmp = pd.read_excel(fp)
        if 'Index' in tmp.columns:
            tmp = tmp.rename(columns={'Index': 'Object_Id'})
        if 'Cavity_Ratio' not in tmp.columns:
            tmp['Cavity_Ratio'] = tmp['Cavity_Volume'] / (tmp['Organoids_Volume_Fill'] + 1e-10)
        tmp = tmp.dropna(subset=feats)
        if len(tmp) == 0: continue
        
        tmp['_well'] = fn.replace('.xlsx', '')
        tmp['_well_id'] = wid
        tmp['_day'] = day_val
        dfs.append(tmp)
        wells.append(wid)
    
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return df, feats, sorted(set(wells))

# Day 1: 20240424
df_d1, feats_d1, wells_d1 = load_icc005_data(
    r'D:\Desktop\music\measure\Data\ICC005_20240424\ICC005_2024', 
    '0424'
)

# Day 2: 20240426  
df_d2, feats_d2, wells_d2 = load_icc005_data(
    r'D:\Desktop\music\measure\Data\ICC005_20240426', 
    '0426'
)

print(f'\nDay 1 (0424): {len(df_d1)} objects, {len(wells_d1)} wells')
print(f'Day 2 (0426): {len(df_d2)} objects, {len(wells_d2)} wells')
print(f'Wells D1: {wells_d1}')
print(f'Wells D2: {wells_d2}')

# ============================================================
# 2. KMeans聚类 (用ICC的KMeans模型)
# ============================================================
pkg = load_model_package(r'model\Kmeans-scatt.pickle')

if len(df_d1) > 0:
    df_d1['Cluster'] = [pkg['raw_to_final'].get(l, l) 
                         for l in pkg['kmeans'].predict(
                             pkg['scaler'].transform(df_d1[feats_d1])
                         )]
    
if len(df_d2) > 0:
    df_d2['Cluster'] = [pkg['raw_to_final'].get(l, l) 
                         for l in pkg['kmeans'].predict(
                             pkg['scaler'].transform(df_d2[feats_d2])
                         )]

# 合并两天
df_all = pd.concat([df_d1, df_d2], ignore_index=True)

# ============================================================
# 3. 提取论文的15个特征 (X11-X43)
# ============================================================
def extract_paper_features_for_well(df, well_id, day_val):
    """提取论文定义的15维特征"""
    wdf = df[(df['_well_id'] == well_id) & (df['_day'] == day_val)]
    
    features = {}
    
    # Cluster 1 (Red/H) - 大囊状健康态
    c1 = wdf[wdf['Cluster'] == 0]
    n1 = len(c1)
    features['X11'] = c1['Cavity_Volume'].mean() if n1 > 0 else 0      # Cavity Volume 1
    features['X12'] = c1['Organoids_Surface'].mean() if n1 > 0 else 0   # Surface Area 1
    features['X13'] = c1['Roughness'].mean() if n1 > 0 else 0           # Roughness 1
    
    # Cluster 2 (Yellow/H) - 大实心健康态
    c2 = wdf[wdf['Cluster'] == 1]
    n2 = len(c2)
    features['X21'] = n2                                                     # Number 2
    features['X22'] = c2['ShortAxis'].mean() if n2 > 0 else 0               # Short Axis 2
    features['X23'] = c2['Organoids_Surface'].mean() if n2 > 0 else 0       # Surface Area 2
    features['X24'] = c2['Organoids_Volume_Fill'].mean() if n2 > 0 else 0   # Fill Volume 2
    
    # Cluster 3 (Green/I) - 小实心中间态
    c3 = wdf[wdf['Cluster'] == 2]
    n3 = len(c3)
    cv3 = c3['Cavity_Volume'].mean() if n3 > 0 else 0
    sa3 = c3['Organoids_Surface'].mean() if n3 > 0 else 1
    features['X31'] = cv3 / sa3 if sa3 > 0 else 0                          # Cyst Thickness 3 (近似)
    features['X32'] = n3                                                     # Number 3
    features['X33'] = c3['Scatt_Mean'].mean() if n3 > 0 else 0             # OAC Mean 3
    features['X34'] = c3['ShortAxis'].mean() if n3 > 0 else 0               # Short Axis 3
    features['X35'] = c3['Organoids_Surface'].mean() if n3 > 0 else 0       # Surface Area 3
    
    # Cluster 4 (Blue/D) - 极小致密损伤态
    c4 = wdf[wdf['Cluster'] == 3]
    n4 = len(c4)
    features['X41'] = c4['LongAxis'].mean() if n4 > 0 else 0                # Long Axis 4
    features['X42'] = c4['Scatt_Mean'].mean() if n4 > 0 else 0             # OAC Mean 4
    features['X43'] = c4['ShortAxis'].mean() if n4 > 0 else 0              # Short Axis 4
    
    return features

# 为每个well-day提取特征
feature_rows = []
common_wells = sorted(set(wells_d1) & set(wells_d2))

for wid in common_wells:
    # Day 1
    feat_d1 = extract_paper_features_for_well(df_all, wid, '0424')
    feat_d1['_well_id'] = wid
    feat_d1['_day'] = '0424'
    feature_rows.append(feat_d1)
    
    # Day 2
    feat_d2 = extract_paper_features_for_well(df_all, wid, '0426')
    feat_d2['_well_id'] = wid
    feat_d2['_day'] = '0426'
    feature_rows.append(feat_d2)

feat_df = pd.DataFrame(feature_rows)
print(f'\n提取了 {len(feat_df)} 条记录 ({len(common_wells)} wells × 2 days)')

# ============================================================
# 4. 用论文公式计算 P1-P4 和 F
# ============================================================

# 论文公式(4.1)-(4.4)的系数
coef_P1 = {
    'X11': -0.092, 'X12': -0.087, 'X13': 0.085,
    'X21': -0.015, 'X22': -0.149, 'X23': -0.145, 'X24': 0.160,
    'X31': 0.108, 'X32': 0.188, 'X33': 0.208, 'X34': 0.148, 'X35': -0.171,
    'X41': -0.106, 'X42': -0.094, 'X43': -0.087
}

coef_P2 = {
    'X11': -0.214, 'X12': -0.080, 'X13': -0.097,
    'X21': -0.271, 'X22': 0.213, 'X23': 0.218, 'X24': -0.238,
    'X31': 0.094, 'X32': 0.082, 'X33': 0.084, 'X34': 0.211, 'X35': -0.081,
    'X41': 0.086, 'X42': -0.105, 'X43': 0.032
}

coef_P3 = {
    'X11': 0.222, 'X12': 0.018, 'X13': 0.199,
    'X21': 0.182, 'X22': 0.238, 'X23': 0.250, 'X24': -0.137,
    'X31': 0.074, 'X32': 0.198, 'X33': 0.159, 'X34': -0.153, 'X35': -0.056,
    'X41': -0.339, 'X42': 0.047, 'X43': -0.146
}

coef_P4 = {
    'X11': -0.240, 'X12': 0.618, 'X13': 0.096,
    'X21': -0.324, 'X22': -0.067, 'X23': -0.017, 'X24': 0.036,
    'X31': -0.256, 'X32': 0.220, 'X33': 0.091, 'X34': -0.102, 'X35': 0.218,
    'X41': -0.190, 'X42': 0.215, 'X43': -0.157
}

# 权重 (论文公式4.7)
w1, w2, w3, w4 = 0.468, 0.247, 0.165, 0.120

feat_names = ['X11','X12','X13','X21','X22','X23','X24',
              'X31','X32','X33','X34','X35',
              'X41','X42','X43']

def calc_component(row, coef_dict):
    """计算主成分得分"""
    return sum(coef_dict[f] * row[f] for f in feat_names)

def calc_composite_score(row):
    """计算综合得分F"""
    P1 = calc_component(row, coef_P1)
    P2 = calc_component(row, coef_P2)
    P3 = calc_component(row, coef_P3)
    P4 = calc_component(row, coef_P4)
    F = w1*P1 + w2*P2 + w3*P3 + w4*P4
    return F, P1, P2, P3, P4

# 计算所有样本的得分
scores = []
for idx, row in feat_df.iterrows():
    F, P1, P2, P3, P4 = calc_composite_score(row)
    scores.append({
        '_well_id': row['_well_id'],
        '_day': row['_day'],
        'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4,
        'F': F
    })

score_df = pd.DataFrame(scores)

# ============================================================
# 5. 计算 ΔF 并匹配ATP
# ============================================================

d1_scores = score_df[score_df['_day'] == '0424'][['_well_id', 'F', 'P1', 'P2', 'P3', 'P4']].copy()
d1_scores.columns = [' _well_id', 'F_D1', 'P1_D1', 'P2_D1', 'P3_D1', 'P4_D1']

d2_scores = score_df[score_df['_day'] == '0426'][['_well_id', 'F', 'P1', 'P2', 'P3', 'P4']].copy()
d2_scores.columns = [' _well_id', 'F_D2', 'P1_D2', 'P2_D2', 'P3_D2', 'P4_D2']

merged = d1_scores.merge(d2_scores, on='_well_id')
merged['Delta_F'] = merged['F_D2'] - merged['F_D1']
merged['Delta_P1'] = merged['P1_D2'] - merged['P1_D1']
merged['Delta_P2'] = merged['P2_D2'] - merged['P2_D1']
merged['Delta_P3'] = merged['P3_D2'] - merged['P3_D1']
merged['Delta_P4'] = merged['P4_D2'] - merged['P4_D1']

print(f'\n合并后: {len(merged)} wells')
print(f'\n{"="*70}')
print(f'  得分统计')
print(f'{"="*70}')
print(f'{"Well":>6s} {"F_D1":>10s} {"F_D2":>10s} {"ΔF":>10s} {"P1":>8s} {"P2":>8s} {"P3":>8s} {"P4":>8s}')
print(f'{"-"*70}')
for _, row in merged.iterrows():
    print(f'{row[" _well_id"]:>6s} {row["F_D1"]:>10.2f} {row["F_D2"]:>10.2f} {row["Delta_F"]:>10.2f} '
          f'{row["Delta_P1"]:>8.2f} {row["Delta_P2"]:>8.2f} {row["Delta_P3"]:>8.2f} {row["Delta_P4"]:>8.2f}')

# ============================================================
# 6. 需要用户提供ATP数据进行相关性分析
# ============================================================
print(f'\n{"="*70}')
print(f'  下一步：需要ATP数据完成验证')
print(f'{"="*70}')
print(f'''
已计算完成的：
  ✓ 两天的15维特征 (X11-X43)  
  ✓ 主成分得分 P1, P2, P3, P4
  ✓ 综合生长得分 F = 0.468×P1 + 0.247×P2 + 0.165×P3 + 0.120×P4
  ✓ 相对生长得分 ΔF = F_Day2 - F_Day1
  
待完成：
  ? ATP检测值（请提供这{len(merged)}个well的ATP数据）
  
验证目标：
  论文声称: ΔF vs ATP 的 Pearson r = 0.887 (p < 0.001)
  
请提供ATP数据格式示例:
  B2: 12345678
  B3: 23456789
  ...
''')

# 保存中间结果供后续使用
output_file = r'D:\Desktop\music\measure\ICC005_paper_scores.xlsx'
merged.to_excel(output_file, index=False)
print(f'\n中间结果已保存: {output_file}')
print('(包含 F_D1, F_D2, Delta_F, P1-P4 等所有得分)')

print('\nDONE')