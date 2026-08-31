import os, sys, glob, pickle, warnings
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

sys.path.insert(0, 'step7_pca_atp_v2_stratified')
from cluster_utils import RAW_FEATURES, load_model_package
from step7_pca_atp_v2_stratified.config import ATP_DATABASE, HEALTHY_CLUSTERS
from step7_pca_atp_v2_stratified.modeling import apply_clustering

print('='*70)
print('  论文模型验证：用论文公式计算Score vs ATP')
print('='*70)

# ============================================================
# 1. 加载数据 + KMeans聚类
# ============================================================
def load_data(base, day_folders):
    feats = list(RAW_FEATURES)
    dfs = []; seen = set()
    for dd, dv in day_folders:
        full = os.path.join(base, dd, 'measure_excel')
        if not os.path.exists(full): continue
        for fp in sorted(glob.glob(os.path.join(full, '*.xlsx'))):
            fn = os.path.basename(fp)
            wid = fn.replace('_0701.xlsx','').replace('_0703.xlsx','').replace('.xlsx','')
            if not wid or (wid,dv) in seen: continue
            tmp = pd.read_excel(fp)
            if 'Index' in tmp.columns: tmp = tmp.rename(columns={'Index':'Object_Id'})
            if 'Cavity_Ratio' not in tmp.columns: tmp['Cavity_Ratio'] = tmp['Cavity_Volume']/(tmp['Organoids_Volume_Fill']+1e-10)
            tmp = tmp.dropna(subset=feats)
            if len(tmp)==0: continue
            tmp['_well'] = fn.replace('.xlsx',''); tmp['_well_id'] = wid; tmp['_day'] = dv
            dfs.append(tmp); seen.add((wid,dv))
    df = pd.concat(dfs, ignore_index=True)
    wells = sorted(set(w for w,_ in seen))
    return df, feats, wells

icc_df, icc_feats, icc_wells = load_data('Data/FXN_2023_new（ICC）',
    [('FXN_20230701','0701'),('FXN_20230703','0703')])
icc_df, ws_icc = apply_clustering(icc_df, icc_feats)

# ============================================================
# 2. 提取论文定义的15个特征 (X11-X43)
# ============================================================
# 论文的Cluster映射:
#   Cluster 1 (Red) = 大囊状健康态
#   Cluster 2 (Yellow) = 大实心健康态  
#   Cluster 3 (Green) = 小实心中间态
#   Cluster 4 (Blue) = 极小致密损伤态
#
# 15个特征定义:
#   X11 = Cavity_Volume_1 (Cluster 1 平均空腔体积)
#   X12 = Surface_Area_1 (Cluster 1 平均表面积)
#   X13 = Roughness_1 (Cluster 1 平均粗糙度)
#   X21 = Number_2 (Cluster 2 数量)
#   X22 = ShortAxis_2 (Cluster 2 平均短轴长度)
#   X23 = Surface_Area_2 (Cluster 2 平均表面积)
#   X24 = Fill_Volume_2 (Cluster 2 平均填充体积)
#   X31 = Cyst_Thickness_3 (Cluster 3 平均囊壁厚度)
#   X32 = Number_3 (Cluster 3 数量)
#   X33 = OAC_Mean_3 (Cluster 3 平均Scatt_Mean)
#   X34 = ShortAxis_3 (Cluster 3 平均短轴长度)
#   X35 = Surface_Area_3 (Cluster 3 平均表面积)
#   X41 = LongAxis_4 (Cluster 4 平均长轴长度)
#   X42 = OAC_Mean_4 (Cluster 4 平均Scatt_Mean)
#   X43 = ShortAxis_4 (Cluster 4 平均短轴长度)

def extract_paper_features(df, ws, well_id, day_val):
    wdf = df[(df['_well_id']==well_id)&(df['_day']==day_val)]
    row = {}
    
    # 每个cluster的统计
    for c in range(4):
        cdf = wdf[wdf['Cluster']==c]
        n_c = len(cdf)
        
        if c == 0:  # Cluster 1 (Red/H)
            row['X11'] = cdf['Cavity_Volume'].mean() if n_c > 0 else 0
            row['X12'] = cdf['Organoids_Surface'].mean() if n_c > 0 else 0
            row['X13'] = cdf['Roughness'].mean() if n_c > 0 else 0
            
        elif c == 1:  # Cluster 2 (Yellow/H)
            row['X21'] = n_c
            row['X22'] = cdf['ShortAxis'].mean() if n_c > 0 else 0
            row['X23'] = cdf['Organoids_Surface'].mean() if n_c > 0 else 0
            row['X24'] = cdf['Organoids_Volume_Fill'].mean() if n_c > 0 else 0
            
        elif c == 2:  # Cluster 3 (Green/I)
            # 囊壁厚度 ≈ Cavity_Volume / Surface (近似)
            cv = cdf['Cavity_Volume'].mean() if n_c > 0 else 0
            sa = cdf['Organoids_Surface'].mean() if n_c > 0 else 1
            row['X31'] = cv / sa if sa > 0 else 0
            row['X32'] = n_c
            row['X33'] = cdf['Scatt_Mean'].mean() if n_c > 0 else 0
            row['X34'] = cdf['ShortAxis'].mean() if n_c > 0 else 0
            row['X35'] = cdf['Organoids_Surface'].mean() if n_c > 0 else 0
            
        elif c == 3:  # Cluster 4 (Blue/D)
            row['X41'] = cdf['LongAxis'].mean() if n_c > 0 else 0
            row['X42'] = cdf['Scatt_Mean'].mean() if n_c > 0 else 0
            row['X43'] = cdf['ShortAxis'].mean() if n_c > 0 else 0
    
    return row

# 为每个well-day提取15维特征
feature_rows = []
for wid in icc_wells:
    for dv in ['0701', '0703']:
        feat_row = extract_paper_features(icc_df, ws_icc, wid, dv)
        feat_row['_well_id'] = wid
        feat_row['_day'] = dv
        feature_rows.append(feat_row)

feat_df = pd.DataFrame(feature_rows)

# ============================================================
# 3. 用论文公式计算 Score
# ============================================================

# P1-P4 的系数 (来自论文公式4.1-4.4)
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

# 权重 (来自论文公式4.7)
w1, w2, w3, w4 = 0.468, 0.247, 0.165, 0.120

feat_names = ['X11','X12','X13','X21','X22','X23','X24',
              'X31','X32','X33','X34','X35',
              'X41','X42','X43']

def calc_score(row, coef_dict):
    return sum(coef_dict[f] * row[f] for f in feat_names)

def calc_F(row):
    P1 = calc_score(row, coef_P1)
    P2 = calc_score(row, coef_P2)
    P3 = calc_score(row, coef_P3)
    P4 = calc_score(row, coef_P4)
    return w1*P1 + w2*P2 + w3*P3 + w4*P4

# 计算F值
feat_df['F'] = feat_df.apply(calc_F, axis=1)

# ============================================================
# 4. 匹配ATP + 计算 ΔF
# ============================================================
feat_df['ATP'] = feat_df['_well_id'].map(ATP_DATABASE)

# 分离D3和D5
d3_df = feat_df[feat_df['_day']=='0701'].copy()
d5_df = feat_df[feat_df['_day']=='0703'].copy()

# 合并计算ΔF
merged = d3_df[[' _well_id','F']].merge(
    d5_df[[' _well_id','F','ATP']], 
    on='_well_id', 
    suffixes=('_D3', '_D5')
)
merged['Delta_F'] = merged['F_D5'] - merged['F_D3']

valid = merged['ATP'].notna()
print(f'\n有效样本: {valid.sum()} 个wells')

if valid.sum() >= 3:
    # 方法1: Delta_F vs ATP
    r_delta, p_delta = pearsonr(merged.loc[valid,'Delta_F'], merged.loc[valid,'ATP'])
    sp_delta, spp_delta = spearmanr(merged.loc[valid,'Delta_F'], merged.loc[valid,'ATP'])
    
    # 方法2: F_D5 vs ATP (终点得分)
    r_f5, p_f5 = pearsonr(merged.loc[valid,'F_D5'], merged.loc[valid,'ATP'])
    sp_f5, spp_f5 = spearmanr(merged.loc[valid,'F_D5'], merged.loc[valid,'ATP'])
    
    print(f'\n{"="*70}')
    print(f'  结果对比')
    print(f'{"="*70}')
    print(f'  {"方法":20s} {"Pearson r":>10s} {"p-value":>12s} {"Spearman ρ":>12s}')
    print(f'  {"-"*55}')
    print(f'  {"ΔF vs ATP":20s} {r_delta:>10.4f} {p_delta:>12.2e} {sp_delta:>12.4f}')
    print(f'  {"F_D5 (终点)":20s} {r_f5:>10.4f} {p_f5:>12.2e} {sp_f5:>12.4f}')
    print(f'\n  论文声称: r = 0.887 (联合用药独立测试)')
    print(f'  我们的结果: r = {r_delta:.4f} (ΔF)')
    print(f'             r = {r_f5:.4f} (F_D5)')
    
    gap_delta = abs(r_delta - 0.887)
    gap_f5 = abs(r_f5 - 0.887)
    print(f'\n  与论文差距:')
    print(f'    ΔF: |{r_delta:.4f} - 0.887| = {gap_delta:.4f}')
    print(f'    F_D5: |{r_f5:.4f} - 0.887| = {gap_f5:.4f}')
    
    if gap_delta < 0.05 or gap_f5 < 0.05:
        print(f'\n  ✓ 论文结果可信！误差在±0.05以内')
    elif gap_delta < 0.15 or gap_f5 < 0.15:
        print(f'\n  △ 论文结果基本可信，误差在合理范围')
    else:
        print(f'\n  ✗ 论文结果存疑！误差较大')

else:
    print('\n  有效样本不足，无法计算')

# ============================================================
# 5. 详细输出
# ============================================================
print(f'\n{"="*70}')
print(f'  详细数据 ({valid.sum()} wells)')
print(f'{"="*70}')
if valid.sum() > 0:
    detail = merged.loc[valid, [' _well_id','F_D3','F_D5','Delta_F','ATP']].copy()
    detail['ATP_norm'] = detail['ATP'] / detail['ATP'].max()
    detail['Delta_F_norm'] = (detail['Delta_F'] - detail['Delta_F'].min()) / (detail['Delta_F'].max() - detail['Delta_F'].min())
    print(detail.to_string(index=False))

print('\nDONE')