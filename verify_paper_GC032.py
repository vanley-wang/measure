import os, sys, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

sys.path.insert(0, 'step7_pca_atp_v2_stratified')
from cluster_utils import RAW_FEATURES, load_model_package

print('='*70)
print('  论文模型验证: GC032 联合用药实验')
print('  验证目标: r = 0.887 (论文声称的泛化能力)')
print('='*70)

# ============================================================
# 1. 加载GC032 ATP数据
# ============================================================
atp_file = r'D:\Desktop\music\measure\Data\GC032\GC032-ATP.xlsx'
atp_df = pd.read_excel(atp_file)

print(f'\nATP数据格式:')
print(atp_df.head(20))

# 解析ATP数据 - 根据实际格式，列是: 药物, Name, 20241101, 20241105, 差值, ATP平均
# 提取well ID和对应的ATP值
atp_dict = {}
print(f'\nATP数据列名: {list(atp_df.columns)}')

for idx, row in atp_df.iterrows():
    well_id = str(row.iloc[1]).strip() if len(row) > 1 else None  # 第2列是Name
    if well_id and len(well_id) > 0 and well_id != 'nan':
        # 尝试找到ATP列（可能是"ATP平均"或包含"ATP"的列）
        for col in ['ATP平均', 'ATP', atp_df.columns[-1] if len(atp_df.columns) > 0 else None]:
            if col and col in atp_df.columns and pd.notna(row[col]):
                try:
                    atp_val = float(row[col])
                    if atp_val > 0:
                        atp_dict[well_id] = atp_val
                except (ValueError, TypeError):
                    pass
                break

print(f'\n提取到 {len(atp_dict)} 个ATP值:')
for k, v in sorted(atp_dict.items()):
    print(f'  {k}: {v:.0f}')

# ============================================================
# 2. 加载GC032两天的形态数据
# ============================================================
def load_gc032_data(base_dir, day_val):
    excel_dir = os.path.join(base_dir, day_val, 'excel')
    if not os.path.exists(excel_dir):
        print(f'  ERROR: {excel_dir} not found')
        return None, None, []
    
    # 先检查一个文件确定实际列名
    sample_files = sorted(glob.glob(os.path.join(excel_dir, '*.xlsx')))
    if len(sample_files) == 0:
        print(f'  ERROR: No xlsx files found')
        return None, None, []
    
    sample_df = pd.read_excel(sample_files[0])
    actual_cols = list(sample_df.columns)
    print(f'  实际列名 ({day_val}): {actual_cols}')
    
    # 检查是否有Scatt/OAC相关列
    has_scatt = any('scatt' in c.lower() or 'oac' in c.lower() for c in actual_cols)
    print(f'  是否有OAC/Scatt列: {has_scatt}')
    
    # 动态选择可用的特征列（排除ID和元数据列）
    exclude_cols = ['Object_Id', 'Index', '_well', '_well_id', '_day', 
                    'Cavity_Ratio', 'Image_Name', 'FileName']
    
    # 基础形态学特征（所有数据都应该有）
    base_feats = [c for c in actual_cols if c not in exclude_cols and 
                  c.startswith(('Organoids_', 'Cavity_', 'LongAxis', 'ShortAxis'))]
    
    # 如果有Scatt/OAC特征，也加入
    if has_scatt:
        scatt_feats = [c for c in actual_cols if 'scatt' in c.lower() or 'oac' in c.lower()]
        feats = base_feats + scatt_feats
    else:
        feats = base_feats
        # 如果没有Scatt特征，添加虚拟列（用于后续聚类）
        print(f'  ⚠️ 警告: 无OAC特征，将使用纯形态学特征')
    
    dfs = []; wells = []
    for fp in sample_files:
        fn = os.path.basename(fp)
        wid = fn.replace(f'_2024{day_val}.xlsx', '').replace('.xlsx', '')
        if not wid or len(wid) < 2: continue
        
        tmp = pd.read_excel(fp)
        if 'Index' in tmp.columns:
            tmp = tmp.rename(columns={'Index': 'Object_Id'})
        
        # 只保留实际存在的特征列
        available_feats = [f for f in feats if f in tmp.columns]
        if len(available_feats) < 3:
            continue
            
        tmp = tmp.dropna(subset=available_feats)
        if len(tmp) == 0: continue
        
        tmp['_well'] = fn.replace('.xlsx', '')
        tmp['_well_id'] = wid
        tmp['_day'] = day_val
        dfs.append(tmp)
        wells.append(wid)
    
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return df, feats, sorted(set(wells))

base_gc032 = r'D:\Desktop\music\measure\Data\GC032'
df_d1, feats_d1, wells_d1 = load_gc032_data(base_gc032, '20241101')
df_d2, feats_d2, wells_d2 = load_gc032_data(base_gc032, '20241105')

print(f'\nDay 1 (20241101): {len(df_d1)} objects, {len(wells_d1)} wells')
print(f'Day 2 (20241105): {len(df_d2)} objects, {len(wells_d2)} wells')

# ============================================================
# 3. KMeans聚类（处理特征不匹配问题）
# ============================================================

# 尝试加载KMeans模型
model_paths = [
    (r'model\Kmeans-scatt.pickle', 'ICC'),
    (r'model\Kmeans-scatt-gc.pickle', 'GC'),
]

pkg = None
model_source = None
for mpath, mname in model_paths:
    if os.path.exists(mpath):
        try:
            pkg = load_model_package(mpath)
            model_source = mname
            print(f'\n尝试使用 {mname} KMeans 模型...')
            
            # 检查模型期望的特征
            model_feats_expected = list(pkg['scaler'].feature_names_in_) if hasattr(pkg['scaler'], 'feature_names_in_') else []
            print(f'  模型期望特征: {len(model_feats_expected)} 个')
            print(f'  实际可用特征: {len(feats_d1)} 个 ({feats_d1[:5]}...)')
            
            # 找到公共特征
            common_feats = [f for f in feats_d1 if f in model_feats_expected]
            if len(common_feats) >= 5:  # 至少需要5个公共特征
                print(f'  公共特征: {len(common_feats)} 个 → 可以使用该模型')
                break
            else:
                print(f'  ⚠️ 公共特征不足({len(common_feats)}个)，尝试下一个模型...')
                pkg = None
                
        except Exception as e:
            print(f'  加载失败: {e}')
            pkg = None

if pkg is None or len(common_feats) < 8:  # 如果公共特征不足或无法加载模型
    print('\n⚠️ 将基于纯形态学特征进行新聚类（因为缺少OAC特征）')
    from sklearn.cluster import KMeans as SimpleKMeans
    from sklearn.preprocessing import StandardScaler
    
    # 使用可用的形态学特征（排除Index等ID列）
    cluster_feats = [f for f in feats_d1 if f in df_d1.columns and 
                     f not in ['Index', 'Cluster']]
    
    if len(cluster_feats) < 3:
        print('ERROR: 特征数量不足，无法聚类')
        sys.exit(1)
    
    print(f'  聚类特征 ({len(cluster_feats)}个): {cluster_feats}')
    
    # 合并两天数据一起训练（确保一致性）
    df_combined = pd.concat([df_d1, df_d2], ignore_index=True)
    scaler_temp = StandardScaler().fit(df_combined[cluster_feats])
    X_scaled = scaler_temp.transform(df_combined[cluster_feats])
    
    # 简单4类聚类（与论文一致）
    simple_kmeans = SimpleKMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
    
    # 分配回原始数据
    clusters_all = simple_kmeans.labels_
    df_d1['Cluster'] = clusters_all[:len(df_d1)]
    df_d2['Cluster'] = clusters_all[len(df_d1):]
    
    print(f'  ✓ 使用新训练的KMeans完成聚类')
else:
    # 使用预训练模型（理论上不应该走到这里，因为GC032没有OAC特征）
    print(f'\n✓ 使用 {model_source} KMeans 模型（{len(common_feats)}个公共特征）')
    
    if len(df_d1) > 0:
        X_d1 = pkg['scaler'].transform(df_d1[common_feats])
        raw_labels_d1 = pkg['kmeans'].predict(X_d1)
        df_d1['Cluster'] = [pkg['raw_to_final'].get(l, l) for l in raw_labels_d1]
        
    if len(df_d2) > 0:
        X_d2 = pkg['scaler'].transform(df_d2[common_feats])
        raw_labels_d2 = pkg['kmeans'].predict(X_d2)
        df_d2['Cluster'] = [pkg['raw_to_final'].get(l, l) for l in raw_labels_d2]

df_all = pd.concat([df_d1, df_d2], ignore_index=True)

# ============================================================
# 4. 提取论文的15个特征 (X11-X43)
# ============================================================
def extract_paper_features_for_well(df, well_id, day_val):
    """提取论文定义的15维特征（自动处理缺失特征）"""
    wdf = df[(df['_well_id'] == well_id) & (df['_day'] == day_val)]
    
    features = {}
    
    # 检查可用列
    has_scatt = 'Scatt_Mean' in df.columns
    has_roughness = 'Roughness' in df.columns
    
    # Cluster 1 (Red/H) - 大囊状健康态
    c1 = wdf[wdf['Cluster'] == 0]
    n1 = len(c1)
    features['X11'] = c1['Cavity_Volume'].mean() if n1 > 0 and 'Cavity_Volume' in c1.columns else 0      # Cavity Volume 1
    features['X12'] = c1['Organoids_Surface'].mean() if n1 > 0 and 'Organoids_Surface' in c1.columns else 0   # Surface Area 1
    features['X13'] = c1['Roughness'].mean() if n1 > 0 and has_roughness else 0           # Roughness 1 (可选)
    
    # Cluster 2 (Yellow/H) - 大实心健康态
    c2 = wdf[wdf['Cluster'] == 1]
    n2 = len(c2)
    features['X21'] = n2                                                     # Number 2
    features['X22'] = c2['ShortAxis'].mean() if n2 > 0 and 'ShortAxis' in c2.columns else 0               # Short Axis 2
    features['X23'] = c2['Organoids_Surface'].mean() if n2 > 0 and 'Organoids_Surface' in c2.columns else 0       # Surface Area 2
    features['X24'] = c2['Organoids_Volume_Fill'].mean() if n2 > 0 and 'Organoids_Volume_Fill' in c2.columns else 0   # Fill Volume 2
    
    # Cluster 3 (Green/I) - 小实心中间态
    c3 = wdf[wdf['Cluster'] == 2]
    n3 = len(c3)
    cv3 = c3['Cavity_Volume'].mean() if n3 > 0 and 'Cavity_Volume' in c3.columns else 0
    sa3 = c3['Organoids_Surface'].mean() if n3 > 0 and 'Organoids_Surface' in c3.columns else 1
    features['X31'] = cv3 / sa3 if sa3 > 0 else 0                          # Cyst Thickness 3 (近似)
    features['X32'] = n3                                                     # Number 3
    features['X33'] = c3['Scatt_Mean'].mean() if n3 > 0 and has_scatt else 0             # OAC Mean 3 (可选)
    features['X34'] = c3['ShortAxis'].mean() if n3 > 0 and 'ShortAxis' in c3.columns else 0               # Short Axis 3
    features['X35'] = c3['Organoids_Surface'].mean() if n3 > 0 and 'Organoids_Surface' in c3.columns else 0       # Surface Area 3
    
    # Cluster 4 (Blue/D) - 极小致密损伤态
    c4 = wdf[wdf['Cluster'] == 3]
    n4 = len(c4)
    features['X41'] = c4['LongAxis'].mean() if n4 > 0 and 'LongAxis' in c4.columns else 0                # Long Axis 4
    features['X42'] = c4['Scatt_Mean'].mean() if n4 > 0 and has_scatt else 0             # OAC Mean 4 (可选)
    features['X43'] = c4['ShortAxis'].mean() if n4 > 0 and 'ShortAxis' in c4.columns else 0              # Short Axis 4
    
    return features

# 为每个well-day提取特征
feature_rows = []

# 直接从df_all中获取所有well-day组合
if len(df_all) > 0:
    well_day_combos = df_all.groupby(['_well_id', '_day']).size().reset_index()
    print(f'\n  发现 {len(well_day_combos)} 个well-day组合')
    
    for _, row in well_day_combos.iterrows():
        wid = row['_well_id']
        day_val = row['_day']
        
        feat_row = extract_paper_features_for_well(df_all, wid, day_val)
        feat_row['_well_id'] = wid
        feat_row['_day'] = 'D1' if day_val == '20241101' else 'D2'
        feature_rows.append(feat_row)
    
    print(f'  提取了 {len(feature_rows)} 条特征记录')

feat_df = pd.DataFrame(feature_rows)
print(f'\n提取了 {len(feat_df)} 条记录')
if len(feat_df) > 0:
    print(f'  包含wells: {sorted(feat_df["_well_id"].unique())}')

# ============================================================
# 5. 用论文公式计算 P1-P4 和 F
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
    return sum(coef_dict[f] * row[f] for f in feat_names)

def calc_composite_score(row):
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
# 6. 计算 ΔF 并匹配ATP
# ============================================================
d1_scores = score_df[score_df['_day'] == 'D1'][['_well_id', 'F', 'P1', 'P2', 'P3', 'P4']].copy()
d1_scores.columns = [' _well_id', 'F_D1', 'P1_D1', 'P2_D1', 'P3_D1', 'P4_D1']

d2_scores = score_df[score_df['_day'] == 'D2'][['_well_id', 'F', 'P1', 'P2', 'P3', 'P4']].copy()
d2_scores.columns = [' _well_id', 'F_D2', 'P1_D2', 'P2_D2', 'P3_D2', 'P4_D2']

merged = d1_scores.merge(d2_scores, on='_well_id')
merged['Delta_F'] = merged['F_D2'] - merged['F_D1']
merged['Delta_P1'] = merged['P1_D2'] - merged['P1_D1']
merged['Delta_P2'] = merged['P2_D2'] - merged['P2_D1']
merged['Delta_P3'] = merged['P3_D2'] - merged['P3_D1']
merged['Delta_P4'] = merged['P4_D2'] - merged['P4_D1']

# 匹配ATP
merged['ATP'] = merged[' _well_id'].map(atp_dict)

valid = merged['ATP'].notna()
print(f'\n{"="*70}')
print(f'  验证结果')
print(f'{"="*70}')
print(f'总wells: {len(merged)}')
print(f'有ATP的wells: {valid.sum()}')

if valid.sum() >= 3:
    # 方法1: Delta_F vs ATP (论文方法)
    r_delta, p_delta = pearsonr(merged.loc[valid,'Delta_F'], merged.loc[valid,'ATP'])
    sp_delta, spp_delta = spearmanr(merged.loc[valid,'Delta_F'], merged.loc[valid,'ATP'])
    
    # 方法2: F_D2 vs ATP (终点得分)
    r_f2, p_f2 = pearsonr(merged.loc[valid,'F_D2'], merged.loc[valid,'ATP'])
    sp_f2, spp_f2 = spearmanr(merged.loc[valid,'F_D2'], merged.loc[valid,'ATP'])
    
    # 方法3: F_D1 vs ATP (基线得分)
    r_f1, p_f1 = pearsonr(merged.loc[valid,'F_D1'], merged.loc[valid,'ATP'])
    
    print(f'\n{"="*80}')
    print(f'  相关性分析结果 (论文声称 r = 0.887)')
    print(f'{"="*80}')
    print(f'{"方法":25s} {"Pearson r":>10s} {"p-value":>12s} {"Spearman ρ":>12s} {"结论":>20s}')
    print(f'{"-"*80}')
    
    result_delta = '✓ 接近论文' if abs(r_delta - 0.887) < 0.1 else ('△ 有差距' if abs(r_delta - 0.887) < 0.2 else '✗ 差距大')
    result_f2 = '✓ 接近论文' if abs(r_f2 - 0.887) < 0.1 else ('△ 有差距' if abs(r_f2 - 0.887) < 0.2 else '✗ 差距大')
    
    print(f'{"ΔF vs ATP":25s} {r_delta:>10.4f} {p_delta:>12.2e} {sp_delta:>12.4f} {result_delta:>20s}')
    print(f'{"F_D2 (终点)":25s} {r_f2:>10.4f} {p_f2:>12.2e} {sp_f2:>12.4f} {result_f2:>20s}')
    print(f'{"F_D1 (基线)":25s} {r_f1:>10.4f} {p_f1:>12.2e} {"-":>12s} {"-":>20s}')
    
    gap_delta = abs(r_delta - 0.887)
    gap_f2 = abs(r_f2 - 0.887)
    
    print(f'\n{"="*80}')
    print(f'  与论文声称 (r=0.887) 的差距分析')
    print(f'{"="*80}')
    print(f'  ΔF: |{r_delta:.4f} - 0.887| = {gap_delta:.4f}')
    print(f'  F_D2: |{r_f2:.4f} - 0.887| = {gap_f2:.4f}')
    
    if gap_delta < 0.05:
        verdict = '✅ 论文结果完全可信！误差<0.05'
    elif gap_delta < 0.15:
        verdict = '⚠️ 论文结果基本可信，误差在合理范围(±0.15)'
    elif gap_delta < 0.30:
        verdict = '❌ 论文结果存疑，误差较大(±0.30)'
    else:
        verdict = '❌ 论文结果不可信！误差超过±0.30'
    
    print(f'\n  最终判定: {verdict}')

else:
    print('\n有效样本不足(<3)，无法计算相关性')

# ============================================================
# 7. 详细输出每个well的结果
# ============================================================
print(f'\n{"="*100}')
print(f'  每孔详细数据 ({valid.sum()} 有效样本)')
print(f'{"="*100}')
if valid.sum() > 0:
    detail = merged.loc[valid].copy()
    detail['ΔF_norm'] = (detail['Delta_F'] - detail['Delta_F'].min()) / (detail['Delta_F'].max() - detail['Delta_F'].min())
    detail['ATP_norm'] = (detail['ATP'] - detail['ATP'].min()) / (detail['ATP'].max() - detail['ATP'].min())
    
    cols_show = [' _well_id', 'F_D1', 'F_D2', 'Delta_F', 'ATP']
    print(detail[cols_show].to_string(index=False))
    
    print(f'\n按药物分组统计:')
    # 尝试根据well ID推断药物分组
    drug_groups = {
        '5-FU': ['C3', 'C4', 'C5'],
        '吉西他滨+5-FU': ['C8', 'C11', 'C12'],
        '顺铂+吉西他滨': ['D4', 'D5', 'D6'],
        '奥沙利铂+5-FU': ['F2', 'F3', 'F4'],
        '奥沙利铂+吉西他滨': ['E4', 'E5', 'E6'],
        'Control': ['F7', 'F8', 'F9']
    }
    
    for drug, wells_in_group in drug_groups.items():
        group_data = detail[detail[' _well_id'].isin(wells_in_group)]
        if len(group_data) > 0:
            r_g, p_g = pearsonr(group_data['Delta_F'], group_data['ATP'])
            print(f'  {drug:20s}: n={len(group_data)}, r={r_g:.4f}, mean_ΔF={group_data["Delta_F"].mean():.2f}, '
                  f'mean_ATP={group_data["ATP"].mean():.0f}')

# 保存完整结果
output_file = r'D:\Desktop\music\measure\GC032_paper_verification.xlsx'
merged.to_excel(output_file, index=False)
print(f'\n完整结果已保存: {output_file}')

print('\nDONE')