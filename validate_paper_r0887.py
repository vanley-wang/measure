"""
验证论文中声称的Scores模型泛化能力（r=0.887）

论文声称：
- 在5-氟尿嘧啶、吉西他滨、顺铂等一线化疗药物及多药联合方案的独立测试中，
- 模型依然保持了极高的一致性（r=0.887），证明该模型拥有良好的泛化能力

本脚本将：
1. 使用ICC（肝内胆管癌）数据训练PCA模型
2. 在GC（胃癌）独立测试集上验证
3. 复现并验证r=0.887这个结果
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def load_atp_data(atp_file):
    """加载ATP数据"""
    print(f"\n{'='*70}")
    print(f"加载ATP数据: {atp_file}")
    print(f"{'='*70}")
    
    df = pd.read_excel(atp_file)
    print(f"ATP数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    atp_dict = {}
    if len(df.columns) >= 2:
        atp_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        print(f"成功加载 {len(atp_dict)} 个样本的ATP值")
        
    return atp_dict


def load_organoid_features(data_dir, day='0701'):
    """加载类器官特征数据"""
    print(f"\n{'='*70}")
    print(f"加载{day}特征数据")
    print(f"{'='*70}")
    
    measure_dir = os.path.join(data_dir, f'FXN_20230701', 'measure_excel')
    if not os.path.exists(measure_dir):
        measure_dir = os.path.join(data_dir, 'measure_excel')
    
    if not os.path.exists(measure_dir):
        raise FileNotFoundError(f"找不到数据目录: {measure_dir}")
    
    all_dfs = []
    xlsx_files = [f for f in os.listdir(measure_dir) if f.endswith('.xlsx') and day in f]
    
    print(f"找到 {len(xlsx_files)} 个xlsx文件")
    
    for fname in sorted(xlsx_files):
        fpath = os.path.join(measure_dir, fname)
        try:
            tmp = pd.read_excel(fpath)
            well_id = fname.replace(f'_{day}.xlsx', '').replace('.xlsx', '')
            
            required_cols = [
                'Organoids_Volume_Fill', 'Organoids_Volume_Solid',
                'Organoids_Surface_Area', 'Sphericity', 'Roughness',
                'Scatt_Mean', 'Scatt_Std'
            ]
            
            available_cols = [c for c in required_cols if c in tmp.columns]
            
            if len(available_cols) >= 5:
                tmp['_well_id'] = well_id
                tmp['_well'] = fname.replace('.xlsx', '')
                tmp['_day'] = day
                all_dfs.append(tmp)
                
        except Exception as e:
            print(f"  跳过 {fname}: {e}")
    
    if not all_dfs:
        raise RuntimeError("没有有效的数据文件")
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"总共加载 {len(df)} 条记录")
    print(f"Well IDs: {sorted(df['_well_id'].unique())}")
    
    return df


def extract_cluster_statistics(df):
    """提取聚类统计特征（模拟K-means++聚类后的统计）"""
    print(f"\n{'='*70}")
    print("提取聚类统计特征")
    print(f"{'='*70}")
    
    morph_features = [
        'Organoids_Volume_Fill', 'Organoids_Volume_Solid',
        'Organoids_Surface_Area', 'Sphericity', 'Roughness',
        'Scatt_Mean', 'Scatt_Std'
    ]
    
    available_feats = [f for f in morph_features if f in df.columns]
    print(f"可用特征 ({len(available_feats)}个): {available_feats}")
    
    stats_list = []
    
    for (well_id, day), group in df.groupby(['_well_id', '_day']):
        n_total = len(group)
        
        stats = {'Well_ID': well_id, '_day': day, 'N_Organoids': n_total}
        
        for feat in available_feats:
            values = group[feat].dropna()
            if len(values) > 0:
                stats[f'{feat}_Mean'] = values.mean()
                stats[f'{feat}_Std'] = values.std()
                stats[f'{feat}_Median'] = values.median()
                
                healthy_threshold = values.quantile(0.6)
                healthy_values = values[values >= healthy_threshold]
                if len(healthy_values) > 0:
                    stats[f'Healthy_{feat}_Mean'] = healthy_values.mean()
                    stats[f'Healthy_{feat}_Median'] = healthy_values.median()
        
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    print(f"生成统计特征矩阵: {stats_df.shape}")
    
    return stats_df, available_feats


def build_composite_features(stats_df, features, day3='0701', day5='0703'):
    """构建综合特征（Day3-Day5差值等）"""
    print(f"\n{'='*70}")
    print("构建综合特征（相对变化量）")
    print(f"{'='*70}")
    
    d3_data = stats_df[stats_df['_day'] == day3].copy()
    d5_data = stats_df[stats_df['_day'] == day5].copy()
    
    common_wells = set(d3_data['Well_ID']) & set(d5_data['Well_ID'])
    print(f"共同Well ID数量: {len(common_wells)}")
    
    rows = []
    for wid in sorted(common_wells):
        row = {'Well_ID': wid}
        d3_row = d3_data[d3_data['Well_ID'] == wid].iloc[0]
        d5_row = d5_data[d5_data['Well_ID'] == wid].iloc[0]
        
        for feat in features:
            d3_val = d3_row.get(f'Healthy_{feat}_Median', d3_row.get(f'{feat}_Median'))
            d5_val = d5_row.get(f'Healthy_{feat}_Median', d5_row.get(f'{feat}_Median'))
            
            if pd.notna(d3_val) and pd.notna(d5_val):
                row[f'Delta_{feat}'] = d5_val - d3_val
                
                if abs(d3_val) > 1e-10:
                    row[f'RelDelta_{feat}'] = (d5_val - d3_val) / abs(d3_val)
                    
                row[f'D5_{feat}'] = d5_val
                row[f'D3_{feat}'] = d3_val
        
        rows.append(row)
    
    feature_df = pd.DataFrame(rows)
    
    delta_feats = [f'Delta_{f}' for f in features]
    reldelta_feats = [f'RelDelta_{f}' for f in features]
    d5_feats = [f'D5_{f}' for f in features]
    
    all_feature_cols = []
    for feat_group in [delta_feats, reldelta_feats, d5_feats]:
        for f in feat_group:
            if f in feature_df.columns and f not in all_feature_cols:
                all_feature_cols.append(f)
    
    print(f"构建特征维度: {len(all_feature_cols)}")
    
    return feature_df, all_feature_cols


def train_pca_model(X_train, n_components=4, var_threshold=0.70):
    """训练PCA模型（按照论文的方法）"""
    print(f"\n{'='*70}")
    print("训练PCA模型")
    print(f"{'='*70}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    max_comp = min(X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca_full = PCA(n_components=max_comp, random_state=42)
    pca_full.fit(X_scaled)
    
    eigenvalues = pca_full.explained_variance_
    explained_var_ratio = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)
    
    print(f"\n主成分分析结果:")
    print(f"{'PC':<5} {'特征值':>10} {'方差贡献率':>12} {'累计方差贡献率':>15}")
    print("-" * 50)
    
    for i in range(min(len(eigenvalues), 10)):
        kaiser_mark = " ✓" if eigenvalues[i] > 1.0 else ""
        cumvar_mark = f" ← >{var_threshold:.0%}" if cumulative_var[i] >= var_threshold else ""
        print(f"PC{i+1:<4} {eigenvalues[i]:>10.4f} {explained_var_ratio[i]:>11.2%} {cumulative_var[i]:>14.2%}{kaiser_mark}{cumvar_mark}")
    
    n_comp_kaiser = max(2, int(np.sum(eigenvalues > 1.0)))
    n_comp_cumvar = int(np.searchsorted(cumulative_var, var_threshold)) + 1
    
    n_components_final = max(n_components, n_comp_kaiser, n_comp_cumvar)
    n_components_final = min(n_components_final, max_comp)
    
    print(f"\n选择标准:")
    print(f"  Kaiser准则 (特征值>1): {n_comp_kaiser} PCs")
    print(f"  累计方差>{var_threshold:.0%}: {n_comp_cumvar} PCs")
    print(f"  论文指定: {n_components} PCs")
    print(f"  最终选择: {n_components_final} PCs")
    
    pca = PCA(n_components=n_components_final, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    vr = pca.explained_variance_ratio_
    wts = vr / vr.sum()  # 权重 = 方差贡献率 / 总方差贡献率
    
    print(f"\n权重计算（按论文公式4.5-4.6）:")
    for i, (v, w) in enumerate(zip(vr, wts)):
        print(f"  w_{i+1} = θ_{i+1}/Σθ = {v:.4f}/{vr.sum():.4f} = {w:.4f}")
    
    score = np.dot(X_pca, wts)
    
    return pca, scaler, wts, score, X_pca, X_scaled


def validate_on_test_set(pca, scaler, wts, X_test, test_ids, atp_dict):
    """在测试集上验证模型"""
    print(f"\n{'='*70}")
    print("在独立测试集上验证模型")
    print(f"{'='*70}")
    
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    test_score = np.dot(X_test_pca, wts)
    
    valid_mask = []
    valid_atp = []
    valid_score = []
    valid_ids = []
    
    for i, wid in enumerate(test_ids):
        if wid in atp_dict and pd.notna(test_score[i]):
            atp_val = atp_dict[wid]
            if pd.notna(atp_val) and atp_val > 0:
                valid_mask.append(True)
                valid_atp.append(atp_val)
                valid_score.append(test_score[i])
                valid_ids.append(wid)
            else:
                valid_mask.append(False)
        else:
            valid_mask.append(False)
    
    valid_atp = np.array(valid_atp)
    valid_score = np.array(valid_score)
    
    print(f"\n测试集统计:")
    print(f"  总样本数: {len(test_ids)}")
    print(f"  有效匹配数: {len(valid_ids)}")
    
    if len(valid_ids) < 3:
        print("\n❌ 错误：有效样本数不足！")
        return None, None
    
    pr, pp = pearsonr(valid_score, valid_atp)
    sp, spp = spearmanr(valid_score, valid_atp)
    
    print(f"\n{'='*50}")
    print("验证结果：Score vs ATP 相关性")
    print(f"{'='*50}")
    print(f"N = {len(valid_ids)}")
    print(f"Pearson r  = {pr:.6f} (p = {pp:.2e})")
    print(f"Spearman ρ = {sp:.6f} (p = {spp:.2e})")
    
    print(f"\n{'='*50}")
    print("与论文声称的 r=0.887 对比")
    print(f"{'='*50}")
    paper_r = 0.887
    diff = abs(pr) - paper_r
    pct_diff = diff / paper_r * 100
    
    print(f"论文声称: r = {paper_r:.3f}")
    print(f"本次复现: r = {abs(pr):.6f}")
    print(f"差异: Δr = {diff:+.6f} ({pct_diff:+.2f}%)")
    
    if abs(diff) < 0.05:
        print(f"\n✅ 验证通过！差异在±0.05范围内")
    elif abs(diff) < 0.10:
        print(f"\n⚠ 基本一致，差异在±0.10范围内")
    else:
        print(f"\n❌ 差异较大，需要进一步调查原因")
    
    result_df = pd.DataFrame({
        'Well_ID': valid_ids,
        'Score': valid_score,
        'ATP': valid_atp
    })
    
    res = {
        'pearson_r': pr,
        'pearson_p': pp,
        'spearman_rho': sp,
        'spearman_p': spp,
        'n_samples': len(valid_ids),
        'paper_r': paper_r,
        'diff_from_paper': diff
    }
    
    return result_df, res


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" " * 20 + "论文 Scores 模型验证")
    print(" " * 15 + "验证目标：独立测试集相关性 r = 0.887")
    print("=" * 80)
    
    base_dir = r'D:\Desktop\music\measure'
    os.chdir(base_dir)
    
    icc_data_dir = os.path.join('Data', 'FXN_2023_new（ICC）')
    gc_data_dir = os.path.join('Data', 'FXN_2023_new（GC）')
    
    icc_atp_file = os.path.join(icc_data_dir, 'FXN_2023_PCA全部.xlsx')
    gc_atp_file = os.path.join(gc_data_dir, 'ATP.xlsx')
    
    print(f"\n项目目录: {base_dir}")
    print(f"ICC训练数据: {icc_data_dir}")
    print(f"GC测试数据: {gc_data_dir}")
    
    try:
        icc_atp_dict = load_atp_data(icc_atp_file)
        gc_atp_dict = load_atp_data(gc_atp_file)
        
        print(f"\n{'='*70}")
        print("步骤1: 加载ICC训练数据")
        print(f"{'='*70}")
        
        icc_df_d3 = load_organoid_features(icc_data_dir, day='0701')
        icc_df_d5 = load_organoid_features(icc_data_dir, day='0703')
        icc_df = pd.concat([icc_df_d3, icc_df_d5], ignore_index=True)
        
        icc_stats, features = extract_cluster_statistics(icc_df)
        icc_feature_df, feature_cols = build_composite_features(icc_stats, features)
        
        icc_feature_df['ATP'] = icc_feature_df['Well_ID'].map(icc_atp_dict)
        
        print(f"\n{'='*70}")
        print("步骤2: 训练PCA模型（使用ICC数据）")
        print(f"{'='*70}")
        
        X_train = icc_feature_df[feature_cols].values
        train_ids = icc_feature_df['Well_ID'].values
        
        valid_train = ~np.isnan(X_train).any(axis=1) & icc_feature_df['ATP'].notna()
        X_train_clean = X_train[valid_train]
        train_ids_clean = train_ids[valid_train]
        train_atp = icc_feature_df.loc[valid_train, 'ATP'].values
        
        print(f"训练集: {X_train_clean.shape[0]} 样本, {X_train_clean.shape[1]} 特征")
        
        pca, scaler, wts, train_score, X_train_pca, X_train_scaled = train_pca_model(
            X_train_clean, n_components=4, var_threshold=0.70
        )
        
        train_pr, train_pp = pearsonr(train_score, train_atp)
        print(f"\n训练集相关性: Pearson r = {train_pr:.6f} (p = {train_pp:.2e})")
        
        print(f"\n{'='*70}")
        print("步骤3: 在GC独立测试集上验证")
        print(f"{'='*70}")
        
        gc_df_d3 = load_organoid_features(gc_data_dir, day='0701')
        gc_stats_gc, _ = extract_cluster_statistics(gc_df_d3)
        
        gc_feature_df, _ = build_composite_features(gc_stats_gc, features)
        gc_feature_df['ATP'] = gc_feature_df['Well_ID'].map(gc_atp_dict)
        
        X_test = gc_feature_df[feature_cols].values
        test_ids = gc_feature_df['Well_ID'].values
        
        valid_test = ~np.isnan(X_test).any(axis=1)
        X_test_clean = X_test[valid_test]
        test_ids_clean = test_ids[valid_test]
        
        print(f"测试集: {X_test_clean.shape[0]} 样本")
        
        result_df, validation_res = validate_on_test_set(
            pca, scaler, wts, X_test_clean, test_ids_clean, gc_atp_dict
        )
        
        if result_df is not None:
            print(f"\n{'='*70}")
            print("验证完成!")
            print(f"{'='*70}")
            
            output_file = 'validation_r0887_result.csv'
            result_df.to_csv(output_file, index=False)
            print(f"结果已保存到: {output_file}")
            
            print(f"\n最终结论:")
            if validation_res and abs(validation_res['diff_from_paper']) < 0.05:
                print(f"✅ 论文结果可信！复现的 r={abs(validation_res['pearson_r']):.4f} 与论文声称的 r=0.887 高度一致")
            elif validation_res and abs(validation_res['diff_from_paper']) < 0.10:
                print(f"⚠ 论文结果基本可信，复现的 r={abs(validation_res['pearson_r']):.4f} 与论文声称的 r=0.887 存在小幅差异")
            else:
                print(f"❌ 论文结果存疑！复现的 r={abs(validation_res['pearson_r']):.4f} 与论文声称的 r=0.887 差异较大")
        
        return validation_res
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print("请确认数据文件路径正确")
        return None
    except Exception as e:
        print(f"\n❌ 发生错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    result = main()
    
    if result is not None:
        print(f"\n{'='*80}")
        print("验证总结")
        print(f"{'='*80}")
        print(f"论文声称的独立测试相关性: r = 0.887")
        print(f"本次复现的相关性: r = {abs(result['pearson_r']):.6f}")
        print(f"差异: Δr = {result['diff_from_paper']:+.6f}")
        print(f"样本数: N = {result['n_samples']}")
        print(f"P值: p = {result['pearson_p']:.2e}")