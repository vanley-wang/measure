"""
Re-run MedNext pipeline steps 5-6, 9-10 (skipping roughness steps 7-8).
Uses the new cavity-boosted KMeans model.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

BASE_DIR = "MedNext_GC032"
BATCHES = ["20230701", "20230703"]
MIN_VOLUME = 50

# Import pipeline module and set globals
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pipeline", os.path.join(REPO_ROOT, "run_mednext_gc032_pipeline.py")
)
pipeline = importlib.util.module_from_spec(spec)
pipeline.BASE_DIR = BASE_DIR
pipeline.BATCHES = BATCHES
pipeline.MIN_VOLUME = MIN_VOLUME
spec.loader.exec_module(pipeline)

import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("MedNext_GC032 — 重跑 step5/6/9/10（跳过粗糙度）")
    print("=" * 60)

    # Step 5: cluster_analysis_1 (孔板级汇总)
    pipeline.step5_cluster_analysis_1()

    # Step 6: cluster_analysis_2 (主表汇总)
    pipeline.step6_cluster_analysis_2()

    # Step 9: PCA (需要修改特征列表，去掉 Roughness_All)
    print("\n" + "=" * 60)
    print("Step 9: PCA 综合评分（不含粗糙度）")
    print("=" * 60)
    _run_pca_no_roughness()

    # Step 10: Feature Optimization (去掉 Roughness_All)
    print("\n" + "=" * 60)
    print("Step 10: 特征优化（不含粗糙度）")
    print("=" * 60)
    _run_feature_opt_no_roughness()

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)


def _run_pca_no_roughness():
    """Run PCA without roughness features."""
    from sklearn.decomposition import PCA as SKPCA
    from sklearn.preprocessing import StandardScaler
    import re

    input_path = os.path.join(BASE_DIR, "MedNext_GC032_Analysis.xlsx")
    details_path = os.path.join(BASE_DIR, "MedNext_GC032_PCA_Model_Details.xlsx")
    result_path = os.path.join(BASE_DIR, "MedNext_GC032_PCA_Result.xlsx")

    Data_All = pd.read_excel(input_path)
    print(f"  输入: {input_path} ({Data_All.shape[0]} 行)")

    # 4 clusters × 9 features each = 36 candidate features
    # (Number, Volume_Fill, Surface, Cavity_Volume, LongAxis, ShortAxis,
    #  Cyst_Thick, Sphericity, Scatt_Mean, Scatt_STD — actually 10 per cluster)
    candidate_features = []
    for ci in range(1, 5):
        for feat in ['Number', 'Volume_Fill_Avg', 'Surface_Avg',
                     'Cavity_Volume_All', 'Long_Axis_Avg', 'Short_Axis_Avg',
                     'Cyst_Thick_Avg', 'Sphericity_Avg',
                     'Scatt_Mean_Avg', 'Scatt_STD_Avg']:
            col = f'{feat}_{ci}'
            if col in Data_All.columns:
                candidate_features.append(col)

    print(f"  候选特征: {len(candidate_features)} 个")

    # Use the same features as before minus Roughness_All
    # Actually just use all available cluster stats features
    features_list = candidate_features

    def get_sort_key(f):
        match = re.search(r'_(\d+)$', f)
        if match:
            return (int(match.group(1)), 0, f)
        return (99, 0, f)

    features_list.sort(key=get_sort_key)

    Data = Data_All[features_list].fillna(0)

    # Standardize
    scaler = StandardScaler()
    Data_std = scaler.fit_transform(Data)

    # PCA
    n_components = min(4, len(features_list))
    pca = SKPCA(n_components=n_components, svd_solver="full", random_state=42)
    Data_Pca = pca.fit_transform(Data_std)

    variance_ratio = pca.explained_variance_ratio_
    weights = variance_ratio / np.sum(variance_ratio)
    result_scores = np.dot(Data_Pca, weights)

    # Save PCA details
    details_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(n_components)],
        'Explained_Variance_Ratio': variance_ratio,
        'Weight': weights,
        'Cumulative_Ratio': np.cumsum(variance_ratio),
    })

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features_list,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    with pd.ExcelWriter(details_path, engine='openpyxl') as writer:
        details_df.to_excel(writer, sheet_name='Variance', index=False)
        loadings.to_excel(writer, sheet_name='Loadings')
    print(f"  PCA 详情: {details_path}")

    # Save results
    df_result = Data_All[['Name']].copy()
    for i in range(n_components):
        df_result[f'PC{i+1}'] = Data_Pca[:, i]
    df_result['Result'] = result_scores

    df_result.to_excel(result_path, index=False)
    print(f"  PCA 结果: {result_path} ({len(df_result)} 行)")
    print(f"  方差解释率: {', '.join(f'{100*v:.1f}%' for v in variance_ratio)}")
    print(f"  累计: {100*sum(variance_ratio):.1f}%")


def _run_feature_opt_no_roughness():
    """Run feature optimization without roughness."""
    from sklearn.decomposition import PCA as SKPCA
    from sklearn.preprocessing import StandardScaler
    from joblib import Parallel, delayed
    import random
    import time

    input_path = os.path.join(BASE_DIR, "MedNext_GC032_Analysis.xlsx")
    log_save_path = os.path.join(BASE_DIR, "Feature_Optimization_Sorted.xlsx")
    n_iterations = 500000
    n_jobs = -1

    # ATP database - read from ATP.xlsx
    atp_path = os.path.join(BASE_DIR, "ATP.xlsx")
    atp_db = {}
    if os.path.exists(atp_path):
        atp_df = pd.read_excel(atp_path)
        # Assume columns: Name, ATP or similar
        name_col = atp_df.columns[0]
        atp_col = None
        for c in atp_df.columns:
            if 'ATP' in str(c).upper() or 'atp' in str(c).lower():
                atp_col = c
                break
        if atp_col is None:
            atp_col = atp_df.columns[-1]
        for _, row in atp_df.iterrows():
            well = str(row[name_col]).split('_')[0]
            atp_val = row[atp_col]
            if pd.notna(atp_val):
                atp_db[well] = float(atp_val)

    Data_All = pd.read_excel(input_path)
    if 'Name' not in Data_All.columns:
        Data_All['Name'] = Data_All.index.astype(str)

    # Build feature pool from 4 cluster stats
    full_feature_pool = []
    for ci in range(1, 5):
        for feat in ['Number', 'Volume_Fill_Avg', 'Surface_Avg',
                     'Cavity_Volume_All', 'Long_Axis_Avg', 'Short_Axis_Avg',
                     'Cyst_Thick_Avg', 'Sphericity_Avg',
                     'Scatt_Mean_Avg', 'Scatt_STD_Avg']:
            col = f'{feat}_{ci}'
            if col in Data_All.columns:
                full_feature_pool.append(col)

    available_pool = [f for f in full_feature_pool if f in Data_All.columns]
    available_pool.sort()
    print(f"  候选特征: {len(available_pool)} 个")

    X_matrix_all = Data_All[available_pool].fillna(0).values

    # Prepare paired data (Day3 vs Day5) with ATP
    Data_All['Well_ID'] = Data_All['Name'].apply(lambda x: x.split('_')[0])
    Data_All['TimePoint'] = Data_All['Name'].apply(lambda x: x.split('_')[1])
    Data_All['ATP'] = Data_All['Well_ID'].map(atp_db)

    start_indices = []
    end_indices = []
    valid_atp = []

    for well in Data_All['Well_ID'].unique():
        row_start = Data_All[(Data_All['Well_ID'] == well) & (Data_All['TimePoint'] == '0701')]
        row_end = Data_All[(Data_All['Well_ID'] == well) & (Data_All['TimePoint'] == '0703')]
        if not row_start.empty and not row_end.empty:
            atp = row_end.iloc[0]['ATP']
            if pd.notna(atp):
                start_indices.append(Data_All.index.get_loc(row_start.index[0]))
                end_indices.append(Data_All.index.get_loc(row_end.index[0]))
                valid_atp.append(atp)

    idx_start = np.array(start_indices)
    idx_end = np.array(end_indices)
    y_atp = np.array(valid_atp)
    atp_mean = np.mean(y_atp)
    atp_std = np.std(y_atp)

    print(f"  有效样本对: {len(y_atp)} 个")

    def run_trial(seed):
        rng = np.random.RandomState(seed)
        n_total = X_matrix_all.shape[1]
        n_select = rng.randint(4, n_total + 1)
        feat_indices = rng.choice(n_total, n_select, replace=False)
        feat_indices.sort()

        X_sub = X_matrix_all[:, feat_indices]

        try:
            sc = StandardScaler()
            X_std = sc.fit_transform(X_sub)
            pca = SKPCA(n_components=min(4, len(feat_indices)),
                        svd_solver='full', random_state=42)
            Data_Pca = pca.fit_transform(X_std)
            variance_ratio = pca.explained_variance_ratio_
            total_var = np.sum(variance_ratio)
            if total_var == 0:
                return None
            weights = variance_ratio / total_var
            result_scores = np.dot(Data_Pca, weights)
        except Exception:
            return None

        res_start = result_scores[idx_start]
        res_end = result_scores[idx_end]
        res_diff = res_end - res_start

        diff_mean = np.mean(res_diff)
        diff_std = np.std(res_diff)
        if diff_std == 0:
            return None
        cov_diff = np.mean((res_diff - diff_mean) * (y_atp - atp_mean))
        r_diff = cov_diff / (diff_std * atp_std)

        end_mean = np.mean(res_end)
        end_std = np.std(res_end)
        cov_end = np.mean((res_end - end_mean) * (y_atp - atp_mean))
        r_end = cov_end / (end_std * atp_std) if end_std > 0 else 0

        return {
            'n_feats': len(feat_indices),
            'feat_idx': feat_indices,
            'diff_r': abs(r_diff),
            'end_r': abs(r_end)
        }

    print(f"  开始 {n_iterations} 次随机搜索...")
    start_time = time.time()

    seeds = [random.randint(0, 100000000) for _ in range(n_iterations)]
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_trial)(seed) for seed in seeds
    )

    clean_results = []
    for res in results:
        if res is not None:
            names = [available_pool[i] for i in res['feat_idx']]
            clean_results.append({
                'Diff_Corr_Abs': res['diff_r'],
                'End_Corr_Abs': res['end_r'],
                'Num_Features': res['n_feats'],
                'Features_List': ", ".join(names)
            })

    df = pd.DataFrame(clean_results)
    df = df.sort_values(by='Diff_Corr_Abs', ascending=False)
    df.head(500).to_excel(log_save_path, index=False)

    elapsed = time.time() - start_time
    print(f"  完成！耗时: {elapsed:.1f}s")
    print(f"\n  TOP 3 最佳特征组合：")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"  No.{i+1}: 差值相关性 = {row['Diff_Corr_Abs']:.6f}")
        print(f"    特征数: {row['Num_Features']}")
        print(f"    特征: {row['Features_List'][:100]}...")

    print(f"\n  结果已保存: {log_save_path}")


if __name__ == "__main__":
    main()
