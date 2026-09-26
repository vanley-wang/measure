# ============================================================
# 新数据预测 Demo
# 证明: 只需加载已有模型做 predict(), 不需要 retrain
# 直接从原始类器官数据 → Score, 验证与之前结果一致
# ============================================================
import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'Data', 'FXN_2023_new（ICC）')
DAY3_DIR = os.path.join(DATA_DIR, 'FXN_20230701', 'measure_excel')
DAY5_DIR = os.path.join(DATA_DIR, 'FXN_20230703', 'measure_excel')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

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

CLUSTER_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD'
]
NUMERIC_MAP = {3: 0, 5: 0, 1: 1, 0: 2, 4: 2, 2: 3}
FEATURE_LONG = {
    'Organoids_Volume': 'Volume_Avg', 'Organoids_Volume_Fill': 'Volume_Fill_Avg',
    'Organoids_Surface': 'Surface_Avg', 'Cavity_Volume': 'Cavity_Volume_All',
    'CavityNum': 'CavityNum_Avg', 'LongAxis': 'Long_Axis_Avg',
    'ShortAxis': 'Short_Axis_Avg', 'Wall_Thickness': 'Cyst_Thick_Avg',
    'Sphericity': 'Sphericity_Avg', 'Scatt_Mean': 'Scatt_Mean_Avg',
    'Scatt_STD': 'Scatt_STD_Avg',
}
N_MERGED = 4


def main():
    # ================================================================
    # Step A: 加载所有已训练模型 (只加载，不训练)
    # ================================================================
    print('Loading saved models...')

    with open(os.path.join(MODEL_DIR, 'kmeans_k6.pkl'), 'rb') as f:
        kmeans = pickle.load(f)       # K=6 聚类模型
    with open(os.path.join(MODEL_DIR, 'scaler_k6.pkl'), 'rb') as f:
        clus_scaler = pickle.load(f)  # 聚类用的标准化器
    with open(os.path.join(MODEL_DIR, 'paper_delta_deploy.pkl'), 'rb') as f:
        deploy = pickle.load(f)       # 完整部署包

    pca_scaler = deploy['scaler']     # Delta特征的标准化器
    pca_model = deploy['pca']         # PCA模型 (只是用于验证公式)
    beta = deploy['beta']             # 综合权重 β
    delta_features = deploy['features']  # 需要的14个特征名

    print(f'  K-means K={kmeans.n_clusters}')
    print(f'  Delta features needed: {len(delta_features)}')
    print(f'  Beta weights: {len(beta)}')

    # ================================================================
    # Step B: 加载新数据 (Demo 用已有数据模拟)
    # ================================================================
    print('\nLoading raw organoid data...')
    wells_d3, wells_d5 = {}, {}
    for dname in os.listdir(DAY3_DIR):
        if not dname.endswith('.xlsx'): continue
        wid = dname.replace('.xlsx', '').split('_')[0]
        wells_d3[wid] = pd.read_excel(os.path.join(DAY3_DIR, dname))
    for dname in os.listdir(DAY5_DIR):
        if not dname.endswith('.xlsx'): continue
        wid = dname.replace('.xlsx', '').split('_')[0]
        wells_d5[wid] = pd.read_excel(os.path.join(DAY5_DIR, dname))

    common = sorted(set(wells_d3) & set(wells_d5))
    print(f'  Wells: {len(common)}')

    # ================================================================
    # Step C: 对每个孔 → 已训练模型 predict() → 聚合 → Delta → Score
    # ================================================================
    print('\nPredicting per well (using saved models only, no retrain)...')

    scores_direct = {}   # 直接用 β 公式
    scores_via_pca = {}  # 走 PCA 路径 (验证公式等价)

    for wid in common:
        # C1: 聚类 predict (不是训练!)
        X3 = wells_d3[wid][CLUSTER_FEATURES].fillna(0).values
        X5 = wells_d5[wid][CLUSTER_FEATURES].fillna(0).values
        k6_3 = kmeans.predict(clus_scaler.transform(X3))
        k6_5 = kmeans.predict(clus_scaler.transform(X5))
        c3 = np.array([NUMERIC_MAP[l] for l in k6_3])
        c5 = np.array([NUMERIC_MAP[l] for l in k6_5])

        # C2: 逐表型均值聚合 (Cavity_Volume用sum，其余用mean)
        def agg(df, labels):
            rec = {}
            for cl in range(N_MERGED):
                mask = labels == cl
                n = mask.sum()
                rec[f'Number_{cl+1}'] = n
                sub = df.loc[mask] if n > 0 else df.iloc[:0]
                for src, alias in FEATURE_LONG.items():
                    if src not in df.columns: continue
                    if src == 'Cavity_Volume':
                        rec[f'{alias}_{cl+1}'] = float(sub[src].sum())
                    else:
                        rec[f'{alias}_{cl+1}'] = float(sub[src].mean()) if n > 0 else 0.0
            return rec

        feats_d3 = agg(wells_d3[wid], c3)
        feats_d5 = agg(wells_d5[wid], c5)

        # C3: Delta = Day5 − Day3, 取14个特征
        delta_vec = []
        for f in delta_features:
            d5_val = feats_d5.get(f, 0.0)
            d3_val = feats_d3.get(f, 0.0)
            delta_vec.append(d5_val - d3_val)
        delta_vec = np.array(delta_vec)

        # C4: Score 计算 — 方法一: 直接 β 公式 (推荐)
        x_std = pca_scaler.transform(delta_vec.reshape(1, -1))[0]
        score_direct = float(np.dot(beta, x_std))
        scores_direct[wid] = score_direct

        # C5: Score 计算 — 方法二: 走 PCA 路径 (验证用)
        pc = pca_model.transform(delta_vec.reshape(1, -1))[0]
        score_pca = float(np.dot(deploy['weights'], pc))
        scores_via_pca[wid] = score_pca

    # ================================================================
    # Step D: 验证
    # ================================================================
    print('\nVerification:')
    atp_list, score_list = [], []
    for wid in common:
        if wid in ATP_DB:
            atp_list.append(ATP_DB[wid])
            score_list.append(scores_direct[wid])

    r, p = pearsonr(score_list, atp_list)
    print(f'  Direct β formula:  r={r:.4f}, p={p:.6f}')

    # 验证两种方法等价
    diffs = [abs(scores_direct[w] - scores_via_pca[w]) for w in common]
    print(f'  β vs PCA path diff: max={max(diffs):.2e}  (should be ~0)')

    # 和之前的对比
    from scipy.stats import spearmanr
    rho, _ = spearmanr(score_list, atp_list)
    prev_search_best = pd.read_excel(
        os.path.join(OUTPUT_DIR, 'delta_search_Paper_Delta_(K6\u21924_mean).xlsx')).iloc[0]
    print(f'\n  Comparison:')
    print(f'    Previous search best: |r|={prev_search_best["|r|"]:.4f}')
    print(f'    Current predict:      |r|={abs(r):.4f}, rho={rho:.4f}')

    # ================================================================
    # Summary
    # ================================================================
    print(f'\n{"="*60}')
    print(f'  Prediction Recap')
    print(f'{"="*60}')
    print(f'''
    For each new well:
      1. Load kmeans_k6.pkl, scaler_k6.pkl  (clustering model)
      2. Load paper_delta_deploy.pkl        (PCA + beta + scaler)
      3. For raw organoids:
         a. X_std = clus_scaler.transform(raw_features)
         b. labels_k6 = kmeans.predict(X_std)     ← predict, not train!
         c. labels_k4 = NUMERIC_MAP[labels_k6]
         d. Per-cluster mean aggregation
      4. Delta = Day5_aggregated − Day3_aggregated (14 features)
      5. delta_std = pca_scaler.transform([delta])
      6. Score = dot(beta, delta_std)     ← one line!
      7. Correlate Score with ATP
    ''')


if __name__ == '__main__':
    main()