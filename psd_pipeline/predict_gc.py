# ============================================================
# GC 新数据预测 — 用已保存模型直接 predict
# ============================================================
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'Data', 'FXN_2023_new（GC）')
DAY3_DIR = os.path.join(GC_DIR, 'FXN_20230701', 'measure_excel')
DAY5_DIR = os.path.join(GC_DIR, 'FXN_20230703', 'measure_excel')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLUSTER_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD'
]
NUMERIC_MAP = {3: 0, 5: 0, 1: 1, 0: 2, 4: 2, 2: 3}
N_MERGED = 4

FEATURE_LONG = {
    'Organoids_Volume': 'Volume_Avg', 'Organoids_Volume_Fill': 'Volume_Fill_Avg',
    'Organoids_Surface': 'Surface_Avg', 'Cavity_Volume': 'Cavity_Volume_All',
    'CavityNum': 'CavityNum_Avg', 'LongAxis': 'Long_Axis_Avg',
    'ShortAxis': 'Short_Axis_Avg', 'Wall_Thickness': 'Cyst_Thick_Avg',
    'Sphericity': 'Sphericity_Avg', 'Scatt_Mean': 'Scatt_Mean_Avg',
    'Scatt_STD': 'Scatt_STD_Avg',
}


def main():
    # 1. 加载所有模型
    print('Loading models...')
    with open(os.path.join(MODEL_DIR, 'kmeans_k6.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler_k6.pkl'), 'rb') as f:
        clus_scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'paper_delta_deploy.pkl'), 'rb') as f:
        deploy = pickle.load(f)

    pca_scaler = deploy['scaler']
    beta = deploy['beta']
    delta_features = deploy['features']
    print(f'  Model loaded: {len(delta_features)} delta features, r_train={deploy["training_r"]:.4f}')

    # 2. 加载 GC 原始数据
    print('\nLoading GC data...')
    wells_d3, wells_d5 = {}, {}
    for fname in os.listdir(DAY3_DIR):
        if not fname.endswith('.xlsx'): continue
        wid = fname.replace('.xlsx', '').split('_')[0]
        wells_d3[wid] = pd.read_excel(os.path.join(DAY3_DIR, fname))
    for fname in os.listdir(DAY5_DIR):
        if not fname.endswith('.xlsx'): continue
        wid = fname.replace('.xlsx', '').split('_')[0]
        wells_d5[wid] = pd.read_excel(os.path.join(DAY5_DIR, fname))

    common = sorted(set(wells_d3) & set(wells_d5))
    print(f'  Paired wells: {len(common)}')

    # 3. 加载 ATP
    atp_df = pd.read_excel(os.path.join(GC_DIR, 'ATP.xlsx'))
    # ATP 文件格式: 第1列well名, 最后非空列为ATP值
    col_names = list(atp_df.columns)
    atp_col = col_names[-1]  # 最后一列
    gc_atp = {}
    for _, row in atp_df.iterrows():
        wid = str(row[col_names[0]]).split('_')[0] if '_' in str(row[col_names[0]]) else str(row[col_names[0]])
        val = row[atp_col]
        if pd.notna(val):
            gc_atp[wid] = float(val)
    print(f'  ATP entries: {len(gc_atp)}')

    # 4. 逐孔预测
    print('\nPredicting per well...')
    results = []

    for wid in common:
        # 4a. 聚类 predict (不训练!)
        X3 = wells_d3[wid][CLUSTER_FEATURES].fillna(0).values
        X5 = wells_d5[wid][CLUSTER_FEATURES].fillna(0).values
        k6_3 = kmeans.predict(clus_scaler.transform(X3))
        k6_5 = kmeans.predict(clus_scaler.transform(X5))
        c3 = np.array([NUMERIC_MAP[l] for l in k6_3])
        c5 = np.array([NUMERIC_MAP[l] for l in k6_5])

        # 4b. 逐表型均值聚合
        def per_cluster_mean(df, labels):
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

        feats_d3 = per_cluster_mean(wells_d3[wid], c3)
        feats_d5 = per_cluster_mean(wells_d5[wid], c5)

        # 4c. Delta = D5 - D3
        delta_vec = []
        for f in delta_features:
            delta_vec.append(feats_d5.get(f, 0.0) - feats_d3.get(f, 0.0))
        delta_vec = np.array(delta_vec)

        # 4d. Score = beta · X_std
        x_std = pca_scaler.transform(delta_vec.reshape(1, -1))[0]
        score = float(np.dot(beta, x_std))

        results.append({
            'Well_ID': wid,
            'Score': score,
            'ATP': gc_atp.get(wid, np.nan),
            'ATP_Log': np.log10(gc_atp.get(wid, np.nan)) if gc_atp.get(wid, np.nan) else np.nan
        })

    df_result = pd.DataFrame(results).dropna(subset=['ATP'])
    print(f'  Matched wells: {len(df_result)}')

    # 5. 相关性
    if len(df_result) >= 5:
        r, p = pearsonr(df_result['Score'], df_result['ATP'])
        rho, p_s = spearmanr(df_result['Score'], df_result['ATP'])
        print(f'\n{"="*50}')
        print(f'  GC Prediction Result')
        print(f'{"="*50}')
        print(f'  Wells: {len(df_result)}')
        print(f'  Pearson r  = {r:.4f}  (p={p:.6f})')
        print(f'  Spearman rho = {rho:.4f}  (p={p_s:.6f})')
        print(f'  Training r (ICC) = {deploy["training_r"]:.4f}')
        print(f'  Delta r = {deploy["training_r"] - abs(r):+.4f}')
        print(f'{"="*50}')
    else:
        print(f'  ERROR: Only {len(df_result)} wells with ATP')

    # 保存
    df_result.to_excel(os.path.join(OUTPUT_DIR, 'GC_prediction.xlsx'), index=False)
    print(f'\nResults saved: output/GC_prediction.xlsx')

    # 打印每个well
    print(f'\n{"Well":<8s} {"Score":>10s} {"ATP":>12s}')
    print('-' * 32)
    for _, row in df_result.iterrows():
        print(f'{row["Well_ID"]:<8s} {row["Score"]:>10.4f} {row["ATP"]:>12.0f}')


if __name__ == '__main__':
    main()