# ============================================================
# Step 1: K-means++ K=6 聚类 + 合并为 4 表型 (复现论文)
# 论文方法: 对 Day3 全量数据做 K=6 聚类，再用 numeric_map 合并为 4 型
# ============================================================
import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'Data', 'FXN_2023_new（ICC）')
DAY3_MEASURE_DIR = os.path.join(DATA_DIR, 'FXN_20230701', 'measure_excel')
DAY5_MEASURE_DIR = os.path.join(DATA_DIR, 'FXN_20230703', 'measure_excel')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# 11 个聚类特征 (含散射)
FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD'
]

# K=6 → 4 类 numeric_map
NUMERIC_MAP = {3: 0, 5: 0, 1: 1, 0: 2, 4: 2, 2: 3}
PHENOTYPE_DESC = {0: '巨大囊泡型', 1: '中等过渡型', 2: '小体积基准型', 3: '高散射实心型'}

K_VALUE = 6


def load_measure_files(measure_dir):
    dfs = []
    for fname in os.listdir(measure_dir):
        if not fname.endswith('.xlsx'):
            continue
        fpath = os.path.join(measure_dir, fname)
        df = pd.read_excel(fpath)
        df['Source_File'] = fname
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f'No xlsx found in {measure_dir}')
    return pd.concat(dfs, ignore_index=True)


def main():
    # 1) 加载 Day3 全量数据
    print('Loading Day3 data...')
    df_day3 = load_measure_files(DAY3_MEASURE_DIR)

    # 2) 提取特征
    missing = [f for f in FEATURES if f not in df_day3.columns]
    if missing:
        raise KeyError(f'Missing features: {missing}')
    X = df_day3[FEATURES].fillna(0).values

    # 3) 标准化 + 保存 scaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    with open(os.path.join(MODEL_DIR, 'scaler_k6.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # 4) 手肘法
    print(f'Elbow analysis (K=1..8)...')
    dispersions = []
    for k in range(1, 9):
        km = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
        km.fit(X_std)
        d = np.sum(np.min(cdist(X_std, km.cluster_centers_, 'euclidean'), axis=1)) / X_std.shape[0]
        dispersions.append(d)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, 9), dispersions, 'ko-', mfc='white', lw=1.5, ms=7)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Dispersion', fontsize=12, fontweight='bold')
    ax.set_title(f'Elbow Method (K={K_VALUE})', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, 'elbow_k6.png'), dpi=300)
    plt.close(fig)

    # 5) 训练 K=6 最终模型
    print(f'Training K={K_VALUE} model...')
    kmeans = KMeans(n_clusters=K_VALUE, init='k-means++', n_init='auto', random_state=42)
    labels_k6 = kmeans.fit_predict(X_std)
    with open(os.path.join(MODEL_DIR, 'kmeans_k6.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)

    # 6) K=6 → K=4 合并
    labels_merged = np.array([NUMERIC_MAP[l] for l in labels_k6])

    # 7) 统计合并后的分布
    unique, counts = np.unique(labels_merged, return_counts=True)
    total = len(labels_merged)
    print('\n合并后表型分布 (Day3全量):')
    for u, c in zip(unique, counts):
        print(f'  Type {u} ({PHENOTYPE_DESC[u]}): {c} ({100*c/total:.1f}%)')

    # 8) 保存标好标签的 Day3 数据供 Step2 使用
    df_day3['Cluster_K6'] = labels_k6
    df_day3['Cluster'] = labels_merged
    df_day3['Phenotype'] = [PHENOTYPE_DESC[l] for l in labels_merged]
    out_path = os.path.join(MODEL_DIR, 'day3_labeled.pkl')
    df_day3.to_pickle(out_path)
    print(f'\nDay3 labeled data saved: {out_path}')

    print('\nStep 1 complete.')


if __name__ == '__main__':
    main()