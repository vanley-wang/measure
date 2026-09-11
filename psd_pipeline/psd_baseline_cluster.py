# ============================================================
# PSD-Pipeline 步骤4：聚类器训练
# 用 Day3+Day5 合并数据训练 K-means++ 聚类器（与论文一致）
# 输出：baseline_kmeans.pkl, baseline_scaler.pkl, 手肘图
# ============================================================

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
# 尝试设置中文字体，避免图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psd_config import (
    DAY3_MEASURE_DIR, DAY5_MEASURE_DIR, MODEL_DIR, OUTPUT_DIR,
    CLUSTER_FEATURES, N_CLUSTERS
)


def load_combined_data():
    df_list = []
    for label, measure_dir in [('Day3', DAY3_MEASURE_DIR), ('Day5', DAY5_MEASURE_DIR)]:
        files = sorted(glob.glob(os.path.join(measure_dir, '*.xlsx')))
        print(f"从 {measure_dir} 读取 {len(files)} 个孔位文件")
        for f in files:
            try:
                df = pd.read_excel(f)
                well_name = os.path.splitext(os.path.basename(f))[0]
                df['Well_ID'] = well_name
                df['TimePoint'] = label
                df_list.append(df)
            except Exception as e:
                print(f"  读取失败: {f} - {e}")

    if not df_list:
        raise ValueError("未在 Day3/Day5 目录中找到有效数据")

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"  Day3+Day5 共加载 {len(df_all)} 个类器官")
    return df_all


def plot_elbow(data_std, save_path):
    K_range = range(1, 9)
    dispersions = []
    for k in K_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
        km.fit(data_std)
        d = sum(np.min(cdist(data_std, km.cluster_centers_, 'euclidean'), axis=1)) / data_std.shape[0]
        dispersions.append(d)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(K_range, dispersions, marker='o', linestyle='-', color='black',
            linewidth=1.5, markersize=7, markerfacecolor='white', markeredgewidth=1.5)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Dispersion', fontsize=12, fontweight='bold')
    ax.set_title('Elbow Method - Day3+Day5 Combined', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"手肘图已保存: {save_path}")


def main():
    print("=" * 60)
    print("PSD 步骤4：聚类器训练（Day3+Day5 合并数据，与论文一致）")
    print("=" * 60)

    # 1. 加载 Day3+Day5 合并数据
    print("\n[1/5] 加载 Day3+Day5 合并数据...")
    df_all = load_combined_data()

    # 2. 提取特征
    print("\n[2/5] 提取聚类特征...")
    missing = [c for c in CLUSTER_FEATURES if c not in df_all.columns]
    if missing:
        raise ValueError(f"缺失特征列: {missing}")
    X = df_all[CLUSTER_FEATURES].fillna(0).values
    print(f"  特征矩阵形状: {X.shape}")

    # 3. 标准化
    print("\n[3/5] 标准化...")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 保存 scaler
    scaler_path = os.path.join(MODEL_DIR, 'baseline_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler 已保存: {scaler_path}")

    # 4. 手肘法
    print("\n[4/5] 手肘法分析...")
    elbow_path = os.path.join(OUTPUT_DIR, 'elbow_day3_baseline.png')
    plot_elbow(X_std, elbow_path)

    # 5. 训练最终模型
    print(f"\n[5/5] 训练 K-means++ (K={N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X_std)

    model
    









    _path = os.path.join(MODEL_DIR, 'baseline_kmeans.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"  模型已保存: {model_path}")

    # 打印聚类中心统计
    print("\n" + "=" * 60)
    print("聚类中心（标准化空间）:")
    centers_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=CLUSTER_FEATURES
    )
    centers_df.index.name = 'Cluster'
    print(centers_df)

    # 各类别样本数
    unique, counts = np.unique(labels, return_counts=True)
    print("\n各类别样本数 (Day3+Day5):")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} 个类器官 ({c/len(labels)*100:.1f}%)")

    # 保存聚类中心
    centers_path = os.path.join(OUTPUT_DIR, 'baseline_cluster_centers.xlsx')
    centers_df.to_excel(centers_path)
    print(f"\n聚类中心已保存: {centers_path}")

    # 保存带标签的合并数据
    df_all['Cluster'] = labels
    labeled_path = os.path.join(OUTPUT_DIR, 'day3_day5_labeled.xlsx')
    df_all.to_excel(labeled_path, index=False)
    print(f"Day3+Day5 标注数据已保存: {labeled_path}")

    print("\n步骤4 完成！")


if __name__ == '__main__':
    main()