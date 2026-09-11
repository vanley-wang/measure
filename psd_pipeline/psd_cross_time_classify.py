# ============================================================
# PSD-Pipeline 步骤5：跨时间点表型匹配
# 加载 Day3 训练的基线聚类器，对 Day5 类器官进行分类预测
# 保证前后类别定义一致，避免独立聚类导致的标签漂移
# ============================================================

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psd_config import (
    DAY3_MEASURE_DIR, DAY5_MEASURE_DIR,
    MODEL_DIR, OUTPUT_DIR, CLUSTER_FEATURES, N_CLUSTERS
)


def load_day_data(measure_dir, time_label):
    df_list = []
    files = sorted(glob.glob(os.path.join(measure_dir, '*.xlsx')))
    print(f"从 {measure_dir} 读取 {len(files)} 个孔位文件")

    for f in files:
        try:
            df = pd.read_excel(f)
            raw_name = os.path.splitext(os.path.basename(f))[0]
            parts = raw_name.rsplit('_', maxsplit=1)
            well_name = parts[0] if len(parts) == 2 else raw_name
            df['Well_ID'] = well_name
            df['TimePoint'] = time_label
            df_list.append(df)
        except Exception as e:
            print(f"  读取失败: {f} - {e}")

    if not df_list:
        raise ValueError(f"未在 {measure_dir} 中找到有效数据")

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"  {time_label} 共加载 {len(df_all)} 个类器官")
    return df_all


def classify_day5(df_day5, kmeans, scaler):
    missing = [c for c in CLUSTER_FEATURES if c not in df_day5.columns]
    if missing:
        raise ValueError(f"Day5 缺失特征列: {missing}")

    X = df_day5[CLUSTER_FEATURES].fillna(0).values
    X_std = scaler.transform(X)
    labels = kmeans.predict(X_std)
    df_day5['Cluster'] = labels
    return df_day5


def main():
    print("=" * 60)
    print("PSD 步骤5：跨时间点表型匹配")
    print("=" * 60)

    # 1. 加载基线模型
    print("\n[1/4] 加载基线聚类器...")
    kmeans_path = os.path.join(MODEL_DIR, 'baseline_kmeans.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'baseline_scaler.pkl')

    with open(kmeans_path, 'rb') as f:
        kmeans = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  模型已加载: K={kmeans.n_clusters}")

    # 2. 加载 Day3 数据并分类（用于后续 Delta 计算的一致性）
    print("\n[2/4] 加载 Day3 基线数据...")
    df_day3 = load_day_data(DAY3_MEASURE_DIR, 'Day3')

    # 用基线模型对 Day3 也做一次分类（确保一致性）
    df_day3 = classify_day5(df_day3, kmeans, scaler)

    # 3. 加载 Day5 数据
    print("\n[3/4] 加载 Day5 终点数据...")
    df_day5 = load_day_data(DAY5_MEASURE_DIR, 'Day5')

    # 4. 用基线聚类器对 Day5 分类
    print("\n[4/4] 基线锚定分类 Day5 类器官...")
    df_day5 = classify_day5(df_day5, kmeans, scaler)

    # 统计
    print("\n" + "=" * 60)
    print("Day3 表型分布:")
    for c in range(N_CLUSTERS):
        n = (df_day3['Cluster'] == c).sum()
        print(f"  Cluster {c}: {n} 个 ({n/len(df_day3)*100:.1f}%)")

    print("\nDay5 表型分布（基线锚定）:")
    for c in range(N_CLUSTERS):
        n = (df_day5['Cluster'] == c).sum()
        print(f"  Cluster {c}: {n} 个 ({n/len(df_day5)*100:.1f}%)")

    # 保存
    day3_save = os.path.join(OUTPUT_DIR, 'day3_baseline_classified.xlsx')
    day5_save = os.path.join(OUTPUT_DIR, 'day5_baseline_classified.xlsx')
    df_day3.to_excel(day3_save, index=False)
    df_day5.to_excel(day5_save, index=False)
    print(f"\nDay3 分类结果已保存: {day3_save}")
    print(f"Day5 分类结果已保存: {day5_save}")

    print("\n步骤5 完成！")


if __name__ == '__main__':
    main()