# ============================================================
# PSD-Pipeline 步骤6：分表型 Delta 特征计算
# 核心创新：不是算整体 Delta，而是按4类表型分别计算
# Day5 中位数 - Day3 中位数，得到 "表型分层时间差特征"
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psd_config import (
    OUTPUT_DIR, N_CLUSTERS, AGGREGATION_PARAMS, PHENOTYPE_LABELS,
    MIN_CLUSTER_SAMPLES
)


def load_classified_data():
    day3_path = os.path.join(OUTPUT_DIR, 'day3_baseline_classified.xlsx')
    day5_path = os.path.join(OUTPUT_DIR, 'day5_baseline_classified.xlsx')

    if not os.path.exists(day3_path):
        raise FileNotFoundError(f"请先运行 psd_cross_time_classify.py: {day3_path}")
    if not os.path.exists(day5_path):
        raise FileNotFoundError(f"请先运行 psd_cross_time_classify.py: {day5_path}")

    print(f"加载 Day3: {day3_path}")
    df_day3 = pd.read_excel(day3_path)
    print(f"加载 Day5: {day5_path}")
    df_day5 = pd.read_excel(day5_path)
    return df_day3, df_day5


def compute_well_phenotype_stats(df, time_label):
    records = []
    wells = sorted(df['Well_ID'].unique())

    for well in wells:
        df_well = df[df['Well_ID'] == well]
        record = {'Name': f"{well}_{time_label}", 'Well_ID': well, 'TimePoint': time_label}

        for c in range(N_CLUSTERS):
            df_cluster = df_well[df_well['Cluster'] == c]
            n = len(df_cluster)
            record[f'Number_{c+1}'] = n

            if n >= MIN_CLUSTER_SAMPLES:
                for src_col, agg_name in AGGREGATION_PARAMS:
                    if src_col in df_cluster.columns:
                        val = np.median(df_cluster[src_col].dropna().values)
                    else:
                        val = np.nan
                    record[f'{agg_name}_Median_{c+1}'] = val
            else:
                for src_col, agg_name in AGGREGATION_PARAMS:
                    record[f'{agg_name}_Median_{c+1}'] = np.nan

        records.append(record)

    return pd.DataFrame(records)


def compute_delta_features(df_day3_stats, df_day5_stats):
    wells_day3 = set(df_day3_stats['Well_ID'].unique())
    wells_day5 = set(df_day5_stats['Well_ID'].unique())
    common_wells = sorted(wells_day3 & wells_day5)

    print(f"\nDay3 孔位数: {len(wells_day3)}")
    print(f"Day5 孔位数: {len(wells_day5)}")
    print(f"共同孔位:   {len(common_wells)}")

    if len(common_wells) == 0:
        raise ValueError("Day3 和 Day5 没有共同的孔位")

    delta_records = []
    for well in common_wells:
        row3 = df_day3_stats[df_day3_stats['Well_ID'] == well].iloc[0]
        row5 = df_day5_stats[df_day5_stats['Well_ID'] == well].iloc[0]

        record = {'Well_ID': well}

        for c in range(N_CLUSTERS):
            for src_col, agg_name in AGGREGATION_PARAMS:
                col_name = f'{agg_name}_Median_{c+1}'
                val3 = row3.get(col_name, np.nan)
                val5 = row5.get(col_name, np.nan)
                if pd.notna(val3) and pd.notna(val5):
                    record[f'{agg_name}_Delta_{c+1}'] = val5 - val3
                else:
                    record[f'{agg_name}_Delta_{c+1}'] = np.nan

            n3 = row3.get(f'Number_{c+1}', 0)
            n5 = row5.get(f'Number_{c+1}', 0)
            if n3 >= MIN_CLUSTER_SAMPLES and n5 >= MIN_CLUSTER_SAMPLES:
                record[f'Number_Delta_{c+1}'] = n5 - n3
            elif n3 >= MIN_CLUSTER_SAMPLES or n5 >= MIN_CLUSTER_SAMPLES:
                record[f'Number_Delta_{c+1}'] = n5 - n3
            else:
                record[f'Number_Delta_{c+1}'] = np.nan

        delta_records.append(record)

    df_delta = pd.DataFrame(delta_records)
    return df_delta


def main():
    print("=" * 60)
    print("PSD 步骤6：分表型 Delta 特征计算")
    print("=" * 60)

    # 1. 加载分类数据
    print("\n[1/4] 加载分类数据...")
    df_day3, df_day5 = load_classified_data()

    # 2. 按孔×表型统计中位数
    print("\n[2/4] 计算 Day3 各孔分表型中位数...")
    df_day3_stats = compute_well_phenotype_stats(df_day3, 'Day3')
    day3_stats_path = os.path.join(OUTPUT_DIR, 'day3_phenotype_stats.xlsx')
    df_day3_stats.to_excel(day3_stats_path, index=False)
    print(f"  已保存: {day3_stats_path}")

    print("\n[3/4] 计算 Day5 各孔分表型中位数...")
    df_day5_stats = compute_well_phenotype_stats(df_day5, 'Day5')
    day5_stats_path = os.path.join(OUTPUT_DIR, 'day5_phenotype_stats.xlsx')
    df_day5_stats.to_excel(day5_stats_path, index=False)
    print(f"  已保存: {day5_stats_path}")

    # 3. 计算 Delta
    print("\n[4/4] 计算分表型 Delta 特征...")
    df_delta = compute_delta_features(df_day3_stats, df_day5_stats)

    delta_path = os.path.join(OUTPUT_DIR, 'psd_delta_features.xlsx')
    df_delta.to_excel(delta_path, index=False)
    print(f"  Delta 特征矩阵已保存: {delta_path}")

    # 统计
    n_wells = len(df_delta)
    n_features = len(df_delta.columns) - 1
    print(f"\n  孔位数: {n_wells}")
    print(f"  Delta 特征数: {n_features}")
    print(f"  每类表型: {n_features // N_CLUSTERS} 个特征")
    print(f"  特征维度 = {N_CLUSTERS} 表型 × 形态参数")

    # 打印前几行
    print("\nDelta 特征矩阵预览（前5行）:")
    print(df_delta.head().to_string())

    print("\n步骤6 完成！")


if __name__ == '__main__':
    main()