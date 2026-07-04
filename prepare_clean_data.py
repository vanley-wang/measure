"""
清洗数据集生成脚本

综合数据诊断结论，生成供后续聚类使用的标准数据集。

输出：
  reports/clean_data_full.parquet   (全部数据，含异常标记)
  reports/clean_data_cleaned.parquet (剔除 IF 异常值)
  reports/preprocessor_fitted.pickle (拟合好的预处理器)
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from cluster_utils import RAW_FEATURES, Preprocessor

# ================= 配置 =================
DATA_FOLDERS = [
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]
REPORTS_DIR = 'reports'
OUTPUT_FULL = os.path.join(REPORTS_DIR, 'clean_data_full.parquet')
OUTPUT_CLEANED = os.path.join(REPORTS_DIR, 'clean_data_cleaned.parquet')
PREPROCESSOR_PATH = os.path.join(REPORTS_DIR, 'preprocessor_fitted.pickle')

# 药物浓度映射
DRUG_MAP = {
    'Control': 0, 'Ctrl': 0,
    '20': 20, '40': 40, '80': 80,
}

os.makedirs(REPORTS_DIR, exist_ok=True)

# ==========================================


def infer_concentration(well_name: str) -> int:
    for key, val in DRUG_MAP.items():
        if key in well_name:
            return val
    return -1


def load_all_data():
    df_list = []
    for folder in DATA_FOLDERS:
        if not os.path.exists(folder):
            continue
        for fp in glob.glob(os.path.join(folder, '*.xlsx')):
            df = pd.read_excel(fp)
            df['_well'] = os.path.basename(fp).replace('.xlsx', '')
            df['_folder'] = folder
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def detect_outliers_if(X):
    """Isolation Forest 异常检测。"""
    clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    preds = clf.fit_predict(X)
    return preds == -1


def main():
    print("=" * 60)
    print("清洗数据集生成")
    print("=" * 60)

    # 1. 加载数据
    print("\n--- 加载数据 ---")
    df = load_all_data()
    print(f"总类器官数: {len(df)}")
    df['_conc'] = df['_well'].apply(infer_concentration)

    # 2. 预处理（统一拟合）
    print("\n--- 拟合预处理器 ---")
    preprocessor = Preprocessor()
    X = preprocessor.fit_transform(df[RAW_FEATURES])
    feature_names = preprocessor.get_feature_names()
    print(f"处理后特征维度: {X.shape}")

    # 保存预处理器
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"预处理器已保存 -> {PREPROCESSOR_PATH}")

    # 3. 异常值检测
    print("\n--- 异常值检测 (Isolation Forest) ---")
    outlier_flags = detect_outliers_if(X)
    df['_outlier_flag'] = outlier_flags
    print(f"异常样本: {outlier_flags.sum()} ({outlier_flags.mean()*100:.2f}%)")

    # 4. PCA 辅助坐标（方便后续直接画图）
    print("\n--- 计算 PCA 坐标 ---")
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(X)
    df['_pc1'] = pca_coords[:, 0]
    df['_pc2'] = pca_coords[:, 1]
    df['_pc3'] = pca_coords[:, 2]
    var = pca.explained_variance_ratio_ * 100
    print(f"  PC1={var[0]:.1f}%, PC2={var[1]:.1f}%, PC3={var[2]:.1f}%, 累计={var.sum():.1f}%")

    # 5. 保存完整版
    print("\n--- 保存数据集 ---")
    # 确保所有列类型兼容 parquet
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    df.to_parquet(OUTPUT_FULL, index=False)
    print(f"  完整数据集 ({len(df)} 条) -> {OUTPUT_FULL}")

    # 6. 保存清洗版（剔除异常值）
    df_clean = df[~df['_outlier_flag']].copy()
    df_clean.to_parquet(OUTPUT_CLEANED, index=False)
    print(f"  清洗数据集 ({len(df_clean)} 条) -> {OUTPUT_CLEANED}")

    # 7. 生成数据说明
    readme_path = os.path.join(REPORTS_DIR, 'clean_data_readme.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("清洗数据集说明\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write(f"原始样本数: {len(df)}\n")
        f.write(f"清洗后样本数: {len(df_clean)}\n")
        f.write(f"剔除异常值比例: {outlier_flags.mean()*100:.2f}%\n\n")
        f.write("列说明:\n")
        f.write("  _well: 来源 well 名称\n")
        f.write("  _conc: 药物浓度 (0=Control, 20, 40, 80)\n")
        f.write("  _outlier_flag: Isolation Forest 判定异常 (True=异常)\n")
        f.write("  _pc1/_pc2/_pc3: PCA 前 3 主成分坐标\n")
        f.write("  其余列: 原始测量特征\n\n")
        f.write("特征列表 (预处理后):\n")
        for fn in feature_names:
            f.write(f"  - {fn}\n")
        f.write(f"\n预处理器保存路径: {PREPROCESSOR_PATH}\n")
    print(f"  数据说明 -> {readme_path}")

    print("\n[Done] 清洗数据集生成完成。")


if __name__ == "__main__":
    main()
