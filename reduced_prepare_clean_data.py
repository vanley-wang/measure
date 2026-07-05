"""
5维精简特征 — 清洗数据集生成

输出：
  reports/clean_data_5d_full.parquet
  reports/clean_data_5d_cleaned.parquet
  reports/preprocessor_5d_fitted.pickle
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from cluster_utils import REDUCED_RAW_FEATURES, Preprocessor

# ================= 配置 =================
DATA_FOLDERS = [
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]
REPORTS_DIR = 'reports'
OUTPUT_FULL = os.path.join(REPORTS_DIR, 'clean_data_5d_full.parquet')
OUTPUT_CLEANED = os.path.join(REPORTS_DIR, 'clean_data_5d_cleaned.parquet')
PREPROCESSOR_PATH = os.path.join(REPORTS_DIR, 'preprocessor_5d_fitted.pickle')

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
    clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    preds = clf.fit_predict(X)
    return preds == -1


def main():
    print("=" * 60)
    print("5维精简特征 — 清洗数据集生成")
    print("=" * 60)

    print("\n--- 加载数据 ---")
    df = load_all_data()
    print(f"总类器官数: {len(df)}")
    df['_conc'] = df['_well'].apply(infer_concentration)

    print("\n--- 拟合预处理器 (5D) ---")
    preprocessor = Preprocessor(mode='reduced')
    X = preprocessor.fit_transform(df[REDUCED_RAW_FEATURES])
    feature_names = preprocessor.get_feature_names()
    print(f"处理后特征维度: {X.shape}")
    print(f"特征列表: {feature_names}")

    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"预处理器已保存 -> {PREPROCESSOR_PATH}")

    print("\n--- 异常值检测 (Isolation Forest) ---")
    outlier_flags = detect_outliers_if(X)
    df['_outlier_flag'] = outlier_flags
    print(f"异常样本: {outlier_flags.sum()} ({outlier_flags.mean()*100:.2f}%)")

    print("\n--- 计算 PCA 坐标 ---")
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(X)
    df['_pc1'] = pca_coords[:, 0]
    df['_pc2'] = pca_coords[:, 1]
    df['_pc3'] = pca_coords[:, 2]
    var = pca.explained_variance_ratio_ * 100
    print(f"  PC1={var[0]:.1f}%, PC2={var[1]:.1f}%, PC3={var[2]:.1f}%, 累计={var.sum():.1f}%")

    print("\n--- 保存数据集 ---")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    df.to_parquet(OUTPUT_FULL, index=False)
    print(f"  完整数据集 ({len(df)} 条) -> {OUTPUT_FULL}")

    df_clean = df[~df['_outlier_flag']].copy()
    df_clean.to_parquet(OUTPUT_CLEANED, index=False)
    print(f"  清洗数据集 ({len(df_clean)} 条) -> {OUTPUT_CLEANED}")

    readme_path = os.path.join(REPORTS_DIR, 'clean_data_5d_readme.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("5维精简特征清洗数据集说明\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write(f"原始样本数: {len(df)}\n")
        f.write(f"清洗后样本数: {len(df_clean)}\n")
        f.write(f"剔除异常值比例: {outlier_flags.mean()*100:.2f}%\n\n")
        f.write("5维特征设计:\n")
        f.write("  - Organoids_Volume_Fill: 体型大小（log1p变换）\n")
        f.write("  - Sphericity: 形状球度\n")
        f.write("  - Scatt_Mean: OAC均值（核心药效指标）\n")
        f.write("  - Scatt_STD: OAC标准差（内部异质性）\n")
        f.write("  - Cavity_Ratio: 空腔化程度（0=实心）\n\n")
        f.write("辅助列:\n")
        f.write("  _well: 来源 well 名称\n")
        f.write("  _conc: 药物浓度 (0=Control, 20, 40, 80)\n")
        f.write("  _outlier_flag: Isolation Forest 判定异常\n")
        f.write("  _pc1/_pc2/_pc3: PCA 前 3 主成分坐标\n\n")
        f.write(f"预处理器保存路径: {PREPROCESSOR_PATH}\n")
    print(f"  数据说明 -> {readme_path}")

    print("\n[Done] 5维清洗数据集生成完成。")


if __name__ == "__main__":
    main()
