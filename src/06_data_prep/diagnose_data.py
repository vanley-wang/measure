import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

"""
数据质量诊断脚本

读取所有 measure_excel 文件，在原始特征空间上做全面诊断：
  - 缺失值 / 零值统计
  - 异常值检测（IQR + Isolation Forest）
  - 分布诊断（直方图、正态性检验、偏度、峰度）

输出：
  reports/data_diagnosis.xlsx
  reports/figures/feature_distributions.png
  reports/figures/outlier_scatter.png
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

from cluster_utils import RAW_FEATURES

# ================= 配置 =================
DATA_FOLDERS = [
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]
REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 药物浓度映射（从 well 名解析）
DRUG_MAP = {
    'Control': 0, 'Ctrl': 0,
    '20': 20, '40': 40, '80': 80,
}

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
            print(f"[WARN] 跳过不存在的目录: {folder}")
            continue
        files = glob.glob(os.path.join(folder, '*.xlsx'))
        print(f"  {folder}: {len(files)} 个文件")
        for fp in files:
            try:
                df = pd.read_excel(fp)
                df['_well'] = os.path.basename(fp).replace('.xlsx', '')
                df['_folder'] = folder
                df_list.append(df)
            except Exception as e:
                print(f"    [ERR] {os.path.basename(fp)}: {e}")
    if not df_list:
        raise ValueError("未读取到任何数据！")
    return pd.concat(df_list, ignore_index=True)


def compute_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """逐特征缺失值统计。"""
    stats_list = []
    for col in RAW_FEATURES:
        if col not in df.columns:
            continue
        total = len(df)
        missing = df[col].isna().sum()
        zero = (df[col] == 0).sum() if df[col].dtype.kind in 'iuf' else 0
        stats_list.append({
            'Feature': col,
            'Total': total,
            'Missing_Count': missing,
            'Missing_Pct': round(missing / total * 100, 2),
            'Zero_Count': zero,
            'Zero_Pct': round(zero / total * 100, 2),
        })
    return pd.DataFrame(stats_list)


def detect_outliers_iqr(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """IQR 法则标记异常值。"""
    outlier_flags = pd.DataFrame(False, index=df.index, columns=[f"{f}_iqr_outlier" for f in features])
    for f in features:
        if f not in df.columns:
            continue
        Q1 = df[f].quantile(0.25)
        Q3 = df[f].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_flags[f"{f}_iqr_outlier"] = (df[f] < lower) | (df[f] > upper)
    return outlier_flags


def detect_outliers_isolation_forest(df: pd.DataFrame, features: list) -> np.ndarray:
    """Isolation Forest 异常检测。"""
    X = df[features].fillna(0).values
    clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    preds = clf.fit_predict(X)
    return preds == -1  # True = outlier


def compute_distribution_stats(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """计算每个特征的分布统计量。"""
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        vals = df[f].dropna()
        # Shapiro 在样本量大时容易拒绝，取前 5000 个
        shapiro_p = np.nan
        if len(vals) <= 5000:
            try:
                _, shapiro_p = stats.shapiro(vals)
            except Exception:
                pass
        # D'Agostino
        dagostino_p = np.nan
        if len(vals) >= 8:
            try:
                _, dagostino_p = stats.normaltest(vals)
            except Exception:
                pass
        rows.append({
            'Feature': f,
            'Mean': vals.mean(),
            'Std': vals.std(),
            'Min': vals.min(),
            'Q25': vals.quantile(0.25),
            'Median': vals.median(),
            'Q75': vals.quantile(0.75),
            'Max': vals.max(),
            'Skewness': vals.skew(),
            'Kurtosis': vals.kurtosis(),
            'Shapiro_p': shapiro_p,
            'DAgostino_p': dagostino_p,
        })
    return pd.DataFrame(rows)


def plot_feature_distributions(df: pd.DataFrame, features: list, save_path: str):
    """绘制每个特征的直方图 + KDE。"""
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for idx, f in enumerate(features):
        ax = axes[idx]
        vals = df[f].dropna()
        ax.hist(vals, bins=50, color='steelblue', edgecolor='white', alpha=0.7, density=True)
        try:
            sns.kdeplot(vals, ax=ax, color='darkred', lw=1.5)
        except Exception:
            pass
        ax.set_title(f, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Density')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 隐藏多余的子图
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  特征分布图已保存 -> {save_path}")


def plot_outlier_scatter(df: pd.DataFrame, save_path: str):
    """在 Volume_Fill vs Scatt_Mean 上标记异常值。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # IQR 异常值（任一特征被标记）
    iqr_cols = [c for c in df.columns if c.endswith('_iqr_outlier')]
    df['_any_iqr'] = df[iqr_cols].any(axis=1)

    ax = axes[0]
    normal = df[~df['_any_iqr']]
    outlier = df[df['_any_iqr']]
    ax.scatter(normal['Organoids_Volume_Fill'], normal['Scatt_Mean'],
               c='steelblue', s=10, alpha=0.5, label='Normal')
    ax.scatter(outlier['Organoids_Volume_Fill'], outlier['Scatt_Mean'],
               c='red', s=15, alpha=0.8, label='IQR Outlier')
    ax.set_xlabel('Organoids Volume Fill', fontsize=11)
    ax.set_ylabel('Scatt Mean (OAC)', fontsize=11)
    ax.set_title('IQR Outliers', fontsize=12, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Isolation Forest
    ax = axes[1]
    normal = df[~df['_if_outlier']]
    outlier = df[df['_if_outlier']]
    ax.scatter(normal['Organoids_Volume_Fill'], normal['Scatt_Mean'],
               c='steelblue', s=10, alpha=0.5, label='Normal')
    ax.scatter(outlier['Organoids_Volume_Fill'], outlier['Scatt_Mean'],
               c='red', s=15, alpha=0.8, label='IF Outlier')
    ax.set_xlabel('Organoids Volume Fill', fontsize=11)
    ax.set_ylabel('Scatt Mean (OAC)', fontsize=11)
    ax.set_title('Isolation Forest Outliers (5%)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  异常值散点图已保存 -> {save_path}")


def main():
    print("=" * 60)
    print("数据质量诊断")
    print("=" * 60)

    # 1. 加载数据
    print("\n--- 加载数据 ---")
    df = load_all_data()
    print(f"总类器官数: {len(df)}")
    df['_conc'] = df['_well'].apply(infer_concentration)

    # 2. 缺失值与零值统计
    print("\n--- 缺失值 / 零值统计 ---")
    missing_stats = compute_missing_stats(df)
    print(missing_stats.to_string(index=False))

    # 3. 异常值检测
    print("\n--- 异常值检测 ---")
    print("  IQR 法则 (1.5×IQR)...")
    iqr_flags = detect_outliers_iqr(df, RAW_FEATURES)
    df = pd.concat([df, iqr_flags], axis=1)

    print("  Isolation Forest (contamination=0.05)...")
    df['_if_outlier'] = detect_outliers_isolation_forest(df, RAW_FEATURES)

    # 统计
    any_iqr = df[[c for c in df.columns if c.endswith('_iqr_outlier')]].any(axis=1)
    print(f"  IQR 异常样本: {any_iqr.sum()} ({any_iqr.mean()*100:.2f}%)")
    print(f"  IF  异常样本: {df['_if_outlier'].sum()} ({df['_if_outlier'].mean()*100:.2f}%)")

    # 按浓度分组看异常值比例
    print("\n  按药物浓度分组的 IF 异常比例:")
    for conc in sorted(df['_conc'].unique()):
        sub = df[df['_conc'] == conc]
        if len(sub) == 0:
            continue
        print(f"    Conc={conc}: {sub['_if_outlier'].mean()*100:.2f}% ({sub['_if_outlier'].sum()}/{len(sub)})")

    # 4. 分布统计
    print("\n--- 分布统计 ---")
    dist_stats = compute_distribution_stats(df, RAW_FEATURES)
    print(dist_stats[['Feature', 'Skewness', 'Kurtosis', 'Shapiro_p', 'DAgostino_p']].to_string(index=False))

    # 5. 绘图
    print("\n--- 生成图表 ---")
    plot_feature_distributions(df, RAW_FEATURES, os.path.join(FIGURES_DIR, 'feature_distributions.png'))
    plot_outlier_scatter(df, os.path.join(FIGURES_DIR, 'outlier_scatter.png'))

    # 6. 保存 Excel 报告
    print("\n--- 保存诊断报告 ---")
    excel_path = os.path.join(REPORTS_DIR, 'data_diagnosis.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        missing_stats.to_excel(writer, sheet_name='Missing_Zero_Stats', index=False)
        dist_stats.to_excel(writer, sheet_name='Distribution_Stats', index=False)
        # 异常值汇总（按 well）
        well_outlier = df.groupby('_well').agg(
            Total=('Organoids_Volume_Fill', 'size'),
            IQR_Outliers=('_any_iqr', 'sum'),
            IF_Outliers=('_if_outlier', 'sum'),
        )
        well_outlier['IQR_Pct'] = (well_outlier['IQR_Outliers'] / well_outlier['Total'] * 100).round(2)
        well_outlier['IF_Pct'] = (well_outlier['IF_Outliers'] / well_outlier['Total'] * 100).round(2)
        well_outlier.to_excel(writer, sheet_name='Well_Outlier_Summary')

    print(f"  诊断报告已保存 -> {excel_path}")
    print("\n[Done] 数据质量诊断完成。")


if __name__ == "__main__":
    main()
