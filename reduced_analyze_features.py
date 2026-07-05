"""
5维精简特征 — 特征重要性 / 冗余度分析

输出：
  reports/figures/reduced_correlation_heatmap.png
  reports/reduced_feature_importance.xlsx
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
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

from cluster_utils import REDUCED_RAW_FEATURES, Preprocessor, load_model_package

warnings.filterwarnings('ignore')

# ================= 配置 =================
DATA_FOLDERS = [
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]
REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
MODEL_PATH = 'model/Kmeans-scatt.pickle'

os.makedirs(FIGURES_DIR, exist_ok=True)

# ==========================================


def load_all_data():
    df_list = []
    for folder in DATA_FOLDERS:
        if not os.path.exists(folder):
            continue
        for fp in glob.glob(os.path.join(folder, '*.xlsx')):
            df = pd.read_excel(fp)
            df['_well'] = os.path.basename(fp).replace('.xlsx', '')
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def get_kmeans_labels(df, features):
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        pkg = load_model_package(MODEL_PATH)
        preprocessor = pkg.get('preprocessor')
        scaler = pkg.get('scaler')
        model = pkg['model']
        raw_to_final = pkg.get('raw_to_final', {})

        if preprocessor is not None:
            X = preprocessor.transform(df[features])
        elif scaler is not None:
            X = scaler.transform(df[features])
        else:
            return None

        raw_labels = model.predict(X)
        if raw_to_final:
            return np.array([raw_to_final.get(l, l) for l in raw_labels])
        return raw_labels
    except Exception as e:
        print(f"[WARN] 加载 K-means 失败: {e}")
        return None


def plot_correlation_heatmaps(df_orig, df_proc, processed_features, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 原始空间（5个特征 + Cavity_Ratio）
    numeric_cols = [c for c in df_orig.columns if c in processed_features or c in REDUCED_RAW_FEATURES]
    numeric_cols = list(dict.fromkeys(numeric_cols))  # 去重保序
    corr_pearson = df_orig[numeric_cols].corr(method='pearson')
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)
    sns.heatmap(corr_pearson, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, ax=axes[0],
                cbar_kws={'shrink': 0.8})
    axes[0].set_title('Pearson Correlation (Raw, 5D)', fontsize=12, fontweight='bold')

    # 预处理后空间
    corr_proc = pd.DataFrame(df_proc, columns=processed_features).corr(method='pearson')
    mask2 = np.triu(np.ones_like(corr_proc, dtype=bool), k=1)
    sns.heatmap(corr_proc, mask=mask2, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, ax=axes[1],
                cbar_kws={'shrink': 0.8})
    axes[1].set_title('Pearson Correlation (Preprocessed, 5D)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  相关性热力图已保存 -> {save_path}")


def compute_spearman_corr(df_orig, features):
    cols = [f for f in features if f in df_orig.columns]
    return df_orig[cols].corr(method='spearman')


def compute_vif(X_proc, feature_names):
    X = pd.DataFrame(X_proc, columns=feature_names).fillna(0)
    vif_data = []
    for i, name in enumerate(feature_names):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan
        vif_data.append({'Feature': name, 'VIF': vif})
    return pd.DataFrame(vif_data)


def compute_mutual_info(X_proc, labels, feature_names):
    if labels is None:
        return pd.DataFrame()
    mi = mutual_info_classif(X_proc, labels, random_state=42)
    return pd.DataFrame({
        'Feature': feature_names,
        'Mutual_Info': mi,
    }).sort_values('Mutual_Info', ascending=False)


def compute_rf_importance(X_proc, labels, feature_names):
    if labels is None:
        return pd.DataFrame()
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_proc, labels)
    return pd.DataFrame({
        'Feature': feature_names,
        'RF_Importance': rf.feature_importances_,
    }).sort_values('RF_Importance', ascending=False)


def main():
    print("=" * 60)
    print("5维精简特征 — 特征重要性 / 冗余度分析")
    print("=" * 60)

    print("\n--- 加载数据 ---")
    df = load_all_data()
    print(f"总类器官数: {len(df)}")

    print("\n--- 预处理 (5D) ---")
    preprocessor = Preprocessor(mode='reduced')
    X = preprocessor.fit_transform(df[REDUCED_RAW_FEATURES])
    feature_names = preprocessor.get_feature_names()
    print(f"处理后的特征: {feature_names}")

    labels = get_kmeans_labels(df, REDUCED_RAW_FEATURES)

    # 计算 Cavity_Ratio 用于原始空间相关分析
    df['Cavity_Ratio'] = df['Cavity_Volume'] / (df['Organoids_Volume_Fill'].clip(lower=1))

    print("\n--- 相关性分析 ---")
    plot_correlation_heatmaps(df, X, feature_names, os.path.join(FIGURES_DIR, 'reduced_correlation_heatmap.png'))

    spearman_corr = compute_spearman_corr(df, REDUCED_RAW_FEATURES + ['Cavity_Ratio'])
    print("\nSpearman 高相关对 (|r| > 0.8):")
    high_corr = []
    cols = spearman_corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = spearman_corr.iloc[i, j]
            if abs(r) > 0.8:
                high_corr.append((cols[i], cols[j], r))
                print(f"  {cols[i]} vs {cols[j]}: r={r:.3f}")
    if not high_corr:
        print("  无 |r| > 0.8 的特征对")

    print("\n--- VIF (方差膨胀因子, 5D) ---")
    vif_df = compute_vif(X, feature_names)
    print(vif_df.to_string(index=False))
    high_vif = vif_df[vif_df['VIF'] > 10]
    if not high_vif.empty:
        print(f"\n  [注意] VIF > 10 的特征（存在多重共线性）:")
        print(high_vif.to_string(index=False))
    else:
        print("\n  [PASS] 所有特征 VIF <= 10，共线性在可控范围！")

    print("\n--- 互信息 (非线性关联) ---")
    mi_df = compute_mutual_info(X, labels, feature_names)
    if not mi_df.empty:
        print(mi_df.to_string(index=False))
    else:
        print("  无标签，跳过互信息计算。")

    print("\n--- 随机森林特征重要性 ---")
    rf_df = compute_rf_importance(X, labels, feature_names)
    if not rf_df.empty:
        print(rf_df.to_string(index=False))
    else:
        print("  无标签，跳过 RF 重要性计算。")

    print("\n--- 保存报告 ---")
    excel_path = os.path.join(REPORTS_DIR, 'reduced_feature_importance.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        vif_df.to_excel(writer, sheet_name='VIF', index=False)
        if not mi_df.empty:
            mi_df.to_excel(writer, sheet_name='Mutual_Info', index=False)
        if not rf_df.empty:
            rf_df.to_excel(writer, sheet_name='RF_Importance', index=False)
        spearman_corr.to_excel(writer, sheet_name='Spearman_Correlation')
        if high_corr:
            pd.DataFrame(high_corr, columns=['Feature_A', 'Feature_B', 'Spearman_r']).to_excel(
                writer, sheet_name='High_Correlation_Pairs', index=False)

    print(f"  特征分析报告已保存 -> {excel_path}")
    print("\n[Done] 5维特征分析完成。")


if __name__ == "__main__":
    main()
