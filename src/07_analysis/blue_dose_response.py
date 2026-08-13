import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

"""
Blue 类器官剂量响应深度分析

结合聚类结果与药物浓度，分析受损（Blue/Cluster 3）类器官比例
随 Icaritin 浓度变化的趋势。同时对比 Control 与加药组的孔体积变化。

输出：
  reports/figures/blue_dose_response.png
  reports/blue_dose_analysis.xlsx
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress

from cluster_utils import (
    RAW_FEATURES,
    REDUCED_RAW_FEATURES,
    Preprocessor,
    load_model_package,
)

warnings.filterwarnings('ignore')

# ================= 配置 =================
DATA_FOLDERS = [
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]

MODEL_PATHS = {
    'KMeans-10d': 'model/Kmeans-scatt.pickle',
    'GMM-10d': 'model/GMM-scatt.pickle',
    'KMeans-5d': 'model/Kmeans-5d.pickle',
    'GMM-5d': 'model/GMM-5d.pickle',
}

# Well name prefix -> drug concentration (μM)
WELL_CONC_MAP = {
    'E11': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 'F7': 0, 'F8': 0, 'F9': 0, 'F10': 0, 'F11': 0,
    'B11': 0, 'C11': 0, 'D11': 0,
    'B2': 20, 'B3': 20, 'B4': 20, 'C2': 20, 'C3': 20, 'C4': 20,
    'B5': 40, 'B6': 40, 'B7': 40, 'C5': 40, 'C6': 40, 'C7': 40,
    'B8': 80, 'B9': 80, 'B10': 80, 'C8': 80, 'C9': 80, 'C10': 80,
}

# 孔体积变化率 (Day 5 vs Day 3), 来自实验记录
WELL_VOLUME_CHANGE = {
    0: 213.3,   # Control
    20: 47.9,
    40: 9.1,
    80: 13.7,
}

REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==========================================


def infer_concentration(well_name: str) -> int:
    prefix = well_name.split('_')[0].upper()
    return WELL_CONC_MAP.get(prefix, -1)


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


def get_model_labels(df, model_path):
    """Load model and predict labels."""
    if not os.path.exists(model_path):
        return None
    try:
        pkg = load_model_package(model_path)
        features = pkg.get('feature_names', RAW_FEATURES)
        preprocessor = pkg.get('preprocessor')
        scaler = pkg.get('scaler')
        model = pkg.get('kmeans') or pkg.get('gmm') or pkg['model']
        raw_to_final = pkg.get('raw_to_final', {})

        # 兼容旧 preprocessor
        if preprocessor is not None:
            if not hasattr(preprocessor, 'mode'):
                preprocessor.mode = 'full'
            if not hasattr(preprocessor, '_log_features'):
                if preprocessor.mode == 'reduced':
                    preprocessor._log_features = ['Organoids_Volume_Fill']
                    preprocessor._processed_features = ['Organoids_Volume_Fill', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'Cavity_Ratio']
                else:
                    preprocessor._log_features = ['Organoids_Volume_Fill', 'Organoids_Surface', 'LongAxis', 'ShortAxis', 'Cavity_Volume']
                    preprocessor._processed_features = ['Organoids_Volume_Fill', 'Organoids_Surface', 'LongAxis', 'ShortAxis', 'Cavity_Volume', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'CavityNum', 'Cavity_Ratio']

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
        print(f"[WARN] Failed to load {model_path}: {e}")
        return None


def compute_dose_stats(df, labels):
    """Compute per-concentration Blue fraction and related stats."""
    df = df.copy()
    df['Cluster'] = labels
    df['_conc'] = df['_well'].apply(infer_concentration)

    # Per-well stats first
    well_stats = df.groupby('_well').agg(
        Conc=('_conc', 'first'),
        Total=('Cluster', 'size'),
        Blue=('Cluster', lambda s: (s == 3).sum()),
        Red=('Cluster', lambda s: (s == 0).sum()),
        Volume_Mean=('Organoids_Volume_Fill', 'mean'),
        OAC_Mean=('Scatt_Mean', 'mean'),
    )
    well_stats['Blue_Fraction'] = well_stats['Blue'] / well_stats['Total']
    well_stats['Red_Fraction'] = well_stats['Red'] / well_stats['Total']

    # Aggregate by concentration
    conc_stats = well_stats.groupby('Conc').agg(
        N_Wells=('Blue', 'size'),
        Total_Organoids=('Total', 'sum'),
        Blue_Mean=('Blue_Fraction', 'mean'),
        Blue_Std=('Blue_Fraction', 'std'),
        Red_Mean=('Red_Fraction', 'mean'),
        Volume_Mean=('Volume_Mean', 'mean'),
        OAC_Mean=('OAC_Mean', 'mean'),
    ).reset_index()

    return well_stats, conc_stats


def plot_blue_dose_response(all_conc_stats, save_path):
    """Plot Blue fraction vs concentration for all 4 models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = {'KMeans-10d': '#1f77b4', 'GMM-10d': '#ff7f0e',
              'KMeans-5d': '#2ca02c', 'GMM-5d': '#d62728'}

    for idx, (name, conc_stats) in enumerate(all_conc_stats.items()):
        ax = axes[idx // 2, idx % 2]
        conc_stats = conc_stats[conc_stats['Conc'] >= 0].sort_values('Conc')
        x = conc_stats['Conc'].values
        y = conc_stats['Blue_Mean'].values
        yerr = conc_stats['Blue_Std'].fillna(0).values

        ax.errorbar(x, y * 100, yerr=yerr * 100, fmt='o-', color=colors[name],
                    capsize=5, capthick=2, lw=2, markersize=8, label=name)
        ax.set_xlabel('Icaritin Concentration (μM)', fontsize=11)
        ax.set_ylabel('Blue Cluster Fraction (%)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks([0, 20, 40, 80])
        ax.set_xlim(-5, 95)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add well volume change annotation
        for conc, vol_change in WELL_VOLUME_CHANGE.items():
            if conc in x:
                ax.annotate(f'+{vol_change}%', xy=(conc, y[list(x).index(conc)] * 100),
                           xytext=(0, 12), textcoords='offset points',
                           ha='center', fontsize=8, color='gray')

        # Spearman correlation
        if len(x) >= 3:
            rho, pval = spearmanr(x, y)
            lr = linregress(x, y)
            ax.axline((x[0], lr.intercept + lr.slope * x[0]),
                     slope=lr.slope, color=colors[name], linestyle='--', alpha=0.5)
            sig = '*' if pval < 0.05 else ''
            ax.text(0.95, 0.05, f"Spearman rho={rho:.3f}{sig}\np={pval:.3f}",
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Blue dose-response 图已保存 -> {save_path}")


def plot_combined_blue_dose(all_conc_stats, save_path):
    """Combined plot: all 4 models on one figure with well volume."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = {'KMeans-10d': '#1f77b4', 'GMM-10d': '#ff7f0e',
              'KMeans-5d': '#2ca02c', 'GMM-5d': '#d62728'}

    for name, conc_stats in all_conc_stats.items():
        conc_stats = conc_stats[conc_stats['Conc'] >= 0].sort_values('Conc')
        x = conc_stats['Conc'].values
        y = conc_stats['Blue_Mean'].values
        ax1.plot(x, y * 100, 'o-', color=colors[name], lw=2, markersize=8, label=name)

    ax1.set_xlabel('Icaritin Concentration (μM)', fontsize=12)
    ax1.set_ylabel('Blue Cluster Fraction (%)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks([0, 20, 40, 80])
    ax1.set_xlim(-5, 95)
    ax1.spines['top'].set_visible(False)

    # Secondary axis: well volume change
    ax2 = ax1.twinx()
    vol_x = list(WELL_VOLUME_CHANGE.keys())
    vol_y = list(WELL_VOLUME_CHANGE.values())
    ax2.bar(vol_x, vol_y, width=8, alpha=0.2, color='gray', label='Well volume change')
    ax2.set_ylabel('Well Volume Change (%)', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.spines['top'].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    ax1.set_title('Blue (Damaged) Organoid Fraction vs Icaritin Concentration\n(4 Models + Well Volume Change)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  综合剂量响应图已保存 -> {save_path}")


def main():
    print("=" * 60)
    print("Blue 类器官剂量响应深度分析")
    print("=" * 60)

    # 1. Load data
    print("\n--- 加载数据 ---")
    df = load_all_data()
    print(f"总类器官数: {len(df)}")

    # 2. Evaluate each model
    print("\n--- 计算各模型剂量响应统计 ---")
    all_conc_stats = {}
    all_well_stats = {}

    for name, path in MODEL_PATHS.items():
        print(f"\n  {name}...")
        labels = get_model_labels(df, path)
        if labels is None:
            print(f"    [SKIP] 模型不可用")
            continue

        well_stats, conc_stats = compute_dose_stats(df, labels)
        all_conc_stats[name] = conc_stats
        all_well_stats[name] = well_stats

        # Print summary
        for _, row in conc_stats.iterrows():
            print(f"    Conc={row['Conc']:3.0f}μM: Blue={row['Blue_Mean']*100:.1f}% ± {row['Blue_Std']*100:.1f}%  (N={row['N_Wells']} wells, n={row['Total_Organoids']} organoids)")

    if len(all_conc_stats) < 2:
        print("\n[ERR] 可用模型不足，无法继续。")
        return

    # 3. Plotting
    print("\n--- 生成图表 ---")
    plot_blue_dose_response(all_conc_stats, os.path.join(FIGURES_DIR, 'blue_dose_response.png'))
    plot_combined_blue_dose(all_conc_stats, os.path.join(FIGURES_DIR, 'blue_dose_response_combined.png'))

    # 4. Statistical summary
    print("\n--- 统计汇总 ---")
    summary_rows = []
    for name, conc_stats in all_conc_stats.items():
        conc_stats = conc_stats[conc_stats['Conc'] >= 0].sort_values('Conc')
        x = conc_stats['Conc'].values
        y = conc_stats['Blue_Mean'].values
        if len(x) >= 3:
            rho, pval = spearmanr(x, y)
            lr = linregress(x, y)
            summary_rows.append({
                'Model': name,
                'Control_Blue_Pct': y[0] * 100,
                '20uM_Blue_Pct': y[1] * 100 if len(y) > 1 else np.nan,
                '40uM_Blue_Pct': y[2] * 100 if len(y) > 2 else np.nan,
                '80uM_Blue_Pct': y[3] * 100 if len(y) > 3 else np.nan,
                'Spearman_Rho': rho,
                'Spearman_P': pval,
                'Linear_Slope': lr.slope,
                'Linear_R2': lr.rvalue ** 2,
                'Trend': 'Up' if rho > 0 else 'Down' if rho < 0 else 'Flat',
            })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # 5. Save Excel
    print("\n--- 保存报告 ---")
    excel_path = os.path.join(REPORTS_DIR, 'blue_dose_analysis.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        for name, conc_stats in all_conc_stats.items():
            sheet_name = name.replace('-', '_')
            conc_stats.to_excel(writer, sheet_name=f'{sheet_name}_Conc', index=False)
        for name, well_stats in all_well_stats.items():
            sheet_name = name.replace('-', '_')
            well_stats.to_excel(writer, sheet_name=f'{sheet_name}_Well', index=False)

    print(f"  报告已保存 -> {excel_path}")
    print("\n[Done] Blue 剂量响应分析完成。")


if __name__ == "__main__":
    main()
