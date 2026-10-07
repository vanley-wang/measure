"""
Generate paper-style clustered bar charts for the nnUNet dataset.

Outputs:
  - reports/figures/nnunet_cluster_bar.png
  - reports/figures/nnunet_cluster_bar_stats.xlsx

The figure contains two rows:
  - Count of organoids per cluster (Day3 vs Day5)
  - Mean OAC per cluster (Day3 vs Day5)

The four columns correspond to Control, 20 μM, 40 μM, and 80 μM.
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


DEFAULT_BASE_DIR = os.path.join('Data', 'nnUNet_FXN_2023')
DEFAULT_OUTPUT_DIR = os.path.join('reports', 'figures')

DAY_MAP = {
    'FXN_0701': 'Day3',
    'FXN_0703': 'Day5',
}

CONC_ORDER = [0, 20, 40, 80]
CONC_LABELS = {
    0: 'Control',
    20: '20 μM',
    40: '40 μM',
    80: '80 μM',
}

WELL_CONC_MAP = {
    'E11': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 'F7': 0, 'F8': 0, 'F9': 0, 'F10': 0, 'F11': 0,
    'B11': 0, 'C11': 0, 'D11': 0,
    'B2': 20, 'B3': 20, 'B4': 20, 'C2': 20, 'C3': 20, 'C4': 20,
    'B5': 40, 'B6': 40, 'B7': 40, 'C5': 40, 'C6': 40, 'C7': 40,
    'B8': 80, 'B9': 80, 'B10': 80, 'C8': 80, 'C9': 80, 'C10': 80,
}

CLUSTER_NAMES = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
DAY_COLORS = {
    'Day3': '#1f77b4',
    'Day5': '#d62728',
}


def infer_concentration(well_name: str) -> int:
    prefix = well_name.split('_')[0].upper()
    return WELL_CONC_MAP.get(prefix, -1)


def find_merge_dir(root: str) -> str | None:
    for candidate in ('cluster_merge_GMM', 'cluster_merge'):
        path = os.path.join(root, candidate)
        if os.path.exists(path):
            return path
    return None


def safe_sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def load_well_level_stats(base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []

    for batch, day in DAY_MAP.items():
        root = os.path.join(base_dir, batch)
        merge_dir = find_merge_dir(root)
        if merge_dir is None:
            continue

        for fp in sorted(glob.glob(os.path.join(merge_dir, '*_merge.xlsx'))):
            well_name = os.path.basename(fp).replace('_merge.xlsx', '')
            df = pd.read_excel(fp)
            if 'Cluster' not in df.columns:
                continue

            conc = infer_concentration(well_name)
            for cluster_id in range(4):
                sub = df[df['Cluster'] == cluster_id]
                records.append({
                    'batch': batch,
                    'day': day,
                    'well': well_name,
                    'conc': conc,
                    'cluster': cluster_id,
                    'count': int(len(sub)),
                    'oac_mean': float(sub['Scatt_Mean'].mean()) if len(sub) else np.nan,
                    'oac_std': float(sub['Scatt_Mean'].std(ddof=1)) if len(sub) > 1 else np.nan,
                })

    well_stats = pd.DataFrame(records)
    if well_stats.empty:
        return well_stats, well_stats

    group_stats = (
        well_stats[well_stats['conc'] >= 0]
        .groupby(['day', 'conc', 'cluster'], as_index=False)
        .agg(
            n_wells=('well', 'nunique'),
            count_mean=('count', 'mean'),
            count_sem=('count', safe_sem),
            oac_mean=('oac_mean', 'mean'),
            oac_sem=('oac_mean', safe_sem),
        )
    )

    return well_stats, group_stats


def plot_bar_figure(group_stats: pd.DataFrame, out_png: str):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')
    x = np.arange(4)
    width = 0.34

    def safe_max(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return 0.0
        return float(np.nanmax(arr))

    for col_idx, conc in enumerate(CONC_ORDER):
        conc_label = CONC_LABELS[conc]
        subset = group_stats[group_stats['conc'] == conc].copy()

        for row_idx, metric in enumerate(['count', 'oac']):
            ax = axes[row_idx, col_idx]
            day3 = subset[subset['day'] == 'Day3'].sort_values('cluster')
            day5 = subset[subset['day'] == 'Day5'].sort_values('cluster')

            if metric == 'count':
                y3 = day3['count_mean'].fillna(0).to_numpy() if len(day3) else np.zeros(4)
                e3 = day3['count_sem'].fillna(0).to_numpy() if len(day3) else np.zeros(4)
                y5 = day5['count_mean'].fillna(0).to_numpy() if len(day5) else np.zeros(4)
                e5 = day5['count_sem'].fillna(0).to_numpy() if len(day5) else np.zeros(4)
                ylabel = 'Count (n)'
                ax.set_yscale('log')
                ax.set_ylim(0.8, max(5.0, safe_max([safe_max(y3), safe_max(y5)]) * 1.8))
            else:
                y3 = day3['oac_mean'].fillna(0).to_numpy() if len(day3) else np.zeros(4)
                e3 = day3['oac_sem'].fillna(0).to_numpy() if len(day3) else np.zeros(4)
                y5 = day5['oac_mean'].fillna(0).to_numpy() if len(day5) else np.zeros(4)
                e5 = day5['oac_sem'].fillna(0).to_numpy() if len(day5) else np.zeros(4)
                ylabel = 'OAC Mean (mm$^{-1}$)'
                ymax = safe_max([safe_max(y3), safe_max(y5)])
                ax.set_ylim(0, max(0.5, ymax * 1.25))

            ax.bar(x - width / 2, y3, width, yerr=e3, color=DAY_COLORS['Day3'],
                   edgecolor='black', linewidth=0.6, capsize=3, label='Day 3')
            ax.bar(x + width / 2, y5, width, yerr=e5, color=DAY_COLORS['Day5'],
                   edgecolor='black', linewidth=0.6, capsize=3, label='Day 5')

            ax.set_xticks(x)
            ax.set_xticklabels(CLUSTER_NAMES, fontsize=9, rotation=0)
            ax.set_title(conc_label, fontsize=12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.set_ylabel(ylabel, fontsize=10)

            if row_idx == 0:
                ax.legend(fontsize=8, frameon=False, loc='upper left')

            # panel labels similar to paper style
            panel_prefix = 'c' if row_idx == 0 else 'd'
            ax.text(0.02, 0.96, f'({panel_prefix}{col_idx + 1})', transform=ax.transAxes,
                    ha='left', va='top', fontsize=11, fontweight='bold')

    fig.suptitle('nnUNet 数据集的类器官聚类统计图', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate paper-style cluster bar charts')
    parser.add_argument('--base-dir', default=DEFAULT_BASE_DIR, help='Dataset root directory')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Directory for figures and stats')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    well_stats, group_stats = load_well_level_stats(args.base_dir)
    if well_stats.empty:
        print('[WARN] 未找到可用的 cluster_merge 数据')
        return

    out_png = os.path.join(args.output_dir, 'nnunet_cluster_bar.png')
    out_xlsx = os.path.join(args.output_dir, 'nnunet_cluster_bar_stats.xlsx')

    plot_bar_figure(group_stats, out_png)

    with pd.ExcelWriter(out_xlsx) as writer:
        well_stats.to_excel(writer, sheet_name='well_stats', index=False)
        group_stats.to_excel(writer, sheet_name='group_stats', index=False)

    print(f'已生成: {out_png}')
    print(f'已生成: {out_xlsx}')


if __name__ == '__main__':
    main()