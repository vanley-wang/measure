import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_FILE = r"Data\FXN_2023_new（ICC）\FXN_2023_PCA全部.xlsx"
OUTPUT_DIR = r"reports\figures\pca_atp_scatter"

# 论文需要展示的 24 个 wells
PAPER_WELLS = {
    'E11', 'F2', 'F6', 'F8', 'F9', 'F11',
    'B2', 'B3', 'B4', 'C2', 'C3', 'C4',
    'B5', 'B6', 'B7', 'C5', 'C6', 'C7',
    'B8', 'B9', 'B10', 'C8', 'C9', 'C10',
}

# 与现有分析脚本保持一致的孔位分组映射
WELL_CONC_MAP = {
    'E11': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 'F7': 0, 'F8': 0, 'F9': 0, 'F10': 0, 'F11': 0,
    'B11': 0, 'C11': 0, 'D11': 0,
    'B2': 20, 'B3': 20, 'B4': 20, 'C2': 20, 'C3': 20, 'C4': 20,
    'B5': 40, 'B6': 40, 'B7': 40, 'C5': 40, 'C6': 40, 'C7': 40,
    'B8': 80, 'B9': 80, 'B10': 80, 'C8': 80, 'C9': 80, 'C10': 80,
}

GROUP_COLORS = {
    0: '#7F7F7F',
    20: '#1F77B4',
    40: '#FF7F0E',
    80: '#D62728',
    -1: '#6C757D',
}

GROUP_LABELS = {
    0: 'Control',
    20: '20 μM',
    40: '40 μM',
    80: '80 μM',
    -1: 'Unknown',
}


def find_column(columns, candidates):
    lower_map = {str(col).strip().lower(): col for col in columns}
    for name in candidates:
        key = name.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def find_score_columns(df, requested_columns=None):
    if requested_columns:
        found = []
        for name in requested_columns:
            col = find_column(df.columns, [name])
            if col is None:
                raise ValueError(f"找不到指定得分列: {name}")
            found.append(col)
        return found

    candidates = ["得分差值", "Result", "Result1"]
    found = []
    for name in candidates:
        col = find_column(df.columns, [name])
        if col is not None and col not in found:
            found.append(col)

    if not found:
        numeric_cols = [c for c in df.columns if c != 'ATP' and pd.api.types.is_numeric_dtype(df[c])]
        found = numeric_cols[:1]

    return found


def infer_group_from_name(name_value):
    if pd.isna(name_value):
        return -1
    well_id = str(name_value).split('_')[0].upper()
    return WELL_CONC_MAP.get(well_id, -1)


def infer_well_id(name_value):
    if pd.isna(name_value):
        return None
    return str(name_value).split('_')[0].upper()


def is_paper_well(name_value):
    well_id = infer_well_id(name_value)
    return well_id in PAPER_WELLS


def safe_pearson(x, y):
    valid = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(valid) < 3:
        raise ValueError(f"可用于相关性计算的数据不足，当前只有 {len(valid)} 行")
    r, p = pearsonr(valid['x'], valid['y'])
    return r, p, len(valid)


def load_workbook(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    return pd.read_excel(path)


def fit_line(x, y):
    valid = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(valid) < 2:
        return None
    coeff = np.polyfit(valid['x'], valid['y'], 1)
    return coeff[0], coeff[1]


def plot_scatter(df, x_col, y_col, out_path, title=None):
    if x_col not in df.columns:
        raise ValueError(f"找不到得分列: {x_col}")
    if y_col not in df.columns:
        raise ValueError(f"找不到 ATP 列: {y_col}")

    plot_df = df[[x_col, y_col]].copy()
    if 'Name' in df.columns:
        plot_df['Name'] = df['Name']
        plot_df = plot_df[plot_df['Name'].apply(is_paper_well)]
    else:
        raise ValueError('找不到 Name 列，无法判断论文 wells')

    if len(plot_df) < 3:
        raise ValueError(f'论文 wells 过滤后可用于绘图的数据不足，当前只有 {len(plot_df)} 行')

    if 'Name' in df.columns:
        plot_df['Group'] = df['Name'].apply(infer_group_from_name)
    else:
        plot_df['Group'] = -1

    stats_r, stats_p, stats_n = safe_pearson(plot_df[x_col], plot_df[y_col])

    plt.figure(figsize=(5.8, 4.8))
    for group_value in [0, 20, 40, 80, -1]:
        group_df = plot_df[plot_df['Group'] == group_value].dropna(subset=[x_col, y_col])
        if len(group_df) == 0:
            continue
        plt.scatter(
            group_df[x_col],
            group_df[y_col],
            s=42,
            alpha=0.88,
            color=GROUP_COLORS[group_value],
            edgecolor='black',
            linewidth=0.3,
            label=GROUP_LABELS[group_value],
        )

    line = fit_line(plot_df[x_col], plot_df[y_col])
    if line is not None:
        slope, intercept = line
        xs = pd.Series(plot_df[x_col].dropna()).sort_values()
        plt.plot(xs, slope * xs + intercept, linestyle='--', color='#4D4D4D', linewidth=2)

    xlabel = x_col
    ylabel = y_col
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    if title is None:
        title = f'{x_col} vs {y_col}'
    plt.title(title, fontsize=12, fontweight='bold')

    annotation = f'Pearson r = {stats_r:.3f}\nP < 0.001' if stats_p < 0.001 else f'Pearson r = {stats_r:.3f}\nP = {stats_p:.3f}'
    plt.text(
        0.05,
        0.93,
        annotation,
        transform=plt.gca().transAxes,
        fontsize=9,
        va='top',
        bbox=dict(facecolor='white', edgecolor='#9AA0A6', boxstyle='square,pad=0.3', alpha=0.95),
    )

    plt.legend(loc='lower right', fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'score_col': x_col,
        'atp_col': y_col,
        'r': stats_r,
        'p': stats_p,
        'n': stats_n,
        'out_path': out_path,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate separate ATP vs score scatter plots, one file per score column')
    parser.add_argument('--file', default=DEFAULT_FILE, help='Workbook path')
    parser.add_argument('--scores', nargs='*', default=[], help='Score columns to plot. If omitted, auto-detect from the workbook')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Directory to save figures')
    parser.add_argument('--atp-col', default='ATP', help='ATP column name')
    args = parser.parse_args()

    df = load_workbook(args.file)
    atp_col = find_column(df.columns, [args.atp_col])
    if atp_col is None:
        raise ValueError(f"找不到 ATP 列: {args.atp_col}，现有列: {list(df.columns)}")

    score_cols = find_score_columns(df, args.scores)
    if not score_cols:
        raise ValueError(f"没有找到可画图的得分列，现有列: {list(df.columns)}")

    os.makedirs(args.output_dir, exist_ok=True)

    print('Separate ATP scatter plots')
    print('=' * 60)
    print(f'文件: {args.file}')
    print(f'ATP列: {atp_col}')
    print(f'得分列: {score_cols}')
    print(f'论文 wells: {sorted(PAPER_WELLS)}')

    summary_rows = []
    for score_col in score_cols:
        out_name = f"{os.path.splitext(os.path.basename(args.file))[0]}__{score_col}.png"
        out_path = os.path.join(args.output_dir, out_name)
        result = plot_scatter(
            df,
            score_col,
            atp_col,
            out_path,
            title=f'{score_col} vs ATP',
        )
        summary_rows.append(result)
        print(f"{score_col}: r = {result['r']:.6f}, p = {result['p']:.6g}, n = {result['n']} -> {out_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, 'scatter_summary.xlsx')
    summary_df.to_excel(summary_path, index=False)
    print(f"汇总表已保存: {summary_path}")


if __name__ == '__main__':
    main()