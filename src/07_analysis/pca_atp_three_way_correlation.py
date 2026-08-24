import argparse
import os

import pandas as pd
from scipy.stats import pearsonr


DEFAULT_FILE = r"Data\FXN_2023_new（ICC）\FXN_2023_PCA全部.xlsx"


def find_column(columns, candidates):
    lower_map = {str(col).strip().lower(): col for col in columns}
    for name in candidates:
        key = name.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def compute_corr(df, value_col, atp_col):
    valid = df[[value_col, atp_col]].dropna()
    if len(valid) < 3:
        raise ValueError(f"{value_col} 与 {atp_col} 可用数据不足，当前只有 {len(valid)} 行")
    r, p = pearsonr(valid[value_col], valid[atp_col])
    return r, p, len(valid)


def main():
    parser = argparse.ArgumentParser(description="Compute three Pearson correlations against ATP from one PCA workbook")
    parser.add_argument("--file", default=DEFAULT_FILE, help="PCA result workbook path")
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    df = pd.read_excel(path)

    atp_col = find_column(df.columns, ["ATP"])
    start_col = find_column(df.columns, ["Result", "Score", "得分", "综合得分", "PCA_Score"])
    end_col = find_column(df.columns, ["Result.1", "Result_End", "终点", "终点得分", "终点结果"])
    diff_col = find_column(df.columns, ["得分差值", "Result_Diff", "Diff", "差值"])

    if atp_col is None:
        raise ValueError(f"{path} 中找不到 ATP 列，现有列: {list(df.columns)}")
    if start_col is None:
        raise ValueError(f"{path} 中找不到起点得分列，现有列: {list(df.columns)}")
    if end_col is None:
        raise ValueError(f"{path} 中找不到终点得分列，现有列: {list(df.columns)}")
    if diff_col is None:
        raise ValueError(f"{path} 中找不到差值列，现有列: {list(df.columns)}")

    print("Pearson correlation with ATP")
    print("=" * 60)

    r_start, p_start, n_start = compute_corr(df, start_col, atp_col)
    r_end, p_end, n_end = compute_corr(df, end_col, atp_col)
    r_diff, p_diff, n_diff = compute_corr(df, diff_col, atp_col)

    print(f"Result vs ATP: r = {r_start:.6f}, p = {p_start:.6g}, n = {n_start}  ({start_col} vs {atp_col})")
    print(f"终点 vs ATP:   r = {r_end:.6f}, p = {p_end:.6g}, n = {n_end}  ({end_col} vs {atp_col})")
    print(f"差值 vs ATP:   r = {r_diff:.6f}, p = {p_diff:.6g}, n = {n_diff}  ({diff_col} vs {atp_col})")

    print("=" * 60)
    print(f"|r| 排序: 差值={abs(r_diff):.6f}, 终点={abs(r_end):.6f}, 起点={abs(r_start):.6f}")


if __name__ == "__main__":
    main()