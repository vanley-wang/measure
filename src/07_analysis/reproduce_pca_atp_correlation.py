import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr


DEFAULT_FILE = r"Data\FXN_2023_new（ICC）\FXN_2023_PCA全部.xlsx"
DEFAULT_BATCH_FILES = {
    "去散射系数": r"Data\FXN_2023_new（ICC）\FXN_2023_PCA去散射系数.xlsx",
    "全部特征": r"Data\FXN_2023_new（ICC）\FXN_2023_PCA全部.xlsx",
}
OUTPUT_DIR = r"reports\figures"


def find_column(columns, candidates):
    lower_map = {str(col).strip().lower(): col for col in columns}
    for name in candidates:
        key = name.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def safe_pearson(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        raise ValueError(f"可用于相关性计算的数据不足，当前只有 {len(valid)} 行")
    r, p = pearsonr(valid["x"], valid["y"])
    return r, p, len(valid)


def load_workbook(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    return pd.read_excel(path)


def get_single_file_values(df):
    atp_col = find_column(df.columns, ["ATP"])
    start_col = find_column(df.columns, ["Result", "Score", "得分", "综合得分", "PCA_Score"])
    end_col = find_column(df.columns, ["Result1", "Result_End", "终点", "终点得分", "终点结果"])
    diff_col = find_column(df.columns, ["得分差值", "Result_Diff", "Diff", "差值"])

    if atp_col is None:
        raise ValueError(f"找不到 ATP 列，现有列: {list(df.columns)}")
    if start_col is None:
        raise ValueError(f"找不到起点得分列，现有列: {list(df.columns)}")
    if end_col is None:
        raise ValueError(f"找不到终点得分列，现有列: {list(df.columns)}")

    diff_series = None
    diff_label = diff_col
    if diff_col is not None and df[diff_col].notna().sum() >= 3:
        diff_series = df[diff_col]
    else:
        diff_series = df[end_col] - df[start_col]
        diff_label = "Result.1 - Result"

    start_r, start_p, n_start = safe_pearson(df[start_col], df[atp_col])
    end_r, end_p, n_end = safe_pearson(df[end_col], df[atp_col])
    diff_r, diff_p, n_diff = safe_pearson(diff_series, df[atp_col])

    return {
        "atp_col": atp_col,
        "start_col": start_col,
        "end_col": end_col,
        "diff_col": diff_label,
        "n": min(n_start, n_end, n_diff),
        "start_r": start_r,
        "start_p": start_p,
        "end_r": end_r,
        "end_p": end_p,
        "diff_r": diff_r,
        "diff_p": diff_p,
    }


def plot_single_scatter(x, y, title, x_label, y_label, out_path):
    plt.figure(figsize=(5.8, 4.6))
    plt.scatter(x, y, s=36, alpha=0.85)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) >= 2:
        m, b = np.polyfit(valid["x"], valid["y"], 1)
        xs = valid["x"].sort_values()
        plt.plot(xs, m * xs + b, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_single(path, save_fig=False):
    df = load_workbook(path)
    stats = get_single_file_values(df)

    print("Pearson correlation with ATP")
    print("=" * 60)
    print(f"文件: {path}")
    print(f"起点: r = {stats['start_r']:.6f}, p = {stats['start_p']:.6g}, n = {stats['n']}  ({stats['start_col']} vs {stats['atp_col']})")
    print(f"终点: r = {stats['end_r']:.6f}, p = {stats['end_p']:.6g}, n = {stats['n']}  ({stats['end_col']} vs {stats['atp_col']})")
    print(f"差值: r = {stats['diff_r']:.6f}, p = {stats['diff_p']:.6g}, n = {stats['n']}  ({stats['diff_col']} vs {stats['atp_col']})")

    if save_fig:
        ensure_dir(OUTPUT_DIR)
        fig_base = os.path.splitext(os.path.basename(path))[0]
        for col_name, label, suffix in [
            (stats['start_col'], '起点 Result vs ATP', 'start'),
            (stats['end_col'], '终点 Result vs ATP', 'end'),
            (stats['diff_col'], '差值 vs ATP', 'diff'),
        ]:
            out_path = os.path.join(OUTPUT_DIR, f"{fig_base}_{suffix}.png")
            plt.figure(figsize=(5.8, 4.6))
            plt.scatter(df[col_name], df[stats['atp_col']], s=36, alpha=0.85)
            plt.xlabel(col_name)
            plt.ylabel('ATP')
            plt.title(label)
            valid = pd.DataFrame({"x": df[col_name], "y": df[stats['atp_col']]}).dropna()
            if len(valid) >= 2:
                m, b = pd.Series(valid["y"]).corr(valid["x"]), None
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()


def run_batch(save_table=False):
    rows = []
    print("Pearson correlation with ATP")
    print("=" * 80)
    for label, path in DEFAULT_BATCH_FILES.items():
        df = load_workbook(path)
        stats = get_single_file_values(df)
        rows.append({
            "文件": label,
            "起点r": stats["start_r"],
            "起点p": stats["start_p"],
            "终点r": stats["end_r"],
            "终点p": stats["end_p"],
            "差值r": stats["diff_r"],
            "差值p": stats["diff_p"],
            "n": stats["n"],
            "起点列": stats["start_col"],
            "终点列": stats["end_col"],
            "差值列": stats["diff_col"],
        })
    table = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if len(table) == 2:
        base = table.iloc[0]
        other = table.iloc[1]
        print("=" * 80)
        print(f"差值相关性对比: {other['文件']} 相比 {base['文件']}，|r| 提升 {abs(other['差值r']) - abs(base['差值r']):.6f}")
        print(f"终点相关性对比: {other['文件']} 相比 {base['文件']}，|r| 提升 {abs(other['终点r']) - abs(base['终点r']):.6f}")
        print(f"起点相关性对比: {other['文件']} 相比 {base['文件']}，|r| 提升 {abs(other['起点r']) - abs(base['起点r']):.6f}")

    if save_table:
        ensure_dir("reports")
        out_path = os.path.join("reports", "pca_atp_correlation_summary.xlsx")
        table.to_excel(out_path, index=False)
        print(f"已保存汇总表: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce PCA/ATP correlations from existing Excel outputs")
    parser.add_argument("--mode", choices=["single", "batch"], default="single", help="single: one workbook; batch: compare two workbooks")
    parser.add_argument("--file", default=DEFAULT_FILE, help="Workbook path for single mode")
    parser.add_argument("--save-table", action="store_true", help="Save summary table to reports/ directory in batch mode")
    args = parser.parse_args()

    if args.mode == "single":
        run_single(args.file)
    else:
        run_batch(save_table=args.save_table)


if __name__ == "__main__":
    main()