import os

import pandas as pd
from scipy.stats import pearsonr


BASE_DIR = r"Data\FXN_2023_new（ICC）"
FILES = {
    "去散射系数": os.path.join(BASE_DIR, "FXN_2023_PCA去散射系数.xlsx"),
    "全部特征": os.path.join(BASE_DIR, "FXN_2023_PCA全部.xlsx"),
}


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


def summarize_file(path):
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
        if start_col is None or end_col is None:
            raise ValueError(f"{path} 中找不到差值列，且无法由起点/终点计算，现有列: {list(df.columns)}")
        diff_col = "__derived_diff__"
        df[diff_col] = df[end_col] - df[start_col]
    else:
        if df[[diff_col, atp_col]].dropna().shape[0] < 3:
            if start_col is None or end_col is None:
                raise ValueError(f"{path} 中差值列不可用，且无法由起点/终点计算，现有列: {list(df.columns)}")
            diff_col = "__derived_diff__"
            df[diff_col] = df[end_col] - df[start_col]

    r_start, p_start, n_start = compute_corr(df, start_col, atp_col)
    r_end, p_end, n_end = compute_corr(df, end_col, atp_col)
    r_diff, p_diff, n_diff = compute_corr(df, diff_col, atp_col)

    return {
        "file": path,
        "start_col": start_col,
        "end_col": end_col,
        "diff_col": diff_col,
        "atp_col": atp_col,
        "n": min(n_start, n_end, n_diff),
        "r_start": r_start,
        "p_start": p_start,
        "r_end": r_end,
        "p_end": p_end,
        "r_diff": r_diff,
        "p_diff": p_diff,
    }


def main():
    rows = []
    print("Pearson correlation with ATP")
    print("=" * 80)

    for label, path in FILES.items():
        res = summarize_file(path)
        rows.append(
            {
                "文件": label,
                "起点r": res["r_start"],
                "起点p": res["p_start"],
                "终点r": res["r_end"],
                "终点p": res["p_end"],
                "差值r": res["r_diff"],
                "差值p": res["p_diff"],
                "n": res["n"],
                "起点列": res["start_col"],
                "终点列": res["end_col"],
                "差值列": res["diff_col"],
            }
        )

    table = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if len(table) == 2:
        base = table.iloc[0]
        other = table.iloc[1]
        print("=" * 80)
        print(
            f"差值相关性对比: {other['文件']} 相比 {base['文件']}，|r| 提升 {abs(other['差值r']) - abs(base['差值r']):.6f}"
        )
        print(
            f"终点相关性对比: {other['文件']} 相比 {base['文件']}，|r| 提升 {abs(other['终点r']) - abs(base['终点r']):.6f}"
        )
        print(
            f"起点相关性对比: {other['文件']} 相比 {base['文件']}，|r| 提升 {abs(other['起点r']) - abs(base['起点r']):.6f}"
        )


if __name__ == "__main__":
    main()