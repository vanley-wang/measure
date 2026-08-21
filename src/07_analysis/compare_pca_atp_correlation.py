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


def load_and_score(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    df = pd.read_excel(path)
    score_col = find_column(df.columns, ["得分差值", "Result_Diff", "Diff", "差值", "Result"])
    atp_col = find_column(df.columns, ["ATP"])

    if score_col is None:
        raise ValueError(f"{path} 中找不到得分列，现有列: {list(df.columns)}")
    if atp_col is None:
        raise ValueError(f"{path} 中找不到 ATP 列，现有列: {list(df.columns)}")

    valid = df[[score_col, atp_col]].dropna()
    if len(valid) < 3:
        raise ValueError(f"{path} 可用于相关性计算的数据不足，当前只有 {len(valid)} 行")

    r, p = pearsonr(valid[score_col], valid[atp_col])
    return {
        "file": path,
        "score_col": score_col,
        "atp_col": atp_col,
        "n": len(valid),
        "r": r,
        "p": p,
    }


def main():
    results = {}
    print("Pearson correlation with ATP")
    print("=" * 60)

    for label, path in FILES.items():
        res = load_and_score(path)
        results[label] = res
        print(f"{label}: r = {res['r']:.6f}, p = {res['p']:.6g}, n = {res['n']}"
              f"  ({res['score_col']} vs {res['atp_col']})")

    if len(results) == 2:
        labels = list(results.keys())
        r1 = results[labels[0]]["r"]
        r2 = results[labels[1]]["r"]
        delta = abs(r2) - abs(r1)
        direction = "提升" if delta > 0 else "下降"
        print("=" * 60)
        print(f"对比结论: {labels[1]} 相比 {labels[0]}，|r| {direction} {abs(delta):.6f}")


if __name__ == "__main__":
    main()