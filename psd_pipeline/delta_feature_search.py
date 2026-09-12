# ============================================================
# Delta 特征穷举搜索 — 直接与论文方法对比
# 核心问题：如果给 Delta 特征也做穷举搜索，r 能追上论文的 0.958 吗？
#
# 两组实验：
#   A) 论文 Delta: 基于论文 K6→4聚合表，先算Delta再PCA
#   B) PSD Delta:  基于原始PSD K=3+中位数，先算Delta再PCA
# ============================================================
import os
import sys
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATP_DB = {
    'B10': 601300, 'B11': 11180000, 'B2': 5391000, 'B3': 6538000,
    'B4': 7103000, 'B5': 1264000, 'B6': 2548000, 'B7': 1579000,
    'B8': 3637000, 'B9': 140300, 'C10': 211800, 'C11': 13930000,
    'C2': 4460000, 'C3': 8336000, 'C4': 6800000, 'C5': 330900,
    'C6': 238900, 'C7': 682100, 'C8': 211300, 'C9': 465900,
    'D11': 11240000, 'E11': 12840000, 'F10': 21910000, 'F11': 14700000,
    'F2': 26980000, 'F3': 14110000, 'F4': 13740000, 'F5': 17250000,
    'F6': 20320000, 'F7': 20000000, 'F8': 17170000, 'F9': 15830000
}

N_ITER = 100000
N_JOBS = -1
N_PC = 4


def build_delta_table_paper():
    """从论文聚合表构建 Delta 表：Day5 - Day3 逐孔逐特征"""
    path = os.path.join(OUTPUT_DIR, 'FXN_2023_Analysis.xlsx')
    df = pd.read_excel(path)
    df['Well_ID'] = df['Name'].apply(lambda x: str(x).split('_')[0])
    df['TimePoint'] = df['Name'].apply(lambda x: str(x).split('_')[1])

    feature_cols = [c for c in df.columns if c not in ('Name', 'Well_ID', 'TimePoint')]

    d3 = df[df['TimePoint'] == '0701'].set_index('Well_ID')[feature_cols]
    d5 = df[df['TimePoint'] == '0703'].set_index('Well_ID')[feature_cols]

    common = sorted(set(d3.index) & set(d5.index))
    delta = d5.loc[common] - d3.loc[common]
    delta = delta.reset_index()
    delta['ATP'] = delta['Well_ID'].map(ATP_DB)
    delta['ATP_Log'] = np.log10(delta['ATP'])

    # 移除常数列/NaN列
    valid_cols = []
    for c in feature_cols:
        vals = delta[c].dropna()
        if len(vals) > 0 and vals.nunique() > 1:
            valid_cols.append(c)

    return delta, valid_cols


def build_delta_table_psd():
    """从 PSD Delta 表加载并清理"""
    path = os.path.join(OUTPUT_DIR, 'psd_delta_features.xlsx')
    df = pd.read_excel(path)
    df['ATP'] = df['Well_ID'].map(ATP_DB)
    df['ATP_Log'] = np.log10(df['ATP'])

    feature_cols = [c for c in df.columns if c not in ('Well_ID', 'Name', 'TimePoint', 'ATP', 'ATP_Log')]
    valid_cols = []
    for c in feature_cols:
        vals = df[c].dropna()
        if len(vals) > 0 and vals.nunique() > 1:
            valid_cols.append(c)

    return df, valid_cols


def search_trial(seed, X_full, feature_pool, y_atp):
    rng = np.random.RandomState(seed)
    n_total = len(feature_pool)
    n_select = rng.randint(3, min(n_total + 1, 25))
    idx = rng.choice(n_total, n_select, replace=False)
    idx.sort()

    X_sub = X_full[:, idx]
    try:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_sub)
        pca = PCA(n_components=N_PC, random_state=42)
        scores = pca.fit_transform(X_std)
        var = pca.explained_variance_ratio_
        total_var = np.sum(var)
        if total_var == 0:
            return None
        w = var / total_var
        result = np.dot(scores, w)
    except Exception:
        return None

    # Delta 场景：Result = score，直接 vs ATP
    if np.std(result) == 0:
        return None
    r_diff, p_diff = pearsonr(result, y_atp)
    rho, _ = spearmanr(result, y_atp)

    names = [feature_pool[i] for i in idx]
    return {
        'n_feats': len(idx),
        '|r|': abs(r_diff),
        'r': r_diff,
        'rho': rho,
        'p': p_diff,
        'features': ', '.join(names)
    }


def run_search(label, delta_df, feature_pool):
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    print(f'  Wells: {len(delta_df)}, Features: {len(feature_pool)}')

    # 全特征基线
    X_full = delta_df[feature_pool].fillna(0).values
    y_atp = delta_df['ATP'].values.astype(float)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_full)
    pca = PCA(n_components=N_PC, random_state=42)
    scores = pca.fit_transform(X_std)
    var = pca.explained_variance_ratio_
    w = var / np.sum(var)
    result_full = np.dot(scores, w)

    r_full, p_full = pearsonr(result_full, y_atp)
    rho_full, _ = spearmanr(result_full, y_atp)
    print(f'  Full features ({len(feature_pool)}): |r|={abs(r_full):.4f}, rho={rho_full:.4f}, p={p_full:.6f}')

    # 穷举搜索
    print(f'  Searching {N_ITER} combinations...')
    seeds = [random.randint(0, 100000000) for _ in range(N_ITER)]
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=3)(
        delayed(search_trial)(s, X_full, feature_pool, y_atp) for s in seeds
    )
    t1 = time.time()
    print(f'  Done in {t1 - t0:.1f}s')

    clean = [r for r in results if r is not None]
    clean.sort(key=lambda x: x['|r|'], reverse=True)
    df_res = pd.DataFrame(clean)

    save_path = os.path.join(OUTPUT_DIR, f'delta_search_{label.replace(" ", "_")}.xlsx')
    df_res.to_excel(save_path, index=False)

    # Top 5
    print(f'\n  Top 5:')
    for i in range(min(5, len(clean))):
        r = clean[i]
        print(f'    #{i+1}: n={r["n_feats"]}  |r|={r["|r|"]:.4f}  rho={r["rho"]:.4f}  p={r["p"]:.6f}')
        print(f'         {r["features"]}')

    return {
        'label': label,
        'full_r': abs(r_full),
        'full_rho': rho_full,
        'full_p': p_full,
        'best_r': clean[0]['|r|'] if clean else 0,
        'best_rho': clean[0]['rho'] if clean else 0,
        'best_p': clean[0]['p'] if clean else 1,
        'best_n': clean[0]['n_feats'] if clean else 0,
        'best_features': clean[0]['features'] if clean else '',
    }


def plot_comparison(results_list):
    """横向对比柱状图"""
    import matplotlib.pyplot as plt
    labels = [r['label'] for r in results_list]
    full_vals = [r['full_r'] for r in results_list]
    best_vals = [r['best_r'] for r in results_list]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - w/2, full_vals, w, label='Full Features', color='lightcoral', edgecolor='white')
    bars2 = ax.bar(x + w/2, best_vals, w, label='Best Selection', color='steelblue', edgecolor='white')

    for bar, val in zip(bars1, full_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, best_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('|r| (Pearson) vs ATP', fontsize=12)
    ax.set_title('Delta Feature Search vs Paper Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9584, color='green', linestyle='--', linewidth=1.5, label='Paper Best (r=0.958)')
    ax.legend(fontsize=9, loc='lower right')

    fig.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'delta_vs_paper_comparison.png')
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'\nComparison chart saved: {save_path}')


def main():
    results_all = []

    # ===== A: 论文风格 Delta (K6→4, mean) =====
    delta_paper, pool_paper = build_delta_table_paper()
    r = run_search('Paper_Delta (K6→4 mean)', delta_paper, pool_paper)
    results_all.append(r)

    # ===== B: PSD 风格 Delta (K=3, median) =====
    try:
        delta_psd, pool_psd = build_delta_table_psd()
        r = run_search('PSD_Delta (K=3 median)', delta_psd, pool_psd)
        results_all.append(r)
    except FileNotFoundError as e:
        print(f'PSD Delta not available: {e}')

    # ===== 汇总对比 =====
    print(f'\n{"="*60}')
    print(f'  最终对比')
    print(f'{"="*60}')
    print(f'  {"方法":<30s} {"全特征|r|":>10s} {"最优|r|":>10s} {"提升":>8s}')
    print(f'  {"-"*58}')
    for r in results_all:
        gain = r['best_r'] - r['full_r']
        print(f'  {r["label"]:<30s} {r["full_r"]:>10.4f} {r["best_r"]:>10.4f} {gain:>+8.4f}')

    print(f'\n  论文方法最优: |r| = 0.9584 (Score_Diff策略, 非Delta策略)')

    plot_comparison(results_all)

    # 保存汇总
    df_summary = pd.DataFrame(results_all)
    df_summary.to_excel(os.path.join(OUTPUT_DIR, 'delta_search_summary.xlsx'), index=False)
    print('\nDone.')


if __name__ == '__main__':
    main()