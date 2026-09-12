# ============================================================
# PSD-Delta 改进验证: 聚类=论文K6→4, 聚合=中位数
# 控制变量，隔离"均值 vs 中位数"的影响
# ============================================================
import os
import sys
import time
import random
import pickle
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

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'Data', 'FXN_2023_new（ICC）')
DAY3_DIR = os.path.join(DATA_DIR, 'FXN_20230701', 'measure_excel')
DAY5_DIR = os.path.join(DATA_DIR, 'FXN_20230703', 'measure_excel')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
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

CLUSTER_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD'
]

NUMERIC_MAP = {3: 0, 5: 0, 1: 1, 0: 2, 4: 2, 2: 3}
N_MERGED = 4

# 聚合参数: 全用 median
AGG_PARAMS = [
    ('Cavity_Volume',           'Cavity_Volume'),
    ('Organoids_Volume_Fill',   'Volume_Fill'),
    ('Organoids_Volume',        'Volume'),
    ('Organoids_Surface',       'Surface'),
    ('LongAxis',                'Long_Axis'),
    ('ShortAxis',               'Short_Axis'),
    ('Wall_Thickness',          'Cyst_Thick'),
    ('Sphericity',              'Sphericity'),
    ('Scatt_Mean',              'Scatt_Mean'),
    ('Scatt_STD',               'Scatt_STD'),
    ('CavityNum',               'CavityNum'),
]

N_ITER = 100000
N_JOBS = -1
N_PC = 4


def load_well_data(measure_dir):
    """加载所有xlsx，返回 {well_id: DataFrame}"""
    wells = {}
    for fname in os.listdir(measure_dir):
        if not fname.endswith('.xlsx'):
            continue
        full_name = fname.replace('.xlsx', '')
        parts = full_name.split('_')
        well_id = parts[0]  # B10_0701 -> B10
        df = pd.read_excel(os.path.join(measure_dir, fname))
        wells[well_id] = df
    return wells


def classify_well(df_well, kmeans, scaler):
    """K=6 → 合并为4"""
    X = df_well[CLUSTER_FEATURES].fillna(0).values
    X_std = scaler.transform(X)
    k6 = kmeans.predict(X_std)
    return np.array([NUMERIC_MAP[l] for l in k6])


def compute_well_median_stats(df_well, labels):
    """逐孔、逐合并表型算中位数"""
    record = {}
    for c in range(N_MERGED):
        mask = labels == c
        n = mask.sum()
        record[f'Number_{c+1}'] = int(n)
        if n >= 3:
            sub = df_well.loc[mask]
            for src, name in AGG_PARAMS:
                record[f'{name}_Median_{c+1}'] = float(np.median(sub[src].values))
        else:
            for src, name in AGG_PARAMS:
                record[f'{name}_Median_{c+1}'] = np.nan
    return record


def search_trial(seed, X_full, feature_pool, y_atp):
    rng = np.random.RandomState(seed)
    n_total = len(feature_pool)
    n_select = rng.randint(3, min(n_total + 1, 22))
    idx = rng.choice(n_total, n_select, replace=False)
    idx.sort()
    X_sub = X_full[:, idx]
    try:
        sc = StandardScaler()
        X_std = sc.fit_transform(X_sub)
        pca = PCA(n_components=N_PC, random_state=42)
        scores = pca.fit_transform(X_std)
        var = pca.explained_variance_ratio_
        total = np.sum(var)
        if total == 0:
            return None
        w = var / total
        result = np.dot(scores, w)
    except Exception:
        return None
    if np.std(result) == 0:
        return None
    r, p = pearsonr(result, y_atp)
    rho, _ = spearmanr(result, y_atp)
    names = [feature_pool[i] for i in idx]
    return {'n': len(idx), '|r|': abs(r), 'r': r, 'rho': rho, 'p': p, 'features': ', '.join(names)}


def main():
    # 加载模型
    with open(os.path.join(MODEL_DIR, 'kmeans_k6.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler_k6.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # 加载原始数据
    wells_d3 = load_well_data(DAY3_DIR)
    wells_d5 = load_well_data(DAY5_DIR)
    common = sorted(set(wells_d3) & set(wells_d5))
    print(f'Common wells: {len(common)}')

    # 对每孔分类 + 算中位数
    stats_d3 = {}
    stats_d5 = {}
    for well in common:
        labels3 = classify_well(wells_d3[well], kmeans, scaler)
        stats_d3[well] = compute_well_median_stats(wells_d3[well], labels3)
        labels5 = classify_well(wells_d5[well], kmeans, scaler)
        stats_d5[well] = compute_well_median_stats(wells_d5[well], labels5)

    # 构建 Delta 表
    feature_keys = []
    for c in range(1, N_MERGED + 1):
        feature_keys.append(f'Number_{c}')
        for _, name in AGG_PARAMS:
            feature_keys.append(f'{name}_Median_{c}')

    delta_rows = []
    for well in common:
        s3 = stats_d3[well]
        s5 = stats_d5[well]
        row = {'Well_ID': well, 'ATP': ATP_DB.get(well, np.nan),
               'ATP_Log': np.log10(ATP_DB.get(well, np.nan)) if ATP_DB.get(well, np.nan) else np.nan}
        for k in feature_keys:
            v3 = s3.get(k, np.nan)
            v5 = s5.get(k, np.nan)
            if pd.notna(v3) and pd.notna(v5):
                row[f'{k}_Delta'] = v5 - v3
            else:
                row[f'{k}_Delta'] = np.nan
        delta_rows.append(row)

    df_delta = pd.DataFrame(delta_rows)
    atp_col = df_delta['ATP'].values.astype(float)

    # 特征筛选
    feature_cols = [c for c in df_delta.columns if c.endswith('_Delta')]
    valid_cols = []
    for c in feature_cols:
        vals = df_delta[c].dropna()
        if len(vals) > 0 and vals.nunique() > 1:
            valid_cols.append(c)
    print(f'Valid Delta features: {len(valid_cols)}')

    X_full = df_delta[valid_cols].fillna(0).values

    # 全特征基线
    sc = StandardScaler()
    X_std = sc.fit_transform(X_full)
    pca = PCA(n_components=N_PC, random_state=42)
    scores = pca.fit_transform(X_std)
    var = pca.explained_variance_ratio_
    w = var / np.sum(var)
    result_full = np.dot(scores, w)
    r_full, p_full = pearsonr(result_full, atp_col)
    rho_full, _ = spearmanr(result_full, atp_col)
    print(f'Full features ({len(valid_cols)}): |r|={abs(r_full):.4f}, rho={rho_full:.4f}, p={p_full:.6f}')

    # 穷举搜索
    print(f'Searching {N_ITER} combos...')
    seeds = [random.randint(0, 100000000) for _ in range(N_ITER)]
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=3)(
        delayed(search_trial)(s, X_full, valid_cols, atp_col) for s in seeds
    )
    t1 = time.time()
    print(f'Done in {t1-t0:.1f}s')

    clean = [r for r in results if r is not None]
    clean.sort(key=lambda x: x['|r|'], reverse=True)

    print(f'\nTop 5:')
    for i in range(min(5, len(clean))):
        r = clean[i]
        print(f'  #{i+1}: n={r["n"]}  |r|={r["|r|"]:.4f}  rho={r["rho"]:.4f}  p={r["p"]:.6f}')
        print(f'       {r["features"]}')

    # 散点图
    best = clean[0]
    best_cols = best['features'].split(', ')
    X_best = df_delta[best_cols].fillna(0).values
    sc2 = StandardScaler()
    Xb_std = sc2.fit_transform(X_best)
    pca2 = PCA(n_components=N_PC, random_state=42)
    s2 = pca2.fit_transform(Xb_std)
    v2 = pca2.explained_variance_ratio_
    w2 = v2 / np.sum(v2)
    score_best = np.dot(s2, w2)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(score_best, atp_col, c='steelblue', edgecolors='white', s=80, zorder=3)
    z = np.polyfit(score_best, atp_col, 1)
    xl = np.linspace(score_best.min(), score_best.max(), 100)
    ax.plot(xl, np.polyval(z, xl), 'r--', lw=2, label=f'|r|={best["|r|"]:.4f}')
    ax.set_xlabel('PSD Score (K6→4 + Median + Delta → PCA)', fontsize=11)
    ax.set_ylabel('ATP', fontsize=12)
    ax.set_title(f'PSD-Hybrid: K6→4 + Median |r|={best["|r|"]:.4f}', fontsize=14)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'psd_hybrid_median.png'), dpi=300)
    plt.close(fig)

    # 保存
    df_delta.to_excel(os.path.join(OUTPUT_DIR, 'psd_hybrid_delta.xlsx'), index=False)
    df_res = pd.DataFrame(clean)
    df_res.to_excel(os.path.join(OUTPUT_DIR, 'psd_hybrid_search.xlsx'), index=False)

    # 最终对比
    print(f'\n{"="*60}')
    print(f'  对比总结')
    print(f'{"="*60}')
    print(f'  {"方法":<35s} {"全特征|r|":>10s} {"最优|r|":>10s}')
    print(f'  {"-"*55}')
    print(f'  {"论文 K6→4 + mean  + Score_Diff":<35s} {"0.3124":>10s} {"0.9584":>10s}')
    print(f'  {"论文 K6→4 + mean  + Delta":<35s} {"0.8816":>10s} {"0.9573":>10s}')
    print(f'  {"PSD  K6→4 + median + Delta":<35s} {abs(r_full):>10.4f} {best["|r|"]:>10.4f}')
    print(f'  {"PSD  K=3   + median + Delta":<35s} {"0.2575":>10s} {"0.8664":>10s}')
    print('\nDone.')


if __name__ == '__main__':
    main()