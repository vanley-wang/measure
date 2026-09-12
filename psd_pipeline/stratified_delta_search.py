# ============================================================
# PSD Stratified Delta — 加入特征穷举搜索
# 聚类: 论文 K6→4 合并, 分类健康(0/1) vs 损伤(2/3)
# 特征: 逐表型中位数 + CV + GrowthRate + Ratio + Fraction
# 先测中位数, 再测均值
# ============================================================
import os, sys, time, random, pickle, warnings
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
    'B4': 7103000, 'B5': 511900, 'B6': 404500, 'B7': 403900,
    'B8': 312700, 'B9': 140300, 'C10': 211800, 'C11': 13930000,
    'C2': 6336000, 'C3': 8336000, 'C4': 6800000, 'C5': 330900,
    'C6': 238900, 'C7': 682100, 'C8': 211300, 'C9': 465900,
    'D11': 11240000, 'E11': 14700000, 'F10': 21910000, 'F11': 11180000,
    'F2': 18240000, 'F3': 14110000, 'F4': 13740000, 'F5': 17250000,
    'F6': 20320000, 'F7': 20000000, 'F8': 17170000, 'F9': 15830000,
}

CLUSTER_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD'
]

NUMERIC_MAP = {3: 0, 5: 0, 1: 1, 0: 2, 4: 2, 2: 3}

# 参数简称
PARAM_SHORT = {
    'Organoids_Volume': 'Vol', 'Organoids_Volume_Fill': 'VolFill', 'Organoids_Surface': 'Surf',
    'Cavity_Volume': 'CavVol', 'CavityNum': 'CavNum', 'LongAxis': 'LongAx',
    'ShortAxis': 'ShortAx', 'Wall_Thickness': 'CystThk', 'Sphericity': 'Spher',
    'Scatt_Mean': 'OACm', 'Scatt_STD': 'OACsd'
}

N_MERGED = 4
HEALTHY_C = {0, 1}
DAMAGED_C = {2, 3}

N_ITER = 100000
N_JOBS = -1
N_PC = 4
MIN_SAMPLES = 5


# ===== 数据加载 =====
def load_well_data(measure_dir):
    wells = {}
    for fname in os.listdir(measure_dir):
        if not fname.endswith('.xlsx'): continue
        well_id = fname.replace('.xlsx', '').split('_')[0]
        df = pd.read_excel(os.path.join(measure_dir, fname))
        wells[well_id] = df
    return wells


# ===== 分类 =====
def classify_well(df_well, kmeans, scaler):
    X = df_well[CLUSTER_FEATURES].fillna(0).values
    k6 = kmeans.predict(scaler.transform(X))
    return np.array([NUMERIC_MAP[l] for l in k6])


# ===== 分层特征提取 =====
def compute_stratified_features(df_well, labels, agg_fn=np.median):
    """agg_fn = np.median or np.mean"""
    record = {}
    for c in range(N_MERGED):
        mask = labels == c
        n = mask.sum()
        record[f'N_C{c}'] = n
        sub = df_well.loc[mask] if n > 0 else df_well.iloc[:0]
        for src, sn in PARAM_SHORT.items():
            if src not in df_well.columns: continue
            vals = sub[src].values
            if n >= MIN_SAMPLES:
                record[f'C{c}_{sn}'] = float(agg_fn(vals))
            else:
                record[f'C{c}_{sn}'] = np.nan

    # Group-level: healthy (C01), damaged (C23), all
    for gname, gset in [('H', HEALTHY_C), ('D', DAMAGED_C)]:
        mask_g = np.isin(labels, list(gset))
        n_g = mask_g.sum()
        record[f'N_{gname}'] = n_g
        sub_g = df_well.loc[mask_g] if n_g > 0 else df_well.iloc[:0]
        for src, sn in PARAM_SHORT.items():
            if src not in df_well.columns: continue
            vals = sub_g[src].values
            if n_g >= MIN_SAMPLES:
                record[f'{gname}_{sn}'] = float(agg_fn(vals))
            else:
                record[f'{gname}_{sn}'] = np.nan

    # All (global)
    n_all = len(df_well)
    record['N_All'] = n_all
    for src, sn in PARAM_SHORT.items():
        if src not in df_well.columns: continue
        vals = df_well[src].values
        if n_all >= MIN_SAMPLES:
            record[f'All_{sn}'] = float(agg_fn(vals))
        else:
            record[f'All_{sn}'] = np.nan

    # CV (heterogeneity) — only for key params in healthy group
    for sn in ['VolFill', 'OACm']:
        mask_h = np.isin(labels, list(HEALTHY_C))
        if mask_h.sum() >= MIN_SAMPLES:
            key_src = [k for k,v in PARAM_SHORT.items() if v == sn][0]
            vals = df_well.loc[mask_h, key_src].values
            med = np.median(vals) if agg_fn == np.median else np.mean(vals)
            record[f'H_CV_{sn}'] = float(np.std(vals) / med) if med and med != 0 else np.nan
        else:
            record[f'H_CV_{sn}'] = np.nan

    # Growth rate H_GR = D5 healthy median / D3 healthy median (computed later as delta)

    # Cross-cluster ratios (C01 vs C23)
    mask_h = np.isin(labels, list(HEALTHY_C))
    mask_d = np.isin(labels, list(DAMAGED_C))
    if mask_h.sum() >= MIN_SAMPLES and mask_d.sum() >= MIN_SAMPLES:
        h_fill = agg_fn(df_well.loc[mask_h, 'Organoids_Volume_Fill'].values)
        d_fill = agg_fn(df_well.loc[mask_d, 'Organoids_Volume_Fill'].values)
        record['HD_VolRatio'] = h_fill / d_fill if d_fill != 0 else np.nan
        record['HD_CountRatio'] = mask_h.sum() / mask_d.sum()
        h_oac = agg_fn(df_well.loc[mask_h, 'Scatt_Mean'].values)
        d_oac = agg_fn(df_well.loc[mask_d, 'Scatt_Mean'].values)
        record['HD_OACRatio'] = h_oac / d_oac if d_oac != 0 else np.nan
    else:
        for k in ['HD_VolRatio', 'HD_CountRatio', 'HD_OACRatio']:
            record[k] = np.nan

    # Fraction features
    tot = len(df_well)
    if tot > 0:
        for c in range(N_MERGED):
            record[f'Frac_C{c}'] = (labels == c).sum() / tot
        record['Frac_H'] = (np.isin(labels, list(HEALTHY_C))).sum() / tot
        record['Frac_D'] = (np.isin(labels, list(DAMAGED_C))).sum() / tot

    return record


# ===== 构建 Delta =====
def build_delta_table(stats_d3, stats_d5, wells):
    rows = []
    all_keys = set()
    for w in wells:
        for k in stats_d3[w]:
            all_keys.add(k)
    for k in list(all_keys):
        if k.startswith('N_') or k.startswith('Frac_'):
            all_keys.add(f'{k}_D5')
            all_keys.discard(k)

    n_keys = sorted([k for k in all_keys if k.startswith('N_') or k.startswith('Frac_')])
    other_keys = sorted(all_keys - set(n_keys))

    for w in wells:
        row = {'Well_ID': w, 'ATP': ATP_DB.get(w, np.nan)}
        s3, s5 = stats_d3[w], stats_d5[w]

        # Count & fraction: keep D5 values (state)
        for k in n_keys:
            row[k] = s5.get(k, np.nan)

        # Other: compute Delta
        for k in other_keys:
            v3, v5 = s3.get(k, np.nan), s5.get(k, np.nan)
            if pd.notna(v3) and pd.notna(v5):
                row[f'{k}_Delta'] = v5 - v3
            else:
                row[f'{k}_Delta'] = np.nan

        # Growth rate (H_D5/H_D3) for key params
        for sn in ['VolFill', 'OACm']:
            v3 = s3.get(f'H_{sn}', np.nan)
            v5 = s5.get(f'H_{sn}', np.nan)
            if pd.notna(v3) and pd.notna(v5) and v3 != 0:
                row[f'H_GR_{sn}'] = v5 / v3
            else:
                row[f'H_GR_{sn}'] = np.nan

        rows.append(row)
    return pd.DataFrame(rows)


# ===== 穷举搜索 =====
def search_trial(seed, X_full, feat_names, y_atp):
    rng = np.random.RandomState(seed)
    n_total = X_full.shape[1]
    n_sel = rng.randint(3, min(n_total + 1, 22))
    idx = rng.choice(n_total, n_sel, replace=False)
    idx.sort()
    X_sub = X_full[:, idx]
    try:
        sc = StandardScaler()
        X_std = sc.fit_transform(X_sub)
        pca = PCA(n_components=N_PC, random_state=42)
        scores = pca.fit_transform(X_std)
        vr = pca.explained_variance_ratio_
        tv = np.sum(vr)
        if tv == 0: return None
        w = vr / tv
        result = np.dot(scores, w)
    except:
        return None
    if np.std(result) == 0: return None
    r, p = pearsonr(result, y_atp)
    rho, _ = spearmanr(result, y_atp)
    names = ', '.join([feat_names[i] for i in idx])
    return {'n': n_sel, '|r|': abs(r), 'r': r, 'rho': rho, 'p': p, 'features': names}


def run_experiment(label, agg_fn):
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')

    with open(os.path.join(MODEL_DIR, 'kmeans_k6.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler_k6.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    wells_d3 = load_well_data(DAY3_DIR)
    wells_d5 = load_well_data(DAY5_DIR)
    common = sorted(set(wells_d3) & set(wells_d5))

    stats_d3, stats_d5 = {}, {}
    for w in common:
        stats_d3[w] = compute_stratified_features(wells_d3[w],
            classify_well(wells_d3[w], kmeans, scaler), agg_fn)
        stats_d5[w] = compute_stratified_features(wells_d5[w],
            classify_well(wells_d5[w], kmeans, scaler), agg_fn)

    df = build_delta_table(stats_d3, stats_d5, common)
    atp_col = df['ATP'].values.astype(float)

    delta_cols = [c for c in df.columns if c not in ('Well_ID', 'ATP')]
    valid = []
    for c in delta_cols:
        vals = df[c].dropna()
        if len(vals) > 0 and vals.nunique() > 1:
            valid.append(c)
    print(f'Valid features: {len(valid)}')

    X = df[valid].fillna(0).values

    # 全特征
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    pca = PCA(n_components=N_PC, random_state=42)
    scores = pca.fit_transform(Xs)
    vr = pca.explained_variance_ratio_
    w = vr / np.sum(vr)
    rf = np.dot(scores, w)
    r_full, pf = pearsonr(rf, atp_col)
    rho_full, _ = spearmanr(rf, atp_col)
    print(f'Full ({len(valid)}): |r|={abs(r_full):.4f}, rho={rho_full:.4f}, p={pf:.6f}')

    # 搜索
    print(f'Searching {N_ITER}...')
    seeds = [random.randint(0, 10**8) for _ in range(N_ITER)]
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, verbose=3)(
        delayed(search_trial)(s, X, valid, atp_col) for s in seeds)
    t1 = time.time()
    print(f'Done {t1-t0:.1f}s')

    clean = [r for r in res if r is not None]
    clean.sort(key=lambda x: x['|r|'], reverse=True)

    print(f'Top 5:')
    for i in range(min(5, len(clean))):
        r = clean[i]
        print(f'  #{i+1}: n={r["n"]}  |r|={r["|r|"]:.4f}  rho={r["rho"]:.4f}  p={r["p"]:.6f}')

    if clean:
        best = clean[0]
        best_cols = [c.strip() for c in best['features'].split(', ')]
        Xb = df[best_cols].fillna(0).values
        sc2 = StandardScaler()
        Xb_s = sc2.fit_transform(Xb)
        pca2 = PCA(n_components=N_PC, random_state=42)
        s2 = pca2.fit_transform(Xb_s)
        v2 = pca2.explained_variance_ratio_
        w2 = v2 / np.sum(v2)
        score_best = np.dot(s2, w2)

        fig, ax = plt.subplots(figsize=(7, 6))
        for wid, s, a in zip(df['Well_ID'], score_best, atp_col):
            ax.scatter(s, a, c='steelblue', edgecolors='white', s=80, zorder=3)
            ax.annotate(wid, (s, a), fontsize=6, ha='center', va='bottom', alpha=0.7)
        z = np.polyfit(score_best, atp_col, 1)
        xl = np.linspace(score_best.min(), score_best.max(), 100)
        ax.plot(xl, np.polyval(z, xl), 'r--', lw=2, label=f'|r|={best["|r|"]:.4f}')
        ax.set_xlabel('Stratified Delta Score', fontsize=12)
        ax.set_ylabel('ATP', fontsize=12)
        ax.set_title(f'{label}  Best {best["n"]} feat |r|={best["|r|"]:.4f}', fontsize=14)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        tag = 'median' if agg_fn == np.median else 'mean'
        fig.savefig(os.path.join(OUTPUT_DIR, f'stratified_delta_{tag}.png'), dpi=300)
        plt.close(fig)

    return abs(r_full), rho_full, clean[0] if clean else None


def main():
    results = []

    # === 中位数 ===
    fr_m, rh_m, best_m = run_experiment('Stratified Delta (K6→4, MEDIAN + CV + Ratio)', np.median)
    results.append(('Stratified median', fr_m, best_m['|r|'] if best_m else 0,
                    best_m['n'] if best_m else 0, best_m['rho'] if best_m else 0))

    # === 均值 ===
    fr_e, rh_e, best_e = run_experiment('Stratified Delta (K6→4, MEAN + CV + Ratio)', np.mean)
    results.append(('Stratified mean', fr_e, best_e['|r|'] if best_e else 0,
                    best_e['n'] if best_e else 0, best_e['rho'] if best_e else 0))

    print(f'\n{"="*60}')
    print(f'  最终对比')
    print(f'{"="*60}')
    print(f'  {"方法":<40s} {"全|r|":>8s} {"最优|r|":>8s} {"rho":>8s} {"n":>5s}')
    print(f'  {"-"*69}')
    for name, fr, br, nf, rho in results:
        print(f'  {name:<40s} {fr:>8.4f} {br:>8.4f} {rho:>8.4f} {nf:>5d}')
    print(f'\n  论文 Score_Diff (K6→4 mean):              0.3124  0.9584')
    print(f'  PSD  Delta (K6→4 median, per-cluster):     0.7101  0.8880')
    print(f'  PSD  Delta (K6→4 mean, per-cluster):       0.8816  0.9573')
    print('Done.')


if __name__ == '__main__':
    main()