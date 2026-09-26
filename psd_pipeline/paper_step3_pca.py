# ============================================================
# Step 3: PCA 建模 + Result_Diff vs ATP (复现论文核心)
# 论文方法:
#   1. 将 Day3+Day5 特征拼接为一张大表
#   2. 在拼接表上 PCA → Result_Score
#   3. 按孔配对: Result_Diff = Result_End - Result_Start
#   4. Result_Diff vs ATP 相关性
#   5. 特征重要性 + 穷举搜索最优子集
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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ANALYSIS_PATH = os.path.join(OUTPUT_DIR, 'FXN_2023_Analysis.xlsx')

ATP_DATABASE = {
    'B10': 601300, 'B11': 11180000, 'B2': 5391000, 'B3': 6538000,
    'B4': 7103000, 'B5': 1264000, 'B6': 2548000, 'B7': 1579000,
    'B8': 3637000, 'B9': 140300, 'C10': 211800, 'C11': 13930000,
    'C2': 4460000, 'C3': 8336000, 'C4': 6800000, 'C5': 330900,
    'C6': 238900, 'C7': 682100, 'C8': 211300, 'C9': 465900,
    'D11': 11240000, 'E11': 12840000, 'F10': 21910000, 'F11': 14700000,
    'F2': 26980000, 'F3': 14110000, 'F4': 13740000, 'F5': 17250000,
    'F6': 20320000, 'F7': 20000000, 'F8': 17170000, 'F9': 15830000
}

N_COMPONENTS = 4
N_ITERATIONS = 200000  # 穷举搜索次数
N_JOBS = -1


def load_analysis_table():
    df = pd.read_excel(ANALYSIS_PATH)
    if 'Name' not in df.columns:
        df['Name'] = df.index.astype(str)
    df['Well_ID'] = df['Name'].apply(lambda x: str(x).split('_')[0])
    df['TimePoint'] = df['Name'].apply(lambda x: str(x).split('_')[1])
    df['ATP'] = df['Well_ID'].map(ATP_DATABASE)
    return df


def build_feature_pool(df):
    exclude = ['Name', 'Well_ID', 'TimePoint', 'ATP']
    cols = [c for c in df.columns if c not in exclude]
    # 移除全为常量或全NaN的列
    valid_cols = []
    for c in cols:
        vals = df[c].dropna()
        if len(vals) == 0 or vals.nunique() <= 1:
            continue
        valid_cols.append(c)
    return valid_cols


def compute_pca_scores(df, feature_list, n_components=N_COMPONENTS):
    X = df[feature_list].fillna(0).values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_std)
    var_ratio = pca.explained_variance_ratio_
    total_var = np.sum(var_ratio)
    if total_var == 0:
        return np.zeros(len(df)), np.zeros(len(df)), None, None, None
    weights = var_ratio / total_var
    result_score = np.dot(scores, weights)
    return result_score, scores[:, 0], pca, scaler, weights


def correlate_with_atp(df, score_col='Result_Score'):
    d3 = df[df['TimePoint'] == '0701'][['Well_ID', score_col]].set_index('Well_ID')
    d5 = df[df['TimePoint'] == '0703'][['Well_ID', score_col, 'ATP']].set_index('Well_ID')
    merged = d5.join(d3, lsuffix='_End', rsuffix='_Start', how='inner')
    merged['Result_Diff'] = merged[f'{score_col}_End'] - merged[f'{score_col}_Start']
    valid = merged.dropna(subset=['Result_Diff', 'ATP'])
    if len(valid) < 3:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0
    r_diff, p_diff = pearsonr(valid['Result_Diff'], valid['ATP'])
    r_end, p_end = pearsonr(valid[f'{score_col}_End'], valid['ATP'])
    rho_diff, _ = spearmanr(valid['Result_Diff'], valid['ATP'])
    return abs(r_diff), abs(r_end), abs(rho_diff), len(valid), p_diff, p_end


# ---- 单次随机搜索 ----
def search_one_trial(seed, X_matrix_all, feature_pool, idx_start, idx_end, y_atp):
    rng = np.random.RandomState(seed)
    n_total = len(feature_pool)
    n_select = rng.randint(4, min(n_total + 1, 25))
    feat_indices = rng.choice(n_total, n_select, replace=False)
    feat_indices.sort()
    X_sub = X_matrix_all[:, feat_indices]
    try:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_sub)
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        scores = pca.fit_transform(X_std)
        var_ratio = pca.explained_variance_ratio_
        total_var = np.sum(var_ratio)
        if total_var == 0:
            return None
        weights = var_ratio / total_var
        result_scores = np.dot(scores, weights)
    except Exception:
        return None

    res_start = result_scores[idx_start]
    res_end = result_scores[idx_end]
    res_diff = res_end - res_start

    diff_mean = np.mean(res_diff)
    diff_std = np.std(res_diff)
    atp_mean = np.mean(y_atp)
    atp_std = np.std(y_atp)
    if diff_std == 0 or atp_std == 0:
        return None
    cov_diff = np.mean((res_diff - diff_mean) * (y_atp - atp_mean))
    r_diff = abs(cov_diff / (diff_std * atp_std))

    end_mean = np.mean(res_end)
    end_std = np.std(res_end)
    if end_std == 0:
        return None
    cov_end = np.mean((res_end - end_mean) * (y_atp - atp_mean))
    r_end = abs(cov_end / (end_std * atp_std))

    names = [feature_pool[i] for i in feat_indices]
    return {
        'n_feats': len(feat_indices),
        'feat_idx': feat_indices,
        'diff_r': float(r_diff),
        'end_r': float(r_end),
        'features': ', '.join(names)
    }


def main():
    df = load_analysis_table()
    feature_pool = build_feature_pool(df)
    print(f'Feature pool: {len(feature_pool)} valid features')

    # ========== 全特征 PCA ==========
    print(f'\n=== Full-feature PCA ({len(feature_pool)} features) ===')
    result_score, pc1, pca_model, scaler_model, weights = compute_pca_scores(df, feature_pool)
    df['Result_Score'] = result_score
    df['PC1'] = pc1

    r_diff, r_end, rho_diff, n_valid, p_diff, p_end = correlate_with_atp(df, 'Result_Score')
    print(f'\n  Result_Diff vs ATP: r={r_diff:.4f} (Pearson), rho={rho_diff:.4f} (Spearman), p={p_diff:.6f}')
    print(f'  Result_End  vs ATP: r={r_end:.4f}, p={p_end:.6f}')
    print(f'  Valid pairs: {n_valid}')

    # 保存全特征结果
    d3 = df[df['TimePoint'] == '0701'][['Well_ID', 'Result_Score']].set_index('Well_ID')
    d5 = df[df['TimePoint'] == '0703'][['Well_ID', 'Result_Score', 'ATP']].set_index('Well_ID')
    merged = d5.join(d3, lsuffix='_End', rsuffix='_Start', how='inner')
    merged['Result_Diff'] = merged['Result_Score_End'] - merged['Result_Score_Start']
    merged.to_excel(os.path.join(OUTPUT_DIR, 'score_diff_full.xlsx'))

    # 散点图
    valid = merged.dropna(subset=['Result_Diff', 'ATP'])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(valid['Result_Diff'], valid['ATP'], c='steelblue', edgecolors='white', s=80, zorder=3)
    z = np.polyfit(valid['Result_Diff'], valid['ATP'], 1)
    x_line = np.linspace(valid['Result_Diff'].min(), valid['Result_Diff'].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', lw=2, label=f'r={r_diff:.4f}')
    ax.set_xlabel('Result_Diff (Score_End - Score_Start)', fontsize=12)
    ax.set_ylabel('ATP', fontsize=12)
    ax.set_title('Paper Method: Result_Diff vs ATP (Full Features)', fontsize=14)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'scatter_diff_vs_atp_full.png'), dpi=300)
    plt.close(fig)

    # ========== 特征搜索 ==========
    print(f'\n=== Feature Search ({N_ITERATIONS} trials) ===')
    X_matrix_all = df[feature_pool].fillna(0).values
    well_ids = df['Well_ID'].unique()
    idx_start = []
    idx_end = []
    valid_atp = []
    for well in well_ids:
        rows_start = df[(df['Well_ID'] == well) & (df['TimePoint'] == '0701')]
        rows_end = df[(df['Well_ID'] == well) & (df['TimePoint'] == '0703')]
        if not rows_start.empty and not rows_end.empty:
            atp = rows_end.iloc[0]['ATP']
            if pd.notna(atp):
                idx_start.append(df.index.get_loc(rows_start.index[0]))
                idx_end.append(df.index.get_loc(rows_end.index[0]))
                valid_atp.append(atp)
    idx_start = np.array(idx_start)
    idx_end = np.array(idx_end)
    y_atp = np.array(valid_atp)

    seeds = [random.randint(0, 100000000) for _ in range(N_ITERATIONS)]
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(search_one_trial)(s, X_matrix_all, feature_pool, idx_start, idx_end, y_atp)
        for s in seeds
    )
    t1 = time.time()
    print(f'Search done in {t1 - t0:.1f}s')

    clean = [r for r in results if r is not None]
    clean.sort(key=lambda x: x['diff_r'], reverse=True)
    df_search = pd.DataFrame(clean)
    df_search.to_excel(os.path.join(OUTPUT_DIR, 'feature_search_results.xlsx'), index=False)

    print(f'\nTop 5 feature combinations:')
    for i in range(min(5, len(clean))):
        r = clean[i]
        print(f'  #{i+1}: n={r["n_feats"]}, |r_diff|={r["diff_r"]:.4f}')
        print(f'       {r["features"]}')

    # ========== 用最优特征组合重新计算 ==========
    if clean:
        best = clean[0]
        best_features = [feature_pool[i] for i in best['feat_idx']]
        print(f'\n=== Best features re-evaluation (n={len(best_features)}) ===')
        result_score2, pc1_2, pca2, scaler2, w2 = compute_pca_scores(df, best_features)
        df['Result_Score_Best'] = result_score2
        r_diff2, r_end2, rho_diff2, n2, p_diff2, p_end2 = correlate_with_atp(df, 'Result_Score_Best')
        print(f'  Result_Diff vs ATP: r={r_diff2:.4f}, rho={rho_diff2:.4f}, p={p_diff2:.6f}')

        # 散点图
        d3b = df[df['TimePoint'] == '0701'][['Well_ID', 'Result_Score_Best']].set_index('Well_ID')
        d5b = df[df['TimePoint'] == '0703'][['Well_ID', 'Result_Score_Best', 'ATP']].set_index('Well_ID')
        merged_b = d5b.join(d3b, lsuffix='_End', rsuffix='_Start', how='inner')
        merged_b['Result_Diff'] = merged_b['Result_Score_Best_End'] - merged_b['Result_Score_Best_Start']
        merged_b.to_excel(os.path.join(OUTPUT_DIR, 'score_diff_best.xlsx'))
        valid_b = merged_b.dropna(subset=['Result_Diff', 'ATP'])

        fig, ax = plt.subplots(figsize=(7, 6))
        for _, row in valid_b.iterrows():
            ax.scatter(row['Result_Diff'], row['ATP'], c='steelblue', edgecolors='white', s=80, zorder=3)
        z2 = np.polyfit(valid_b['Result_Diff'], valid_b['ATP'], 1)
        x_line2 = np.linspace(valid_b['Result_Diff'].min(), valid_b['Result_Diff'].max(), 100)
        ax.plot(x_line2, np.polyval(z2, x_line2), 'r--', lw=2, label=f'r={r_diff2:.4f}')
        ax.set_xlabel('Result_Diff (Score_End - Score_Start)', fontsize=12)
        ax.set_ylabel('ATP', fontsize=12)
        ax.set_title(f'Paper Method: Best {len(best_features)} Features |r|={r_diff2:.4f}', fontsize=14)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'scatter_diff_vs_atp_best.png'), dpi=300)
        plt.close(fig)

    print('\nStep 3 complete.')


if __name__ == '__main__':
    main()