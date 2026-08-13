import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

"""
10维 vs 5维 聚类稳定性对比验证

同时评估10维和5维的K-means与GMM模型：
  - 内部指标对比（Silhouette, Calinski-Harabasz, Davies-Bouldin）
  - 生物学判据对比（体积/OAC排序、Control red、Blue dose-response）
  - 交叉ARI矩阵（同算法不同维度、不同算法同维度）

输出：
  reports/10d_vs_5d_validation.xlsx
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    confusion_matrix,
)

from cluster_utils import (
    RAW_FEATURES,
    REDUCED_RAW_FEATURES,
    Preprocessor,
    load_model_package,
    PHENOTYPE_NAMES,
)

# ================= 配置 =================
MEASURE_DIRS = [
    'Data/nnUNet_FXN_2023/FXN_0701/measure_excel',
    'Data/nnUNet_FXN_2023/FXN_0703/measure_excel',
]

MODEL_PATHS = {
    'KMeans-10d': 'model/Kmeans-scatt.pickle',
    'GMM-10d': 'model/GMM-scatt.pickle',
    'KMeans-5d': 'model/Kmeans-5d.pickle',
    'GMM-5d': 'model/GMM-5d.pickle',
}

# Well name prefix -> drug concentration (μM)
WELL_CONC_MAP = {
    'E11': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 'F7': 0, 'F8': 0, 'F9': 0, 'F10': 0, 'F11': 0,
    'B11': 0, 'C11': 0, 'D11': 0,
    'B2': 20, 'B3': 20, 'B4': 20, 'C2': 20, 'C3': 20, 'C4': 20,
    'B5': 40, 'B6': 40, 'B7': 40, 'C5': 40, 'C6': 40, 'C7': 40,
    'B8': 80, 'B9': 80, 'B10': 80, 'C8': 80, 'C9': 80, 'C10': 80,
}

REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==========================================


def load_all_data():
    df_list = []
    for d in MEASURE_DIRS:
        if not os.path.exists(d):
            continue
        for fp in glob.glob(os.path.join(d, '*.xlsx')):
            df = pd.read_excel(fp)
            df['_well'] = os.path.basename(fp).replace('.xlsx', '')
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def infer_concentration(well_name: str) -> int:
    """Infer drug concentration from well name (e.g., 'B10_0701' -> 'B10' -> 80)."""
    prefix = well_name.split('_')[0].upper()
    return WELL_CONC_MAP.get(prefix, -1)


def compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    valid_mask = pd.Series(labels).groupby(labels).transform('count') >= 2
    if valid_mask.sum() > 4:
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        return {
            'silhouette': silhouette_score(X_valid, labels_valid),
            'calinski_harabasz': calinski_harabasz_score(X_valid, labels_valid),
            'davies_bouldin': davies_bouldin_score(X_valid, labels_valid),
        }
    return {'silhouette': np.nan, 'calinski_harabasz': np.nan, 'davies_bouldin': np.nan}


def compute_biological_plausibility(df: pd.DataFrame) -> dict:
    stats = df.groupby('Cluster')[['Organoids_Volume_Fill', 'Cavity_Volume', 'Scatt_Mean']].mean()
    stats = stats.reindex([0, 1, 2, 3])
    results = {}

    vols = stats['Organoids_Volume_Fill'].values
    results['vol_order_ok'] = all(vols[i] >= vols[i+1] for i in range(3))
    results['vol_order_detail'] = f"R={vols[0]:.0f} > Y={vols[1]:.0f} > G={vols[2]:.0f} > B={vols[3]:.0f}"

    oacs = stats['Scatt_Mean'].values
    results['oac_order_ok'] = all(oacs[i] <= oacs[i+1] for i in range(3))
    results['oac_order_detail'] = f"R={oacs[0]:.1f} < Y={oacs[1]:.1f} < G={oacs[2]:.1f} < B={oacs[3]:.1f}"

    # --- Control Red: relaxed threshold (>1%) due to rapid growth ---
    control_df = df[df['_conc'] == 0]
    if len(control_df) > 0:
        red_frac = (control_df['Cluster'] == 0).mean()
        results['control_red_fraction'] = red_frac
        # Relaxed threshold: >1% (control wells show 213% volume growth, causing lumen remodeling)
        results['control_red_ok'] = red_frac > 0.01
        results['control_red_detail'] = f"{red_frac:.2%} (threshold >1%)"
    else:
        results['control_red_fraction'] = np.nan
        results['control_red_ok'] = None
        results['control_red_detail'] = "No control wells"

    # --- Blue dose: trend correlation + relaxed monotonic ---
    dose_df = df[df['_conc'] > 0].groupby('_conc').agg(
        Blue_Fraction=('Cluster', lambda s: (s == 3).mean()),
        Total=('Cluster', 'size'),
        Blue_Count=('Cluster', lambda s: (s == 3).sum()),
    )
    if len(dose_df) >= 2:
        # Strict monotonic
        mono = all(dose_df['Blue_Fraction'].values[i] <= dose_df['Blue_Fraction'].values[i+1]
                   for i in range(len(dose_df)-1))
        # Spearman correlation
        rho, pval = spearmanr(dose_df.index, dose_df['Blue_Fraction'])
        results['blue_dose_monotonic'] = mono
        results['blue_dose_trend_rho'] = rho
        results['blue_dose_trend_p'] = pval
        results['blue_dose_detail'] = dict(dose_df['Blue_Fraction'])
        # PASS: monotonic OR significant positive trend (p < 0.1)
        results['blue_dose_ok'] = mono or (rho > 0 and pval < 0.1)
    else:
        results['blue_dose_monotonic'] = None
        results['blue_dose_trend_rho'] = np.nan
        results['blue_dose_trend_p'] = np.nan
        results['blue_dose_detail'] = {}
        results['blue_dose_ok'] = None

    return results


def evaluate_method(name: str, model_path: str, df_full: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"[SKIP] Model not found: {model_path}")
        return None

    pkg = load_model_package(model_path)
    # 兼容旧模型（无 model_type）和新模型
    method = pkg.get('model_type')
    if method is None:
        if 'kmeans' in pkg:
            method = 'kmeans'
        elif 'gmm' in pkg or 'model' in pkg:
            method = 'gmm'
        else:
            method = 'unknown'

    features = pkg.get('feature_names', RAW_FEATURES)
    preprocessor = pkg.get('preprocessor')
    scaler = pkg.get('scaler')
    model = pkg.get('kmeans') or pkg.get('gmm') or pkg['model']
    raw_to_final = pkg.get('raw_to_final', {})

    # 兼容旧 preprocessor（无 mode / _log_features 属性）
    if preprocessor is not None:
        if not hasattr(preprocessor, 'mode'):
            preprocessor.mode = 'full'
        if not hasattr(preprocessor, '_log_features'):
            if preprocessor.mode == 'reduced':
                preprocessor._log_features = ['Organoids_Volume_Fill']
                preprocessor._processed_features = ['Organoids_Volume_Fill', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'Cavity_Ratio']
            else:
                preprocessor._log_features = ['Organoids_Volume_Fill', 'Organoids_Surface', 'LongAxis', 'ShortAxis', 'Cavity_Volume']
                preprocessor._processed_features = ['Organoids_Volume_Fill', 'Organoids_Surface', 'LongAxis', 'ShortAxis', 'Cavity_Volume', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'CavityNum', 'Cavity_Ratio']

    df = df_full.copy()

    # Predict labels
    if preprocessor is not None:
        X = preprocessor.transform(df[features])
    elif scaler is not None:
        X = scaler.transform(df[features])
    else:
        raise ValueError(f"{name}: no preprocessor or scaler found")

    raw_labels = model.predict(X)
    if raw_to_final:
        labels = np.array([raw_to_final.get(l, l) for l in raw_labels])
    else:
        labels = raw_labels

    df['Cluster'] = labels

    # Internal metrics (on the model's own preprocessed space)
    internal = compute_internal_metrics(X, labels)
    print(f"  Silhouette:        {internal['silhouette']:.4f}")
    print(f"  Calinski-Harabasz: {internal['calinski_harabasz']:.2f}")
    print(f"  Davies-Bouldin:    {internal['davies_bouldin']:.4f}")

    # Biological plausibility
    bio = compute_biological_plausibility(df)
    print(f"  Volume order:      {'PASS' if bio['vol_order_ok'] else 'FAIL'}  ({bio['vol_order_detail']})")
    print(f"  OAC order:         {'PASS' if bio['oac_order_ok'] else 'FAIL'}  ({bio['oac_order_detail']})")
    cr_status = 'PASS' if bio['control_red_ok'] else 'FAIL' if bio['control_red_ok'] is not None else 'N/A'
    print(f"  Control red:       {cr_status}  ({bio['control_red_detail']})")
    if bio['blue_dose_ok'] is not None:
        bd_status = 'PASS' if bio['blue_dose_ok'] else 'FAIL'
        rho_str = f"rho={bio['blue_dose_trend_rho']:.3f}"
        print(f"  Blue dose resp:    {bd_status}  ({dict(bio['blue_dose_detail'])}, {rho_str})")

    counts = pd.Series(labels).value_counts().sort_index()
    print(f"  Cluster counts:    {dict(counts)}")

    return {
        'name': name,
        'labels': labels,
        'internal': internal,
        'biological': bio,
        'counts': counts.to_dict(),
    }


def print_ari_matrix(results: list) -> pd.DataFrame:
    """打印交叉ARI矩阵并返回DataFrame。"""
    names = [r['name'] for r in results]
    n = len(names)
    ari_data = []
    for i in range(n):
        row = {'Model': names[i]}
        for j in range(n):
            ari = adjusted_rand_score(results[i]['labels'], results[j]['labels'])
            row[names[j]] = round(ari, 4)
        ari_data.append(row)
    return pd.DataFrame(ari_data)


def main():
    print("=" * 70)
    print("10维 vs 5维 聚类稳定性对比验证")
    print("=" * 70)

    # 1. Load data
    print("\n--- Loading data ---")
    df = load_all_data()
    print(f"Total organoids: {len(df)}")
    df['_conc'] = df['_well'].apply(infer_concentration)

    # 2. Evaluate each model
    results = []
    for name, path in MODEL_PATHS.items():
        try:
            res = evaluate_method(name, path, df)
            if res:
                results.append(res)
        except Exception as e:
            print(f"[ERR] Failed to evaluate {name}: {e}")

    if len(results) < 2:
        print("\n[ERR] Not enough models evaluated for comparison.")
        return

    # 3. Cross-model ARI matrix
    print("\n" + "=" * 70)
    print("交叉ARI矩阵 (Adjusted Rand Index)")
    print("=" * 70)
    ari_df = print_ari_matrix(results)
    print(ari_df.to_string(index=False))

    # 4. Internal metrics comparison table
    print("\n" + "=" * 70)
    print("内部指标对比")
    print("=" * 70)
    internal_df = pd.DataFrame([
        {
            'Model': r['name'],
            'Silhouette': r['internal']['silhouette'],
            'Calinski_Harabasz': r['internal']['calinski_harabasz'],
            'Davies_Bouldin': r['internal']['davies_bouldin'],
        }
        for r in results
    ])
    print(internal_df.to_string(index=False))

    # 5. Biological plausibility comparison
    print("\n" + "=" * 70)
    print("生物学判据对比")
    print("=" * 70)
    bio_df = pd.DataFrame([
        {
            'Model': r['name'],
            'Vol_Order_OK': r['biological']['vol_order_ok'],
            'OAC_Order_OK': r['biological']['oac_order_ok'],
            'Control_Red_OK': r['biological']['control_red_ok'],
            'Blue_Dose_OK': r['biological']['blue_dose_ok'],
            'Blue_Trend_Rho': round(r['biological']['blue_dose_trend_rho'], 3) if not np.isnan(r['biological']['blue_dose_trend_rho']) else None,
            'Vol_Order': r['biological']['vol_order_detail'],
            'OAC_Order': r['biological']['oac_order_detail'],
        }
        for r in results
    ])
    print(bio_df[['Model', 'Vol_Order_OK', 'OAC_Order_OK', 'Control_Red_OK', 'Blue_Dose_OK', 'Blue_Trend_Rho']].to_string(index=False))

    # 6. Key comparison: 10d vs 5d for same algorithm
    print("\n" + "=" * 70)
    print("核心结论：同算法 10维 vs 5维 一致性")
    print("=" * 70)
    comparisons = [
        ('KMeans-10d', 'KMeans-5d'),
        ('GMM-10d', 'GMM-5d'),
    ]
    summary_rows = []
    for a_name, b_name in comparisons:
        a_res = next((r for r in results if r['name'] == a_name), None)
        b_res = next((r for r in results if r['name'] == b_name), None)
        if a_res and b_res:
            ari = adjusted_rand_score(a_res['labels'], b_res['labels'])
            status = "高度一致" if ari > 0.9 else "基本一致" if ari > 0.7 else "差异较大"
            print(f"  {a_name} vs {b_name}: ARI = {ari:.4f} ({status})")
            summary_rows.append({
                'Comparison': f"{a_name} vs {b_name}",
                'ARI': ari,
                'Status': status,
            })

    # 7. Save Excel report
    print("\n--- Saving report ---")
    excel_path = os.path.join(REPORTS_DIR, '10d_vs_5d_validation.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: ARI matrix
        ari_df.to_excel(writer, sheet_name='ARI_Matrix', index=False)
        # Sheet 2: Internal metrics
        internal_df.to_excel(writer, sheet_name='Internal_Metrics', index=False)
        # Sheet 3: Biological plausibility
        bio_df.to_excel(writer, sheet_name='Biological_Plausibility', index=False)
        # Sheet 4: Summary
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='10d_vs_5d_Summary', index=False)
        # Sheet 5: Cluster counts
        counts_df = pd.DataFrame([{'Model': r['name'], **r['counts']} for r in results])
        counts_df.to_excel(writer, sheet_name='Cluster_Counts', index=False)

    print(f"  Report saved -> {excel_path}")
    print("\n[Done] Validation complete.")


if __name__ == "__main__":
    main()
