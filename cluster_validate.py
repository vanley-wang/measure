"""
Validation framework for comparing clustering methods.

Computes:
  - Internal metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
  - Biological plausibility (phenotype ordering, dose-response)
  - Cross-method agreement (confusion matrices, ARI)

Usage:
    python cluster_validate.py
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    confusion_matrix,
)

from cluster_utils import (
    RAW_FEATURES,
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
    'Kmeans': 'model/Kmeans-scatt.pickle',
    'GMM': 'model/GMM-scatt.pickle',
    'Rule': 'model/Rule-scatt.pickle',
}

# Drug concentration mapping (extracted from well name prefixes)
# Adjust these mappings based on your actual naming convention
DRUG_MAP = {
    'Control': 0,
    'Ctrl': 0,
    '20': 20,
    '40': 40,
    '80': 80,
}

# ==========================================

def load_all_data():
    """Load all measure_excel files into a single DataFrame."""
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
    """Infer drug concentration from well name."""
    for key, val in DRUG_MAP.items():
        if key in well_name:
            return val
    return -1  # unknown


def compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute internal clustering quality metrics."""
    return {
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
    }


def compute_biological_plausibility(df: pd.DataFrame) -> dict:
    """
    Compute domain-specific biological plausibility scores.

    Returns dict of pass/fail and quantitative scores.
    """
    stats = df.groupby('Cluster')[['Organoids_Volume_Fill', 'Cavity_Volume', 'Scatt_Mean']].mean()
    stats = stats.reindex([0, 1, 2, 3])

    results = {}

    # 1. Volume ordering: Red > Yellow > Green > Blue
    vols = stats['Organoids_Volume_Fill'].values
    vol_ordered = all(vols[i] >= vols[i+1] for i in range(3))
    results['vol_order_ok'] = vol_ordered
    results['vol_order_detail'] = f"R={vols[0]:.0f} > Y={vols[1]:.0f} > G={vols[2]:.0f} > B={vols[3]:.0f}"

    # 2. OAC ordering: Blue > Green > Yellow > Red
    oacs = stats['Scatt_Mean'].values
    oac_ordered = all(oacs[i] <= oacs[i+1] for i in range(3))
    results['oac_order_ok'] = oac_ordered
    results['oac_order_detail'] = f"R={oacs[0]:.1f} < Y={oacs[1]:.1f} < G={oacs[2]:.1f} < B={oacs[3]:.1f}"

    # 3. Red fraction in control wells
    control_df = df[df['_conc'] == 0]
    if len(control_df) > 0:
        red_frac = (control_df['Cluster'] == 0).mean()
        results['control_red_fraction'] = red_frac
        results['control_red_ok'] = red_frac > 0.05
    else:
        results['control_red_fraction'] = 0
        results['control_red_ok'] = False

    # 4. Blue fraction dose-response
    dose_df = df[df['_conc'] > 0].groupby('_conc')['Cluster'].apply(lambda s: (s == 3).mean())
    if len(dose_df) >= 2:
        # Check monotonic increase
        mono = all(dose_df.values[i] <= dose_df.values[i+1] for i in range(len(dose_df)-1))
        results['blue_dose_monotonic'] = mono
        results['blue_dose_detail'] = dict(dose_df)
    else:
        results['blue_dose_monotonic'] = None
        results['blue_dose_detail'] = {}

    return results


def evaluate_method(name: str, model_path: str, df_full: pd.DataFrame) -> dict:
    """Evaluate a single clustering method."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    pkg = load_model_package(model_path)
    method = pkg['model_type']
    features = pkg['feature_names']

    df = df_full.copy()

    # Predict labels
    if method == 'rule':
        from cluster_utils import rule_classify_df
        labels = rule_classify_df(df[features]).values
    elif method in ('kmeans', 'gmm'):
        preprocessor = pkg.get('preprocessor')
        scaler = pkg.get('scaler')
        model = pkg['model']
        raw_to_final = pkg.get('raw_to_final', {})

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
    else:
        raise ValueError(f"Unknown model_type: {method}")

    df['Cluster'] = labels

    # --- Internal metrics (on preprocessed space) ---
    # Build a fresh preprocessor for fair comparison across methods
    preproc = Preprocessor()
    X_all = preproc.fit_transform(df[features])

    # Exclude noise/outliers if any cluster has < 2 samples
    valid_mask = pd.Series(labels).groupby(labels).transform('count') >= 2
    if valid_mask.sum() > 4:
        X_valid = X_all[valid_mask]
        labels_valid = labels[valid_mask]
        internal = compute_internal_metrics(X_valid, labels_valid)
    else:
        internal = {'silhouette': np.nan, 'calinski_harabasz': np.nan, 'davies_bouldin': np.nan}

    print(f"  Silhouette:     {internal['silhouette']:.4f}")
    print(f"  Calinski-Harabasz: {internal['calinski_harabasz']:.2f}")
    print(f"  Davies-Bouldin: {internal['davies_bouldin']:.4f}")

    # --- Biological plausibility ---
    bio = compute_biological_plausibility(df)
    print(f"  Volume order:   {'PASS' if bio['vol_order_ok'] else 'FAIL'}  ({bio['vol_order_detail']})")
    print(f"  OAC order:      {'PASS' if bio['oac_order_ok'] else 'FAIL'}  ({bio['oac_order_detail']})")
    print(f"  Control red:    {'PASS' if bio['control_red_ok'] else 'FAIL'}  ({bio['control_red_fraction']:.2%})")
    if bio['blue_dose_monotonic'] is not None:
        print(f"  Blue dose resp: {'PASS' if bio['blue_dose_monotonic'] else 'FAIL'}  ({bio['blue_dose_detail']})")

    # --- Cluster counts ---
    counts = pd.Series(labels).value_counts().sort_index()
    print(f"  Cluster counts: {dict(counts)}")

    return {
        'name': name,
        'labels': labels,
        'internal': internal,
        'biological': bio,
        'counts': counts.to_dict(),
    }


def compare_methods(results: list) -> None:
    """Print cross-method agreement tables."""
    print("\n" + "=" * 60)
    print("Cross-Method Agreement (Adjusted Rand Index)")
    print("=" * 60)

    names = [r['name'] for r in results]
    n = len(names)
    for i in range(n):
        for j in range(i+1, n):
            ari = adjusted_rand_score(results[i]['labels'], results[j]['labels'])
            print(f"  {names[i]} vs {names[j]}: ARI = {ari:.4f}")

    print("\n" + "=" * 60)
    print("Confusion Matrices")
    print("=" * 60)

    for i in range(n):
        for j in range(i+1, n):
            cm = confusion_matrix(results[i]['labels'], results[j]['labels'],
                                  labels=[0, 1, 2, 3])
            print(f"\n{names[i]} (rows) vs {names[j]} (cols):")
            cm_df = pd.DataFrame(cm, index=[f"R{i}" for i in range(4)],
                                      columns=[f"C{j}" for j in range(4)])
            print(cm_df)


def main():
    print("=" * 60)
    print("Clustering Validation Framework")
    print("=" * 60)

    # 1. Load data
    print("\n--- Loading data ---")
    df = load_all_data()
    print(f"Total organoids: {len(df)}")

    # Add concentration column
    df['_conc'] = df['_well'].apply(infer_concentration)

    # 2. Evaluate each method
    results = []
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"[WARN] Model not found, skipping: {path}")
            continue
        try:
            res = evaluate_method(name, path, df)
            results.append(res)
        except Exception as e:
            print(f"[ERR] Failed to evaluate {name}: {e}")

    # 3. Cross-method comparison
    if len(results) >= 2:
        compare_methods(results)

    print("\n[Done] Validation complete.")


if __name__ == "__main__":
    main()
