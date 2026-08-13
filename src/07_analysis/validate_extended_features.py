import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

"""
Validate extended feature set (5D + OCT stats) for collinearity and clustering performance.

Steps:
  1. Load merged measure_excel data
  2. Compute Spearman correlation matrix
  3. Compute VIF
  4. Select non-redundant features (|r| < 0.9, VIF < 10)
  5. Train KMeans and GMM on selected features
  6. Compare with 5D baseline

Output:
  reports/extended_feature_validation.xlsx
  reports/figures/extended_feature_correlation.png
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from cluster_utils import Preprocessor

# ================= 配置 =================
DATA_FOLDERS = [
    'Data/nnUNet_FXN_2023/FXN_0701/measure_excel',
    'Data/nnUNet_FXN_2023/FXN_0703/measure_excel',
]

REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 5D baseline features
BASE_5D = ['Organoids_Volume_Fill', 'Sphericity', 'Scatt_Mean', 'Scatt_STD', 'Cavity_Volume']

# All candidate features (5D + OCT stats)
# Note: Cavity_Ratio is engineered in Preprocessor
CANDIDATE_FEATURES = BASE_5D + [
    'Scatt_Skewness', 'Scatt_Kurtosis', 'Scatt_Entropy',
    'Scatt_IQR', 'Scatt_Median', 'Scatt_CV',
    'Scatt_Range', 'Scatt_10pct', 'Scatt_90pct',
]

# ==========================================


def load_all_data():
    df_list = []
    for folder in DATA_FOLDERS:
        if not os.path.exists(folder):
            continue
        for fp in glob.glob(os.path.join(folder, '*.xlsx')):
            df = pd.read_excel(fp)
            df['_well'] = os.path.basename(fp).replace('.xlsx', '')
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def compute_vif(X, feature_names):
    """Compute VIF for each feature."""
    X_df = pd.DataFrame(X, columns=feature_names).fillna(0)
    vif_data = []
    for i, name in enumerate(feature_names):
        try:
            vif = variance_inflation_factor(X_df.values, i)
        except Exception:
            vif = np.nan
        vif_data.append({'Feature': name, 'VIF': vif})
    return pd.DataFrame(vif_data)


def select_features(df, features, corr_threshold=0.9, vif_threshold=10):
    """Select non-redundant features."""
    # Step 1: Preprocess
    preprocessor = Preprocessor(mode='reduced')
    X = preprocessor.fit_transform(df[features])
    proc_names = preprocessor.get_feature_names()

    # Step 2: Correlation matrix on raw features
    corr = df[features].corr(method='spearman')

    # Step 3: VIF
    vif_df = compute_vif(X, proc_names)
    print("\n--- VIF ---")
    print(vif_df.to_string(index=False))

    # Step 4: Iterative selection
    # Start with 5D baseline, then add new features one by one
    selected_raw = BASE_5D.copy()
    new_features = [f for f in features if f not in BASE_5D]

    for feat in new_features:
        trial = selected_raw + [feat]
        trial_preproc = Preprocessor(mode='reduced')
        X_trial = trial_preproc.fit_transform(df[trial])
        trial_names = trial_preproc.get_feature_names()

        # Check VIF
        vif_trial = compute_vif(X_trial, trial_names)
        max_vif = vif_trial['VIF'].max()

        # Check correlation with existing
        corr_with_existing = [abs(df[feat].corr(df[s], method='spearman')) for s in selected_raw]
        max_corr = max(corr_with_existing) if corr_with_existing else 0

        if max_vif < vif_threshold and max_corr < corr_threshold:
            selected_raw.append(feat)
            print(f"  [KEEP] {feat}: VIF_max={max_vif:.1f}, max_corr={max_corr:.2f}")
        else:
            print(f"  [DROP] {feat}: VIF_max={max_vif:.1f}, max_corr={max_corr:.2f}")

    return selected_raw, corr, vif_df


def plot_correlation_heatmap(corr, selected_features, save_path):
    """Plot correlation heatmap."""
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                xticklabels=corr.columns, yticklabels=corr.columns)
    plt.title('Spearman Correlation: 5D + OCT Stats', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Correlation heatmap saved -> {save_path}")


def train_and_evaluate(df, features, label):
    """Train KMeans and GMM, return metrics."""
    # For extended features, we need manual preprocessing:
    # log1p on Volume_Fill, then StandardScaler on all
    df_proc = df[features].copy()
    if 'Organoids_Volume_Fill' in df_proc.columns:
        df_proc['Organoids_Volume_Fill'] = np.log1p(df_proc['Organoids_Volume_Fill'])
    # Cavity_Ratio engineering if Cavity_Volume present
    if 'Cavity_Volume' in df_proc.columns and 'Organoids_Volume_Fill' in df_proc.columns:
        df_proc['Cavity_Ratio'] = df_proc['Cavity_Volume'] / df_proc['Organoids_Volume_Fill'].clip(lower=1)
        df_proc.drop(columns=['Cavity_Volume'], inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc.fillna(0))

    # KMeans
    km = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    km_labels = km.fit_predict(X)

    # GMM
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    gmm_labels = gmm.fit_predict(X)

    results = {'label': label, 'n_features': X.shape[1]}
    for name, labels in [('KMeans', km_labels), ('GMM', gmm_labels)]:
        results[f'{name}_silhouette'] = silhouette_score(X, labels)
        results[f'{name}_calinski'] = calinski_harabasz_score(X, labels)
        results[f'{name}_davies'] = davies_bouldin_score(X, labels)

        # Cluster counts
        counts = pd.Series(labels).value_counts().sort_index()
        results[f'{name}_counts'] = dict(counts)

    return results


def main():
    print("=" * 60)
    print("Extended Feature Validation")
    print("=" * 60)

    # 1. Load data
    print("\n--- Loading data ---")
    df = load_all_data()
    print(f"Total organoids: {len(df)}")

    # Check which features are available
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    missing = [f for f in CANDIDATE_FEATURES if f not in df.columns]
    print(f"Available features: {available}")
    if missing:
        print(f"Missing features: {missing}")

    # 2. Feature selection
    print("\n--- Feature Selection ---")
    selected, corr, vif_df = select_features(df, available)
    print(f"\nSelected features ({len(selected)}): {selected}")

    # 3. Plot correlation
    print("\n--- Plotting ---")
    plot_correlation_heatmap(
        corr,
        selected,
        os.path.join(FIGURES_DIR, 'extended_feature_correlation.png')
    )

    # 4. Compare 5D vs extended
    print("\n--- Clustering Comparison ---")
    baseline = train_and_evaluate(df, BASE_5D, '5D_Baseline')
    extended = train_and_evaluate(df, selected, 'Extended')

    comparison = pd.DataFrame([baseline, extended])
    print(comparison[['label', 'KMeans_silhouette', 'KMeans_calinski', 'KMeans_davies',
                      'GMM_silhouette', 'GMM_calinski', 'GMM_davies']].to_string(index=False))

    # 5. PCA comparison
    print("\n--- PCA Comparison ---")
    for name, feats in [('5D', BASE_5D), ('Extended', selected)]:
        df_proc = df[feats].copy()
        if 'Organoids_Volume_Fill' in df_proc.columns:
            df_proc['Organoids_Volume_Fill'] = np.log1p(df_proc['Organoids_Volume_Fill'])
        if 'Cavity_Volume' in df_proc.columns and 'Organoids_Volume_Fill' in df_proc.columns:
            df_proc['Cavity_Ratio'] = df_proc['Cavity_Volume'] / df_proc['Organoids_Volume_Fill'].clip(lower=1)
            df_proc.drop(columns=['Cavity_Volume'], inplace=True)
        scaler = StandardScaler()
        X = scaler.fit_transform(df_proc.fillna(0))
        pca = PCA(n_components=min(10, X.shape[1]))
        pca.fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
        print(f"  {name} (dim={X.shape[1]}): PC1-3 = {cumvar[2]:.1f}%, PC1-5 = {cumvar[min(4, len(cumvar)-1)]:.1f}%")

    # 6. Save report
    print("\n--- Saving report ---")
    excel_path = os.path.join(REPORTS_DIR, 'extended_feature_validation.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        comparison.to_excel(writer, sheet_name='Comparison', index=False)
        vif_df.to_excel(writer, sheet_name='VIF', index=False)
        corr.to_excel(writer, sheet_name='Spearman_Correlation')
        pd.DataFrame({'Selected_Features': selected}).to_excel(writer, sheet_name='Selected', index=False)

    print(f"  Report saved -> {excel_path}")
    print("\n[Done]")


if __name__ == "__main__":
    main()
