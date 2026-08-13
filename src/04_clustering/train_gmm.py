import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

"""
Train Gaussian Mixture Model (GMM) for organoid phenotype clustering.

Usage:
    python train_gmm.py
    python train_gmm.py --data-folders path1 path2 --output model/GMM-scatt.pickle
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from cluster_utils import (
    RAW_FEATURES,
    REDUCED_RAW_FEATURES,
    Preprocessor,
    map_phenotypes_by_centroids,
    compute_phenotype_prototypes,
    build_means_init,
    save_model_package,
    PHENOTYPE_NAMES,
)

# ================= 配置 =================
DEFAULT_DATA_FOLDERS = [
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]
MODEL_DIR = 'model'
OUTPUT_PATH = os.path.join(MODEL_DIR, 'GMM-scatt.pickle')

# ==========================================

def load_data(folders):
    """Load and concatenate all measure_excel files."""
    df_list = []
    for folder in folders:
        if not os.path.exists(folder):
            print(f"[WARN] Skipping non-existent folder: {folder}")
            continue
        files = glob.glob(os.path.join(folder, '*.xlsx'))
        print(f"  {folder}: {len(files)} files")
        for fp in files:
            try:
                df_list.append(pd.read_excel(fp))
            except Exception as e:
                print(f"    [ERR] {os.path.basename(fp)}: {e}")
    if not df_list:
        raise ValueError("No data loaded!")
    return pd.concat(df_list, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description='Train GMM for organoid clustering')
    parser.add_argument('--data-folders', nargs='+', default=DEFAULT_DATA_FOLDERS,
                        help='Paths to measure_excel directories')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Output model path')
    parser.add_argument('--covariance-type', default='full',
                        choices=['full', 'tied', 'diag', 'spherical'],
                        help='GMM covariance type')
    parser.add_argument('--max-iter', type=int, default=500,
                        help='Maximum EM iterations')
    parser.add_argument('--reduced', action='store_true',
                        help='Use 5-dim reduced feature set')
    args = parser.parse_args()

    if args.reduced:
        features = REDUCED_RAW_FEATURES
        model_type_label = '5D Reduced'
        if args.output == OUTPUT_PATH:
            args.output = os.path.join(MODEL_DIR, 'GMM-5d.pickle')
    else:
        features = RAW_FEATURES
        model_type_label = '10D Full'

    print("=" * 60)
    print(f"GMM Training for Organoid Phenotype Clustering ({model_type_label})")
    print("=" * 60)

    # 1. Load data
    print("\n--- Loading data ---")
    Data_All = load_data(args.data_folders)
    print(f"Total organoids: {len(Data_All)}")

    # Ensure all raw features exist
    missing = [f for f in features if f not in Data_All.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 2. Preprocess
    print(f"\n--- Preprocessing ({model_type_label}: log1p + engineer + standardize) ---")
    preprocessor = Preprocessor(mode='reduced' if args.reduced else 'full')
    X = preprocessor.fit_transform(Data_All[features])
    print(f"Processed features: {preprocessor.get_feature_names()}")
    print(f"Feature matrix shape: {X.shape}")

    # 4. Train GMM
    # Note: phenotype-informed means_init was tried but caused numerical issues
    # (Cavity_Ratio@90 = 0 due to 95.7% zeros, collapsing components).
    # Default k-means++ init + full covariance works well on log-transformed data.
    print(f"\n--- Training GMM (K=4, covariance={args.covariance_type}) ---")
    gmm = GaussianMixture(
        n_components=4,
        covariance_type=args.covariance_type,
        max_iter=args.max_iter,
        random_state=42,
        init_params='kmeans',
    )
    gmm.fit(X)
    raw_labels = gmm.predict(X)

    print(f"  Converged: {gmm.converged_}")
    print(f"  N iterations: {gmm.n_iter_}")
    print(f"  Log-likelihood: {gmm.lower_bound_:.2f}")

    # 5. Map raw clusters to phenotype labels
    print("\n--- Mapping raw clusters to phenotypes ---")
    raw_to_final = map_phenotypes_by_centroids(Data_All[features], raw_labels)

    for raw_id, final_id in sorted(raw_to_final.items()):
        print(f"  Raw {raw_id} -> {final_id} ({PHENOTYPE_NAMES[final_id]})")

    # 6. Print cluster statistics
    print("\n--- Cluster Statistics ---")
    Data_All['Cluster'] = [raw_to_final[l] for l in raw_labels]
    summary = Data_All.groupby('Cluster')[['Organoids_Volume_Fill', 'Cavity_Volume', 'Scatt_Mean']].mean()
    print(summary)

    # 7. Save model package
    print("\n--- Saving model ---")
    save_model_package(
        path=args.output,
        model=gmm,
        model_type='gmm',
        preprocessor=preprocessor,
        feature_names=features,
        extra={
            'raw_to_final': raw_to_final,
            'covariance_type': args.covariance_type,
        }
    )

    print("\n[Done] GMM model trained and saved.")


if __name__ == "__main__":
    main()
