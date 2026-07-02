"""
Train rule-based phenotype classifier for organoid clustering.

Unlike unsupervised methods, this uses biologically-defined hierarchical rules
with per-well adaptive thresholds. It is deterministic and fully interpretable.

Usage:
    python train_rule.py
    python train_rule.py --data-folders path1 path2 --output model/Rule-scatt.pickle
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd

from cluster_utils import (
    RAW_FEATURES,
    compute_well_stats,
    rule_classify_df,
    save_model_package,
    PHENOTYPE_NAMES,
)

# ================= 配置 =================
DEFAULT_DATA_FOLDERS = [
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0701', 'measure_excel'),
    os.path.join('Data', 'nnUNet_FXN_2023', 'FXN_0703', 'measure_excel'),
]
MODEL_DIR = 'model'
OUTPUT_PATH = os.path.join(MODEL_DIR, 'Rule-scatt.pickle')

# ==========================================

def load_data(folders):
    """Load all measure_excel files and return concatenated DataFrame."""
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


def compute_global_thresholds(df: pd.DataFrame) -> dict:
    """
    Compute global fallback thresholds from the full dataset.
    Used when a well has too few organoids for reliable percentile estimation.
    """
    vol = df['Organoids_Volume_Fill']
    oac = df['Scatt_Mean']
    return {
        'vol_median': vol.median(),
        'vol_75': vol.quantile(0.75),
        'vol_60': vol.quantile(0.60),
        'vol_90': vol.quantile(0.90),
        'oac_85': oac.quantile(0.85),
        'oac_90': oac.quantile(0.90),
    }


def main():
    parser = argparse.ArgumentParser(description='Train rule-based organoid classifier')
    parser.add_argument('--data-folders', nargs='+', default=DEFAULT_DATA_FOLDERS,
                        help='Paths to measure_excel directories')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Output model path')
    args = parser.parse_args()

    print("=" * 60)
    print("Rule-Based Phenotype Classifier Calibration")
    print("=" * 60)

    # 1. Load data
    print("\n--- Loading data ---")
    Data_All = load_data(args.data_folders)
    print(f"Total organoids: {len(Data_All)}")

    missing = [f for f in RAW_FEATURES if f not in Data_All.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 2. Compute global thresholds (fallback for small wells)
    print("\n--- Computing global fallback thresholds ---")
    global_thresholds = compute_global_thresholds(Data_All)
    for k, v in global_thresholds.items():
        print(f"  {k}: {v:.2f}")

    # 3. Apply rule classification on full dataset (for validation)
    print("\n--- Applying rules to full dataset ---")
    labels = rule_classify_df(Data_All[RAW_FEATURES])
    Data_All['Cluster'] = labels

    # 4. Print statistics
    print("\n--- Cluster Statistics ---")
    summary = Data_All.groupby('Cluster')[['Organoids_Volume_Fill', 'Cavity_Volume', 'Scatt_Mean']].mean()
    print(summary)

    counts = Data_All['Cluster'].value_counts().sort_index()
    print("\n--- Cluster Counts ---")
    print(counts)

    # 5. Save model package
    print("\n--- Saving model ---")
    save_model_package(
        path=args.output,
        model=None,  # rule-based has no estimator object
        model_type='rule',
        preprocessor=None,
        feature_names=RAW_FEATURES,
        extra={
            'global_thresholds': global_thresholds,
            'rule_description': {
                0: 'Volume_Fill > well_vol_75 AND Cavity_Ratio > 0.12',
                1: 'Volume_Fill > well_vol_median (after Red/Blue exclusion)',
                2: 'Volume_Fill <= well_vol_median (after Red/Blue exclusion)',
                3: 'Scatt_Mean > well_oac_85 AND Volume_Fill < well_vol_60',
            }
        }
    )

    print("\n[Done] Rule-based classifier calibrated and saved.")
    print("  Note: This model uses per-well adaptive thresholds at inference time.")


if __name__ == "__main__":
    main()
