"""
Merge OCT signal statistics into existing measure_excel files.

Input:
  - reports/oct_stats_0701.xlsx
  - reports/oct_stats_0703.xlsx
  - Data/nnUNet_FXN_2023/FXN_0701/measure_excel/*.xlsx
  - Data/nnUNet_FXN_2023/FXN_0703/measure_excel/*.xlsx

Output:
  - Data/nnUNet_FXN_2023/FXN_0701/measure_excel/*.xlsx (with new texture columns)
  - Data/nnUNet_FXN_2023/FXN_0703/measure_excel/*.xlsx (with new texture columns)
  - reports/extended_features_summary.xlsx
"""
import os
import glob
import pandas as pd

# ================= 配置 =================
DATA_CONFIGS = [
    {
        'name': 'FXN_0701',
        'stats_file': 'reports/oct_stats_0701.xlsx',
        'measure_dir': 'Data/nnUNet_FXN_2023/FXN_0701/measure_excel',
    },
    {
        'name': 'FXN_0703',
        'stats_file': 'reports/oct_stats_0703.xlsx',
        'measure_dir': 'Data/nnUNet_FXN_2023/FXN_0703/measure_excel',
    },
]

REPORTS_DIR = 'reports'

# ==========================================


def merge_stats_for_dataset(config):
    print(f"\n{'='*60}")
    print(f"Merging: {config['name']}")
    print(f"{'='*60}")

    # Load stats
    df_stats = pd.read_excel(config['stats_file'])
    print(f"Loaded stats: {len(df_stats)} rows")

    # Find all measure_excel files
    measure_files = sorted(glob.glob(os.path.join(config['measure_dir'], '*.xlsx')))
    print(f"Found {len(measure_files)} measure_excel files")

    merged_count = 0
    for measure_path in measure_files:
        well_name = os.path.basename(measure_path).replace('.xlsx', '')

        # Load measure
        df_measure = pd.read_excel(measure_path)

        # Merge on Index
        df_merged = df_measure.merge(df_stats, on='Index', how='left')

        # Check merge result
        new_cols = [c for c in df_stats.columns if c != 'Index']
        n_merged = df_merged[new_cols[0]].notna().sum()

        # Save back (overwrite)
        df_merged.to_excel(measure_path, index=False)
        merged_count += n_merged

    print(f"Merged {merged_count} organoids with OCT stats")

    # Return summary for report
    return merged_count


def generate_summary():
    """Generate a summary of all extended features."""
    print("\n--- Generating summary ---")

    # Load one sample to get column names
    sample = pd.read_excel('Data/nnUNet_FXN_2023/FXN_0701/measure_excel/B10_0701.xlsx')
    all_cols = sample.columns.tolist()

    original_cols = ['Index', 'Organoids_Volume', 'Organoids_Volume_Fill',
                     'Organoids_Surface', 'Cavity_Volume', 'CavityNum',
                     'LongAxis', 'ShortAxis', 'Wall_Thickness',
                     'Sphericity', 'Scatt_Mean', 'Scatt_STD']
    new_cols = [c for c in all_cols if c not in original_cols]

    print(f"Original features: {len(original_cols)}")
    print(f"New OCT stats features: {len(new_cols)}")
    print(f"New features: {new_cols}")

    # Save summary
    summary = pd.DataFrame({
        'Feature_Type': ['Original'] * len(original_cols) + ['OCT_Stats'] * len(new_cols),
        'Feature_Name': original_cols + new_cols,
    })
    summary.to_excel(os.path.join(REPORTS_DIR, 'extended_features_summary.xlsx'), index=False)
    print(f"Summary saved to {REPORTS_DIR}/extended_features_summary.xlsx")


def main():
    print("=" * 60)
    print("Merge OCT Stats into Measure Excel")
    print("=" * 60)

    total = 0
    for config in DATA_CONFIGS:
        count = merge_stats_for_dataset(config)
        total += count

    print(f"\n{'='*60}")
    print(f"Total merged: {total} organoids")
    print(f"{'='*60}")

    generate_summary()
    print("\n[Done]")


if __name__ == "__main__":
    main()
