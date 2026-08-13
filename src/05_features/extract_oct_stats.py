"""
Fast OCT signal statistics extraction for each organoid.

Extracts intensity distribution features from original OCT within each organoid ROI:
  - Scatt_Skewness: asymmetry of scattering distribution
  - Scatt_Kurtosis: tail heaviness (presence of outliers)
  - Scatt_Entropy: Shannon entropy of intensity histogram
  - Scatt_IQR: interquartile range
  - Scatt_Median: robust central tendency
  - Scatt_CV: coefficient of variation (STD/Mean)
  - Scatt_Range: max - min
  - Scatt_10pct / Scatt_90pct: lower/upper deciles

Input:
  - Original OCT: Data/nnUNet_FXN_2023/FXN_0701_fixed/*.nii.gz
  - Instance labels: Data/nnUNet_FXN_2023/FXN_0701/seg_label/*.mat
  - Measure excel: Data/nnUNet_FXN_2023/FXN_0701/measure_excel/*.xlsx

Output:
  - reports/oct_stats_0701.xlsx
  - reports/oct_stats_0703.xlsx
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ================= 配置 =================
DATA_CONFIGS = [
    {
        'name': 'FXN_0701',
        'nii_dir': 'Data/nnUNet_FXN_2023/FXN_0701_fixed',
        'label_dir': 'Data/nnUNet_FXN_2023/FXN_0701/seg_label',
        'measure_dir': 'Data/nnUNet_FXN_2023/FXN_0701/measure_excel',
        'output': 'reports/oct_stats_0701.xlsx',
    },
    {
        'name': 'FXN_0703',
        'nii_dir': 'Data/nnUNet_FXN_2023/FXN_0703_fixed',
        'label_dir': 'Data/nnUNet_FXN_2023/FXN_0703/seg_label',
        'measure_dir': 'Data/nnUNet_FXN_2023/FXN_0703/measure_excel',
        'output': 'reports/oct_stats_0703.xlsx',
    },
]

REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==========================================


def compute_intensity_stats(vals):
    """Compute OCT intensity distribution statistics."""
    if len(vals) == 0:
        return {k: np.nan for k in [
            'Scatt_Skewness', 'Scatt_Kurtosis', 'Scatt_Entropy',
            'Scatt_IQR', 'Scatt_Median', 'Scatt_CV',
            'Scatt_Range', 'Scatt_10pct', 'Scatt_90pct'
        ]}

    # Basic stats
    mean_val = np.mean(vals)
    std_val = np.std(vals, ddof=1)
    median_val = np.median(vals)
    min_val = np.min(vals)
    max_val = np.max(vals)
    q25, q75 = np.percentile(vals, [25, 75])
    p10, p90 = np.percentile(vals, [10, 90])

    # Distribution shape
    sk = skew(vals, bias=False) if len(vals) > 2 else 0.0
    kt = kurtosis(vals, bias=False) if len(vals) > 3 else 0.0

    # Coefficient of variation
    cv = std_val / mean_val if mean_val > 1e-9 else np.nan

    # Entropy (Shannon, 16 bins)
    hist, _ = np.histogram(vals, bins=16, range=(min_val, max_val + 1e-9))
    probs = hist / hist.sum()
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))

    return {
        'Scatt_Skewness': sk,
        'Scatt_Kurtosis': kt,
        'Scatt_Entropy': entropy,
        'Scatt_IQR': q75 - q25,
        'Scatt_Median': median_val,
        'Scatt_CV': cv,
        'Scatt_Range': max_val - min_val,
        'Scatt_10pct': p10,
        'Scatt_90pct': p90,
    }


def extract_stats_for_well_fast(nii_path, label_path, measure_path):
    """Extract OCT stats for all organoids in one well (vectorized, fast)."""
    # Load original OCT
    img = nib.load(nii_path)
    data_nii = img.get_fdata()
    data_nii = np.transpose(data_nii, (1, 2, 0))

    # Load instance labels
    mat = loadmat(label_path)
    labels = mat['Data_label'].astype(np.int32)

    # Load measure excel
    df_measure = pd.read_excel(measure_path)
    organoid_ids = [int(str(row['Index']).split('_')[-1]) for _, row in df_measure.iterrows()]
    index_strs = [str(row['Index']) for _, row in df_measure.iterrows()]

    # Flatten for vectorized processing
    vol_flat = data_nii.ravel()
    lab_flat = labels.ravel()

    # Keep only labeled voxels (label > 0)
    valid = lab_flat > 0
    vol_flat = vol_flat[valid]
    lab_flat = lab_flat[valid]

    # Sort by label for contiguous processing
    sort_idx = np.argsort(lab_flat)
    vol_sorted = vol_flat[sort_idx]
    lab_sorted = lab_flat[sort_idx]

    # Find boundaries between different labels
    boundaries = np.where(np.diff(lab_sorted) != 0)[0] + 1
    boundaries = np.concatenate(([0], boundaries, [len(lab_sorted)]))

    # Pre-compute all stats
    results = []
    for idx, oid in enumerate(organoid_ids):
        # Binary search for label position
        # Find all occurrences of this label
        pos = np.searchsorted(lab_sorted, oid)
        if pos >= len(lab_sorted) or lab_sorted[pos] != oid:
            # Label not found in this well
            stats = {k: np.nan for k in [
                'Scatt_Skewness', 'Scatt_Kurtosis', 'Scatt_Entropy',
                'Scatt_IQR', 'Scatt_Median', 'Scatt_CV',
                'Scatt_Range', 'Scatt_10pct', 'Scatt_90pct'
            ]}
            stats['Index'] = index_strs[idx]
            results.append(stats)
            continue

        # Find start and end of this label block
        start = pos
        end = pos
        while end < len(lab_sorted) and lab_sorted[end] == oid:
            end += 1

        vals = vol_sorted[start:end]
        stats = compute_intensity_stats(vals)
        stats['Index'] = index_strs[idx]
        results.append(stats)

    return pd.DataFrame(results)


def process_dataset(config):
    print(f"\n{'='*60}")
    print(f"Processing: {config['name']}")
    print(f"{'='*60}")

    measure_files = sorted(glob.glob(os.path.join(config['measure_dir'], '*.xlsx')))
    print(f"Found {len(measure_files)} measure_excel files")

    all_results = []
    for measure_path in tqdm(measure_files, desc="Wells"):
        well_name = os.path.basename(measure_path).replace('.xlsx', '')
        well_prefix = well_name.split('_')[0]
        nii_path = os.path.join(config['nii_dir'], f"{well_prefix}_1_0000.nii.gz")
        label_path = os.path.join(config['label_dir'], f"{well_name}_label.mat")

        if not os.path.exists(nii_path) or not os.path.exists(label_path):
            continue

        try:
            df_stats = extract_stats_for_well_fast(nii_path, label_path, measure_path)
            all_results.append(df_stats)
        except Exception as e:
            print(f"  [ERR] {well_name}: {e}")

    if not all_results:
        print("[ERR] No data extracted.")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_excel(config['output'], index=False)
    print(f"\n[Done] Saved to {config['output']} ({len(df_all)} rows)")

    # Summary
    stat_cols = [c for c in df_all.columns if c != 'Index']
    print("\nFeature summary:")
    for col in stat_cols:
        valid = df_all[col].notna().sum()
        print(f"  {col}: {valid}/{len(df_all)} valid, mean={df_all[col].mean():.3f}, std={df_all[col].std():.3f}")


def main():
    print("=" * 60)
    print("OCT Signal Statistics Extraction")
    print("=" * 60)

    for config in DATA_CONFIGS:
        process_dataset(config)

    print("\n[All Done]")


if __name__ == "__main__":
    main()
