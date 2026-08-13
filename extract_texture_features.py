"""
Extract GLCM + GLRLM texture features for each organoid from OCT data.

Input:
  - Original OCT: Data/nnUNet_FXN_2023/FXN_0701_fixed/*.nii.gz
  - Instance labels: Data/nnUNet_FXN_2023/FXN_0701/seg_label/*.mat
  - Measure excel: Data/nnUNet_FXN_2023/FXN_0701/measure_excel/*.xlsx

Output:
  - reports/texture_features_0701.xlsx (one row per organoid)
  - reports/texture_features_0703.xlsx

Strategy:
  - For each organoid with Volume_Fill >= 200 voxels, extract masked 3D GLCM + GLRLM
  - For smaller organoids, fill with NaN (too few voxels for stable statistics)
  - Uses fast numpy-based GLCM/GLRLM (no skimage dependency for texture)
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from scipy import ndimage
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ================= 配置 =================
DATA_CONFIGS = [
    {
        'name': 'FXN_0701',
        'nii_dir': 'Data/nnUNet_FXN_2023/FXN_0701_fixed',
        'label_dir': 'Data/nnUNet_FXN_2023/FXN_0701/seg_label',
        'measure_dir': 'Data/nnUNet_FXN_2023/FXN_0701/measure_excel',
        'output': 'reports/texture_features_0701.xlsx',
    },
    {
        'name': 'FXN_0703',
        'nii_dir': 'Data/nnUNet_FXN_2023/FXN_0703_fixed',
        'label_dir': 'Data/nnUNet_FXN_2023/FXN_0703/seg_label',
        'measure_dir': 'Data/nnUNet_FXN_2023/FXN_0703/measure_excel',
        'output': 'reports/texture_features_0703.xlsx',
    },
]

MIN_VOLUME = 200   # Minimum voxels for texture extraction
LEVELS = 32        # Gray-level quantization
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==========================================


def quantize_volume(volume, mask, levels=32):
    """Quantize volume intensities within mask to [0, levels-1]."""
    if not np.any(mask):
        return np.zeros_like(volume, dtype=np.uint8)
    roi_vals = volume[mask]
    vmin, vmax = roi_vals.min(), roi_vals.max()
    if vmax - vmin < 1e-9:
        return np.zeros_like(volume, dtype=np.uint8)
    quantized = np.zeros_like(volume, dtype=np.uint8)
    quantized[mask] = np.clip(((roi_vals - vmin) / (vmax - vmin) * (levels - 1)).astype(int), 0, levels - 1)
    return quantized


def fast_glcm_3d(volume, mask, levels=32):
    """
    Fast 3D GLCM via averaged orthogonal 2D planes (XY, XZ, YZ).
    Returns dict of texture features.
    """
    if not np.any(mask):
        return {k: np.nan for k in ['Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Dissimilarity', 'Entropy', 'ASM']}

    # Crop to bbox for speed
    coords = np.where(mask)
    zmin, zmax = int(coords[0].min()), int(coords[0].max()) + 1
    ymin, ymax = int(coords[1].min()), int(coords[1].max()) + 1
    xmin, xmax = int(coords[2].min()), int(coords[2].max()) + 1
    vol_crop = volume[zmin:zmax, ymin:ymax, xmin:xmax]
    mask_crop = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    # Build co-occurrence matrices for 3 directions
    glcms = []
    for axis in [(0, 1), (0, 2), (1, 2)]:
        glcm = np.zeros((levels, levels), dtype=np.float64)
        # Sum over the remaining axis
        ax = 3 - sum(axis)  # the third axis
        for i in range(vol_crop.shape[ax]):
            slicer = [slice(None)] * 3
            slicer[ax] = i
            plane = vol_crop[tuple(slicer)]
            plane_mask = mask_crop[tuple(slicer)]
            if not np.any(plane_mask):
                continue
            # For each valid pixel, count pair with neighbor (offset=1)
            p = plane.astype(int)
            p_shift = np.roll(p, shift=-1, axis=1)
            m = plane_mask.astype(bool)
            m_shift = np.roll(m, shift=-1, axis=1)
            valid = m & m_shift
            if np.any(valid):
                np.add.at(glcm, (p[valid], p_shift[valid]), 1)
        if glcm.sum() > 0:
            glcm /= glcm.sum()
            glcms.append(glcm)

    if not glcms:
        return {k: np.nan for k in ['Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Dissimilarity', 'Entropy', 'ASM']}

    # Average GLCM
    glcm = np.mean(glcms, axis=0)
    if glcm.sum() == 0:
        return {k: np.nan for k in ['Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Dissimilarity', 'Entropy', 'ASM']}

    # Features
    i_idx, j_idx = np.indices(glcm.shape)
    contrast = np.sum(glcm * (i_idx - j_idx) ** 2)
    dissimilarity = np.sum(glcm * np.abs(i_idx - j_idx))
    energy = np.sum(glcm ** 2)
    asm = energy
    homogeneity = np.sum(glcm / (1 + (i_idx - j_idx) ** 2))
    # Entropy
    glcm_nonzero = glcm[glcm > 0]
    entropy = -np.sum(glcm_nonzero * np.log2(glcm_nonzero))
    # Correlation
    mu_i = np.sum(i_idx * glcm)
    mu_j = np.sum(j_idx * glcm)
    sigma_i = np.sqrt(np.sum((i_idx - mu_i) ** 2 * glcm))
    sigma_j = np.sqrt(np.sum((j_idx - mu_j) ** 2 * glcm))
    if sigma_i > 1e-9 and sigma_j > 1e-9:
        correlation = np.sum(glcm * (i_idx - mu_i) * (j_idx - mu_j)) / (sigma_i * sigma_j)
    else:
        correlation = 0.0

    return {
        'Texture_Contrast': contrast,
        'Texture_Correlation': correlation,
        'Texture_Energy': energy,
        'Texture_Homogeneity': homogeneity,
        'Texture_Dissimilarity': dissimilarity,
        'Texture_Entropy': entropy,
        'Texture_ASM': asm,
    }


def fast_glrlm_3d(volume, mask, levels=32):
    """
    Fast 3D GLRLM via principal-axis run-length counting.
    Returns dict of run-length features.
    """
    if not np.any(mask):
        return {k: np.nan for k in ['SRE', 'LRE', 'RLNU', 'RP', 'GLNU', 'LGRE', 'HGRE']}

    coords = np.where(mask)
    zmin, zmax = int(coords[0].min()), int(coords[0].max()) + 1
    ymin, ymax = int(coords[1].min()), int(coords[1].max()) + 1
    xmin, xmax = int(coords[2].min()), int(coords[2].max()) + 1
    vol_crop = volume[zmin:zmax, ymin:ymax, xmin:xmax]
    mask_crop = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    # Run-length matrices for 3 principal axes
    rlms = []
    for ax in [0, 1, 2]:
        rlm = np.zeros((levels, max(vol_crop.shape)), dtype=np.float64)
        other_axes = [i for i in range(3) if i != ax]
        for i in range(vol_crop.shape[other_axes[0]]):
            for j in range(vol_crop.shape[other_axes[1]]):
                slicer = [slice(None)] * 3
                slicer[other_axes[0]] = i
                slicer[other_axes[1]] = j
                line = vol_crop[tuple(slicer)]
                line_mask = mask_crop[tuple(slicer)]
                if not np.any(line_mask):
                    continue
                # Count runs within mask
                current_val = -1
                run_len = 0
                for k in range(len(line)):
                    if line_mask[k]:
                        v = int(line[k])
                        if v == current_val:
                            run_len += 1
                        else:
                            if run_len > 0 and current_val >= 0:
                                rlm[current_val, min(run_len - 1, rlm.shape[1] - 1)] += 1
                            current_val = v
                            run_len = 1
                    else:
                        if run_len > 0 and current_val >= 0:
                            rlm[current_val, min(run_len - 1, rlm.shape[1] - 1)] += 1
                        current_val = -1
                        run_len = 0
                if run_len > 0 and current_val >= 0:
                    rlm[current_val, min(run_len - 1, rlm.shape[1] - 1)] += 1
        if rlm.sum() > 0:
            rlms.append(rlm)

    if not rlms:
        return {k: np.nan for k in ['SRE', 'LRE', 'RLNU', 'RP', 'GLNU', 'LGRE', 'HGRE']}

    rlm = np.mean(rlms, axis=0)
    if rlm.sum() == 0:
        return {k: np.nan for k in ['SRE', 'LRE', 'RLNU', 'RP', 'GLNU', 'LGRE', 'HGRE']}

    # Features
    Nr = rlm.sum()
    Ng, Nr_max = rlm.shape
    j_idx = np.arange(1, Nr_max + 1)
    i_idx = np.arange(1, Ng + 1).reshape(-1, 1)

    sre = np.sum(rlm / (j_idx ** 2)) / Nr
    lre = np.sum(rlm * (j_idx ** 2)) / Nr
    glnu = np.sum((rlm.sum(axis=1) ** 2)) / Nr
    rl_nu = np.sum((rlm.sum(axis=0) ** 2)) / Nr
    rp = Nr / np.sum(rlm * j_idx)
    # Low / High gray level run emphasis
    lgre = np.sum(rlm / (i_idx ** 2)) / Nr
    hgre = np.sum(rlm * (i_idx ** 2)) / Nr

    return {
        'Texture_GLRLM_SRE': sre,
        'Texture_GLRLM_LRE': lre,
        'Texture_GLRLM_RLNU': rl_nu,
        'Texture_GLRLM_RP': rp,
        'Texture_GLRLM_GLNU': glnu,
        'Texture_GLRLM_LGRE': lgre,
        'Texture_GLRLM_HGRE': hgre,
    }


def extract_textures_for_well(nii_path, label_path, measure_path):
    """Extract texture features for all organoids in one well."""
    # Load original OCT
    img = nib.load(nii_path)
    data_nii = img.get_fdata()
    # Transpose to match .mat label dimensions
    data_nii = np.transpose(data_nii, (1, 2, 0))

    # Load label
    mat = loadmat(label_path)
    labels = mat['Data_label']

    # Load measure excel
    df_measure = pd.read_excel(measure_path)

    results = []
    for _, row in df_measure.iterrows():
        index_str = str(row['Index'])
        organoid_id = int(index_str.split('_')[-1])
        volume = row['Organoids_Volume_Fill']

        if volume < MIN_VOLUME:
            # Too small for stable texture statistics
            res = {'Index': index_str}
            for k in ['Texture_Contrast', 'Texture_Correlation', 'Texture_Energy',
                      'Texture_Homogeneity', 'Texture_Dissimilarity', 'Texture_Entropy', 'Texture_ASM',
                      'Texture_GLRLM_SRE', 'Texture_GLRLM_LRE', 'Texture_GLRLM_RLNU',
                      'Texture_GLRLM_RP', 'Texture_GLRLM_GLNU', 'Texture_GLRLM_LGRE', 'Texture_GLRLM_HGRE']:
                res[k] = np.nan
            results.append(res)
            continue

        mask = labels == organoid_id
        if not np.any(mask):
            continue

        qvol = quantize_volume(data_nii, mask, levels=LEVELS)
        glcm_feats = fast_glcm_3d(qvol, mask, levels=LEVELS)
        glrlm_feats = fast_glrlm_3d(qvol, mask, levels=LEVELS)

        res = {'Index': index_str}
        res.update(glcm_feats)
        res.update(glrlm_feats)
        results.append(res)

    return pd.DataFrame(results)


def process_dataset(config):
    """Process one dataset (0701 or 0703)."""
    print(f"\n{'='*60}")
    print(f"Processing: {config['name']}")
    print(f"{'='*60}")

    # Find all measure excel files
    measure_files = glob.glob(os.path.join(config['measure_dir'], '*.xlsx'))
    print(f"Found {len(measure_files)} measure_excel files")

    all_results = []
    for measure_path in measure_files:
        well_name = os.path.basename(measure_path).replace('.xlsx', '')
        # well_name is like "B10_0701", nii is like "B10_1_0000.nii.gz"
        well_prefix = well_name.split('_')[0]
        nii_path = os.path.join(config['nii_dir'], f"{well_prefix}_1_0000.nii.gz")
        label_path = os.path.join(config['label_dir'], f"{well_name}_label.mat")

        if not os.path.exists(nii_path):
            print(f"  [SKIP] NII not found: {nii_path}")
            continue
        if not os.path.exists(label_path):
            print(f"  [SKIP] Label not found: {label_path}")
            continue

        try:
            df_tex = extract_textures_for_well(nii_path, label_path, measure_path)
            all_results.append(df_tex)
            print(f"  [OK] {well_name}: {len(df_tex)} organoids, {df_tex['Texture_Contrast'].notna().sum()} with texture")
        except Exception as e:
            print(f"  [ERR] {well_name}: {e}")

    if not all_results:
        print("[ERR] No texture data extracted.")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_excel(config['output'], index=False)
    print(f"\n[Done] Saved to {config['output']} ({len(df_all)} rows)")

    # Print summary
    tex_cols = [c for c in df_all.columns if c.startswith('Texture_')]
    print("\nTexture feature summary (non-NaN count):")
    for col in tex_cols:
        n_valid = df_all[col].notna().sum()
        print(f"  {col}: {n_valid} / {len(df_all)} ({n_valid/len(df_all)*100:.1f}%)")


def main():
    print("=" * 60)
    print("3D OCT Texture Feature Extraction")
    print("=" * 60)

    for config in DATA_CONFIGS:
        process_dataset(config)

    print("\n[All Done]")


if __name__ == "__main__":
    main()
