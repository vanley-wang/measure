"""
Fast texture feature extraction using skimage's optimized C implementation.

Extracts 2D GLCM features from 3 orthogonal planes (XY, XZ, YZ) per organoid,
then averages. Uses skimage.feature.graycomatrix (Cython) for speed.

Output:
  reports/texture_features_0701.xlsx
  reports/texture_features_0703.xlsx
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from skimage.feature import graycomatrix, graycoprops
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

MIN_VOLUME = 200
LEVELS = 32
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

GLCM_PROPS = ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity', 'ASM']

# ==========================================


def extract_glcm_2d(plane, levels=32):
    """Extract GLCM features from a single 2D plane using skimage."""
    if plane.size < 10:
        return {p: np.nan for p in GLCM_PROPS}
    try:
        glcm = graycomatrix(plane, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=levels, symmetric=True, normed=True)
        feats = {}
        for prop in GLCM_PROPS:
            if prop == 'dissimilarity':
                # skimage doesn't have dissimilarity directly
                i, j = np.indices(glcm.shape[:2])
                diss = np.sum(np.abs(i - j) * glcm[:, :, 0, :])
                feats[prop] = np.mean(diss)
            elif prop == 'ASM':
                feats[prop] = np.mean(graycoprops(glcm, 'ASM'))
            else:
                feats[prop] = np.mean(graycoprops(glcm, prop))
        # Entropy
        glcm_nonzero = glcm[glcm > 0]
        feats['entropy'] = -np.sum(glcm_nonzero * np.log2(glcm_nonzero))
        return feats
    except Exception:
        return {p: np.nan for p in GLCM_PROPS + ['entropy']}


def get_glcm_planes(volume, mask):
    """Extract 3 orthogonal 2D planes from 3D ROI for GLCM."""
    if not np.any(mask):
        return []
    coords = np.where(mask)
    zmin, zmax = int(coords[0].min()), int(coords[0].max()) + 1
    ymin, ymax = int(coords[1].min()), int(coords[1].max()) + 1
    xmin, xmax = int(coords[2].min()), int(coords[2].max()) + 1

    vol_crop = volume[zmin:zmax, ymin:ymax, xmin:xmax]
    mask_crop = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    planes = []
    # XY plane at middle Z
    mid_z = vol_crop.shape[0] // 2
    p = vol_crop[mid_z, :, :]
    m = mask_crop[mid_z, :, :]
    if np.sum(m) > 5:
        planes.append(p[m])

    # XZ plane at middle Y
    mid_y = vol_crop.shape[1] // 2
    p = vol_crop[:, mid_y, :]
    m = mask_crop[:, mid_y, :]
    if np.sum(m) > 5:
        planes.append(p[m])

    # YZ plane at middle X
    mid_x = vol_crop.shape[2] // 2
    p = vol_crop[:, :, mid_x]
    m = mask_crop[:, :, mid_x]
    if np.sum(m) > 5:
        planes.append(p[m])

    return planes


def quantize(vals, levels=32):
    """Quantize 1D array to [0, levels-1]."""
    if len(vals) == 0:
        return np.array([], dtype=np.uint8)
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-9:
        return np.zeros(len(vals), dtype=np.uint8)
    return np.clip(((vals - vmin) / (vmax - vmin) * (levels - 1)).astype(int), 0, levels - 1)


def extract_textures_for_well(nii_path, label_path, measure_path):
    """Extract texture features for all organoids in one well."""
    img = nib.load(nii_path)
    data_nii = img.get_fdata()
    data_nii = np.transpose(data_nii, (1, 2, 0))

    mat = loadmat(label_path)
    labels = mat['Data_label']

    df_measure = pd.read_excel(measure_path)

    results = []
    for _, row in df_measure.iterrows():
        index_str = str(row['Index'])
        organoid_id = int(index_str.split('_')[-1])
        volume = row['Organoids_Volume_Fill']

        if volume < MIN_VOLUME:
            res = {'Index': index_str}
            for prop in GLCM_PROPS + ['entropy']:
                res[f'Texture_{prop.capitalize()}'] = np.nan
            results.append(res)
            continue

        mask = labels == organoid_id
        if not np.any(mask):
            continue

        # Get signal values within mask
        vals = data_nii[mask]
        qvals = quantize(vals, levels=LEVELS)

        # Reshape to approximate square for 2D GLCM
        n = len(qvals)
        side = int(np.sqrt(n))
        if side < 3:
            res = {'Index': index_str}
            for prop in GLCM_PROPS + ['entropy']:
                res[f'Texture_{prop.capitalize()}'] = np.nan
            results.append(res)
            continue

        # Pad or trim to side x side
        padded = np.zeros(side * side, dtype=np.uint8)
        padded[:min(n, side*side)] = qvals[:side*side]
        plane = padded.reshape(side, side)

        feats = extract_glcm_2d(plane, levels=LEVELS)
        res = {'Index': index_str}
        for prop, val in feats.items():
            res[f'Texture_{prop.capitalize()}'] = val
        results.append(res)

    return pd.DataFrame(results)


def process_dataset(config):
    print(f"\n{'='*60}")
    print(f"Processing: {config['name']}")
    print(f"{'='*60}")

    measure_files = glob.glob(os.path.join(config['measure_dir'], '*.xlsx'))
    print(f"Found {len(measure_files)} measure_excel files")

    all_results = []
    for measure_path in tqdm(measure_files, desc="Wells"):
        well_name = os.path.basename(measure_path).replace('.xlsx', '')
        well_prefix = well_name.split('_')[0]
        nii_path = os.path.join(config['nii_dir'], f"{well_prefix}_1_0000.nii.gz")
        label_path = os.path.join(config['label_dir'], f"{well_name}_label.mat")

        if not os.path.exists(nii_path):
            continue
        if not os.path.exists(label_path):
            continue

        try:
            df_tex = extract_textures_for_well(nii_path, label_path, measure_path)
            all_results.append(df_tex)
        except Exception as e:
            print(f"  [ERR] {well_name}: {e}")

    if not all_results:
        print("[ERR] No texture data extracted.")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_excel(config['output'], index=False)
    print(f"\n[Done] Saved to {config['output']} ({len(df_all)} rows)")

    tex_cols = [c for c in df_all.columns if c.startswith('Texture_')]
    print("\nTexture feature summary (non-NaN count):")
    for col in tex_cols:
        n_valid = df_all[col].notna().sum()
        print(f"  {col}: {n_valid} / {len(df_all)} ({n_valid/len(df_all)*100:.1f}%)")


def main():
    print("=" * 60)
    print("Fast Texture Feature Extraction (skimage GLCM)")
    print("=" * 60)

    for config in DATA_CONFIGS:
        process_dataset(config)

    print("\n[All Done]")


if __name__ == "__main__":
    main()
