"""
对比验证：用同一批 FXN 原始 .mat mask，分别用我的 Python 代码重新测量，
与原始 measure_excel 中的值做逐 organoid 对比，找出测量代码差异。
测试两种 Surface 计算方法：边界体素法 vs marching_cubes
"""
import os, gc
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import ndimage
from skimage.measure import label as sk_label, marching_cubes, mesh_surface_area

ICC_DIR = r'D:\Desktop\music\measure\Data\FXN_2023_new（ICC）'
MIN_VOLUME = 50

MORPH_FEATURES = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity'
]


def compute_surface_boundary(mask_3d):
    coords = np.where(mask_3d)
    if len(coords[0]) == 0:
        return 0.0
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1
    cropped = mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]
    eroded = ndimage.binary_erosion(cropped)
    boundary = cropped & (~eroded)
    return float(np.count_nonzero(boundary))


def compute_surface_marching(mask_3d):
    coords = np.where(mask_3d)
    if len(coords[0]) == 0:
        return 0.0
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1
    cropped = mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]
    try:
        verts, faces, _, _ = marching_cubes(cropped.astype(float), level=0.5)
        return mesh_surface_area(verts, faces)
    except Exception:
        return compute_surface_boundary(mask_3d)


def measure_from_label(label_3d, surface_method='marching'):
    """从 label 矩阵提取每个 organoid 的形态学特征"""
    surface_fn = compute_surface_marching if surface_method == 'marching' else compute_surface_boundary
    num_features = label_3d.max()
    if num_features == 0:
        return []
    slices = ndimage.find_objects(label_3d)
    rows = []
    for label_id in range(1, num_features + 1):
        sl = slices[label_id - 1]
        if sl is None:
            continue
        crop_label = label_3d[sl]
        inst_mask = (crop_label == label_id)
        volume_raw = int(np.count_nonzero(inst_mask))
        if volume_raw < MIN_VOLUME:
            continue
        try:
            fill_mask = ndimage.binary_fill_holes(inst_mask)
        except Exception:
            fill_mask = inst_mask
        volume_fill = int(np.count_nonzero(fill_mask))
        cavity_mask = fill_mask & (~inst_mask)
        cavity_volume = int(np.count_nonzero(cavity_mask))
        _, cavity_num = sk_label(cavity_mask, connectivity=3, return_num=True)
        surface = surface_fn(fill_mask)
        dz = sl[0].stop - sl[0].start
        dy = sl[1].stop - sl[1].start
        dx = sl[2].stop - sl[2].start
        long_axis = float(max(dz, dy, dx))
        short_axis = float(min(dz, dy, dx))
        wall_thickness = max(1.0, volume_fill / (surface + 1e-6) * 0.25)
        sphericity = (np.pi ** (1.0 / 3)) * ((6.0 * volume_fill) ** (2.0 / 3)) / (surface + 1e-6)
        sphericity = min(sphericity, 5.0)
        rows.append({
            'label_id': label_id,
            'Organoids_Volume': volume_raw,
            'Organoids_Volume_Fill': volume_fill,
            'Organoids_Surface': round(surface, 2),
            'Cavity_Volume': cavity_volume,
            'CavityNum': cavity_num,
            'LongAxis': round(long_axis, 6),
            'ShortAxis': round(short_axis, 6),
            'Wall_Thickness': round(wall_thickness, 4),
            'Sphericity': round(sphericity, 4),
        })
    return rows


print('=' * 70)
print('  Validation: marching_cubes vs boundary-voxel vs Original')
print('=' * 70)

WELLS = ['B4', 'B6', 'B9', 'B10', 'B11', 'F4', 'F5', 'F7', 'F8', 'F10']
DATES = [('0701', 'FXN_20230701')]

for method in ['boundary', 'marching']:
    print(f'\n{"="*70}')
    print(f'  Method: {method}')
    print(f'{"="*70}')
    all_ratios = {f: [] for f in MORPH_FEATURES}

    for date_tag, date_dir in DATES:
        label_dir = os.path.join(ICC_DIR, date_dir, 'seg_label')
        excel_dir = os.path.join(ICC_DIR, date_dir, 'measure_excel')

        for well in WELLS:
            mat_path = os.path.join(label_dir, f'{well}_{date_tag}_label.mat')
            xlsx_path = os.path.join(excel_dir, f'{well}_{date_tag}.xlsx')
            if not os.path.exists(mat_path) or not os.path.exists(xlsx_path):
                continue

            df_orig = pd.read_excel(xlsx_path)
            mat = sio.loadmat(mat_path)
            label_key = None
            for k in mat:
                if not k.startswith('_') and isinstance(mat[k], np.ndarray) and mat[k].ndim == 3:
                    label_key = k
                    break
            if label_key is None:
                continue
            label_3d = mat[label_key].astype(np.int32)
            rows_new = measure_from_label(label_3d, surface_method=method)
            df_new = pd.DataFrame(rows_new)

            df_orig_s = df_orig.sort_values('Organoids_Volume').reset_index(drop=True)
            df_new_s = df_new.sort_values('Organoids_Volume').reset_index(drop=True)
            n_match = min(len(df_orig_s), len(df_new_s))

            for f in MORPH_FEATURES:
                if f in df_orig_s.columns and f in df_new_s.columns:
                    orig_vals = df_orig_s[f].values[:n_match]
                    new_vals = df_new_s[f].values[:n_match]
                    mask = (orig_vals != 0) & (new_vals != 0)
                    if mask.sum() > 0:
                        ratios = new_vals[mask] / orig_vals[mask]
                        all_ratios[f].extend(ratios.tolist())

    print(f'  {"Feature":<25s} {"Median Ratio":>12s} {"Mean Ratio":>12s} {"Verdict":>15s}')
    print('-' * 70)
    for f in MORPH_FEATURES:
        ratios = np.array(all_ratios[f])
        if len(ratios) > 0:
            med = np.median(ratios)
            mean = np.mean(ratios)
            if abs(med - 1.0) > 0.5:
                verdict = '⚠⚠ LARGE DIFF'
            elif abs(med - 1.0) > 0.1:
                verdict = '⚠ MODERATE'
            else:
                verdict = '✓ OK'
            print(f'  {f:<25s} {med:>12.4f} {mean:>12.4f} {verdict:>15s}')

# Detail: B4_0701 with marching_cubes
print(f'\n{"="*70}')
print(f'  DETAIL: B4_0701 marching_cubes vs original (first 15 organoids)')
print(f'{"="*70}')
mat = sio.loadmat(os.path.join(ICC_DIR, 'FXN_20230701', 'seg_label', 'B4_0701_label.mat'))
label_key = [k for k in mat if not k.startswith('_') and isinstance(mat[k], np.ndarray) and mat[k].ndim == 3][0]
label_3d = mat[label_key].astype(np.int32)
rows_mc = measure_from_label(label_3d, 'marching')
rows_bv = measure_from_label(label_3d, 'boundary')
df_mc = pd.DataFrame(rows_mc)
df_bv = pd.DataFrame(rows_bv)
df_orig = pd.read_excel(os.path.join(ICC_DIR, 'FXN_20230701', 'measure_excel', 'B4_0701.xlsx'))

df_mc_s = df_mc.sort_values('Organoids_Volume').reset_index(drop=True)
df_bv_s = df_bv.sort_values('Organoids_Volume').reset_index(drop=True)
df_orig_s = df_orig.sort_values('Organoids_Volume').reset_index(drop=True)
n = min(15, len(df_orig_s), len(df_mc_s))

print(f'  {"Vol_o":>7s} {"Vol_m":>7s} {"Surf_o":>8s} {"Surf_mc":>8s} {"Surf_bv":>8s} {"mc/o":>6s} {"bv/o":>6s} {"Spher_o":>8s} {"Spher_mc":>8s}')
for i in range(n):
    vo = df_orig_s.loc[i, 'Organoids_Volume']
    so = df_orig_s.loc[i, 'Organoids_Surface']
    spo = df_orig_s.loc[i, 'Sphericity']
    smc = df_mc_s.loc[i, 'Organoids_Surface']
    sbv = df_bv_s.loc[i, 'Organoids_Surface']
    spmc = df_mc_s.loc[i, 'Sphericity']
    r_mc = smc / so if so != 0 else 0
    r_bv = sbv / so if so != 0 else 0
    print(f'  {vo:>7d} {vo:>7d} {so:>8.1f} {smc:>8.1f} {sbv:>8.1f} {r_mc:>6.3f} {r_bv:>6.3f} {spo:>8.4f} {spmc:>8.4f}')

print(f'\n[Done]')