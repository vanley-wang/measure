import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import ndimage
from skimage.measure import label as sk_label
from tqdm import tqdm
import argparse

# ================= 配置区域 =================
BASE_DIR = "Data/nnUNet_FXN_2023"
BATCHES = ['FXN_0701', 'FXN_0703']
MIN_VOLUME = 50          # 最小体积阈值（体素数），小于此值的噪声将被过滤
USE_MARCHING_CUBES = True  # True=用 marching_cubes 计算表面积（慢但更准）；False=边界体素 proxy（快）


def compute_surface_area(mask_3d):
    """
    计算 3D 单实例 mask 的表面积 proxy。
    关键优化：先裁切到 bounding box，再算边界体素，避免在全体积上操作。
    """
    coords = np.where(mask_3d)
    if len(coords[0]) == 0:
        return 0.0
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1

    cropped = mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]

    if USE_MARCHING_CUBES:
        try:
            from skimage.measure import marching_cubes, mesh_surface_area
            verts, faces, _, _ = marching_cubes(cropped, level=0.5)
            return mesh_surface_area(verts, faces)
        except Exception:
            pass

    eroded = ndimage.binary_erosion(cropped)
    boundary = cropped & (~eroded)
    return float(np.count_nonzero(boundary))


def process_one_well(label_path, fill_path, date_suffix):
    """
    处理单个孔位的 seg_label，提取形态学特征。
    核心优化：使用 scipy.ndimage.find_objects 一次性获取所有标签的 bbox（C 实现），
    之后每个器官只操作小裁切，避免在全尺寸 512x800x800 上反复扫描。
    """
    rows = []
    try:
        label_mat = loadmat(label_path)
        label_data = label_mat['Data_label']           # 填充后标记
        label_raw = label_mat['Data_label_raw']        # 原始未填充标记

        well_id = os.path.basename(label_path).split('_')[0]

        # 一次性获取所有填充标签的 bbox（C 实现，非常快，只需扫描全体积 1 次）
        slices = ndimage.find_objects(label_data)
        if not slices:
            return [], None

        for fill_id in range(1, len(slices) + 1):
            sl = slices[fill_id - 1]
            if sl is None:
                continue

            # 裁切后的填充区域（通常只有几十×几十×几十）
            crop_fill = label_data[sl]
            fill_mask = (crop_fill == fill_id)
            volume_fill = int(np.count_nonzero(fill_mask))

            if volume_fill < MIN_VOLUME:
                continue

            # 在对应裁切区域中找重叠最多的原始标签
            crop_raw = label_raw[sl]
            overlap = crop_raw[fill_mask]
            overlap = overlap[overlap > 0]
            if len(overlap) == 0:
                continue
            raw_id = int(np.bincount(overlap).argmax())

            raw_mask = (crop_raw == raw_id)
            volume = int(np.count_nonzero(raw_mask))

            # 囊腔（小裁切上计算）
            cavity_mask = fill_mask & (~raw_mask)
            cavity_volume = int(np.count_nonzero(cavity_mask))
            _, cavity_num = sk_label(cavity_mask, connectivity=3, return_num=True)

            # 表面积（小裁切上计算）
            surface = compute_surface_area(fill_mask)

            # bbox 尺寸直接从 slice 获取，无需 regionprops
            long_axis = float(max(
                sl[0].stop - sl[0].start,
                sl[1].stop - sl[1].start,
                sl[2].stop - sl[2].start
            ))
            short_axis = float(min(
                sl[0].stop - sl[0].start,
                sl[1].stop - sl[1].start,
                sl[2].stop - sl[2].start
            ))

            wall_thickness = max(1.0, volume_fill / (surface + 1e-6) * 0.25)
            sphericity = (np.pi ** (1 / 3)) * ((6 * volume_fill) ** (2 / 3)) / (surface + 1e-6)
            sphericity = min(sphericity, 5.0)

            rows.append({
                'Index': f"{well_id}_{date_suffix}_{fill_id}",
                'Organoids_Volume': volume,
                'Organoids_Volume_Fill': volume_fill,
                'Organoids_Surface': round(surface, 2),
                'Cavity_Volume': cavity_volume,
                'CavityNum': cavity_num,
                'LongAxis': long_axis,
                'ShortAxis': short_axis,
                'Wall_Thickness': round(wall_thickness, 4),
                'Sphericity': round(sphericity, 4),
                'Scatt_Mean': np.nan,
                'Scatt_STD': np.nan,
            })

    except Exception as e:
        return [], f"[ERR] {os.path.basename(label_path)}: {e}"

    return rows, None


def process_batch(batch, base_dir):
    """处理一个批次（如 FXN_0701）"""
    label_dir = os.path.join(base_dir, batch, 'seg_label')
    fill_dir = os.path.join(base_dir, batch, 'seg_fill')
    out_dir = os.path.join(base_dir, batch, 'measure_excel')
    os.makedirs(out_dir, exist_ok=True)

    date_suffix = batch.replace('FXN_', '')  # e.g. 0701

    label_files = sorted(glob.glob(os.path.join(label_dir, '*_label.mat')))
    if not label_files:
        print(f"[WARN] No label files in {label_dir}")
        return 0

    print(f"\n[Batch] {batch} | {len(label_files)} wells")

    total_organs = 0
    errors = []

    # 单线程处理（Windows 下多进程导入重型库会导致卡死）
    for label_path in tqdm(label_files, desc=f"{date_suffix} wells"):
        well_id = os.path.basename(label_path).split('_')[0]
        fill_name = os.path.basename(label_path).replace('_label', '_fill')
        fill_path = os.path.join(fill_dir, fill_name)

        if not os.path.exists(fill_path):
            print(f"  [WARN] Missing fill: {fill_name}, skip")
            continue

        rows, err = process_one_well(label_path, fill_path, date_suffix)

        if err:
            errors.append(err)
            continue

        if rows:
            df = pd.DataFrame(rows)
            out_path = os.path.join(out_dir, f"{well_id}_{date_suffix}.xlsx")
            df.to_excel(out_path, index=False)
            total_organs += len(rows)
            print(f"  [OK] {well_id}_{date_suffix}: {len(rows)} organoids")

    for e in errors:
        print(e)

    print(f"  [Done] {batch}: {total_organs} organoids saved to {out_dir}")
    return total_organs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract morphological features from seg_label/*.mat')
    parser.add_argument('--base-dir', default=BASE_DIR, help='Project root directory')
    parser.add_argument('--batch', choices=['0701', '0703', 'all'], default='all',
                        help='Process single batch only')
    parser.add_argument('--min-volume', type=int, default=MIN_VOLUME,
                        help='Minimum organoid volume (voxels) to keep')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast boundary-voxel proxy instead of marching_cubes')
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    MIN_VOLUME = args.min_volume
    USE_MARCHING_CUBES = not args.fast

    batches_to_run = BATCHES if args.batch == 'all' else [
        b for b in BATCHES if b.endswith(args.batch)
    ]

    grand_total = 0
    for batch in batches_to_run:
        grand_total += process_batch(batch, BASE_DIR)

    print(f"\n[All Done] Total {grand_total} organoids extracted")
