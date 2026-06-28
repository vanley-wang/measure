import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage.measure import regionprops, marching_cubes, mesh_surface_area
from tqdm import tqdm
import argparse

# ================= 配置区域 =================
BASE_DIR = "Data/nnUNet_FXN_2023"
BATCHES = ['FXN_0701', 'FXN_0703']
MIN_VOLUME = 50          # 最小体积阈值（体素数），小于此值的噪声将被过滤
MAX_WORKERS = 4          # 并行进程数，Windows 报错请改为 1
USE_MARCHING_CUBES = True  # True=用 marching_cubes 计算表面积（慢但更准）；False=边界体素 proxy（快）


def compute_surface_area(mask_3d):
    """计算 3D 单实例 mask 的表面积 proxy"""
    if USE_MARCHING_CUBES:
        try:
            verts, faces, _, _ = marching_cubes(mask_3d, level=0.5)
            return mesh_surface_area(verts, faces)
        except Exception:
            pass  # 回退到边界体素法
    # 回退：3D 边界体素计数 × 1.0（作为 proxy）
    from scipy import ndimage
    eroded = ndimage.binary_erosion(mask_3d)
    boundary = mask_3d & (~eroded)
    return float(np.count_nonzero(boundary))


def map_raw_to_fill_id(label_raw, label_fill):
    """
    建立原始标记 ID -> 填充后标记 ID 的映射。
    对每个原始实例，在填充后标记中找到重叠体素最多的对应 ID。
    """
    mapping = {}
    raw_ids = np.unique(label_raw)
    raw_ids = raw_ids[raw_ids > 0]
    for rid in raw_ids:
        raw_mask = (label_raw == rid)
        overlap = label_fill[raw_mask]
        overlap = overlap[overlap > 0]
        if len(overlap) == 0:
            continue
        # 众数 = 重叠最多的填充后 ID
        fid = int(np.bincount(overlap).argmax())
        mapping[rid] = fid
    return mapping


def process_one_well(label_path, fill_path, date_suffix):
    """
    处理单个孔位的 seg_label + seg_fill，提取形态学特征。
    利用 Data_label_raw（原始未填充标记）计算原始体积与囊腔信息，
    利用 Data_label（填充后标记）计算表面积等后续特征，保证与 scatt.py ID 一致。
    返回 list of dict（每个 dict 对应一个 organoid）。
    """
    rows = []
    try:
        label_mat = loadmat(label_path)
        fill_mat = loadmat(fill_path)

        label_data = label_mat['Data_label']           # 填充后标记（供 scatt.py）
        label_raw = label_mat['Data_label_raw']        # 原始未填充标记（供囊腔计算）

        well_id = os.path.basename(label_path).split('_')[0]  # e.g. B2

        # 建立原始 ID -> 填充后 ID 映射
        id_mapping = map_raw_to_fill_id(label_raw, label_data)

        raw_ids = np.unique(label_raw)
        raw_ids = raw_ids[raw_ids > 0]

        for raw_id in raw_ids:
            if raw_id not in id_mapping:
                continue  # 原始实例在填充后找不到对应（极少见）
            fill_id = id_mapping[raw_id]

            raw_mask = (label_raw == raw_id).astype(np.uint8)
            fill_mask = (label_data == fill_id).astype(np.uint8)

            volume = int(np.count_nonzero(raw_mask))
            volume_fill = int(np.count_nonzero(fill_mask))

            if volume_fill < MIN_VOLUME:
                continue  # 过滤噪声（用填充后体积判断更稳）

            # 囊腔 = 填充后 mask 减去原始 mask
            cavity_mask = fill_mask & (~raw_mask.astype(bool))
            cavity_volume = int(np.count_nonzero(cavity_mask))
            # 孔洞连通域数（26-连通）
            from skimage.measure import label as sk_label
            _, cavity_num = sk_label(cavity_mask, connectivity=3, return_num=True)

            # 表面积等形态特征基于填充后的实心 mask（更稳定）
            surface = compute_surface_area(fill_mask)

            props = regionprops(fill_mask)
            if not props:
                continue
            p = props[0]

            bbox = p.bbox
            dims = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]
            long_axis = float(max(dims))
            short_axis = float(min(dims))

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
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help='Parallel workers (set 1 if Windows error)')
    parser.add_argument('--min-volume', type=int, default=MIN_VOLUME,
                        help='Minimum organoid volume (voxels) to keep')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast boundary-voxel proxy instead of marching_cubes')
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    MAX_WORKERS = args.workers
    MIN_VOLUME = args.min_volume
    USE_MARCHING_CUBES = not args.fast

    batches_to_run = BATCHES if args.batch == 'all' else [
        b for b in BATCHES if b.endswith(args.batch)
    ]

    grand_total = 0
    for batch in batches_to_run:
        grand_total += process_batch(batch, BASE_DIR)

    print(f"\n[All Done] Total {grand_total} organoids extracted")
