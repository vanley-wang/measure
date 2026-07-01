"""
3D cluster rendering via depth-buffered projection + normal-based shading.

Instead of extracting meshes, this script:
1. Builds a cluster-ID volume (each voxel = cluster 0-3 or -1 for background)
2. For each pixel (y, x), finds the front-most non-background voxel along Z
3. Computes surface normals from the depth map using gradients
4. Applies diffuse lighting per pixel
5. Outputs a 3000x3000 PNG

This is much faster than marching-cubes + mesh rendering and avoids
all OpenGL/headless issues. The visual style is orthographic with
surface shading, giving clear 3D shape cues.
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import ndimage
from PIL import Image
from tqdm import tqdm

# ================= 配置 =================
BASE = "Data/nnUNet_FXN_2023"
BATCHES = ['FXN_0701', 'FXN_0703']

# Amira 颜色映射 (cluster_id -> RGB 0-255)
CLUSTER_COLORS = {
    0: np.array([220,  24,  24], dtype=np.float32),   # #dc1818 red
    1: np.array([242, 217,  24], dtype=np.float32),   # #f2d918 yellow
    2: np.array([  0, 214,  42], dtype=np.float32),   # #00d62a green
    3: np.array([ 59,  59, 247], dtype=np.float32),   # #3b3bf7 blue
}

# 光照参数
LIGHT_DIR = np.array([0.4, -0.3, -1.0], dtype=np.float32)
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
AMBIENT = 0.30
DIFFUSE = 0.70
DEPTH_FOG = 0.15  # 越深的像素越暗

OUTPUT_SIZE = 3000


def extract_id_from_index(val):
    if isinstance(val, (int, float, np.integer)):
        return int(val)
    if isinstance(val, str):
        match = re.search(r'_(\d+)$', val)
        if match:
            return int(match.group(1))
    return -1


def build_id_to_cluster(df):
    mapping = {}
    for _, row in df.iterrows():
        oid = extract_id_from_index(row['Index'])
        if oid > 0:
            mapping[oid] = int(row['Cluster'])
    return mapping


def render_well_depth(label_vol, id_to_cluster):
    """
    Depth-buffered projection with normal shading.
    label_vol: (Z, Y, X) instance ID volume
    Returns: RGB image (Y, X, 3) uint8
    """
    z_size, y_size, x_size = label_vol.shape

    # 1. 构建 cluster 体积: -1 = 背景, 0-3 = cluster
    cluster_vol = np.full_like(label_vol, -1, dtype=np.int8)
    for oid, cid in id_to_cluster.items():
        cluster_vol[label_vol == oid] = cid

    # 2. Z-buffer: 找每个 (y, x) 最前面的 voxel
    # 从前往后扫描 Z 轴
    has_obj = cluster_vol >= 0

    # 对每个 (y, x) 列，argmax 返回第一个 True 的位置
    # 如果全 False，argmax 返回 0，我们需要额外处理
    front_z = np.argmax(has_obj, axis=0)  # shape (Y, X)
    any_obj = has_obj.max(axis=0)         # shape (Y, X), True if any voxel

    # 获取 front-most cluster ID
    y_idx, x_idx = np.meshgrid(np.arange(y_size), np.arange(x_size), indexing='ij')
    front_cid = cluster_vol[front_z, y_idx, x_idx]
    front_cid = np.where(any_obj, front_cid, -1)

    # 3. 深度图 (front_z), NaN for background
    depth_map = np.where(any_obj, front_z.astype(np.float32), np.nan)

    # 4. 从深度图计算法向量
    # 使用 Sobel 或 central difference
    # 处理 NaN: 先填充 NaN，计算完后再 mask
    depth_filled = np.nan_to_num(depth_map, nan=0.0)

    # 计算梯度
    dz_dy, dz_dx = np.gradient(depth_filled)

    # 对背景区域，梯度设为 0
    dz_dy = np.where(any_obj, dz_dy, 0.0)
    dz_dx = np.where(any_obj, dz_dx, 0.0)

    # 法向量 N = (-dz_dx, -dz_dy, 1) 然后归一化
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(dz_dx)
    norm_len = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = np.where(any_obj, nx / norm_len, 0.0)
    ny = np.where(any_obj, ny / norm_len, 0.0)
    nz = np.where(any_obj, nz / norm_len, 0.0)

    # 5. 光照计算
    diffuse = np.clip(nx * LIGHT_DIR[0] + ny * LIGHT_DIR[1] + nz * LIGHT_DIR[2], 0, 1)
    intensity = AMBIENT + DIFFUSE * diffuse

    # 深度雾效: 越深的像素越暗
    depth_norm = front_z.astype(np.float32) / z_size
    fog = 1.0 - DEPTH_FOG * depth_norm
    intensity = intensity * fog

    # 6. 合成颜色
    rgb = np.zeros((y_size, x_size, 3), dtype=np.float32)
    for cid in range(4):
        mask = (front_cid == cid)
        if not np.any(mask):
            continue
        color = CLUSTER_COLORS[cid] / 255.0
        for ch in range(3):
            rgb[:, :, ch] = np.where(mask, color[ch] * intensity, rgb[:, :, ch])

    # 7. 转为 uint8 并放大到 3000x3000
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    img = img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    return np.array(img)


def process_one_well(well_name, label_path, merge_path, out_dir):
    try:
        mat = loadmat(label_path)
        if 'Data_label' in mat:
            label_vol = mat['Data_label'].astype(np.int32)
        else:
            keys = [k for k in mat.keys() if not k.startswith('__')]
            label_vol = mat[keys[0]].astype(np.int32)

        df = pd.read_excel(merge_path)
        id_to_cluster = build_id_to_cluster(df)

        if not id_to_cluster:
            return f"[WARN] {well_name}: 无有效 ID 映射"

        rgb = render_well_depth(label_vol, id_to_cluster)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{well_name}_3D.png")
        Image.fromarray(rgb).save(out_path)
        print(f"  [OK] {well_name}: 已保存 {out_path}")
        return f"[OK] {well_name}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[ERR] {well_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='3D cluster depth-buffered rendering')
    parser.add_argument('--base-dir', default=BASE, help='Project root directory')
    parser.add_argument('--batch', choices=['0701', '0703', 'all'], default='all')
    parser.add_argument('--well', default=None, help='Process single well only')
    args = parser.parse_args()

    tasks = []
    batches = BATCHES if args.batch == 'all' else [b for b in BATCHES if b.endswith(args.batch)]

    for batch in batches:
        root = os.path.join(args.base_dir, batch)
        label_dir = os.path.join(root, 'seg_label')
        merge_dir = os.path.join(root, 'cluster_merge')
        out_dir = os.path.join(root, 'cluster_3d')

        if not os.path.exists(merge_dir):
            print(f"[WARN] 跳过 {batch}: 无 cluster_merge 目录")
            continue

        merge_files = [f for f in os.listdir(merge_dir) if f.endswith('_merge.xlsx')]
        for mf in merge_files:
            well_name = mf.replace('_merge.xlsx', '')
            if args.well and well_name != args.well:
                continue
            label_path = os.path.join(label_dir, f"{well_name}_label.mat")
            merge_path = os.path.join(merge_dir, mf)
            if not os.path.exists(label_path):
                print(f"  [WARN] 缺失标签文件: {label_path}")
                continue
            tasks.append((well_name, label_path, merge_path, out_dir))

    if not tasks:
        print("[WARN] 未找到任何可处理任务")
        return

    print(f">>> 共 {len(tasks)} 个孔位待处理...")
    for task in tqdm(tasks, desc="3D depth rendering"):
        result = process_one_well(*task)
        if "[ERR]" in result or "[WARN]" in result:
            print(result)

    print("\n[Done] 全部完成！输出目录: */cluster_3d/")


if __name__ == "__main__":
    main()
