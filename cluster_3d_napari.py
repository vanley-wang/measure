import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import ndimage
from skimage.measure import marching_cubes
import napari
from PIL import Image
from tqdm import tqdm

# ================= 配置 =================
BASE = "Data/nnUNet_FXN_2023"
BATCHES = ['FXN_0701', 'FXN_0703']

# Amira 颜色映射 (cluster_id -> RGB, 0-1)
# 来源: Data/FXN_2023_new（ICC）/cluster_amira/My_colormap.am
CLUSTER_COLORS = {
    0: np.array([0.8627, 0.0941, 0.0941], dtype=np.float32),   # #dc1818 red
    1: np.array([0.9490, 0.8471, 0.0706], dtype=np.float32),   # #f2d918 yellow
    2: np.array([0.0000, 0.8392, 0.1647], dtype=np.float32),   # #00d62a green
    3: np.array([0.2314, 0.2314, 0.9686], dtype=np.float32),   # #3b3bf7 blue
}

# 是否对 mask 做高斯平滑（模拟 Amira smoothingExtent=5）
SMOOTH_SIGMA = 1.0

# 输出分辨率
OUTPUT_SIZE = (3000, 3000)

# 相机初始参数（参考 Amira B4.hx，需微调）
CAMERA_ELEVATION = -60.0
CAMERA_AZIMUTH = 30.0
CAMERA_ROLL = 0.0
CAMERA_PERSPECTIVE = 45.0


def extract_id_from_index(val):
    """从 Index 字符串提取数字 ID，如 B4_0701_2 -> 2"""
    if isinstance(val, (int, float, np.integer)):
        return int(val)
    if isinstance(val, str):
        match = re.search(r'_(\d+)$', val)
        if match:
            return int(match.group(1))
    return -1


def build_id_to_cluster(df):
    """从 cluster_merge DataFrame 构建 {instance_id: cluster_id} 映射"""
    mapping = {}
    for _, row in df.iterrows():
        oid = extract_id_from_index(row['Index'])
        if oid > 0:
            mapping[oid] = int(row['Cluster'])
    return mapping


def extract_cluster_mesh(label_vol, id_to_cluster, cluster_id, smooth_sigma=0.0):
    """
    为指定 cluster 提取表面 mesh。
    返回 (vertices, faces, normals) 或 None（如果该 cluster 为空）。
    """
    oids = [oid for oid, cid in id_to_cluster.items() if cid == cluster_id]
    if not oids:
        return None

    mask = np.isin(label_vol, oids)
    if not np.any(mask):
        return None

    if smooth_sigma > 0:
        mask_float = ndimage.gaussian_filter(mask.astype(np.float32), sigma=smooth_sigma)
    else:
        mask_float = mask.astype(np.float32)

    try:
        verts, faces, normals, _ = marching_cubes(mask_float, level=0.5)
    except ValueError:
        return None

    return verts, faces, normals


def render_well(well_name, label_vol, id_to_cluster, out_dir,
                output_size=OUTPUT_SIZE, smooth_sigma=SMOOTH_SIGMA,
                elevation=CAMERA_ELEVATION, azimuth=CAMERA_AZIMUTH,
                roll=CAMERA_ROLL, perspective=CAMERA_PERSPECTIVE,
                zoom_override=None):
    """
    用 napari 渲染单个孔位的 3D cluster 表面，并截图保存。
    """
    all_vertices = []
    all_faces = []
    all_colors = []
    face_offset = 0

    for cid in range(4):
        result = extract_cluster_mesh(label_vol, id_to_cluster, cid, smooth_sigma=smooth_sigma)
        if result is None:
            continue
        verts, faces, _ = result

        color = CLUSTER_COLORS[cid]
        vertex_colors = np.tile(color, (len(verts), 1))

        all_vertices.append(verts)
        all_faces.append(faces + face_offset)
        all_colors.append(vertex_colors)
        face_offset += len(verts)

    if not all_vertices:
        print(f"  [WARN] {well_name}: 无有效 mesh")
        return False

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    vertex_colors = np.vstack(all_colors)

    # napari 颜色需要 0-1 的 RGBA
    rgba = np.concatenate([vertex_colors, np.ones((len(vertex_colors), 1), dtype=np.float32)], axis=1)

    viewer = napari.Viewer(show=False)
    viewer.add_surface(
        (vertices, faces),
        vertex_colors=rgba,
        shading='smooth',
        name=f'{well_name}_clusters'
    )
    viewer.dims.ndisplay = 3
    viewer.theme = 'dark'

    # 设置相机
    viewer.camera.perspective = perspective
    viewer.camera.angles = (elevation, azimuth, roll)

    # 自动 zoom：让场景居中并填满画面
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    diagonal = np.linalg.norm(bbox_max - bbox_min)

    viewer.camera.center = center
    if zoom_override is not None:
        viewer.camera.zoom = zoom_override
    else:
        viewer.camera.zoom = max(output_size) / diagonal * 0.7

    # 高分辨率截图
    qv = viewer.window._qt_viewer
    qv.canvas.size = output_size[::-1]  # vispy canvas size 是 (width, height)
    img = viewer.screenshot(canvas_only=True)
    viewer.close()

    # 裁剪 alpha 通道，保留 RGB
    rgb = img[..., :3]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{well_name}_3D.png")
    Image.fromarray(rgb).save(out_path)
    print(f"  [OK] {well_name}: 已保存 {out_path} ({img.shape[1]}x{img.shape[0]})")
    return True


def process_one_well(well_name, label_path, merge_path, out_dir,
                     output_size=OUTPUT_SIZE, smooth_sigma=SMOOTH_SIGMA,
                     elevation=CAMERA_ELEVATION, azimuth=CAMERA_AZIMUTH,
                     zoom_override=None):
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

        success = render_well(well_name, label_vol, id_to_cluster, out_dir,
                              output_size=output_size, smooth_sigma=smooth_sigma,
                              elevation=elevation, azimuth=azimuth,
                              zoom_override=zoom_override)
        return f"[OK] {well_name}" if success else f"[WARN] {well_name}: 渲染失败"

    except Exception as e:
        return f"[ERR] {well_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='3D cluster surface rendering with napari')
    parser.add_argument('--base-dir', default=BASE, help='Project root directory')
    parser.add_argument('--batch', choices=['0701', '0703', 'all'], default='all',
                        help='Process single batch only')
    parser.add_argument('--well', default=None, help='Process single well only (e.g. B4_0701)')
    parser.add_argument('--smooth', type=float, default=SMOOTH_SIGMA,
                        help='Gaussian smoothing sigma before marching cubes (0=disable)')
    parser.add_argument('--size', type=int, default=3000, help='Output image size in pixels')
    parser.add_argument('--elevation', type=float, default=CAMERA_ELEVATION,
                        help='Camera elevation angle')
    parser.add_argument('--azimuth', type=float, default=CAMERA_AZIMUTH,
                        help='Camera azimuth angle')
    parser.add_argument('--zoom', type=float, default=None,
                        help='Override camera zoom (auto if not set)')
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
            tasks.append((well_name, label_path, merge_path, out_dir,
                           (args.size, args.size), args.smooth,
                           args.elevation, args.azimuth, args.zoom))

    if not tasks:
        print("[WARN] 未找到任何可处理任务")
        return

    print(f">>> 共 {len(tasks)} 个孔位待处理...")
    for task in tqdm(tasks, desc="3D rendering"):
        result = process_one_well(*task)
        if "[ERR]" in result or "[WARN]" in result:
            print(result)

    print("\n[Done] 全部完成！输出目录: */cluster_3d/")


if __name__ == "__main__":
    main()
