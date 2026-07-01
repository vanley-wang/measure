import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import ndimage
from skimage.measure import marching_cubes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

# ================= 配置 =================
BASE = "Data/nnUNet_FXN_2023"
BATCHES = ['FXN_0701', 'FXN_0703']

# Amira 颜色映射 (cluster_id -> RGB 0-1)
# 来源: Data/FXN_2023_new（ICC）/cluster_amira/My_colormap.am
CLUSTER_COLORS = {
    0: np.array([0.8627, 0.0941, 0.0941], dtype=np.float32),   # #dc1818 red
    1: np.array([0.9490, 0.8471, 0.0706], dtype=np.float32),   # #f2d918 yellow
    2: np.array([0.0000, 0.8392, 0.1647], dtype=np.float32),   # #00d62a green
    3: np.array([0.2314, 0.2314, 0.9686], dtype=np.float32),   # #3b3bf7 blue
}

# 光照方向 (从左上前方照射)
LIGHT_DIR = np.array([0.5, -0.5, -1.0], dtype=np.float32)
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
AMBIENT = 0.35
DIFFUSE = 0.65

# 输出分辨率
FIGSIZE_INCH = 20
DPI = 150  # 20 * 150 = 3000

# 下采样因子 (2=体积缩小8倍，显著加速 marching cubes)
DOWNSAMPLE = 2

# 相机角度
DEFAULT_ELEV = 60.0
DEFAULT_AZIM = 140.0


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


def shade_colors(base_color, normals, light_dir=LIGHT_DIR, ambient=AMBIENT, diffuse=DIFFUSE):
    """
    基于面法向量和光照方向计算着色后的颜色。
    base_color: (3,) RGB 0-1
    normals: (N, 3) 面法向量
    返回: (N, 3) 着色后的 RGB
    """
    # dot product of each normal with light direction
    intensity = np.dot(normals, light_dir)
    intensity = np.clip(intensity, 0, 1)
    # ambient + diffuse
    shaded = base_color * (ambient + diffuse * intensity[:, np.newaxis])
    return np.clip(shaded, 0, 1)


def render_well(well_name, label_vol, id_to_cluster, out_dir,
                elev=DEFAULT_ELEV, azim=DEFAULT_AZIM,
                downsample=DOWNSAMPLE):
    """
    用 matplotlib 3D 渲染单个孔位的 cluster 表面。
    """
    all_verts = []
    all_faces = []
    all_colors = []
    face_offset = 0

    for cid in range(4):
        oids = [oid for oid, c in id_to_cluster.items() if c == cid]
        if not oids:
            continue
        mask = np.isin(label_vol, oids)
        if not np.any(mask):
            continue

        # 可选高斯平滑 + 下采样
        if downsample > 1:
            zoom_factor = 1.0 / downsample
            mask_proc = ndimage.zoom(mask.astype(np.float32), zoom_factor, order=1)
        else:
            mask_proc = mask.astype(np.float32)

        try:
            verts, faces, normals, _ = marching_cubes(mask_proc, level=0.5)
        except ValueError:
            continue

        # 顶点坐标缩放回原始尺寸
        if downsample > 1:
            verts = verts * downsample

        # 着色
        base_color = CLUSTER_COLORS[cid]
        face_colors = shade_colors(base_color, normals)

        all_verts.append(verts)
        all_faces.append(faces + face_offset)
        all_colors.append(face_colors)
        face_offset += len(verts)

    if not all_verts:
        print(f"  [WARN] {well_name}: 无有效 mesh")
        return False

    vertices = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    face_colors = np.vstack(all_colors)

    # 计算 mesh 的 tight bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    size = bbox_max - bbox_min
    max_size = size.max()
    pad = max_size * 0.05  # 5% padding

    # 创建 figure
    fig = plt.figure(figsize=(FIGSIZE_INCH, FIGSIZE_INCH))
    ax = fig.add_subplot(111, projection='3d')

    # 关键：每个 cluster 单独创建一个 Poly3DCollection
    # 避免 matplotlib 在 facecolors 包含多色时的 N^2 内存爆炸 bug
    face_offset = 0
    for cid in range(4):
        n_faces_cluster = len(all_faces[cid])
        if n_faces_cluster == 0:
            continue
        cluster_faces = faces[face_offset:face_offset + n_faces_cluster]
        cluster_colors = face_colors[face_offset:face_offset + n_faces_cluster]
        # 使用 cluster 的平均颜色作为 uniform facecolor
        # shade=False 避免 matplotlib _shade_colors bug； shade=True 在大 mesh 上会崩
        avg_color = cluster_colors.mean(axis=0)
        mesh = Poly3DCollection(
            vertices[cluster_faces],
            alpha=1.0,
            facecolor=avg_color,
            edgecolors='none',
            shade=False
        )
        ax.add_collection3d(mesh)
        face_offset += n_faces_cluster

    # 设置等比例 tight limits
    ax.set_xlim(center[0] - max_size/2 - pad, center[0] + max_size/2 + pad)
    ax.set_ylim(center[1] - max_size/2 - pad, center[1] + max_size/2 + pad)
    ax.set_zlim(center[2] - max_size/2 - pad, center[2] + max_size/2 + pad)

    # 等比例
    ax.set_box_aspect([1, 1, 1])

    # 相机角度
    ax.view_init(elev=elev, azim=azim)

    # 关闭坐标轴
    ax.axis('off')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{well_name}_3D.png")
    plt.savefig(out_path, dpi=DPI, facecolor='black', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] {well_name}: 已保存 {out_path}")
    return True


def process_one_well(well_name, label_path, merge_path, out_dir,
                     elev=DEFAULT_ELEV, azim=DEFAULT_AZIM):
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
                              elev=elev, azim=azim)
        return f"[OK] {well_name}" if success else f"[WARN] {well_name}: 渲染失败"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[ERR] {well_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='3D cluster surface rendering with matplotlib')
    parser.add_argument('--base-dir', default=BASE, help='Project root directory')
    parser.add_argument('--batch', choices=['0701', '0703', 'all'], default='all',
                        help='Process single batch only')
    parser.add_argument('--well', default=None, help='Process single well only (e.g. B4_0701)')
    parser.add_argument('--elev', type=float, default=DEFAULT_ELEV,
                        help='Camera elevation angle')
    parser.add_argument('--azim', type=float, default=DEFAULT_AZIM,
                        help='Camera azimuth angle')
    parser.add_argument('--ds', type=int, default=DOWNSAMPLE,
                        help='Downsample factor before marching cubes (1=full res)')
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
            tasks.append((well_name, label_path, merge_path, out_dir, args.elev, args.azim))

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
