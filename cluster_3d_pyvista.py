"""
3D cluster surface rendering with pyvista.

Generates per-well 3000x3000 PNG images showing 3D organoid clusters
with smooth shading, specular lighting, and depth occlusion.

Usage:
    python cluster_3d_pyvista.py --well B4_0701
    python cluster_3d_pyvista.py --batch all
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import ndimage
from skimage.measure import marching_cubes
import pyvista as pv
from PIL import Image
from tqdm import tqdm

# ================= 配置 =================
BASE = "Data/nnUNet_FXN_2023"
BATCHES = ['FXN_0701', 'FXN_0703']

# PyVista 颜色映射 (cluster_id -> RGB 0-1)
# 论文标准四类表型 (cluster_category.md):
# 0=大囊状健康类器官(红/Cluster1), 1=大实心健康类器官(黄/Cluster2),
# 2=小实心休眠/幼类器官(绿/Cluster3), 3=极小高致密受损类器官(蓝/Cluster4)
CLUSTER_COLORS = {
    0: [0.8627, 0.0941, 0.0941],   # #dc1818 red   - Cluster1 大囊状
    1: [0.9490, 0.8471, 0.0706],   # #f2d918 yellow - Cluster2 大实心
    2: [0.0000, 0.8392, 0.1647],   # #00d62a green  - Cluster3 小实心
    3: [0.2314, 0.2314, 0.9686],   # #3b3bf7 blue   - Cluster4 高致密受损
}

LABEL_DESC = {
    0: '大囊状健康类器官 (Cluster 1)',
    1: '大实心健康类器官 (Cluster 2)',
    2: '小实心休眠/幼类器官 (Cluster 3)',
    3: '极小高致密受损类器官 (Cluster 4)',
}

OUTPUT_SIZE = 3000

# 下采样因子 (2 = 体积缩小8倍，显著加速 marching cubes)
DOWNSAMPLE = 2

SPECULAR = 0.5
SPECULAR_POWER = 15


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


def render_well_pyvista(well_name, label_vol, id_to_cluster, out_dir,
                        downsample=DOWNSAMPLE):
    """
    用 pyvista 渲染单个孔位的 3D cluster 表面。
    """
    meshes = []
    for cid in range(4):
        oids = [oid for oid, c in id_to_cluster.items() if c == cid]
        if not oids:
            continue
        mask = np.isin(label_vol, oids)
        if not np.any(mask):
            continue

        # 高斯平滑 + 下采样
        if downsample > 1:
            zoom_factor = 1.0 / downsample
            mask_proc = ndimage.zoom(mask.astype(np.float32), zoom_factor, order=1)
        else:
            mask_proc = mask.astype(np.float32)

        try:
            verts, faces, normals, _ = marching_cubes(mask_proc, level=0.5)
        except ValueError:
            continue

        # 缩放回原始坐标
        if downsample > 1:
            verts = verts * downsample

        # 坐标重排: marching_cubes 返回 (z, y, x)，转为 pyvista 标准 (x, y, z)
        verts = verts[:, [2, 1, 0]]

        # pyvista faces 格式: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        pv_faces = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
        mesh = pv.PolyData(verts, pv_faces)
        meshes.append((mesh, CLUSTER_COLORS[cid]))

    if not meshes:
        print(f"  [WARN] {well_name}: 无有效 mesh")
        return False

    # 计算场景中心
    all_verts = np.vstack([m.points for m, _ in meshes])
    center = all_verts.mean(axis=0)

    # 创建 off-screen plotter
    plotter = pv.Plotter(off_screen=True, window_size=[OUTPUT_SIZE, OUTPUT_SIZE])
    plotter.set_background('black')

    for mesh, color in meshes:
        plotter.add_mesh(
            mesh,
            color=color,
            show_edges=False,
            smooth_shading=True,
            specular=SPECULAR,
            specular_power=SPECULAR_POWER
        )

    # 自定义光照
    plotter.remove_all_lights()
    # 主光源：参考 Amira 的 specular lighting 方向
    plotter.add_light(pv.Light(
        position=(center[0] + 300, center[1] - 400, center[2] + 500),
        focal_point=center,
        color='white',
        intensity=1.0
    ))
    # 补光：右上方
    plotter.add_light(pv.Light(
        position=(center[0] - 300, center[1] + 400, center[2] + 300),
        focal_point=center,
        color='white',
        intensity=0.4
    ))

    # 相机设置：Z 轴俯视，动态计算中心与距离
    bounds = all_verts.ptp(axis=0).max()
    plotter.camera.position = (center[0], center[1], center[2] + bounds * 2.5)
    plotter.camera.focal_point = center
    plotter.camera.view_up = (0, -1, 0)
    plotter.camera.zoom(0.8)

    # 截图
    img = plotter.screenshot()
    plotter.close()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{well_name}_3D.png")
    Image.fromarray(img).save(out_path)
    print(f"  [OK] {well_name}: 已保存 {out_path}")
    return True


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

        success = render_well_pyvista(well_name, label_vol, id_to_cluster, out_dir)
        return f"[OK] {well_name}" if success else f"[WARN] {well_name}: 渲染失败"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[ERR] {well_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='3D cluster rendering with pyvista')
    parser.add_argument('--base-dir', default=BASE, help='Project root directory')
    parser.add_argument('--batch', choices=['0701', '0703', 'all'], default='all')
    parser.add_argument('--well', default=None, help='Process single well only')
    parser.add_argument('--ds', type=int, default=DOWNSAMPLE,
                        help='Downsample factor before marching cubes (1=full res)')
    args = parser.parse_args()

    tasks = []
    batches = BATCHES if args.batch == 'all' else [b for b in BATCHES if b.endswith(args.batch)]

    for batch in batches:
        root = os.path.join(args.base_dir, batch)
        label_dir = os.path.join(root, 'seg_label')
        merge_dir = os.path.join(root, 'cluster_merge')
        out_dir = os.path.join(root, 'cluster_3d_new')

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
    for task in tqdm(tasks, desc="3D rendering"):
        result = process_one_well(*task)
        if "[ERR]" in result or "[WARN]" in result:
            print(result)

    print("\n[Done] 全部完成！输出目录: */cluster_3d/")


if __name__ == "__main__":
    main()
