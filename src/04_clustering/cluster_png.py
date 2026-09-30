"""
Generate direct-viewable four-color cluster PNGs.

The script renders each organoid cluster as a colored 3D surface and saves
one PNG per well into the dataset's cluster_png directory.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.io import loadmat
from skimage.measure import marching_cubes
import pyvista as pv
from tqdm import tqdm


# ================= 配置 =================
DEFAULT_BASE = r"Data/nnUNet_FXN_2023"
BATCHES = ["FXN_0701", "FXN_0703"]
OUTPUT_DIR_NAME = "cluster_png"

# 四种聚类颜色 (RGB 0-255)
# 0=红, 1=黄, 2=绿, 3=蓝
CLUSTER_COLORS = {
    0: np.array([220, 24, 24], dtype=np.float32),
    1: np.array([242, 217, 24], dtype=np.float32),
    2: np.array([0, 214, 42], dtype=np.float32),
    3: np.array([59, 59, 247], dtype=np.float32),
}

SPECULAR = 0.5
SPECULAR_POWER = 15
OUTPUT_SIZE = 3000
DOWNSAMPLE = 2


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
    if 'Index' not in df.columns or 'Cluster' not in df.columns:
        return mapping

    for _, row in df.iterrows():
        oid = extract_id_from_index(row['Index'])
        if oid > 0:
            mapping[oid] = int(row['Cluster'])
    return mapping


def render_well_png(well_name, label_vol, id_to_cluster, out_dir, downsample=DOWNSAMPLE):
    meshes = []

    for cid in range(4):
        oids = [oid for oid, c in id_to_cluster.items() if c == cid]
        if not oids:
            continue

        mask = np.isin(label_vol, oids)
        if not np.any(mask):
            continue

        if downsample > 1:
            mask_proc = ndimage.zoom(mask.astype(np.float32), 1.0 / downsample, order=1)
        else:
            mask_proc = mask.astype(np.float32)

        try:
            verts, faces, _, _ = marching_cubes(mask_proc, level=0.5)
        except ValueError:
            continue

        if downsample > 1:
            verts = verts * downsample

        # marching_cubes gives (z, y, x); pyvista uses (x, y, z)
        verts = verts[:, [2, 1, 0]]
        pv_faces = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
        mesh = pv.PolyData(verts, pv_faces)
        meshes.append((mesh, CLUSTER_COLORS[cid]))

    if not meshes:
        print(f"  [WARN] {well_name}: 无有效 mesh")
        return False

    all_verts = np.vstack([mesh.points for mesh, _ in meshes])
    center = all_verts.mean(axis=0)

    plotter = pv.Plotter(off_screen=True, window_size=[OUTPUT_SIZE, OUTPUT_SIZE])
    plotter.set_background('black')

    for mesh, color in meshes:
        plotter.add_mesh(
            mesh,
            color=color / 255.0,
            show_edges=False,
            smooth_shading=True,
            specular=SPECULAR,
            specular_power=SPECULAR_POWER,
        )

    plotter.remove_all_lights()
    bounds = np.ptp(all_verts, axis=0).max()

    plotter.add_light(pv.Light(
        position=(center[0] + 300, center[1] - 400, center[2] + 500),
        focal_point=center,
        color='white',
        intensity=1.0,
    ))
    plotter.add_light(pv.Light(
        position=(center[0] - 300, center[1] + 400, center[2] + 300),
        focal_point=center,
        color='white',
        intensity=0.4,
    ))

    plotter.camera.position = (center[0], center[1], center[2] + bounds * 2.5)
    plotter.camera.focal_point = center
    plotter.camera.view_up = (0, -1, 0)
    plotter.camera.zoom(0.8)
    plotter.camera.enable_parallel_projection()

    img = plotter.screenshot()
    plotter.close()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{well_name}.png")
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

        ok = render_well_png(well_name, label_vol, id_to_cluster, out_dir)
        return f"[OK] {well_name}" if ok else f"[WARN] {well_name}: 渲染失败"

    except Exception as e:
        return f"[ERR] {well_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='Generate four-color cluster PNGs')
    parser.add_argument('--base-dir', default=DEFAULT_BASE, help='Dataset root directory')
    parser.add_argument('--batch', choices=['0701', '0703', 'all'], default='all')
    args = parser.parse_args()

    base_dir = args.base_dir
    batches = BATCHES if args.batch == 'all' else [b for b in BATCHES if b.endswith(args.batch)]

    tasks = []

    for batch in batches:
        root = os.path.join(base_dir, batch)
        label_dir = os.path.join(root, 'seg_label')

        merge_dir = None
        for candidate in ('cluster_merge_GMM', 'cluster_merge'):
            candidate_dir = os.path.join(root, candidate)
            if os.path.exists(candidate_dir):
                merge_dir = candidate_dir
                break

        if merge_dir is None:
            print(f"[WARN] 跳过 {batch}: 无 cluster_merge 目录")
            continue

        out_dir = os.path.join(base_dir, OUTPUT_DIR_NAME)
        merge_files = [f for f in os.listdir(merge_dir) if f.endswith('_merge.xlsx')]

        for mf in merge_files:
            well_name = mf.replace('_merge.xlsx', '')
            label_path = os.path.join(label_dir, f"{well_name}_label.mat")
            merge_path = os.path.join(merge_dir, mf)

            if not os.path.exists(label_path):
                print(f"  [WARN] 缺失标签文件: {label_path}")
                continue

            tasks.append((well_name, label_path, merge_path, out_dir))

    if not tasks:
        print("[WARN] 未找到任何可处理任务")
        return

    print(f">>> 共 {len(tasks)} 个孔位待处理，开始生成 cluster_png...")
    for task in tqdm(tasks, desc='cluster_png'):
        result = process_one_well(*task)
        if "[ERR]" in result or "[WARN]" in result:
            print(result)

    print(f"\n[Done] 全部完成！输出目录: {os.path.join(base_dir, OUTPUT_DIR_NAME)}")


if __name__ == "__main__":
    main()
