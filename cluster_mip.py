import os
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ================= 配置 =================
N_WORKERS = 8
BASE = "Data/nnUNet_FXN_2023"
BATCHES = ['FXN_0701', 'FXN_0703']

# 聚类颜色映射 (R, G, B)
# 0=巨大囊泡型(黄), 1=中等过渡型(蓝), 2=小体积基准型(绿), 3=高散射实心型(红)
CLUSTER_COLORS = {
    0: np.array([255, 255, 0], dtype=np.float32),   # 黄
    1: np.array([0, 0, 255], dtype=np.float32),     # 蓝
    2: np.array([0, 255, 0], dtype=np.float32),     # 绿
    3: np.array([255, 0, 0], dtype=np.float32),     # 红
}

LABEL_DESC = {
    0: '巨大囊泡型',
    1: '中等过渡型',
    2: '小体积基准型',
    3: '高散射实心型',
}


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


def create_color_mip(label_vol, id_to_cluster, axis=0):
    """
    生成按聚类着色的 MIP 投影图。
    每个 cluster 单独做二值 MIP，再按颜色叠加。
    重叠区域颜色会混合（加法混色）。
    """
    if axis == 0:   # Z 投影 -> (Y, X)
        h, w = label_vol.shape[1], label_vol.shape[2]
    elif axis == 1: # Y 投影 -> (Z, X)
        h, w = label_vol.shape[0], label_vol.shape[2]
    elif axis == 2: # X 投影 -> (Z, Y)
        h, w = label_vol.shape[0], label_vol.shape[1]
    else:
        raise ValueError("axis must be 0, 1 or 2")

    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # 为提速：一次性把 label_vol 映射为 cluster_vol（0=背景, 1~4=cluster+1）
    cluster_vol = np.zeros_like(label_vol, dtype=np.uint8)
    for oid, cid in id_to_cluster.items():
        cluster_vol[label_vol == oid] = cid + 1  # +1 避免和背景 0 冲突

    for cid, color in CLUSTER_COLORS.items():
        mask = (cluster_vol == cid + 1)
        if not np.any(mask):
            continue
        mip = np.max(mask, axis=axis).astype(np.float32)
        for ch in range(3):
            rgb[:, :, ch] += mip * color[ch]

    # 裁剪并转 uint8
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def process_one_well(args):
    """处理单个孔位：生成 Z-MIP 和 Y-MIP"""
    well_name, label_path, merge_path, out_dir = args
    try:
        # 1. 读取标签体积
        mat = loadmat(label_path)
        if 'Data_label' in mat:
            label_vol = mat['Data_label'].astype(np.int32)
        else:
            keys = [k for k in mat.keys() if not k.startswith('__')]
            label_vol = mat[keys[0]].astype(np.int32)

        # 2. 读取聚类表
        df = pd.read_excel(merge_path)
        id_to_cluster = build_id_to_cluster(df)

        if not id_to_cluster:
            return f"⚠️ {well_name}: 无有效 ID 映射"

        # 3. 生成 Z-MIP (axis=0) 和 Y-MIP (axis=1)
        rgb_z = create_color_mip(label_vol, id_to_cluster, axis=0)
        rgb_y = create_color_mip(label_vol, id_to_cluster, axis=1)

        # 4. 保存
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(rgb_z).save(os.path.join(out_dir, f"{well_name}_Z_MIP.png"))
        Image.fromarray(rgb_y).save(os.path.join(out_dir, f"{well_name}_Y_MIP.png"))

        return f"✅ {well_name}: Z/Y MIP 已保存"

    except Exception as e:
        return f"❌ {well_name}: {e}"


def main():
    tasks = []
    for batch in BATCHES:
        root = os.path.join(BASE, batch)
        label_dir = os.path.join(root, 'seg_label')
        merge_dir = os.path.join(root, 'cluster_merge')
        out_dir = os.path.join(root, 'cluster_mip')

        if not os.path.exists(merge_dir):
            print(f"[WARN] 跳过 {batch}: 无 cluster_merge 目录")
            continue

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
        print("⚠️ 未找到任何可处理任务")
        return

    print(f">>> 共 {len(tasks)} 个孔位待处理，启动 {N_WORKERS} 进程...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(tqdm(executor.map(process_one_well, tasks), total=len(tasks)))

    for r in results:
        if r and ("❌" in r or "⚠️" in r):
            print(r)

    print("\n🎉 全部完成！输出目录: */cluster_mip/")
    print("   每孔包含: {well}_Z_MIP.png, {well}_Y_MIP.png")


if __name__ == "__main__":
    main()
