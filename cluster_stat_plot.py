import os
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ================= 配置 =================
N_WORKERS = 8
BASE = "Data/nnUNet_FXN_2023"
BATCHES = ['FXN_0701', 'FXN_0703']

# 聚类颜色 (与 cluster_mip.py 保持一致)
CLUSTER_COLORS = {
    0: '#FFFF00',  # 黄 - 巨大囊泡型
    1: '#0000FF',  # 蓝 - 中等过渡型
    2: '#00FF00',  # 绿 - 小体积基准型
    3: '#FF0000',  # 红 - 高散射实心型
}

LABEL_DESC = {
    0: '巨大囊泡型',
    1: '中等过渡型',
    2: '小体积基准型',
    3: '高散射实心型',
}


def extract_id_from_index(val):
    """从 Index 字符串提取数字 ID"""
    if isinstance(val, (int, float, np.integer)):
        return int(val)
    if isinstance(val, str):
        match = re.search(r'_(\d+)$', val)
        if match:
            return int(match.group(1))
    return -1


def compute_centroids(label_vol, id_to_cluster):
    """
    计算每个实例的 centroid (z, y, x)。
    返回 dict: {cluster_id: [(z, y, x, volume), ...]}
    """
    coords_by_cluster = {i: [] for i in range(4)}

    for oid, cid in id_to_cluster.items():
        mask = (label_vol == oid)
        if not np.any(mask):
            continue
        vol = int(np.count_nonzero(mask))
        cz, cy, cx = ndimage.center_of_mass(mask)
        coords_by_cluster[cid].append((cz, cy, cx, vol))

    return coords_by_cluster


def process_one_well(args):
    """处理单个孔位：生成统计图"""
    well_name, label_path, merge_path, out_dir = args
    try:
        # 1. 读取聚类表
        df = pd.read_excel(merge_path)
        id_to_cluster = {}
        for _, row in df.iterrows():
            oid = extract_id_from_index(row['Index'])
            if oid > 0:
                id_to_cluster[oid] = int(row['Cluster'])

        if not id_to_cluster:
            return f"⚠️ {well_name}: 无有效 ID 映射"

        # 2. 统计信息
        counts = df['Cluster'].value_counts().to_dict()
        cluster_ids = [0, 1, 2, 3]
        count_list = [counts.get(i, 0) for i in cluster_ids]

        avg_vol = df.groupby('Cluster')['Organoids_Volume_Fill'].mean().to_dict()
        avg_vol_list = [avg_vol.get(i, 0) for i in cluster_ids]

        # 3. 读取标签体积并计算 centroid
        mat = loadmat(label_path)
        if 'Data_label' in mat:
            label_vol = mat['Data_label'].astype(np.int32)
        else:
            keys = [k for k in mat.keys() if not k.startswith('__')]
            label_vol = mat[keys[0]].astype(np.int32)

        coords_by_cluster = compute_centroids(label_vol, id_to_cluster)

        # 4. 绘图
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.2])

        # --- 子图1: 数量饼图 ---
        ax1 = fig.add_subplot(gs[0, 0])
        colors = [CLUSTER_COLORS[i] for i in cluster_ids if count_list[i] > 0]
        labels = [f"{LABEL_DESC[i]}\n({count_list[i]}个)" for i in cluster_ids if count_list[i] > 0]
        sizes = [count_list[i] for i in cluster_ids if count_list[i] > 0]

        if sizes:
            wedges, texts, autotexts = ax1.pie(
                sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 9}
            )
            for t in autotexts:
                t.set_color('white')
                t.set_fontweight('bold')
        ax1.set_title('聚类数量占比', fontsize=12, fontweight='bold')

        # --- 子图2: 平均体积柱状图 ---
        ax2 = fig.add_subplot(gs[0, 1])
        x_pos = np.arange(4)
        bars = ax2.bar(x_pos, avg_vol_list, color=[CLUSTER_COLORS[i] for i in cluster_ids], edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([LABEL_DESC[i] for i in cluster_ids], fontsize=9, rotation=15, ha='right')
        ax2.set_ylabel('平均体积 (voxels)', fontsize=10)
        ax2.set_title('各类平均体积', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        # 在柱顶标数值
        for bar, val in zip(bars, avg_vol_list):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                         f'{val:.0f}', ha='center', va='bottom', fontsize=8)

        # --- 子图3: Z-Y 平面空间分布 ---
        ax3 = fig.add_subplot(gs[1, 0])
        z_max, y_max, x_max = label_vol.shape
        for cid in cluster_ids:
            pts = coords_by_cluster.get(cid, [])
            if pts:
                zs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax3.scatter(ys, zs, c=CLUSTER_COLORS[cid], s=50, alpha=0.7,
                            edgecolors='black', linewidths=0.5, label=LABEL_DESC[cid])
        ax3.set_xlim(0, y_max)
        ax3.set_ylim(z_max, 0)  # Z 轴翻转，顶部在上
        ax3.set_xlabel('Y (pixels)', fontsize=10)
        ax3.set_ylabel('Z (depth)', fontsize=10)
        ax3.set_title('Z-Y 平面空间分布', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_aspect('equal', adjustable='box')

        # --- 子图4: Z-X 平面空间分布 ---
        ax4 = fig.add_subplot(gs[1, 1])
        for cid in cluster_ids:
            pts = coords_by_cluster.get(cid, [])
            if pts:
                zs = [p[0] for p in pts]
                xs = [p[2] for p in pts]
                ax4.scatter(xs, zs, c=CLUSTER_COLORS[cid], s=50, alpha=0.7,
                            edgecolors='black', linewidths=0.5, label=LABEL_DESC[cid])
        ax4.set_xlim(0, x_max)
        ax4.set_ylim(z_max, 0)
        ax4.set_xlabel('X (pixels)', fontsize=10)
        ax4.set_ylabel('Z (depth)', fontsize=10)
        ax4.set_title('Z-X 平面空间分布', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.set_aspect('equal', adjustable='box')

        # --- 子图5: 体积分布箱线图 ---
        ax5 = fig.add_subplot(gs[0:2, 2])
        box_data = []
        box_colors = []
        box_labels = []
        for cid in cluster_ids:
            vols = df[df['Cluster'] == cid]['Organoids_Volume_Fill'].values
            if len(vols) > 0:
                box_data.append(vols)
                box_colors.append(CLUSTER_COLORS[cid])
                box_labels.append(f"{LABEL_DESC[cid]}\n(n={len(vols)})")

        if box_data:
            bp = ax5.boxplot(box_data, labels=box_labels, patch_artist=True,
                             showmeans=True, meanline=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)
            ax5.set_ylabel('体积 (voxels)', fontsize=10)
            ax5.set_title('各类体积分布', fontsize=12, fontweight='bold')
            ax5.grid(axis='y', linestyle='--', alpha=0.5)
            # 对数刻度如果差异大
            if max([max(d) for d in box_data]) / max(1, min([min(d) for d in box_data])) > 100:
                ax5.set_yscale('log')
        else:
            ax5.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax5.transAxes)

        fig.suptitle(f"孔位 {well_name} 聚类统计", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{well_name}_stat.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        return f"✅ {well_name}: 统计图已保存"

    except Exception as e:
        return f"❌ {well_name}: {e}"


def main():
    tasks = []
    for batch in BATCHES:
        root = os.path.join(BASE, batch)
        label_dir = os.path.join(root, 'seg_label')
        merge_dir = os.path.join(root, 'cluster_merge')
        out_dir = os.path.join(root, 'cluster_stat')

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

    print("\n🎉 全部完成！输出目录: */cluster_stat/")
    print("   每孔包含: {well}_stat.png")


if __name__ == "__main__":
    main()
