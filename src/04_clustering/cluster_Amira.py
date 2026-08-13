import pandas as pd
import numpy as np
import os
import scipy.io as sio
import tifffile
import re
from multiprocessing import Pool, cpu_count
import time

# ================= 1. 配置区 =================
# 数据的根目录
base_path = r'Data\nnUNet_FXN_2023'

# 需要处理的子文件夹列表
target_folders = [
    'FXN_0701',
    'FXN_0703'
]

# 颜色/材料映射 (Cluster ID -> Amira Material ID)
# 论文标准四类表型 (cluster_category.md):
# 0=红/大囊状健康类器官(Cluster1), 1=黄/大实心健康类器官(Cluster2),
# 2=绿/小实心休眠/幼类器官(Cluster3), 3=蓝/极小高致密受损类器官(Cluster4)
# Amira Material ID 顺序与 Cluster ID 一一对应
CLUSTER_TO_LABEL_MAP = {
    0: 1,   # 红 -> Material 1
    1: 2,   # 黄 -> Material 2
    2: 3,   # 绿 -> Material 3
    3: 4,   # 蓝 -> Material 4
}


# ================= 2. 核心处理函数 (工作单元) =================
def process_single_well_task(args):
    """
    并行任务函数
    args: (root_folder, well_name, output_dir)
    """
    root_folder, well_name, output_dir = args

    # 构建路径
    # 读取 seg_label 下的 _label.mat (这是最原始、准确的 ID 来源)
    mat_path = os.path.join(root_folder, "seg_label", f"{well_name}_label.mat")
    excel_path = os.path.join(root_folder, "cluster_merge", f"{well_name}_merge.xlsx")

    # 检查文件是否存在
    if not os.path.exists(mat_path) or not os.path.exists(excel_path):
        return None  # 静默跳过

    try:
        # 1. 读取数据
        df = pd.read_excel(excel_path)
        mat_content = sio.loadmat(mat_path)

        # 2. 智能获取图像矩阵 (Data_label)
        # 优先查找 'Data_label'，如果没有则取第一个非系统变量
        # 保持原始 dtype (uint32)，避免 astype 额外分配 1.2 GB 内存
        if 'Data_label' in mat_content:
            instance_vol = mat_content['Data_label']
        else:
            keys = [k for k in mat_content.keys() if not k.startswith('__')]
            if not keys: return f"⚠️ {well_name} .mat 文件为空"
            instance_vol = mat_content[keys[0]]

        # 立即释放 mat 字典，减少内存峰值
        mat_content.clear()
        del mat_content

        # 3. 清洗 Excel 中的 Index 列 (关键修复！)
        # 将 "B4_0701_1" 这种字符串转换为数字 1
        def extract_pure_id(val):
            # 如果本身就是数字
            if isinstance(val, (int, float, np.integer)):
                return int(val)
            # 如果是字符串，提取最后下划线后的数字
            if isinstance(val, str):
                match = re.search(r'_(\d+)$', val)
                if match: return int(match.group(1))
            return -1  # 匹配失败

        # 如果有 Index 列就清洗，没有就用行号兜底
        if 'Index' in df.columns:
            df['Clean_ID'] = df['Index'].apply(extract_pure_id)
        else:
            df['Clean_ID'] = df.index + 1

        # 4. 创建 Amira 标签场
        final_label_vol = np.zeros(instance_vol.shape, dtype=np.uint8)
        unique_clusters = sorted(df['Cluster'].unique())

        match_count = 0

        # 遍历每个 Cluster 进行赋值
        for cluster_id in unique_clusters:
            label_val = CLUSTER_TO_LABEL_MAP.get(cluster_id, 255)

            # 找出该 Cluster 包含的所有 ID
            target_ids = df[df['Cluster'] == cluster_id]['Clean_ID'].tolist()

            # 使用 numpy 掩码批量赋值
            # assume_unique=False 安全；uint32 与 Python int 比较无需额外转型
            mask = np.isin(instance_vol, target_ids)
            final_label_vol[mask] = label_val
            del mask  # 立即释放掩码内存，降低峰值

                # 5. 保存结果
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except:
                pass  # 防止多进程并发创建报错

        save_name = f"{well_name}_Unified_Labels.tif"
        save_path = os.path.join(output_dir, save_name)

        tifffile.imwrite(save_path, final_label_vol)

        return f"✅ {well_name}: 处理完成"

    except Exception as e:
        return f"❌ {well_name} 出错: {type(e).__name__}: {e}"


# ================= 3. 主程序 (并行控制器) =================
if __name__ == "__main__":
    start_time = time.time()

    # 1. 收集任务
    all_tasks = []
    print(">>> 正在扫描任务列表...")

    for folder_name in target_folders:
        full_folder_path = os.path.join(base_path, folder_name)
        output_dir = os.path.join(full_folder_path, "cluster_amira")
        merge_dir = os.path.join(full_folder_path, "cluster_merge")

        if not os.path.exists(merge_dir): continue

        files = [f for f in os.listdir(merge_dir) if f.endswith('_merge.xlsx')]

        for f in files:
            well_name = f.replace('_merge.xlsx', '')
            all_tasks.append((full_folder_path, well_name, output_dir))

    total_files = len(all_tasks)
    if total_files == 0:
        print("未找到任何任务！请检查路径。")
        exit()

    # 2. 配置并行池
    # 每个进程峰值内存约 2-3 GB（加载 .mat + 生成掩码 + 输出图像）
    # 限制并发数以避免总内存撑爆；8 核是保守安全值
    max_safe_cores = 8
    num_cores = min(max_safe_cores, cpu_count() - 1)
    num_cores = max(1, num_cores)

    print(f"\n🚀 启动并行加速引擎")
    print(f"   - 检测到 CPU 逻辑核心数: {cpu_count()}")
    print(f"   - 实际使用核心数: {num_cores} (已限制在 Windows 安全范围内)")
    print(f"   - 待处理文件总数: {total_files}")
    print("-" * 50)

    # 3. 开始并行执行
    # 使用 imap_unordered 实现乱序输出，处理完一个打印一个
    with Pool(processes=num_cores) as pool:
        for result in pool.imap_unordered(process_single_well_task, all_tasks):
            if result:
                print(result)

    end_time = time.time()
    duration = end_time - start_time
    print("-" * 50)
    print(f"🎉 全部完成！耗时: {duration:.2f} 秒")
    print(f"📁 结果已保存至各文件夹下的 'cluster_amira' 目录")