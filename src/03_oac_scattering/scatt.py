import os
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

def extract_scatt_stats_fast(label_map, scatt_map):
    """
    用 np.bincount 向量化计算每个 label 的散射系数均值和标准差。
    避免 Python 循环，内存占用更小，速度更快。
    """
    assert label_map.shape == scatt_map.shape

    # ravel() 产生视图，不额外复制数据
    flat_label = label_map.ravel()
    flat_scatt = scatt_map.ravel()

    # 只保留前景体素（通常只占 10-20%，大幅减少内存）
    mask = flat_label > 0
    labels = flat_label[mask]
    values = flat_scatt[mask]

    max_label = int(labels.max())

    # bincount 向量化：一次扫描完成所有 label 的求和/计数/平方和
    sum_per_label = np.bincount(labels, weights=values, minlength=max_label + 1)
    count_per_label = np.bincount(labels, minlength=max_label + 1)
    sq_sum_per_label = np.bincount(labels, weights=values ** 2, minlength=max_label + 1)

    # 过滤有效 label
    valid_mask = count_per_label > 0
    valid_labels = np.where(valid_mask)[0]

    counts = count_per_label[valid_mask]
    means = (sum_per_label[valid_mask] / counts).round().astype(int)

    # std = sqrt(E[x^2] - E[x]^2)
    variances = (sq_sum_per_label[valid_mask] / counts) - (means.astype(np.float64) ** 2)
    variances = np.maximum(variances, 0)  # 防止浮点误差导致负数
    stds = np.sqrt(variances).round().astype(int)

    return valid_labels, means, stds

def process_one_sample(seg_label_dir, scatt_mat_dir, output_dir, measure_dir, fname):
    try:
        basename = fname.replace('_label.mat', '')
        label_path = os.path.join(seg_label_dir, fname)
        scatt_path = os.path.join(scatt_mat_dir, f"{basename}_scatt.mat")
        output_path = os.path.join(output_dir, f"{basename}_scatt.xlsx")
        measure_path = os.path.join(measure_dir, f"{basename}.xlsx")

        # 加载标签图和散射系数图
        label_data = loadmat(label_path)['Data_label']
        # scatt_data = loadmat(scatt_path)['Data_scatt']
        scatt_mat = loadmat(scatt_path)
        
        # 兼容不同文件里的 key 大小写差异
        scatt_key = 'Data_scatt' if 'Data_scatt' in scatt_mat else 'data_scatt'
        scatt_data = scatt_mat[scatt_key]

        # 提取统计信息
        labels, means, stds = extract_scatt_stats_fast(label_data, scatt_data)
        index = [f"{basename}_{i+1}" for i in range(len(labels))]

        # 保存单独的散射表格
        df_scatt = pd.DataFrame({
            "Index": index,
            "Scatt_Mean": means,
            "Scatt_STD": stds
        })
        df_scatt.to_excel(output_path, index=False)

        # 合并到原有的量化表格（如果存在）
        if os.path.isfile(measure_path):
            df_measure = pd.read_excel(measure_path)

            # 删除旧的 Scatt_Mean 和 Scatt_STD（若存在）
            for col in ["Scatt_Mean", "Scatt_STD"]:
                if col in df_measure.columns:
                    df_measure.drop(columns=[col], inplace=True)

            df_merged = pd.merge(df_measure, df_scatt, on="Index", how="left")

            # 保存覆盖原表
            df_merged.to_excel(measure_path, index=False)

        return f"✅ 合并完成: {basename}"

    except Exception as e:
        return f"❌ 错误处理 {fname}: {e}"

def process_one_root_folder(root_dir):
    seg_label_dir = os.path.join(root_dir, "seg_label")
    scatt_mat_dir = os.path.join(root_dir, "scatt_mat")
    # output_dir = os.path.join(root_dir, "scatt")
    # measure_dir = os.path.join(root_dir, "measure_excel")
    
    # 输出到根目录下的 scatt/ 和 measure_excel/
    output_dir = os.path.join(root_dir, "scatt")
    measure_dir = os.path.join(root_dir, "measure_excel")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(measure_dir, exist_ok=True)

    mat_files = sorted([f for f in os.listdir(seg_label_dir) if f.endswith('_label.mat')])
    folder_suffix = os.path.basename(root_dir)[-4:]

    print(f"\n📁 正在处理大文件夹: {root_dir}, 共 {len(mat_files)} 个孔位文件夹")

    # 单线程处理（Windows 下多进程加载 mat 文件会导致内存爆炸）
    for fname in tqdm(mat_files, desc=f"{folder_suffix} wells", unit="well"):
        result = process_one_sample(seg_label_dir, scatt_mat_dir, output_dir, measure_dir, fname)
        print(result)

if __name__ == "__main__":
    roots = [
        r"Data\nnUNet_FXN_2023\FXN_0701",
        r"Data\nnUNet_FXN_2023\FXN_0703"
    ]
    print(f"总共 {len(roots)} 个大文件夹；")
    for i, root in enumerate(roots):
        print(f"\n[{i+1}/{len(roots)}] 开始处理大文件夹: {root}")
        start_time = time.time()
        process_one_root_folder(root)
        print(f"✅ 处理完成: {root}, 耗时 {time.time() - start_time:.2f}s\n")
