import os
import glob
import numpy as np
import scipy.io as sio
import tifffile
from joblib import Parallel, delayed
from datetime import datetime

# ================= 配置区域 =================
# 根目录 (请修改为您实际的硬盘路径)
# 注意：在 Windows 下路径建议使用 r'' 前缀防止转义
BASE_DIR = r'Data\nnUNet_FXN_2023'

# 需要处理的时间点文件夹
DATE_FOLDERS = ['FXN_0701', 'FXN_0703']

# 输入子文件夹名称
SCATT_INPUT_NAME = 'scatt_mat'
MASK_INPUT_NAME = 'seg_fill'

# 输出子文件夹名称
OUTPUT_FOLDER_NAME = 'scatt_seg'

# 并行核心数 (-1 表示使用所有可用核心, 4 表示使用4个核心)
N_JOBS = 16


# ===========================================

def process_single_sample(scatt_path, mask_dir, output_dir):
    """
    处理单个样本的函数，将被并行调用
    """
    try:
        # 1. 解析文件名和ID
        scatt_filename = os.path.basename(scatt_path)
        # 文件名示例: B2_0701_scatt.mat -> ID: B2_0701
        name_parts = scatt_filename.split('_')
        sample_id = f"{name_parts[0]}_{name_parts[1]}"

        # 构造掩膜路径 (假设掩膜命名为 B2_0701_fill.mat)
        mask_filename = f"{sample_id}_fill.mat"
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            return f"[跳过] {sample_id}: 缺失掩膜文件"

        # 2. 加载数据 (.mat)
        # scipy.io.loadmat 读取后是一个字典
        try:
            scatt_data = sio.loadmat(scatt_path)['data_scatt']
            mask_data = sio.loadmat(mask_path)['Data_fill']
        except KeyError as e:
            return f"[错误] {sample_id}: .mat文件中未找到变量 {e}"
        except Exception as e:
            return f"[错误] {sample_id}: 文件读取失败 - {e}"

        # 3. 数据清洗与尺寸对齐
        # 转换为 float32 节省内存 (MATLAB默认可能是 double)
        vol_scatt = scatt_data.astype(np.float32)
        vol_mask = mask_data.astype(np.float32)

        # 检查尺寸并裁剪 (防止维度不一致报错)
        if vol_scatt.shape != vol_mask.shape:
            min_z = min(vol_scatt.shape[0], vol_mask.shape[0])
            min_y = min(vol_scatt.shape[1], vol_mask.shape[1])
            min_x = min(vol_scatt.shape[2], vol_mask.shape[2])

            vol_scatt = vol_scatt[:min_z, :min_y, :min_x]
            vol_mask = vol_mask[:min_z, :min_y, :min_x]
            # 注意: Python numpy 维度通常是 (Z, Y, X) 或 (X, Y, Z) 取决于保存方式
            # 这里只需保证两者切片一致即可

        # 应用掩膜 (背景置零)
        vol_clean = vol_scatt * vol_mask

        # 4. 保存为 TIFF Stack
        # 注意：tifffile 默认保存顺序通常是 (Z, Y, X)
        # 如果需要在 Amira 中正确显示，可能需要根据原始数据的维度顺序调整 transpose
        # 这里假设原始数据已经是 [Width, Height, Depth] 或 [Depth, Height, Width] 且兼容

        output_filename = os.path.join(output_dir, f"{sample_id}.tif")
        tifffile.imwrite(output_filename, vol_clean, compression='zlib')  # 使用压缩减小体积

        return f"[完成] {sample_id}"

    except Exception as e:
        return f"[异常] {sample_id}: {str(e)}"


def main():
    print(f"开始处理... 时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"并行核心数: {N_JOBS}")

    # 收集所有任务
    tasks = []

    for date_folder in DATE_FOLDERS:
        root_path = os.path.join(BASE_DIR, date_folder)
        scatt_dir = os.path.join(root_path, SCATT_INPUT_NAME)
        mask_dir = os.path.join(root_path, MASK_INPUT_NAME)
        output_dir = os.path.join(root_path, OUTPUT_FOLDER_NAME)

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建目录: {output_dir}")

        # 获取该时间点下的所有散射文件
        search_pattern = os.path.join(scatt_dir, '*_scatt.mat')
        files = glob.glob(search_pattern)

        if not files:
            print(f"[警告] {date_folder} 下未找到数据")
            continue

        # 将任务添加到列表
        for scatt_path in files:
            tasks.append((scatt_path, mask_dir, output_dir))

    print(f"共收集到 {len(tasks)} 个样本任务，开始并行计算...")

    # 执行并行计算
    # backend='loky' 是 joblib 的默认后端，适合这种混合 IO 和 计算 的任务
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_single_sample)(s, m, o) for s, m, o in tasks
    )

    # 打印统计结果
    print("\n" + "=" * 30)
    print("处理报告:")
    print("=" * 30)
    for res in results:
        if "[错误]" in res or "[异常]" in res:
            print(res)  # 只打印错误信息，避免刷屏

    success_count = sum(1 for r in results if "[完成]" in r)
    print(f"\n成功: {success_count} / {len(tasks)}")
    print(f"结束时间: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    # Windows下使用多进程必须放在 if __name__ == '__main__': 之下
    main()