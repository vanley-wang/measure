import os
import glob
import numpy as np
import scipy.io as sio
import tifffile
from PIL import Image
from joblib import Parallel, delayed
from natsort import natsorted
from datetime import datetime

# ==================== 全局配置区域 ====================
# 1. 数据根目录 (请修改为您实际的路径)
BASE_DIR = r'Data\nnUNet_FXN_2023'
DATE_FOLDERS = ['FXN_0701', 'FXN_0703']

# 2. 输入文件夹名称
DIR_IMG = 'image'  # 原始强度图 (PNG文件夹)
DIR_SCATT = 'scatt_mat'  # 散射数据 (.mat)
DIR_MASK = 'seg_fill'  # 分割掩膜 (.mat)

# 3. 输出文件夹名称
OUT_IMG = 'seg_amira'  # 输出: 配准后的强度图 TIFF
OUT_SCATT = 'scatt_seg'  # 输出: 配准后的散射图 TIFF

# 4. 配准核心参数 (控制所有数据的几何方向)
# -----------------------------------------------------
TARGET_Z = 512  # 强制将最接近此数值的维度设为 Z轴 (层数)
# ROTATE_MAT = False  # 【关键开关】 True = 将 .mat 数据 (Mask/Scatt) 旋转90度以匹配图片
ROTATE_MAT = True  # 【关键开关】 True = 将 .mat 数据 (Mask/Scatt) 旋转90度以匹配图片

# 如果你在Amira里发现掩膜方向和图片垂直，请改为 True
# -----------------------------------------------------

# 5. 并行核心数 (-1 为全速运行)
N_JOBS = 30


# ======================================================

def read_image_stack(folder_path):
    """读取PNG序列并堆叠为3D矩阵 (默认顺序: Z, H, W)"""
    files = glob.glob(os.path.join(folder_path, '*.png'))
    files = natsorted(files)
    if not files: return None

    img0 = np.array(Image.open(files[0]))
    h, w = img0.shape
    d = len(files)

    # 预分配 float32 矩阵
    volume = np.zeros((d, h, w), dtype=np.float32)
    for i, f in enumerate(files):
        volume[i, :, :] = np.array(Image.open(f))
    return volume


def load_mat_data(path, var_name):
    """安全读取 .mat 文件中的指定变量"""
    try:
        data = sio.loadmat(path)[var_name]
        return data.astype(np.float32)
    except Exception as e:
        return None


def align_volume(vol, source_type):
    """
    统一的几何对齐函数
    source_type: 'Image' (PNG) 或 'Mat' (Mask/Scatt)
    """
    shape = vol.shape

    # --- 1. Z轴统一 (寻找最接近 TARGET_Z 的维度移至 Axis 0) ---
    diffs = [abs(s - TARGET_Z) for s in shape]
    z_idx = np.argmin(diffs)

    if z_idx != 0:
        vol = np.moveaxis(vol, z_idx, 0)
        msg_z = f"维度{z_idx}移至Z"
    else:
        msg_z = "Z轴默认"

    # --- 2. XY平面旋转 (仅针对 .mat 数据响应 ROTATE_MAT 开关) ---
    # 假设 PNG 图片是视觉基准，通常不转。只转 .mat 数据来适配图片。
    msg_xy = "XY直通"
    if source_type == 'Mat' and ROTATE_MAT:
        vol = np.swapaxes(vol, 1, 2)  # 交换 X/Y
        msg_xy = "XY旋转90度"

    return vol, f"[{msg_z}, {msg_xy}]"


def crop_to_intersection(vol_a, vol_b):
    """将两个矩阵裁剪到公共的最小尺寸 (左上对齐)"""
    if vol_a.shape == vol_b.shape:
        return vol_a, vol_b, "尺寸完美"

    min_z = min(vol_a.shape[0], vol_b.shape[0])
    min_y = min(vol_a.shape[1], vol_b.shape[1])
    min_x = min(vol_a.shape[2], vol_b.shape[2])

    # 执行裁剪
    new_a = vol_a[:min_z, :min_y, :min_x]
    new_b = vol_b[:min_z, :min_y, :min_x]

    return new_a, new_b, "已裁剪对齐"


def save_tiff(vol, path, dtype='auto'):
    """保存为压缩的TIFF"""
    if dtype == 'uint':
        if np.max(vol) > 255:
            data = vol.astype(np.uint16)
        else:
            data = vol.astype(np.uint8)
    else:  # float
        data = vol.astype(np.float32)

    tifffile.imwrite(path, data, compression='zlib')


# ==================== 任务处理函数 ====================

def process_intensity_task(folder_path, well_name, date_suffix, mask_dir, out_dir):
    """处理强度图: Image(PNG) + Mask(Mat)"""
    try:
        # 1. 路径检查
        mask_path = os.path.join(mask_dir, f"{well_name}_{date_suffix}_fill.mat")
        if not os.path.exists(mask_path): return f"[跳过] {well_name}: 无掩膜"

        # 2. 读取数据
        vol_img = read_image_stack(folder_path)
        if vol_img is None: return f"[跳过] {well_name}: 无PNG图片"

        vol_mask = load_mat_data(mask_path, 'Data_fill')
        if vol_mask is None: return f"[错误] {well_name}: 掩膜读取失败"

        # 3. 几何对齐 (应用 Z轴和旋转)
        vol_img, msg_i = align_volume(vol_img, 'Image')
        vol_mask, msg_m = align_volume(vol_mask, 'Mat')

        # 4. 尺寸求交集 (防止微小误差)
        vol_img, vol_mask, msg_c = crop_to_intersection(vol_img, vol_mask)

        # 5. 应用掩膜
        vol_out = vol_img * vol_mask

        # 6. 保存 (Intensity保存为整数以兼容灰度显示)
        save_name = f"{well_name}_{date_suffix}.tif"
        save_tiff(vol_out, os.path.join(out_dir, save_name), dtype='uint')

        return f"[强度图 OK] {well_name} | {msg_m} | {msg_c}"

    except Exception as e:
        return f"[强度图 异常] {well_name}: {str(e)}"


def process_scattering_task(scatt_path, mask_dir, out_dir):
    """处理散射图: Scatt(Mat) + Mask(Mat)"""
    try:
        # 1. 解析ID
        filename = os.path.basename(scatt_path)  # B2_0701_scatt.mat
        parts = filename.split('_')
        well_name, date_suffix = parts[0], parts[1]
        sample_id = f"{well_name}_{date_suffix}"

        # 2. 路径检查
        mask_path = os.path.join(mask_dir, f"{sample_id}_fill.mat")
        if not os.path.exists(mask_path): return f"[跳过] {sample_id}: 无掩膜"

        # 3. 读取数据
        vol_scatt = load_mat_data(scatt_path, 'data_scatt')
        if vol_scatt is None: return f"[错误] {sample_id}: 散射数据读取失败"

        vol_mask = load_mat_data(mask_path, 'Data_fill')
        if vol_mask is None: return f"[错误] {sample_id}: 掩膜读取失败"

        # 4. 几何对齐 (注意: Scatt和Mask都是Mat，都受ROTATE_MAT控制)
        vol_scatt, msg_s = align_volume(vol_scatt, 'Mat')
        vol_mask, msg_m = align_volume(vol_mask, 'Mat')

        # 5. 尺寸求交集
        vol_scatt, vol_mask, msg_c = crop_to_intersection(vol_scatt, vol_mask)

        # 6. 应用掩膜
        vol_out = vol_scatt * vol_mask

        # 7. 保存 (Scattering保存为float32以保留物理数值)
        save_name = f"{sample_id}.tif"
        save_tiff(vol_out, os.path.join(out_dir, save_name), dtype='float')

        return f"[散射图 OK] {sample_id} | {msg_s} | {msg_c}"

    except Exception as e:
        return f"[散射图 异常] {os.path.basename(scatt_path)}: {str(e)}"


# ==================== 主程序 ====================

def main():
    print(f"=== 全能配准处理脚本 (Z={TARGET_Z}, 旋转Mat={ROTATE_MAT}) ===")
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")

    all_tasks = []

    # 遍历所有日期文件夹
    for date_folder in DATE_FOLDERS:
        root = os.path.join(BASE_DIR, date_folder)

        # 路径定义
        path_img_in = os.path.join(root, DIR_IMG)
        path_scatt_in = os.path.join(root, DIR_SCATT)
        path_mask_in = os.path.join(root, DIR_MASK)

        path_img_out = os.path.join(root, OUT_IMG)
        path_scatt_out = os.path.join(root, OUT_SCATT)

        # 创建输出目录
        if not os.path.exists(path_img_out): os.makedirs(path_img_out)
        if not os.path.exists(path_scatt_out): os.makedirs(path_scatt_out)

        # 解析日期后缀 (用于匹配文件名)
        try:
            date_suffix = date_folder.split('_')[-1][-4:]  # 取 0701
        except:
            date_suffix = "0000"

        # --- 收集 强度图 处理任务 ---
        if os.path.exists(path_img_in):
            wells = [d for d in os.listdir(path_img_in) if os.path.isdir(os.path.join(path_img_in, d))]
            for w in wells:
                # 任务格式: (类型, 参数...)
                all_tasks.append(
                    ('INTENSITY', os.path.join(path_img_in, w), w, date_suffix, path_mask_in, path_img_out))

        # --- 收集 散射图 处理任务 ---
        if os.path.exists(path_scatt_in):
            files = glob.glob(os.path.join(path_scatt_in, '*_scatt.mat'))
            for f in files:
                all_tasks.append(('SCATTERING', f, path_mask_in, path_scatt_out))

    print(f"共收集任务: {len(all_tasks)} 个 | 并行核心: {N_JOBS}")

    # 执行并行处理
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_intensity_task)(*t[1:]) if t[0] == 'INTENSITY' else
        delayed(process_scattering_task)(*t[1:])
        for t in all_tasks
    )

    # 打印简要报告
    print("\n=== 处理报告 ===")
    err_count = 0
    for res in results:
        if "异常" in res or "错误" in res:
            print(res)
            err_count += 1

    success_count = len(results) - err_count
    print(f"\n成功: {success_count} | 失败: {err_count}")
    print(f"数据已分别保存至: {OUT_IMG} 和 {OUT_SCATT}")


if __name__ == '__main__':
    main()