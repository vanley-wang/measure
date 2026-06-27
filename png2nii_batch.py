import os
import re
import glob
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

# ==================== 配置区 ====================
INPUT_DIRS = [
    r"E:\student\Private\student13\Measure_copy\Data\FXN\20230701\data\image",
    r"E:\student\Private\student13\Measure_copy\Data\FXN\20230703\data\image",
]

# NIfTI affine 矩阵 (从已有文件提取: diag(-1, -1, 1, 1))
AFFINE = np.array([
    [-1.0,  0.0,  0.0, -0.0],
    [ 0.0, -1.0,  0.0, -0.0],
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
])

# 是否跳过已存在的 nii.gz 文件
SKIP_EXISTING = True
# ================================================


def natural_sort_key(s):
    """自然排序 key，用于正确处理 B2_1.png, B2_2.png, ..., B2_10.png"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]


def convert_png_folder_to_nii(png_folder, out_nii_path):
    """将一个文件夹内的 PNG 切片序列合并为单个 3D NIfTI 文件"""
    png_files = sorted(glob.glob(os.path.join(png_folder, "*.png")),
                       key=lambda x: natural_sort_key(os.path.basename(x)))

    if not png_files:
        print(f"  警告: 在 {png_folder} 中未找到 PNG 文件")
        return False

    # 读取第一张图获取尺寸
    first_img = np.array(Image.open(png_files[0]))
    h, w = first_img.shape  # PIL: (height=512, width=800)
    z = len(png_files)

    # 预分配 volume: 目标 shape (800, 512, 800)
    # 对应关系: nii[:, :, z] == png.T, 即 (w, h)
    volume = np.zeros((w, h, z), dtype=np.uint8)

    for i, png_path in enumerate(tqdm(png_files, desc=f"  Reading {os.path.basename(png_folder)}", leave=False)):
        img = np.array(Image.open(png_path))
        if img.shape != (h, w):
            print(f"  警告: {png_path} 尺寸不匹配 {img.shape} != {(h, w)}")
            continue
        volume[:, :, i] = img.T  # (512, 800) -> (800, 512)

    # 创建 NIfTI 对象
    nifti_img = nib.Nifti1Image(volume, affine=AFFINE)
    nifti_img.header.set_data_dtype(np.uint8)

    # 保存
    nib.save(nifti_img, out_nii_path)
    return True


def process_image_dir(image_dir):
    """处理一个 image 目录下的所有子文件夹"""
    print(f"\n处理目录: {image_dir}")

    subdirs = sorted([d for d in os.listdir(image_dir)
                      if os.path.isdir(os.path.join(image_dir, d))])

    success_count = 0
    skip_count = 0

    for subdir in subdirs:
        png_folder = os.path.join(image_dir, subdir)
        out_nii_name = f"{subdir}_1.nii.gz"
        out_nii_path = os.path.join(image_dir, out_nii_name)

        if SKIP_EXISTING and os.path.exists(out_nii_path):
            print(f"  跳过 (已存在): {out_nii_name}")
            skip_count += 1
            continue

        print(f"  转换: {subdir} -> {out_nii_name}")
        ok = convert_png_folder_to_nii(png_folder, out_nii_path)
        if ok:
            success_count += 1
            print(f"  完成: {out_nii_name}")

    print(f"目录 {image_dir} 处理完毕: 成功 {success_count}, 跳过 {skip_count}, 共 {len(subdirs)} 个")
    return success_count, skip_count


if __name__ == "__main__":
    print("开始批量转换 PNG 切片序列为 NIfTI 格式...")
    total_success = 0
    total_skip = 0

    for image_dir in INPUT_DIRS:
        if os.path.exists(image_dir):
            s, sk = process_image_dir(image_dir)
            total_success += s
            total_skip += sk
        else:
            print(f"警告: 目录不存在 {image_dir}")

    print(f"\n全部完成! 成功转换 {total_success} 个, 跳过 {total_skip} 个")
