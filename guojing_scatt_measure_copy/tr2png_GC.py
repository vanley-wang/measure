# -*- coding: utf-8 -*-
import os
import cv2 as cv
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# !!! 关键：解除 PIL 像素限制，否则读不进那个巨大的伪装 PNG !!!
Image.MAX_IMAGE_PIXELS = None

# ==================== 路径配置 ====================
ROOTS = [
    r"Data/FXN_2023_new/FXN_20230701",
    r"Data/FXN_2023_new/FXN_20230703",
]


# ==================== 核心处理函数 ====================
def process_tr_file(args):
    """
    单个文件的处理逻辑：
    1. PIL读取 -> 2. 镜像翻转 -> 3. Reshape -> 4. 切片旋转(270度) -> 5. 保存
    """
    tr_path, save_subdir, tr_name = args

    try:
        # 1. 检查是否需要创建子文件夹
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir, exist_ok=True)

        # 2. 使用 PIL 读取并展平数据 (跳过文件头)
        # 转换为灰度 'L' 模式
        with Image.open(tr_path).convert('L') as im:
            pix = np.array(im)

        # 3. 【关键修正】镜像翻转 (Fix Mirror)
        # 新数据的列方向被镜像了，必须翻回来
        pix = np.flip(pix, axis=1)

        # 4. 【关键修正】Reshape 为 (Z=800, Y=800, X=512)
        # 这样第 0 维直接就是深度轴 (800层)
        volume = pix.reshape(800, 800, 512)

        # 获取深度 (应该为 800)
        total_z = volume.shape[0]

        # 5. 循环保存每一层
        for z in range(total_z):
            # 提取切片: 形状应该是 (800, 512)
            raw_slice = volume[z, :, :]

            # 6. 【关键修正】顺时针旋转 270 度 (即逆时针 90 度)
            # 结果形状变为 (512, 800)
            rotate_slice = cv.rotate(raw_slice, cv.ROTATE_90_COUNTERCLOCKWISE)

            # 7. 保存
            # 文件名格式: F10_001.png
            save_name = f"{tr_name}_{z + 1:03d}.png"
            cv.imwrite(os.path.join(save_subdir, save_name), rotate_slice)

        return f"✅ 成功: {tr_name} (共 {total_z} 层)"

    except Exception as e:
        return f"❌ 错误 {tr_name}: {str(e)}"


# ==================== 主程序 ====================
if __name__ == "__main__":
    tasks = []

    print("正在扫描任务...")
    for root in ROOTS:
        tr_dir = os.path.join(root, 'tr')
        img_root_dir = os.path.join(root, 'image')  # 图片总目录

        if not os.path.exists(tr_dir):
            continue

        # 获取所有 .tr 文件
        tr_files = [f for f in os.listdir(tr_dir) if f.endswith('.tr')]

        for tr_file in tr_files:
            tr_name = os.path.splitext(tr_file)[0]
            tr_path = os.path.join(tr_dir, tr_file)

            # 为每个文件创建一个独立的子文件夹
            # 例如: Data/.../image/F10/
            save_subdir = os.path.join(img_root_dir, tr_name)

            tasks.append((tr_path, save_subdir, tr_name))

    print(f"准备处理 {len(tasks)} 个文件，使用多进程加速...")

    # 根据你的电脑性能，max_workers 可以设为 None (自动) 或手动指定 (如 4, 8)
    # 因为涉及大文件读写，进程数不宜过多，以免内存溢出
    with ProcessPoolExecutor(max_workers = 12) as executor:
        futures = [executor.submit(process_tr_file, task) for task in tasks]

        # 进度条
        for f in tqdm(as_completed(futures), total=len(futures), desc="Total Progress"):
            result = f.result()
            # 只打印错误信息，保持控制台清爽
            if "❌" in result:
                print(result)

    print("\n所有转换任务完成！")