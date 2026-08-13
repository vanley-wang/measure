import os
import cv2 as cv
import struct
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 原始图像尺寸
x_size = 512
y_size = 800
z_size = 800

# 多个大文件夹路径，自己替换成实际路径
ROOTS = [
    r"Data\nnUNet_FXN_2023\FXN_0701\data",
    r"Data\nnUNet_FXN_2023\FXN_0703\data"
]

# 1个用法
def process_tr_file(args):
    tr_path, image_path, tr_name = args

    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    # 注意：这里 IDE 显示的 shape: 和 dtype= 是提示，实际代码如下
    slice = np.zeros((y_size, x_size), dtype='uint8')

    with open(tr_path, 'rb') as tr_file:
        for z in range(z_size):
            for i in range(y_size):
                # ！！！这里就是导致错误的根源！！！
                # 二进制直接读取会导致文件头被当成像素读入，产生锯齿
                data = tr_file.read(x_size)
                # 这里的 frombuffer 如果读到的 data 不足 512 字节就会报错 ValueError
                slice[i, :] = np.frombuffer(data, dtype=np.uint8)

            rotate_slice = cv.rotate(slice, cv.ROTATE_90_CLOCKWISE)
            # rotate_slice = np.rot90(rotate_slice, 2)

            save_path = os.path.join(image_path, f"{tr_name}_{z+1}.png")
            cv.imwrite(save_path, rotate_slice)

    return f"{tr_path} Transform Complete!"

if __name__ == "__main__":
    tasks = []

    for ROOT in ROOTS:
        TR_ROOT = os.path.join(ROOT, 'tr')
        IMG_ROOT = os.path.join(ROOT, 'image')
        os.makedirs(IMG_ROOT, exist_ok=True)

        # 过滤文件
        if os.path.exists(TR_ROOT):
            tr_files = [f for f in os.listdir(TR_ROOT) if f.endswith('.tr')]

            for tr in tr_files:
                tr_name = os.path.splitext(tr)[0]
                tr_path = os.path.join(TR_ROOT, tr)
                image_path = os.path.join(IMG_ROOT, tr_name)
                tasks.append((tr_path, image_path, tr_name))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_tr_file, task) for task in tasks]

        # 使用tqdm显示进度条
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing .tr files"):
            print(f.result())