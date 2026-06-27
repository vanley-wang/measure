# -*- coding: utf-8 -*-
import os
import sys
import torch
import cv2 as cv
import numpy as np
from natsort import natsorted
from tqdm import tqdm

# 将模型代码所在目录加入路径
MODEL_DIR = r'models/Parasam2unet_models-maxpool-skloss'
sys.path.insert(0, MODEL_DIR)
from ParaSamCNN2 import ParaSamCNN

# ================= 配置区域 =================
MODEL_PATH = r'models/Parasam2unet_models-maxpool-skloss/epoch_32.pth'
SAM2_CHECKPOINT = r'checkpoints/sam2_hiera_large.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16  # 根据显存调整，RTX 3090 可尝试 16
THRESHOLD = 0.5

# 输入文件夹列表
INPUT_ROOTS = [
    r'Data/FXN/20230701/data/image',
    r'Data/FXN/20230703/data/image',
]

# 输出文件夹后缀（与输入平级）
OUTPUT_SUFFIX = 'seg'
# ===========================================


def load_model():
    """加载 ParaSamCNN 模型和 epoch_32 权重"""
    print(f'Loading model on {DEVICE} ...')
    net = ParaSamCNN(num_classes=1, checkpoint_path=SAM2_CHECKPOINT).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    print('Model loaded successfully.')
    return net


def preprocess_image(img_path):
    """读取并预处理单张图片 -> (1, H, W) numpy float32"""
    image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'无法读取图片: {img_path}')
    image = image.astype(np.float32)
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image)
    return image[np.newaxis, :, :]  # (1, H, W)


def run_inference(net, img_batch):
    """对一批图片进行推理，返回二值 numpy 数组列表"""
    # img_batch: (B, 1, H, W)
    with torch.no_grad():
        pred = net(img_batch)          # (B, 1, H, W)
        pred = torch.sigmoid(pred)     # (B, 1, H, W)
        pred = pred.squeeze(1)         # (B, H, W)
        binary = (pred >= THRESHOLD).cpu().numpy()  # (B, H, W) bool
    return binary


def process_folder(net, input_dir, output_dir):
    """处理单个 well 文件夹内的所有 PNG 图片"""
    os.makedirs(output_dir, exist_ok=True)
    img_files = natsorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    if not img_files:
        print(f'  [跳过] 无PNG文件: {input_dir}')
        return

    # 收集所有预处理后的图片
    images = []
    valid_files = []
    for f in img_files:
        img_path = os.path.join(input_dir, f)
        try:
            img = preprocess_image(img_path)
            images.append(img)
            valid_files.append(f)
        except Exception as e:
            print(f'  [错误] {e}')

    if not images:
        return

    # Batch 推理
    total = len(images)
    pbar = tqdm(total=total, desc=f'  {os.path.basename(input_dir)}', unit='img', leave=False)
    for start_idx in range(0, total, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch_np = np.stack(images[start_idx:end_idx], axis=0)  # (B, 1, H, W)
        batch_tensor = torch.from_numpy(batch_np).float().to(DEVICE)

        binary_batch = run_inference(net, batch_tensor)  # (B, H, W) bool

        # 保存结果
        for i in range(binary_batch.shape[0]):
            out_name = valid_files[start_idx + i]
            out_path = os.path.join(output_dir, out_name)
            mask = (binary_batch[i] * 255).astype(np.uint8)
            cv.imwrite(out_path, mask)

        pbar.update(binary_batch.shape[0])
    pbar.close()


def main():
    net = load_model()

    for input_root in INPUT_ROOTS:
        if not os.path.isdir(input_root):
            print(f'[跳过] 目录不存在: {input_root}')
            continue

        # 输出目录与输入目录平级，例如 .../data/seg
        output_root = os.path.join(os.path.dirname(input_root), OUTPUT_SUFFIX)
        os.makedirs(output_root, exist_ok=True)

        # 获取所有 well 子文件夹
        subdirs = natsorted([d for d in os.listdir(input_root)
                             if os.path.isdir(os.path.join(input_root, d))])
        print(f'\n处理: {input_root}')
        print(f'共 {len(subdirs)} 个子文件夹 -> 输出到: {output_root}')

        for subdir in subdirs:
            input_dir = os.path.join(input_root, subdir)
            output_dir = os.path.join(output_root, subdir)
            process_folder(net, input_dir, output_dir)

    print('\n✅ 所有分割任务完成！')


if __name__ == '__main__':
    main()
