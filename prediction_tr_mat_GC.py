# -*- coding: utf-8 -*-
import os
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import savemat
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
# 假设你的模型文件叫 SAM2UNet.py
from SAM2UNet import SAM2UNet

# ================= 配置区域 (根据显存调整) =================
BATCH_SIZE = 40  # 并行数
NUM_WORKERS = 0  # Windows下设为0
SAVE_PNG = True  # 是否保存中间图片
SAVE_MAT = True  # 是否保存预测结果

# 模型与权重路径
MODEL_PATH = 'models/SAM2-UM-2.pth'
CHECKPOINT_PATH = 'checkpoints/sam2_hiera_large.pt'

# 数据根目录
ROOTS = [
    r"Data/nnUNet_FXN_2023/FXN_0701",
    r"Data/nnUNet_FXN_2023/FXN_0703",
]

# 解除 PIL 像素限制，否则读不进那个巨大的伪装 PNG
Image.MAX_IMAGE_PIXELS = None

# ================= 数据加载器 =================
class OrganoidDataset(Dataset):
    """
    专门针对伪装成 .tr 的大尺寸 PNG 文件的内存数据集
    逻辑：读取 -> 镜像翻转 -> Reshape(800,800,512) -> 旋转270度 -> 归一化
    """

    def __init__(self, tr_path):
        self.slices = []
        self.raw_uint8_images = []  # 用于保存PNG的原始数据

        # --- 1. 读取并预处理 (集成的新逻辑) ---
        # A. 使用 PIL 读取
        img = Image.open(tr_path).convert('L')  # 确保灰度
        pix = np.array(img)

        # B. 镜像翻转 (Fix Mirror)
        pix = np.flip(pix, axis=1)

        # C. Reshape 为 (Z=800, Y=800, X=512)
        # 这样第0维就是 800 层
        volume = pix.reshape(800, 800, 512)

        self.depth = volume.shape[0]  # 应该是 800

        # --- 2. 预处理循环 (切片、旋转、归一化) ---
        # CPU 密集型，初始化时一次性做完存入内存
        for z in range(self.depth):
            # 取出第 z 层: (800, 512)
            raw_slice = volume[z, :, :]

            # D. 旋转: 顺时针 270 度 (即逆时针 90 度)
            # 旋转后形状变为 (512, 800)
            img_rot = cv.rotate(raw_slice, cv.ROTATE_90_COUNTERCLOCKWISE)

            # 保存一份 uint8 用于写 PNG (不归一化)
            self.raw_uint8_images.append(img_rot)

            # E. 归一化 (用于模型输入)
            # 转为 float32
            img_float = img_rot.astype(np.float32)
            img_min, img_max = img_float.min(), img_float.max()

            if img_max > img_min:
                image_norm = (img_float - img_min) / (img_max - img_min)
            else:
                image_norm = np.zeros_like(img_float)

            # 增加维度 (C, H, W) -> (1, 512, 800)
            tensor_input = image_norm[np.newaxis, :, :]
            self.slices.append(tensor_input)

    def __len__(self):
        return self.depth

    def __getitem__(self, idx):
        # 返回: Tensor数据, 原始图片(用于保存), 索引
        return torch.from_numpy(self.slices[idx]).float(), self.raw_uint8_images[idx], idx


# ================= 异步保存函数 =================
def save_batch_images(images_list, indices, save_dir, tr_name):
    """后台线程调用：批量写入PNG"""
    for img, idx in zip(images_list, indices):
        # 文件名: F10_001.png
        save_path = os.path.join(save_dir, f"{tr_name}_{idx + 1:03d}.png")
        cv.imwrite(save_path, img)


# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 初始化模型
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model to {device}...")
    # 需要确保 SAM2UNet 的定义和你的权重文件匹配
    net = SAM2UNet(CHECKPOINT_PATH).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()

    # 建立线程池用于保存图片 (避免IO阻塞GPU)
    executor = ThreadPoolExecutor(max_workers=4)

    for ROOT in ROOTS:
        TR_ROOT = os.path.join(ROOT, 'tr')
        if not os.path.isdir(TR_ROOT):
            continue

        # 准备输出目录
        IMG_ROOT = os.path.join(ROOT, 'image')  # 存放PNG
        SEG_ROOT = os.path.join(ROOT, 'seg_mat_test')  # 存放MAT
        os.makedirs(IMG_ROOT, exist_ok=True)
        os.makedirs(SEG_ROOT, exist_ok=True)

        tr_files = sorted([f for f in os.listdir(TR_ROOT) if f.endswith('.tr')])

        # 遍历文件
        for tr_file in tqdm(tr_files, desc=f"Processing {os.path.basename(ROOT)}"):
            tr_name = os.path.splitext(tr_file)[0]
            tr_path = os.path.join(TR_ROOT, tr_file)

            # 为当前文件创建独立的图片保存子目录
            current_img_save_dir = os.path.join(IMG_ROOT, tr_name)
            if SAVE_PNG:
                os.makedirs(current_img_save_dir, exist_ok=True)

            try:
                # --- Step 1: 加载数据到内存 (这里会执行我们修改后的 Dataset 逻辑) ---
                dataset = OrganoidDataset(tr_path)

                # --- Step 2: 批量推理 (DataLoader加速) ---
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)

                predictions = []  # 用于收集分割结果 (Batch, H, W)

                with torch.no_grad():
                    for batch_data, batch_raw_imgs, batch_indices in loader:
                        # 1. 推理
                        inputs = batch_data.to(device)  # (B, 1, 512, 800)

                        # 前向传播 (根据你的模型返回值调整，这里假设返回3个值)
                        preds, _, _ = net(inputs)
                        preds = torch.softmax(preds, dim=1)

                        # 二分类取前景 (假设前景在通道1)
                        # 输出形状 (B, 512, 800)
                        binary = (preds[:, 1] >= 0.5).float().cpu().numpy()
                        predictions.append(binary)

                        # 2. 异步保存图片 (不阻塞GPU)
                        if SAVE_PNG:
                            # 将 tensor (uint8 list) 转回 numpy list
                            # batch_raw_imgs 是一个 list of tensors (来自collate_fn的默认行为) 或者 list of arrays
                            # 为了保险，转换一下
                            raw_imgs_np = [
                                img.numpy().astype(np.uint8) if isinstance(img, torch.Tensor) else img.astype(np.uint8)
                                for img in batch_raw_imgs]
                            indices_np = batch_indices.numpy()

                            # 扔给后台线程处理
                            executor.submit(save_batch_images, raw_imgs_np, indices_np, current_img_save_dir, tr_name)

                # --- Step 3: 整合结果并保存 .MAT ---
                if SAVE_MAT:
                    # 拼接所有 batch 的结果 -> (800, 512, 800) [Z, H, W]
                    full_volume = np.concatenate(predictions, axis=0)

                    # 转置为 (H, W, Z) 格式以符合 MATLAB/医疗影像 习惯
                    # 原始: (Z=800, H=512, W=800)
                    # 目标: (H=512, W=800, Z=800)
                    # axes=(1, 2, 0)
                    final_mat = np.transpose(full_volume, axes=(1, 2, 0))

                    suffix = os.path.basename(ROOT)[8:]  # 提取日期后缀
                    mat_name = f'{tr_name}_{suffix}.mat'

                    # 保存
                    savemat(os.path.join(SEG_ROOT, mat_name), mdict={'Data_Seg': final_mat.astype(bool)})

            except Exception as e:
                print(f"\n❌ Error handling {tr_name}: {e}")
                import traceback

                traceback.print_exc()

    # 关闭线程池
    executor.shutdown(wait=True)
    print("\n✅ 所有处理完成！")