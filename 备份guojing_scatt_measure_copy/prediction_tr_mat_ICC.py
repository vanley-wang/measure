import os
import torch
import struct
import cv2 as cv
import numpy as np
from tqdm import tqdm
from SAM2UNet import SAM2UNet
from scipy.io import savemat

# 数据格式
x_size = 512
y_size = 800
z_size = 800

# 权重路径
model_path = 'models/SAM2-UM-2.pth'
sam2_checkpoints = 'checkpoints/sam2_hiera_large.pt'

# 初始化模型
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device('cuda')
net = SAM2UNet(sam2_checkpoints).to(device)
net.load_state_dict(torch.load(model_path))
net.eval()

# 数据路径列表
ROOTS = [
    r"E:\student\Private\student11\Measure\Measure_copy\Data\FXN_2023\FXN_20230701",
    r"E:\student\Private\student11\Measure\Measure_copy\Data\FXN_2023\FXN_20230703",
]


for ROOT in ROOTS:
    TR_ROOT = os.path.join(ROOT, 'tr')
    assert os.path.isdir(TR_ROOT), f'TR folder not found: {TR_ROOT}'

    SEG_ROOT = os.path.join(ROOT, 'seg_mat_test')
    os.makedirs(SEG_ROOT, exist_ok=True)

    tr_files = sorted(f for f in os.listdir(TR_ROOT) if f.endswith('.tr'))

    with tqdm(total=len(tr_files), desc=f'[{ROOT}] Processing .tr files', unit='file', dynamic_ncols=True) as pbar:

        for tr_file in tr_files:
            tr_name = tr_file[:-3]
            tr_path = os.path.join(TR_ROOT, tr_file)
            slice = np.zeros(shape=(y_size, x_size), dtype=np.float32)
            res = []

            image_save_dir = os.path.join(ROOT, 'image', tr_name)
            os.makedirs(image_save_dir, exist_ok=True)

            with open(tr_path, 'rb') as f:
                for z in range(z_size):
                    for i in range(y_size):
                        data = f.read(x_size)
                        slice[i, :] = np.frombuffer(data, dtype=np.uint8)

                    image = cv.rotate(slice, cv.ROTATE_90_CLOCKWISE)
                    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

                    # 新增：保存当前层为 PNG
                    image_uint8 = (image_norm * 255).astype(np.uint8)
                    cv.imwrite(os.path.join(image_save_dir, f'{z:04d}.png'), image_uint8)

                    image_input = np.expand_dims(image_norm, axis=(0, 1))  # (1, 1, H, W)
                    img_torch = torch.from_numpy(image_input).to(device)

                    with torch.no_grad():
                        pred, _, _ = net(img_torch)
                        pred = torch.softmax(pred, dim=1)
                        binary = (pred[:, 1] >= 0.5).float()
                        res.append(binary.squeeze().cpu().numpy())

            res = np.array(res)
            res = np.transpose(res, axes=(1, 2, 0))  # (H, W, Z)
            folder_name = os.path.basename(ROOT)
            suffix = folder_name[8:]
            save_name = f'{tr_name}_{suffix}.mat'
            savemat(os.path.join(SEG_ROOT, save_name), mdict={'Data_Seg': res.astype(bool)})

            pbar.update(1)
