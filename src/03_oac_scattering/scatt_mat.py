import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

file_path = r'Data/nnUNet_FXN_2023/FXN_0701/scatt_mat/F5_0701_scatt.mat'
D_METERS = 5.5e-6  # 轴向分辨率 5.5 um

def inverse_physics_model(mu_s_matrix, d_meters):
    """
    第一步：物理反演 (Beer-Lambert Law)
    从 散射系数(mu_s) 得到 相对强度(dB)
    """
    mu = np.nan_to_num(mu_s_matrix, nan=1e-6)
    mu = np.maximum(mu, 1e-6)

    # 积分计算衰减 (Optical Depth)
    optical_depth = np.cumsum(mu, axis=0) * d_meters

    # 双程衰减
    attenuation = np.exp(-2.0 * optical_depth)

    # 线性强度 = 局部反射率 * 衰减
    linear_intensity = mu * attenuation

    # 转为 dB (此时是相对值，通常是负数或很小的值)
    # +1e-15 防止 log(0)
    raw_db = 10 * np.log10(linear_intensity + 1e-15)

    return raw_db

def db_to_gray_exact_inverse(raw_db, floor_p=2, ceil_p=99.8):

    # 1. 确定物理信号的有效范围 (Dynamic Range)
    # 我们认为物理信号中最弱的对应 10dB (底噪)，最强的对应 50dB (饱和)
    vmin = np.percentile(raw_db, floor_p)  # 物理信号底 (对应 10dB)
    vmax = np.percentile(raw_db, ceil_p)  # 物理信号顶 (对应 50dB)

    # 2. 归一化并映射到 [10, 50] dB 空间
    norm = (raw_db - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)  # 截断溢出值
    mapped_db = 10.0 + norm * 40.0
    gray_img = (mapped_db - 10.0) / 40.0 * 255.0

    return gray_img, mapped_db

def main():
    # 1. 加载数据
    if not os.path.exists(file_path):
        print("文件未找到");
        return

    mat = sio.loadmat(file_path)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    data_vol = mat[keys[0]]

    H, W, D = data_vol.shape
    slice_idx = D // 2
    print(f"处理切片: {slice_idx}, 尺寸: {H}x{W}")

    scatt_slice = data_vol[:, :, slice_idx]

    # 2. 物理反演
    raw_physics_db = inverse_physics_model(scatt_slice, D_METERS)

    # 3. 严格灰度映射
    final_gray_img, final_db_map = db_to_gray_exact_inverse(raw_physics_db, floor_p=5, ceil_p=99.5)

    # 4. 绘图验证
    plt.figure(figsize=(12, 6))
    # 左图：最终生成的灰度图 (模拟原始PNG)
    plt.subplot(1, 2, 1)
    plt.imshow(final_gray_img, cmap='gray', vmin=0, vmax=255, aspect='auto')
    plt.title("Simulated Original Image (Gray 0-255)\nFormula: Gray = (dB-10)/40 * 255")
    plt.colorbar(label='Grayscale Value')
    plt.axis('off')
    # 右图：对应的 dB 热力图 (10-50 dB)
    plt.subplot(1, 2, 2)
    plt.imshow(final_db_map, cmap='jet', vmin=10, vmax=50, aspect='auto')
    plt.title("Mapped Intensity (10-50 dB)\nFormula: dB = 10 + (Gray/255)*40")
    plt.colorbar(label='Intensity (dB)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()