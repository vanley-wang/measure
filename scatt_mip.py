import os
import numpy as np
import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ================= 配置 =================
N_WORKERS = 8  # 并行进程数
BASE = "Data/nnUNet_FXN_2023"
FOLDERS = {
    '0701': {
        'scatt_mat': os.path.join(BASE, 'FXN_0701/scatt_mat'),
        'seg_mat':   os.path.join(BASE, 'FXN_0701/seg_mat'),
        'output':    os.path.join(BASE, 'FXN_0701/scatt_mip'),
    },
    '0703': {
        'scatt_mat': os.path.join(BASE, 'FXN_0703/scatt_mat'),
        'seg_mat':   os.path.join(BASE, 'FXN_0703/seg_mat'),
        'output':    os.path.join(BASE, 'FXN_0703/scatt_mip'),
    },
}
P_LOW, P_HIGH = 0.0, 99.9  # 用于色阶的分位数范围（同一方向，两个时间点合并后计算）

# ================= I/O 工具 =================
def load_mat_variable(filepath, varname):
    """自动判断 HDF5 / v7 mat 读取指定变量"""
    with open(filepath, 'rb') as f:
        header = f.read(128)
    is_hdf5 = b'HDF5' in header

    if is_hdf5:
        with h5py.File(filepath, 'r') as f:
            if varname not in f:
                raise KeyError(f"{varname} not in {filepath}")
            return np.array(f[varname]).transpose()  # HDF5 通常需要转置
    else:
        mat = loadmat(filepath)
        if varname not in mat:
            raise KeyError(f"{varname} not in {filepath}")
        return np.squeeze(mat[varname])

# ================= 投影函数 =================
def mean_projection_with_mask(data, mask, axis):
    """
    仅在掩膜内做均值投影（axis=0→Z投影，axis=1→Y投影）
    返回：proj(2D)
    """
    m = (mask > 0)
    masked = np.where(m, data, 0)
    counts = np.sum(m, axis=axis)
    sums   = np.sum(masked, axis=axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = np.true_divide(sums, counts)
    proj[counts == 0] = 0  # 没有参与平均的像素置0
    return proj

def save_with_fixed_range(arr2d, save_path, vmin, vmax):
    """用固定 vmin/vmax 保存伪彩图"""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        # 兜底：用图自身 min/max
        vmin, vmax = float(np.nanmin(arr2d)), float(np.nanmax(arr2d))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            print(f"⚠️ 数值异常，跳过保存：{save_path}")
            return False
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, arr2d, cmap='jet', vmin=vmin, vmax=vmax)
    return True

# ================= 单样本处理（同一方向两时间点合并取分位数） =================
def process_sample_pair(args):
    """
    对一个样本（如 C11）：
      1) 读 0701/0703 的 scatt_mat + seg_mat（掩膜变量名 Data_Seg，不旋转）
      2) 计算均值投影：Z、Y（各时间点各一张）
      3) 色阶：
         - Z方向：将 projZ_0701 与 projZ_0703 合并，取 0–99.9% 分位数为 vminZ/vmaxZ
         - Y方向：同理 vminY/vmaxY
      4) 保存到各自 scatt_mip 目录，文件名：{sample}_0701_scatt_Z.png 等
    """
    sample, paths_0701, paths_0703, out_0701, out_0703 = args
    try:
        # 读数据与掩膜
        d1 = load_mat_variable(paths_0701['scatt'], 'data_scatt')
        d3 = load_mat_variable(paths_0703['scatt'], 'data_scatt')
        m1 = load_mat_variable(paths_0701['mask'],  'Data_Seg')  # 不旋转
        m3 = load_mat_variable(paths_0703['mask'],  'Data_Seg')

        if d1.shape != m1.shape or d3.shape != m3.shape:
            print(f"⚠️ 尺寸不匹配，跳过 {sample}: d1{d1.shape}, m1{m1.shape}, d3{d3.shape}, m3{m3.shape}")
            return False

        # 均值投影
        z1 = mean_projection_with_mask(d1, m1, axis=0)
        y1 = mean_projection_with_mask(d1, m1, axis=1)
        z3 = mean_projection_with_mask(d3, m3, axis=0)
        y3 = mean_projection_with_mask(d3, m3, axis=1)

        # 方向内（两时间点合并）分位数色阶
        valsZ = np.concatenate([z1.ravel(), z3.ravel()])
        valsY = np.concatenate([y1.ravel(), y3.ravel()])

        # 去掉 NaN/Inf
        valsZ = valsZ[np.isfinite(valsZ)]
        valsY = valsY[np.isfinite(valsY)]

        if valsZ.size == 0 and valsY.size == 0:
            print(f"⚠️ 样本 {sample} 无有效投影值，跳过")
            return False

        if valsZ.size > 0:
            vminZ = float(np.percentile(valsZ, P_LOW))
            vmaxZ = float(np.percentile(valsZ, P_HIGH))
            if not np.isfinite(vminZ) or not np.isfinite(vmaxZ) or vmaxZ <= vminZ:
                vminZ, vmaxZ = float(valsZ.min()), float(valsZ.max())
        else:
            vminZ = vmaxZ = 0.0

        if valsY.size > 0:
            vminY = float(np.percentile(valsY, P_LOW))
            vmaxY = float(np.percentile(valsY, P_HIGH))
            if not np.isfinite(vminY) or not np.isfinite(vmaxY) or vmaxY <= vminY:
                vminY, vmaxY = float(valsY.min()), float(valsY.max())
        else:
            vminY = vmaxY = 0.0

        # 保存（各时间点用同一方向的统一色阶）
        os.makedirs(out_0701, exist_ok=True)
        os.makedirs(out_0703, exist_ok=True)

        save_with_fixed_range(z1, os.path.join(out_0701, f"{sample}_0701_scatt_Z.png"), vminZ, vmaxZ)
        save_with_fixed_range(y1, os.path.join(out_0701, f"{sample}_0701_scatt_Y.png"), vminY, vmaxY)
        save_with_fixed_range(z3, os.path.join(out_0703, f"{sample}_0703_scatt_Z.png"), vminZ, vmaxZ)
        save_with_fixed_range(y3, os.path.join(out_0703, f"{sample}_0703_scatt_Y.png"), vminY, vmaxY)

        return True

    except Exception as e:
        print(f"❌ 错误 {sample}: {e}")
        return False

# ================= 主程序：收集成对样本并并行处理 =================
def main():
    # 收集样本（只保留 0701 & 0703 都齐的）
    pairs = {}
    for day in ['0701', '0703']:
        scatt_dir = FOLDERS[day]['scatt_mat']
        seg_dir   = FOLDERS[day]['seg_mat']
        files = [f for f in os.listdir(scatt_dir) if f.endswith('_scatt.mat')]
        for f in files:
            prefix = f.replace('_scatt.mat', '')  # e.g. C11_0701
            sample = prefix.split('_')[0]         # C11
            pairs.setdefault(sample, {})[day] = {
                'scatt': os.path.join(scatt_dir, f),
                'mask':  os.path.join(seg_dir,  f'{prefix}.mat'),
            }

    tasks = []
    for sample, d in pairs.items():
        if '0701' in d and '0703' in d:
            tasks.append((
                sample,
                d['0701'],
                d['0703'],
                FOLDERS['0701']['output'],
                FOLDERS['0703']['output'],
            ))

    if not tasks:
        print("⚠️ 未找到成对样本（0701 & 0703）")
        return

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        list(tqdm(ex.map(process_sample_pair, tasks), total=len(tasks), desc="处理样本对"))

if __name__ == "__main__":
    main()
