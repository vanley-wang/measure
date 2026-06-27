import os
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import ball, erosion
from scipy.fftpack import fftn, fftshift, ifftn, ifftshift
from skimage.filters import threshold_otsu
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings

# 引入进度条库
try:
    from tqdm import tqdm
except ImportError:
    print("建议安装 tqdm 以获得更好体验: pip install tqdm")
    tqdm = lambda x, **kwargs: x

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 这里填数据里真实的类别值 (0 和 1)
TARGET_CLUSTERS = [0, 1]


def but_worth_filter3D(img, mode, n, d):
    # (保持原有的滤波函数不变)
    shape = img.shape
    fft3 = fftshift(fftn(img.astype(np.float32)))
    rows, cols, deps = [np.fft.fftfreq(n, 1 / n) for n in shape]
    z, y, x = np.meshgrid(deps, cols, rows, indexing='ij')
    D = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    B = np.sqrt(2) - 1
    if mode == 'low':
        H = 1 / (1 + B * (D / d) ** (2 * n))
    else:
        H = 1 / (1 + B * (d / D) ** (2 * n))
    out_fft = fft3 * H
    cur_psval = np.abs(out_fft) ** 2
    img_out = np.real(ifftn(ifftshift(out_fft)))
    img_out = (img_out - img_out.min()) / (img_out.max() - img_out.min())
    return img_out.astype(np.float32), cur_psval


def roughness_func3D(Data_Fill_Filt, Index_list):
    FF = label(Data_Fill_Filt)
    props = regionprops(FF)
    roughness = []

    for j, name in enumerate(Index_list):
        try:
            parts = str(name).split('_')
            # 假设文件名格式是 B10_0701_1 (MATLAB索引从1开始)
            id = int(parts[2]) - 1

            if id < 0 or id >= len(props):
                roughness.append(0)
                continue

            Index_Temp = props[id].coords
            Data_Fill_ROI = np.zeros_like(Data_Fill_Filt, dtype=np.uint8)
            Data_Fill_ROI[tuple(Index_Temp.T)] = Data_Fill_Filt[tuple(Index_Temp.T)]

            sub_props = regionprops(label(Data_Fill_ROI))
            if not sub_props:
                roughness.append(0)
                continue

            bbox = sub_props[0].bbox
            Region = Data_Fill_ROI[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]

            shape = Region.shape
            obj_size = max(shape)
            objVol = np.zeros((obj_size, obj_size, obj_size), dtype=np.uint8)
            offset = [(obj_size - s) // 2 for s in shape]
            objVol[offset[0]:offset[0] + shape[0], offset[1]:offset[1] + shape[1],
            offset[2]:offset[2] + shape[2]] = Region

            img_src_bin = objVol
            rows = objVol.shape[0]
            src_psval = np.abs(fftshift(fftn(img_src_bin))) ** 2

            eroded = erosion(img_src_bin, ball(3))
            edge = np.logical_xor(img_src_bin, eroded)
            vol_table = regionprops_table(label(edge), properties=["area"])
            sum_val = vol_table["area"][0] if len(vol_table["area"]) > 0 else 1

            n = 2
            vol = []
            psval = []

            for it in range(100):
                d = (it + 1) / 1000.0 * rows
                img_filt, cur_psval = but_worth_filter3D(img_src_bin, mode='low', n=n, d=d)
                img_bin = img_filt > threshold_otsu(img_filt)
                diff = np.logical_xor(img_src_bin, img_bin)
                vol.append(np.sum(diff) / sum_val)
                psval.append(np.sum(cur_psval) / np.sum(src_psval))

            diffs = [abs(p - 0.95) for p in psval[:100]]
            obj_i = np.argmin(diffs)
            roughness.append(round(vol[obj_i], 4))

        except Exception as e:
            roughness.append(0)

    return roughness

def process_one_file(seg_file, seg_path, cluster_path, save_path):
    try:
        name1 = seg_file[:-4]
        name2 = seg_file[:-9]

        mat_path = os.path.join(seg_path, seg_file)
        excel_path = os.path.join(cluster_path, f"{name2}_merge.xlsx")
        save_excel = os.path.join(save_path, f"{name2}_roughness.xlsx")

        # 1. 读取 MAT
        try:
            with h5py.File(mat_path, mode='r') as f:
                Data_fill = f['Data_fill'][:]
                Data_fill = np.transpose(Data_fill, axes=(2, 1, 0)).astype(np.uint8)
        except:
            mat_data = loadmat(mat_path)
            Data_fill = mat_data['Data_fill']

        # 2. 读取 Excel 并筛选
        if not os.path.exists(excel_path):
            return f"SKIP: 找不到分类表 {name2}"

        df_cluster = pd.read_excel(excel_path)

        # 筛选逻辑：只保留 Cluster 为 0 和 1 的
        mask = df_cluster['Cluster'].isin(TARGET_CLUSTERS)
        Index = df_cluster.loc[mask, 'Index'].tolist()
        Cluster = df_cluster.loc[mask, 'Cluster'].tolist()

        if len(Index) == 0:
            return f"SKIP: {name1} 无目标类别"

        # 3. 计算
        Roughness = roughness_func3D(Data_fill, Index)

        # 4. 保存
        df_out = pd.DataFrame({'Index': Index, 'Roughness': Roughness, 'Cluster': Cluster})
        df_out.to_excel(save_excel, index=False)

        return None  # 成功返回 None

    except Exception as e:
        return f"ERROR: {seg_file} -> {e}"


if __name__ == '__main__':
    # 请确保路径正确
    LOOPS = [
        r"E:\student\Private\student11\Measure\Measure_copy\Data\FXN_2023_new\FXN_20230701",
        r"E:\student\Private\student11\Measure\Measure_copy\Data\FXN_2023_new\FXN_20230703"
    ]

    print(f" 开始计算 | 目标数据类别: {TARGET_CLUSTERS} (将在汇总时显示为 {[x + 1 for x in TARGET_CLUSTERS]})")

    for loop in LOOPS:
        seg_path = os.path.join(loop, "seg_fill")
        cluster_path = os.path.join(loop, "cluster_merge")
        save_path = os.path.join(loop, "roughness")
        os.makedirs(save_path, exist_ok=True)

        seg_list = [f for f in os.listdir(seg_path) if f.endswith('.mat')]

        print(f"\n📂 处理文件夹: {os.path.basename(loop)}")
        print(f"📊 文件总数: {len(seg_list)}")

        # 显示进度条
        with ProcessPoolExecutor(max_workers = 8) as executor:
            func = partial(process_one_file, seg_path=seg_path, cluster_path=cluster_path, save_path=save_path)

            # 显示处理进度
            results = list(tqdm(executor.map(func, seg_list), total=len(seg_list), unit="file"))

            # 打印错误或跳过的信息
            for res in results:
                if res: print(res)

    print("\n✅ 所有计算任务完成！")