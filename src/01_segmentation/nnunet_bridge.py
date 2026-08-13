import os
import glob
import nibabel as nib
import numpy as np
from scipy.io import savemat
from scipy import ndimage
from skimage.measure import label
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# ================= 配置区域 =================
# 默认路径，可按实际情况修改
BASE_DIR = r"Data\nnUNet_FXN_2023"

# 两个时间点配置
BATCHES = [
    {
        'date_folder': 'FXN_0701',
        'raw_dir': 'FXN_0701_fixed',
        'pred_dir': 'prediction\\FXN_0701',
    },
    {
        'date_folder': 'FXN_0703',
        'raw_dir': 'FXN_0703_fixed',
        'pred_dir': 'prediction\\FXN_0703',
    }
]

MAX_WORKERS = 4  # Windows 下多进程建议不要太大，若报错改 1


def process_one_well(raw_path, pred_path, out_root, date_suffix):
    """
    处理单个孔位：
    1. 读取原始 OCT → scatt_mat (data_scatt)
    2. 读取 nnUNet 预测 mask → 二值化 → 填充孔洞 → seg_fill (Data_fill)
    3. 对填充后的 mask 做 3D 连通域标记 → seg_label (Data_label)

    维度转换：nnUNet .nii.gz 为 (800, 512, 800)，
    为兼容历史 .mat 格式，保存时 transpose(1,0,2) → (512, 800, 800)
    """
    try:
        # --- 提取孔位名 ---
        raw_name = os.path.basename(raw_path).replace('.nii.gz', '')  # e.g. B2_1_0000
        # 取第一个 '_' 前的部分作为孔位 ID
        well_id = raw_name.split('_')[0]  # e.g. B2
        out_name = f"{well_id}_{date_suffix}"  # e.g. B2_0701

        # --- 1. 读取原始 OCT 强度 ---
        raw_img = nib.load(raw_path)
        raw_data = raw_img.get_fdata().astype(np.float32)

        # 读取预测 mask
        pred_img = nib.load(pred_path)
        pred_data = pred_img.get_fdata()

        # mask 二值化（nnUNet 输出已是 0/1，但以防万一）
        binary_mask = (pred_data >= 0.5).astype(np.uint8)

        # --- 2. 对原始 mask 做 3D 连通域标记（供 measure_from_label.py 计算囊腔） ---
        labeled_raw, _ = label(binary_mask, connectivity=3, return_num=True)
        labeled_raw = labeled_raw.astype(np.uint32)

        # --- 3. 整孔填充 + 再标记（供 scatt.py / roughness.py 使用实心 mask） ---
        filled_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
        labeled_fill, num_features = label(filled_mask, connectivity=3, return_num=True)
        labeled_fill = labeled_fill.astype(np.uint32)

        # --- 4. 维度对齐（兼容历史 .mat）---
        # .nii.gz 维度: (800, 512, 800)
        # 历史 .mat 维度: (512, 800, 800)
        # 转换: transpose(1,0,2) 把前两个轴交换
        raw_mat = np.transpose(raw_data, (1, 0, 2))
        filled_mat = np.transpose(filled_mask, (1, 0, 2))
        labeled_raw_mat = np.transpose(labeled_raw, (1, 0, 2))
        labeled_fill_mat = np.transpose(labeled_fill, (1, 0, 2))

        # --- 5. 保存 .mat 文件 ---
        os.makedirs(os.path.join(out_root, 'seg_fill'), exist_ok=True)
        os.makedirs(os.path.join(out_root, 'seg_label'), exist_ok=True)
        os.makedirs(os.path.join(out_root, 'scatt_mat'), exist_ok=True)

        savemat(os.path.join(out_root, 'seg_fill', f'{out_name}_fill.mat'),
                {'Data_fill': filled_mat})
        savemat(os.path.join(out_root, 'seg_label', f'{out_name}_label.mat'),
                {'Data_label': labeled_fill_mat, 'Data_label_raw': labeled_raw_mat})
        savemat(os.path.join(out_root, 'scatt_mat', f'{out_name}_scatt.mat'),
                {'data_scatt': raw_mat})

        return out_name, num_features, None

    except Exception as e:
        return os.path.basename(raw_path), 0, str(e)


def process_batch(batch_cfg, base_dir):
    """处理一个批次（如 FXN_0701）"""
    date_folder = batch_cfg['date_folder']
    raw_dir = os.path.join(base_dir, batch_cfg['raw_dir'])
    pred_dir = os.path.join(base_dir, batch_cfg['pred_dir'])
    out_root = os.path.join(base_dir, date_folder)
    date_suffix = date_folder.replace('FXN_', '')  # e.g. 0701

    if not os.path.exists(raw_dir):
        print(f"[WARN] Raw dir not exist, skip: {raw_dir}")
        return
    if not os.path.exists(pred_dir):
        print(f"[WARN] Prediction dir not exist, skip: {pred_dir}")
        return

    # 收集原始图文件
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*_0000.nii.gz')))
    if not raw_files:
        print(f"[WARN] No raw images found: {raw_dir}")
        return

    print(f"\n[Batch] {date_folder} | Found {len(raw_files)} raw images")

    tasks = []
    for raw_path in raw_files:
        raw_name = os.path.basename(raw_path)
        # 预测 mask 文件名：去掉 _0000 后缀
        pred_name = raw_name.replace('_0000.nii.gz', '.nii.gz')
        pred_path = os.path.join(pred_dir, pred_name)

        if not os.path.exists(pred_path):
            print(f"  [WARN] Missing prediction mask: {pred_name}, skip")
            continue

        tasks.append((raw_path, pred_path, out_root, date_suffix))

    if not tasks:
        print(f"  [WARN] No matched tasks")
        return

    # 执行
    success = 0
    errors = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one_well, *args): args[0]
            for args in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc=f"{date_suffix} 处理"):
            out_name, num_features, err = future.result()
            if err is None:
                success += 1
                tqdm.write(f"  [OK] {out_name}: {num_features} organoids")
            else:
                errors.append(f"  [ERR] {out_name}: {err}")

    for e in errors:
        print(e)

    print(f"  Done: {success}/{len(tasks)} success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nnUNet .nii.gz → .mat 转换')
    parser.add_argument('--base-dir', default=BASE_DIR, help='项目根目录')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='并行进程数')
    parser.add_argument('--batch', choices=['0701', '0703', 'all'], default='all',
                        help='只处理单个批次，默认 all')
    args = parser.parse_args()

    base_dir = args.base_dir
    MAX_WORKERS = args.workers

    batches_to_run = BATCHES if args.batch == 'all' else [
        b for b in BATCHES if b['date_folder'].endswith(args.batch)
    ]

    for batch in batches_to_run:
        process_batch(batch, base_dir)

    print("\n[Done] All finished!")
