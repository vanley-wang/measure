import glob
import os
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage.measure import label as cc_label
from tqdm import tqdm


def _stem_without_nii_suffix(path):
    name = Path(path).name
    if name.endswith('.nii.gz'):
        return name[:-7]
    return Path(path).stem


def _load_mat_volume(mat_path, preferred_keys):
    mat_data = loadmat(mat_path)
    for key in preferred_keys:
        
        if key in mat_data:
            return mat_data[key]
    available = [k for k in mat_data.keys() if not k.startswith('__')]
    raise KeyError(f"{mat_path} 中未找到变量 {preferred_keys}，可用键: {available}")


def load_scatt_volume(scatt_path):
    if scatt_path.lower().endswith(('.nii', '.nii.gz')):
        return nib.load(scatt_path).get_fdata()
    return _load_mat_volume(scatt_path, ['Data_scatt', 'data_scatt'])


def load_label_map(label_path):
    if label_path.lower().endswith(('.nii', '.nii.gz')):
        seg = nib.load(label_path).get_fdata()
        unique_vals = np.unique(seg)

        if unique_vals.size <= 2 and np.all(np.isin(unique_vals, [0, 1])):
            return cc_label(seg > 0.5, connectivity=3).astype(np.int32)

        if np.allclose(seg, np.round(seg)):
            return np.where(seg > 0, np.round(seg), 0).astype(np.int32)

        return cc_label(seg > 0.5, connectivity=3).astype(np.int32)

    return _load_mat_volume(label_path, ['Data_label']).astype(np.int32)

def extract_scatt_stats_fast(label_map, scatt_map):
    """提取每个 label 区域的散射系数均值和标准差"""
    assert label_map.shape == scatt_map.shape
    flat_label = label_map.flatten()
    flat_scatt = scatt_map.flatten()

    mask = flat_label > 0
    labels = flat_label[mask]
    values = flat_scatt[mask]

    unique_labels = np.unique(labels)
    means = []
    stds = []

    for lbl in unique_labels:
        region_vals = values[labels == lbl]
        means.append(int(round(np.mean(region_vals))))
        stds.append(int(round(np.std(region_vals))))

    return unique_labels, means, stds


def build_sample_id(sample_base, date_suffix):
    if re.search(r'_\d{4}$', sample_base):
        return sample_base
    return f"{sample_base}_{date_suffix}"


def process_one_sample(label_path, scatt_path, output_dir, measure_dir, sample_base, date_suffix):
    try:
        sample_id = build_sample_id(sample_base, date_suffix)
        output_path = os.path.join(output_dir, f"{sample_base}_scatt.xlsx")
        measure_path = os.path.join(measure_dir, f"{sample_id}.xlsx")

        # 加载标签图和散射系数图
        label_data = load_label_map(label_path)
        scatt_data = load_scatt_volume(scatt_path)

        # 提取统计信息
        labels, means, stds = extract_scatt_stats_fast(label_data, scatt_data)
        index = [f"{sample_id}_{i+1}" for i in range(len(labels))]

        # 保存单独的散射表格
        df_scatt = pd.DataFrame({
            "Index": index,
            "Scatt_Mean": means,
            "Scatt_STD": stds
        })
        df_scatt.to_excel(output_path, index=False)

        # 合并到原有的量化表格（如果存在）
        if os.path.isfile(measure_path):
            df_measure = pd.read_excel(measure_path)

            # 删除旧的 Scatt_Mean 和 Scatt_STD（若存在）
            for col in ["Scatt_Mean", "Scatt_STD"]:
                if col in df_measure.columns:
                    df_measure.drop(columns=[col], inplace=True)

            df_merged = pd.merge(df_measure, df_scatt, on="Index", how="left")

            # 保存覆盖原表
            df_merged.to_excel(measure_path, index=False)

        return f"✅ 合并完成: {sample_id}"

    except Exception as e:
        return f"❌ 错误处理 {sample_base}: {e}"

def process_one_root_folder(root_dir):
    seg_label_dir = os.path.join(root_dir, "seg_label")
    scatt_mat_dir = os.path.join(root_dir, "scatt_mat")
    nii_dir = os.path.join(root_dir, "nii")
    seg_dir = os.path.join(root_dir, "seg")
    output_dir = os.path.join(root_dir, "scatt")
    measure_dir = os.path.join(root_dir, "measure_excel")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(measure_dir, exist_ok=True)

    folder_suffix = os.path.basename(root_dir)[-4:]

    if os.path.isdir(seg_label_dir) and os.path.isdir(scatt_mat_dir):
        file_items = [
            (
                os.path.join(seg_label_dir, fname),
                os.path.join(scatt_mat_dir, f"{fname.replace('_label.mat', '')}_scatt.mat"),
                fname.replace('_label.mat', '')
            )
            for fname in sorted(os.listdir(seg_label_dir))
            if fname.endswith('_label.mat')
        ]
        mode_name = 'MAT'
    elif os.path.isdir(nii_dir) or os.path.isdir(seg_dir):
        all_nii_files = sorted(glob.glob(os.path.join(root_dir, '**', '*.nii*'), recursive=True))
        paired = {}

        for path in all_nii_files:
            stem = _stem_without_nii_suffix(path)
            if stem.endswith('-seg'):
                sample_base = stem[:-4]
                bucket = paired.setdefault(sample_base, {'raw': [], 'seg': []})
                bucket['seg'].append(path)
            else:
                sample_base = stem
                bucket = paired.setdefault(sample_base, {'raw': [], 'seg': []})
                bucket['raw'].append(path)

        file_items = []
        for sample_base in sorted(paired):
            raw_candidates = paired[sample_base]['raw']
            seg_candidates = paired[sample_base]['seg']

            if not raw_candidates or not seg_candidates:
                print(f"⚠️ 缺少配对文件: {sample_base}")
                continue

            raw_path = raw_candidates[0]
            seg_path = seg_candidates[0]
            file_items.append((seg_path, raw_path, sample_base))
        mode_name = 'NII'
    else:
        print(f"⚠️ 未找到可用输入目录: {root_dir}")
        return

    print(f"\n📁 正在处理大文件夹: {root_dir}, 模式: {mode_name}, 共 {len(file_items)} 个样本")

    for label_path, scatt_path, sample_base in tqdm(file_items, desc=f"{folder_suffix} 样本", unit="sample"):
        result = process_one_sample(label_path, scatt_path, output_dir, measure_dir, sample_base, folder_suffix)
        print(result)

if __name__ == "__main__":
    roots = [
        r"E:\student\Private\student13\Measure_copy\ICC005\20230701",
        r"E:\student\Private\student13\Measure_copy\ICC005\20230703",
    ]
    print(f"总共 {len(roots)} 个大文件夹；")
    for i, root in enumerate(roots):
        print(f"\n[{i+1}/{len(roots)}] 开始处理大文件夹: {root}")
        start_time = time.time()
        process_one_root_folder(root)
        print(f"✅ 处理完成: {root}, 耗时 {time.time() - start_time:.2f}s\n")
