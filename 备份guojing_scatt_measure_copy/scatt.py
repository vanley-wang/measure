import glob
import os
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.io import loadmat
from skimage.measure import label as cc_label, regionprops_table
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
    """提取每个 label 区域的散射系数统计和空间元数据"""
    assert label_map.shape == scatt_map.shape
    label_ids = np.unique(label_map[label_map > 0]).astype(np.int32)
    if label_ids.size == 0:
        return pd.DataFrame(columns=[
            'Component_ID', 'Voxel_Count',
            'Centroid_Axis0', 'Centroid_Axis1', 'Centroid_Axis2',
            'BBox_Min_Axis0', 'BBox_Min_Axis1', 'BBox_Min_Axis2',
            'BBox_Max_Axis0', 'BBox_Max_Axis1', 'BBox_Max_Axis2',
            'Scatt_Mean', 'Scatt_STD', 'Index'
        ])

    mean_values = np.asarray(ndi.mean(scatt_map, labels=label_map, index=label_ids), dtype=np.float64)
    std_values = np.sqrt(np.maximum(
        np.asarray(ndi.variance(scatt_map, labels=label_map, index=label_ids), dtype=np.float64),
        0.0,
    ))

    props = regionprops_table(
        label_map,
        properties=('label', 'area', 'centroid', 'bbox')
    )
    df_props = pd.DataFrame(props)
    df_props.rename(columns={
        'label': 'Component_ID',
        'area': 'Voxel_Count',
        'centroid-0': 'Centroid_Axis0',
        'centroid-1': 'Centroid_Axis1',
        'centroid-2': 'Centroid_Axis2',
        'bbox-0': 'BBox_Min_Axis0',
        'bbox-1': 'BBox_Min_Axis1',
        'bbox-2': 'BBox_Min_Axis2',
        'bbox-3': 'BBox_Max_Axis0',
        'bbox-4': 'BBox_Max_Axis1',
        'bbox-5': 'BBox_Max_Axis2',
    }, inplace=True)

    df_props['Scatt_Mean'] = np.round(mean_values, 3)
    df_props['Scatt_STD'] = np.round(std_values, 3)
    df_props['Component_ID'] = df_props['Component_ID'].astype(np.int32)
    df_props['Voxel_Count'] = df_props['Voxel_Count'].astype(np.int64)
    df_props.sort_values('Component_ID', inplace=True)
    df_props.reset_index(drop=True, inplace=True)

    return df_props


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
        df_scatt = extract_scatt_stats_fast(label_data, scatt_data)
        if not df_scatt.empty:
            df_scatt.insert(0, "Index", [f"{sample_id}_{int(component_id)}" for component_id in df_scatt["Component_ID"]])
            df_scatt.insert(1, "Sample_ID", sample_id)

        # 保存单独的散射表格
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
    elif os.path.isdir(nii_dir) and os.path.isdir(seg_dir):
        raw_files = [
            path for path in sorted(glob.glob(os.path.join(nii_dir, '*.nii*')))
            if not _stem_without_nii_suffix(path).endswith('-seg')
        ]
        seg_files = sorted(glob.glob(os.path.join(seg_dir, '*.nii*')))

        raw_map = {}
        for path in raw_files:
            sample_base = _stem_without_nii_suffix(path)
            raw_map[sample_base] = path

        seg_map = {}
        for path in seg_files:
            sample_base = _stem_without_nii_suffix(path)
            if sample_base.endswith('-seg'):
                sample_base = sample_base[:-4]
            seg_map[sample_base] = path

        file_items = []
        for sample_base in sorted(raw_map):
            raw_path = raw_map[sample_base]
            seg_path = seg_map.get(sample_base)

            if seg_path is None:
                print(f"⚠️ 缺少分割文件: {sample_base}")
                continue

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
