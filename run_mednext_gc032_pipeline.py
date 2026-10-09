"""
MedNext_GC032 完整流水线脚本
将 nnUNet 风格的流水线适配到 MedNext_GC032 数据格式。

数据结构:
    MedNext_GC032/
    ├── 20230701/
    │   ├── nii/       (原始OCT, .nii)
    │   └── seg/       (分割mask, .nii.gz)
    ├── 20230703/
    │   ├── nii/
    │   └── seg/
    ├── ATP.xlsx
    └── cluster_png/

输出:
    MedNext_GC032/20230701/ 和 20230703/ 下的 seg_fill, seg_label, scatt_mat,
    measure_excel, cluster_merge, roughness 等
    MedNext_GC032/MedNext_GC032_Analysis.xlsx
    MedNext_GC032/MedNext_GC032_PCA_Result.xlsx
    MedNext_GC032/Feature_Optimization_Sorted.xlsx
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

# ============================================================
# 配置区
# ============================================================
BASE_DIR = r"MedNext_GC032"
BATCHES = ["20230701", "20230703"]
RAW_SUBDIR = "nii"       # 原始图子目录 (.nii)
SEG_SUBDIR = "seg"       # 分割子目录 (.nii.gz)
MIN_VOLUME = 50          # 最小体积阈值

# ATP 文件路径
ATP_PATH = os.path.join(BASE_DIR, "ATP.xlsx")


# ============================================================
# Step 1: 格式转换 (nii + seg → .mat)
# ============================================================
def step1_nnunet_bridge():
    """
    读取原始 OCT + 分割 mask，生成 seg_fill, seg_label, scatt_mat 三个 .mat 目录。
    适配 MedNext_GC032 的目录结构 (nii/ 和 seg/，无 _0000 后缀)。
    """
    import nibabel as nib
    from scipy.io import savemat
    from scipy import ndimage
    from skimage.measure import label
    from tqdm import tqdm

    print("\n" + "="*60)
    print("Step 1: 格式转换 (.nii/.nii.gz → .mat)")
    print("="*60)

    for batch in BATCHES:
        raw_dir = os.path.join(BASE_DIR, batch, RAW_SUBDIR)
        seg_dir = os.path.join(BASE_DIR, batch, SEG_SUBDIR)
        out_root = os.path.join(BASE_DIR, batch)
        date_suffix = batch[-4:]  # e.g. 0701

        if not os.path.exists(raw_dir):
            print(f"[WARN] Raw dir not exist: {raw_dir}")
            continue
        if not os.path.exists(seg_dir):
            print(f"[WARN] Seg dir not exist: {seg_dir}")
            continue

        # 收集原始图文件 (.nii)
        raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.nii")))
        # 也兼容 .nii.gz
        raw_files += sorted(glob.glob(os.path.join(raw_dir, "*.nii.gz")))

        if not raw_files:
            print(f"[WARN] No raw images in {raw_dir}")
            continue

        print(f"\n[Batch] {batch} | Found {len(raw_files)} raw images")

        os.makedirs(os.path.join(out_root, "seg_fill"), exist_ok=True)
        os.makedirs(os.path.join(out_root, "seg_label"), exist_ok=True)
        os.makedirs(os.path.join(out_root, "scatt_mat"), exist_ok=True)

        success = 0
        for raw_path in tqdm(raw_files, desc=f"{batch} 转换"):
            raw_name = os.path.basename(raw_path)
            # 去掉 .nii 或 .nii.gz 后缀
            if raw_name.endswith(".nii.gz"):
                well_id = raw_name.replace(".nii.gz", "")
            else:
                well_id = raw_name.replace(".nii", "")

            # 查找对应的 seg 文件
            seg_name = well_id + ".nii.gz"
            seg_path = os.path.join(seg_dir, seg_name)
            if not os.path.exists(seg_path):
                # 尝试 .nii
                seg_name2 = well_id + ".nii"
                seg_path = os.path.join(seg_dir, seg_name2)
                if not os.path.exists(seg_path):
                    print(f"  [WARN] Missing seg for {well_id}, skip")
                    continue

            out_name = f"{well_id}_{date_suffix}"

            try:
                # 读取原始 OCT
                raw_img = nib.load(raw_path)
                raw_data = raw_img.get_fdata().astype(np.float32)

                # 读取预测 mask
                pred_img = nib.load(seg_path)
                pred_data = pred_img.get_fdata()

                # 二值化
                binary_mask = (pred_data >= 0.5).astype(np.uint8)

                # 原始 mask 连通域标记
                labeled_raw, _ = label(binary_mask, connectivity=3, return_num=True)
                labeled_raw = labeled_raw.astype(np.uint32)

                # 填充孔洞 + 再标记
                filled_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
                labeled_fill, num_features = label(filled_mask, connectivity=3, return_num=True)
                labeled_fill = labeled_fill.astype(np.uint32)

                # 维度对齐: (800, 512, 800) → (512, 800, 800)  (兼容历史 .mat 格式)
                raw_mat = np.transpose(raw_data, (1, 0, 2))
                filled_mat = np.transpose(filled_mask, (1, 0, 2))
                labeled_raw_mat = np.transpose(labeled_raw, (1, 0, 2))
                labeled_fill_mat = np.transpose(labeled_fill, (1, 0, 2))

                # 保存 .mat
                savemat(os.path.join(out_root, "seg_fill", f"{out_name}_fill.mat"),
                        {"Data_fill": filled_mat})
                savemat(os.path.join(out_root, "seg_label", f"{out_name}_label.mat"),
                        {"Data_label": labeled_fill_mat, "Data_label_raw": labeled_raw_mat})
                savemat(os.path.join(out_root, "scatt_mat", f"{out_name}_scatt.mat"),
                        {"data_scatt": raw_mat})

                success += 1
                tqdm.write(f"  [OK] {out_name}: {num_features} organoids")

            except Exception as e:
                print(f"  [ERR] {well_id}: {e}")

        print(f"  Done: {success}/{len(raw_files)} success")

    print("\nStep 1 完成!")


# ============================================================
# Step 2: 形态学特征提取
# ============================================================
def step2_measure_from_label():
    """从 seg_label/*.mat 提取形态学特征，输出 measure_excel/*.xlsx"""
    from scipy.io import loadmat
    from scipy import ndimage
    from skimage.measure import label as sk_label, marching_cubes, mesh_surface_area
    from tqdm import tqdm

    print("\n" + "="*60)
    print("Step 2: 形态学特征提取")
    print("="*60)

    def compute_surface_area(mask_3d):
        coords = np.where(mask_3d)
        if len(coords[0]) == 0:
            return 0.0
        z_min, z_max = coords[0].min(), coords[0].max() + 1
        y_min, y_max = coords[1].min(), coords[1].max() + 1
        x_min, x_max = coords[2].min(), coords[2].max() + 1
        cropped = mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]
        try:
            verts, faces, _, _ = marching_cubes(cropped, level=0.5)
            return mesh_surface_area(verts, faces)
        except Exception:
            eroded = ndimage.binary_erosion(cropped)
            boundary = cropped & (~eroded)
            return float(np.count_nonzero(boundary))

    def process_one_well(label_path, fill_path, date_suffix):
        rows = []
        try:
            label_mat = loadmat(label_path)
            label_data = label_mat["Data_label"]
            label_raw = label_mat["Data_label_raw"]
            well_id = os.path.basename(label_path).split("_")[0]

            slices = ndimage.find_objects(label_data)
            if not slices:
                return [], None

            for fill_id in range(1, len(slices) + 1):
                sl = slices[fill_id - 1]
                if sl is None:
                    continue

                crop_fill = label_data[sl]
                fill_mask = (crop_fill == fill_id)
                volume_fill = int(np.count_nonzero(fill_mask))

                if volume_fill < MIN_VOLUME:
                    continue

                crop_raw = label_raw[sl]
                overlap = crop_raw[fill_mask]
                overlap = overlap[overlap > 0]
                if len(overlap) == 0:
                    continue
                raw_id = int(np.bincount(overlap).argmax())

                raw_mask = (crop_raw == raw_id)
                volume = int(np.count_nonzero(raw_mask))

                cavity_mask = fill_mask & (~raw_mask)
                cavity_volume = int(np.count_nonzero(cavity_mask))
                _, cavity_num = sk_label(cavity_mask, connectivity=3, return_num=True)

                surface = compute_surface_area(fill_mask)

                long_axis = float(max(
                    sl[0].stop - sl[0].start,
                    sl[1].stop - sl[1].start,
                    sl[2].stop - sl[2].start
                ))
                short_axis = float(min(
                    sl[0].stop - sl[0].start,
                    sl[1].stop - sl[1].start,
                    sl[2].stop - sl[2].start
                ))

                wall_thickness = max(1.0, volume_fill / (surface + 1e-6) * 0.25)
                sphericity = (np.pi ** (1/3)) * ((6 * volume_fill) ** (2/3)) / (surface + 1e-6)
                sphericity = min(sphericity, 5.0)

                rows.append({
                    "Index": f"{well_id}_{date_suffix}_{fill_id}",
                    "Organoids_Volume": volume,
                    "Organoids_Volume_Fill": volume_fill,
                    "Organoids_Surface": round(surface, 2),
                    "Cavity_Volume": cavity_volume,
                    "CavityNum": cavity_num,
                    "LongAxis": long_axis,
                    "ShortAxis": short_axis,
                    "Wall_Thickness": round(wall_thickness, 4),
                    "Sphericity": round(sphericity, 4),
                    "Scatt_Mean": np.nan,
                    "Scatt_STD": np.nan,
                })

        except Exception as e:
            return [], f"[ERR] {os.path.basename(label_path)}: {e}"

        return rows, None

    grand_total = 0
    for batch in BATCHES:
        label_dir = os.path.join(BASE_DIR, batch, "seg_label")
        fill_dir = os.path.join(BASE_DIR, batch, "seg_fill")
        out_dir = os.path.join(BASE_DIR, batch, "measure_excel")
        os.makedirs(out_dir, exist_ok=True)

        date_suffix = batch[-4:]

        label_files = sorted(glob.glob(os.path.join(label_dir, "*_label.mat")))
        if not label_files:
            print(f"[WARN] No label files in {label_dir}")
            continue

        print(f"\n[Batch] {batch} | {len(label_files)} wells")

        total_organs = 0
        for label_path in tqdm(label_files, desc=f"{date_suffix} wells"):
            well_id = os.path.basename(label_path).split("_")[0]
            fill_name = os.path.basename(label_path).replace("_label", "_fill")
            fill_path = os.path.join(fill_dir, fill_name)

            if not os.path.exists(fill_path):
                print(f"  [WARN] Missing fill: {fill_name}, skip")
                continue

            rows, err = process_one_well(label_path, fill_path, date_suffix)
            if err:
                print(err)
                continue

            if rows:
                df = pd.DataFrame(rows)
                out_path = os.path.join(out_dir, f"{well_id}_{date_suffix}.xlsx")
                df.to_excel(out_path, index=False)
                total_organs += len(rows)
                tqdm.write(f"  [OK] {well_id}_{date_suffix}: {len(rows)} organoids")

        print(f"  [Done] {batch}: {total_organs} organoids")
        grand_total += total_organs

    print(f"\nStep 2 完成! 总计 {grand_total} 个类器官")


# ============================================================
# Step 3: 散射统计
# ============================================================
def step3_scatt():
    """计算每个类器官的散射均值/标准差，合并到 measure_excel"""
    from scipy.io import loadmat
    from tqdm import tqdm

    print("\n" + "="*60)
    print("Step 3: 散射统计 (Scatt_Mean / Scatt_STD)")
    print("="*60)

    def extract_scatt_stats_fast(label_map, scatt_map):
        assert label_map.shape == scatt_map.shape
        flat_label = label_map.ravel()
        flat_scatt = scatt_map.ravel()
        mask = flat_label > 0
        labels = flat_label[mask]
        values = flat_scatt[mask]
        max_label = int(labels.max())

        sum_per_label = np.bincount(labels, weights=values, minlength=max_label + 1)
        count_per_label = np.bincount(labels, minlength=max_label + 1)
        sq_sum_per_label = np.bincount(labels, weights=values**2, minlength=max_label + 1)

        valid_mask = count_per_label > 0
        valid_labels = np.where(valid_mask)[0]
        counts = count_per_label[valid_mask]
        means = (sum_per_label[valid_mask] / counts).round().astype(int)
        variances = (sq_sum_per_label[valid_mask] / counts) - (means.astype(np.float64) ** 2)
        variances = np.maximum(variances, 0)
        stds = np.sqrt(variances).round().astype(int)
        return valid_labels, means, stds

    for batch in BATCHES:
        root_dir = os.path.join(BASE_DIR, batch)
        seg_label_dir = os.path.join(root_dir, "seg_label")
        scatt_mat_dir = os.path.join(root_dir, "scatt_mat")
        output_dir = os.path.join(root_dir, "scatt")
        measure_dir = os.path.join(root_dir, "measure_excel")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(measure_dir, exist_ok=True)

        mat_files = sorted([f for f in os.listdir(seg_label_dir) if f.endswith("_label.mat")])
        if not mat_files:
            print(f"[WARN] No label files in {seg_label_dir}")
            continue

        print(f"\n[Batch] {batch} | {len(mat_files)} wells")

        for fname in tqdm(mat_files, desc=f"{batch[-4:]} scatt"):
            try:
                basename = fname.replace("_label.mat", "")
                label_path = os.path.join(seg_label_dir, fname)
                scatt_path = os.path.join(scatt_mat_dir, f"{basename}_scatt.mat")
                output_path = os.path.join(output_dir, f"{basename}_scatt.xlsx")
                measure_path = os.path.join(measure_dir, f"{basename}.xlsx")

                label_data = loadmat(label_path)["Data_label"]
                scatt_mat = loadmat(scatt_path)
                scatt_key = "Data_scatt" if "Data_scatt" in scatt_mat else "data_scatt"
                scatt_data = scatt_mat[scatt_key]

                labels, means, stds = extract_scatt_stats_fast(label_data, scatt_data)
                index = [f"{basename}_{i+1}" for i in range(len(labels))]

                df_scatt = pd.DataFrame({
                    "Index": index,
                    "Scatt_Mean": means,
                    "Scatt_STD": stds
                })
                df_scatt.to_excel(output_path, index=False)

                if os.path.isfile(measure_path):
                    df_measure = pd.read_excel(measure_path)
                    for col in ["Scatt_Mean", "Scatt_STD"]:
                        if col in df_measure.columns:
                            df_measure.drop(columns=[col], inplace=True)
                    df_merged = pd.merge(df_measure, df_scatt, on="Index", how="left")
                    df_merged.to_excel(measure_path, index=False)

                tqdm.write(f"  [OK] {basename}")
            except Exception as e:
                print(f"  [ERR] {fname}: {e}")

    print("\nStep 3 完成!")


# ============================================================
# Step 4: 聚类分型
# ============================================================
def step4_cluster_merge():
    """加载 KMeans 模型，对每个孔做聚类分型"""
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from cluster_utils import load_model_package, PHENOTYPE_NAMES

    print("\n" + "="*60)
    print("Step 4: 聚类分型 (KMeans)")
    print("="*60)

    model_path = "model/Kmeans-scatt.pickle"
    print(f"加载模型: {model_path}")
    model_pkg = load_model_package(model_path)
    print(f"  模型类型: {model_pkg.get('model_type', 'unknown')}")
    print(f"  特征数: {len(model_pkg['feature_names'])}")
    print(f"  特征: {model_pkg['feature_names']}")

    for batch in BATCHES:
        root = os.path.join(BASE_DIR, batch)
        input_dir = os.path.join(root, "measure_excel")
        output_dir = os.path.join(root, "cluster_merge")

        if not os.path.exists(input_dir):
            print(f"[WARN] Skipping {root}: measure_excel not found")
            continue

        os.makedirs(output_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
        print(f"\n[Batch] {batch}: {len(files)} files")

        for fp in files:
            try:
                df = pd.read_excel(fp)
                features = model_pkg["feature_names"]

                if not all(col in df.columns for col in features):
                    missing = [c for c in features if c not in df.columns]
                    print(f"  [WARN] {os.path.basename(fp)} 缺少特征: {missing}")
                    continue

                preprocessor = model_pkg.get("preprocessor")
                scaler = model_pkg.get("scaler")
                # 兼容不同模型的 key 命名: kmeans / gmm / model
                model = None
                for key in ("model", "kmeans", "gmm"):
                    if key in model_pkg:
                        model = model_pkg[key]
                        break
                if model is None:
                    raise ValueError(f"Model package has no model key. Keys: {list(model_pkg.keys())}")
                raw_to_final = model_pkg.get("raw_to_final", {})

                if preprocessor is not None:
                    X = preprocessor.transform(df[features])
                elif scaler is not None:
                    X = scaler.transform(df[features])
                else:
                    raise ValueError("Model package missing preprocessor/scaler")

                raw_labels = model.predict(X)
                if raw_to_final:
                    final_labels = [raw_to_final.get(l, l) for l in raw_labels]
                else:
                    final_labels = raw_labels

                df["Cluster"] = final_labels
                df["Phenotype_Desc"] = [PHENOTYPE_NAMES.get(int(c), "Unknown") for c in df["Cluster"]]

                fname = os.path.basename(fp).replace(".xlsx", "_merge.xlsx")
                out_path = os.path.join(output_dir, fname)
                df.to_excel(out_path, index=False)
                print(f"  [OK] {fname}")

            except Exception as e:
                print(f"  [ERR] {os.path.basename(fp)}: {e}")

    print("\nStep 4 完成!")


# ============================================================
# Step 5: 孔板级汇总 (cluster_analysis_1)
# ============================================================
def step5_cluster_analysis_1():
    """对每个 cluster_merge 文件追加 Sheet2（孔板级汇总统计）"""
    from pathlib import Path

    print("\n" + "="*60)
    print("Step 5: 孔板级汇总 (追加 Sheet2)")
    print("="*60)

    all_roots = []
    for batch in BATCHES:
        cluster_dir = os.path.join(BASE_DIR, batch, "cluster_merge")
        if os.path.exists(cluster_dir):
            for f in sorted(os.listdir(cluster_dir)):
                if f.endswith("_merge.xlsx"):
                    all_roots.append(os.path.join(cluster_dir, f))

    print(f"共 {len(all_roots)} 个文件需要处理")

    for ROOT in all_roots:
        try:
            DF = pd.read_excel(ROOT)
            cluster = np.array(DF[["Cluster"]]).T
            p = Path(ROOT)
            name = p.stem.rsplit(sep="_", maxsplit=1)[0]

            number = []
            volume_fill_avg = []
            surface_avg = []
            cavity_volume_all = []
            long_axis_avg = []
            short_axis_avg = []
            cyst_thick_avg = []
            sphericity_avg = []
            scatt_mean_avg = []
            scatt_std_avg = []

            for i in range(4):
                cluster_temp = np.where(cluster == i)[1]
                temp = DF.loc[cluster_temp]
                number_temp = len(cluster_temp)
                if number_temp > 0:
                    number.append(number_temp)
                    volume_fill_avg.append(np.mean(np.array(temp["Organoids_Volume"])))
                    surface_avg.append(np.mean(np.array(temp["Organoids_Surface"])))
                    cavity_volume_all.append(np.sum(np.array(temp["Cavity_Volume"])))
                    long_axis_avg.append(np.mean(np.array(temp["LongAxis"])))
                    short_axis_avg.append(np.mean(np.array(temp["ShortAxis"])))
                    cyst_thick_avg.append(np.mean(np.array(temp["Wall_Thickness"])))
                    sphericity_avg.append(np.mean(np.array(temp["Sphericity"])))
                    scatt_mean_avg.append(np.mean(np.array(temp["Scatt_Mean"])))
                    scatt_std_avg.append(np.mean(np.array(temp["Scatt_STD"])))
                else:
                    number.append(0)
                    volume_fill_avg.append(0)
                    surface_avg.append(0)
                    cavity_volume_all.append(0)
                    long_axis_avg.append(0)
                    short_axis_avg.append(0)
                    cyst_thick_avg.append(0)
                    sphericity_avg.append(0)
                    scatt_mean_avg.append(0)
                    scatt_std_avg.append(0)

            data = {"Name": name}
            for ci in range(4):
                prefix = f"_{ci+1}"
                data[f"Number{prefix}"] = number[ci]
                data[f"Volume_Fill_Avg{prefix}"] = volume_fill_avg[ci]
                data[f"Surface_Avg{prefix}"] = surface_avg[ci]
                data[f"Cavity_Volume_All{prefix}"] = cavity_volume_all[ci]
                data[f"Long_Axis_Avg{prefix}"] = long_axis_avg[ci]
                data[f"Short_Axis_Avg{prefix}"] = short_axis_avg[ci]
                data[f"Cyst_Thick_Avg{prefix}"] = cyst_thick_avg[ci]
                data[f"Sphericity_Avg{prefix}"] = sphericity_avg[ci]
                data[f"Scatt_Mean_Avg{prefix}"] = scatt_mean_avg[ci]
                data[f"Scatt_STD_Avg{prefix}"] = scatt_std_avg[ci]

            DF_cluster = pd.DataFrame(data, index=[0])
            with pd.ExcelWriter(ROOT, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                DF_cluster.to_excel(writer, sheet_name="Sheet2", index=False)
            print(f"  [OK] {os.path.basename(ROOT)}")
        except Exception as e:
            print(f"  [ERR] {ROOT}: {e}")

    print("\nStep 5 完成!")


# ============================================================
# Step 6: 汇总主表 (cluster_analysis_2)
# ============================================================
def step6_cluster_analysis_2():
    """合并所有 Sheet2 为主表 Analysis.xlsx"""
    print("\n" + "="*60)
    print("Step 6: 汇总主表 (Analysis.xlsx)")
    print("="*60)

    all_data = []
    per_batch_paths = {}

    for batch in BATCHES:
        cluster_dir = os.path.join(BASE_DIR, batch, "cluster_merge")
        if not os.path.exists(cluster_dir):
            print(f"[WARN] 路径不存在: {cluster_dir}")
            continue

        files = [f for f in sorted(os.listdir(cluster_dir)) if f.endswith("_merge.xlsx")]
        batch_data = []

        for file in files:
            file_path = os.path.join(cluster_dir, file)
            try:
                df = pd.read_excel(file_path, sheet_name="Sheet2")
                batch_data.append(df)
            except Exception as e:
                print(f"  [ERR] 无法读取 {file} 的 Sheet2：{e}")

        if batch_data:
            merged_df = pd.concat(batch_data, ignore_index=True)
            save_name = f"{batch}_Analysis.xlsx"
            save_path = os.path.join(BASE_DIR, batch, save_name)
            merged_df.to_excel(save_path, index=False)
            per_batch_paths[batch] = save_path
            print(f"  [OK] {batch}: {save_path} ({len(merged_df)} 行)")
            all_data.append(merged_df)

    # 合并两个批次的主表
    if all_data:
        df_merged = pd.concat(all_data, ignore_index=True)
        master_path = os.path.join(BASE_DIR, "MedNext_GC032_Analysis.xlsx")
        df_merged.to_excel(master_path, index=False)
        print(f"\n[OK] 主表已保存: {master_path} ({len(df_merged)} 行)")
    else:
        print("[WARN] 没有数据可合并")

    print("\nStep 6 完成!")


# ============================================================
# Step 7: 粗糙度计算
# ============================================================
def step7_roughness():
    """计算每个类器官的表面粗糙度 (仅 Cluster 0 和 1)"""
    from scipy.io import loadmat
    from scipy.fftpack import fftn, fftshift, ifftn, ifftshift
    from skimage.measure import label, regionprops
    from tqdm import tqdm

    print("\n" + "="*60)
    print("Step 7: 粗糙度计算")
    print("="*60)

    TARGET_CLUSTERS = [0, 1]

    def but_worth_filter3D(img, mode, n, d):
        shape = img.shape
        fft3 = fftshift(fftn(img.astype(np.float32)))
        rows, cols, deps = [np.fft.fftfreq(n, 1/n) for n in shape]
        z, y, x = np.meshgrid(deps, cols, rows, indexing="ij")
        D = np.sqrt(x**2 + y**2 + z**2)
        B = np.sqrt(2) - 1
        if mode == "low":
            H = 1 / (1 + B * (D / d) ** (2 * n))
        else:
            H = 1 / (1 + B * (d / D) ** (2 * n))
        out_fft = fft3 * H
        cur_psval = np.abs(out_fft) ** 2
        img_out = np.real(ifftn(ifftshift(out_fft)))
        img_out = (img_out - img_out.min()) / (img_out.max() - img_out.min() + 1e-8)
        return img_out.astype(np.float32), cur_psval

    def roughness_func3D(Data_Fill_Filt, Index_list):
        FF = label(Data_Fill_Filt)
        props = regionprops(FF)
        roughness = []
        for prop in props:
            if prop.label not in Index_list:
                continue
            bbox = prop.bbox
            cropped = Data_Fill_Filt[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
            if cropped.size == 0 or cropped.min() == cropped.max():
                roughness.append(0)
                continue
            try:
                _, psval = but_worth_filter3D(cropped.astype(np.float32), "high", 3, 0.1)
                total_power = np.sum(psval)
                if total_power > 0:
                    roughness.append(float(total_power / cropped.size))
                else:
                    roughness.append(0)
            except Exception:
                roughness.append(0)
        return roughness

    for batch in BATCHES:
        root_dir = os.path.join(BASE_DIR, batch)
        seg_fill_dir = os.path.join(root_dir, "seg_fill")
        cluster_dir = os.path.join(root_dir, "cluster_merge")
        out_dir = os.path.join(root_dir, "roughness")
        os.makedirs(out_dir, exist_ok=True)

        cluster_files = sorted([f for f in os.listdir(cluster_dir) if f.endswith("_merge.xlsx")])
        if not cluster_files:
            print(f"[WARN] No cluster files in {cluster_dir}")
            continue

        print(f"\n[Batch] {batch} | {len(cluster_files)} wells")

        for cf in tqdm(cluster_files, desc=f"{batch[-4:]} roughness"):
            try:
                basename = cf.replace("_merge.xlsx", "")
                well_id = basename.split("_")[0]
                cluster_path = os.path.join(cluster_dir, cf)
                fill_path = os.path.join(seg_fill_dir, f"{basename}_fill.mat")

                if not os.path.exists(fill_path):
                    print(f"  [WARN] Missing fill: {basename}_fill.mat")
                    continue

                df = pd.read_excel(cluster_path)
                # 只保留目标 cluster
                target_df = df[df["Cluster"].isin(TARGET_CLUSTERS)].copy()

                if len(target_df) == 0:
                    # 保存空表
                    out_path = os.path.join(out_dir, f"{basename}_roughness.xlsx")
                    pd.DataFrame(columns=["Index", "Cluster", "Roughness"]).to_excel(out_path, index=False)
                    continue

                fill_mat = loadmat(fill_path)["Data_fill"]
                # 需要从 seg_label 获取每个 organoid 的 mask
                label_path = os.path.join(root_dir, "seg_label", f"{basename}_label.mat")
                if not os.path.exists(label_path):
                    print(f"  [WARN] Missing label: {basename}_label.mat")
                    continue

                label_data = loadmat(label_path)["Data_label"]

                roughness_results = []
                for _, row in target_df.iterrows():
                    idx = row["Index"]
                    # Index 格式: C3_0701_1 → 最后一个数字是 label id
                    label_id = int(idx.rsplit("_", 1)[-1])
                    organoid_mask = (label_data == label_id).astype(np.uint8)

                    if organoid_mask.sum() < MIN_VOLUME:
                        roughness_results.append(0)
                        continue

                    try:
                        r = roughness_func3D(organoid_mask, [1])
                        roughness_results.append(r[0] if r else 0)
                    except Exception:
                        roughness_results.append(0)

                target_df["Roughness"] = roughness_results
                out_path = os.path.join(out_dir, f"{basename}_roughness.xlsx")
                target_df[["Index", "Cluster", "Roughness"]].to_excel(out_path, index=False)
                tqdm.write(f"  [OK] {basename}: {len(target_df)} organoids")

            except Exception as e:
                print(f"  [ERR] {cf}: {e}")

    print("\nStep 7 完成!")


# ============================================================
# Step 8: 粗糙度汇总到主表
# ============================================================
def step8_roughness_analysis():
    """汇总粗糙度到 Analysis 主表"""
    print("\n" + "="*60)
    print("Step 8: 粗糙度汇总到主表")
    print("="*60)

    master_path = os.path.join(BASE_DIR, "MedNext_GC032_Analysis.xlsx")
    if not os.path.exists(master_path):
        print(f"[ERR] 主表不存在: {master_path}")
        return

    df_master = pd.read_excel(master_path)
    print(f"主表行数: {len(df_master)}")

    roughness_all_list = []
    roughness_avg_1_list = []
    roughness_avg_2_list = []

    for _, row in df_master.iterrows():
        name = row["Name"]
        well_id = name.rsplit("_", 1)[0]
        date_suffix = name.rsplit("_", 1)[1]
        # 找到批次
        batch = None
        for b in BATCHES:
            if b.endswith(date_suffix):
                batch = b
                break

        if batch is None:
            roughness_all_list.append(np.nan)
            roughness_avg_1_list.append(np.nan)
            roughness_avg_2_list.append(np.nan)
            continue

        roughness_dir = os.path.join(BASE_DIR, batch, "roughness")
        rough_file = os.path.join(roughness_dir, f"{name}_roughness.xlsx")

        if not os.path.exists(rough_file):
            roughness_all_list.append(0)
            roughness_avg_1_list.append(0)
            roughness_avg_2_list.append(0)
            continue

        try:
            df_rough = pd.read_excel(rough_file)
            if len(df_rough) == 0:
                roughness_all_list.append(0)
                roughness_avg_1_list.append(0)
                roughness_avg_2_list.append(0)
                continue

            roughness_all_list.append(df_rough["Roughness"].sum())
            c1 = df_rough[df_rough["Cluster"] == 0]
            c2 = df_rough[df_rough["Cluster"] == 1]
            roughness_avg_1_list.append(c1["Roughness"].mean() if len(c1) > 0 else 0)
            roughness_avg_2_list.append(c2["Roughness"].mean() if len(c2) > 0 else 0)
        except Exception:
            roughness_all_list.append(0)
            roughness_avg_1_list.append(0)
            roughness_avg_2_list.append(0)

    df_master["Roughness_All"] = roughness_all_list
    df_master["Roughness_Avg_1"] = roughness_avg_1_list
    df_master["Roughness_Avg_2"] = roughness_avg_2_list

    df_master.to_excel(master_path, index=False)
    print(f"[OK] 粗糙度已汇总到: {master_path}")
    print("\nStep 8 完成!")


# ============================================================
# Step 9: PCA 综合评分
# ============================================================
def step9_pca():
    """PCA 降维，输出 PCA_Model_Details 和 PCA_Result"""
    from sklearn.decomposition import PCA as SKPCA
    from sklearn.preprocessing import StandardScaler
    import re

    print("\n" + "="*60)
    print("Step 9: PCA 综合评分")
    print("="*60)

    input_path = os.path.join(BASE_DIR, "MedNext_GC032_Analysis.xlsx")
    details_path = os.path.join(BASE_DIR, "MedNext_GC032_PCA_Model_Details.xlsx")
    result_path = os.path.join(BASE_DIR, "MedNext_GC032_PCA_Result.xlsx")

    if not os.path.exists(input_path):
        print(f"[ERR] 输入文件不存在: {input_path}")
        return

    print(f"读取数据: {input_path}")
    Data_All = pd.read_excel(input_path)
    print(f"  数据形状: {Data_All.shape}")

    features_list = [
        'Cavity_Volume_All_1', 'Cyst_Thick_Avg_3', 'Long_Axis_Avg_4', 'Number_2', 'Number_3',
        'Roughness_All',
        'Scatt_Mean_Avg_3',
        'Scatt_Mean_Avg_4',
        'Short_Axis_Avg_2',
        'Short_Axis_Avg_3',
        'Short_Axis_Avg_4',
        'Surface_Avg_1', 'Surface_Avg_2', 'Surface_Avg_3', 'Volume_Fill_Avg_2'
    ]

    def get_feature_sort_key(feature_name):
        if 'Roughness' in feature_name:
            return (1, 999, feature_name)
        match = re.search(r'_(\d+)$', feature_name)
        if match:
            return (int(match.group(1)), 0, feature_name)
        return (99, 0, feature_name)

    features_list.sort(key=get_feature_sort_key)

    existing_cols = [c for c in features_list if c in Data_All.columns]
    missing = [c for c in features_list if c not in Data_All.columns]
    print(f"  可用特征: {len(existing_cols)} / {len(features_list)}")
    if missing:
        print(f"  缺失特征: {missing}")

    Data = Data_All[existing_cols].fillna(0)

    # 标准化
    scaler = StandardScaler()
    Data_std = scaler.fit_transform(Data)

    # PCA
    n_components = min(4, len(existing_cols))
    pca = SKPCA(n_components=n_components, svd_solver="full", random_state=42)
    Data_Pca = pca.fit_transform(Data_std)

    variance_ratio = pca.explained_variance_ratio_
    weights = variance_ratio / np.sum(variance_ratio)

    # --- 准备输出 ---
    group_data = []
    current_group = None
    g_idx = 0
    for feat in existing_cols:
        key = get_feature_sort_key(feat)
        group_num = key[0]
        if group_num != current_group:
            current_group = group_num
            g_idx = 1
        else:
            g_idx += 1
        group_data.append({
            "Feature_Name": feat,
            "Group_ID": group_num,
            "Symbol_LaTeX": f"X_{{{group_num}{g_idx}}}",
            "Symbol_Simple": f"X_{group_num}{g_idx}"
        })
    df_groups = pd.DataFrame(group_data)

    pc_names = [f"PC{i+1}" for i in range(n_components)]
    df_eigen = pd.DataFrame({
        "Principal_Component": pc_names,
        "Eigenvalue": pca.explained_variance_,
        "Variance_Ratio": pca.explained_variance_ratio_,
        "Cumulative_Ratio": np.cumsum(pca.explained_variance_ratio_),
        "Weight_in_Score": weights
    })

    df_loadings = pd.DataFrame(
        pca.components_.T,
        index=existing_cols,
        columns=[f"Loading_{pc}" for pc in pc_names]
    )
    df_loadings = pd.concat([df_groups.set_index("Feature_Name")[["Symbol_Simple"]], df_loadings], axis=1)

    final_coefs = np.dot(pca.components_.T, weights)
    df_final_coef = pd.DataFrame({
        "Feature_Name": existing_cols,
        "Symbol": df_groups["Symbol_Simple"].values,
        "Coefficient": final_coefs,
        "Abs_Coefficient": np.abs(final_coefs)
    }).sort_values(by="Abs_Coefficient", ascending=False)

    # 保存 PCA Model Details
    with pd.ExcelWriter(details_path, engine="openpyxl") as writer:
        df_eigen.to_excel(writer, sheet_name="1_Eigenvalues", index=False, float_format="%.4f")
        df_loadings.to_excel(writer, sheet_name="2_Loadings_Matrix", float_format="%.4f")
        df_final_coef.to_excel(writer, sheet_name="3_Final_Coefficients", index=False, float_format="%.4f")
        df_groups.to_excel(writer, sheet_name="4_Feature_Meta", index=False)

    print(f"  PCA Model Details → {details_path}")

    # --- 计算每个样本的 PCA Score ---
    pc_scores = pd.DataFrame(Data_Pca, columns=pc_names)
    result_score = np.dot(Data_Pca, weights)

    df_result = pd.DataFrame({
        "Name": Data_All["Name"].values if "Name" in Data_All.columns else range(len(Data_All)),
    })
    for pc in pc_names:
        df_result[pc] = pc_scores[pc].values
    df_result["Result"] = result_score

    df_result.to_excel(result_path, index=False)
    print(f"  PCA Result → {result_path}")

    print("\nStep 9 完成!")


# ============================================================
# Step 10: 特征优化搜索 (Feature_Optimization_Sorted)
# ============================================================
def step10_feature_optimization():
    """随机搜索最佳特征组合，生成 Feature_Optimization_Sorted.xlsx"""
    from sklearn.decomposition import PCA as SKPCA
    from sklearn.preprocessing import StandardScaler
    from joblib import Parallel, delayed
    import random
    import time

    print("\n" + "="*60)
    print("Step 10: 特征优化搜索 (Feature_Optimization_Sorted)")
    print("="*60)

    input_path = os.path.join(BASE_DIR, "MedNext_GC032_Analysis.xlsx")
    output_path = os.path.join(BASE_DIR, "Feature_Optimization_Sorted.xlsx")

    if not os.path.exists(input_path):
        print(f"[ERR] 输入文件不存在: {input_path}")
        return

    # 加载 ATP 数据
    atp_dict = {}
    if os.path.exists(ATP_PATH):
        df_atp = pd.read_excel(ATP_PATH)
        for _, row in df_atp.iterrows():
            name = str(row.iloc[0]).strip()
            atp_val = row.iloc[2] if row.shape[0] > 2 else None
            try:
                atp_dict[name] = float(atp_val)
            except (ValueError, TypeError):
                pass
    print(f"  ATP 数据条目数: {len(atp_dict)}")

    Data_All = pd.read_excel(input_path)
    if "Name" not in Data_All.columns:
        Data_All["Name"] = Data_All.index.astype(str)

    # 候选特征池
    full_feature_pool = [
        'Number_1', 'Volume_Fill_Avg_1', 'Surface_Avg_1', 'Cavity_Volume_All_1',
        'Long_Axis_Avg_1', 'Short_Axis_Avg_1', 'Cyst_Thick_Avg_1', 'Sphericity_Avg_1',
        'Roughness_All', 'Scatt_Mean_Avg_1', 'Scatt_STD_Avg_1',
        'Number_2', 'Volume_Fill_Avg_2', 'Surface_Avg_2', 'Cavity_Volume_All_2',
        'Long_Axis_Avg_2', 'Short_Axis_Avg_2', 'Cyst_Thick_Avg_2', 'Sphericity_Avg_2',
        'Scatt_Mean_Avg_2', 'Scatt_STD_Avg_2',
        'Number_3', 'Volume_Fill_Avg_3', 'Surface_Avg_3', 'Cavity_Volume_All_3',
        'Long_Axis_Avg_3', 'Short_Axis_Avg_3', 'Cyst_Thick_Avg_3', 'Sphericity_Avg_3',
        'Scatt_Mean_Avg_3', 'Scatt_STD_Avg_3',
        'Number_4', 'Volume_Fill_Avg_4', 'Surface_Avg_4',
        'Long_Axis_Avg_4', 'Short_Axis_Avg_4', 'Cyst_Thick_Avg_4', 'Sphericity_Avg_4',
        'Scatt_Mean_Avg_4', 'Scatt_STD_Avg_4',
    ]

    available_pool = [f for f in full_feature_pool if f in Data_All.columns]
    available_pool.sort()
    print(f"  有效候选特征数: {len(available_pool)}")

    X_matrix_all = Data_All[available_pool].fillna(0).values

    # 准备时间点对 + ATP
    Data_All["Well_ID"] = Data_All["Name"].apply(lambda x: x.split("_")[0])
    Data_All["TimePoint"] = Data_All["Name"].apply(lambda x: x.split("_")[-1] if "_" in str(x) else "")

    start_indices = []
    end_indices = []
    valid_atp = []

    tp1 = "0701"  # Day3
    tp2 = "0703"  # Day5

    for well in Data_All["Well_ID"].unique():
        row_start = Data_All[(Data_All["Well_ID"] == well) & (Data_All["TimePoint"].str.contains(tp1[-4:]))]
        row_end = Data_All[(Data_All["Well_ID"] == well) & (Data_All["TimePoint"].str.contains(tp2[-4:]))]

        if not row_start.empty and not row_end.empty:
            # 用终点 ATP 值
            end_name = str(row_end.iloc[0]["Name"]).strip()
            start_name = str(row_start.iloc[0]["Name"]).strip()
            atp = atp_dict.get(end_name, atp_dict.get(start_name, None))
            if atp is not None and pd.notna(atp):
                start_indices.append(Data_All.index.get_loc(row_start.index[0]))
                end_indices.append(Data_All.index.get_loc(row_end.index[0]))
                valid_atp.append(atp)

    idx_start = np.array(start_indices)
    idx_end = np.array(end_indices)
    y_atp = np.array(valid_atp)

    print(f"  有效样本对: {len(y_atp)}")
    if len(y_atp) < 3:
        print("[WARN] 有效样本对太少，跳过特征优化")
        return

    atp_mean = np.mean(y_atp)
    atp_std = np.std(y_atp)

    n_iterations = 500000
    n_jobs = -1

    def run_trial(seed):
        rng = np.random.RandomState(seed)
        n_total = X_matrix_all.shape[1]
        n_select = rng.randint(4, min(n_total + 1, 20))
        feat_indices = rng.choice(n_total, n_select, replace=False)
        feat_indices.sort()

        X_sub = X_matrix_all[:, feat_indices]
        try:
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X_sub)
            pca = SKPCA(n_components=min(4, n_select))
            Data_Pca = pca.fit_transform(X_std)
            variance_ratio = pca.explained_variance_ratio_
            total_var = np.sum(variance_ratio)
            if total_var == 0:
                return None
            weights = variance_ratio / total_var
            result_scores = np.dot(Data_Pca, weights)
        except Exception:
            return None

        res_start = result_scores[idx_start]
        res_end = result_scores[idx_end]
        res_diff = res_end - res_start

        diff_mean = np.mean(res_diff)
        diff_std = np.std(res_diff)
        if diff_std == 0:
            return None
        cov_diff = np.mean((res_diff - diff_mean) * (y_atp - atp_mean))
        r_diff = cov_diff / (diff_std * atp_std)

        end_mean = np.mean(res_end)
        end_std = np.std(res_end)
        if end_std == 0:
            return None
        cov_end = np.mean((res_end - end_mean) * (y_atp - atp_mean))
        r_end = cov_end / (end_std * atp_std)

        return {
            "n_feats": len(feat_indices),
            "feat_idx": feat_indices,
            "diff_r": abs(r_diff),
            "end_r": abs(r_end)
        }

    print(f"  开始搜索 ({n_iterations} 次迭代)...")
    start_time = time.time()
    seeds = [random.randint(0, 100000000) for _ in range(n_iterations)]
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_trial)(seed) for seed in seeds
    )

    print(f"\n  计算完成! 耗时: {time.time() - start_time:.2f} 秒")

    clean_results = []
    for res in results:
        if res is not None:
            names = [available_pool[i] for i in res["feat_idx"]]
            clean_results.append({
                "Diff_Corr_Abs": res["diff_r"],
                "End_Corr_Abs": res["end_r"],
                "Num_Features": res["n_feats"],
                "Features_List": ", ".join(names)
            })

    df = pd.DataFrame(clean_results)
    df = df.sort_values(by="Diff_Corr_Abs", ascending=False)
    df.head(500).to_excel(output_path, index=False)

    print(f"\n  TOP 3 最佳特征组合:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"    No.{i+1}: 差值相关性={row['Diff_Corr_Abs']:.6f}")
        print(f"      特征: {row['Features_List']}")

    print(f"\n  结果已保存: {output_path}")
    print("\nStep 10 完成!")


# ============================================================
# 主函数
# ============================================================
def main():
    global BASE_DIR, ATP_PATH
    parser = argparse.ArgumentParser(description="MedNext_GC032 完整流水线")
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        help="指定运行的步骤 (1-10)，默认运行全部")
    parser.add_argument("--base-dir", default=BASE_DIR, help="数据根目录")
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    ATP_PATH = os.path.join(BASE_DIR, "ATP.xlsx")

    steps_to_run = args.steps if args.steps else list(range(1, 11))

    step_funcs = {
        1: step1_nnunet_bridge,
        2: step2_measure_from_label,
        3: step3_scatt,
        4: step4_cluster_merge,
        5: step5_cluster_analysis_1,
        6: step6_cluster_analysis_2,
        7: step7_roughness,
        8: step8_roughness_analysis,
        9: step9_pca,
        10: step10_feature_optimization,
    }

    print("="*60)
    print(f"MedNext_GC032 流水线 - 运行步骤: {steps_to_run}")
    print(f"数据根目录: {BASE_DIR}")
    print("="*60)

    for step in steps_to_run:
        if step in step_funcs:
            step_funcs[step]()
        else:
            print(f"[WARN] 未知步骤: {step}")

    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)


if __name__ == "__main__":
    main()
