# ============================================================
# nnUNet .nii.gz 分割 → per-organoid morphological measurements
# 输入: nnUNet_FXN/FXN_0701_seg/*.nii.gz (二值掩膜, 0/1)
#      nnUNet_FXN/FXN_0701/*_0000.nii.gz  (OCT强度图)
# 输出: nnUNet_FXN/FXN_0701/measure_excel/*.xlsx
# ============================================================
import os, sys, glob, gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.measure import label as sk_label, marching_cubes, mesh_surface_area
from tqdm import tqdm

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nnUNet_FXN')
BATCHES = ['FXN_0701', 'FXN_0703']
MIN_VOLUME = 50          # 最小体素数（过滤噪声）
MARCHING_CUBES = False   # 快速表面积（边界体素proxy，速度×10）

def compute_surface_area(mask_3d):
    coords = np.where(mask_3d)
    if len(coords[0]) == 0:
        return 0.0
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1
    cropped = mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]

    if MARCHING_CUBES:
        try:
            verts, faces, _, _ = marching_cubes(cropped, level=0.5)
            return mesh_surface_area(verts, faces)
        except Exception:
            pass
    eroded = ndimage.binary_erosion(cropped)
    boundary = cropped & (~eroded)
    return float(np.count_nonzero(boundary))


def process_one_well(seg_path, img_path, well_id, date_suffix):
    seg = nib.load(seg_path).get_fdata()
    img = nib.load(img_path).get_fdata()

    # 确保尺寸一致
    min_z = min(seg.shape[0], img.shape[0])
    min_y = min(seg.shape[1], img.shape[1])
    min_x = min(seg.shape[2], img.shape[2])
    seg = seg[:min_z, :min_y, :min_x]
    img = img[:min_z, :min_y, :min_x]

    # 连通域标记（每个个体一个 ID，从 1 开始）
    labeled, num_features = ndimage.label(seg > 0)
    if num_features == 0:
        return [], None

    # 一次性获取所有 bbox
    slices = ndimage.find_objects(labeled)
    rows = []

    for label_id in range(1, num_features + 1):
        sl = slices[label_id - 1]
        if sl is None:
            continue

        # 裁切区域
        crop_label = labeled[sl]
        inst_mask = (crop_label == label_id)
        volume_raw = int(np.count_nonzero(inst_mask))
        if volume_raw < MIN_VOLUME:
            continue

        # 填孔得到 filled mask
        try:
            fill_mask = ndimage.binary_fill_holes(inst_mask)
        except Exception:
            fill_mask = inst_mask
        volume_fill = int(np.count_nonzero(fill_mask))

        # 空腔：filled 有但 raw 没有的部分
        cavity_mask = fill_mask & (~inst_mask)
        cavity_volume = int(np.count_nonzero(cavity_mask))
        _, cavity_num = sk_label(cavity_mask, connectivity=3, return_num=True)

        # 表面积
        surface = compute_surface_area(fill_mask)

        # 长轴/短轴
        dz = sl[0].stop - sl[0].start
        dy = sl[1].stop - sl[1].start
        dx = sl[2].stop - sl[2].start
        long_axis = float(max(dz, dy, dx))
        short_axis = float(min(dz, dy, dx))

        # 壁厚
        wall_thickness = max(1.0, volume_fill / (surface + 1e-6) * 0.25)

        # 球形度
        sphericity = (np.pi ** (1.0 / 3)) * ((6.0 * volume_fill) ** (2.0 / 3)) / (surface + 1e-6)
        sphericity = min(sphericity, 5.0)

        # 散射：从 OCT 强度图像中取对应区域的均值和标准差
        crop_img = img[sl]
        img_vals = crop_img[inst_mask]
        scatt_mean = float(np.mean(img_vals)) if len(img_vals) > 0 else 0.0
        scatt_std = float(np.std(img_vals)) if len(img_vals) > 0 else 0.0

        rows.append({
            'Index': f"{well_id}_{date_suffix}_{label_id}",
            'Organoids_Volume': volume_raw,
            'Organoids_Volume_Fill': volume_fill,
            'Organoids_Surface': round(surface, 2),
            'Cavity_Volume': cavity_volume,
            'CavityNum': cavity_num,
            'LongAxis': long_axis,
            'ShortAxis': short_axis,
            'Wall_Thickness': round(wall_thickness, 4),
            'Sphericity': round(sphericity, 4),
            'Scatt_Mean': round(scatt_mean, 4),
            'Scatt_STD': round(scatt_std, 4),
        })

    return rows, None


def main():
    for batch in BATCHES:
        seg_dir = os.path.join(BASE_DIR, f'{batch}_seg')
        img_dir = os.path.join(BASE_DIR, batch)
        out_dir = os.path.join(BASE_DIR, batch, 'measure_excel')
        os.makedirs(out_dir, exist_ok=True)

        date_suffix = batch.replace('FXN_', '')

        # 匹配 seg 和 img
        seg_files = sorted(glob.glob(os.path.join(seg_dir, '*.nii.gz')))
        if not seg_files:
            print(f'[WARN] No seg files in {seg_dir}')
            continue

        print(f'\n{"="*60}')
        print(f'  Processing {batch}: {len(seg_files)} wells')
        print(f'{"="*60}')

        total = 0
        for seg_path in tqdm(seg_files, desc=date_suffix):
            fname = os.path.basename(seg_path)  # B10_1.nii.gz
            well_id = fname.replace('.nii.gz', '').replace('_1', '')

            # 找对应 OCT 图像
            img_path = os.path.join(img_dir, f'{well_id}_1_0000.nii.gz')
            if not os.path.exists(img_path):
                # 尝试其他命名模式
                alt = os.path.join(img_dir, f'{well_id}_0000.nii.gz')
                if os.path.exists(alt):
                    img_path = alt
                else:
                    print(f'  [WARN] Missing image for {well_id}, skip')
                    continue

            rows, err = process_one_well(seg_path, img_path, well_id, date_suffix)
            if err:
                print(f'  {err}')
                continue

            if rows:
                df = pd.DataFrame(rows)
                out_path = os.path.join(out_dir, f'{well_id}_{date_suffix}.xlsx')
                df.to_excel(out_path, index=False)
                total += len(rows)

            gc.collect()

        print(f'  [Done] {batch}: {total} organoids → {out_dir}')

    print(f'\n[All Done]')

    # 自检：对比新旧nnUNet vs Original ICC的CV
    print(f'\n{"="*60}')
    print(f'  SELF-CHECK: Morphological Diversity Comparison')
    print(f'{"="*60}')
    CF=['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
        'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
        'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']
    CF_S=['Vol','FillVol','Surf','CavVol','CavNum',
          'LongAx','ShortAx','WallTh','Spher','ScattM','ScattS']

    # 读新nnUNet数据
    new_vals = {f:[] for f in CF}
    org_vals = {f:[] for f in CF}
    me_dir = os.path.join(BASE_DIR, 'FXN_0701', 'measure_excel')
    for xf in os.listdir(me_dir):
        if not xf.endswith('.xlsx'): continue
        df=pd.read_excel(os.path.join(me_dir,xf))
        for f in CF:
            if f in df.columns:
                new_vals[f].extend(df[f].dropna().tolist())

    # 读原始ICC数据
    icc_dir = os.path.join(os.path.dirname(BASE_DIR), 'Data','FXN_2023_new（ICC）','FXN_20230701','measure_excel')
    for xf in os.listdir(icc_dir):
        if not xf.endswith('.xlsx'): continue
        df=pd.read_excel(os.path.join(icc_dir,xf))
        for f in CF:
            if f in df.columns:
                org_vals[f].extend(df[f].dropna().tolist())

    print(f'  {"Feature":<22s} {"CV New":>8s} {"CV Orig":>8s} {"Ratio":>8s} {"Verdict":>10s}')
    print('-'*62)
    for f in CF:
        an=np.array(new_vals[f]); ao=np.array(org_vals[f])
        cv_n=an.std()/an.mean() if an.mean()!=0 else 0
        cv_o=ao.std()/ao.mean() if ao.mean()!=0 else 0
        ratio=cv_n/cv_o if cv_o>0 else 0
        if ratio<0.7: v='⚠ DIVERSITY↓'
        elif ratio<0.9: v='~ slightly↓'
        elif ratio>1.3: v='★ DIVERSITY↑'
        else: v='≈ similar'
        print(f'  {f:<22s} {cv_n:>8.3f} {cv_o:>8.3f} {ratio:>7.2f}x {v:>12s}')

    # 专项：Cavity分布
    nc=np.array(new_vals['CavityNum']); oc=np.array(org_vals['CavityNum'])
    print(f'\n  CavityNum > 0: New={(nc>0).sum()/len(nc)*100:.1f}%, Orig={(oc>0).sum()/len(oc)*100:.1f}%')
    print(f'  CavityNum mean: New={nc.mean():.4f}, Orig={oc.mean():.4f}')
    nz=nc[nc>0]; oz=oc[oc>0]
    if len(nz)>0 and len(oz)>0:
        print(f'  CavityNum (nonzero): New mean={nz.mean():.2f}, Orig mean={oz.mean():.2f}')

if __name__ == '__main__':
    main()