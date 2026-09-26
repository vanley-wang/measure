# ============================================================
# MedNext .nii.gz 分割 → per-organoid morphological measurements
# 输入: MedNext_FXN_2023/FXN_0701/*.nii.gz (二值掩膜)
#      nnUNet_FXN/FXN_0701/*_0000.nii.gz  (借用nnUNet目录的OCT强度图)
# 输出: MedNext_FXN_2023/FXN_0701/measure_excel/*.xlsx
# ============================================================
import os, sys, glob, gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.measure import label as sk_label
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(ROOT, 'MedNext_FXN_2023')
OCT_DIR  = os.path.join(ROOT, 'nnUNet_FXN')   # 借用nnUNet的OCT图像
BATCHES = ['FXN_0701', 'FXN_0703']
MIN_VOLUME = 50
MARCHING_CUBES = False

def compute_surface_area(mask_3d):
    coords = np.where(mask_3d)
    if len(coords[0]) == 0:
        return 0.0
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1
    cropped = mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]
    eroded = ndimage.binary_erosion(cropped)
    boundary = cropped & (~eroded)
    return float(np.count_nonzero(boundary))

def process_one_well(seg_path, img_path, well_id, date_suffix):
    seg = nib.load(seg_path).get_fdata()
    img = nib.load(img_path).get_fdata()
    min_z = min(seg.shape[0], img.shape[0])
    min_y = min(seg.shape[1], img.shape[1])
    min_x = min(seg.shape[2], img.shape[2])
    seg = seg[:min_z, :min_y, :min_x]
    img = img[:min_z, :min_y, :min_x]

    labeled, num_features = ndimage.label(seg > 0)
    if num_features == 0:
        return [], None

    slices = ndimage.find_objects(labeled)
    rows = []

    for label_id in range(1, num_features + 1):
        sl = slices[label_id - 1]
        if sl is None:
            continue
        crop_label = labeled[sl]
        inst_mask = (crop_label == label_id)
        volume_raw = int(np.count_nonzero(inst_mask))
        if volume_raw < MIN_VOLUME:
            continue
        try:
            fill_mask = ndimage.binary_fill_holes(inst_mask)
        except Exception:
            fill_mask = inst_mask
        volume_fill = int(np.count_nonzero(fill_mask))
        cavity_mask = fill_mask & (~inst_mask)
        cavity_volume = int(np.count_nonzero(cavity_mask))
        _, cavity_num = sk_label(cavity_mask, connectivity=3, return_num=True)
        surface = compute_surface_area(fill_mask)
        dz = sl[0].stop - sl[0].start
        dy = sl[1].stop - sl[1].start
        dx = sl[2].stop - sl[2].start
        long_axis = float(max(dz, dy, dx))
        short_axis = float(min(dz, dy, dx))
        wall_thickness = max(1.0, volume_fill / (surface + 1e-6) * 0.25)
        sphericity = (np.pi ** (1.0 / 3)) * ((6.0 * volume_fill) ** (2.0 / 3)) / (surface + 1e-6)
        sphericity = min(sphericity, 5.0)
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
        oct_dir = os.path.join(OCT_DIR, batch)
        out_dir = os.path.join(BASE_DIR, batch, 'measure_excel')
        os.makedirs(out_dir, exist_ok=True)
        date_suffix = batch.replace('FXN_', '')

        seg_files = sorted(glob.glob(os.path.join(seg_dir, '*.nii.gz')))
        seg_files = [f for f in seg_files if 'plans' not in f]
        if not seg_files:
            print(f'[WARN] No seg files in {seg_dir}')
            continue

        print(f'\n{"="*60}')
        print(f'  Processing {batch}: {len(seg_files)} wells')
        print(f'{"="*60}')

        total = 0
        for seg_path in tqdm(seg_files, desc=date_suffix):
            fname = os.path.basename(seg_path)
            well_id = fname.replace('.nii.gz', '').replace('_1', '')
            img_path = os.path.join(oct_dir, f'{well_id}_1_0000.nii.gz')
            if not os.path.exists(img_path):
                print(f'  [WARN] Missing OCT for {well_id}, skip')
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

        print(f'  [Done] {batch}: {total} organoids -> {out_dir}')

    # ---- Self-check ----
    print(f'\n{"="*60}')
    print(f'  SELF-CHECK: MedNext vs Original ICC')
    print(f'{"="*60}')
    CF = ['Organoids_Volume','Organoids_Volume_Fill','Organoids_Surface',
          'Cavity_Volume','CavityNum','LongAxis','ShortAxis',
          'Wall_Thickness','Sphericity','Scatt_Mean','Scatt_STD']

    med_vals = {f:[] for f in CF}
    org_vals = {f:[] for f in CF}

    me_dir = os.path.join(BASE_DIR, 'FXN_0701', 'measure_excel')
    for xf in os.listdir(me_dir):
        if not xf.endswith('.xlsx'): continue
        df = pd.read_excel(os.path.join(me_dir, xf))
        for f in CF:
            if f in df.columns:
                med_vals[f].extend(df[f].dropna().tolist())

    icc_dir = os.path.join(ROOT, 'Data', 'FXN_2023_new（ICC）', 'FXN_20230701', 'measure_excel')
    for xf in os.listdir(icc_dir):
        if not xf.endswith('.xlsx'): continue
        df = pd.read_excel(os.path.join(icc_dir, xf))
        for f in CF:
            if f in df.columns:
                org_vals[f].extend(df[f].dropna().tolist())

    print(f'  {"Feature":<22s} {"CV MedN":>8s} {"CV Orig":>8s} {"Ratio":>8s} {"Verdict":>10s}')
    print('-'*62)
    for f in CF:
        am = np.array(med_vals[f]); ao = np.array(org_vals[f])
        cvm = am.std() / am.mean() if am.mean() != 0 else 0
        cvo = ao.std() / ao.mean() if ao.mean() != 0 else 0
        ratio = cvm / cvo if cvo > 0 else 0
        if ratio < 0.7:    v = 'WARNING DIV'
        elif ratio < 0.9:  v = '~ slight'
        elif ratio > 1.3:  v = 'STAR UP'
        else:              v = '~ similar'
        print(f'  {f:<22s} {cvm:>8.3f} {cvo:>8.3f} {ratio:>7.2f}x {v:>12s}')

    nc = np.array(med_vals['CavityNum']); oc = np.array(org_vals['CavityNum'])
    print(f'\n  CavityNum > 0: MedNext={(nc>0).sum()/len(nc)*100:.1f}%, Orig={(oc>0).sum()/len(oc)*100:.1f}%')
    print(f'  CavityNum mean: MedNext={nc.mean():.4f}, Orig={oc.mean():.4f}')
    nz = nc[nc>0]; oz = oc[oc>0]
    if len(nz)>0 and len(oz)>0:
        print(f'  CavityNum (nonzero): MedNext={nz.mean():.2f}, Orig={oz.mean():.2f}')
    print(f'\n[All Done]')


if __name__ == '__main__':
    main()