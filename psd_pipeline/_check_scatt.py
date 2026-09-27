import pandas as pd
import nibabel as nib
from scipy import ndimage
import numpy as np
import os

WELLS = ['B4','B6','B9','B10','F4','F5','F7','F8','F10','G4','G6','G9','G11']
MIN_VOL = 50

print(f"{'Well_Date':<12} {'Scatt':>6} {'MedNext':>7} {'Match':>6}")
print('-' * 35)

all_match = True
for date_tag, date_dir in [('0701', '20230701'), ('0703', '20230703')]:
    for well in WELLS:
        scatt_path = os.path.join(r'D:\Desktop\music\measure\ICC005', date_dir, 'scatt', well + '_scatt.xlsx')
        seg_path = os.path.join(r'D:\Desktop\music\measure\ICC005', date_dir, 'seg', well + '.nii.gz')

        if not os.path.exists(scatt_path) or not os.path.exists(seg_path):
            continue

        scatt = pd.read_excel(scatt_path)
        n_scatt = len(scatt)

        seg = nib.load(seg_path).get_fdata()
        labeled, nf = ndimage.label(seg > 0)
        slices = ndimage.find_objects(labeled)
        n_mednext = 0
        for lid in range(1, nf + 1):
            sl = slices[lid - 1]
            if sl is not None and (labeled[sl] == lid).sum() >= MIN_VOL:
                n_mednext += 1

        match = 'OK' if n_scatt == n_mednext else 'FAIL'
        if match == 'FAIL':
            all_match = False
        label = well + '_' + date_tag
        print(f"{label:<12} {n_scatt:>6} {n_mednext:>7} {match:>6}")

print()
if all_match:
    print('>>> ALL MATCH! Scatt is aligned with MedNext segmentation.')
else:
    print('>>> MISMATCH! Scatt organoid count does not match MedNext seg.')

# Also check scatt columns and sample values
print()
scatt_sample = pd.read_excel(r'D:\Desktop\music\measure\ICC005\20230701\scatt\B4_scatt.xlsx')
print('Scatt columns:', scatt_sample.columns.tolist())
print('Sample rows:')
print(scatt_sample.head(3))