import numpy as np
import nibabel as nib
import glob
n_data = nib.load(r'Data/训练数据样本/organoid_010_0000.nii.gz').get_fdata()
n_mean = np.mean(n_data)
print('Nifti 010 mean:', n_mean)
best, mdiff = '', 999
for f in glob.glob(r'Data/nnUNet_FXN_2023/FXN_0701/tr/*.tr'):
    r = np.fromfile(f, dtype=np.uint8)
    diff = abs(np.mean(r) - n_mean)
    if diff < mdiff:
        mdiff = diff
        best = f
print('Best:', best, 'diff:', mdiff)