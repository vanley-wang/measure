import os
import numpy as np
import scipy.io as sio

try:
    import nibabel as nib
except Exception as e:
    nib = None
    print('NO_NIBABEL', repr(e))

mat_paths = [
    r'Data/FXN_2023/FXN_20230701/scatt_mat/B4_0701_scatt.mat',
    r'Data/FXN_2023/FXN_20230701/seg_fill/B4_0701_fill.mat',
    r'Data/FXN_2023/FXN_20230701/seg_label/B4_0701_label.mat',
    r'Data/FXN_2023/FXN_20230701/seg_mat/B4_0701.mat',
]
for p in mat_paths:
    print('\nMAT', p, 'exists=', os.path.exists(p), 'size=', os.path.getsize(p) if os.path.exists(p) else None)
    mat = sio.loadmat(p)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print(' keys=', keys)
    for k in keys:
        v = mat[k]
        print(' ', k, 'shape=', v.shape, 'dtype=', v.dtype, 'min=', np.nanmin(v), 'max=', np.nanmax(v), 'nonzero=', np.count_nonzero(v))
        if v.size < 20_000_000:
            print('  unique[:20]=', np.unique(v)[:20])

nii_paths = [
    r'Data/nnUNet_FXN_2023/FXN_0701/organoid_003_0000.nii.gz',
    r'Data/nnUNet_FXN_2023/FXN_0701_prediction/organoid_003.nii.gz',
    r'Data/nnUNet_FXN_2023/FXN_0703/organoid_003_0000.nii.gz',
    r'Data/nnUNet_FXN_2023/FXN_0703_prediction/organoid_003.nii.gz',
]
if nib is not None:
    for p in nii_paths:
        print('\nNII', p, 'exists=', os.path.exists(p), 'size=', os.path.getsize(p) if os.path.exists(p) else None)
        img = nib.load(p)
        data = np.asanyarray(img.dataobj)
        print(' shape=', data.shape, 'dtype=', data.dtype, 'min=', np.nanmin(data), 'max=', np.nanmax(data), 'nonzero=', np.count_nonzero(data))
        vals = np.unique(data)
        print(' unique_count=', len(vals), 'unique[:20]=', vals[:20])
        print(' affine=')
        print(img.affine)
