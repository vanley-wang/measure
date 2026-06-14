import numpy as np
import nibabel as nib
import itertools
import glob

n_data = nib.load(r'E:/student/Private/student13/Measure_copy/训练数据样本/organoid_010_0000.nii.gz').get_fdata()

files = glob.glob(r'E:/student/Private/student13/Measure_copy/Data/FXN_2023_new0/FXN_20230701/tr/*.tr')

def get_perms(vol):
    perms = []
    for p in itertools.permutations([0,1,2]):
        v = np.transpose(vol, p)
        for flips in itertools.product([False, True], repeat=3):
            vf = v.copy()
            if flips[0]: vf = np.flip(vf, 0)
            if flips[1]: vf = np.flip(vf, 1)
            if flips[2]: vf = np.flip(vf, 2)
            if vf.shape == n_data.shape:
                perms.append((p, flips, vf))
    return perms

found = False
for f in files:
    print('Testing', f)
    raw = np.fromfile(f, dtype=np.uint8).reshape(800, 800, 512)
    perms = get_perms(raw)
    for p, flips, vf in perms:
        if np.mean(np.abs(vf - n_data)) < 0.1:
            print('MATCH FOUND!')
            print('File:', f, 'Perm:', p, 'Flips:', flips)
            found = True
            break
    if found: break

if not found:
    print('No simple match. Trying the 40000x8192 flip logic...')
    for f in files:
        raw = np.fromfile(f, dtype=np.uint8).reshape(40000, 8192)
        raw = np.flip(raw, axis=1).reshape(800, 800, 512)
        perms = get_perms(raw)
        for p, flips, vf in perms:
            if np.mean(np.abs(vf - n_data)) < 0.1:
                print('MATCH FOUND with 40000x8192 flip!')
                print('File:', f, 'Perm:', p, 'Flips:', flips)
                found = True
                break
        if found: break
