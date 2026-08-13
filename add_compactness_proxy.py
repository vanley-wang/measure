"""
Add Compactness as a roughness proxy to all measure_excel files.

Compactness = Surface / Volume^(2/3)
- High compactness = smooth, spherical surface
- Low compactness = rough, irregular surface

This is a fast alternative to the expensive 3D FFT-based roughness.
"""
import os
import glob
import pandas as pd

DATA_FOLDERS = [
    'Data/nnUNet_FXN_2023/FXN_0701/measure_excel',
    'Data/nnUNet_FXN_2023/FXN_0703/measure_excel',
]

for folder in DATA_FOLDERS:
    files = glob.glob(os.path.join(folder, '*.xlsx'))
    for fp in files:
        df = pd.read_excel(fp)
        df['Compactness'] = df['Organoids_Surface'] / (df['Organoids_Volume_Fill'] ** (2/3))
        df.to_excel(fp, index=False)
    print(f"Added Compactness to {len(files)} files in {folder}")

print("[Done]")
