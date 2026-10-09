import pandas as pd
import numpy as np
import os

def analyze_scatt_means(folder, dataset_name):
    """Analyze Scatt_Mean values in Excel files"""
    excel_files = [f for f in os.listdir(folder) if f.endswith('.xlsx')]
    means = []
    
    print(f"\n=== {dataset_name} Scatt_Mean Analysis ===")
    print(f"Found {len(excel_files)} Excel files")
    
    for file in excel_files[:10]:  # 分析前10个文件
        file_path = os.path.join(folder, file)
        try:
            df = pd.read_excel(file_path)
            if 'Scatt_Mean' in df.columns:
                file_means = df['Scatt_Mean'].dropna()
                if len(file_means) > 0:
                    mean_val = file_means.mean()
                    means.extend(file_means.tolist())
                    print(f"  {file}: {len(file_means)} objects, mean = {mean_val:.1f}")
        except Exception as e:
            print(f"  {file}: ERROR - {e}")
    
    if means:
        print(f"\n  OVERALL: Total {len(means)} objects, Mean = {np.mean(means):.1f}, Median = {np.median(means):.1f}")
    
    return means

# 分析两个数据集
gc_folder = r'E:\student\Private\student13\Measure_copy\Data\FXN_2023_new（GC）\FXN_20230701\measure_excel'
nnunet_folder = r'E:\student\Private\student13\Measure_copy\Data\nnUNet_FXN_2023\FXN_0701\measure_excel'

gc_means = analyze_scatt_means(gc_folder, 'GC (FXN_2023_new)')
nnunet_means = analyze_scatt_means(nnunet_folder, 'nnUNet')

print(f"\n=== SUMMARY of DISCOVERY ===")
if gc_means and nnunet_means:
    print(f"GC Scatt_Mean average: {np.mean(gc_means):.1f}")
    print(f"nnUNet Scatt_Mean average: {np.mean(nnunet_means):.1f}")
    print(f"Ratio (GC/nnUNet): {np.mean(gc_means)/np.mean(nnunet_means):.1f}x")

