import os
import glob
import nibabel as nib

# 检查一个目录下的nii文件维度
check_dir = r"E:\student\Private\student13\Measure_copy\Data\FXN_2023_new（闭10新聚类）\FXN_20230703\scatt_mat"

nii_files = glob.glob(os.path.join(check_dir, "*.nii.gz"))

if nii_files:
    print(f"检查目录: {check_dir}")
    print(f"找到 {len(nii_files)} 个nii.gz文件\n")
    
    # 检查前3个文件
    for i, nii_file in enumerate(nii_files[:3]):
        img = nib.load(nii_file)
        data = img.get_fdata()
        print(f"文件 {i+1}: {os.path.basename(nii_file)}")
        print(f"  维度: {data.shape}")
        print(f"  数据类型: {data.dtype}\n")
else:
    print("没有找到nii.gz文件，请等待转换完成")
