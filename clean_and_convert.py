import os
import glob
import numpy as np
import nibabel as nib
import scipy.io as sio
from tqdm import tqdm

# ==================== 配置区 ====================
# 要处理的目录列表
INPUT_DIRS = [
    r"E:\student\Private\student13\Measure_copy\Data\FXN_2023_new（闭10新聚类）\FXN_20230703\scatt_mat",
    r"E:\student\Private\student13\Measure_copy\Data\FXN_2023_new（闭10新聚类）\FXN_20230703\seg_fill",
    r"E:\student\Private\student13\Measure_copy\Data\FXN_2023_new（闭10新聚类）\FXN_20230703\seg_label",
    r"E:\student\Private\student13\Measure_copy\Data\FXN_2023_new（闭10新聚类）\FXN_20230703\seg_mat_test",
    r"E:\student\Private\student13\Measure_copy\Data\FXN_2023_new（闭10新聚类）\FXN_20230701\seg_fill",
    r"E:\student\Private\student13\Measure_copy\Data\FXN_2023_new（闭10新聚类）\FXN_20230701\seg_label",
    r"E:\student\Private\student13\Measure_copy\Data\FXN_2023_new（闭10新聚类）\FXN_20230701\seg_mat_test",
]

# 不同目录对应的变量名映射
VAR_NAME_MAPPING = {
    'scatt_mat': 'data_scatt',
    'seg_fill': 'Data_fill', 
    'seg_label': 'Data_label',
    'seg_mat_test': 'Data_Seg'
}

# ================================================

def get_var_name_from_dir(dir_path):
    """根据目录名获取对应的变量名"""
    dir_name = os.path.basename(dir_path)
    return VAR_NAME_MAPPING.get(dir_name, None)

def convert_mat_to_nifti(mat_path, var_name, out_nii_path):
    """将mat文件转换为nii文件"""
    # 读取mat文件
    mat_data = sio.loadmat(mat_path)
    
    if var_name not in mat_data:
        print(f"警告: 在文件 {mat_path} 中未找到变量 {var_name}")
        return False
    
    # 获取数据
    volume = mat_data[var_name]
    
    # 调整维度: 从(512, 800, 800)转换为(800, 512, 800)
    # 使用transpose交换前两个维度
    if volume.shape == (512, 800, 800):
        volume = np.transpose(volume, (1, 0, 2))
    
    # 创建NIfTI对象
    nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
    
    # 保存文件
    nib.save(nifti_img, out_nii_path)
    return True

def process_directory(input_dir):
    """处理单个目录下的所有mat文件"""
    var_name = get_var_name_from_dir(input_dir)
    if var_name is None:
        print(f"警告: 无法确定目录 {input_dir} 对应的变量名，跳过")
        return
    
    # 先删除旧的nii.gz文件
    old_nii_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    for nii_file in old_nii_files:
        os.remove(nii_file)
    print(f"删除了 {len(old_nii_files)} 个旧的nii.gz文件")
    
    # 获取所有mat文件
    mat_files = sorted(glob.glob(os.path.join(input_dir, "*.mat")))
    if not mat_files:
        print(f"在目录 {input_dir} 中未找到mat文件")
        return
    
    print(f"\n处理目录: {input_dir}")
    print(f"找到 {len(mat_files)} 个mat文件，变量名: {var_name}")
    
    for mat_path in tqdm(mat_files):
        # 生成输出文件名
        file_name = os.path.basename(mat_path)
        nii_name = os.path.splitext(file_name)[0] + ".nii.gz"
        out_nii_path = os.path.join(input_dir, nii_name)
        
        # 转换
        convert_mat_to_nifti(mat_path, var_name, out_nii_path)
    
    print(f"目录 {input_dir} 处理完成")

if __name__ == "__main__":
    print("开始清理旧文件并重新批量转换mat文件到nii文件...")
    
    for input_dir in INPUT_DIRS:
        if os.path.exists(input_dir):
            process_directory(input_dir)
        else:
            print(f"警告: 目录不存在 {input_dir}")
    
    print("\n所有转换完成！")
