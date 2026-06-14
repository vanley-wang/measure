import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# ==================== 配置区 ====================
# 这里修改为你存放 32 个待预测 .tr 文件的目录
INPUT_DIR = r"E:\student\Private\student13\Measure_copy\Data\FXN_2023\FXN_20230703\tr"

# nnUNet 预测需要的 NIfTI 输入目录
OUTPUT_DIR = r"E:\student\Private\student13\Measure_copy\Data\nnUNet_raw_input_20230703"
# 你为这批数据起的前缀名称
PREFIX = "organoid" 

# ================================================

def convert_tr_to_nifti(tr_path, out_nii_path):
    # 1. 自动判断这批文件究竟是 Raw 数据 还是 被压缩成伪装的 PNG
    file_size = os.path.getsize(tr_path)
    
    if file_size > 300_000_000:
        # 大于300MB为旧版裸二进制数据 (327,680,000 Byte)
        raw_data = np.fromfile(tr_path, dtype=np.uint8)
        volume = raw_data.reshape(800, 800, 512)
        # 旧版通常不需要再次翻转列 (基于以前代码的备注)
        
    else:
        # 大概率是新版 GC 里的压缩大尺寸 PNG (~180MB)
        with Image.open(tr_path).convert('L') as im:
            pix = np.array(im)
            
        # GC 数据里因为拼接原因列反了，需要翻转回来
        pix = np.flip(pix, axis=1)
        volume = pix.reshape(800, 800, 512)

    # 2. 空间旋转：将 (800, 800, 512) 的后两个轴 (即截面轴) 旋转 90 度
    #    相当于代码原本的 cv.rotate(... ROTATE_90_COUNTERCLOCKWISE)
    #    使得最终体积形状变为 (800, 512, 800)，和你的 NIfTI 训练文件完美契合
    volume_rot = np.rot90(volume, k=1, axes=(1, 2))
    
    # 将 uint8 缩放或转为常用医学影像格式 float32 防止截断
    volume_float = volume_rot.astype(np.float32)

    # 3. 创建 NIfTI 对象 (无缩放与仿射偏移)
    nifti_img = nib.Nifti1Image(volume_float, affine=np.eye(4))
    
    # 4. 写入文件
    nib.save(nifti_img, out_nii_path)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有的 tr 文件
    tr_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tr")))
    print(f"找到 {len(tr_files)} 个 .tr 文件。开始转换为 nnUNet 格式...")

    for i, tr_path in enumerate(tqdm(tr_files)):
        # nnUNet 要求的严格命名法: prefix_001_0000.nii.gz 
        # 我们用 i+1 来排序映射，如 organoid_001_0000.nii.gz
        file_idx = str(i + 1).zfill(3)
        nii_name = f"{PREFIX}_{file_idx}_0000.nii.gz"
        out_nii_path = os.path.join(OUTPUT_DIR, nii_name)
        
        convert_tr_to_nifti(tr_path, out_nii_path)
    
    print("\n转换完成！你可以将输出目录中的数据直接喂给 nnUNetv2_predict 命令了。")