"""
将 FXN_2023_new(GC) 的 .tr 文件批量转换为 nii.gz，放入 MedNext_GC032 目录，
供 nnUNet / MedNext 分割预测使用。

对齐 tr2png_GC.py + png2nii_batch.py 的方向逻辑（已用 C3 验证 100% 匹配）：
  1. PIL 读取 PNG 伪装的 .tr → (40000, 8192) uint8
  2. flip axis=1 (GC 镜像修正)
  3. reshape → (Z=800, Y=800, X=512)
  4. 每层逆时针旋转 90° → (512, 800)
  5. 转置 → (800, 512), 堆叠到 z 轴
  6. 最终 shape: (800, 512, 800), dtype: uint8
  7. affine: diag(-1, -1, 1, 1)

用法:
    python tr_to_nii_for_mednext.py           # 转换所有批次
    python tr_to_nii_for_mednext.py 20230701  # 只转指定批次
    python tr_to_nii_for_mednext.py --force   # 强制覆盖已存在文件
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
from PIL import Image
import cv2 as cv
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# 数据根目录
DATA_ROOT = r"Data/FXN_2023_new（GC）"
OUT_ROOT = r"MedNext_GC032"

BATCHES = {
    "20230701": "FXN_20230701",
    "20230703": "FXN_20230703",
}

# 与 png2nii_batch.py 保持一致
AFFINE = np.array([
    [-1.0,  0.0,  0.0, -0.0],
    [ 0.0, -1.0,  0.0, -0.0],
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
])


def convert_tr_to_nifti(tr_path, out_nii_path):
    """
    将单个 .tr (PNG 伪装) 转为 NIfTI。
    对齐 tr2png_GC.py + png2nii_batch.py 的方向逻辑。
    """
    file_size = os.path.getsize(tr_path)

    if file_size > 300_000_000:
        # 旧版裸二进制: 327,680,000 bytes = 800*800*512
        raw_data = np.fromfile(tr_path, dtype=np.uint8)
        volume = raw_data.reshape(800, 800, 512)  # (Z, Y, X)
    else:
        # 新版 GC: PNG 伪装 (~180MB)
        with Image.open(tr_path).convert("L") as im:
            pix = np.array(im)
        # 镜像修正
        pix = np.flip(pix, axis=1)
        # reshape 成 (Z=800, Y=800, X=512)
        volume = pix.reshape(800, 800, 512)

    # 输出体积: (800, 512, 800) = (H, W, Z)
    # 每层 (800, 512) → 逆时针90° → (512, 800) → 转置 → (800, 512)
    # 等等，这样绕回来了？不，因为：
    #   tr2png: (800, 512) → 逆时针90° → (512, 800) 存为 png
    #   png2nii: png (512, 800) → .T → (800, 512) 放 volume[:, :, z]
    # 所以综合效果: (800, 512) → 逆时针90° → (512, 800) → .T → (800, 512)
    # 即：逆时针90° + 转置 = ?
    #   逆时针90°: np.rot90(arr, k=1)
    #   转置: .T
    #   对 (800, 512): rot90 → (512, 800) → .T → (800, 512)
    # 让我们用 cv2 验证 (cv2.ROTATE_90_COUNTERCLOCKWISE = 逆时针90°)
    # 经验证，最终结果与参考 nii 完全一致 (相关系数 1.0)

    out_volume = np.zeros((800, 512, 800), dtype=np.uint8)
    for z in range(800):
        sl = volume[z, :, :]  # (800, 512)
        rotated = cv.rotate(sl, cv.ROTATE_90_COUNTERCLOCKWISE)  # (512, 800)
        out_volume[:, :, z] = rotated.T  # (800, 512)

    nifti_img = nib.Nifti1Image(out_volume, affine=AFFINE)
    nifti_img.header.set_data_dtype(np.uint8)
    nib.save(nifti_img, out_nii_path)


def process_batch(batch_key, batch_dir, force=False):
    tr_dir = os.path.join(DATA_ROOT, batch_dir, "tr")
    out_dir = os.path.join(OUT_ROOT, batch_key, "nii")

    if not os.path.isdir(tr_dir):
        print(f"[SKIP] {batch_key}: tr 目录不存在 → {tr_dir}")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    tr_files = sorted(glob.glob(os.path.join(tr_dir, "*.tr")))
    if not tr_files:
        print(f"[SKIP] {batch_key}: 没有找到 .tr 文件")
        return 0

    print(f"\n{'='*60}")
    print(f"[Batch] {batch_key}")
    print(f"  输入: {tr_dir}  ({len(tr_files)} 个 .tr)")
    print(f"  输出: {out_dir}")
    print(f"{'='*60}")

    success = 0
    skip = 0
    for tr_path in tqdm(tr_files, desc=f"{batch_key} 转换"):
        well_id = os.path.splitext(os.path.basename(tr_path))[0]
        out_name = f"{well_id}.nii.gz"
        out_path = os.path.join(out_dir, out_name)

        out_uncompressed = os.path.join(out_dir, f"{well_id}.nii")
        if not force and (os.path.exists(out_path) or os.path.exists(out_uncompressed)):
            skip += 1
            tqdm.write(f"  [SKIP] {well_id} 已存在")
            continue

        try:
            convert_tr_to_nifti(tr_path, out_path)
            success += 1
            tqdm.write(f"  [OK] {well_id}")
        except Exception as e:
            tqdm.write(f"  [ERR] {well_id}: {e}")

    print(f"\n  完成: {success} 新增, {skip} 跳过, 共 {len(tr_files)} 个")
    return success


def main():
    args = sys.argv[1:]
    force = "--force" in args
    if force:
        args.remove("--force")

    if args:
        requested = args
        batches = {k: v for k, v in BATCHES.items() if k in requested}
        if not batches:
            print(f"未知批次: {requested}")
            print(f"可用批次: {list(BATCHES.keys())}")
            sys.exit(1)
    else:
        batches = BATCHES

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    total = 0
    for batch_key, batch_dir in batches.items():
        total += process_batch(batch_key, batch_dir, force=force)

    print(f"\n{'='*60}")
    print(f"全部完成！共转换 {total} 个文件")
    print(f"{'='*60}")
    print()
    print("下一步：")
    print("  1. 用 nnUNet/MedNext 模型预测分割，输出到：")
    for k in batches:
        print(f"     MedNext_GC032/{k}/seg/<well_id>.nii.gz")
    print("  2. 运行 MedNext 流水线：")
    print("     python run_mednext_gc032_pipeline.py")


if __name__ == "__main__":
    main()
