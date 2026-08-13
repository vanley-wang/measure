import numpy as np
import nibabel as nib

print("=== 分析tr2nii.py的转换流程 ===")
print("\n原图转换步骤：")
print("1. 原始数据: (800, 800, 512)")  
print("2. rot90(volume, k=1, axes=(1, 2))")
print("3. 结果: (800, 512, 800)")
print("\n我们现在的mat文件：")
print("1. 原始mat: (512, 800, 800)")
print("\n让我们模拟一下可能的正确转换方式...")

# 模拟数据
simulated_mat = np.random.rand(512, 800, 800)
print(f"\n模拟mat形状: {simulated_mat.shape}")

# 尝试1: 先transpose再rot90
try:
    step1 = np.transpose(simulated_mat, (1, 0, 2))  # (800, 512, 800)
    print(f"transpose后: {step1.shape}")
    
    # 不对，这和原图的处理不同
    # 原图是 (800,800,512) -> rot90 -> (800,512,800)
    # 我们的mat是 (512,800,800)
    
    # 让我们先把mat变成类似原图的形状
    print("\n=== 重新思考 ===")
    print("mat: (512, 800, 800)")
    print("我们需要把它变成类似原图的 (800, 800, 512)，然后再用同样的rot90")
    
    # 尝试方案：
    print("\n方案A:")
    stepA1 = np.transpose(simulated_mat, (1, 2, 0))  # (800,800,512)
    print(f"  step1 transpose(1,2,0): {stepA1.shape}")
    stepA2 = np.rot90(stepA1, k=1, axes=(1, 2))  # (800,512,800)
    print(f"  step2 rot90: {stepA2.shape}")
    
    print("\n方案B:")
    stepB1 = np.transpose(simulated_mat, (2, 1, 0))  # (800,800,512)
    print(f"  step1 transpose(2,1,0): {stepB1.shape}")
    stepB2 = np.rot90(stepB1, k=1, axes=(1, 2))
    print(f"  step2 rot90: {stepB2.shape}")
    
except Exception as e:
    print(f"错误: {e}")
