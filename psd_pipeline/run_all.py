# ============================================================
# PSD-Pipeline 一键运行脚本
# 按顺序执行所有步骤
# ============================================================

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def run_step(step_name, script_name):
    print("\n" + "=" * 70)
    print(f"  >>> 执行: {step_name} <<<")
    print("=" * 70)

    script_path = os.path.join(BASE_DIR, script_name)
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        print(f"\n!!! 步骤失败: {step_name} (exit code: {result.returncode})")
        sys.exit(1)

    print(f"\n>>> {step_name} 完成 <<<")


def main():
    print("=" * 70)
    print("PSD-Pipeline: 表型分层时间差特征分析流水线")
    print("=" * 70)

    steps = [
        ("步骤4: 基线聚类器训练",           "psd_baseline_cluster.py"),
        ("步骤5: 跨时间点表型匹配",          "psd_cross_time_classify.py"),
        ("步骤6: 分表型Delta特征计算",       "psd_delta_features.py"),
        ("步骤7: PCA降维+ATP验证",           "psd_pca_analysis.py"),
    ]

    for step_name, script_name in steps:
        run_step(step_name, script_name)

    print("\n" + "=" * 70)
    print("  全部步骤完成！")
    print(f"  输出目录: {os.path.join(BASE_DIR, 'output')}")
    print(f"  模型目录: {os.path.join(BASE_DIR, 'model')}")
    print("=" * 70)


if __name__ == '__main__':
    main()