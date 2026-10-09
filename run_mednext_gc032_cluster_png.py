"""
Generate cluster PNGs for MedNext_GC032 dataset.
Adapts src/04_clustering/cluster_png.py for the different batch naming.
"""
import sys
import os

# Add repo root and cluster_png dir to path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src', '04_clustering'))

# Patch BATCHES before importing cluster_png
import importlib.util
spec = importlib.util.spec_from_file_location('cluster_png', os.path.join(REPO_ROOT, 'src', '04_clustering', 'cluster_png.py'))
cluster_png = importlib.util.module_from_spec(spec)

# Pre-set module-level variables
cluster_png.BATCHES = ['20230701', '20230703']
cluster_png.DEFAULT_BASE = 'MedNext_GC032'

spec.loader.exec_module(cluster_png)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate cluster PNGs for MedNext_GC032')
    parser.add_argument('--base-dir', default='MedNext_GC032')
    parser.add_argument('--batch', choices=['0701', '0703', 'all'], default='all')
    args = parser.parse_args()

    base_dir = args.base_dir
    batches = ['20230701', '20230703']
    if args.batch != 'all':
        batches = [b for b in batches if b.endswith(args.batch)]

    tasks = []
    out_dir = os.path.join(base_dir, 'cluster_png')

    for batch in batches:
        root = os.path.join(base_dir, batch)
        label_dir = os.path.join(root, 'seg_label')
        merge_dir = os.path.join(root, 'cluster_merge')

        if not os.path.exists(merge_dir):
            print(f"[WARN] 跳过 {batch}: 无 cluster_merge 目录")
            continue
        if not os.path.exists(label_dir):
            print(f"[WARN] 跳过 {batch}: 无 seg_label 目录")
            continue

        merge_files = sorted([f for f in os.listdir(merge_dir) if f.endswith('_merge.xlsx')])
        print(f"\n>>> {batch}: 找到 {len(merge_files)} 个 merge 文件")

        for mf in merge_files:
            well_name = mf.replace('_merge.xlsx', '')
            label_path = os.path.join(label_dir, f"{well_name}_label.mat")
            merge_path = os.path.join(merge_dir, mf)

            if not os.path.exists(label_path):
                print(f"  [WARN] 缺失标签文件: {label_path}")
                continue

            tasks.append((well_name, label_path, merge_path, out_dir))

    if not tasks:
        print("[WARN] 未找到任何可处理任务")
        return

    from tqdm import tqdm
    print(f"\n>>> 共 {len(tasks)} 个孔位待处理，开始生成 cluster_png...")
    print(f">>> 输出目录: {out_dir}")

    success = 0
    warnings = 0
    errors = 0
    for task in tqdm(tasks, desc='cluster_png'):
        result = cluster_png.process_one_well(*task)
        if "[ERR]" in result:
            errors += 1
            print(result)
        elif "[WARN]" in result:
            warnings += 1
            print(result)
        else:
            success += 1

    print(f"\n[Done] 完成！成功: {success}, 警告: {warnings}, 错误: {errors}")
    print(f"输出目录: {out_dir}")


if __name__ == '__main__':
    main()
