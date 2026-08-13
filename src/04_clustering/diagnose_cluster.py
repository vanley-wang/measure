import pickle
import pandas as pd
import os
import glob

# ================= 诊断脚本 =================

print("=" * 60)
print("诊断 1: 检查模型包里的映射")
print("=" * 60)

with open('model/Kmeans-scatt.pickle', 'rb') as f:
    pkg = pickle.load(f)

print(f"模型类型: {type(pkg)}")
if isinstance(pkg, dict):
    print(f"Keys: {pkg.keys()}")
    print(f"raw_to_final 映射: {pkg.get('raw_to_final')}")
    print(f"feature_names: {pkg.get('feature_names')}")
    km = pkg.get('kmeans')
    if km:
        print(f"kmeans n_clusters: {km.n_clusters}")
        print(f"cluster_centers_.shape: {km.cluster_centers_.shape}")

print("\n" + "=" * 60)
print("诊断 2: 抽查 merge 文件 Cluster 分布")
print("=" * 60)

root = 'Data/nnUNet_FXN_2023/FXN_0701'
merge_dir = os.path.join(root, 'cluster_merge')
files = sorted(glob.glob(os.path.join(merge_dir, '*_merge.xlsx')))

if not files:
    print("未找到 merge 文件！")
else:
    # 打印前 3 个文件的分布
    for fp in files[:3]:
        df = pd.read_excel(fp)
        fname = os.path.basename(fp)
        print(f"\n{fname}:")
        if 'Cluster' not in df.columns:
            print("  [错误] 缺少 Cluster 列")
            continue
        print(df['Cluster'].value_counts().sort_index().to_string())
        # 显示体积最大的 5 个类器官的 Cluster
        top5 = df.nlargest(5, 'Organoids_Volume_Fill')[['Index', 'Organoids_Volume_Fill', 'Cavity_Volume', 'Scatt_Mean', 'Cluster']]
        print("  体积最大的5个:")
        print(top5.to_string(index=False))

print("\n" + "=" * 60)
print("诊断 3: 检查是否有文件被 cluster-merge.py 跳过")
print("=" * 60)
# 对比 measure_excel 和 cluster_merge 的文件时间戳
measure_dir = os.path.join(root, 'measure_excel')
measure_files = {os.path.basename(f).replace('.xlsx', ''): os.path.getmtime(f)
                 for f in glob.glob(os.path.join(measure_dir, '*.xlsx'))}
merge_files = {os.path.basename(f).replace('_merge.xlsx', ''): os.path.getmtime(f)
               for f in glob.glob(os.path.join(merge_dir, '*_merge.xlsx'))}

stale = []
for well, mtime in measure_files.items():
    if well not in merge_files:
        stale.append((well, '缺失'))
    elif merge_files[well] < mtime:
        stale.append((well, '过期'))

if stale:
    print(f"发现 {len(stale)} 个孔位需要重新生成 merge:")
    for w, reason in stale[:10]:
        print(f"  {w}: {reason}")
else:
    print("所有 merge 文件都是最新的")
