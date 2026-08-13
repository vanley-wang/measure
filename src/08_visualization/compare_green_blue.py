import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 读取一个药物处理组的 merge 文件（蓝类会比较多）
fp = 'Data/nnUNet_FXN_2023/FXN_0701/cluster_merge/B9_0701_merge.xlsx'
df = pd.read_excel(fp)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Volume_Fill 分布
for cid, color in [(2, 'green'), (3, 'blue')]:
    subset = df[df['Cluster'] == cid]['Organoids_Volume_Fill']
    axes[0].hist(np.log1p(subset), bins=30, alpha=0.6, color=color,
                 label=f'Cluster {cid} (n={len(subset)})')
axes[0].set_xlabel('log1p(Volume_Fill)')
axes[0].set_ylabel('Count')
axes[0].set_title('Volume Distribution: Green vs Blue')
axes[0].legend()

# 2. OAC 分布
for cid, color in [(2, 'green'), (3, 'blue')]:
    subset = df[df['Cluster'] == cid]['Scatt_Mean']
    axes[1].hist(subset, bins=30, alpha=0.6, color=color,
                 label=f'Cluster {cid} (n={len(subset)})')
axes[1].set_xlabel('Scatt_Mean (OAC)')
axes[1].set_ylabel('Count')
axes[1].set_title('OAC Distribution: Green vs Blue')
axes[1].legend()

# 3. 散点图：Volume vs OAC
for cid, color, label in [(2, 'green', 'Green/Small'), (3, 'blue', 'Blue/Damaged')]:
    subset = df[df['Cluster'] == cid]
    axes[2].scatter(subset['Organoids_Volume_Fill'], subset['Scatt_Mean'],
                    c=color, s=20, alpha=0.6, label=label)
axes[2].set_xlabel('Volume_Fill')
axes[2].set_ylabel('Scatt_Mean (OAC)')
axes[2].set_title('Volume vs OAC: Green vs Blue')
axes[2].set_xscale('log')
axes[2].legend()

plt.tight_layout()
plt.savefig('green_blue_comparison.png', dpi=200)
print("Saved to green_blue_comparison.png")

# 打印统计
print("\n=== Cluster 2 (Green) vs Cluster 3 (Blue) Stats ===")
summary = df[df['Cluster'].isin([2, 3])].groupby('Cluster')[['Organoids_Volume_Fill', 'Scatt_Mean']].agg(['mean', 'median', 'min', 'max'])
print(summary)
