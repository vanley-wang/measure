import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import pickle
import os
import glob

from cluster_utils import (
    RAW_FEATURES, REDUCED_RAW_FEATURES,
    RECONSTRUCTED_PROCESSED_FEATURES, SIZE_FEATURES,
    Preprocessor,
)

# 1. 基础配置
parser = argparse.ArgumentParser(description='Train KMeans for organoid clustering')
parser.add_argument('--reduced', action='store_true', help='Use 5-dim reduced feature set')
parser.add_argument('--reconstructed', action='store_true', help='Use 6-dim PCA-reconstructed feature set')
args = parser.parse_args()

data_folders = [
    os.path.join('Data', 'FXN_2023_new（ICC）', 'FXN_20230701', 'measure_excel'),
    os.path.join('Data', 'FXN_2023_new（ICC）', 'FXN_20230703', 'measure_excel')
]

model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if args.reconstructed:
    features_list = RAW_FEATURES  # Preprocessor 需要原始特征来计算 PCA
    preprocessor_mode = 'reconstructed'
    print("【使用 6维 PCA 重构特征集】")
    print(f"  组 A (大小): {SIZE_FEATURES} → log1p → PCA → Size_PC1")
    print(f"  组 B (散射): Scatt_Mean, Scatt_STD → 保持原值")
    print(f"  组 C (形态): Cavity_Ratio, CavityNum, Sphericity")
    print(f"  最终特征: {RECONSTRUCTED_PROCESSED_FEATURES}")
elif args.reduced:
    features_list = REDUCED_RAW_FEATURES
    preprocessor_mode = 'reduced'
    print("【使用 5维精简特征集】")
else:
    features_list = RAW_FEATURES
    preprocessor_mode = 'full'
    print("【使用 10维完整特征集】")

# 设置 Pandas 显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 设置 Pandas 显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# ==========================================
# 2. 数据读取与合并
# ==========================================
print("--- 正在读取数据 ---")
df_list = []
for folder in data_folders:
    if not os.path.exists(folder):
        print(f"路径跳过 (不存在): {folder}")
        continue

    files = glob.glob(os.path.join(folder, '*.xlsx'))
    print(f"目录 {os.path.basename(folder)}: 发现 {len(files)} 个文件")

    for file in files:
        try:
            temp_df = pd.read_excel(file)
            df_list.append(temp_df)
        except Exception as e:
            print(f"读取失败: {file}")

if not df_list:
    raise ValueError("未读取到数据，请检查路径!")

Data_All = pd.concat(df_list, ignore_index=True)

before = len(Data_All)
Data_All = Data_All.dropna(subset=features_list)
after = len(Data_All)
if before != after:
    print(f"  已剔除 {before - after} 个含 NaN 的样本 (剩余 {after})")

Data_Features = Data_All[features_list].copy()

# ==========================================
# 3. 标准化 (使用 Preprocessor，支持 full/reduced/reconstructed)
# ==========================================
print("\n--- 正在标准化 ---")
if args.reduced or args.reconstructed:
    preprocessor = Preprocessor(mode=preprocessor_mode)
    Data_Std = preprocessor.fit_transform(Data_Features)
    scaler = preprocessor.scaler
    print(f"Preprocessor ({preprocessor_mode}) 已拟合，输出维度: {Data_Std.shape}")
    if args.reconstructed:
        pca_info = preprocessor.get_size_pca_info()
        print(f"  Size PCA 方差解释率: {pca_info['explained_variance_ratio']:.3f} "
              f"({pca_info['explained_variance_ratio']*100:.1f}%)")
        print(f"  Size PCA 载荷: {pca_info['components']}")
else:
    scaler = StandardScaler()
    Data_Std = scaler.fit_transform(Data_Features)
    scaler_path = os.path.join(model_dir, 'scaler-scatt.pickle')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"StandardScaler 已保存 -> {scaler_path}")

# ==========================================
# 4. 手肘法分析 (可视化 - 期刊风格 & 保存)
# ==========================================
print("\n--- 正在生成手肘图 (计算 K=1 到 K=8) ---")
meanDispersions = []
K_range = range(1, 9)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans_temp.fit(Data_Std)
    # 计算平均离差
    m_Disp = sum(np.min(cdist(Data_Std, kmeans_temp.cluster_centers_, 'euclidean'), axis=1)) / Data_Std.shape[0]
    meanDispersions.append(m_Disp)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(7, 5))

# 绘制折线
plt.plot(K_range, meanDispersions, marker='o', linestyle='-', color='black',
         linewidth=1.5, markersize=7, markerfacecolor='white', markeredgewidth=1.5)

# 标签设置
plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('Mean Dispersion (Euclidean)', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold', pad=15)

# 去除顶部和右侧边框 (Clean Style)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 加粗坐标轴线
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.tick_params(width=1.2, labelsize=10)

plt.tight_layout()

# --- 保存图片
elbow_plot_path = os.path.join(model_dir, 'Elbow_Method_Plot.png')
plt.savefig(elbow_plot_path, dpi=600, bbox_inches='tight')
print(f"手肘图已保存 -> {elbow_plot_path}")

plt.show()

# ==========================================
# 5. 最终模型训练与结果打印
# ==========================================
# 【根据手肘图与论文生物学可解释性确定最佳 K 值 = 4】
optimal_k = 4
print(f"\n--- 开始最终聚类训练 (K={optimal_k}, K-means++) ---")

final_model = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)
cluster_labels = final_model.fit_predict(Data_Std)

# ==========================================
# 6. 根据质心特征自动判定四类表型 (论文 4.2.2)
# ==========================================
print("\n--- 正在根据质心特征判定四类表型 ---")
Data_All['RawCluster'] = cluster_labels

# 计算每簇核心指标均值（在原始特征空间更直观）
cluster_stats = Data_All.groupby('RawCluster')[['Organoids_Volume_Fill', 'Cavity_Volume', 'Scatt_Mean']].mean()

print("\n【Raw Cluster 质心统计】")
print(cluster_stats)
print("-" * 50)

raw_ids = list(range(optimal_k))

# Cluster 4 (蓝色/极小高致密受损): OAC (Scatt_Mean) 显著最高
raw_c4 = cluster_stats['Scatt_Mean'].idxmax()

# 论文标准判定逻辑：
# Cluster 1 (红/大囊状) 核心特征 = 填充体积最大，空腔多为伴随现象
# Cluster 2 (黄/大实心) = 体积次大
# Cluster 3 (绿/小实心) = 体积最小
remaining = [c for c in raw_ids if c != raw_c4]
# 按填充体积从大到小排序
remaining_sorted = cluster_stats.loc[remaining, 'Organoids_Volume_Fill'].sort_values(ascending=False)
raw_c1 = remaining_sorted.index[0]   # 红：体积最大
raw_c2 = remaining_sorted.index[1]   # 黄：体积次大
raw_c3 = remaining_sorted.index[2]   # 绿：体积最小

# 映射到论文标准编号 (0~3，与可视化脚本一致)
# 0=大囊状(红), 1=大实心(黄), 2=小实心(绿), 3=高致密受损(蓝)
raw_to_final = {
    raw_c1: 0,
    raw_c2: 1,
    raw_c3: 2,
    raw_c4: 3,
}

Data_All['Cluster'] = Data_All['RawCluster'].map(raw_to_final)

print(f"原始类别 {raw_c1} -> 大囊状(红, 0)   [体积最大]")
print(f"原始类别 {raw_c2} -> 大实心(黄, 1)   [体积次大]")
print(f"原始类别 {raw_c3} -> 小实心(绿, 2)   [体积最小]")
print(f"原始类别 {raw_c4} -> 高致密受损(蓝, 3) [OAC最高]")

# ==========================================
# 7. 保存模型包 (含 scaler/preprocessor + 类别映射)
# ==========================================
model_package = {
    'kmeans': final_model,
    'scaler': scaler,
    'raw_to_final': raw_to_final,
    'feature_names': features_list,
}
if args.reduced:
    model_package['preprocessor'] = preprocessor
    model_path = os.path.join(model_dir, 'Kmeans-5d.pickle')
elif args.reconstructed:
    model_package['preprocessor'] = preprocessor
    model_package['feature_names'] = RECONSTRUCTED_PROCESSED_FEATURES
    model_path = os.path.join(model_dir, 'Kmeans-reconstructed.pickle')
else:
    model_path = os.path.join(model_dir, 'Kmeans-scatt.pickle')
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)
print(f"模型包已保存 -> {model_path}")

# ==========================================
# 8. 打印详细统计结果
# ==========================================
print("\n" + "=" * 50)
print(f"聚类结果统计 (共 {optimal_k} 类)")
print("=" * 50)

# 计算每个类在所有特征上的均值
summary_mean = Data_All.groupby('Cluster')[features_list].mean()

# 打印完整表格
print(summary_mean)

# 保存 Excel
summary_path = os.path.join(model_dir, 'Cluster_scatt.xlsx')
summary_mean.to_excel(summary_path)
print(f"\n[提示] 统计表格已额外保存至: {summary_path}")