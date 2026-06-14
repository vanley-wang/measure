import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import pickle
import os
import glob

# 1. 基础配置
data_folders = [
    os.path.join('Data', 'FXN_2023_new', 'FXN_20230701', 'measure_excel'),
    os.path.join('Data', 'FXN_2023_new', 'FXN_20230703', 'measure_excel')
]

model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 特征列表 (共11个，含形态学 + 散射系数)
features_list = [
    'Organoids_Volume',
    'Organoids_Volume_Fill',
    'Organoids_Surface',
    'Cavity_Volume',
    'CavityNum',
    'LongAxis',
    'ShortAxis',
    'Wall_Thickness',
    'Sphericity',
    'Scatt_Mean',
    'Scatt_STD'
]

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
Data_Features = Data_All[features_list].copy()
Data_Features = Data_Features.fillna(0)  # 简单填充缺失值，防止报错

# ==========================================
# 3. 标准化 (并保存 scaler)
# ==========================================
print("\n--- 正在标准化 ---")
scaler = StandardScaler()
Data_Std = scaler.fit_transform(Data_Features)

# 保存 Scaler
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
# 【修改根据手肘图确定的最佳 K 值】
optimal_k = 6
print(f"\n--- 开始最终聚类训练 (K={optimal_k}) ---")

final_model = KMeans(n_clusters=optimal_k, init='k-means++', n_init='auto', random_state=42)
cluster_labels = final_model.fit_predict(Data_Std)

# 保存模型
model_path = os.path.join(model_dir, 'Kmeans-scatt.pickle')
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"模型已保存 -> {model_path}")

# 将标签合并回数据
Data_All['Cluster'] = cluster_labels

# ==========================================
# 6. 打印详细统计结果
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