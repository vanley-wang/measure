import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

# ================= 配置 =================
# 必须加载你刚才生成的包含 Scatt 的数据
# 这里我们直接读取你刚刚生成的汇总表（如果没保存，就重新读一遍原始数据+预测）
# 为了演示方便，假设你已经有了带标签的 Excel，或者我们重新加载模型预测
model_path = 'model/Kmeans-scatt.pickle'
scaler_path = 'model/scaler-scatt.pickle'
# 随便找一个处理好的结果文件来看中心点，或者直接读取模型中心
# 这里我们直接从模型里提取中心点，这是最准确的

# ================= 加载模型 =================
with open(model_path, 'rb') as f:
    kmeans = pickle.load(f)

# 聚类中心的坐标 (Cluster Centers)
centers = kmeans.cluster_centers_

# 特征列表 (顺序必须完全一致)
features = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD'
]

# 转为 DataFrame
df_centers = pd.DataFrame(centers, columns=features)
df_centers.index.name = 'Cluster_ID'

# ================= 绘图：聚类中心热图 =================
plt.figure(figsize=(12, 6))

# 标准化显示 (因为体积是几千，球形度是0-1，不标准化没法画在一张图上)
# 这里用 heatmap 的 z_score 参数让每一列都在内部比较 (0=平均, 红=高, 蓝=低)
sns.heatmap(df_centers, annot=True, cmap='RdBu_r', fmt='.2f', center=0, cbar=True)
# 注意：这张图显示的是“相对值”。红色代表该特征在该类中偏高，蓝色代表偏低。

plt.title('Feature Heatmap of 6 Clusters (用于判断哪些类可以合并)')
plt.ylabel('Cluster ID (类别)')
plt.xlabel('Features (特征)')
plt.tight_layout()
plt.show()

print("分析指南：")
print("1. 观察每一行（横向）。")
print("2. 如果有两行（例如 Cluster 0 和 Cluster 3）的红蓝分布模式非常像，")
print("   说明它们是同一类生物学状态，只是在数值上有轻微差别。")
print("3. 你可以在论文里把它们合并讨论。")