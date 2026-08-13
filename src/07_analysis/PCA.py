import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import re

# ================= 1. 初始化与配置 =================
# 您原来的文件路径
input_path = 'Data/nnUNet_FXN_2023/nnUNet_Analysis.xlsx'
# 新增：保存所有模型参数的 Excel 路径
details_save_path = 'Data/nnUNet_FXN_2023/PCA_Model_Details.xlsx'

# 确保输出目录存在
os.makedirs(os.path.dirname(details_save_path), exist_ok=True)

# ================= 2. 读取数据 (保留原逻辑) =================
print(f"正在读取数据: {input_path} ...")
# 为了演示，如果文件不存在，我这里生成模拟数据。您运行时请直接用您的 pd.read_excel
if os.path.exists(input_path):
    Data_All = pd.read_excel(input_path)
else:
    print("警告: 未找到文件，生成模拟数据用于演示...")
    Data_All = pd.DataFrame(np.random.rand(100, 20), columns=[
        'Cavity_Volume_All_1', 'Cyst_Thick_Avg_3', 'Long_Axis_Avg_4', 'Number_2', 'Number_3',
        'Roughness_All', 'Scatt_Mean_Avg_4', 'Short_Axis_Avg_2', 'Short_Axis_Avg_3',
        'Short_Axis_Avg_4', 'Surface_Avg_1', 'Surface_Avg_2', 'Surface_Avg_3', 'Volume_Fill_Avg_2',
        'Extra_Col1', 'Extra_Col2', 'Name', 'TimePoint', 'Well_ID', 'ATP'
    ])

# ================= 3. 特征定义与排序 (关键修改：加入分组逻辑) =================
features_list = [
    'Cavity_Volume_All_1', 'Cyst_Thick_Avg_3', 'Long_Axis_Avg_4', 'Number_2', 'Number_3',
    'Roughness_All',
    'Scatt_Mean_Avg_3',
    'Scatt_Mean_Avg_4',
    'Short_Axis_Avg_2',
    'Short_Axis_Avg_3',
    'Short_Axis_Avg_4',
    'Surface_Avg_1', 'Surface_Avg_2', 'Surface_Avg_3', 'Volume_Fill_Avg_2'
]


# --- 定义排序与分组函数 ---
def get_feature_sort_key(feature_name):
    """
    排序规则：
    1. Roughness 强制归为第1组最后
    2. 其他按末尾数字归组 (e.g. _1 -> Group 1)
    """
    if 'Roughness' in feature_name:
        return (1, 999, feature_name)  # Group 1, Priority Last
    match = re.search(r'_(\d+)$', feature_name)
    if match:
        return (int(match.group(1)), 0, feature_name)
    return (99, 0, feature_name)


# 执行排序
features_list.sort(key=get_feature_sort_key)
print(f"特征已排序，共 {len(features_list)} 个")

# ================= 4. PCA 计算 (保留原逻辑) =================
print("正在执行 PCA 计算...")
# 提取数据
existing_cols = [c for c in features_list if c in Data_All.columns]
print(f"实际可用特征: {len(existing_cols)} / {len(features_list)}")
if len(existing_cols) != len(features_list):
    missing = [c for c in features_list if c not in Data_All.columns]
    print(f"缺失特征: {missing}")
Data = Data_All[existing_cols].fillna(0)

# 标准化
scaler = StandardScaler()
Data_std = scaler.fit_transform(Data)

# PCA
n_components = 4
pca = PCA(n_components=n_components, svd_solver='full', random_state=42)
Data_Pca = pca.fit_transform(Data_std)

# 权重计算
variance_ratio = pca.explained_variance_ratio_
weights = variance_ratio / np.sum(variance_ratio)

# ================= 5. 数据整理与保存 (核心新增部分) =================
print(f"正在生成详细数据表: {details_save_path} ...")

# --- A. 准备特征分组表 (X_ij) ---
group_data = []
current_group = None
g_idx = 0
for feat in existing_cols:
    key = get_feature_sort_key(feat)
    group_num = key[0]

    if group_num != current_group:
        current_group = group_num
        g_idx = 1
    else:
        g_idx += 1

    symbol = f"X_{{{group_num}{g_idx}}}"  # LaTeX格式
    symbol_simple = f"X_{group_num}{g_idx}"  # 简单格式

    group_data.append({
        'Feature_Name': feat,
        'Group_ID': group_num,
        'Symbol_LaTeX': symbol,
        'Symbol_Simple': symbol_simple
    })
df_groups = pd.DataFrame(group_data)

# --- B. 准备特征根与贡献率表 ---
pc_names = [f'PC{i + 1}' for i in range(n_components)]
df_eigen = pd.DataFrame({
    'Principal_Component': pc_names,
    'Eigenvalue (特征根)': pca.explained_variance_,
    'Variance_Ratio (方差贡献率)': pca.explained_variance_ratio_,
    'Cumulative_Ratio (累积贡献率)': np.cumsum(pca.explained_variance_ratio_),
    'Weight_in_Score (综合评分权重)': weights
})

# --- C. 准备载荷矩阵表 (Loadings) ---
# 行是特征，列是PC
df_loadings = pd.DataFrame(
    pca.components_.T,
    index=existing_cols,
    columns=[f'Loading_{pc}' for pc in pc_names]
)
# 合并符号信息，方便查看
df_loadings = pd.concat([df_groups.set_index('Feature_Name')[['Symbol_Simple']], df_loadings], axis=1)

# --- D. 准备最终评分公式系数表 ---
# Final_Coef = Loadings * Weights
# 也就是每个原始特征(标准化后)在最终 Score 中的系数
final_coefs = np.dot(pca.components_.T, weights)
df_final_coef = pd.DataFrame({
    'Feature_Name': existing_cols,
    'Symbol': df_groups['Symbol_Simple'],
    'Coefficient (最终系数)': final_coefs,
    'Abs_Coefficient (系数绝对值)': np.abs(final_coefs)
})
# 按绝对值大小排序，方便看谁最重要
df_final_coef = df_final_coef.sort_values(by='Abs_Coefficient (系数绝对值)', ascending=False)

# ================= 6. 写入 Excel (多 Sheet 排版) =================
with pd.ExcelWriter(details_save_path, engine='openpyxl') as writer:
    # 1. 特征根信息
    df_eigen.to_excel(writer, sheet_name='1_Eigenvalues', index=False, float_format="%.4f")

    # 2. 载荷矩阵
    df_loadings.to_excel(writer, sheet_name='2_Loadings_Matrix', float_format="%.4f")

    # 3. 最终系数
    df_final_coef.to_excel(writer, sheet_name='3_Final_Coefficients', index=False, float_format="%.4f")

    # 4. 特征分组元数据
    df_groups.to_excel(writer, sheet_name='4_Feature_Meta', index=False)

print("-" * 50)
print(f"所有数据已成功保存至:\n{os.path.abspath(details_save_path)}")
print("-" * 50)
print("包含以下 Sheet:")
print("1. 1_Eigenvalues      -> 特征根、方差贡献率、用于计算Score的权重")
print("2. 2_Loadings_Matrix  -> 载荷矩阵 (Table 5)")
print("3. 3_Final_Coefficients -> 原始特征在综合评分中的权重 (按重要性排序)")
print("4. 4_Feature_Meta     -> 特征分组与数学符号映射 (X_ij)")