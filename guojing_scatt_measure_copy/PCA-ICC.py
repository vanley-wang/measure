import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import os
import joblib

# ================= 1. ATP 数据 =================
# # GC
# atp_database = {
#      # 'C10': 54230,
#     'C11': 24330, 'C12': 71930,
#     # 'C1': 2961000, 'C2': 6731000,
#     'C3': 8638000,
#     'C4': 7800000, 'C5': 7260000,
#     # 'C6': 6671000,
#     # 'C7': 77960,
#     'C8': 185300,
#     # 'C9': 46930,
#     # 'D1': 45350, 'D2': 133600, 'D3': 28820,
#     'D4': 66260, 'D5': 27630, 'D6': 25920,
#     # 'E1': 32980, 'E2': 51460, 'E3': 69330,
#     'E4': 63070, 'E5': 15450, 'E6': 68250,
#     # 'F10': 56050000,
#     'F11': 55300000, 'F12': 46400000,
#     # 'F1': 12407000,
#     'F2': 14970000, 'F3': 14070000,
#     'F4': 10820000,
#     # 'F5': 6017000, 'F6': 12140000,
#     'F7': 54140000, 'F8': 50680000, 'F9': 65380000
# }

# ICC
atp_database = {
    'B10': 601300, 'B11': 11180000, 'B2': 5391000, 'B3': 6538000,
    'B4': 7103000, 'B5': 1264000, 'B6': 2548000, 'B7': 1579000,
    'B8': 3637000, 'B9': 140300, 'C10': 211800, 'C11': 13930000,
    'C2': 4460000, 'C3': 8336000, 'C4': 6800000, 'C5': 330900,
    'C6': 238900, 'C7': 682100, 'C8': 211300, 'C9': 465900,
    'D11': 11240000, 'E11': 12840000, 'F10': 21910000, 'F11': 14700000,
    'F2': 26980000, 'F3': 14110000, 'F4': 13740000, 'F5': 17250000,
    'F6': 20320000, 'F7': 20000000, 'F8': 17170000, 'F9': 15830000
}

# ================= 2. 读取数据 =================
input_path = 'Data/FXN_2023_new/FXN_2023_Analysis.xlsx'
output_path = 'Data/FXN_2023_new/FXN_2023_PCA_Result.xlsx'

print("读取数据...")
Data_All = pd.read_excel(input_path)
if 'Name' not in Data_All.columns: Data_All['Name'] = Data_All.index.astype(str)

# ================= 3. 特征选择 (在此处粘贴自动筛选出的列表) =================
# 举例：粘贴你筛选出的 Top 1 特征
features_list = [
    # 'Cyst_Thick_Avg_4', 'Long_Axis_Avg_2', 'Long_Axis_Avg_3', 'Number_1', 'Number_2',
    # 'Roughness_All', 'Surface_Avg_1', 'Volume_Fill_Avg_1', 'Volume_Fill_Avg_2'
]

# 【关键步骤】强制按字母排序
features_list.sort()
print(f"已强制排序特征，共 {len(features_list)} 个")

existing_cols = [col for col in features_list if col in Data_All.columns]
Data = Data_All[existing_cols].fillna(0)

# ================= 4. PCA 计算 (由排序保证一致性) =================
print("计算 PCA...")
scaler = StandardScaler()
Data_std = scaler.fit_transform(Data)

# 使用 full solver 和固定种子
pca = PCA(n_components=4, svd_solver='full', random_state=42)
Data_Pca = pca.fit_transform(Data_std)

variance_ratio = pca.explained_variance_ratio_
weights = variance_ratio / np.sum(variance_ratio)
Result_Score = np.dot(Data_Pca, weights)

# 写入结果
Data_All['Result'] = Result_Score
for i in range(4):
    Data_All[f'PC{i + 1}'] = Data_Pca[:, i]

# ================= 5. 计算相关性 =================
# 解析 ATP
Data_All['Well_ID'] = Data_All['Name'].apply(lambda x: x.split('_')[0])
Data_All['TimePoint'] = Data_All['Name'].apply(lambda x: x.split('_')[1])
Data_All['ATP'] = Data_All['Well_ID'].map(atp_database)

# 配对计算
df_0701 = Data_All[Data_All['TimePoint'] == '0701'][['Well_ID', 'Result']].set_index('Well_ID')
df_0703 = Data_All[Data_All['TimePoint'] == '0703'][['Well_ID', 'Result', 'ATP']].set_index('Well_ID')
merged = df_0703.join(df_0701, on='Well_ID', lsuffix='_End', rsuffix='_Start', how='inner')

merged['Result_Diff'] = merged['Result_End'] - merged['Result_Start']
valid = merged.dropna(subset=['Result_Diff', 'ATP'])

if len(valid) > 2:
    r_diff, _ = pearsonr(valid['Result_Diff'], valid['ATP'])
    r_end, _ = pearsonr(valid['Result_End'], valid['ATP'])

    print(f"\n>>> 最终验证结果 <<<")
    print(f"差值相关性 (Diff vs ATP): {abs(r_diff):.6f}")  # 取绝对值对比
    print(f"终点相关性 (End vs ATP):  {abs(r_end):.6f}")
else:
    print("数据不足")

# 保存
Output_Df = Data_All[['Name', 'PC1', 'PC2', 'PC3', 'PC4', 'Result', 'ATP']].copy()
Output_Df['差值相关'] = np.nan
Output_Df.loc[0, '差值相关'] = r_diff
Output_Df.loc[0, '终点相关'] = r_end
Output_Df.to_excel(output_path, index=False)
print(f"结果已保存: {output_path}")

# ==========================================
# 6. 保存模型参数
# ==========================================
print("\n>>> 正在保存模型架构...")

# 1. 定义文件夹和路径
current_dir = os.getcwd()  # 获取当前工作目录
model_dir = os.path.join(current_dir, 'PCA_model')  # 目标文件夹名为 PCA_model

# 如果文件夹不存在，则创建它
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"已新建文件夹: {model_dir}")

# 定义保存文件的完整路径
model_filename = 'organoid_pca_scoring.pkl'
save_path = os.path.join(model_dir, model_filename)

# 2. 打包关键对象
model_package = {
    'features_list': features_list,  # 核心：锁死特征的顺序
    'scaler': scaler,  # 核心：锁死均值和方差标准
    'pca': pca,  # 核心：锁死投影方向
    'weights': weights,  # 核心：锁死权重计算公式

    # 以下是可选的备注信息，方便以后查看
    'description': '基于多维特征的类器官生长评价模型',
    'variance_ratio': pca.explained_variance_ratio_  # 记录当时的解释度
}

# 3. 保存文件
# joblib.dump(model_package, save_path)

print(f"模型已成功保存至: {save_path}")
print("下次验证新数据时，直接读取该 .pkl 文件即可，无需重新训练。")