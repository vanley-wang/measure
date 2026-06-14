import os
import pandas as pd
from pathlib import Path
import pickle  # 确保导入 pickle

# ================= 修改点 1: 加载新模型和标准化器 =================
# 加载你刚刚训练好的 K=6 的模型
# model_path = 'model/Kmeans-form.pickle'
model_path = 'model/Kmeans-scatt.pickle'

# 【重要】必须同时加载训练时保存的 scaler，保证评价标准一致
# scaler_path = 'model/scaler-form.pickle'
scaler_path = 'model/scaler-scatt.pickle'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
# =============================================================

# 多文件夹路径 (保持你原来的不变)
root_folders = [
    'Data/FXN_2023_new/FXN_20230701',
    'Data/FXN_2023_new/FXN_20230703', # 如果有第二个文件夹就取消注释
]

for root in root_folders:
    measure_folder = os.path.join(root, 'measure_excel')
    save_folder = os.path.join(root, 'cluster_excel')
    os.makedirs(save_folder, exist_ok=True)

    # 获取所有 Excel 文件
    excel_files = [f for f in os.listdir(measure_folder) if f.endswith('.xlsx')]

    for excel_file in excel_files:
        excel_path = os.path.join(measure_folder, excel_file)
        df = pd.read_excel(excel_path)

        # ================= 修改点 2: 更新特征列表 =================
        # 必须与训练时的 11 个特征完全一致，顺序也不能乱
        required_columns = [
            'Organoids_Volume',
            'Organoids_Volume_Fill',
            'Organoids_Surface',
            'Cavity_Volume',
            'CavityNum',
            'LongAxis',
            'ShortAxis',
            'Wall_Thickness',
            'Sphericity',
            'Scatt_Mean',  # 新增
            'Scatt_STD'  # 新增
        ]
        # ========================================================

        # 提取数据
        X = df[required_columns]

        # ================= 修改点 3: 使用保存的 scaler 进行转换 =================
        # ❌ 原代码：X_std = StandardScaler().fit_transform(X)  <--这是错的，不要用 fit
        # ✅ 新代码：直接用加载的 scaler 转换
        X_std = scaler.transform(X)
        # ======================================================================

        # 预测
        clusters = model.predict(X_std)

        # 添加聚类列
        df['Cluster'] = clusters

        # 保存到新文件夹
        stem = Path(excel_file).stem
        new_name = f"{stem}_cluster.xlsx"
        save_path = os.path.join(save_folder, new_name)

        df.to_excel(save_path, index=False)
        print(f"{excel_file} 聚类完成并保存至: {new_name}")