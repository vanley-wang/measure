import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

# 聚类模型路径
model_path = 'model/Recover-Km5.pickle'
# model_path = 'model/Kmeans-5.pickle'
with open(model_path, 'rb') as f:
    model = pickle.load(f)


root_folders = [
    'Data/nnUNet_FXN_2023/FXN_0701',
    'Data/nnUNet_FXN_2023/FXN_0703'
]
# 每个文件夹下的 measure 文件夹
for root in root_folders:
    measure_folder = os.path.join(root, 'measure_excel')
    save_folder = os.path.join(root, 'cluster_merge')
    os.makedirs(save_folder, exist_ok=True)

    # 获取所有 Excel 文件
    excel_files = [
        f for f in os.listdir(measure_folder)
        if f.endswith('.xlsx')
    ]

    for excel_file in excel_files:
        excel_path = os.path.join(measure_folder, excel_file)
        df = pd.read_excel(excel_path)

        required_columns = [
            'Organoids_Volume',
            'Organoids_Volume_Fill',
            'Organoids_Surface',
            'Cavity_Volume',
            'CavityNum',
            'LongAxis',
            'ShortAxis',
            'Wall_Thickness',
            'Sphericity'
        ]

        # 提取数据 & 聚类
        X = df[required_columns]
        X_std = StandardScaler().fit_transform(X)
        clusters = model.predict(X_std)

        # 添加聚类列
        df['Cluster'] = clusters

        # 保存到新文件夹
        stem =Path(excel_file).stem
        new_name = f"{stem}_cluster.xlsx"
        save_path = os.path.join(save_folder, new_name)
        df.to_excel(save_path, index=False)
        print(f"{excel_file} 聚类完成并保存至: {new_name}")
