import pandas as pd
import os
import glob
import pickle

# ================= 1. 配置路径 =================
# 模型包路径 (含 kmeans + scaler + raw_to_final 映射)
model_path = 'model/Kmeans-scatt.pickle'

# 要处理的数据根目录
root_folders = [
    'Data/nnUNet_FXN_2023/FXN_0701',
    'Data/nnUNet_FXN_2023/FXN_0703'
]

# 输出文件夹名称
output_folder_name = 'cluster_merge'

# 对应的中文说明 (与论文四类表型一致)
# 0=红/大囊状, 1=黄/大实心, 2=绿/小实心, 3=蓝/高致密受损
label_desc = {
    0: '大囊状健康类器官',
    1: '大实心健康类器官',
    2: '小实心休眠/幼类器官',
    3: '极小高致密受损类器官',
}

# ================= 2. 执行批处理 =================
print("正在加载模型包...")
with open(model_path, 'rb') as f:
    model_pkg = pickle.load(f)

kmeans = model_pkg['kmeans']
scaler = model_pkg['scaler']
raw_to_final = model_pkg['raw_to_final']
features = model_pkg['feature_names']

for root in root_folders:
    input_dir = os.path.join(root, 'measure_excel')
    output_dir = os.path.join(root, output_folder_name)

    if not os.path.exists(input_dir):
        print(f"跳过: 找不到 {input_dir}")
        continue
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(input_dir, '*.xlsx'))
    print(f"\n>>> 处理文件夹: {os.path.basename(root)} (共 {len(files)} 个文件) <<<")

    for file_path in files:
        try:
            # 1. 读取
            df = pd.read_excel(file_path)

            if not all(col in df.columns for col in features):
                print(f"  [跳过] 缺少特征列: {os.path.basename(file_path)}")
                continue

            # 2. 预测原始分类 (得到 0~3)
            X = df[features]
            X_std = scaler.transform(X)  # 必须使用 transform
            raw_labels = kmeans.predict(X_std)

            # 3. 应用论文标准表型映射 (根据质心特征自动判定)
            final_labels = [raw_to_final[l] for l in raw_labels]

            # 4. 写入结果
            df['Cluster'] = final_labels
            df['Phenotype_Desc'] = [label_desc[mid] for mid in final_labels]

            # 5. 保存
            file_name = os.path.basename(file_path)
            save_name = file_name.replace('.xlsx', '_merge.xlsx')
            save_path = os.path.join(output_dir, save_name)

            df.to_excel(save_path, index=False)

        except Exception as e:
            print(f"  [错误] {os.path.basename(file_path)}: {e}")

    print(f"  -> 结果已保存至: {output_dir}")

print("\n全部完成！Cluster 列已按论文标准四类表型赋值 (0=红/大囊状, 1=黄/大实心, 2=绿/小实心, 3=蓝/高致密受损)。")