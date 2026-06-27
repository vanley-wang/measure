import pandas as pd
import os
import glob
import pickle

# ================= 1. 配置路径 =================
# 模型与标准化器路径
model_path = 'model/Kmeans-scatt.pickle'
scaler_path = 'model/scaler-scatt.pickle'

# 要处理的数据根目录
root_folders = [
    'Data/nnUNet_FXN_2023/FXN_0701',
    'Data/nnUNet_FXN_2023/FXN_0703'
]

# 输出文件夹名称
output_folder_name = 'cluster_merge'

# ================= 2. 定义数值合并规则 =================

numeric_map = {
    # --- 大 ---
    3: 0,
    5: 0,
    # --- 中 ---
    1: 1,
    # --- 小  ---
    0: 2,
    # 1: 2,
    4: 2,
    # --- 高散射 ---
    2: 3,
}

# 对应的中文说明
label_desc = {
    0: '巨大囊泡型',
    1: '中等过渡型',
    2: '小体积基准型',
    3: '高散射实心型'
}

# 特征列表
features = [
    'Organoids_Volume', 'Organoids_Volume_Fill', 'Organoids_Surface',
    'Cavity_Volume', 'CavityNum', 'LongAxis', 'ShortAxis',
    'Wall_Thickness', 'Sphericity', 'Scatt_Mean', 'Scatt_STD'

]

# ================= 3. 执行批处理 =================
print("正在加载模型...")
with open(model_path, 'rb') as f:
    kmeans = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

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

            # 2. 预测原始分类 (得到 0~5)
            X = df[features]
            X_std = scaler.transform(X)  # 必须使用 transform
            raw_labels = kmeans.predict(X_std)

            # 3. 核心：应用数值映射
            # 将 0-5 映射为 0, 1, 2
            merged_ids = [numeric_map[l] for l in raw_labels]

            # 4. 写入结果
            df['Cluster'] = merged_ids

            # 增加一列原始ID和中文标签，方便追溯
            df['Original_K6_ID'] = raw_labels
            df['Phenotype_Desc'] = [label_desc[mid] for mid in merged_ids]

            # 5. 保存
            file_name = os.path.basename(file_path)
            save_name = file_name.replace('.xlsx', '_merge.xlsx')
            save_path = os.path.join(output_dir, save_name)

            df.to_excel(save_path, index=False)

        except Exception as e:
            print(f"  [错误] {os.path.basename(file_path)}: {e}")

    print(f"  -> 结果已保存至: {output_dir}")

print("\n全部完成！Cluster 列现已包含合并后的数值 ID (0, 1, 2)。")