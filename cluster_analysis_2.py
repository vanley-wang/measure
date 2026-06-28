import os
import pandas as pd

# 定义两个大文件夹路径
folders = [
    'Data/nnUNet_FXN_2023/FXN_0701',
    'Data/nnUNet_FXN_2023/FXN_0703'
    # 'Data/icc016p6/icc016p620230911004/0913',
    # 'Data/icc016p6/icc016p620230911004/0915'
]

for folder in folders:
    # cluster_dir = os.path.join(folder, 'cluster_merge')
    cluster_dir = os.path.join(folder, 'cluster_merge')

    if not os.path.exists(cluster_dir):
        print(f"⚠️ 路径不存在: {cluster_dir}")
        continue

    # 获取所有 _cluster.xlsx 文件
    # files = [f for f in os.listdir(cluster_dir) if f.endswith('_merge.xlsx')]
    files = [f for f in os.listdir(cluster_dir) if f.endswith('_cluster.xlsx')]

    all_data = []

    for file in files:
        file_path = os.path.join(cluster_dir, file)
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet2')  #可选 sheet
            # 可选：记录来源
            # df.insert(0, 'Source_File', file)
            all_data.append(df)
        except Exception as e:
            print(f"❌ 无法读取 {file_path} 的 Sheet2：{e}")

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)

        # 生成保存文件名：例如 folder 下保存为 All_Analysis.xlsx
        save_name = os.path.basename(folder.rstrip('/\\')) + '_Analysis.xlsx'
        save_path = os.path.join(folder, save_name)

        merged_df.to_excel(save_path, index=False)
        print(f"✅ 合并完成：{save_path}")
    else:
        print(f"⚠️ 没有可用的Sheet2文件在 {cluster_dir}")
path1 = 'Data/nnUNet_FXN_2023/FXN_0701/FXN_0701_Analysis.xlsx'
path2 = 'Data/nnUNet_FXN_2023/FXN_0703/FXN_0703_Analysis.xlsx'
# path1 = 'Data/icc016p6/icc016p620230911004/0913/0913_Analysis.xlsx'
# path2 = 'Data/icc016p6/icc016p620230911004/0915/0915_Analysis.xlsx'
df1 = pd.read_excel(path1)
df2 = pd.read_excel(path2)
df_merged = pd.concat([df1, df2], ignore_index=True)
df_merged.to_excel('Data/nnUNet_FXN_2023/nnUNet_Analysis.xlsx', index=False)
# df_merged.to_excel('Data/icc016p6/icc016p620230911004/ICCO16P6_2023_Analysis.xlsx', index=False)

