import os
import pandas as pd

# 定义两个大文件夹路径
folders = [
    'Data/nnUNet_FXN_2023/FXN_0701',
    'Data/nnUNet_FXN_2023/FXN_0703'
]

for folder in folders:
    cluster_dir = os.path.join(folder, 'cluster_merge')

    if not os.path.exists(cluster_dir):
        print(f"[WARN] 路径不存在: {cluster_dir}")
        continue

    # 获取所有 _merge.xlsx 文件（cluster-merge.py 输出）
    files = [f for f in os.listdir(cluster_dir) if f.endswith('_merge.xlsx')]

    all_data = []

    for file in files:
        file_path = os.path.join(cluster_dir, file)
        try:
            # 读取 Sheet2（由 cluster_analysis_1.py 追加的汇总统计）
            df = pd.read_excel(file_path, sheet_name='Sheet2')
            all_data.append(df)
        except Exception as e:
            print(f"[ERR] 无法读取 {file_path} 的 Sheet2：{e}")

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        save_name = os.path.basename(folder.rstrip('/\\')) + '_Analysis.xlsx'
        save_path = os.path.join(folder, save_name)
        merged_df.to_excel(save_path, index=False)
        print(f"[OK] 合并完成：{save_path}")
    else:
        print(f"[WARN] 没有可用的Sheet2文件在 {cluster_dir}")

# 合并两个批次的主表
path1 = 'Data/nnUNet_FXN_2023/FXN_0701/FXN_0701_Analysis.xlsx'
path2 = 'Data/nnUNet_FXN_2023/FXN_0703/FXN_0703_Analysis.xlsx'

df1 = pd.read_excel(path1)
df2 = pd.read_excel(path2)
df_merged = pd.concat([df1, df2], ignore_index=True)
df_merged.to_excel('Data/nnUNet_FXN_2023/nnUNet_Analysis.xlsx', index=False)
print("[OK] 主表合并完成：Data/nnUNet_FXN_2023/nnUNet_Analysis.xlsx")
