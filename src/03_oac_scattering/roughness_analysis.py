import os
import pandas as pd
import numpy as np
import warnings

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
A_path = r'Data\nnUNet_FXN_2023\nnUNet_Analysis.xlsx'
Path1 = r'Data\nnUNet_FXN_2023\FXN_0701\wwl_measure\roughness'
Path2 = r'Data\nnUNet_FXN_2023\FXN_0703\wwl_measure\roughness'

# 真实数据类别 (0, 1)
TARGET_CLUSTERS = [0, 1]
# ===========================================

print("📂 正在加载主表...")
try:
    df_A = pd.read_excel(A_path)
except FileNotFoundError:
    print(f"❌ 找不到主表文件: {A_path}")
    exit()

ROOTS = [os.path.join(Path1, f) for f in os.listdir(Path1) if f.endswith('.xlsx')] + \
        [os.path.join(Path2, f) for f in os.listdir(Path2) if f.endswith('.xlsx')]

print(f"🔍 找到 {len(ROOTS)} 个数据文件，开始汇总...")
print("📝 映射逻辑: 数据[0]->Avg_1, 数据[1]->Avg_2, [0+1]->All(Sum)")

# 使用 tqdm 显示进度
for ROOT in tqdm(ROOTS, desc="汇总进度", unit="file"):
    filename = os.path.basename(ROOT)
    index_key = filename.replace('_roughness.xlsx', '')

    try:
        DF = pd.read_excel(ROOT)

        # ---------------------------------------------------------
        # 计算 Roughness_All (0和1类的总和)
        # ---------------------------------------------------------
        # 筛选属于 0 或 1 的所有行
        mask_all = DF['Cluster'].isin(TARGET_CLUSTERS)

        if mask_all.any():
            # 【重要】这里按你的要求计算的是“加起来”(Sum)
            # 如果以后想改平均值，把 .sum() 改成 .mean() 即可
            val_all = DF.loc[mask_all, 'Roughness'].sum()
        else:
            val_all = 0

        # -> 写入小文件 (记录在第一行)
        col_name_all = 'Roughness_All'
        if len(DF) > 0:
            DF.loc[0, col_name_all] = round(val_all, 4)

        # -> 写入主表
        row_indices = df_A.index[df_A.iloc[:, 0] == index_key]
        if len(row_indices) > 0:
            idx = row_indices[0]
            # 动态插入列 (插在最后面，或者指定位置)
            if col_name_all not in df_A.columns:
                # 这里我把它插在表格比较靠后的位置，以免打乱 Number_1 附近的结构
                df_A.insert(loc=len(df_A.columns), column=col_name_all, value=np.nan)

            df_A.loc[idx, col_name_all] = round(val_all, 4)

        # ---------------------------------------------------------
        # 分别计算 Cluster 0 (Avg_1) 和 Cluster 1 (Avg_2)
        # ---------------------------------------------------------
        for target in reversed(TARGET_CLUSTERS):

            display_num = target + 1
            col_name_file = f'roughness_cluster{display_num}_avg'
            summary_col_name = f'Roughness_Avg_{display_num}'

            mask = DF['Cluster'] == target

            if mask.any():
                avg_val = DF.loc[mask, 'Roughness'].mean()  # 这里保持计算均值
                first_idx = DF[mask].index[0]
                DF.loc[first_idx, col_name_file] = round(avg_val, 4)
            else:
                avg_val = 0

                # 写入主表
            if len(row_indices) > 0:
                idx = row_indices[0]
                if summary_col_name not in df_A.columns:
                    if 'Number_1' in df_A.columns:
                        insert_pos = df_A.columns.get_loc('Number_1') + 1
                    else:
                        insert_pos = len(df_A.columns)
                    df_A.insert(loc=insert_pos, column=summary_col_name, value=np.nan)

                df_A.loc[idx, summary_col_name] = round(avg_val, 4)

        # 保存
        DF.to_excel(ROOT, index=False)

    except Exception as e:
        tqdm.write(f'❌ 处理 {filename} 失败: {e}')

# 保存主表
try:
    print("\n💾 正在保存主表...")
    df_A.to_excel(A_path, index=False)
    print(f'✅ 汇总完成！包含特征: Roughness_Avg_1, Roughness_Avg_2, Roughness_All')
except PermissionError:
    print(f'\n❌ 保存失败！请先关闭 Excel 文件: {A_path}')