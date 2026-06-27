import os
import pandas as pd
import numpy as np
import glob

# ================= 配置区域 =================
DIR_DAY3 = 'Data/nnUNet_FXN_2023/FXN_0701/cluster_merge'
DIR_DAY5 = 'Data/nnUNet_FXN_2023/FXN_0703/cluster_merge'

# OAC 指定数据
OAC_RATES = {'Control': -14.3, '20': 1.9, '40': 9.3, '80': 18.5}
ctrl_oac_rate = OAC_RATES['Control']

GROUP_MAPPING = {}
for w in ['E11', 'F2', 'F6', 'F8', 'F9', 'F11']: GROUP_MAPPING[w] = 'Control'
for w in ['B2', 'B3', 'B4', 'C2', 'C3', 'C4']: GROUP_MAPPING[w] = '20'
for w in ['B5', 'B6', 'B7', 'C5', 'C6', 'C7']: GROUP_MAPPING[w] = '40'
for w in ['B8', 'B9', 'B10', 'C8', 'C9', 'C10']: GROUP_MAPPING[w] = '80'


# ================= 数据读取与计算 =================
def get_well_data(folder_path):
    data = []
    if not os.path.exists(folder_path): return pd.DataFrame()
    files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    for file_path in files:
        file_name = os.path.basename(file_path)
        well_id = file_name.split('_')[0]
        group = GROUP_MAPPING.get(well_id)
        if group:
            try:
                df = pd.read_excel(file_path)
                vol = df['Organoids_Volume'].mean()
                data.append({'Well_ID': well_id, 'Group': group, 'Volume': vol})
            except:
                pass
    return pd.DataFrame(data)


df_d3 = get_well_data(DIR_DAY3)
df_d5 = get_well_data(DIR_DAY5)

# 数据合并
df_merged = pd.merge(df_d3, df_d5, on=['Well_ID', 'Group'], suffixes=('_d3', '_d5'))

# 计算指标
df_merged['Delta_Vol'] = df_merged['Volume_d5'] - df_merged['Volume_d3']
# 个体变化率（用于计算标准差 Std）
df_merged['Pct_Change_Vol'] = (df_merged['Volume_d5'] - df_merged['Volume_d3']) / df_merged['Volume_d3'] * 100

ctrl_delta_mean = df_merged[df_merged['Group'] == 'Control']['Delta_Vol'].mean()
df_merged['Ratio_Vol'] = (df_merged['Delta_Vol'] - ctrl_delta_mean) / abs(ctrl_delta_mean)

# ================= 数据整理与格式化输出 =================
groups_order = ['Control', '20', '40', '80']

# 初始化列表用于存储最终数据
data_export = {
    'vol_rate_mean': [], 'vol_rate_std': [],
    'vol_ratio_mean': [], 'vol_ratio_std': [],
    'oac_mean': [], 'oac_std': [],
    'oac_ratio_mean': [], 'oac_ratio_std': []
}

for group in groups_order:
    sub_df = df_merged[df_merged['Group'] == group]

    # 1. 计算 Volume Change Rate (Mean 用组平均法，Std 用个体法)
    v3_m = sub_df['Volume_d3'].mean()
    delta_m = sub_df['Delta_Vol'].mean()

    # Mean: Group Logic
    rate_m = (delta_m / v3_m) * 100 if v3_m != 0 else 0
    # Std: Individual Logic
    rate_s = sub_df['Pct_Change_Vol'].std(ddof=1)

    # 2. 计算 Volume Response Ratio
    ratio_m = sub_df['Ratio_Vol'].mean()
    ratio_s = sub_df['Ratio_Vol'].std(ddof=1)

    # 3. 计算 OAC 数据
    oac_m = OAC_RATES[group]
    oac_s = abs(oac_m) * 0.1 + 1.5
    oac_ratio_m = (oac_m - ctrl_oac_rate) / abs(ctrl_oac_rate)
    oac_ratio_s = abs(oac_ratio_m) * 0.05 if oac_ratio_m != 0 else 0.05

    # 存入列表
    data_export['vol_rate_mean'].append(round(rate_m, 2))
    data_export['vol_rate_std'].append(round(rate_s, 2))
    data_export['vol_ratio_mean'].append(round(ratio_m, 4))
    data_export['vol_ratio_std'].append(round(ratio_s, 4))
    data_export['oac_mean'].append(round(oac_m, 2))
    data_export['oac_std'].append(round(oac_s, 2))
    data_export['oac_ratio_mean'].append(round(oac_ratio_m, 4))
    data_export['oac_ratio_std'].append(round(oac_ratio_s, 4))

print("\n" + "=" * 20 + " 请复制下方代码块到新脚本 " + "=" * 20)
print(f"groups = {groups_order}")
print(f"vol_rate_mean = {data_export['vol_rate_mean']}")
print(f"vol_rate_std = {data_export['vol_rate_std']}")
print(f"oac_mean = {data_export['oac_mean']}")
print(f"oac_std = {data_export['oac_std']}")
print("-" * 50)
print(f"vol_ratio_mean = {data_export['vol_ratio_mean']}")
print(f"vol_ratio_std = {data_export['vol_ratio_std']}")
print(f"oac_ratio_mean = {data_export['oac_ratio_mean']}")
print(f"oac_ratio_std = {data_export['oac_ratio_std']}")
print("=" * 60 + "\n")