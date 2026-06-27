import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import random
import time
import os

# ================= 1. 配置参数 =================
input_path = 'Data/nnUNet_FXN_2023/nnUNet_Analysis.xlsx'
log_save_path = 'Data/nnUNet_FXN_2023/Feature_Optimization_Sorted.xlsx'

# 搜索次数 (建议 50,000 或更多)
n_iterations = 5000000

# 并行核心数 (-1 代表使用所有核心)
n_jobs = -1

# ATP 数据库
# atp_database = {
#     'B10': 601300, 'B11': 11180000, 'B2': 5391000, 'B3': 6538000,
#     'B4': 7103000, 'B5': 1264000, 'B6': 2548000, 'B7': 1579000,
#     'B8': 3637000, 'B9': 140300, 'C10': 211800, 'C11': 13930000,
#     'C2': 4460000, 'C3': 8336000, 'C4': 6800000, 'C5': 330900,
#     'C6': 238900, 'C7': 682100, 'C8': 211300, 'C9': 465900,
#     'D11': 11240000, 'E11': 12840000, 'F10': 21910000, 'F11': 14700000,
#     'F2': 26980000, 'F3': 14110000, 'F4': 13740000, 'F5': 17250000,
#     'F6': 20320000, 'F7': 20000000, 'F8': 17170000, 'F9': 15830000
# }

# 更新后的 ATP 数据库（根据表格提取）
atp_database = {
    # 'C10': 54230,
    'C11': 24330, 'C12': 71930,
    # 'C1': 2961000, 'C2': 6731000,
    'C3': 8638000,
    'C4': 7800000, 'C5': 7260000,
    # 'C6': 6671000,
    # 'C7': 77960,
    'C8': 185300,
    # 'C9': 46930,
    # 'D1': 45350, 'D2': 133600, 'D3': 28820,
    'D4': 66260, 'D5': 27630, 'D6': 25920,
    # 'E1': 32980, 'E2': 51460, 'E3': 69330,
    'E4': 63070, 'E5': 15450, 'E6': 68250,
    # 'F10': 56050000,
    'F11': 55300000, 'F12': 46400000,
    # 'F1': 12407000,
    'F2': 14970000, 'F3': 14070000,
    'F4': 10820000,
    # 'F5': 6017000, 'F6': 12140000,
    'F7': 54140000, 'F8': 50680000, 'F9': 65380000
}

# 候选特征池
full_feature_pool = [
     'Number_1',
    'Volume_Fill_Avg_1',
    'Surface_Avg_1',
    'Cavity_Volume_All_1',
    'Long_Axis_Avg_1',
    'Short_Axis_Avg_1',
    'Cyst_Thick_Avg_1',
    'Sphericity_Avg_1',
    # 'Roughness_Avg_1',
    # 'Roughness_Avg_2',
    'Roughness_All',
    'Scatt_Mean_Avg_1',
    'Scatt_STD_Avg_1',

    'Number_2',
    'Volume_Fill_Avg_2',
    'Surface_Avg_2',
    'Cavity_Volume_All_2',
    'Long_Axis_Avg_2',
    'Short_Axis_Avg_2',
    'Cyst_Thick_Avg_2',
    'Sphericity_Avg_2',
    'Scatt_Mean_Avg_2',
    'Scatt_STD_Avg_2',

    'Number_3',
    'Volume_Fill_Avg_3',
    'Surface_Avg_3',
    'Cavity_Volume_All_3',
    'Long_Axis_Avg_3',
    'Short_Axis_Avg_3',
    'Cyst_Thick_Avg_3',
    'Sphericity_Avg_3',
    'Scatt_Mean_Avg_3',
    'Scatt_STD_Avg_3',

    'Number_4',
    'Volume_Fill_Avg_4',  # 原代码删掉 4
    'Surface_Avg_4',
    # 'Cavity_Volume_All_4',
    'Long_Axis_Avg_4',
    'Short_Axis_Avg_4',
    'Cyst_Thick_Avg_4',
    'Sphericity_Avg_4',
    'Scatt_Mean_Avg_4',
    'Scatt_STD_Avg_4',
]

# ================= 2. 数据准备 =================
print("正在读取数据...")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"找不到文件: {input_path}")

Data_All = pd.read_excel(input_path)
if 'Name' not in Data_All.columns: Data_All['Name'] = Data_All.index.astype(str)

# 仅保留存在的特征
available_pool = [f for f in full_feature_pool if f in Data_All.columns]
# 也可以在这里先对 pool 进行排序，双重保险
available_pool.sort()
print(f"有效候选特征数: {len(available_pool)}")

# 转为 NumPy 矩阵
X_matrix_all = Data_All[available_pool].fillna(0).values

# 准备索引和 ATP
Data_All['Well_ID'] = Data_All['Name'].apply(lambda x: x.split('_')[0])
Data_All['TimePoint'] = Data_All['Name'].apply(lambda x: x.split('_')[1])
Data_All['ATP'] = Data_All['Well_ID'].map(atp_database)

start_indices = []
end_indices = []
valid_atp = []

for well in Data_All['Well_ID'].unique():
    row_start = Data_All[(Data_All['Well_ID'] == well) & (Data_All['TimePoint'] == '0701')]
    row_end = Data_All[(Data_All['Well_ID'] == well) & (Data_All['TimePoint'] == '0703')]

    if not row_start.empty and not row_end.empty:
        atp = row_end.iloc[0]['ATP']
        if pd.notna(atp):
            start_indices.append(Data_All.index.get_loc(row_start.index[0]))
            end_indices.append(Data_All.index.get_loc(row_end.index[0]))
            valid_atp.append(atp)

idx_start = np.array(start_indices)
idx_end = np.array(end_indices)
y_atp = np.array(valid_atp)

# ATP 的统计量
atp_mean = np.mean(y_atp)
atp_std = np.std(y_atp)

print(f"数据准备完毕。样本对数量: {len(y_atp)}。启动并行计算...")


# ================= 3. 核心计算函数 (强制排序版) =================
def run_sorted_sklearn_trial(seed):
    # 必须在函数内导入，否则 joblib 可能报错
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)

    n_total = X_matrix_all.shape[1]
    n_select = rng.randint(4, n_total + 1)

    # 随机选择索引
    feat_indices = rng.choice(n_total, n_select, replace=False)

    # 【关键步骤】强制排序！
    # 确保特征列的顺序永远是从小到大，避免 PCA 符号翻转
    feat_indices.sort()

    # 1. 提取子矩阵
    X_sub = X_matrix_all[:, feat_indices]

    try:
        # 2. 调用 sklearn (确保与验证脚本算法一致)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_sub)

        pca = PCA(n_components=4)
        Data_Pca = pca.fit_transform(X_std)

        # 3. 计算权重
        variance_ratio = pca.explained_variance_ratio_
        total_var = np.sum(variance_ratio)
        if total_var == 0: return None

        weights = variance_ratio / total_var
        result_scores = np.dot(Data_Pca, weights)

    except Exception:
        return None

    # 4. 计算相关性
    res_start = result_scores[idx_start]
    res_end = result_scores[idx_end]
    res_diff = res_end - res_start

    # 差值相关性 (Diff vs ATP)
    diff_mean = np.mean(res_diff)
    diff_std = np.std(res_diff)
    if diff_std == 0: return None
    cov_diff = np.mean((res_diff - diff_mean) * (y_atp - atp_mean))
    r_diff = cov_diff / (diff_std * atp_std)

    # 终点相关性 (End vs ATP)
    end_mean = np.mean(res_end)
    end_std = np.std(res_end)
    cov_end = np.mean((res_end - end_mean) * (y_atp - atp_mean))
    r_end = cov_end / (end_std * atp_std)

    return {
        'n_feats': len(feat_indices),
        'feat_idx': feat_indices,  # 已排序的索引
        'diff_r': abs(r_diff),
        'end_r': abs(r_end)
    }


# ================= 4. 并行执行 =================
if __name__ == '__main__':
    start_time = time.time()

    seeds = [random.randint(0, 100000000) for _ in range(n_iterations)]

    print(f"正在全速计算 ({n_iterations} 次)...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_sorted_sklearn_trial)(seed) for seed in seeds
    )

    end_time = time.time()
    print(f"\n计算完成！耗时: {end_time - start_time:.2f} 秒")

    # ================= 5. 结果整理 =================
    print("正在整理结果...")
    clean_results = []
    for res in results:
        if res is not None:
            # 还原特征名 (因为索引已排序，这里的特征名也会按顺序排列)
            names = [available_pool[i] for i in res['feat_idx']]
            clean_results.append({
                'Diff_Corr_Abs': res['diff_r'],
                'End_Corr_Abs': res['end_r'],
                'Num_Features': res['n_feats'],
                'Features_List': ", ".join(names)
            })

    df = pd.DataFrame(clean_results)
    df = df.sort_values(by='Diff_Corr_Abs', ascending=False)

    # 保存 Top 2000
    df.head(500).to_excel(log_save_path, index=False)

    print("\n" + "=" * 40)
    print("TOP 3 最佳特征组合 (已强制排序)：")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\nNo.{i + 1}: 差值相关性={row['Diff_Corr_Abs']:.6f}")
        print(f"特征: {row['Features_List']}")

    print(f"\n结果已保存至: {log_save_path}")
    print("提示：在验证代码中使用这些特征时，记得也加上 .sort() 以确保顺序一致！")