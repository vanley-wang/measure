import os
from pathlib import Path
import pandas as pd
import numpy as np

path1 = 'Data/nnUNet_FXN_2023/FXN_0701/cluster_excel'

file_ls = os.listdir(path1)
ROOTS1 = [os.path.join(path1, file) for file in file_ls]
path2 = 'Data/nnUNet_FXN_2023/FXN_0703/cluster_excel'
# path2 = 'Data/nnUNet_FXN_2023/FXN_0703/cluster_excel'

file_ls = os.listdir(path2)
ROOTS2 = [os.path.join(path2, file) for file in file_ls]
ROOTS = ROOTS1 + ROOTS2

# path2 = 'Data/nnUNet_FXN_2023/FXN_0703/cluster_excel'
# file_ls = os.listdir(path2)
# ROOTS2 = [os.path.join(path2, file) for file in file_ls]
# ROOTS = ROOTS2

for ROOT in ROOTS:
    DF = pd.read_excel(ROOT)
    cluster = DF[['Cluster']]
    cluster = np.array(cluster).T
    p = Path(ROOT)
    name = p.stem
    name = name.rsplit(sep="_", maxsplit=1)[0]

    # 参数计算
    cluster_name = []
    number = []
    volume_fill_avg = []
    surface_avg = []
    cavity_volume_all = []
    long_axis_avg = []
    short_axis_avg = []
    cyst_thick_avg = []
    sphericity_avg = []

    scatt_mean_avg = []         # 新增
    scatt_std_avg = []          # 新增

    for i in range(4):  # i -> 聚类数目
        cluster_temp = np.where(cluster == i)[1]
        temp = DF.loc[cluster_temp]
        number_temp = len(cluster_temp)
        if number_temp > 0:
            number.append(number_temp)
            volume = np.array(temp['Organoids_Volume'])
            volume_fill_avg.append(np.mean(volume))
            surface = np.array(temp['Organoids_Surface'])
            surface_avg.append(np.mean(surface))
            cavity_volume = np.array(temp['Cavity_Volume'])
            cavity_volume_all.append(np.sum(cavity_volume))
            long_axis = np.array(temp['LongAxis'])
            short_axis = np.array(temp['ShortAxis'])
            long_axis_avg.append(np.mean(long_axis))
            short_axis_avg.append(np.mean(short_axis))
            cyst_thick = np.array(temp['Wall_Thickness'])
            cyst_thick_avg.append(np.mean(cyst_thick))
            sphericity = np.array(temp['Sphericity'])
            sphericity_avg.append(np.mean(sphericity))
            scatt_mean = np.array(temp['Scatt_Mean'])   # 新增
            scatt_std = np.array(temp['Scatt_STD'])     # 新增
            scatt_mean_avg.append(np.mean(scatt_mean))  # 新增
            scatt_std_avg.append(np.mean(scatt_std))    # 新增
        else:
            number.append(0)
            volume_fill_avg.append(0)
            surface_avg.append(0)
            cavity_volume_all.append(0)
            long_axis_avg.append(0)
            short_axis_avg.append(0)
            cyst_thick_avg.append(0)
            sphericity_avg.append(0)
            scatt_mean_avg.append(0)     # 新增
            scatt_std_avg.append(0)      # 新增

    # 数据保存
    data = {'Name': name,
            # Cluster-1
            'Number_1': number[0],
            'Volume_Fill_Avg_1': volume_fill_avg[0],
            'Surface_Avg_1': surface_avg[0],
            'Cavity_Volume_All_1': cavity_volume_all[0],
            'Long_Axis_Avg_1': long_axis_avg[0],
            'Short_Axis_Avg_1': short_axis_avg[0],
            'Cyst_Thick_Avg_1': cyst_thick_avg[0],
            'Sphericity_Avg_1': sphericity_avg[0],

            'Scatt_Mean_Avg_1': scatt_mean_avg[0],  # 新增
            'Scatt_STD_Avg_1': scatt_std_avg[0],    # 新增

            # Cluster-2
            'Number_2': number[1],
            'Volume_Fill_Avg_2': volume_fill_avg[1],
            'Surface_Avg_2': surface_avg[1],
            'Cavity_Volume_All_2': cavity_volume_all[1],
            'Long_Axis_Avg_2': long_axis_avg[1],
            'Short_Axis_Avg_2': short_axis_avg[1],
            'Cyst_Thick_Avg_2': cyst_thick_avg[1],
            'Sphericity_Avg_2': sphericity_avg[1],

            'Scatt_Mean_Avg_2': scatt_mean_avg[1],
            'Scatt_STD_Avg_2': scatt_std_avg[1],

            # Cluster-3
            'Number_3': number[2],
            'Volume_Fill_Avg_3': volume_fill_avg[2],
            'Surface_Avg_3': surface_avg[2],
            'Cavity_Volume_All_3': cavity_volume_all[2],
            'Long_Axis_Avg_3': long_axis_avg[2],
            'Short_Axis_Avg_3': short_axis_avg[2],
            'Cyst_Thick_Avg_3': cyst_thick_avg[2],
            'Sphericity_Avg_3': sphericity_avg[2],

            'Scatt_Mean_Avg_3': scatt_mean_avg[2],
            'Scatt_STD_Avg_3': scatt_std_avg[2],

            # Cluster-4
            'Number_4': number[3],
            'Volume_Fill_Avg_4': volume_fill_avg[3],
            'Surface_Avg_4': surface_avg[3],
            'Cavity_Volume_All_4': cavity_volume_all[3],
            'Long_Axis_Avg_4': long_axis_avg[3],
            'Short_Axis_Avg_4': short_axis_avg[3],
            'Cyst_Thick_Avg_4': cyst_thick_avg[3],
            'Sphericity_Avg_4': sphericity_avg[3],

            'Scatt_Mean_Avg_4': scatt_mean_avg[3],
            'Scatt_STD_Avg_4': scatt_std_avg[3],

            }

    DF_cluster = pd.DataFrame(data, index=[0])
    with pd.ExcelWriter(ROOT, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        DF_cluster.to_excel(writer, sheet_name='Sheet2', index=False)
    print('{} Complete!'.format(ROOT))
