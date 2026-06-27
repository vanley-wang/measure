import os
import pandas as pd


Path = 'Data/FXN_2023_new/FXN_2023_Analysis.xlsx'
# Path = 'Data/icc016p6/icc016p620230911004/ICCO16P6_2023_Analysis.xlsx'
Data_All = pd.read_excel(Path)

Data = Data_All[[
    'Number_1',
    'Volume_Fill_Avg_1',
    'Surface_Avg_1',
    'Cavity_Volume_All_1',
    'Long_Axis_Avg_1',
    'Short_Axis_Avg_1',
    'Cyst_Thick_Avg_1',
    'Sphericity_Avg_1',
    # 'Roughness_Avg_1',
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
    'Cavity_Volume_All_4',
    'Long_Axis_Avg_4',
    'Short_Axis_Avg_4',
    'Cyst_Thick_Avg_4',
    'Sphericity_Avg_4',
    'Scatt_Mean_Avg_4',
    'Scatt_STD_Avg_4',

    # 'Number_5',
    # 'Volume_Fill_Avg_5',
    # 'Surface_Avg_5',
    # 'Cavity_Volume_All_5',
    # 'Long_Axis_Avg_5',
    # 'Short_Axis_Avg_5',
    # 'Cyst_Thick_Avg_5',
    # 'Sphericity_Avg_5',
    # 'Scatt_Mean_Avg_5',
    # 'Scatt_STD_Avg_5',
    #
    # 'Number_6',
    # 'Volume_Fill_Avg_6',
    # 'Surface_Avg_6',
    # 'Cavity_Volume_All_6',
    # 'Long_Axis_Avg_6',
    # 'Short_Axis_Avg_6',
    # # 'Cyst_Thick_Avg_6',
    # 'Sphericity_Avg_6',
    # 'Scatt_Mean_Avg_6',
    # 'Scatt_STD_Avg_6',
]]

# 输出各特征的特征值
from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=10)
res = vt.fit_transform(Data)
print(vt.variances_)

# 输出选择的特征名
print(vt.get_feature_names_out())