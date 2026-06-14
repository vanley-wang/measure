import os
import numpy as np
import pandas as pd
from pathlib import Path


# 降采样后总数据所在excel路径
DataPath = 'Data/FXN/FXN-undersampled.xlsx'
Data_UnderSample = pd.read_excel(DataPath)

# 原始数据excel所在文件夹路径
path = 'Data/FXN/20230703/excel'
file_ls = os.listdir(path)
ROOTS = [os.path.join(path, file) for file in file_ls]

# 将降采样后总数据拆分到各自的excel中
list = Data_UnderSample['Index']
for ROOT in ROOTS:
    p = Path(ROOT)
    name = p.stem
    index_temp = []
    for i in range(len(list)):
        if name in list[i]:
            index_temp.append(True)
        else:
            index_temp.append(False)
    temp = Data_UnderSample.loc[index_temp]
    temp.to_excel(ROOT.replace('excel', 'excel_undersample'), index=False)
    print('{} Complete!'.format(ROOT))