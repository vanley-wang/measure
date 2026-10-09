import os
import pandas as pd
from pathlib import Path

# # Rename Excel
# path = 'Data/GC006-P9/20240826/excel'
# date = os.path.basename(os.path.dirname(path))
# list = os.listdir(path)
# for i in range(len(list)):
#     old_name = list[i]
#     old_path = os.path.join(path, old_name)
#     p = Path(old_path)
#     new_name = p.stem + '_' + date + p.suffix
#     new_path = os.path.join(path, new_name)
#     os.rename(old_path, new_path)
#     print('{} Complete!'.format(new_name))

# Rename 'Index' in Excel
path = 'Data/GC006-P9/20240826/excel'
list = os.listdir(path)
for i in range(len(list)):
    ROOT = os.path.join(path, list[i])
    DF = pd.read_excel(ROOT)

    p = Path(ROOT)
    name = p.stem
    index = []
    for i in range(len(DF)):
        temp = name + '_' + str(i+1)
        index.append(temp)
    DF['Index'] = index

    DF.to_excel(ROOT, index=False)
    print('{} Complete!'.format(ROOT))


# # Rename '.mat'
# path = 'Data/ICC027-P6-11-10-1/20231115/seg_filt'
# date = os.path.basename(os.path.dirname(path))
# list = os.listdir(path)
# for i in range(len(list)):
#     old_name = list[i]
#     old_path = os.path.join(path, old_name)
#     p = Path(old_path)
#     new_name = p.stem + '_' + date + p.suffix
#     new_path = os.path.join(path, new_name)
#     os.rename(old_path, new_path)
#     print('{} Complete!'.format(new_name))