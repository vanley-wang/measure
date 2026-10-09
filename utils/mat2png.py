import os
import numpy as np
import cv2 as cv
import h5py


# 数据格式
x_size = 512
y_size = 800
z_size = 800 

# 图像加载
path = 'Data/FXN-2024/20230703/seg_cluster/F8_20230703.mat'
save_path = 'Data/FXN-2024/20230703/images/F8_20230703'

with h5py.File(path,'r') as file:
    data = file['Data_Fill_Cluster'][:]
    data = data.transpose(2,0,1)
temp_data = [data == v+1 for v in range(4)]
for c in range(4):
    temp = temp_data[c]
    temp_save_path = os.path.join(save_path, 'cluster_' + str(c+1))
    if not os.path.isdir(temp_save_path):
        os.mkdir(temp_save_path)

    for z in range(z_size):
        slice = temp[:,:,z]
        slice = slice.astype(np.uint8)
        img_path = os.path.join(temp_save_path, str(z+1) + '.png')
        cv.imwrite(img_path, slice)
    print('{} Transform Complete!'.format(temp_save_path))