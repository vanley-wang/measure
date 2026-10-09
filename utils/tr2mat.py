import os
import cv2 as cv
import struct
import numpy as np
from scipy.io import savemat

# 数据格式
x_size = 512
y_size = 800
z_size = 800 

# 图像加载
with open('utils/FilePath.txt', 'r') as file:
        ROOTS = file.read().splitlines()
for ROOT in ROOTS:
        TR_ROOT = os.path.join(ROOT, 'tr')
        assert os.path.isdir(TR_ROOT), 'Tr do not exit!'
        MAT_ROOT = os.path.join(ROOT, 'mat')
        if not os.path.isdir(MAT_ROOT):
                os.mkdir(MAT_ROOT)

        for tr in os.listdir(TR_ROOT):
                tr_name = tr[:-3]
                tr_path = os.path.join(TR_ROOT, tr)
                
                res = []
                slice = np.zeros((y_size, x_size), dtype='uint8')        
                with open(tr_path, 'rb') as tr:
                        for j in range(z_size):
                                for i in range(y_size):
                                        data = tr.read(x_size)
                                        slice[i, :] = np.asarray(struct.unpack('B' * x_size, data))
                                rotate_slice = cv.rotate(slice, cv.ROTATE_90_CLOCKWISE)
                                res.append(rotate_slice)

                res = np.array(res)
                res = np.transpose(res,(1,2,0))
                
                savemat(os.path.join(MAT_ROOT, tr_name + '.mat'), {'data':res.astype(np.uint8)})             
                print('{} Transform Complete!'.format(tr_path))