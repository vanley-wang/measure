import os
import cv2 as cv
import struct
import numpy as np

# 数据格式
x_size = 512
y_size = 800
z_size = 800 

# 图像加载
with open('FilePath.txt', 'r') as file:
        ROOTS = file.read().splitlines()
for ROOT in ROOTS:
        TR_ROOT = os.path.join(ROOT, 'tr')
        if not os.path.isdir(os.path.join(ROOT, 'images')):
                os.mkdir(os.path.join(ROOT, 'images'))
        IMG_ROOT = os.path.join(ROOT, 'images')
        
        for tr in os.listdir(TR_ROOT):
                tr_name = tr[:-3]
                tr_path = os.path.join(TR_ROOT, tr)
                if not os.path.isdir(os.path.join(IMG_ROOT, tr_name)): 
                        os.mkdir(os.path.join(IMG_ROOT, tr_name))
                image_path = os.path.join(IMG_ROOT, tr_name)
                
                slice = np.zeros((y_size, x_size), dtype='uint8')        
                with open(tr_path, 'rb') as tr:
                        for z in range(z_size):
                                for i in range(y_size):
                                        data = tr.read(x_size)
                                        slice[i, :] = np.asarray(struct.unpack('B' * x_size, data))
                                rotate_slice = cv.rotate(slice, cv.ROTATE_90_CLOCKWISE)
                                # save_path = os.path.join(image_path, tr_name + '_' + str(z+1) + '.png')
                                save_path = os.path.join(image_path, str(z) + '.png')
                                cv.imwrite(save_path, rotate_slice)
                print('{} Transform Complete!'.format(tr_path))