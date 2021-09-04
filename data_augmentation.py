#coding=utf-8#
'''
@File    :   data_augmentation.py
@Time    :   2021/08/22 22:18:56
@Author  :   Li Yang 
@Version :   1.0
@Contact :   liyang259@mail2.sysu.edu.cn
@License :   (C)Copyright 2020-2021, Li Yang
@Desc    :   Data Augmentation includes rotation, translation & flip
'''
import cv2 
import numpy as np
import random
import utility


data_path = '/Users/menglidaren/Desktop/SrcData/IMG'

sample_array = np.load(data_path+'/'+'sample.npy')
label_array = np.load(data_path + '/' + 'label.npy')
# get index
idx_range = sample_array.shape[0]
'''
idx_r = random.sample(range(0,idx_range), int(0.1*idx_range))
idx_t = random.sample(range(0,idx_range), int(0.2*idx_range))
idx_f = random.sample(range(0,idx_range), int(0.2*idx_range))
'''
idx_r, idx_f, idx_t = idx_range, idx_range, idx_range
shape = sample_array.shape[2:4]
mat_rotate = []
for i in range(1,5):
    m_p = cv2.getRotationMatrix2D((shape[0]//2,shape[1]//2), i*0.5 ,1)
    m_n = cv2.getRotationMatrix2D((shape[0]//2,shape[1]//2), -i*0.5 ,1)
    mat_rotate.append(m_p)
    mat_rotate.append(m_n)
# rotation
print('rotate 10% of images')
for i in range(0,idx_r):
    sample2rotate = sample_array[i,0,:,:]
    label2roatate = label_array[i,0,:,:]
    sample_rotated = []
    label_rotated = []
    for rotate in mat_rotate:
        s = cv2.warpAffine(sample2rotate,rotate,shape)
        l = cv2.warpAffine(label2roatate,rotate,shape)
        # binary threshold
        l[l>=0.5] = 1
        l[l<0.5] = 0
        sample_rotated.append(s)
        label_rotated.append(l)
    sample_rotated = np.stack(sample_rotated,0)
    label_rotated = np.stack(label_rotated,0)
    sample_rotated = np.expand_dims(sample_rotated,1)
    label_rotated = np.expand_dims(label_rotated,1)
    '''
    for m in range(0,sample_rotated.shape[0]):
        utility.show_sample_label(sample_rotated[m,0,:,:],label_rotated[m,0,:,:])
    '''
    sample_array = np.concatenate((sample_array,sample_rotated),0)
    label_array = np.concatenate((label_array, label_rotated),0)
print(str(label_rotated.shape[0]*idx_r) + ' images are created')
print('========================================================')
print('translate 20% of images')
    
# translation
for i in range(0,idx_t):
    tx = float(random.randint(1,3))
    ty = float(random.randint(1,3))
    mat_translation = []
    mat_translation.append(np.array([[1,0,tx],[0,1,ty]]))
    mat_translation.append(np.array([[1,0,-tx],[0,1,-ty]]))
    mat_translation.append(np.array([[1,0,tx],[0,1,-ty]]))
    mat_translation.append(np.array([[1,0,-tx],[0,1,ty]]))
    sample2tran = sample_array[i,0,:,:]
    label2tran = label_array[i,0,:,:]
    sample_translated = []
    label_translated = []
    for trans in mat_translation:
        s = cv2.warpAffine(sample2tran,trans,shape)
        l = cv2.warpAffine(label2tran,trans,shape)
        # binary threshold
        l[l>=0.5] = 1
        l[l<0.5] = 0
        sample_translated.append(s)
        label_translated.append(l)
    sample_translated = np.stack(sample_translated,0)
    label_translated = np.stack(label_translated,0)
    sample_translated = np.expand_dims(sample_translated, 1)
    label_translated = np.expand_dims(label_translated,1)
    '''
    for m in range(0,sample_translated.shape[0]):
        utility.show_sample_label(sample_translated[m,0,:,:],label_translated[m,0,:,:])
    '''
    sample_array = np.concatenate((sample_array,sample_translated),0)  
    label_array = np.concatenate((label_array, label_translated),0)
print(str(label_translated.shape[0]*idx_t) + ' images are created')
print('========================================================')
print('flip 20% of images')
# flip
for i in range(0,idx_f):
    sample2flip = sample_array[i,0,:,:]
    label2flip = label_array[i,0,:,:]
    sample_fliped = cv2.flip(sample2flip,0)
    label_fliped = cv2.flip(label2flip,0)
    sample_fliped = np.expand_dims(sample_fliped,0)
    sample_fliped = np.expand_dims(sample_fliped,0)
    label_fliped = np.expand_dims(label_fliped,0)
    label_fliped = np.expand_dims(label_fliped,0)
    # utility.show_sample_label(sample_fliped[0,0,:,:],label_fliped[0,0,:,:])
    sample_array = np.concatenate((sample_array,sample_fliped),0)
    label_array = np.concatenate((label_array, label_fliped),0)
print(str(label_fliped.shape[0]*idx_f) + ' images are created')

    
np.save(data_path+'/augmentation/sample.npy',sample_array)
np.save(data_path+'/augmentation/label.npy',label_array)
    
    