# -*-coding:utf-8 -*-
'''
@File    :   dataVisualize.py
@Time    :   2021/07/28 20:01:53
@Author  :   Li Yang 
@Version :   1.0
@Contact :   liyang259@mail2.sysu.edu.cn
@License :   (C)Copyright 2020-2021, Li Yang
@Desc    :   None
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

dataset = 'NPC-seg'
dataRootPath = '../npys/'+ dataset + '/IMG'
sample = np.load(dataRootPath + '/sample/sample.npy')
label = np.load(dataRootPath+'/label/label.npy')
length = label.shape[0]
'''
for i in range(0,length-1):
    sample_sampled = sample[i,:,:,:]
    sample_sampled = np.squeeze(sample_sampled)
    label_sampled = label[i,:,:,:]
    label_sampled = np.squeeze(label_sampled)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(sample_sampled,cmap='gray')
    plt.axis('off')
    plt.title(str(i))
    plt.subplot(1,2,2)
    plt.imshow(label_sampled, cmap='gray')
    plt.axis('off')
    plt.title(str(i))
    plt.show()
'''
sample2save = sample[1339,:,:,:]
sample2save = np.squeeze(sample2save)
label2save = label[1339,:,:,:]
label2save = np.squeeze(label2save)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(sample2save,cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(label2save, cmap='gray')
plt.axis('off')
fig = plt.gcf()
fig.savefig('../biglabeled.eps',dpi=300,format='eps')
plt.show()