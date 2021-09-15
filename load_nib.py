# -*-coding:utf-8 -*-
'''
@File    :   nii2npy.py
@Time    :   2021/07/19 17:32:22
@Author  :   Li Yang 
@Version :   1.0
@Contact :   liyang259@mail2.sysu.edu.cn
@License :   (C)Copyright 2020-2021, Li Yang
@Desc    :   None
'''

'该模块将指定路径的MRI图像(nii.gz格式)与标签Resample并reszie到设定大小并将sample和label分别合并为两个npy数组并保存到指定路径.'

from numpy.core.arrayprint import dtype_is_implied
from config import Config as cg
import nibabel as nib
import nibabel.processing as proc
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys, getopt, os
import time
import math
import re

def resize(img, size):
    nx, ny, nz = img.shape
    # crop
    if nx > size:
        center = nx // 2
        sidx = center - math.floor(size/2)
        eidx = center + math.ceil(size/2)
        img = img[sidx:eidx, :, :]
    else: 
    # pad
        diff = size - nx
        lSize = math.floor(diff/2)
        rSize = math.ceil(diff/2)
        lBlock = np.zeros([lSize, ny, nz], dtype= np.uint16)
        rBlock = np.zeros([rSize, ny, nz], dtype= np.uint16)
        img = np.concatenate((lBlock,img,rBlock), axis= 0)
    # crop
    if ny > size:
        center = ny // 2
        sidx = center - math.floor(size/2)
        eidx = center + math.ceil(size/2)
        img = img[:,sidx:eidx,:]
    else: 
    # pad
        diff = size - ny
        lSize = math.floor(diff/2)
        rSize = math.ceil(diff/2)
        lBlock = np.zeros([nx,lSize,nz], dtype= np.uint16)
        rBlock = np.zeros([nx,rSize, nz], dtype= np.uint16)
        img = np.concatenate((lBlock,img,rBlock), axis= 1)

    return img

def main(Args):
    inputfolder = '../SrcData'
    outputfolder = '../train_data/IMG'
    labelFolder = 'label'
    sampleFolder = 'sample'
    if os.path.exists(inputfolder +'/'+sampleFolder+'/'+'.DS_Store'):
        os.remove(inputfolder +'/'+sampleFolder+'/'+'.DS_Store')
    try:
        opts, args = getopt.getopt(Args, "i:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('nii2png.py -i <inputfolder> -o <outputfolder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('nii2png.py -i <inputfolder> -o <outputfolder>')
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfolder = arg
        elif opt in ("-o", "--output"):
            outputfolder = arg

    sampleFile = os.listdir(inputfolder + '/' + sampleFolder)
    # labelFile = os.listdir(inputfolder + '/' + labelFolder) 由于随机性，故不可以直接读取

    if os.path.exists(outputfolder):
        pass 
    else: 
        print("Created ouput directory: " + outputfolder +'/' + 'sample')
        os.mkdir(outputfolder)
    

    
    sample_a = np.zeros([cg.image_size,cg.image_size, 1], dtype = np.uint16)
    label_a = np.zeros([cg.image_size,cg.image_size, 1], dtype = np.uint16)

    print('Load samples')

# Convert to numpy & data augmentation
    
    for file in sampleFile:
        labelName = re.sub(r'sample', 'mask-', file)
        sample = nib.load(inputfolder + '/' + sampleFolder + '/' + file)
        label = nib.load(inputfolder + '/' + labelFolder + '/' + labelName)
        slice_shape = sample.shape
        shape = [256,256,slice_shape[2]]
        print(file+' with a size of '+ str(slice_shape) + ' loaded')
        print('========================================================')
        print('Data Cleaning')
        print('========================================================')
        img_array = sample.get_fdata()
        label_array = label.get_fdata()
        idx=[]
        for i in range(0,slice_shape[2]):
            if np.max(img_array[:,:,i])== 0:
                pass
            elif np.max(label_array[:,:,i]) == 0:
                pass
            else:
                idx.append(i)
        img_resampled = proc.conform(sample, out_shape=shape, voxel_size=[1,1,sample.header['pixdim'][3]])
        img_resampled_array = img_resampled.get_fdata()
        label_resampled = proc.conform(label,out_shape=shape, voxel_size=[1,1,label.header['pixdim'][3]])
        label_resampled_array = label_resampled.get_fdata()
        img_s = img_resampled_array[:,:,idx]
        label_s = label_resampled_array[:,:,idx]
        
        #Patche-based
        '''
        img_temp = []
        img_temp.append(img_s[0:256,0:256,:])
        img_temp.append(img_s[-256:,0:256:,:])
        img_temp.append(img_s[0:256,-256:,:])
        img_temp.append(img_s[-256:,-256:,:])
        
        label_temp=[]
        label_temp.append(label_s[0:256,0:256,:])
        label_temp.append(label_s[-256:,0:256,:])
        label_temp.append(label_s[0:256,-256:,:])
        label_temp.append(label_s[-256:,-256:,:])
        
        img_s = np.concatenate(img_temp, axis=2)
        label_s = np.concatenate(label_temp,axis=2)
        '''

        #histogram equalization

        # binary threshold
        label_s[label_s>=0.5] = 1
        label_s[label_s<0.5] = 0
        
        # Normalization
        ## slice
        '''
        for i in range(1,len(idx)):
            img_n = img_s[:,:,i]
            _range = np.max(img_n)
            img_s[:,:,i] = img_s[:,:,i]/_range
        '''
        ## volume
        _range = np.max(img_s)
        img_s = img_s/_range
        
        # Visualization
        for i in range(0,img_s.shape[2]):
            img2show = img_s[:,:,i]
            label2show = label_s[:,:,i]
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(img2show, cmap="gray")
            plt.title('sample')
            plt.subplot(1,2,2)
            plt.imshow(label2show, cmap="gray")
            plt.title('label')

        print(str(len(idx)) + ' out of ' + str(slice_shape[2]) + ' selected.')
        sample_a = np.concatenate((sample_a,img_s), axis=2)
        label_a = np.concatenate((label_a,label_s), axis=2)
        print(file + ' loaded & appended.')
        print('========================================================')
    
    
    sample_a = sample_a[:,:,1:]
    label_a = label_a[:,:,1:]
    sample_a = np.transpose(sample_a, (2,0,1))
    label_a = np.transpose(label_a, (2,0,1))
    sample_a = np.expand_dims(sample_a, axis=1)
    label_a = np.expand_dims(label_a, axis =1)

    print('All samples loaded & npy shape is:' + str(sample_a.shape))
    print('All labels loaded & npy shape is:' + str(label_a.shape))

    np.save(outputfolder + '/' + 'sample.npy', sample_a)
    np.save(outputfolder + '/' + 'label.npy', label_a)
    
    print('samples have been saved in' + outputfolder)
    print('labels have been saved in' + outputfolder)


    
if __name__ == '__main__':
    main(sys.argv)
     
    