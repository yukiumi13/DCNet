from numpy.core.arrayprint import dtype_is_implied
from config import Config as cg
import nibabel as nib
import numpy as np
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
    outputfolder = '../SrcData/IMG'
    labelFolder = 'Label'
    sampleFolder = 'Sample'
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
    labelFile = []
    for strs in sampleFile:
        if strs == '.DS_Store':
            pass #隐藏文件
        else:
            labelName = re.sub(r'sample', 'mask-', strs)
            labelFile.append(labelName)


    if os.path.exists(outputfolder + '/' + 'sample'):
        pass 
    else: 
        print("Created ouput directory: " + outputfolder + 'sample')
        os.mkdir(outputfolder + '/' + 'sample')
    
    if os.path.exists(outputfolder + '/' + 'label'):
        pass 
    else: 
        print("Created ouput directory: " + outputfolder + 'label')
        os.mkdir(outputfolder + '/' + 'label')
    

    
    sample = np.zeros([cg.image_size,cg.image_size, 1], dtype = np.uint16)
    label = np.zeros([cg.image_size,cg.image_size, 1], dtype = np.uint16)

    print('Load samples')

    shape = []
    
    for file in sampleFile:
        try:
            sampleArray = nib.load(inputfolder + '/' + sampleFolder + '/' + file).get_fdata()
            shape.append(sampleArray.shape)
            if sampleArray.shape == cg.image_size:
                pass
            else:
                print(file + ' ' + 'size:' + str(sampleArray.shape))
                sampleArray = resize(sampleArray, cg.image_size)
                print(file + ' ' + 'Resized:' + str(sampleArray.shape))
            print(sampleArray.shape)
            sample = np.concatenate((sample,sampleArray), axis=2)
            print(file + ' loaded & appended.')
        except nib.filebasedimages.ImageFileError :
            pass
        # 防止隐藏文件

    cout = 0
    
    for file in labelFile:
        try:
            labelArray = nib.load(inputfolder + '/' + labelFolder + '/' + file).get_fdata()
            
            if shape[cout] == labelArray.shape:
                pass
            else: 
                print(file + '\'s shape doesn\'t match.')
                print('sample size:' + str(shape[cout]))
                print('label size:' + str(labelArray.shape))
                raise AssertionError('Shape doesn\'t match')
            
            if labelArray.shape[:2]== (cg.image_size, cg.image_size):
                pass
            else:
               print(file + ' ' + 'size:' + str(labelArray.shape))
               labelArray = resize(labelArray, cg.image_size)
               print(file + ' ' + 'Resized:' + str(labelArray.shape)) 
            label = np.concatenate((label, labelArray), axis= 2)
            print(file + ' ' + 'loaded & appended.')
            cout += 1
        except nib.filebasedimages.ImageFileError :
            pass
        # 防止隐藏文件
    
    sample = sample[:,:,1:]
    label = label[:,:,1:]
    # nii体素值非灰度值，可以在[0,255]之外

    print('All samples loaded & npy shape is:' + str(sample.shape))
    print('All labels loaded & npy shape is:' + str(label.shape))

    np.save(outputfolder + '/' + sampleFolder + '/' + 'sample1-' + str(len(labelFile)) + '_' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+'.npy', sample)
    np.save(outputfolder + '/' + labelFolder + '/' + 'label1-' + str(len(labelFile)) + '_' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+'.npy', label)
    
    print('samples have been saved in' + outputfolder + '/' + sampleFolder)
    print('labels have been saved in' + outputfolder + '/' + labelFolder)


    
if __name__ == '__main__':
    main(sys.argv)
     
    