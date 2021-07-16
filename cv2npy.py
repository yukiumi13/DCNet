#coding=utf-8#
from skimage import io
import numpy as np
import os
import sys
import cv2
from config import Config as cg

def main():
    #########################  生成npy文件 #####################################
    ##############################################################
    dataRoot = '../CVdataset'
    datasetName = 'HKU-IS'
    imgPath = dataRoot + '/' + datasetName + '/' + 'img'
    imgFiles = os.listdir(imgPath)
    img = data_process_img(imgPath, imgFiles)
    print('img shape is ' + str(img.shape))
    '''若img格式为jpg
    maskFiles = []
    for i in imgFiles:
        temp = i.replace('jpg','png')
        maskFiles.append(temp)  
    '''
    np.save('../' + 'npys' + '/' + datasetName + '/'+'IMG' + '/' + 'sample' + '/' + 'sample' + '.npy', img)
    print('numpy saved in' + '../' + 'npys' + '/' + datasetName + '/'+'IMG' + '/' + 'sample' + '/' + 'sample' + '.npy')
    maskPath = dataRoot + '/' + datasetName + '/' + 'gt'
    mask = data_process_mask(maskPath, imgFiles)
    print('mask shape is ' + str(mask.shape)) 
    np.save('../' + 'npys' + '/' + datasetName + '/'+'IMG' + '/' + 'label' + '/' + 'label' + '.npy', mask)
    print('numpy saved in' + '../' + 'npys' + '/' + datasetName + '/'+'IMG' + '/' + 'label' + '/' + 'label' + '.npy')


    ###############################################################
    # path = path_curr + "\\raw_data\\IVUS\\mask_ma\\"
    # img_flat = data_process_mask_ivus(path)
    # np.save("train_masks_ma_ivus.npy", img_flat)
    # ##############################################################
    # path = path_curr + "\\raw_data\\OCT\\img\\"
    # img_flat = data_process_img_oct(path)
    # np.save("train_images_oct.npy", img_flat)
    # # # ###############################################################
    # path = path_curr + "\\raw_data\\oct\\mask_lumen\\"
    # img_flat = data_process_mask_oct(path)
    # np.save("train_masks_lumen_oct.npy", img_flat)
    #
    # ##########################  测试npy文件 #####################################



def data_process_img(path, content):

    imgs=[] 
    for i in content :
        img_path = path + '/' + i
        print('processing image', img_path)
        img = cv2.imread(img_path) # cv2.imread是按照BGR的顺序读的图像
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])

        # img[1:20, 1:77, :] = 30
        # img[1:22, 279:384, :] = 30
        # img[360:384, 1:93, :] = 30
        # img[361:384, 329:384, :] = 30

        img = cv2.resize(img, (cg.image_size, cg.image_size))
        #img = img_normalize(img) # 图像标准化
        # io.imshow(img)
        # io.show()
        imgs.append(img)
    
    imgs = np.stack(imgs)
    print(str(len(content)) +' images loaded')
    print(imgs.shape)
    imgs = np.transpose(imgs,(0,3,1,2))
    print('img shape is ' + str(imgs.shape))

    return imgs

def data_process_mask(path, content):
    imgs = []
    for i in content:
        img_path = path + '/' + i
        print('processing mask', img_path)
        img = cv2.imread(img_path)
        if 3 == img.ndim:
            img = img[:, :, 0]

        img = cv2.resize(img, (256, 256))

        # 插值后会有数据变得不是0和1, 下面进行二值化
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # io.imshow(img)
        # io.show()
        imgs.append(img)

    imgs = np.stack(imgs,axis=0)
    imgs = np.expand_dims(imgs, axis = 1)
    print(str(len(content)) +' masks loaded')
    print('mask shape is ' + str(imgs.shape)) 

    return imgs

def data_process_img_oct(path):
    content = os.listdir(path)

    img_flat_mat = []
    for i in range(len(content)):
        img_path = path + content[i]
        img = cv2.imread(img_path)  # cv2.imread是按照BGR的顺序读的图像
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])

        # 有三种不同尺寸的OCT
        if 960 == img.shape[0] & 960 == img.shape[1]:
            img[1:65, 1:255, :] = 0
            img[1:70, 730:960, :] = 0
            img[640:710, 750:960, :] = 0
            img = img[1:719, 120:120+720-1, :]
        elif 704 == img.shape[0] & 704 == img.shape[1]:
            img[1:60, 1:250, :] = 0
            img[1:53, 497:704, :] = 0
            img[462:520, 550:704, :] = 0
            img = img[1:527, 88:88+527-1, :]
        elif 848 == img.shape[0] & 848 == img.shape[1]:
            img[1:60, 1:250, :] = 0
            img[1:53, 640:848, :] = 0
            img[560:625, 670:848, :] = 0
            img = img[1:635, 106:106+635-1, :]


        img = cv2.resize(img, (256, 256))
        img_flat = np.reshape(img, [1, -1])
        if 0 == i:
            img_flat_mat = img_flat
        else:
            img_flat_mat = np.vstack((img_flat_mat, img_flat))

        # io.imshow(img)
        # io.show()
        # a=0

    return img_flat_mat

def data_process_mask_oct(path):
    content = os.listdir(path)

    img_flat_mat = []
    for i in range(len(content)):
        img_path = path + content[i]
        img = cv2.imread(img_path)
        if 3 == img.ndim:
            img = img[:, :, 0]

            # 有三种不同尺寸的OCT
            if 960 == img.shape[0] & 960 == img.shape[1]:
                img = img[1:719, 120:120+720-1]
            elif 704 == img.shape[0] & 704 == img.shape[1]:
                img = img[1:527, 88:88+527-1]
            elif 848 == img.shape[0] & 848 == img.shape[1]:
                img = img[1:635, 106:106+635-1]

        img = cv2.resize(img, (256, 256))

        # 插值后会有数据变得不是0和1, 下面进行二值化
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        img_flat = np.reshape(img, [1, -1])
        if 0 == i:
            img_flat_mat = img_flat
        else:
            img_flat_mat = np.vstack((img_flat_mat, img_flat))

        # io.imshow(img)
        # io.show()
        # a=0

    return img_flat_mat
'''
def img_normalize(images):
  images = images.astype(np.float64)
  images[:, :, 2] -= np.mean(images[:,:,2])
  images[:, :, 1] -= np.mean(images[:,:,1])
  images[:, :, 0] -= np.mean(images[:,:,0])

  images[:, :, 2] /= ( np.std(images[:,:,2]) + 1e-12)
  images[:, :, 1] /= ( np.std(images[:,:,1]) + 1e-12)
  images[:, :, 0] /= ( np.std(images[:,:,0]) + 1e-12)
  return images
'''

if __name__ == '__main__':
    main()