import numpy as np
from numpy.core.fromnumeric import shape
import torch
import cv2
'''
for file in os.listdir('../SrcData/Sample'):
    try:
        a = nib.load('../SrcData/Sample/' + file).get_fdata()
        print(file)
        print(a.shape)
    except nib.filebasedimages.ImageFileError :
        pass
'''
'''
imgs = []
img = cv2.imread('../CVdataset/HKU-IS/img/1804.png')
print(img.shape)
imgs = imgs.append(img)
print(imgs.shape)
'''
t=[]
a = np.random.randn(5,5,3)
t.append(a)
b = np.random.randn(5,5,3)
t.append(b)
c = np.stack(t,axis=0)
print(c.shape)