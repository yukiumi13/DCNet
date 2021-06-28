import numpy as np
import torch
import cv2
import os
'''
for file in os.listdir('../SrcData/Sample'):
    try:
        a = nib.load('../SrcData/Sample/' + file).get_fdata()
        print(file)
        print(a.shape)
    except nib.filebasedimages.ImageFileError :
        pass
'''
print(os.path.exists('../CVdataset/HKU-IS/img/.DS_store'))