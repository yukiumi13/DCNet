import numpy as np
import nibabel as nib
import os
import math
import time
'''
for file in os.listdir('../SrcData/Sample'):
    try:
        a = nib.load('../SrcData/Sample/' + file).get_fdata()
        print(file)
        print(a.shape)
    except nib.filebasedimages.ImageFileError :
        pass
'''
a = (1, 2, 3)
print(a[:2])