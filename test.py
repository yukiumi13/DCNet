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
a = np.random.normal(size = [10,9,8])
print(a.shape)
b = np.transpose(a, (2,0,1))
print(b.shape)
assert a[4,5,6]==b[6,4,5]