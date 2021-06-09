import nibabel, cv2 , os
import numpy as np
import imageio
from nii2png import *


#########################
# imageio 自动类型转换
#########################
'''
sample = nibabel.load('../SrcData/Sample/sample1.nii.gz').get_fdata()
sample_1 = sample[:,:,50]
sample_1 = sample_1.astype('uint8').T
sample_2 =sample[:,:,51]
imageio.imwrite('test.png',sample_1)
'''
#########################
# ceshi.py
#########################
# convert('../SrcData/Sample/sample1.nii.gz','../test','test')
#########################
# 手动uint8
#########################
sample = nibabel.load('../SrcData/Sample/sample1.nii.gz').get_fdata()
sample_1 = sample[:,:,50]
sample_1 = sample_1/(np.max(sample_1))*255
sample_1 = sample_1.astype('uint8')
imageio.imwrite('test.png',sample_1)

