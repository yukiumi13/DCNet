import numpy as np
import torch
'''
for file in os.listdir('../SrcData/Sample'):
    try:
        a = nib.load('../SrcData/Sample/' + file).get_fdata()
        print(file)
        print(a.shape)
    except nib.filebasedimages.ImageFileError :
        pass
'''
print(torch.cuda.is_available())