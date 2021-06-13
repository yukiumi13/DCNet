import numpy as np
'''
for file in os.listdir('../SrcData/Sample'):
    try:
        a = nib.load('../SrcData/Sample/' + file).get_fdata()
        print(file)
        print(a.shape)
    except nib.filebasedimages.ImageFileError :
        pass
'''
data = np.load('../SrcData/IMG/sample/sample.npy')
data = np.squeeze(data)