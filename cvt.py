from utility import *
for i in [22,24,25,26,29,31,33,37,40]:
    cvt_dcm_nii('../test_data/sample/sample'+str(i), '../test_data/'+str(i)+'.nii.gz')