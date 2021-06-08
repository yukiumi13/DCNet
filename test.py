import nibabel, cv2 , os

sample = nibabel.load('../SrcData/Sample/sample1.nii.gz').get_fdata()
sample = sample[:,:,1]
cv2.imshow('imshow',sample)