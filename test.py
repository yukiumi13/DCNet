import nibabel
import cv2
image_array = nibabel.load('../SrcData/Label/mask-1.nii.gz').get_fdata()
data = image_array[:,:,0]
print(data)
cv2.imshow(data)
data = data.astype('uint8')
print(data)
cv2.imshow(data)