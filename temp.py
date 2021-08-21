import numpy as np
# import SimpleITK as sitk
import matplotlib.pyplot as plt
# from utility import *
import nibabel as nib
import nibabel.processing as proc
'''Create Grid Image
grid = sitk.GridSource(outputPixelType=sitk.sitkUInt16,
    size=(250, 250),
    sigma=(0.2, 0.2),
    gridSpacing=(5.0, 5.0),
    gridOffset=(0.0, 0.0),
    spacing=(0.2,0.2))
myshow(grid)
'''
'''
def threshold_based_crop(image):
    """
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.                                 
    """
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute( sitk.OtsuThreshold(image, inside_value, outside_value) )
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
    

mr_image = sitk.ReadImage('../SrcData/Sample/sample1.nii.gz')

mr_roi = threshold_based_crop(mr_image)
disp_images(mr_roi)

'''

img = nib.load('../SrcData/Sample/sample20.nii.gz')
obj = nib.load('../SrcData/Sample/sample7.nii.gz')
# print(header)
# img = img.get_fdata()
shape = img.header.get_data_shape()
slice_num = shape[2]
'''
img_resized = proc.conform(img,out_shape=(256,256,slice_num))
print(img_resized.header.get_data_shape())
img_resized_array = img_resized.get_fdata()
print(np.max(img_resized_array))
'''
# image_resize_array = img_resized.get_fdata()
# img_resampled = proc.resample_to_output(img,voxel_sizes=[1,1,3])
'''
img_resampled = proc.conform(img,out_shape=shape,voxel_size=[1,1,img.header['pixdim'][3]])
img_array = img.get_fdata()
img_resampled_array = img_resampled.get_fdata() 
img2show = img_array[:,:,54]
img_resampled2show = img_resampled_array[:,:,54]
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img2show, cmap="gray")
plt.title('before')
plt.subplot(1,2,2)
plt.imshow(img_resampled2show, cmap="gray")
plt.title('after')
print(np.max(abs(img2show-img_resampled2show)))
aff = nib.orientations.aff2axcodes(img.affine)
aff_resampled = nib.orientations.aff2axcodes(img_resampled.affine)
print(img.header['pixdim'][1:4])
print(img_resampled.header['pixdim'][1:4])
# print(img.header)
# print(img_resampled.header)
'''
label = nib.load('../SrcData/Label/mask-20.nii.gz')
img_array = img.get_fdata()
label_array = label.get_fdata()

idx = []
for i in range(0,slice_num):
    if np.max(img_array[:,:,i])== 0:
        pass
    elif np.max(label_array[:,:,i]) == 0:
        pass
    else:
        idx.append(i)
img_resampled = proc.conform(img, out_shape=shape, voxel_size=[1,1,img.header['pixdim'][3]])
img_resampled_array = img_resampled.get_fdata()
label_resampled = proc.conform(label,out_shape=shape, voxel_size=[1,1,img.header['pixdim'][3]])
label_resampled_array = label_resampled.get_fdata()
img_s = img_resampled_array[:,:,idx]
label_s = label_resampled_array[:,:,idx]
for i in range(0,len(idx)):
    img2show = img_s[:,:,i]
    label2show = label_s[:,:,i]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img2show, cmap="gray")
    plt.title('sample')
    plt.subplot(1,2,2)
    plt.imshow(label2show, cmap="gray")
    plt.title('label')

    
    
    
    