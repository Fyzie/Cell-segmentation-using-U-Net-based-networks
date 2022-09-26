import os
from patchify import patchify
import tifffile as tiff
import cv2

image_directory = '1-data/x4/images/'
mask_directory = '1-data/x4/masks/'

images = os.listdir(image_directory)
images.sort()
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if image_name.split('.')[1] == 'tiff':
        image = tiff.imread(image_directory+image_name)
        patches_img = patchify(image, (256, 256,3), step=256)
        for j in range(patches_img.shape[0]):
            for k in range(patches_img.shape[1]):
                single_patch_img = patches_img[j, k, 0, :, :, :]
                tiff.imwrite('1-data/image_patches/' + 'image_' + '_' + str(i) + '_' +str(j) + str(k) + '.tiff', single_patch_img)

masks = os.listdir(mask_directory)
# masks.sort()
for i, image_name in enumerate(masks):
    if image_name.split('.')[1] == 'tiff':
        image = tiff.imread(mask_directory+image_name)
        patches_mask = patchify(image, (256, 256), step=256)
        for j in range(patches_mask.shape[0]):
            for k in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[j,k, :,:]
                tiff.imwrite('1-data/mask_patches/' + 'mask_' + '_' + str(i) + '_' + str(j)+str(k)+ ".tiff", single_patch_mask)
                single_patch_mask = single_patch_mask / 255.
