import os
from patchify import patchify
import tifffile as tiff
import cv2
import numpy as np

image_directory = 'D:/Pycharm Projects/Cell Segmentation 2/data_256/unfiltered/x10/images/'
mask_directory = 'D:/Pycharm Projects/Cell Segmentation 2/data_256/unfiltered/x10/masks/'

image_to = 'D:/Pycharm Projects/Cell Segmentation 2/data_256/x10/images/'
mask_to = 'D:/Pycharm Projects/Cell Segmentation 2/data_256/x10/masks/'

image_throw = 'D:/Pycharm Projects/Cell Segmentation 2/data_256/throw/x10/images/'
mask_throw = 'D:/Pycharm Projects/Cell Segmentation 2/data_256/throw/x10/masks/'

min_pixels = 2000

os.makedirs(image_to, exist_ok = True)
os.makedirs(mask_to, exist_ok = True)
os.makedirs(image_throw, exist_ok = True)
os.makedirs(mask_throw, exist_ok = True)


images = os.listdir(image_directory)
images.sort()
masks = os.listdir(mask_directory)
masks.sort()

for i, mask_name in enumerate(masks):
    mask = cv2.imread(mask_directory+masks[i])
    image = cv2.imread(image_directory+images[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_name = masks[i].split('.')[0]
    image_name = images[i].split('.')[0]
    if np.sum(mask == 255)>min_pixels:
        # pixels=np.sum(mask == 255)
        # print(pixels)
        
        tiff.imwrite(mask_to + mask_name + ".tiff", mask)
        tiff.imwrite(image_to + image_name + ".tiff", image)
        print(i)
    else:
        tiff.imwrite(mask_throw + mask_name + ".tiff", mask)
        tiff.imwrite(image_throw + image_name + ".tiff", image)
        