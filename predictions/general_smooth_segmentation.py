import tensorflow as tf
from keras.models import load_model
from keras.utils.np_utils import normalize
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import copy
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
import segmentation_models as sm
from skimage import io, img_as_float
from skimage.restoration import denoise_tv_chambolle
from skimage import io, filters, restoration, util
from skimage.restoration import denoise_nl_means, estimate_sigma
import bm3d

####################################################################################################################
# GPU usage

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # change to CPU
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

####################################################################################################################

def segmentation_map_to_image(
    result: np.ndarray, colormap: np.ndarray, remove_holes=False
) -> np.ndarray:

    if len(result.shape) != 2 and result.shape[0] != 1:
        raise ValueError(
            f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
        )

    if len(np.unique(result)) > colormap.shape[0]:
        raise ValueError(
            f"Expected max {colormap[0]} classes in result, got {len(np.unique(result))} "
            "different output values. Please make sure to convert the network output to "
            "pixel values before calling this function."
        )
    elif result.shape[0] == 1:
        result = result.squeeze(0)

    result = result.astype(np.uint8)

    contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
    mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for label_index, color in enumerate(colormap):
        label_index_map = result == label_index
        label_index_map = label_index_map.astype(np.uint8) * 255
        contours, hierarchies = cv2.findContours(
            label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            mask,
            contours,
            contourIdx=-1,
            color=color.tolist(),
            thickness=cv2.FILLED,
        )

    return mask


root = Tk()
# hide root window
root.overrideredirect(True)
root.geometry('0x0+0+0')

# lift window to top level
root.lift()
root.attributes('-topmost',True)
root.after_idle(root.attributes,'-topmost',False)

# model_path = askopenfilename(title='Choose model', \
#                               initialdir='D:/Pycharm Projects/Cell Segmentation/4-model/')

# image_magnification = 'x4'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/unet_3/x4/3/S256_BS24_LR0.001_E50_unet_3.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/msunet_2.0/x4/5/S256_BS24_LR0.001_E50_msunet_2.0.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/dilated_1/x4/4/S256_BS24_LR0.001_E50_dilated_1.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/residual_1_2.0/x4/5/S256_BS24_LR0.001_E50_residual_1_2.0.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/unet_5/x4/1/S256_BS24_LR0.001_E50_unet_5.h5'

# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/mobilenet/x4/3/S256_BS24_LR0.001_E50_mobilenet.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/EfficientNetV2BO/x4/1/S256_BS24_LR0.001_E50_EfficientNetV2BO.h5'


# image_magnification = 'x10'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/unet_3/x10/4/S256_BS24_LR0.001_E50_unet_3.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/msunet_2.0/x10/5/S256_BS24_LR0.001_E50_msunet_2.0.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/dilated_like_msunet/x10/2/S256_BS24_LR0.001_E50_dilated_1.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/residual_1_2.0/x10/2/S256_BS24_LR0.001_E50_residual_1_2.0.h5'

# image_magnification = 'x20'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/unet_3/x20/1/S256_BS24_LR0.001_E50_unet_3.h5'
_model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/msunet_2.0/x20/3/S256_BS24_LR0.001_E50_msunet_2.0.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/dilated_1/x20/5/S256_BS24_LR0.001_E50_dilated_1.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/residual_1/x20/4/S256_BS24_LR0.001_E50_residual_1.h5'

# merged dataset
# image_magnifications = ['x4', 'x10', 'x20']
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/merged_network/x20/4/S256_BS24_LR0.001_E50_merged_network.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/merged_inception/x20/5/S256_BS24_LR0.001_E50_merged_inception.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/merged_dilation/x20/1/S256_BS24_LR0.001_E50_merged_dilation.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/merged_residual/x20/2/S256_BS24_LR0.001_E50_merged_residual.h5'

# _model = 

model_path = _model

# model = load_model(model_path, compile=True)
iou_score = sm.metrics.IOUScore(threshold=0.5)
f_score = sm.metrics.FScore(threshold=0.5)
hybrid_metrics = [iou_score, f_score]

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
hybrid_loss = dice_loss + (1 * focal_loss)
######################################################################################
# model = load_model(model_path, custom_objects={"binary_focal_loss":focal_loss,
#                    "iou_score":iou_score},
#                    compile=True)

######################################################################################
model = load_model(model_path,
                    custom_objects={"dice_loss_plus_1binary_focal_loss":hybrid_loss,
                                    "iou_score":iou_score,
                                    "f1-score":f_score},
                    compile=True)
from memory_usage import check_memory
# check_memory(24, model, by_layer = True)
######################################################################################
# model = load_model(model_path, compile=True)

print('MODEL NAME:', model_path.split('/')[-1])

# path = askopenfilename(title='Choose test image', \
#                         initialdir='D:/Pycharm Projects/cell images (all magnifications)/')

path = 'D:/Pycharm Projects/cell images (all magnifications)/x4/M4-N80k-B35-0001.tif'
path = 'D:/Pycharm Projects/cell images (all magnifications)/x10/10X_55_002.jpg'
path = 'D:/Pycharm Projects/cell images (all magnifications)/x20/d1-c2-007.jpg'
# path = 'D:/Pycharm Projects/Cell Segmentation 2/all scales/x10/10X_55_001.jpg'
print('IMAGE FILE:', path.split('/')[-1])
large_image = cv2.imread(path)
file_name = os.path.basename(path)
split_name = file_name.split('.')
bgr_image = large_image
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
#############################################################################
lab_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_img)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl_img = clahe.apply(l)
updated_lab_img2 = cv2.merge((cl_img, a, b))
bgr_image = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
large_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

#############################################################################
# img = img_as_float(large_image)
# img_denoise = denoise_tv_chambolle(img, weight=0.1, eps=0.0002, max_num_iter=200, channel_axis=False)
# img_denoise = util.img_as_ubyte(img_denoise)
# img_denoise = cv2.cvtColor(img_denoise, cv2.COLOR_BGR2RGB)
# large_image = cv2.resize(img_denoise, (large_image.shape[1], large_image.shape[0]))
# large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
#############################################################################

# predict by patches
# large image to small patches
# patches = patchify(large_image, (256, 256), step=256)
mean = []
start = time.time()
patch_size = (256, 256)
stride = (128, 128)

# Create an empty mask to store the segmented image
mask = np.zeros_like(large_image)
# Iterate over each patch
k =0
for y in range(0, large_image.shape[0] - patch_size[0] + stride[0], stride[0]):
    for x in range(0, large_image.shape[1] - patch_size[1] + stride[1], stride[1]):

        # Extract the current patch
        patch = large_image[y:y+patch_size[0], x:x+patch_size[1]]
        position = '{}:{}, {}:{}'.format(y, y+patch_size[0], x ,x+patch_size[1])
        single_patch_norm = np.expand_dims(normalize(np.array(patch), axis=1), 2)
        single_patch_input = np.expand_dims(single_patch_norm, 0)

        # Perform the segmentation on the patch using your deep learning model
        predicted_patch = (model.predict(single_patch_input, verbose=1)[0, :, :, 0] > 0.5).astype(np.uint8)
        predicted_patch = predicted_patch.reshape(256, 256)

        # Blend the predicted patch with the mask
        mask[y:y+patch_size[0], x:x+patch_size[1]] += predicted_patch
        # plt.figure(figsize=(60, 30))
        # plt.title(position, fontsize = 80)
        # plt.imshow(mask)
        # plt.axis('off')
        k = k + 1
        print(k)

# plt.figure(figsize=(60, 30))
# plt.title(position, fontsize = 80)
# plt.imshow(mask)
# plt.axis('off')
# Normalize the mask
mask = mask / np.max(mask)
# plt.figure(figsize=(60, 30))
# plt.title(position, fontsize = 80)
# plt.imshow(mask)
# plt.axis('off')

# Apply a threshold to convert the mask to a binary image
threshold = 0.2
reconstructed_image = (mask > threshold).astype(np.uint8)
end = time.time()
print('Time taken: {:.2f} seconds'.format(end - start))
# plt.imshow(reconstructed_image)
#############################################################################

num_white = np.sum(reconstructed_image == 1)
num_black = np.sum(reconstructed_image == 0)
confluency = (num_white/(num_white+num_black))*100
print('Confluency: %.2f' % confluency + '%')

#############################################################################
# plt.imshow(reconstructed_image, cmap='gray')
# plt.imsave('results/'+split_name[0]+'.jpg', reconstructed_image, cmap='gray')

# plt.figure(figsize=(8, 8))
# plt.subplot(121)
# plt.title('Large Image')
# plt.imshow(large_image, cmap='gray')
# plt.subplot(122)
# plt.title('Prediction of large Image')
# plt.imshow(reconstructed_image, cmap='gray')
# plt.imsave('predicted_' + split_name[0] + '.jpg', reconstructed_image, cmap='gray')
# plt.show()

#############################################################################

height, width = reconstructed_image.shape

colormap = np.array([[0, 0, 0], [0, 255, 0]])

# Define the transparency of the segmentation mask on the photo
alpha = 0.2

# Use function from notebook_utils.py to transform mask to an RGB image
mask = segmentation_map_to_image(reconstructed_image, colormap)

resized_mask = cv2.resize(mask, (width, height))

# Create image with mask put on
image_with_mask1 = cv2.addWeighted(bgr_image, 1-alpha, resized_mask, alpha, 0)
image_with_mask = cv2.cvtColor(image_with_mask1, cv2.COLOR_BGR2RGB)

##############################################################################

# masked = np.ma.masked_where(reconstructed_image == 0, reconstructed_image)

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(large_image, 'gray', interpolation='none')
# plt.axis('off')

# plt.subplot(1,3,2)
# plt.imshow(reconstructed_image, 'gray', interpolation='none')
# plt.axis('off')

# plt.subplot(1,3,3)
# plt.imshow(large_image, 'gray', interpolation='none')
# plt.imshow(masked, 'jet', interpolation='none', alpha=0.5)
# plt.axis('off')

# plt.show()
plt.figure(figsize=(60, 30))
plt.imshow(rgb_image, interpolation='none')
plt.axis('off')

# plt.savefig('clahe_image_0.png')
plt.show()

# plt.imshow(large_image, 'gray', interpolation='none')
# plt.imshow(rgb_image, interpolation='none')
# plt.imshow(masked, 'jet', interpolation='none', alpha=0.5)


plt.figure(figsize=(60, 30))
# plt.title('Masked Image')
plt.imshow(image_with_mask)
plt.axis('off')

# plt.savefig('masked_image_0.png')
plt.show()

plt.figure(figsize=(60, 30))
plt.imshow(reconstructed_image, 'gray', interpolation='none')
plt.axis('off') 

# plt.savefig('segmented_image_0.png')
plt.show()

# cv2.imwrite('10x20_image_with_mask.jpg', image_with_mask1)
