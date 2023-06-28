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
from PIL import Image

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

# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/msunet_2.0/x4/5/S256_BS24_LR0.001_E50_msunet_2.0.h5'

# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/msunet_2.0/x10/5/S256_BS24_LR0.001_E50_msunet_2.0.h5'

# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/msunet_2.0/x20/3/S256_BS24_LR0.001_E50_msunet_2.0.h5'

# x4
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/unet_3/x4/3/S256_BS24_LR0.001_E50_unet_3.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/msunet_2.0/x4/5/S256_BS24_LR0.001_E50_msunet_2.0.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/mobilenet/x4/3/S256_BS24_LR0.001_E50_mobilenet.h5'
# _model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/EfficientNetV2BO/x4/1/S256_BS24_LR0.001_E50_EfficientNetV2BO.h5'

# x10
_model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/unet_3/x10/4/S256_BS24_LR0.001_E50_unet_3.h5'
_model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/msunet_2.0/x10/5/S256_BS24_LR0.001_E50_msunet_2.0.h5'
_model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/mobilenet/x10/5/S256_BS24_LR0.001_E50_mobilenet.h5'
_model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/EfficientNetV2BO/x10/4/S256_BS24_LR0.001_E50_EfficientNetV2BO.h5'

# x20
_model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/unet_3/x20/1/S256_BS24_LR0.001_E50_unet_3.h5'
_model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/msunet_2.0/x20/3/S256_BS24_LR0.001_E50_msunet_2.0.h5'
_model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/mobilenet/x20/3/S256_BS24_LR0.001_E50_mobilenet.h5'
_model = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/EfficientNetV2BO/x20/1/S256_BS24_LR0.001_E50_EfficientNetV2BO.h5'

iou_score = sm.metrics.IOUScore(threshold=0.5)
f_score = sm.metrics.FScore(threshold=0.5)
hybrid_metrics = [iou_score, f_score]

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
hybrid_loss = dice_loss + (1 * focal_loss)
######################################################################################
model = load_model(_model,
                    custom_objects={"dice_loss_plus_1binary_focal_loss":hybrid_loss,
                                    "iou_score":iou_score,
                                    "f1-score":f_score},
                    compile=True)
######################################################################################
# image = askopenfilename(title='Choose test image', \
#                        initialdir='D:/Academics/3 - Master/IMR Collaboration/images/')
# image = 'D:/Academics/3 - Master/IMR Collaboration/images/2-20220119-01.tiff'
# image = 'D:/Academics/3 - Master/IMR Collaboration/images/1-F1_10X_60_013.tiff'
image = 'D:/Academics/3 - Master/IMR Collaboration/images/3-d1-c3-007.tiff'
mask_path = 'D:/Academics/3 - Master/IMR Collaboration/masks'
name = image.split('/')[-1].split('-')[0]
print(name)
masks = os.listdir(mask_path)
masks.sort()
for mask in masks:
    if mask.startswith(name):
        mask = os.path.join(mask_path, mask)
        break

large_image = cv2.imread(image)

lab_img = cv2.cvtColor(large_image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_img)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl_img = clahe.apply(l)
updated_lab_img2 = cv2.merge((cl_img, a, b))
bgr_image = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
large_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

#############################################################################
# # predict by patches
# # large image to small patches
# patches = patchify(large_image, (256, 256), step=256)

# # start = time.time()
# predicted_patches = []
# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         # print(i, j)

#         single_patch = patches[i, j, :, :]
#         single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
#         single_patch_input = np.expand_dims(single_patch_norm, 0)

#         # Predict and threshold for values above 0.5 probability
#         single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(np.uint8)
#         predicted_patches.append(single_patch_prediction)

# predicted_patches = np.array(predicted_patches)
# predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256, 256))

# reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
# # end = time.time()
# # print('Time taken: {:.2f} seconds'.format(end - start))

#############################################################################
start = time.time()
patch_size = (256, 256)
stride = (128, 128)

# Create an empty mask to store the segmented image
mask_ = np.zeros_like(large_image)
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
        mask_[y:y+patch_size[0], x:x+patch_size[1]] += predicted_patch

# Normalize the mask
mask_ = mask_ / np.max(mask_)

# Apply a threshold to convert the mask to a binary image
threshold = 0.2
reconstructed_image = (mask_ > threshold).astype(np.uint8)
end = time.time()
print('Time taken: {:.2f} seconds'.format(end - start))
#############################################################################
# plt.imshow(reconstructed_image)
plt.show()
print(np.unique(reconstructed_image))
num_white = np.sum(reconstructed_image == 1)
num_black = np.sum(reconstructed_image == 0)
p_confluency = (num_white/(num_white+num_black))*100
print('Predicted confluency: %.2f' % p_confluency + '%')

#############################################################################

large_mask = cv2.imread(mask,0)
# plt.imshow(large_mask)
plt.show()
print(np.unique(large_mask))
num_white = np.sum(large_mask == 1)
num_black = np.sum(large_mask == 0)
gt_confluency = (num_white/(num_white+num_black))*100
print('Ground truth confluency: %.2f' % gt_confluency + '%')

#############################################################################

def numpytoimage(numpy):
    numpy = numpy * 255
    image= Image.fromarray(numpy.astype(np.uint8))
    return image


C = np.zeros(shape=(len(large_mask), len(large_mask[0]), 3))

green = 0
red = 0
blue = 0
for i in range (0, large_mask.shape[0],1):
    for j in range(0, large_mask.shape[1], 1):
        if large_mask[i][j] == reconstructed_image[i][j] and large_mask[i][j] == 0:
            C[i][j] = 0
        elif large_mask[i][j] == 0:
            C[i][j][0] = 0
            C[i][j][1] = 0
            C[i][j][2] = 1
            blue = blue + 1
        elif reconstructed_image[i][j] == 0:
            C[i][j][0] = 1
            C[i][j][1] = 0
            C[i][j][2] = 0
            red = red + 1
        else:
            C[i][j][0] = 0
            C[i][j][1] = 1
            C[i][j][2] = 0
            green = green + 1

V_image = numpytoimage(C)
precision = green/(green+red+blue)*100
print('Precision: ', precision, '%')
#############################################################################

# plt.figure(figsize=(60, 30))
plt.figure(figsize=(60, 30))
plt.subplot(131)
plt.title('Original Image')
plt.imshow(rgb_image)
plt.axis('off')

plt.subplot(132)
plt.title('Ground Truth')
plt.imshow(large_mask, cmap = 'gray')
plt.text(700,1650,'Confluency: {:.2f} %'.format(gt_confluency))
plt.axis('off')

# plt.subplot(133)
# plt.title('Predicted Mask')
# plt.imshow(reconstructed_image, cmap = 'gray')
# plt.text(700,1650,'Confluency: {:.2f} %'.format(p_confluency))
# plt.axis('off')

plt.subplot(133)
plt.title('Predicted Mask')
plt.imshow(V_image, cmap = 'gray')
plt.text(700,1650,'Confluency: {:.2f} %'.format(p_confluency))
plt.axis('off')

plt.show()

plt.figure(figsize=(60, 30))
plt.imshow(rgb_image)
plt.axis('off')
plt.show()

plt.figure(figsize=(60, 30))
plt.imshow(large_mask, cmap = 'gray')
plt.axis('off')
plt.show()

plt.figure(figsize=(60, 30))
plt.imshow(V_image, cmap = 'gray')
plt.axis('off')

plt.show()

# basename = image.split('/')[-1]
# image_name = 'image_' + basename
# ground_name = 'ground_' + basename
# predict_name = 'predict_' + basename
# cv2.imwrite(image_name, rgb_image)
# cv2.imwrite(ground_name, large_mask)
# cv2.imwrite(predict_name, V_image)
