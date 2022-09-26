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

root = Tk()
# hide root window
root.overrideredirect(True)
root.geometry('0x0+0+0')

# lift window to top level
root.lift()
root.attributes('-topmost',True)
root.after_idle(root.attributes,'-topmost',False)

model_path = askopenfilename(title='Choose model', \
                             initialdir='C:/Users/Hafizi/PycharmProjects/Cell Segmentation/4-model/')
model = load_model(model_path, compile=True)
print('MODEL NAME:', model_path.split('/')[-1])

path = askopenfilename(title='Choose test image', \
                       initialdir='C:/Users/Hafizi/PycharmProjects/Cell Segmentation/1-data/all scales/')
print('IMAGE FILE:', path.split('/')[-1])

large_image = cv2.imread(path)
file_name = os.path.basename(path)
split_name = file_name.split('.')
lab_img = cv2.cvtColor(large_image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_img)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl_img = clahe.apply(l)
updated_lab_img2 = cv2.merge((cl_img, a, b))
large_image = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
rgb_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)
large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)

#############################################################################
# predict by patches
# large image to small patches
patches = patchify(large_image, (256, 256), step=256)

predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        # print(i, j)

        single_patch = patches[i, j, :, :]
        single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
        single_patch_input = np.expand_dims(single_patch_norm, 0)

        # Predict and threshold for values above 0.5 probability
        single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(np.uint8)
        predicted_patches.append(single_patch_prediction)

predicted_patches = np.array(predicted_patches)
predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256, 256))

reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)

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

masked = np.ma.masked_where(reconstructed_image == 0, reconstructed_image)

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

plt.imshow(rgb_image, interpolation='none')
plt.axis('off')

plt.savefig('clahe_image_0.png')
plt.show()

plt.imshow(reconstructed_image, 'gray', interpolation='none')
plt.axis('off') 

plt.savefig('segmented_image_0.png')
plt.show()

plt.imshow(large_image, 'gray', interpolation='none')
plt.imshow(masked, 'jet', interpolation='none', alpha=0.5)
plt.axis('off')

plt.savefig('masked_image_0.png')
plt.show()
