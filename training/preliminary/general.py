from keras.models import Model, load_model
from keras.optimizers import adam_v2
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
from keras.utils.np_utils import normalize
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # change to CPU
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
from tabulate import tabulate
from unet_hybrid import segment_model
import segmentation_models as sm
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.losses import BinaryFocalCrossentropy

kernel_size = 3
image_magnification = 'x20'

type_model = 'unet_%s_hybrid' % kernel_size

SIZE = 256 
batch_size = 24
epochs = 50

data_directory = 'D:/Pycharm Projects/Cell Segmentation/1-data/'
model_directory = 'D:/Pycharm Projects/Cell Segmentation/4-model/%s/%s/' % (type_model, image_magnification)
    
trial_num = 2

model_directory = model_directory + '%s/' % (trial_num)

if image_magnification == 'x4':
    image_format = 'tiff'
elif image_magnification == 'x20' or 'all':
    image_format = 'tif'

image_directory = data_directory + '%s/images/' % image_magnification
mask_directory = data_directory + '%s/masks/' % image_magnification

save_logs = model_directory + 'logs.txt'
save_model = model_directory + '%s.%s.{epoch:02d}-{val_loss:.4f}.hdf5' % (type_model, batch_size)
save_history = model_directory + 'trainHistoryDict'
graph_file = model_directory + 'graph{%s}.png' % batch_size
dm_file = model_directory + 'cm{%s}.png' % batch_size
result_file = '/'.join(model_directory.split('/')[:-2]) + '/results_%s.txt' % type_model

image_dataset = [] 
mask_dataset = []  

images = os.listdir(image_directory)
images.sort()
# print(images)

num_images = 2000 # # limit number of training images

for i, image_name in enumerate(images):  
    if i == num_images: 
        i = i - 1
        break
    if (image_name.split('.')[1] == image_format):
        # print(image_directory+image_name)
        image = cv2.imread(image_directory + image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))
print("number of images:\n", i + 1)

masks = os.listdir(mask_directory)
masks.sort()
# print(masks)

for i, image_name in enumerate(masks):
    if i == num_images: 
        i = i - 1
        break
    if (image_name.split('.')[1] == image_format):
        # print(mask_directory+image_name)
        image = cv2.imread(mask_directory + image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))
print("number of masks:\n", i + 1)

# make sure number of images and masks are same

# Normalize images
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

import random
for x in range(10):
    
    image_number = random.randint(0, len(X_train))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
    plt.show()

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

model = segment_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel_size)

_model = model_directory + '%s.h5'%type_model
csv_logger =CSVLogger(save_logs, append=True, separator=";")

checkpointer = ModelCheckpoint(_model, verbose=1, save_best_only=True, monitor='val_loss')
# checkpointer = ModelCheckpoint(save_model, verbose=1, save_weights_only=True, monitor='val_loss', mode ='min', save_best_only=True)

callbacks = [csv_logger, checkpointer]

start = datetime.now()

iou_score = sm.metrics.IOUScore(threshold=0.5)
f_score = sm.metrics.FScore(threshold=0.5)
hybrid_metrics = [iou_score, f_score]

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
hybrid_loss = dice_loss + (1 * focal_loss)

model.compile(optimizer='adam', loss=hybrid_loss, metrics=[hybrid_metrics])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    verbose=1,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(X_test, y_test),
                    shuffle=False)

run_time = datetime.now() - start

# model.save(_model)

with open(save_history, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
# history = pickle.load(open('/trainHistoryDict'), "rb")

###############################################################################################
# evaluation

model = load_model(_model,
                   custom_objects={"dice_loss_plus_1binary_focal_loss":hybrid_loss,
                                   "iou_score":iou_score,
                                   "f1-score":f_score},
                   compile=True)
# model.load_weights(model_directory+latest_model)

# evaluate model
loss_, iou_score_, fscore_ = model.evaluate(X_test, y_test)
print(_model)
print("Loss = ", (loss_ * 100.0), "%")
print("IoU = ", iou_score_)
print("F1 = ", fscore_)

###############################################################################################
# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.ylim(0, 1)

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.ylim(0, 1)
plt.savefig(graph_file)
plt.show()
###############################################################################################
# confusion matrix of test data
import seaborn as sns

y_pred = model.predict(X_test)
y_pred_thresholded = y_pred > 0.5
FP = len(np.where(y_pred_thresholded - y_test == -1)[0])
FN = len(np.where(y_pred_thresholded - y_test == 1)[0])
TP = len(np.where(y_pred_thresholded + y_test == 2)[0])
TN = len(np.where(y_pred_thresholded + y_test == 0)[0])
cmat = [[TP, FN], [FP, TN]]

plt.figure(figsize=(6, 6))
sns.heatmap(cmat / np.sum(cmat), cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
plt.xlabel("predictions")
plt.ylabel("real values")
plt.savefig(dm_file)
plt.show()

###############################################################################################
# IoU of test data
y_pred = model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU score is: ", iou_score)

###############################################################################################
# Predict on a few images
model = load_model(_model,
                   custom_objects={"dice_loss_plus_1binary_focal_loss":hybrid_loss,
                   "iou_score":iou_score, "f1-score":f_score}, compile=True)


test_img_number = random.randint(0, len(X_test-1))
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_norm = test_img[:, :, 0][:, :, None]
test_img_input = np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.2).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, 0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.show()
