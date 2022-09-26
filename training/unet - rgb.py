from keras.models import Model, load_model
from keras.optimizers import adam_v2
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
    


################################################################
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # optimizer = adam_v2.Adam(learning_rate=lr, decay=lr/epochs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model


from keras.utils.np_utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
import pickle

# !unzip -u "/content/drive/MyDrive/datasets/patchesx4.zip" -d "/content/drive/MyDrive/datasets"

image_directory = 'C:/Users/Hafizi/PycharmProjects/Cell Segmentation/1-data/x20/images/'
mask_directory = 'C:/Users/Hafizi/PycharmProjects/Cell Segmentation/1-data/x20/masks/'

SIZE = 256
image_dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
mask_dataset = []  # Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
images.sort()

for i, image_name in enumerate(images):  # Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(image_directory + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))
print("number of images:\n", i + 1)

masks = os.listdir(mask_directory)
masks.sort()
# print(masks)

for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(mask_directory + image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))
print("number of masks:\n", i + 1)

# make sure number of images and masks are same

# Normalize images
image_dataset = normalize(np.array(image_dataset), axis=1)
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

import random

image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
img = np.reshape(X_train[image_number], (256, 256, 3)).astype(np.float64)
plt.imshow(img)
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]


def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


model = get_model()

batch_size=24

# save_logs = 'C:/Users/Hafizi/PycharmProjects/Cell Segmentation/4-model/unet/x20/logs.txt'
save_model = 'C:/Users/Hafizi/PycharmProjects/Cell Segmentation/4-model/rgb_unet/x20/rgb_unet{%s}.hdf5'%batch_size
# save_history = 'C:/Users/Hafizi/PycharmProjects/Cell Segmentation/4-model/unet/x20/trainHistoryDict'

# csv_logger =CSVLogger(save_logs, append=True, separator=";")

# checkpointer = ModelCheckpoint(save_model, verbose=1, save_best_only=True, monitor='val_loss')

# callbacks = [csv_logger, checkpointer]

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    verbose=1,
                    epochs=50,
                    # callbacks=callbacks,
                    validation_data=(X_test, y_test),
                    shuffle=False)

model.save(save_model)

# with open(save_history, 'wb') as file_pi:
#         pickle.dump(history.history, file_pi)
        
# history = pickle.load(open('/trainHistoryDict'), "rb")

###############################################################################################
# evaluation
model = load_model(save_model, compile=True)

# evaluate model
loss_, acc_ = model.evaluate(X_test, y_test)
print("Loss = ", (loss_ * 100.0), "%")
print("Accuracy = ", (acc_ * 100.0), "%")

graph_file = 'C:/Users/Hafizi/PycharmProjects/Cell Segmentation/4-model/rgb_unet/x20/graph{%s}.png'%batch_size
dm_file = 'C:/Users/Hafizi/PycharmProjects/Cell Segmentation/4-model/rgb_unet/x20/cm{%s}.png'%batch_size
result_file = "C:/Users/Hafizi/PycharmProjects/Cell Segmentation/4-model/rgb_unet/x20/results.txt"

###############################################################################################
# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

fig, ax1 = plt.subplots()
ax1.plot(epochs, loss, 'b', label='train_loss')
ax1.plot(epochs, val_loss, 'g', label='val_loss')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')

ax2 = ax1.twinx()
ax2.plot(epochs, acc, 'r', label='train_acc')
ax2.plot(epochs, val_acc, 'k', label='val_acc')
ax2.set_ylabel('Accuracy')
plt.title('UNet')
ax1.grid()
ax2.grid()
handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
fig.legend(handles, labels, bbox_to_anchor=(0.9, 0.7))
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

# num_epoch = (latest_model.split('.')[2]).split('-')[0]

# f= open(result_file,"a+")
# f.write("Batch size: {}\nMax Epochs: {}\nAccuracy: {:.2f}%\nLoss: {:.2f}%\nIoU: {:.3f}\n\n".format(batch_size, num_epoch, acc_*100,loss_*100,iou_score))
# f.close()

#######################################################################
import random
from keras.models import load_model
model = load_model(save_model)

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.show


#plt.imsave('input.jpg', test_img[:,:,0], cmap='gray')
#plt.imsave('data/results/output2.jpg', prediction_other, cmap='gray')