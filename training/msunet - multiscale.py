from keras.models import Model, load_model
from keras.optimizers import adam_v2
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
from keras.utils.np_utils import normalize
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
from tabulate import tabulate

def encoder_block(input1, feature, kernel1, kernel2, dropout):
    c1 = Conv2D(feature, (kernel1, kernel1), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
    c2 = Conv2D(feature, (kernel2, kernel2), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
    c1 = Dropout(dropout)(c1)
    c2 = Dropout(dropout)(c2)
    c1 = Conv2D(feature, (kernel1, kernel1), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c2 = Conv2D(feature, (kernel2, kernel2), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c = concatenate([c1, c2], axis=3)
    # c = Conv2D(16, (1,1), activation='relu', kernel_initializer='he_normal', padding='same')(c)
    return c
    
def decoder_block(input1, input2, feature, kernel1, kernel2, dropout):
    u = Conv2DTranspose(feature, (2, 2), strides=(2, 2), padding='same')(input1)
    u = concatenate([u, input2])
    c1 = Conv2D(feature, (kernel1, kernel1), activation='relu', kernel_initializer='he_normal', padding='same')(u)
    c2 = Conv2D(feature, (kernel2, kernel2), activation='relu', kernel_initializer='he_normal', padding='same')(u)
    c1 = Dropout(dropout)(c1)
    c2 = Dropout(dropout)(c2)
    c1 = Conv2D(feature, (kernel1, kernel1), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c2 = Conv2D(feature, (kernel2, kernel2), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c = concatenate([c1, c2], axis=3)
    # c = Conv2D(128,(1,1), activation='relu', kernel_initializer='he_normal', padding='same')(c)
    return c

################################################################
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel1, kernel2):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    # Contraction path
    c1 = encoder_block(s, 16, kernel1, kernel2, 0.1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = encoder_block(p1, 32, kernel1, kernel2, 0.1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = encoder_block(p2, 64, kernel1, kernel2, 0.2)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = encoder_block(p3, 128, kernel1, kernel2, 0.2)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = encoder_block(p4, 256, kernel1, kernel2, 0.3)

    # Expansive path
    c6 = decoder_block(c5, c4, 128, kernel1, kernel2, 0.2)

    c7 = decoder_block(c6, c3, 64, kernel1, kernel2, 0.2)

    c8 = decoder_block(c7, c2, 32, kernel1, kernel2, 0.1)

    c9 = decoder_block(c8, c1, 16, kernel1, kernel2, 0.1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # optimizer = adam_v2.Adam(learning_rate=lr, decay=lr/epochs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model
################################################################
def get_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel1, kernel2):
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel1, kernel2)

################################################################
def run_trials(data_directory, model_directory, trial_num, kernel1, kernel2, image_magnification, type_model):
    
    SIZE = 256 # size images
    batch_size = 24
    
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
    
    image_dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
    mask_dataset = []  # Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.
    
    images = os.listdir(image_directory)
    images.sort()
    # print(images)
    
    for i, image_name in enumerate(images):  # Remember enumerate method adds a counter and returns the enumerate object
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
    
    # import random
    
    # image_number = random.randint(0, len(X_train))
    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
    # plt.subplot(122)
    # plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
    # plt.show()
    
    IMG_HEIGHT = image_dataset.shape[1]
    IMG_WIDTH = image_dataset.shape[2]
    IMG_CHANNELS = image_dataset.shape[3]
    
    model = get_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel1, kernel2)
    
    csv_logger =CSVLogger(save_logs, append=True, separator=";")
    
    checkpointer = ModelCheckpoint(save_model, verbose=1, save_best_only=True, monitor='val_loss')
    
    callbacks = [csv_logger, checkpointer]
    
    start = datetime.now()
    
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        verbose=1,
                        epochs=50,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test),
                        shuffle=False)
    
    run_time = datetime.now() - start

    # model.save('C:/Users/Hafizi/PycharmProjects/Cell Segmentation/4-model/unet/_.hdf5')
    
    with open(save_history, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
    # history = pickle.load(open('/trainHistoryDict'), "rb")
    
    ###############################################################################################
    # evaluation
    
    models = os.listdir(model_directory)
    models.sort()
    for i, model_name in enumerate(models):  # Remember enumerate method adds a counter and returns the enumerate object
        if (model_name.split('.')[-1] == 'hdf5'):
            latest_model = model_name
    
    model = load_model(model_directory+latest_model, compile=True)
    
    # evaluate model
    loss_, acc_ = model.evaluate(X_test, y_test)
    print(latest_model)
    print("Loss = ", (loss_ * 100.0), "%")
    print("Accuracy = ", (acc_ * 100.0), "%")
    
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
    
    num_epoch = (latest_model.split('.')[2]).split('-')[0]
    
    f= open(result_file,"a+")
    f.write("Trial: {}\nBatch size: {}\nMax Epochs: {}\nAccuracy: {:.2f}%\nLoss: {:.2f}%\nIoU: {:.3f}\nTraining time: {}\n\n".format(trial_num, batch_size, num_epoch, acc_*100,loss_*100, iou_score, run_time))
    f.close()
    
    ###############################################################################################
    # Predict on a few images
    import random
    
    model = load_model(model_directory+latest_model, compile=False)
    
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
    
    return [trial_num, batch_size, num_epoch, acc_*100,loss_*100, iou_score, run_time]
   
################################################################    
kernel1 = 3
kernel2 = 5
image_magnification = 'all'

kernel_size = str(kernel1) + str(kernel2)
type_model = 'msunet_%s' % kernel_size

data_directory = 'D:/Pycharm Projects/Cell Segmentation/1-data/'
model_directory = 'D:/Pycharm Projects/Cell Segmentation/4-model/%s/%s/' % (type_model, image_magnification)
table_file = model_directory + '/table_%s.txt' % type_model  

table = []

col_names = ['Trial', 'Batch size', 'Max Epochs', 'Accuracy', 'Loss', 'IoU', 'Training Time']

init_num = 1
trials_left = 11-init_num
for t in range(trials_left):
    data = run_trials(data_directory, model_directory, init_num, kernel1, kernel2, image_magnification, type_model)
    table.append(data)
    init_num += 1

cal_table = np.array(table)
mean = np.mean(cal_table[:,3:6], axis = 0)
std = np.std(np.array(cal_table[:,3:6], dtype = np.float32), axis = 0)

mean = ['Mean', '-', '-', mean[0], mean[1], mean[2], '-']
std = ['STD', '-', '-', std[0], std[1], std[2], '-']

table.append(mean)
table.append(std)

f= open(table_file,"a+")
f.write(tabulate(table, headers = col_names))
f.close()
    