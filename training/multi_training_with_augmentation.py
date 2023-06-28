# lines that need changes: 14, 22, 23, 278
from keras.models import load_model
from keras.utils.np_utils import normalize
from keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
from tabulate import tabulate
from unet_3 import segment_model
import segmentation_models as sm
import albumentations as A
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # change to CPU
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.833)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def multi_trials(trial):
    kernel_size = 3
    image_magnification = 'x20'

    type_model = 'augment_unet_%s' % kernel_size
    # type_model = 'msunet_%s' % kernel_size

    SIZE = 256 
    batch_size = 24
    lr = 0.001
    epochs = 50

    iou_score = sm.metrics.IOUScore(threshold=0.5)
    f_score = sm.metrics.FScore(threshold=0.5)
    hybrid_metrics = ['accuracy', iou_score, f_score]

    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    losses = dice_loss + (1 * focal_loss)
    # losses = focal_loss

    optimizer ='adam'
    # optimizer = adam_v2.Adam(learning_rate=lr, decay=lr/epochs)

    data_directory = 'D:/Pycharm Projects/Cell Segmentation 2/data_%s/' % SIZE
    model_directory = 'D:/Pycharm Projects/Cell Segmentation 2/multi_model/%s/%s/%s/' % (type_model, image_magnification, trial)

    ####################################################################################################################
        
    try:
        os.makedirs(model_directory, exist_ok = True)
        print('Model directory created successfully')
    except FileExistsError as error:
        print('Model directory already existed')

    image_directory = data_directory + '%s/filtered/images/' % image_magnification
    mask_directory = data_directory + '%s/filtered/masks/' % image_magnification

    save_logs = model_directory + '{S%s_BS%s_LR%s_E%s_M%s}logs.txt' % (SIZE, batch_size, lr, epochs, type_model)
    save_history = model_directory + '{S%s_BS%s_LR%s_E%s_M%s}trainHistoryDict' % (SIZE, batch_size, lr, epochs, type_model)
    graph_file = model_directory + '{S%s_BS%s_LR%s_E%s_M%s}graph.png' % (SIZE, batch_size, lr, epochs, type_model)
    dm_file = model_directory + '{S%s_BS%s_LR%s_E%s_M%s}cm.png' % (SIZE, batch_size, lr, epochs, type_model)
    table_file = model_directory + '{S%s_BS%s_LR%s_E%s_M%s}table.txt' % (SIZE, batch_size, lr, epochs, type_model)
    _model = model_directory + 'S%s_BS%s_LR%s_E%s_%s.h5' % (SIZE, batch_size, lr, epochs, type_model)

    image_dataset = [] 
    mask_dataset = []  

    images = os.listdir(image_directory)
    images.sort()
    # print(images)
    
    masks = os.listdir(mask_directory)
    masks.sort()
    # print(masks)

    # num_images = 2000 # # limit number of training images

    for i, image_name in enumerate(images):  
        # if i == num_images: 
        #     i = i - 1
        #     break
        image = cv2.imread(image_directory + images[i], 0)
        mask = cv2.imread(mask_directory + masks[i], 0)
        
        original_height, original_width = image.shape[:2]
        
        aug = A.Compose([
            A.OneOf([
                # A.RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.8),
                A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
            ],p=0.7),
            A.Transpose(p=1),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.8),
            A.OneOf([
                A.ElasticTransform(p=0.8, alpha=50, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.8),
            ], p=1)])
        
        num = random.random()
        # print(num)
        random.seed(num)
        augmented = aug(image=image, mask=mask)

        augment_image = augmented['image']
        augment_mask = augmented['mask']
        
        augment_mask = np.where(augment_mask>0, 255, augment_mask)
        # print(np.unique(augment_mask))
        
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))
        
        # print(mask)
        mask = Image.fromarray(mask)
        mask = mask.resize((SIZE, SIZE))
        mask_dataset.append(np.array(mask))
        
        augment_image = Image.fromarray(augment_image)
        augment_image = augment_image.resize((SIZE, SIZE))
        image_dataset.append(np.array(augment_image))
        
        # print(augment_mask)
        augment_mask = Image.fromarray(augment_mask)
        augment_mask = augment_mask.resize((SIZE, SIZE))
        mask_dataset.append(np.array(augment_mask))
        
    print("number of images:\n", len(image_dataset))
    print("number of images:\n", len(mask_dataset))

    # make sure number of images and masks are same

    # Normalize images
    image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
    mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.30, random_state=0)

    # for x in range(10):
    #     image_number = random.randint(0, len(X_train)-1)
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(121)
    #     plt.imshow(np.reshape(X_train[image_number], (SIZE, SIZE)), cmap='gray')
    #     plt.subplot(122)
    #     plt.imshow(np.reshape(y_train[image_number], (SIZE, SIZE)), cmap='gray')
    #     plt.show()

    IMG_HEIGHT = image_dataset.shape[1]
    IMG_WIDTH = image_dataset.shape[2]
    IMG_CHANNELS = image_dataset.shape[3]

    model = segment_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    csv_logger =CSVLogger(save_logs, append=True, separator=";")

    checkpointer = ModelCheckpoint(_model, verbose=1, save_best_only=True, mode='max', monitor='val_iou_score')
    # checkpointer = ModelCheckpoint(save_model, verbose=1, save_weights_only=True, monitor='val_loss', mode ='min', save_best_only=True)

    callbacks = [csv_logger, checkpointer]

    start = datetime.now()

    # model.compile(optimizer=optimizer, loss=hybrid_loss, metrics=[hybrid_metrics])
    model.compile(optimizer=optimizer, loss=losses, metrics=[hybrid_metrics])

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
                        custom_objects={"dice_loss_plus_1binary_focal_loss":losses,
                                        "iou_score":iou_score,
                                        "f1-score":f_score},
                        compile=True)
    # model = load_model(_model,
    #                     custom_objects={"binary_focal_loss":losses,
    #                                     "iou_score":iou_score,
    #                                     "f1-score":f_score},
    #                     compile=True)
    # model.load_weights(model_directory+latest_model)

    # evaluate model
    loss_, accuracy_, iou_score_, fscore_ = model.evaluate(X_test, y_test)
    print(_model)
    print("Loss = ", (loss_ * 100.0), "%")
    print("Accuracy = ", (accuracy_ * 100.0), "%")
    print("IoU = ", iou_score_)
    print("F1 = ", fscore_)

    ###############################################################################################
    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))

    plt.subplot(131)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.ylim(0, 1)

    plt.subplot(132)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.ylim(0, 1)

    # Plot training & validation loss values
    plt.subplot(133)
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
                        custom_objects={"dice_loss_plus_1binary_focal_loss":losses,
                        "iou_score":iou_score, "f1-score":f_score}, compile=True)
    # model = load_model(_model,
    #                    custom_objects={"binary_focal_loss":losses,
    #                    "iou_score":iou_score, "f1-score":f_score}, compile=True)


    test_img_number = random.randint(0, len(X_test)-1)
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]
    test_img_norm = test_img[:, :, 0][:, :, None]
    test_img_input = np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.2).astype(np.uint8)

    # kernel = np.ones((3, 3),np.uint8)
    # prediction = cv2.morphologyEx(prediction,cv2.MORPH_OPEN,kernel, iterations = 2)

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
    
    return [trial, loss_, accuracy_, iou_score_, fscore_, run_time]

table_file = 'table_augment_unet_3.txt'
table = []

col_names = ['Trial', 'Loss', 'Accuracy', 'IoU', 'F_score','Training Time']
trial = 1
while trial<=5:
    data = multi_trials(trial)
    if data[3]>0.1:
        table.append(data)
        trial = trial + 1
    else:
        trial = trial
    
    
cal_table = np.array(table)
mean = np.mean(cal_table[:,1:5], axis = 0)
std = np.std(np.array(cal_table[:,1:5], dtype = np.float32), axis = 0)

mean = ['Mean', mean[0], mean[1], mean[2], mean[3], '-']
std = ['STD', std[0], std[1], std[2], std[3], '-']

table.append(mean)
table.append(std)
    
f= open(table_file,"a+")
f.write(tabulate(table, headers = col_names))
f.close() 