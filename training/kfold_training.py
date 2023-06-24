# lines that need changes: 
from keras.models import load_model
from keras.utils.np_utils import normalize
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
# from keras.optimizers import adam_v2
import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
from tabulate import tabulate
from msunet import segment_model
import segmentation_models as sm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import class_weight

# subjected to changes
####################################################################################################################
# GPU usage

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # change to CPU
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

####################################################################################################################
type_model = 'test'

# image_magnifications = ['x4']
# image_magnifications = ['x10']
image_magnifications = ['x20']
# image_magnifications = ['x4','x20']
# image_magnifications = ['x10','x20']
# image_magnifications = ['x4','x10','x20']
col_iou = []
mean_ = []

iou_score = sm.metrics.IOUScore(threshold=0.5)
f_score = sm.metrics.FScore(threshold=0.5)
hybrid_metrics = ['accuracy', iou_score, f_score]

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
losses = dice_loss + (1 * focal_loss)
# losses = focal_loss

optimizer ='adam'

for image_magnification in image_magnifications:

    # 256/ 512
    SIZE = 256
    batch_size = 24
    lr = 0.001
    epochs = 50
    
    data_directory = 'D:/Pycharm Projects/Cell Segmentation 2/data_%s/' % SIZE
    model_directory = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/%s/%s/' % (type_model, image_magnification)
    
    ###################################################################################################################
    
    image_format = 'tiff'
    
    image_directory = data_directory + '%s/images/' % image_magnification
    mask_directory = data_directory + '%s/masks/' % image_magnification
    
    ###############################################################################################
    # data preparation
    
    image_dataset = [] 
    mask_dataset = []  
    
    images = os.listdir(image_directory)
    images.sort()
    # print(images)
    
    num_images = 650 # # limit number of training images for merge datasets
    
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
    # image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
    # mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.20, random_state=0)
    
    X_train = np.expand_dims(normalize(np.array(X_train), axis=1), 3)
    X_test = np.expand_dims(normalize(np.array(X_test), axis=1), 3)
    y_train = np.expand_dims((np.array(y_train)), 3) / 255.
    y_test = np.expand_dims((np.array(y_test)), 3) / 255.
    
    import random
    # for x in range(10):
    #     image_number = random.randint(0, len(X_train)-1)
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(121)
    #     plt.imshow(np.reshape(X_train[image_number], (SIZE, SIZE)), cmap='gray')
    #     plt.subplot(122)
    #     plt.imshow(np.reshape(y_train[image_number], (SIZE, SIZE)), cmap='gray')
    #     plt.show()
    ###############################################################################################
    # training settings
    
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    
    table = []
    
    # Merge inputs and targets
    # inputs = np.concatenate((X_train, X_test), axis=0)
    # targets = np.concatenate((y_train, y_test), axis=0)
    inputs = X_train
    targets = y_train
    
    num_folds = 5
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        run = 1
        while run == 1:
            data = []
            x=inputs[train]
            y=inputs[test]
            
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            model_directory = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/%s/%s/%s/' % (type_model, image_magnification, fold_no)
            
            try:
                os.makedirs(model_directory, exist_ok = True)
                print('Model directory created successfully')
            except FileExistsError as error:
                print('Model directory already existed')
            
            save_logs = model_directory + 'S%s_BS%s_LR%s_E%s_M%s_logs.txt' % (SIZE, batch_size, lr, epochs, type_model)
            save_history = model_directory + 'S%s_BS%s_LR%s_E%s_M%s_trainHistoryDict' % (SIZE, batch_size, lr, epochs, type_model)
            graph_file = model_directory + 'S%s_BS%s_LR%s_E%s_M%s_graph.png' % (SIZE, batch_size, lr, epochs, type_model)
            dm_file = model_directory + 'S%s_BS%s_LR%s_E%s_M%s_cm.png' % (SIZE, batch_size, lr, epochs, type_model)
            tab = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(model_directory))))
            tab = os.path.join(tab, 'summary', image_magnification)
            try:
                os.makedirs(tab, exist_ok = True)
                print('Table directory created successfully')
            except FileExistsError as error:
                print('Table directory already existed')
            _model = model_directory + 'S%s_BS%s_LR%s_E%s_%s.h5' % (SIZE, batch_size, lr, epochs, type_model)
        
            model = segment_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            print(model.summary())
            
            from memory_usage import check_memory
            check_memory(batch_size, model, by_layer = True)
            
            csv_logger =CSVLogger(save_logs, append=True, separator=";")
            
            checkpointer = ModelCheckpoint(_model, verbose=1, save_best_only=True, mode='min', monitor='val_loss')
            # checkpointer = ModelCheckpoint(save_model, verbose=1, save_weights_only=True, monitor='val_loss', mode ='min', save_best_only=True)
        
            from tensorflow.keras.callbacks import TensorBoard
            tensorboard = TensorBoard(log_dir="{}\\logs\\{}".format(model_directory, type_model))
            # python -m tensorboard.main --logdir=logs/
            callbacks = [csv_logger, checkpointer, tensorboard]
            
            start = datetime.now()
            
            # model.compile(optimizer=optimizer, loss=hybrid_loss, metrics=[hybrid_metrics])
            model.compile(optimizer=optimizer, loss=losses, metrics=[hybrid_metrics])
            
            history = model.fit(inputs[train], targets[train],
                                batch_size=batch_size,
                                verbose=1,
                                epochs=epochs,
                                callbacks=callbacks,
                                validation_data=(inputs[test], targets[test]),
                                shuffle=False)
            
            run_time = datetime.now() - start
            run_time = run_time.total_seconds() / 60
            print("Runtime: {:.2f} minutes".format(run_time))
            
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
            loss_, accuracy_, iou_score_, fscore_ = model.evaluate(X_test,y_test)
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
            cal_iou = np.sum(intersection) / np.sum(union)
            print("IoU score is: ", cal_iou)
            
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
            
            plt.figure(figsize=(60, 30))
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
            
            if iou_score_ > 0.1:
                data = [fold_no, loss_, accuracy_, iou_score_, fscore_, run_time]
                table.append(data)
                fold_no = fold_no + 1
                run = 0
                col_iou.append(iou_score_)
            else:
                run = 1
            
    
    ###############################################################################################
    col_names = ['Fold', 'Loss', 'Accuracy', 'IoU', 'F_score','Training Time']
    cal_table = np.array(table)
    
    mean = np.mean(cal_table[:,1:5], axis = 0)
    std = np.std(np.array(cal_table[:,1:5], dtype = np.float32), axis = 0)
    mean = ['Mean', mean[0], mean[1], mean[2], mean[3], '-']
    std = ['STD', std[0], std[1], std[2], std[3], '-']
    
    table.append(mean)
    table.append(std)
    mean_.append(mean[3])
    
    table_file = tab + '/%s_table.txt' % type_model
    f= open(table_file,"a+")
    f.write(tabulate(table, headers = col_names))
    f.close() 
print(col_iou)
print(mean_)