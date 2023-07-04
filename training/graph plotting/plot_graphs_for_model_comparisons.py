import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import os

# List of paths where models are located
path_dirs = ['D:/Pycharm Projects/Cell Segmentation 2/kfold/unet_3/x20',
             'D:/Pycharm Projects/Cell Segmentation 2/kfold/msunet_2.0/x20',
             'D:/Pycharm Projects/Cell Segmentation 2/kfold/dilated_1/x20',
             'D:/Pycharm Projects/Cell Segmentation 2/kfold/residual_1/x20'
             ]

data_frames = []
for path_dir in path_dirs:
    path = os.listdir(path_dir)
    # Load the text files into a list of DataFrames
    for i in range(5):
        data_path = os.path.join(path_dir, str(i+1))
        files = os.listdir(data_path)
        files.sort()
        for file in files:
            if file.endswith('logs.txt'):
                file_name = os.path.join(data_path, file)
                df = pd.read_csv(file_name, delimiter=';', skiprows=(1), header=None,
                                 names=['epoch', 'accuracy', 'f1-score', 'iou_score', 'loss', 'val_accuracy',
                                        'val_f1-score', 'val_iou_score', 'val_loss'])
                data_frames.append(df)

train_datas = []
val_datas = []
for df in data_frames:
    train_data = df['val_iou_score']
    val_data = df['val_loss']
    train_datas.append(train_data)
    val_datas.append(val_data)

# Compute the mean and standard deviation of the training and validation accuracy across trials for each path
mean_train_accs = [pd.concat(train_datas[i:i+5], axis=1).mean(axis=1) for i in range(0, len(train_datas), 5)]
std_train_accs = [pd.concat(train_datas[i:i+5], axis=1).std(axis=1) for i in range(0, len(train_datas), 5)]
mean_val_accs = [pd.concat(val_datas[i:i+5], axis=1).mean(axis=1) for i in range(0, len(val_datas), 5)]
std_val_accs = [pd.concat(val_datas[i:i+5], axis=1).std(axis=1) for i in range(0, len(val_datas), 5)]

# Plot the training accuracy and validation accuracy for each path along with the mean accuracy and deviation
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

name1 = ["U-Net", "Inception UNet", "Dilated UNet", "ResDilated UNet"]
name = ["Model A", "Model B", "Model C", "Model D"]
for i, (mean_train_acc, mean_val_acc, std_train_acc, std_val_acc) in enumerate(zip(mean_train_accs, mean_val_accs, std_train_accs, std_val_accs)):
    ax1.plot(mean_train_acc, linestyle='-', label=name[i], linewidth=1)
    ax2.plot(mean_val_acc, linestyle='-.', label=name[i], linewidth=1)
    # ax.fill_between(mean_train_acc.index, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc,
    #                 alpha=0.5, label=f'Train IoU Deviation ({i+1})')
    # ax.fill_between(mean_val_acc.index, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc,
    #                 alpha=0.5, label=f'Val IoU Deviation ({i+1})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation IoU Score')
    ax2.set_ylabel('Validation Loss')
    ax1.legend(loc='center right')
    # ax.legend(loc='upper right')
    # ax1.set_ylim(0,1)
    # ax2.set_ylim(0,1)
    
    
plt.show()