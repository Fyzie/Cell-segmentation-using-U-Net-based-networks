# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:08:10 2023

@author: Hafizi
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import os

path_dir = 'D:/Pycharm Projects/Cell Segmentation 2/kfold/unet_3/x4'
path = os.listdir(path_dir)
# Load the text files into a list of DataFrames
data_frames = []
for i in range(5):  
    data_path = os.path.join(path_dir, str(i+1))
    files = os.listdir(data_path)
    files.sort()
    for file in files:
        if file.endswith('logs.txt'):
            file_name = os.path.join(data_path, file)
            df = pd.read_csv(file_name, delimiter=';', skiprows=(1), header=None, names=['epoch', 'accuracy', 'f1-score', 'iou_score','loss', 'val_accuracy', 'val_f1-score', 'val_iou_score', 'val_loss'])
            data_frames.append(df)

train_datas = []
val_datas = []
for df in data_frames:
    train_data = df['iou_score']
    val_data = df['val_iou_score']
    train_datas.append(train_data)
    val_datas.append(val_data)

# Compute the mean and standard deviation of the training and validation accuracy across trials
mean_train_acc = pd.concat(train_datas, axis=1).mean(axis=1)
std_train_acc = pd.concat(train_datas, axis=1).std(axis=1)
mean_val_acc = pd.concat(val_datas, axis=1).mean(axis=1)
std_val_acc = pd.concat(val_datas, axis=1).std(axis=1)

# Calculate the difference between the mean training and validation accuracy for each epoch
diff = mean_train_acc - mean_val_acc

# Create a new figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Plot the training accuracy and validation accuracy for each trial and the mean accuracy with shaded regions for deviation
ax1.plot(mean_train_acc, linestyle = '-.', color='k', label='Mean Train IoU', linewidth = 1)
ax1.plot(mean_val_acc, linestyle = '-.', color='r', label='Mean Val IoU', linewidth = 1)
ax1.fill_between(mean_train_acc.index, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha=0.5, label='Train IoU Deviation')
ax1.fill_between(mean_val_acc.index, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, alpha=0.5, label='Val IoU Deviation')
ax1.set_ylabel('IoU Score')
ax1.legend(loc = 'lower right')
ax1.set_ylim(0,1)

# Plot the histogram of the difference between the mean training and validation accuracy with respect to the epochs
# ax2.plot(diff, linestyle = '-.', color='k', label='Mean Train IoU', linewidth = 1)
ax2.bar(mean_train_acc.index, diff, color='b')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Difference')
# ax2.set_xlim(-0.1, 0.1)
ax2.set_ylim(-0.15,0.15)

# Set the x-axis limits for both subplots
ax1.set_xlim(mean_train_acc.index[0], mean_train_acc.index[-1])

plt.show()