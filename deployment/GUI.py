from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import os

from tflite_runtime.interpreter import Interpreter
from keras.models import Model,load_model
from keras.utils.np_utils import normalize
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
import cv2
import numpy as np
from patchify import patchify, unpatchify
import threading

import adafruit_dht
import adafruit_vcnl4040
import board
import time  

class Ui_Dialog(object):
    def fileBrowser(self):
        folder_directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Open File', directory='.')
        if folder_directory:
            self.previous_directory = folder_directory
        else:
            folder_directory = self.previous_directory
        self.lineEdit.setText(folder_directory)
        files = os.listdir(folder_directory)
        self.listWidget.clear()
        image_format = ['.jpg', '.png', '.tif', '.tiff']
        for file in files:
            for i in image_format:
                if file.endswith(i):
                    self.listWidget.addItem(file)
                
    def showImage(self):
        self.image = QImage(self.image_path)
        pic = QtWidgets.QGraphicsPixmapItem()
        pic.setPixmap(QPixmap.fromImage(self.image))
        pic = self.scene.addItem(pic)
        resize = self.scene.sceneRect()
        self.graphicsView.fitInView(resize, QtCore.Qt.KeepAspectRatio)
        
    def itemClicked(self, item):
        # print("Selected items: ", item.text())
        directory = self.lineEdit.text()
        self.image_path = directory + '/' + item.text()
        # print(image_path)
        self.showImage()
        
    def segmentation_map_to_image(self,
        result: np.ndarray, colormap: np.ndarray, remove_holes=False
    ) -> np.ndarray:
        """
        Convert network result of floating point numbers to an RGB image with
        integer values from 0-255 by applying a colormap.

        :param result: A single network result after converting to pixel values in H,W or 1,H,W shape.
        :param colormap: A numpy array of shape (num_classes, 3) with an RGB value per class.
        :param remove_holes: If True, remove holes in the segmentation result.
        :return: An RGB image where each pixel is an int8 value according to colormap.
        """
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
    
    def encoder_block(self,input1, feature, kernel1, kernel2, dropout):
        c1 = Conv2D(feature, (kernel1, kernel1), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
        c2 = Conv2D(feature, (kernel2, kernel2), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
        c1 = Dropout(dropout)(c1)
        c2 = Dropout(dropout)(c2)
        c1 = Conv2D(feature, (kernel1, kernel1), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        c2 = Conv2D(feature, (kernel2, kernel2), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        c = concatenate([c1, c2], axis=3)
        c = Conv2D(feature, (1,1), activation='relu', kernel_initializer='he_normal', padding='same')(c)
        return c
    
    def decoder_block(self,input1, input2, feature, kernel1, kernel2, dropout):
        u = Conv2DTranspose(feature, (2, 2), strides=(2, 2), padding='same')(input1)
        u = concatenate([u, input2])
        c = self.encoder_block(u, feature, kernel1, kernel2, dropout)
        return c

################################################################
    def multiscale_unet_model(self,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel1, kernel2):
        # Build the model
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
        s = inputs

        # Contraction path
        c1 = self.encoder_block(s, 16, kernel1, kernel2, 0.1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = self.encoder_block(p1, 32, kernel1, kernel2, 0.1)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = self.encoder_block(p2, 64, kernel1, kernel2, 0.2)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = self.encoder_block(p3, 128, kernel1, kernel2, 0.2)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = self.encoder_block(p4, 256, kernel1, kernel2, 0.3)

        # Expansive path
        c6 = self.decoder_block(c5, c4, 128, kernel1, kernel2, 0.2)

        c7 = self.decoder_block(c6, c3, 64, kernel1, kernel2, 0.2)

        c8 = self.decoder_block(c7, c2, 32, kernel1, kernel2, 0.1)

        c9 = self.decoder_block(c8, c1, 16, kernel1, kernel2, 0.1)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        # optimizer = adam_v2.Adam(learning_rate=lr, decay=lr/epochs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        return model
    
    def segmentCell(self):
        model = self.multiscale_unet_model(256, 256, 1, 3, 5)
        model.load_weights('model_35.hdf5')
        large_image = cv2.imread(self.image_path)
        file_name = os.path.basename(self.image_path)
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
        
        num_white = np.sum(reconstructed_image == 1)
        num_black = np.sum(reconstructed_image == 0)
        confluency = (num_white/(num_white+num_black))*100
        confluency = '{:.2f} %'.format(confluency)
        # print('Confluency: {}'.format(confluency))
        
        height, width = reconstructed_image.shape
        
        # Define colormap, each color represents a class
        colormap = np.array([[68, 1, 84], [255, 216, 52]])

        # Define the transparency of the segmentation mask on the photo
        alpha = 0.3

        # Use function from notebook_utils.py to transform mask to an RGB image
        mask = self.segmentation_map_to_image(reconstructed_image, colormap)

        resized_mask = cv2.resize(mask, (width, height))

        rgb_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)

        # Create image with mask put on
        image_with_mask = cv2.addWeighted(resized_mask, alpha, rgb_image, 1 - alpha, 0)
        
        self.image_path = 'segmented_image.jpg'
        cv2.imwrite(self.image_path, image_with_mask)
        
        self.showImage()
        self.label_13.setText(confluency)
        
    def adafruitSensors(self):
        try:
            dht_pin = board.D4
            i2c = board.I2C()

            dht = adafruit_dht.DHT22(dht_pin, use_pulseio=False)
            vcnl = adafruit_vcnl4040.VCNL4040(i2c)
            temperature = dht.temperature
            humidity = dht.humidity
            proximity = vcnl.proximity
            light = vcnl.light
            white = vcnl.white
            if humidity is not None and temperature is not None and light is not None and proximity is not None and white is not None:
                self.label_8.setText('{:.2f}*C'.format(temperature))
                self.label_9.setText('{:.2f}%'.format(humidity))
                self.label_10.setText('{:.2f}'.format(white))
                self.label_11.setText('{:.2f}'.format(light))
                self.label_12.setText('{:.2f}'.format(proximity))
        except RuntimeError as e:
            pass
        
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(798, 384)
        Dialog.setMouseTracking(False)
        
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(20, 40, 400, 300))
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.cell_image = QtWidgets.QLabel(Dialog)
        self.cell_image.setGeometry(QtCore.QRect(170, 10, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.cell_image.setFont(font)
        self.cell_image.setTextFormat(QtCore.Qt.AutoText)
        self.cell_image.setAlignment(QtCore.Qt.AlignCenter)
        self.cell_image.setWordWrap(False)
        self.cell_image.setObjectName("cell_image")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(490, 200, 81, 16))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(660, 200, 81, 16))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(580, 320, 81, 16))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(490, 260, 81, 16))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(660, 260, 81, 16))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(480, 220, 111, 21))
        self.label_8.setAutoFillBackground(False)
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_8.setLineWidth(2)
        self.label_8.setText("")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setGeometry(QtCore.QRect(640, 220, 111, 21))
        self.label_9.setAutoFillBackground(False)
        self.label_9.setFrameShape(QtWidgets.QFrame.Box)
        self.label_9.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_9.setLineWidth(2)
        self.label_9.setText("")
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setGeometry(QtCore.QRect(640, 280, 111, 21))
        self.label_10.setAutoFillBackground(False)
        self.label_10.setFrameShape(QtWidgets.QFrame.Box)
        self.label_10.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_10.setLineWidth(2)
        self.label_10.setText("")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(Dialog)
        self.label_11.setGeometry(QtCore.QRect(480, 280, 111, 21))
        self.label_11.setAutoFillBackground(False)
        self.label_11.setFrameShape(QtWidgets.QFrame.Box)
        self.label_11.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_11.setLineWidth(2)
        self.label_11.setText("")
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(Dialog)
        self.label_12.setGeometry(QtCore.QRect(570, 340, 111, 21))
        self.label_12.setAutoFillBackground(False)
        self.label_12.setFrameShape(QtWidgets.QFrame.Box)
        self.label_12.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_12.setLineWidth(2)
        self.label_12.setText("")
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(450, 180, 331, 191))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(710, 10, 71, 23))
        self.pushButton.setDefault(False)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(550, 10, 151, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(450, 40, 331, 131))
        self.listWidget.setObjectName("listWidget")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(450, 10, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_13.setGeometry(QtCore.QRect(90, 350, 221, 21))
        self.label_13.setAutoFillBackground(False)
        self.label_13.setFrameShape(QtWidgets.QFrame.Box)
        self.label_13.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_13.setLineWidth(2)
        self.label_13.setText("")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(Dialog)
        self.label_14.setGeometry(QtCore.QRect(10, 350, 81, 16))
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(330, 350, 91, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.groupBox.raise_()
        self.graphicsView.raise_()
        self.cell_image.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.label_9.raise_()
        self.label_10.raise_()
        self.label_11.raise_()
        self.label_12.raise_()
        self.pushButton.raise_()
        self.lineEdit.raise_()
        self.listWidget.raise_()
        self.label_2.raise_()
        self.label_13.raise_()
        self.label_14.raise_()
        self.pushButton_2.raise_()
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.adafruitSensors)
        self.timer.start()

        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(lambda: self.fileBrowser())
        self.pushButton_2.clicked.connect(lambda: self.segmentCell())
        self.listWidget.itemClicked.connect(self.itemClicked)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Cell Segmentation"))
        self.cell_image.setText(_translate("Dialog", "Cell Image"))
        self.label_3.setText(_translate("Dialog", "Temperature"))
        self.label_4.setText(_translate("Dialog", "Humidity"))
        self.label_5.setText(_translate("Dialog", "Proximity"))
        self.label_6.setText(_translate("Dialog", "Light"))
        self.label_7.setText(_translate("Dialog", "White"))
        self.groupBox.setTitle(_translate("Dialog", "Sensors"))
        self.pushButton.setText(_translate("Dialog", "Browse"))
        self.label_2.setText(_translate("Dialog", "Image Folder"))
        self.label_14.setText(_translate("Dialog", "Confluency"))
        self.pushButton_2.setText(_translate("Dialog", "Segment"))
        
class Dialog(QtWidgets.QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)     

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Dialog()
    w.show()
    sys.exit(app.exec_())
