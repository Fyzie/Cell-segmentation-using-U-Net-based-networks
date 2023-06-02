from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Activation
from keras.models import Model
from keras import backend as K
import segmentation_models as sm
import tensorflow as tf

# def encoder_block(input1, feature, kernel, dropout):
#     c = Conv2D(feature, (kernel, kernel), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
#     c = Dropout(dropout)(c)
#     c = Conv2D(feature, (kernel, kernel), activation='relu', kernel_initializer='he_normal', padding='same')(c)
#     return c
    
# def decoder_block(input1, input2, feature, kernel, dropout):
#     u = Conv2DTranspose(feature, (2, 2), strides=(2, 2), padding='same')(input1)
#     u = concatenate([u, input2])
#     c = encoder_block(u, feature, kernel, dropout)
#     return c

################################################################
def segment_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs
    dropout = 0.3

    # # Contraction path
    # c1 = encoder_block(s, 16, kernel_size, 0.1)
    # p1 = MaxPooling2D((2, 2))(c1)

    # c2 = encoder_block(p1, 32, kernel_size, 0.1)
    # p2 = MaxPooling2D((2, 2))(c2)

    # c3 = encoder_block(p2, 64, kernel_size, 0.2)
    # p3 = MaxPooling2D((2, 2))(c3)

    # c4 = encoder_block(p3, 128, kernel_size, 0.2)
    # p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # c5 = encoder_block(p4, 256, kernel_size, 0.3)

    # # Expansive path
    # c6 = decoder_block(c5, c4, 128, kernel_size, 0.2)

    # c7 = decoder_block(c6, c3, 64, kernel_size, 0.2)

    # c8 = decoder_block(c7, c2, 32, kernel_size, 0.1)

    # c9 = decoder_block(c8, c1, 16, kernel_size, 0.1)

    # outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(dropout)(c1)
    c1 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding = 'same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(dropout)(c2)
    c2 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(dropout)(c3)
    c3 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
    c4 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(dropout)(c4)
    c4 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)
    
    c5 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(dropout)(c5)
    c5 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    c6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    c6 = concatenate([c6, c4])
    c6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = Dropout(dropout)(c6)
    c6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    c7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    c7 = concatenate([c7, c3])
    c7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = Dropout(dropout)(c7)
    c7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    c8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    c8 = concatenate([c8, c2])
    c8 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = Dropout(dropout)(c8)
    c8 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    c9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    c9 = concatenate([c9, c1])
    c9 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = Dropout(dropout)(c9)
    c9 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])

    # model.summary()

    return model