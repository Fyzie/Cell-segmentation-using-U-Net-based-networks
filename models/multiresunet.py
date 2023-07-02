from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Activation, Add
from keras.models import Model

################################################################ convolutional blocks
def encoder_block(input1, feature):
    feature = int(feature*1.67)
    feature1 = feature//6
    feature2 = feature//3
    feature3 = feature//2
    
    c1 = Conv2D(feature1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
    c2 = Conv2D(feature2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c3 = Conv2D(feature3, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c = concatenate([c1, c2, c3], axis=3)
    
    feature4 = feature1 + feature2 + feature3
    r = Conv2D(feature4, (1,1), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
    c = Add()([c, r])
    
    return c
    
def decoder_block(input1, input2, feature):
    u = Conv2DTranspose(feature, (2, 2), strides=(2, 2), padding='same')(input1)
    u = concatenate([u, input2])
    c = encoder_block(u, feature)
    return c

def res_path(input1, feature, layer):
    for x in range(layer):
        c = Conv2D(feature, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
        r = Conv2D(feature, (1,1), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
        input1 = Add()([c, r])

    return input1

################################################################ whole model
def segment_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel1=3):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    # Contraction path
    c1 = encoder_block(s, 32)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = encoder_block(p1, 16)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = encoder_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = encoder_block(p3, 256)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = encoder_block(p4, 512)

    # Expansive path
    c4 = res_path(c4, 256, 1)
    c6 = decoder_block(c5, c4, 256)
    
    c3 = res_path(c3, 128, 2)
    c7 = decoder_block(c6, c3, 128)
    
    c2 = res_path(c2, 64, 3)
    c8 = decoder_block(c7, c2, 64)

    c1 = res_path(c1, 32, 4)    
    c9 = decoder_block(c8, c1, 32)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    # model.summary()

    return model