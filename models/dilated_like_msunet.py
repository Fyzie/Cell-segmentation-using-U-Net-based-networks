from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Activation, SeparableConv2D, Add
from keras.models import Model
from keras import regularizers

################################################################ convolutional blocks
def encoder_block(input1, feature, kernel, dropout):
    
    # Scale 1
    x = Conv2D(feature, (kernel, kernel), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
    c1 = Dropout(dropout)(x)
    c1 = Conv2D(feature, (kernel, kernel), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        
    # Scale 2
    y = Conv2D(feature, (kernel, kernel), activation='relu', kernel_initializer='he_normal', padding='same', dilation_rate = 2)(input1)
    c2 = Dropout(dropout)(y)
    c2 = Conv2D(feature, (kernel, kernel), activation='relu', kernel_initializer='he_normal', padding='same', dilation_rate = 2)(c2)
    
    # Dilated addition
    # m1 = Add()([y, x])
    # m2 = Add()([c2, c1])
    # m3 = Add()([m1, m2])
    
    # concatenation
    c = concatenate([c1, c2], axis=3)
    # c = concatenate([c1, c2, m3], axis=3)
    c = Conv2D(feature, (1,1), activation='relu', kernel_initializer='he_normal', padding='same')(c)
    return c
    
def decoder_block(input1, input2, feature, kernel, dropout):
    u = Conv2DTranspose(feature, (2, 2), strides=(2, 2), padding='same')(input1)
    u = concatenate([u, input2])
    c = encoder_block(u, feature, kernel, dropout)
    return c

################################################################ whole model
def segment_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel=3, dropout = 0.3):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    # Contraction path
    c1 = encoder_block(s, 16, kernel, dropout)
    p1 = MaxPooling2D((2, 2))(s)

    c2 = encoder_block(p1, 32, kernel, dropout)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = encoder_block(p2, 64, kernel, dropout)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = encoder_block(p3, 128, kernel, dropout)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = encoder_block(p4, 256, kernel, dropout)

    # Expansive path
    c6 = decoder_block(c5, c4, 128, kernel, dropout)

    c7 = decoder_block(c6, c3, 64, kernel, dropout)

    c8 = decoder_block(c7, c2, 32, kernel, dropout)

    c9 = decoder_block(c8, c1, 16, kernel, dropout)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    # model.summary()

    return model

