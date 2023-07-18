from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Activation, SeparableConv2D, DepthwiseConv2D
from keras.models import Model

################################################################ convolutional blocks
def separable_block(input1, feature, kernel1, kernel2, dropout):
    c1 = SeparableConv2D(feature, (kernel1, kernel1), activation='relu', pointwise_initializer='he_normal', padding='same')(input1)
    c2 = SeparableConv2D(feature, (kernel2, kernel2), activation='relu', pointwise_initializer='he_normal', padding='same')(input1)
    c1 = Dropout(dropout)(c1)
    c2 = Dropout(dropout)(c2)
    c1 = SeparableConv2D(feature, (kernel1, kernel1), activation='relu', pointwise_initializer='he_normal', padding='same')(c1)
    c2 = SeparableConv2D(feature, (kernel2, kernel2), activation='relu', pointwise_initializer='he_normal', padding='same')(c2)
    c = concatenate([c1, c2], axis=3)
    c = Conv2D(feature, (1,1), activation='relu', kernel_initializer='he_normal', padding='same')(c)
    return c

def encoder_block(input1, feature, kernel1, kernel2, dropout):
    c1 = Conv2D(feature, (kernel1, kernel1), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
    c2 = Conv2D(feature, (kernel2, kernel2), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
    c1 = Dropout(dropout)(c1)
    c2 = Dropout(dropout)(c2)
    c1 = Conv2D(feature, (kernel1, kernel1), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c2 = Conv2D(feature, (kernel2, kernel2), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c = concatenate([c1, c2], axis=3)
    c = Conv2D(feature, (1,1), activation='relu', kernel_initializer='he_normal', padding='same')(c)
    return c
    
def decoder_block(input1, input2, feature, kernel1, kernel2, dropout):
    u = Conv2DTranspose(feature, (2, 2), strides=(2, 2), padding='same')(input1)
    u = concatenate([u, input2])
    c = separable_block(u, feature, kernel1, kernel2, dropout)
    return c

################################################################ whole model
def segment_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel1=3, kernel2=5):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    # Contraction path
    c1 = encoder_block(s, 16, kernel1, kernel2, 0.3)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = separable_block(p1, 32, kernel1, kernel2, 0.3)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = separable_block(p2, 64, kernel1, kernel2, 0.3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = separable_block(p3, 128, kernel1, kernel2, 0.3)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = separable_block(p4, 256, kernel1, kernel2, 0.3)

    # Expansive path
    c6 = decoder_block(c5, c4, 128, kernel1, kernel2, 0.3)

    c7 = decoder_block(c6, c3, 64, kernel1, kernel2, 0.3)

    c8 = decoder_block(c7, c2, 32, kernel1, kernel2, 0.3)

    c9 = decoder_block(c8, c1, 16, kernel1, kernel2, 0.3)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    # model.summary()

    return model