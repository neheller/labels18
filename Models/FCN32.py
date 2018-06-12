from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from .metrics import *


# No VGG is implemented yet
def create_fcn32(IMG_DIM, IMG_CHANNELS, filter_size=(3, 3),
                initial_filters=6, num_classes=2):

    input = Input((IMG_DIM, IMG_DIM, IMG_CHANNELS))
    x = input
    pool_size = (2, 2)
    stride_size = (2, 2)
    num_filters = initial_filters

    # Block 1
    for i in range(2):
        x = Conv2D(num_filters, filter_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size, strides=stride_size)(x)
    f1 = x

    # Block 2
    num_filters *= 2
    for i in range(2):
        x = Conv2D(num_filters, filter_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size, strides=stride_size)(x)
    f2 = x

    # Block 3
    num_filters *= 2
    for i in range(3):
        x = Conv2D(num_filters, filter_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size, strides=stride_size)(x)
    f3 = x

    # Block 4
    num_filters *= 2
    for i in range(3):
        x = Conv2D(num_filters, filter_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size, strides=stride_size)(x)
    f4 = x

    # Block 5
    num_filters *= 2
    for i in range(3):
        x = Conv2D(num_filters, filter_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size, strides=stride_size)(x)
    f5 = x

    o = f5
    o = (Conv2D(4096, (7, 7), activation='relu', padding='same'))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu', padding='same'))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(num_classes, (1, 1), kernel_initializer='he_normal'))(o)

    o = Conv2DTranspose(num_classes, kernel_size=(64, 64), strides=(32, 32),
                    padding='same', activation='softmax', use_bias=False)(o)
    flat_preds = Reshape([-1, 2])(o)

    return Model(input, flat_preds)
