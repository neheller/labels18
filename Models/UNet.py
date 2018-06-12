import numpy as np

import keras
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import BatchNormalization as BNorm
import tensorflow as tf

from initializers.convtransposeinterp import ConvTransposeInterp

from .metrics import *




def create_unet(IMG_DIM, IMG_CHANNELS, filter_size=3,
                transpose_filter_size=2, depth=5, initial_filters=6):
    # Build U-Net model
    inpt = Input((IMG_DIM, IMG_DIM, IMG_CHANNELS))

    fltr = (filter_size, filter_size)
    trp_fltr = (transpose_filter_size, transpose_filter_size)
    pool_size = (2,2)
    num_filters = initial_filters

    conv_layers = []

    x = inpt
    #Descend
    for i in range(depth):
        x = Conv2D(num_filters, fltr, activation='relu', padding='same')(x)
        x = BNorm()(x)
        x = Conv2D(num_filters, fltr, activation='relu', padding='same')(x)
        x = BNorm()(x)
        conv_layers.append(x)
        x = MaxPooling2D(pool_size)(x)
        num_filters *= 2

    #Bottom
    x = Conv2D(num_filters, fltr, activation='relu', padding='same')(x)
    x = BNorm()(x)
    x = Conv2D(num_filters, fltr, activation='relu', padding='same')(x)
    x = BNorm()(x)

    #Ascend
    for i in range(depth-1, -1, -1):
        num_filters = (int)(num_filters / 2)
        x = Conv2DTranspose(
            num_filters, trp_fltr, strides=(2, 2), padding='same',
            kernel_initializer=ConvTransposeInterp()
        )(x)
        x = BNorm()(x)
        x = concatenate([x, conv_layers[i]])
        x = Conv2D(num_filters, fltr, activation='relu', padding='same')(x)
        x = BNorm()(x)
        x = Conv2D(num_filters, fltr, activation='relu', padding='same')(x)
        x = BNorm()(x)

    output = Conv2D(2, (1, 1), activation='softmax', name="output_conv")(x)
    flat_preds = keras.layers.Reshape([-1, 2])(output)
    return Model(inputs=inpt, outputs=flat_preds)
