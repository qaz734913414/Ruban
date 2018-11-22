# -*- coding: utf-8 -*-
# @Author: Hawkin
# @License: Apache Licence
# @File: densenet.py
# @Time: 2018/11/19 21:40
from keras import Input, Model
from keras.layers.core import Dropout, Activation, Dense
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import numpy as np

from ..nets.stn import STN


def conv_block(input_tensor, growth_rate, dropout_rate=None):
    x = Activation('relu')(input_tensor)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, dropout_rate)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(input_tensor, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = Activation('relu')(input_tensor)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    if pooltype == 2:
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif pooltype == 1:
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif pooltype == 3:
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter


def MiniSTNDenseNet(input_tensor):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    x = Conv2D(_nb_filter, (3, 3), strides=(2, 2), padding='same', activation='relu',
               kernel_regularizer=l2(_weight_decay))(x)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)

    # 64 + 8 * 8 = 128
    x, _ = dense_block(x, 8, _nb_filter, 8, None)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _ = dense_block(x, 8, _nb_filter, 8, None)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _ = dense_block(x, 8, _nb_filter, 8, None)

    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)

    '''----------------------STN-------------------------'''
    stn_input_shape = x.get_shape()
    loc_input_shape = (stn_input_shape[1].value, stn_input_shape[2].value, stn_input_shape[3].value)

    loc_b = np.zeros((2, 3), dtype='float32')
    loc_b[0, 0] = 1
    loc_b[1, 1] = 1
    loc_w = np.zeros((64, 6), dtype='float32')
    loc_weights = [loc_w, loc_b.flatten()]

    loc_input = Input(loc_input_shape)

    loc_x = Conv2D(16, (3, 3), padding='same', activation='relu')(loc_input)
    loc_x = Conv2D(32, (3, 3), padding='same', strides=(2, 2), activation='relu')(loc_x)
    loc_x = Conv2D(64, (3, 3), padding='same', activation='relu')(loc_x)
    # x = Flatten()(x)
    loc_x = GlobalAveragePooling2D()(loc_x)
    # x = Dense(64, activation='relu')(x)
    loc_x = Dense(6, weights=loc_weights)(loc_x)

    loc_output = Model(inputs=loc_input, outputs=loc_x)

    x = STN(localization_net=loc_output, output_size=(loc_input_shape[0], loc_input_shape[1]))(x)
    '''----------------------STN-------------------------'''

    encoder = GlobalAveragePooling2D()(x)

    return encoder
