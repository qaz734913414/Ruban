# -*- coding: utf-8 -*-
# @Author: Hawkin
# @License: Apache Licence
# @File: resnet.py
# @Time: 2018/10/28 10:38


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras import backend as K
from keras import layers
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, RepeatVector, GRU, Bidirectional, TimeDistributed, Dropout
from keras.models import Model
from keras.regularizers import l2

from .stn import STN


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def MiniStnResNet(img_input=None, classes=62):
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(img_input)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)

    x = conv_block(x, 3, [64, 64, 192], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 192], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 192], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 192], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 192], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 192], stage=3, block='c')
    # x = identity_block(x, 3, [128, 128, 192], stage=3, block='d')

    x = conv_block(x, 3, [128, 128, 192], stage=4, block='a')
    x = identity_block(x, 3, [128, 128, 192], stage=4, block='b')
    x = identity_block(x, 3, [128, 128, 192], stage=4, block='c')
    # x = identity_block(x, 3, [128, 128, 192], stage=4, block='d')
    # x = identity_block(x, 3, [128, 128, 192], stage=4, block='e')
    # x = identity_block(x, 3, [128, 128, 192], stage=4, block='f')

    x = conv_block(x, 3, [128, 128, 192], stage=5, block='a')
    x = identity_block(x, 3, [128, 128, 192], stage=5, block='b')
    x = identity_block(x, 3, [128, 128, 192], stage=5, block='c')

    '''----------------------STN-------------------------'''
    stn_input_shape = x.get_shape()
    loc_input_shape = (stn_input_shape[1].value, stn_input_shape[2].value, stn_input_shape[3].value)

    loc_b = np.zeros((2, 3), dtype='float32')
    loc_b[0, 0] = 1
    loc_b[1, 1] = 1
    loc_w = np.zeros((32, 6), dtype='float32')
    loc_weights = [loc_w, loc_b.flatten()]

    loc_input = Input(loc_input_shape)

    loc_x = Conv2D(16, (3, 3), padding='same', activation='relu')(loc_input)
    loc_x = Conv2D(32, (3, 3), padding='same', activation='relu')(loc_x)
    # x = Flatten()(x)
    loc_x = GlobalAveragePooling2D()(loc_x)
    # x = Dense(64, activation='relu')(x)
    loc_x = Dense(6, weights=loc_weights)(loc_x)

    loc_output = Model(inputs=loc_input, outputs=loc_x)

    x = STN(localization_net=loc_output, output_size=(loc_input_shape[0], loc_input_shape[1]))(x)
    '''----------------------STN-------------------------'''

    encoder = GlobalAveragePooling2D()(x)

    return encoder
