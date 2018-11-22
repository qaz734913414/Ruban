# -*- coding: utf-8 -*-
# @Author: Hawkin
# @License: Apache Licence
# @File: tensent_captcha_ocr.py
# @Time: 2018/8/5 9:14

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, RepeatVector, GRU, TimeDistributed, Bidirectional, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage
import pickle

# from ..nets.resnet import MiniStnResNet
from ..nets.densenet import MiniSTNDenseNet
import keras.callbacks
import os
from keras.utils import np_utils
import cv2
from ..utils.data_utils import load_character_sample

np.random.seed(55)


# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

def speckle(img):
    uniform_range = np.random.random() * 0.6
    severity = np.random.uniform(0, uniform_range)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


# import random
def read_sample(img_path, img_w, img_h, is_train):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (img_w, img_h))
    img = np.array(img, dtype=np.float) / 255.0
    if is_train:
        img = speckle(img)

    return img


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret


# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(keras.callbacks.Callback):
    def __init__(self, output_dir, pic_path_list, label_list, minibatch_size, nb_classes, label_length, img_w, img_h,
                 sample_count, val_split, generator):
        super().__init__()
        self.output_dir = output_dir
        self.pic_path_list = pic_path_list
        self.label_list = label_list
        self.minibatch_size = minibatch_size
        self.sample_count = sample_count
        self.val_split = val_split
        self.cur_train_index = 0
        self.cur_val_index = val_split
        self.nb_classes = nb_classes
        self.label_length = label_length
        self.img_w = img_w
        self.img_h = img_h
        self.acc = 0
        self.generator = generator
        self.train_paint_func = lambda img_path: read_sample(img_path, img_w, img_h, True)
        self.test_paint_func = lambda img_path: read_sample(img_path, img_w, img_h, False)
        # self.test_result_list = list()

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, is_train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            x_data = np.ones([size, 3, self.img_h, self.img_w])
        else:
            x_data = np.ones([size, self.img_h, self.img_w, 3])

        y_data = np.ones([size, self.label_length, self.nb_classes])
        for i in range(size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if is_train:
                if K.image_data_format() == 'channels_first':
                    x_data[i, :, :, :] = self.train_paint_func(self.pic_path_list[index + i])[:, :, :].T
                else:
                    x_data[i, :, :, :] = self.train_paint_func(self.pic_path_list[index + i])[:, :, :]
            else:
                if K.image_data_format() == 'channels_first':
                    x_data[i, :, :, :] = self.test_paint_func(self.pic_path_list[index + i])[:, :, :].T
                else:
                    x_data[i, :, :, :] = self.test_paint_func(self.pic_path_list[index + i])[:, :, :]
            y_data[i] = np_utils.to_categorical(self.label_list[index + i], num_classes=self.nb_classes)
        if is_train:
            gen = self.generator.flow(x_data, batch_size=size, shuffle=False)
            x_data = gen.next()
        inputs = x_data
        outputs = y_data
        return inputs, outputs

    def next_train(self):
        while 1:
            if self.cur_train_index + self.minibatch_size <= self.val_split:
                ret = self.get_batch(self.cur_train_index, self.minibatch_size, True)
                self.cur_train_index += self.minibatch_size
            else:
                ret = self.get_batch(self.cur_train_index, self.val_split - self.cur_train_index, True)
                self.cur_train_index = 0
                (self.pic_path_list, self.label_list) = shuffle_mats_or_lists(
                    [self.pic_path_list, self.label_list], self.val_split)

            yield ret

    def next_val(self):
        while 1:
            if self.cur_val_index + self.minibatch_size <= self.sample_count:
                ret = self.get_batch(self.cur_val_index, self.minibatch_size, False)
                self.cur_val_index += self.minibatch_size
            else:
                ret = self.get_batch(self.cur_val_index, self.sample_count - self.cur_val_index, False)
                self.cur_val_index = self.val_split

            yield ret

    def print_acc(self, pred_y, y):
        pred_y = np.argmax(pred_y, axis=-1)
        correct_count = ((pred_y == y).sum(axis=1) == self.label_length).sum()
        acc_count = ((pred_y == y).sum(axis=1)).sum()
        correct = correct_count * 100.0 / len(y)
        acc = acc_count * 100.0 / (len(y) * self.label_length)
        return correct, acc

    def cal_acc(self, x, y):
        pred_y = self.model.predict(x)
        # if len(self.test_result_list) == 100:
        #     self.test_result_list = self.test_result_list[1:]
        #     self.test_result_list.append(pred_y)
        # else:
        #     self.test_result_list.append(pred_y)
        # total_pred_y = self.test_result_list[-1].copy()
        # for i in range(1, len(self.test_result_list) // 10):
        #     total_pred_y += self.test_result_list[-1 - i * 10]
        y = np.argmax(y, axis=-1)
        correct, acc = self.print_acc(pred_y, y)
        # total_correct, total_acc = self.print_acc(total_pred_y, y)
        print('\ncorrect is %f%% acc is %f%%' % (correct, acc))
        # print('\ntotal correct is %f%% acc is %f%%' % (total_correct, total_acc))
        return acc

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.get_batch(self.val_split, self.sample_count - self.val_split, False)
        acc = self.cal_acc(x, y)
        if acc >= self.acc:
            self.model.save_weights(os.path.join(self.output_dir, 'weights.h5'))
            self.acc = acc


def load_sample_list(root):
    img_name_list = os.listdir(root)
    label_to_index = {}
    index = 0
    sample_list = []
    for img_name in img_name_list:
        if img_name.endswith('.jpg') or img_name.endswith('.png') or img_name.endswith('.bmp'):
            img_path = os.path.join(root, img_name)
            label = img_name.split('.')[0]
            index_str = []
            for char in label:
                if char not in label_to_index:
                    label_to_index[char] = index
                    index += 1
                index_str.append(label_to_index[char])
            sample_list.append([img_path, index_str])

    return sample_list, label_to_index


class InkFountain:
    def __init__(self, sample_path, lr=0.001):
        self.sample_path = sample_path
        self.lr = lr
        self.model = None
        self.sample_list = list()
        self.idx_char_dict = dict()
        self.img_w = 128
        self.img_h = 32
        self.nb_classes = 32
        self.max_label_len = 4
        self.init_model()

    def init_model(self):
        if self.sample_path is None or not os.path.exists(self.sample_path):
            raise ValueError('must give sample path.'
                             'include img and txt catalog.')

        label_path = os.path.join(self.sample_path, r'label.pkl')
        if os.path.exists(label_path):
            with open(label_path, 'rb') as label_file:
                self.idx_char_dict = pickle.load(label_file)
        try:
            self.sample_list, self.idx_char_dict, self.max_label_len, self.img_w, self.img_h = load_character_sample(
                self.sample_path)
        except Exception as e:
            print(e)

        self.nb_classes = len(self.idx_char_dict) + 1

        if K.image_data_format() == 'channels_first':
            input_shape = (3, self.img_h, self.img_w)
        else:
            input_shape = (self.img_h, self.img_w, 3)

        input_image = Input(shape=input_shape)
        encoder = MiniSTNDenseNet(input_tensor=input_image)

        decoder = RepeatVector(self.max_label_len)(encoder)

        rnn_size = 128
        decoder = Bidirectional(GRU(rnn_size, return_sequences=True))(decoder)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        pre_trained_weight = os.path.join(current_dir, r'../models/character_weights.h5')
        # pre_trained_weight = r'D:\python\Ruban\ruban\models\character_weights.h5'
        if os.path.exists(pre_trained_weight):
            pre_trained_pred = TimeDistributed(Dense(62, activation='softmax'))(Dropout(0.2)(decoder))
            pre_trained_model = Model(inputs=input_image, outputs=pre_trained_pred)
            pre_trained_model.load_weights(pre_trained_weight)
        predict = TimeDistributed(Dense(self.nb_classes, activation='softmax'))(Dropout(0.2)(decoder))
        self.model = Model(inputs=input_image, outputs=predict)
        model_path = os.path.join(self.sample_path, 'weights.h5')
        if os.path.exists(model_path):
            self.model.load_weights(model_path)

    def train(self):
        mini_batch_size = 32
        train_rate = 0.9

        label_path = os.path.join(self.sample_path, r'label.pkl')
        with open(label_path, 'wb') as label_file:
            pickle.dump(self.idx_char_dict, label_file)

        for i in range(10):
            np.random.shuffle(self.sample_list)
        # Input Parameters
        sample_count = len(self.sample_list)
        train_count = int(sample_count * train_rate)

        pic_path_list = []
        label_list = []
        for pic_path, label in self.sample_list:
            pic_path_list.append(pic_path)
            label_list.append(label)

        label_list = np.array(label_list)

        generator = ImageDataGenerator(rotation_range=3,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       channel_shift_range=0.05,
                                       zoom_range=0.1)

        img_gen = TextImageGenerator(output_dir=self.sample_path, pic_path_list=pic_path_list, label_list=label_list,
                                     minibatch_size=mini_batch_size, nb_classes=self.nb_classes,
                                     label_length=self.max_label_len, img_w=self.img_w, img_h=self.img_h,
                                     sample_count=sample_count, val_split=train_count, generator=generator)

        # adam = SGD(lr=self.lr)
        adam = Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        steps_per_epoch = train_count / mini_batch_size
        steps_per_epoch = steps_per_epoch if steps_per_epoch * mini_batch_size == train_count else steps_per_epoch + 1

        validation_steps = (sample_count - train_count) / mini_batch_size
        validation_steps = validation_steps if validation_steps * mini_batch_size == (
                sample_count - train_count) else validation_steps + 1

        self.model.fit_generator(generator=img_gen.next_train(), steps_per_epoch=steps_per_epoch,
                                 epochs=200, validation_steps=validation_steps,
                                 callbacks=[img_gen], initial_epoch=0)

    def predict(self, src):
        img = cv2.resize(src, (self.img_w, self.img_h))
        img = np.array(img, dtype=np.float) / 255.0
        y_pred = self.model.predict(np.array([img]))[0]
        y_pred = np.argmax(y_pred, axis=-1)
        text = ''.join([self.idx_char_dict[idx] for idx in y_pred])
        return text


if __name__ == '__main__':
    pass
