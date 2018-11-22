# -*- coding: utf-8 -*-
# @Author: Hawkin
# @License: Apache Licence
# @File: data_utils.py
# @Time: 2018/10/26 0:00


import os
import re
import codecs
import numpy as np
import cv2


def img_read(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print('{} error:{}'.format(img_path, e))
        return None


def load_character_sample(sample_root):
    char_idx_dict = dict()
    idx_char_dict = dict()
    char_count_dict = dict()
    sample_list = list()
    label_len_count_dict = dict()
    img_w_list = list()
    img_h_list = list()

    img_root = os.path.join(sample_root, r'img')
    txt_root = os.path.join(sample_root, r'txt')
    img_name_list = os.listdir(img_root)
    for img_name in img_name_list:
        img_path = os.path.join(img_root, img_name)
        txt_name = re.sub('[.][^.]+$', '.txt', img_name)
        txt_path = os.path.join(txt_root, txt_name)
        img = img_read(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        img_h_list.append(img_h)
        img_w_list.append(img_w)
        if os.path.exists(txt_path):
            with codecs.open(txt_path, 'rb', encoding='utf-8') as txt_file:
                lines = txt_file.readlines()
                chars = lines[0].split('\t')
                label = list()
                for char in chars:
                    if char not in char_idx_dict:
                        idx_char_dict[len(char_idx_dict)] = char
                        char_idx_dict[char] = len(char_idx_dict)
                        char_count_dict[char] = 1
                    else:
                        char_count_dict[char] += 1
                    label.append(char_idx_dict[char])
                label_len = len(label)
                if label_len not in label_len_count_dict:
                    label_len_count_dict[label_len] = 1
                else:
                    label_len_count_dict[label_len] += 1
                sample_list.append([img_path, label])

    print('---------------char count---------------')
    for char in char_count_dict:
        print('{}:\t{}'.format(char, char_count_dict[char]))
    print('----------------------------------------')
    print('-----------label length count-----------')
    for label_len in label_len_count_dict:
        print('{}:\t{}'.format(label_len, label_len_count_dict[label_len]))
    print('----------------------------------------')

    max_label_len = np.max([label_len for label_len in label_len_count_dict])

    avg_img_h = int(np.average(img_h_list))
    avg_img_w = int(np.average(img_w_list))

    return sample_list, idx_char_dict, max_label_len, avg_img_w, avg_img_h
