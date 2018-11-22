# -*- coding: utf-8 -*-
# @Author: Hawkin
# @License: Apache Licence
# @File: characterTest.py
# @Time: 2018/11/19 22:16

from ruban.applications import InkFountain
import cv2


def train():
    ink_fountain = InkFountain(sample_path=r'InkFountainData', lr=0.001)
    ink_fountain.train()


def test():
    ink_fountain = InkFountain(sample_path=r'InkFountainData')
    img = cv2.imread(r'InkFountainData\img\aacl.jpg')
    print(ink_fountain.predict(img))


if __name__ == '__main__':
    train()
    test()
