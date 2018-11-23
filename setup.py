# -*- coding: utf-8 -*-
# @Author: Hawkin
# @License: Apache Licence
# @File: setup.py
# @Time: 2018/10/24 23:08

from setuptools import setup
from setuptools import find_packages
print(find_packages())

setup(name='ruban',
      version='1.0.0',
      author='Ruban Seven',
      author_email='rubanseven@163.com',
      license='Apache',
      install_requires=['keras',
                        'pyyaml',
                        'h5py'],
      packages=find_packages(),
      data_files=[('ruban/models', ['ruban/models/character_weights.h5'])],
      )
