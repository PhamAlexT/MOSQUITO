# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd

FEATURE_EXT = "_feat.name"
X_TRAIN_EXT = "_train.data"
Y_TRAIN_EXT = "_train.solution"
X_TEST_EXT = "_test.data"
X_VALID_EXT = "_valid.data"
CHALLENGE_NAME = "malaria"
RAW_DATA_DIR = "/home/estrade/Bureau/Teaching/L2-ML/challenges/MediChal/raw/malaria_input_data"
PRE_DATA_DIR = "/home/estrade/Bureau/Teaching/L2-ML/challenges/MediChal/preprocessed/malaria_input_data"
DATA_DIR = PRE_DATA_DIR


def load_images(data_dir=RAW_DATA_DIR, challenge_name=CHALLENGE_NAME):
    X_train = _load_data(data_dir, challenge_name, X_TRAIN_EXT)
    y_train = _load_data(data_dir, challenge_name, Y_TRAIN_EXT, dtype=np.int32)
    X_train = X_train.reshape(-1, 1, 50, 50)
    y_train = y_train.reshape(-1)
    return X_train, y_train

def load_train(data_dir=PRE_DATA_DIR, challenge_name=CHALLENGE_NAME):
    X_train = _load_data(data_dir, challenge_name, X_TRAIN_EXT)
    y_train = _load_data(data_dir, challenge_name, Y_TRAIN_EXT, dtype=np.int32)
    y_train = y_train.reshape(-1)
    feature_names = _load_feature_name(data_dir, challenge_name)
    data_train = pd.DataFrame(X_train, columns=feature_names)
    data_train['target'] = y_train
    return data_train

def load_valid(data_dir=PRE_DATA_DIR, challenge_name=CHALLENGE_NAME):
    X_valid = _load_data(data_dir, challenge_name, X_VALID_EXT)
    feature_names = _load_feature_name(data_dir, challenge_name)
    data_valid = pd.DataFrame(X_valid, columns=feature_names)
    return data_valid

def load_test(data_dir=PRE_DATA_DIR, challenge_name=CHALLENGE_NAME):
    X_test = _load_data(data_dir, challenge_name, X_TEST_EXT)
    feature_names = _load_feature_name(data_dir, challenge_name)
    data_test = pd.DataFrame(X_test, columns=feature_names)
    return data_test

MAX_ROWS = None # TODO set to None when everything works

def _load_data(data_dir, challenge_name, extension, dtype=np.float32):
    path = os.path.join(data_dir, challenge_name+extension)
    data = pd.read_csv(path, sep=" ", header=None).values
    return data

def _load_feature_name(data_dir, challenge_name):
    path = os.path.join(data_dir, challenge_name+FEATURE_EXT)
    with open(path, 'r') as f:
        feature_names = f.read().splitlines()
    return feature_names

