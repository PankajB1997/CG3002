# import standard python libraries
import logging
import os, pickle, json, h5py, operator
import numpy as np
import pandas as pd
import obspy.signal.filter
import scipy.signal
from collections import Counter

#import modules for scipy savgol and obspy butterworth and use normalization from scikit
from sklearn import preprocessing

# initialise logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CG3002_FILEPATH = ""

#temporarily using x_train instead of raw data

X_TRAIN_TXT_PATH = os.path.join(CG3002_FILEPATH, "dummy_dataset/Train/X_train.txt")
Y_TRAIN_TXT_PATH = os.path.join(CG3002_FILEPATH, "dummy_dataset/Train/y_train.txt")

ENC_DICT = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING',
    6: 'STAND_TO_SIT',
    7: 'SIT_TO_STAND',
    8: 'SIT_TO_LIE',
    9: 'LIE_TO_SIT',
    10: 'STAND_TO_LIE',
    11: 'LIE_TO_STAND'
}

def loadDataset(X_PATH, Y_PATH):
    X = []
    Y = []
    with open(X_PATH) as x_file:
        for input in x_file:
            X.append(list(map(float, input[:-1].split(" "))))
    with open(Y_PATH) as y_file:
        for input in y_file:
            Y.append(ENC_DICT[int(input) - 1])
    classes_removed = [
        'STAND_TO_SIT',
        'SIT_TO_STAND',
        'SIT_TO_LIE',
        'LIE_TO_SIT',
        'STAND_TO_LIE',
        'LIE_TO_STAND'
    ]

    del_idx = [ idx for idx, val in enumerate(Y) if val in classes_removed ]
    X = np.delete(X, del_idx, axis=0)
    Y = np.delete(Y, del_idx)
    return X, Y

if __name__ == "__main__":

    logger.info("Preprocessing...")
    X, Y = loadDataset(X_TRAIN_TXT_PATH, Y_TRAIN_TXT_PATH)

    logger.info("X...")
    logger.info(X)

    #scipy.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    X_savgol = scipy.signal.savgol_filter(X, 11, 5)

    logger.info("X_savgol...")
    logger.info(X_savgol)

    #highpass(data, freq, df, corners=4, zerophase=False) - we choose 3hz for cutoff due to summary from uci dataset
    X_highpass = obspy.signal.filter.highpass(X_savgol, 3, 50)

    logger.info("X_highpass...")
    logger.info(X_highpass)

    #normalization maybe using minmaxscaler from scikit - (to range -1, 1)
    min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    X_train_minmax = min_max_scaler.fit_transform(X_highpass)

    logger.info("X_train_minmax...")
    logger.info(X_train_minmax)
