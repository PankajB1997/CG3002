import os, pickle, logging
import numpy as np
from statsmodels import robust
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, Normalizer
from obspy.signal.filter import highpass
from scipy.signal import savgol_filter, periodogram, welch
from scipy.fftpack import fft, ifft, rfft

def extract_feature_vector(X):
    # extract time domain features
    X_mean = np.mean(X, axis=0)
    X_var = np.var(X, axis=0)
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    X_off = np.subtract(X_max, X_min)
    X_mad = robust.mad(X, axis=0)
    # extract frequency domain features
    X_fft_abs = np.abs(fft(X)) #np.abs() if you want the absolute val of complex number
    X_fft_mean = np.mean(X_fft_abs, axis=0)
    X_fft_var = np.var(X_fft_abs, axis=0)
    X_fft_max = np.max(X_fft_abs, axis=0)
    X_fft_min = np.min(X_fft_abs, axis=0)
    # logger.info("hello ")
    # logger.info(X)

    # X_psd = np.mean(periodogram(X))
    # logger.info("hello ")
    # logger.info(X_psd)

    X_peakF = []
    # return feature vector by appending all vectors above as one d-dimension feature vector
    return np.append(X_mean, [])

chicken = []
clist = []
with open('mermaid.txt') as f:
    lines = f.read().split("\n")[:-1]
    # print(lines)
    for line in lines:
        l = line.split("\t")
        if not len(l) == 9:
            continue
        clist.append([ float(val) for val in l ])
    N = 64
    for i in range(0, len(clist), N):
        chicken.append(extract_feature_vector(clist[i : i+N]))

s = []
for c in chicken:
    s.append("\t".join([ str(round(val, 2)) for val in c ]))

with open('output.txt', 'w') as f:
    f.write("\n".join(s))
