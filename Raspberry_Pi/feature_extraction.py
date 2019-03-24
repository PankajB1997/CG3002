import os, pickle, logging
import numpy as np
from statsmodels import robust
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, Normalizer
from obspy.signal.filter import highpass
from scipy.signal import savgol_filter, periodogram, welch
from scipy.fftpack import fft, ifft, rfft
from scipy.stats import entropy
import math

# Fix seed value for reproducibility
np.random.seed(1234)

# initialise logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATASET_FILEPATH = "dataset"
SCALER_FILEPATH_PREFIX = ""

SEGMENT_SIZE = 32
OVERLAP = 0
MDL = "_segment-" + str(SEGMENT_SIZE) + "_overlap-newf-" + str(OVERLAP * 100)

# for every segment of data, extract the feature vector
def extract_feature_vector(X):
    # # extract acceleration and angular velocity
    # X_accA = math.sqrt(sum(map(lambda x:x*x, np.mean(X[:, 0:3], axis=0))))
    # X_accB = math.sqrt(sum(map(lambda x:x*x, np.mean(X[:, 3:6], axis=0))))
    # X_gyro = math.sqrt(sum(map(lambda x:x*x, np.mean(X[:, 6:9], axis=0))))
    # X_mag = np.asarray([ X_accA, X_accB, X_gyro ])

    # extract time domain features
    X_mean = np.mean(X, axis=0)
    X_median = np.median(X, axis=0)
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
    X_entr = entropy(np.abs(np.fft.rfft(X, axis=0))[1:], base=2)

    # return feature vector by appending all vectors above as one d-dimension feature vector
    return np.append(X_off, [ X_mean, X_median, X_var, X_mad, X_entr, X_min, X_max, X_fft_mean, X_fft_var, X_fft_max, X_fft_min ])

# segment data from the raw data files, return list of tuples (segments, move_class)
# where every tuple represents raw data for that segment and the move_class for that segment
def get_all_segments(raw_data, move_class, scaler):
    # preprocess data
    raw_data = savgol_filter(raw_data, 3, 2)
    raw_data = highpass(raw_data, 3, 50)
    raw_data = scaler.transform(raw_data)
    # extract segments
    limit = (len(raw_data) // SEGMENT_SIZE ) * SEGMENT_SIZE
    segments = []
    for i in range(0, limit, int(SEGMENT_SIZE * (1 - OVERLAP))):
        segment = raw_data[i: (i + SEGMENT_SIZE)]
        segments.append(segment)
    return segments

if __name__ == "__main__":
    # Get all segments for every move one by one
    # for every segment for a given move, extract the feature vector
    # in the end, store a list of tuple pairs of (feature_vector, move_class) to pickle file
    raw_data_all_moves = pickle.load(open(os.path.join(DATASET_FILEPATH, 'left_right.pkl'), 'rb'))
    raw_data = {}
    for move in raw_data_all_moves:
        if len(raw_data_all_moves[move]) > 0:
            raw_data[move] = raw_data_all_moves[move]
    scaler = MinMaxScaler((-1,1))
    raw_data_all = []
    for move in raw_data:
        raw_data_all.extend(raw_data[move])
    # raw_data_all = [ data[2:6] for data in raw_data_all ]
    scaler.fit(raw_data_all)
    pickle.dump(scaler, open(os.path.join(SCALER_FILEPATH_PREFIX + 'scaler', 'min_max_scaler' + MDL + '.pkl'), 'wb'))
    data = []
    for move in raw_data:
        segments = get_all_segments(raw_data[move], move, scaler)
        isPrinted = False
        for segment in segments:
            X = extract_feature_vector(segment)
            if isPrinted == False:
                logger.info(move)
                logger.info("\n" + str(X))
                isPrinted = True
            data.append((X, move))
    X, Y = zip(*data)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True, stratify=Y)
    pickle.dump([X_train, Y_train], open(os.path.join(DATASET_FILEPATH, 'train.pkl'), 'wb'))
    pickle.dump([X_val, Y_val], open(os.path.join(DATASET_FILEPATH, 'test.pkl'), 'wb'))
