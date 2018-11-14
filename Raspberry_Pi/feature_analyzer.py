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
SCALER_FILEPATH_PREFIX = "nn_"

SEGMENT_SIZE = 64
OVERLAP = 0.95
MDL = "_segment-" + str(SEGMENT_SIZE) + "_overlap-newf-" + str(OVERLAP * 100)

# for every segment of data, extract the feature vector
def extract_feature_vector(X):
    # extract acceleration and angular velocity
    X_accA = math.sqrt(sum(map(lambda x:x*x, np.mean(X[:, 0:3], axis=0))))
    X_accB = math.sqrt(sum(map(lambda x:x*x, np.mean(X[:, 3:6], axis=0))))
    X_gyro = math.sqrt(sum(map(lambda x:x*x, np.mean(X[:, 6:9], axis=0))))
    X_mag = np.asarray([ X_accA, X_accB, X_gyro ])
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
    res = np.append(X_mean, [ X_median, X_var, X_off, X_mad, X_entr ])
    # res = np.append(res, [ X_mag ])
    return res

# segment data from the raw data files, return list of tuples (segments, move_class)
# where every tuple represents raw data for that segment and the move_class for that segment
def get_all_segments(raw_data, move_class, scaler):
    # preprocess data
    # raw_data = [ data[:6] for data in raw_data ]
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
    raw_data_all_moves = pickle.load(open(os.path.join(DATASET_FILEPATH, 'data_by_move.pkl'), 'rb'))
    moves_to_filter = [ 'number7', 'salute' ]
    raw_data = {}
    for move in raw_data_all_moves:
        if len(raw_data_all_moves[move]) > 0:
            raw_data[move] = raw_data_all_moves[move]
    raw_data_all = []
    for move in raw_data:
        raw_data_all.extend(raw_data[move])
    scaler = pickle.load(open(os.path.join(SCALER_FILEPATH_PREFIX + 'scaler', 'min_max_scaler' + MDL + '.pkl'), 'rb'))
    data = []
    movesSampleData = {}
    for move in raw_data:
        segments = get_all_segments(raw_data[move], move, scaler)
        for segment in segments[:100]:
            X = extract_feature_vector(segment)
            print(move)
            print("\n" + str(X))
            data.append(X)
        movesSampleData[move] = np.mean(data, axis=0)
        data = []

    disp = list(map(list, np.transpose(disp)))
    count = 1
    rs = []
    for l in disp:
        range = round(np.max(l) - np.min(l), 2)
        c = str(count)
        if count < 10:
            c = "0" + c
        print(str(c) + ". " + " ".join([["", "+"][v > 0] + str('{0:.2f}'.format(v)) for v in l]) + " = range : " + '{0:.2f}'.format(range))
        rs.append((range, count))
        count += 1
    rs.sort(reverse=True)
    rs = [ str(t[1]) for t in rs ]
    print(", ".join(rs))
    print("Top 30 features: ")
    print(", ".join(rs[:30]))
    rs = [ int(t) for t in rs[:30] ]
    rs.sort()
    rs = [ str(t) for t in rs ]
    print(", ".join(rs))

    # for move in movesSampleData:
    #     print(str(move) + " : " + str(movesSampleData[move].tolist()))
    # print("diff : " + str(np.subtract(movesSampleData['number7'], movesSampleData['salute']).tolist()))
