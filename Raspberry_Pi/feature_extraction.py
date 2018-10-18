import os, pickle, logging
import numpy as np
from statsmodels import robust
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, Normalizer
from obspy.signal.filter import highpass
from scipy.signal import savgol_filter

# initialise logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATASET_FILEPATH = "dataset"

# Default: 64 sets per segment with 0% overlap
SEGMENT_SIZE = 64
OVERLAP = 0

# for every segment of data, extract the feature vector
def extract_feature_vector(X):
    # extract time domain features
    X_mean = np.mean(X, axis=0)
    X_var = np.var(X, axis=0)
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    X_off = np.subtract(X_max, X_min)
    X_mad = robust.mad(X, axis=0)
    # extract frequency domain features
    X_psd = []
    X_peakF = []
    # return feature vector by appending all vectors above as one d-dimension feature vector
    return np.append(X_mean, [X_var, X_max, X_min, X_off, X_mad])

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
    raw_data_all_moves = pickle.load(open(os.path.join(DATASET_FILEPATH, 'data_by_move.pkl'), 'rb'))
    raw_data = {}
    for move in raw_data_all_moves:
        if len(raw_data_all_moves[move]) > 0:
            raw_data[move] = raw_data_all_moves[move]
    scaler = MinMaxScaler((-1,1))
    raw_data_all = []
    for move in raw_data:
        raw_data_all.extend(raw_data[move])
    scaler.fit(raw_data_all)
    pickle.dump(scaler, open(os.path.join('scaler', 'min_max_scaler.pkl'), 'wb'))
    data = []
    for move in raw_data:
        segments = get_all_segments(raw_data[move], move, scaler)
        for segment in segments:
            X = extract_feature_vector(segment)
            logger.info(move + " " + str(X))
            data.append((X, move))
    X, Y = zip(*data)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True, stratify=Y)
    pickle.dump([X_train, Y_train], open(os.path.join(DATASET_FILEPATH, 'train.pkl'), 'wb'))
    pickle.dump([X_val, Y_val], open(os.path.join(DATASET_FILEPATH, 'test.pkl'), 'wb'))
