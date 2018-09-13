import numpy as np
from statsmodels import robust
import pickle

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
    return np.append(X_mean, [X_var, X_max, X_min, X_off, X_mad, X_psd, X_peakF])

# segment data from the raw data files, return list of tuples (segments, move_class)
# where every tuple represents raw data for that segment and the move_class for that segment
def get_all_segments(raw_data, move_class):


if __name__ == "__main__":
    # Get all segments for every move one by one
    # for every segment for a given move, extract the feature vector
    # in the end, store a list of tuple pairs of (feature_vector, move_class) to pickle file
