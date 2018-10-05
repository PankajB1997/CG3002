import numpy as np
from statsmodels import robust
import pickle
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, Normalizer
from obspy.signal.filter import highpass
from scipy.signal import savgol_filter
from keras.models import load_model

# initialise logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ENC_LIST = [
    ('idle', 0),
    ('logout', 1),
    ('number_six', 2)
]

ENC_DICT = {
    0: 'idle',
    1: 'logout',
    2: 'number_six'
}

CLASSLIST = [ pair[0] for pair in ENC_LIST ]

# Obtain best class from a given list of class probabilities for every prediction
def onehot2str(onehot):
       enc_dict = dict([(i[1],i[0]) for i in ENC_LIST])
       idx_list = np.argmax(onehot, axis=1).tolist()
       result_str = []
       for i in idx_list:
               result_str.append(enc_dict[i])
       return np.asarray(result_str)

# Convert a class to its corresponding one hot vector
def str2onehot(Y):
   enc_dict = dict(ENC_LIST)
   new_Y = []
   for y in Y:
       vec = np.zeros((1,len(ENC_LIST)),dtype='float64')
       vec[ 0, enc_dict[y] ] = 1.
       new_Y.append(vec)
   del Y
   new_Y = np.vstack(new_Y)
   return new_Y

# Load model from pickle/hdf5 file
model = load_model('nn_models\\nn_model.hdf5')
# model = pickle.load(open('classifier_models\\model_RandomForestClassifier200.pkl', 'rb'))
# Load scalers
min_max_scaler = pickle.load(open('scaler\\min_max_scaler.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler\\standard_scaler.pkl', 'rb'))

# for every segment of data, extract the feature vector
def extract_feature_vector(X):
    # Default: 128 sets per segment with 50% overlap; currently, 8 segments per set is used due to insufficient data
    SEGMENT_SIZE = 8
    OVERLAP = 0.5
    # preprocess data
    X = savgol_filter(X, 3, 2)
    X = highpass(X, 3, 50)
    X = min_max_scaler.transform(X)
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
    # obtain feature vector by appending all vectors above as one d-dimension feature vector
    X = np.append(X_mean, [X_var, X_max, X_min, X_off, X_mad])
    return standard_scaler.transform([X])

def predict_dance_move(segment):
    X = extract_feature_vector(segment)
    # return model.predict(X).tolist()[0]
    return onehot2str(model.predict(X))[0]

if __name__ == '__main__':
    segment_idle = [
        [0.00, -0.01, 0.99],
        [0.01, -0.01, 0.99],
        [0.00, 0.00, 0.99],
        [-0.00, 0.00, 0.99],
        [-0.00, -0.00, 0.99],
        [-0.00, -0.00, 1.00],
        [-0.00, -0.00, 1.00],
        [0.00, -0.01, 0.99]
    ]

    segment_logout = [
        [-0.11, 0.05, 0.66],
        [-0.19, 0.11, 0.72],
        [-0.20, 0.11, 0.54],
        [-0.18, 0.07, 0.45],
        [-0.16, 0.04, 0.39],
        [-0.16, 0.09, 0.45],
        [-0.14, 0.11, 0.56],
        [-0.09, 0.11, 0.74]
    ]

    segment_numbersix = [
        [0.57, -0.67, 0.91],
        [0.52, -0.69, 0.92],
        [0.54, -0.67, 0.96],
        [0.68, -0.65, 1.11],
        [0.67, -0.63, 1.18],
        [0.72, -0.72, 1.13],
        [0.72, -0.77, 1.04],
        [0.65, -0.75, 1.01]
    ]

    print("Dance Move 1: " + predict_dance_move(segment_idle))
    print("Dance Move 2: " + predict_dance_move(segment_logout))
    print("Dance Move 3: " + predict_dance_move(segment_numbersix))
