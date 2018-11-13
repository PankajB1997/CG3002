#!/usr/bin/python3

import time

wait_time = int(round(time.time() * 1000))

import traceback
import serial
import os
import sys
import threading
import socket
import base64
from Crypto import Random
from Crypto.Cipher import  AES

import numpy as np
from statsmodels import robust
import pickle
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, Normalizer
from obspy.signal.filter import highpass
from scipy.signal import savgol_filter
from scipy.fftpack import fft, ifft, rfft
from keras.models import load_model

# Fix seed value for reproducibility
np.random.seed(1234)

N = 128
OVERLAP = 0.75
EXTRACT_SIZE = int((1 - OVERLAP) * N)
MDL = "_segment-" + str(N) + "_overlap-newf-" + str(OVERLAP * 100)
CONFIDENCE_THRESHOLD = 0.75
INITIAL_WAIT = 61500 # in milliseconds
MOVE_BUFFER_MIN_SIZE = 2

# Initialize WAIT value in milliseconds depending on N and OVERLAP values
WAIT = 940 # for best case prediction time 3.5 seconds for N=64 and OVERLAP=0
if N == 128 and OVERLAP == 0.75:
    WAIT = 1000 # for best case prediction time 4.2 seconds
elif N == 128 and OVERLAP == 0.50:
    WAIT = 1160 # for best case prediction time 5.0 seconds
elif N == 128 and OVERLAP == 0.25:
    WAIT = 1020 # for best case prediction time 5.5 seconds
elif N == 128 and OVERLAP == 0:
    WAIT = 880 # for best case prediction time 6.0 seconds
elif N == 64 and OVERLAP == 0.75:
    WAIT = 1400 # for best case prediction time 3.0 seconds
elif N == 64 and OVERLAP == 0.50:
    WAIT = 1080 # for best case prediction time 3.0 seconds
elif N == 64 and OVERLAP == 0.25:
    WAIT = 1260 # for best case prediction time 3.5 seconds

secret_key = "1234123412341234"  # must be at least 16
BLOCK_SIZE = 32 # AES.block_size

'''
The following move states are used:
IDLE (with move_state as 1)
DETERMINING_DANCE_MOVE (with move_state as 2)

The machine begins in IDLE state. When idle dance move is received, it moves to DETERMINING_DANCE_MOVE state.
When dance move is successfully determined and sent to the server, it moves back to IDLE state.
'''

# Begin with IDLE state
move_state = 1

STATE = {
    1: 'IDLE',
    2: 'DETERMINING_DANCE_MOVE'
}

ENC_LIST = [
    ('sidestep', 0),
    ('number7', 1),
    ('chicken', 2),
    ('wipers', 3),
    ('turnclap', 4),
    ('numbersix', 5),
    ('salute', 6),
    ('mermaid', 7),
    ('swing', 8),
    ('cowboy', 9),
    ('logout', 10)
    # ('IDLE', 11),
]

ENC_DICT = {
    0: 'sidestep',
    1: 'number7',
    2: 'chicken',
    3: 'wipers',
    4: 'turnclap',
    5: 'numbersix',
    6: 'salute',
    7: 'mermaid',
    8: 'swing',
    9: 'cowboy',
    10: 'logout'
    # 11: 'IDLE',
}

CLASSLIST = [ pair[0] for pair in ENC_LIST ]

danceMoveBuffer = []

previousPacketData = []

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

try:
    # Load model from h5py file
    model = load_model(os.path.join('nn_models', 'nn_model' + MDL + '.hdf5'))
except:
    traceback.print_exc()
    print("Error in loading pretrained model!")
    exit()

try:
    # Load scalers
    min_max_scaler = pickle.load(open(os.path.join('nn_scaler', 'min_max_scaler' + MDL + '.pkl'), 'rb'))
    standard_scaler = pickle.load(open(os.path.join('nn_scaler', 'standard_scaler' + MDL + '.pkl'), 'rb'))
except:
    traceback.print_exc()
    print("Error in loading scaler objects!")
    exit()

# for every segment of data, extract the feature vector
def extract_feature_vector(X):
    try:
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
        X_fft_abs = np.abs(fft(X)) #np.abs() if you want the absolute val of complex number
        X_fft_mean = np.mean(X_fft_abs, axis=0)
        X_fft_var = np.var(X_fft_abs, axis=0)
        X_fft_max = np.max(X_fft_abs, axis=0)
        X_fft_min = np.min(X_fft_abs, axis=0)
        X_entr = entropy(np.abs(np.fft.rfft(X, axis=0))[1:], base=2)
        # return feature vector by appending all vectors above as one d-dimension feature vector
        return standard_scaler.transform([np.append(X_mean, [ X_var, X_entr, X_off, X_mad ])])
        # , X_fft_mean, X_fft_var, X_fft_max, X_fft_min
    except:
        traceback.print_exc()
        print("Error in predicting dance move!")

def predict_dance_move(segment):
    try:
        X = extract_feature_vector(segment)
        Y = model.predict(np.expand_dims(X, axis=2))
        print(Y)
        return onehot2str(Y)[0], max(Y[0])
    except:
        traceback.print_exc()
        print("Error in predicting dance move!")

def readLineCR(port):
    rv = ""
    while True:
        ch = port.read().decode()
        rv += ch
        # print("I'm reading " + ch)
        if ch == "\r" or ch == "":
        # if ch == "\r":
            return rv

def inputData():
    #'#action | voltage | current | power | cumulativepower|'
    action = str(input('Manually enter data: '))
    data = '#' + action + '|2.0|1.5|5.6|10.10|'
    return data

def encryption(data, secret_key):
	#Padding
	length = BLOCK_SIZE-(len(data)%BLOCK_SIZE)
	msg = data+((chr(length))*(length))
	print(msg)

	#encryption
	iv = Random.new().read(AES.block_size)
	cipher = AES.new(secret_key, AES.MODE_CBC, iv)
	encoded = base64.b64encode(iv + cipher.encrypt(msg))

	return encoded

def sendToServer(s, data):
	encryptedData = encryption(data, secret_key)
	s.send(encryptedData)
	print('output sent to server')

def lastXDanceMovesSame(danceMoveBuffer):
        lastXMoves = danceMoveBuffer[-MOVE_BUFFER_MIN_SIZE:]
        return len(set(lastXMoves)) == 1

try:
    #Establish socket connection
    #input on console in this format: IP_address Port_number
    TCP_IP = sys.argv[1]
    TCP_PORT = int(sys.argv[2])
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
except:
    traceback.print_exc()
    print("Error in establishing socket connection with the server!")
    exit()

#Check if connected to server
if (s):
	print("connected to server")
else:
	print("not connected to server")

#Establish arduino-rpi connection
# dataArray = [] # N objects in array, per 20ms
handshake_flag = False
data_flag = False
print("test")
port=serial.Serial("/dev/serial0", baudrate=115200, timeout=3.0)
print("set up")
port.reset_input_buffer()
port.reset_output_buffer()

while (handshake_flag == False):
    try:
        port.write("H".encode())
        print("H sent")
        response = port.read(1)
        time.sleep(0.5)
        if (response.decode() == "A"):
            print("A received, sending N")
            port.write("N".encode())
            time.sleep(0.5)
            handshake_flag= True
        else:
            time.sleep(0.5)
    except:
        traceback.print_exc()
        print("Error while attempting a handshake!")

port.reset_input_buffer()
port.reset_output_buffer()
print("connected")

countMovesSent = 0
stoptime = int(round(time.time() * 1000))
ite = N
while (data_flag == False):

    print("ENTERING")

    movementData = []
    otherData = []
    try:
        if not len(previousPacketData) == 0:
            ite = EXTRACT_SIZE
        else:
            ite = N
        for i in range(ite): # extract from 0->N-1 = N sets of readings
            data = readLineCR(port).split(',')
            print(data)
            if not len(data) == 13:
               continue
            data = [ float(val.strip()) for val in data ]
            movementData.append(data[:9]) # extract acc1[3], and acc2[3] values
            otherData.append(data[9:]) # extract voltage, current, power and cumulative power
    except:
        traceback.print_exc()
        print("Error in reading packet!")
        continue

    if int(round(time.time() * 1000)) - wait_time <= INITIAL_WAIT:
        continue

    print(len(previousPacketData))
    print(len(movementData))

    diff = int(round(time.time() * 1000)) - stoptime
    if diff <= WAIT:
        continue

    # print(otherData)

    # Add overlapping logic
    if len(previousPacketData) == N - EXTRACT_SIZE and not EXTRACT_SIZE == N:
        rawData = previousPacketData + movementData
        print("Overlap done")
    else:
        rawData = movementData[:]
        print("Overlap not done")

    # Add ML Logic
    # Precondition 1: dataArray has values for acc1[3], acc2[3], gyro[3], voltage[1], current[1], power[1] and energy[1] in that order
    # Precondition 2: dataArray has N sets of readings, where N is the segment size, hence it has dimensions N*13
    try:
        danceMove, predictionConfidence = predict_dance_move(rawData)
        if predictionConfidence > CONFIDENCE_THRESHOLD:
            danceMoveBuffer.append(danceMove)
        print(len(rawData))
        print(danceMoveBuffer)
    except:
        traceback.print_exc()
        print("Error in prediction!")
        continue

    isMoveSent = False
    if len(danceMoveBuffer) >= MOVE_BUFFER_MIN_SIZE and lastXDanceMovesSame(danceMoveBuffer) == True:
        try:
            otherData = np.mean(otherData, axis=0).tolist() # only calculated for overlapped part of the segment
            voltage = otherData[0]
            current = otherData[1]
            power = otherData[2]
            energy = otherData[3]
            output = "#" + danceMove + "|" + str(round(voltage, 2)) + "|" + str(round(current, 2)) + "|" + str(round(power, 2)) + "|" + str(round(energy, 2)) + "|"
            if danceMove == "logout" and not countMovesSent >= 40: # only allow logout to be sent once 40 moves have been sent
                continue
            # Send output to server
            sendToServer(s, output)
            port.reset_input_buffer()
            port.reset_output_buffer()
            print("Sent to server: " + str(output) + ".")
            danceMoveBuffer = []
            stoptime = int(round(time.time() * 1000))
            isMoveSent = True
            countMovesSent += 1
        except:
            traceback.print_exc()
            print("Error in sending dance move to the server!")
            continue

    if isMoveSent == False:
        print("System did not change state. Dance move is " + str(danceMove) + " with prediction confidence " + str(predictionConfidence) + " and move buffer size is " + str(len(danceMoveBuffer)) + ".")

    # Add overlapping logic
    if EXTRACT_SIZE >= 0 and EXTRACT_SIZE < N:
        previousPacketData = rawData[EXTRACT_SIZE:]
    else:
        previousPacketData = []

    if isMoveSent == True:
        previousPacketData = []

    # data_flag = True
