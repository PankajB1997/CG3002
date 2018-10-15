#!/usr/bin/python3

import time
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
from keras.models import load_model

N = 128
CONFIDENCE_THRESHOLD = 0.85

BLOCK_SIZE = 32 #AES.block_size

'''
The following move states are used:
IDLE (with move_state as 1)
DETERMINING_DANCE_MOVE (with move_state as 2)

The machine begins in IDLE state. When idle dance move is received, it moves to DETERMINING_DANCE_MOVE state.
When dance move is successfully determined and sent to the server, it moves back to IDLE state.
'''

# Begin with IDLE state
move_state = 1

ENC_LIST = [
    ('IDLE', 0),
    ('logout', 1),
    ('Wipers', 2),
    ('NumberSeven', 3),
    ('Chicken', 4),
    ('SideStep', 5),
    ('Turnclap', 6),
    # ('NumberSix', 7),
    # ('Salute', 8),
    # ('Mermaid', 9),
    # ('Swing', 10),
    # ('Cowboy', 11)
]

ENC_DICT = {
    0: 'IDLE',
    1: 'logout',
    2: 'Wipers',
    3: 'NumberSeven',
    4: 'Chicken',
    5: 'SideStep',
    6: 'Turnclap',
    # 7: 'NumberSix',
    # 8: 'Salute',
    # 9: 'Mermaid',
    # 10: 'Swing',
    # 11: 'Cowboy'
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

# for every segment of data (128 sets per segment with 0% overlap for now), extract the feature vector
def extract_feature_vector(X):
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
    # extract frequency domain features - unused for now
    X_psd = []
    X_peakF = []
    # obtain feature vector by appending all vectors above as one d-dimension feature vector
    X = np.append(X_mean, [X_var, X_max, X_min, X_off, X_mad])
    return standard_scaler.transform([X])

def predict_dance_move(segment):
    X = extract_feature_vector(segment)
    Y = model.predict(X)
    # return model.predict(X).tolist()[0]
    return onehot2str(Y)[0], max(Y[0])

def readLineCR(port):
    rv = ""
    while True:
        ch = port.read().decode()
        rv += ch
        if ch == '\r':
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

#Establish socket connection
#input on console in this format: IP_address Port_number
TCP_IP = sys.argv[1]
TCP_PORT = int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

secret_key = "1234123412341234"  #must be at least 16

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
# port.reset_input_buffer()
# port.reset_output_buffer()
while (handshake_flag == False):
    port.write('H'.encode())
    print("H sent")
    response = port.read()
    if (response.decode() == 'A'):
        print("A received, sending N")
        port.write('N'.encode())
        handshake_flag= True
        port.read()

# port.reset_input_buffer()
# port.reset_output_buffer()
print("connected")

while (data_flag == False):

    print("ENTERING")

    movementData = []
    otherData = []
    for i in range(N): # extract from 0->N-1 = N sets of readings
        data = readLineCR(port).split(',')
        data = [ float(val.strip()) for val in data ]
        movementData.append(data[:10]) # extract acc1[3], acc2[3] and gyro[3] values
        otherData.append(data[10:]) # extract voltage, current, power and cumulativepower

    # Add ML Logic
    # Precondition 1: dataArray has values for acc1[3], acc2[3], gyro[3], voltage[1], current[1], power[1] and energy[1] in that order
    # Precondition 2: dataArray has N sets of readings, where N is the segment size, hence it has dimensions N*13
    danceMove, predictionConfidence = predict_dance_move(movementData)
    if move_state == 2 and not danceMove == "IDLE" and predictionConfidence >= CONFIDENCE_THRESHOLD:
        voltage, current, power, energy = tuple(map(tuple, np.mean(otherData, axis=0)))
        output = "#" + danceMove + "|" + str(round(voltage, 2)) + "|" + str(round(current, 2)) + "|" + str(round(power, 2)) + "|" + str(round(energy, 2)) + "|"
        if danceMove == "logout":
            output = danceMove # with logout command, no other values are sent
        # Send output to server above
        sendToServer(s, output)
        move_state = 1
    elif move_state == 1 and danceMove == "IDLE":
        move_state = 2

    #print("Print array: ")
    #output = "1.0,2.0,3.0,4.0,5.0"
    #output = output.replace(',', '|')
    print(output)
    action, voltage, current, power, cumulativepower = output.split('|')
    print("action: " + action + '\n' + "voltage: " + voltage + '\n' + "current: " + current + '\n' +
          "power: " + power + '\n' + "cumulativepower: " + cumulativepower + '\n')

    data_flag = True
