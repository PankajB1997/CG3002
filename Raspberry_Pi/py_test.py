#!/usr/bin/python3

import time
import serial
import os
import sys
import threading
import socket
import base64

def readLineCR(port):
    rv = ""
    while True:
        ch = port.read().decode()
        rv += ch
        if ch == '\r':
            return rv

def testReadLineCR(port):
    read_flag = 1
    rv = ""
    while (read_flag == 1):
        ch = port.read()
        rv += ch
        if ch == '\r':
            data = rv
            read_flag = 0

#dataArray = [] #128 objects in array, per 20ms
handshake_flag = False
data_flag = False
print("test")
port=serial.Serial("/dev/serial0", baudrate=115200, timeout=3.0)
print("set up")
#port.reset_input_buffer()
#port.reset_output_buffer()
while (handshake_flag == False):
    port.write('H'.encode())
    print("H sent")
    response = port.read()
    if (response.decode() == 'A'):
        print("A received, sending N")
        port.write('N'.encode())
        handshake_flag= True
        port.read()
        
#port.reset_input_buffer()
#port.reset_output_buffer()
print("connected")


while (data_flag == False):
    
    print("ENTERING")
    
    #port.write('A')
    
    #dataArray.clear()
    dataArray = []
    for i in range(64): #print from 0->127 = 128 sets of readings
        data = readLineCR(port)
        #print(str(i) + ": " + data)
        
        #str = str(data)
        #data.replace(',', '\t')        
        dataArray.append(data)
        print(str(i) + ': ' + dataArray[i])
    
    print("Print array: ")
    #print(dataArray)
    sendData = dataArray[63]
    print(sendData)
    output = "1.0,2.0,3.0,4.0,5.0"
    output = output.replace(',', '|')
    print(output)
    action, voltage, current, power, cumulativepower = output.split('|')
    print("action: " + action + '\n' + "voltage: " + voltage + '\n' + "current: " + current + '\n' +
          "power: " + power + '\n' + "cumulativepower: " + cumulativepower + '\n')
    data_flag = True