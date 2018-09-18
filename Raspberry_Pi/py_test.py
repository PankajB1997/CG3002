#!/usr/bin/python3

import time
import serial

def readlineCR(port):
    rv=""
    while True:
        ch=port.read()
        rv+=ch
        if ch=='\r' or ch=='':
            return rv

handshake_flag = False
port=serial.Serial("/dev/serial0", baudrate=115200, timeout=3.0)
print("set up")
while (handshake_flag == False):
    port.write('H')
    time.sleep(0.5)
    response = port.read(1)
    if (response == 'A'):
        port.write('N')
        handshake_flag= True
        
print("connected")

while 1:
    data = port.readline()
    print(data)