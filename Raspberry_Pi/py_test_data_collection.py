#!/usr/bin/python3
import time
import serial
import os
import sys
import threading
import socket
import base64
import pickle

N = 128
count = 1
MAX_BUF_LEN = 2800 # bytes
TOTAL_PACKETS = 20
PACKET_SIZE = 64 # number of elements in list = len(array); actual size = sys.getsizeof(array)

danceMove = "turnclap"
dancer = "pankaj"
SAVEPATH = os.path.join("dataset", "RawData", dancer, danceMove + ".txt")

def readLineCR(port):
    rv = ""
    while True:
        ch = port.read().decode()
        rv += ch
        if ch == "\r" or ch == "":
            return rv
        
def testReadLineCR(port):
    read_flag = 1
    rv = ""
    while (read_flag == 1):
        ch = port.read()
        rv += ch
        if ch == "\r":
            data = rv
            read_flag = 0

def comppute_checksum(data):
	data, correct_checksum = data.rsplit("," , 1)
	data = data.split(",")
    for i in range(len(data)):
		cs ^= data[i] #to be changed
	
	return cs

def handle_checksum(cs):
	if (cs):
		print("Packet OK")
		return True
	else:
		print("Packet Bad Checksum ")
		return False

handshake_flag = False
data_flag = False
print("test")
port = serial.Serial("/dev/serial0", baudrate=115200, timeout=3.0)
print("set up")
port.reset_input_buffer()
port.reset_output_buffer()
#port.flushInput()
#port.flushOutput()

while (handshake_flag == False):
    port.write("H".encode())
    print("H sent")
    response = port.read(1)
    time.sleep(0.5)
    if (response.decode() == "A"):
        print("A received, sending N")
        port.write("N".encode())
        time.sleep(0.5)
        handshake_flag= True
        # port.readline()
    else:
        time.sleep(0.5)
    
print("connected")
#port.flush() # waits till all in buffer is written then flush

while (data_flag == False):
    print("ENTERING")
    with open(SAVEPATH, "a") as txtfile:
        for i in range(N): # print from 0->127 = 128 sets of readings
            data = readLineCR(port)
			
			#checksum
			cs = compute_checksum(data)
			data_flag = handle_checksum(cs)
			
            data = data.split(",")[0:9] # extract acc1[3], acc2[3] and gyro[3] values
            data = [ val.strip() for val in data ]
            output = "\t".join(data) + "\n"
            # output = output.replace(' ', '').replace(',', '\t').replace('[', '').replace(']', '')
            txtfile.write(output)
            print(str(count) + ". " + output)
            count += 1
            #else:
                #time.sleep(1)
    # data_flag = True
