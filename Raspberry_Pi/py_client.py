import os
import sys
import time
import threading
import serial
import socket
import base64
from Crypto import Random
from Crypto.Cipher import  AES
#from Crypto.Util.Padding import pad

BLOCK_SIZE = 32 #AES.block_size

#references
#https://docs.python.org/3/howto/sockets.html
#https://gist.github.com/Integralist/3f004c3594bbf8431c15ed6db15809ae
#https://wiki.python.org/moin/TcpCommunication

def inputDummyData():
    #'#action | voltage | current | power | cumulativepower|'
    action = str(input('Manually enter data: '))
    data = '#' + action + '|2.0|1.5|5.6|10.10|'
    return data
	
def inputData():
    #'#action | voltage | current | power | cumulativepower|'
    action = str(input('Manually enter data: '))
    data = '#' + action + '|2.0|1.5|5.6|10.10|'
    return data

def encryption(data, secret_key):
	length = BLOCK_SIZE-(len(data)%BLOCK_SIZE)
	msg = data+((chr(length))*(length))
	print(msg)
	
	iv = Random.new().read(AES.block_size)
	cipher = AES.new(secret_key, AES.MODE_CBC, iv)
	encoded = base64.b64encode(iv + cipher.encrypt(msg))

	return encoded

#def sendToServer():
    

TCP_IP = sys.argv[1]
TCP_PORT = int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

secret_key = "1234123412341234"  #must be at least 16

try:
    #connect to server
    if (s):
        print('connected to server')

        while (True):
            data = inputData()
            encryptedData = encryption(data, secret_key)
            #send to socket
            s.send(encryptedData)
            print('sent')
            
    else:
        print('not connected')

    
    

    #connect to arduino
    #handshaking with arduino

except KeyboardInterrupt:
    s.close()
    sys.exit(1)
