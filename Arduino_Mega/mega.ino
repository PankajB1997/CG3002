#include<Arduino_FreeRTOS.h>
#include<task.h>
#include<avr/io.h>
#include<semphr.h>
#define STACK_SIZE 200
#include<stdio.h>
#include<stdlib.h>

typedef struct Packet {
  float ID = 0;
  float GYRO_ID = 0;
  float gyro[3];
  float ACC_ID = 1;
  float acc1[3];
  float acc2[3];
  float power;
  float current;
  float voltage;
} Packet;   //Size of packet is 60 (15 of 4 bytes)

char* acc1_x;
char* acc1_y;
char* acc1_z;
char* acc2_x;
char* acc2_y;
char* acc2_z;
char* gyro_x;
char* gyro_y;
char* gyro_z;
char* voltage_c;
char* current_c;
char* power_c;

DataPacket packet;

int test_flag = 0;
int dataReady = 0;
char databuf[3000];

SemaphoreHandle_t taskSemaphore = xSemaphoreCreateMutex();

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);

  handshake();
  collectData();
  sendToPi();
// //xTaskCreate(collectData, "collectData", STACK_SIZE, (void *)NULL, 1, NULL);
// xTaskCreate(collectData, "collectData_c", STACK_SIZE, (void *)NULL, 1, NULL);
// xTaskCreate(sendToPi, "sendToPi", STACK_SIZE, (void *)NULL, 2, NULL);
// vTaskStartScheduler();
}

void loop() {
}

/**
 *  To perform handshalke to ensure that communication between Rpi and Aduino is ready
 */
void handshake() {
  int h_flag = 0;
  int n_flag = 0;

  while (h_flag == 0) {
    if (Serial1.available()) {
      if ((Serial1.read() == 'H')) {
        h_flag = 1;
      }
    }
  }

  while (n_flag == 0) {
    if (Serial1.available()) {
      Serial1.write('A');
      if (Serial1.read() == 'N') {
        Serial.println("Handshake done");
        n_flag = 1;
      }
    }
  }

}
/**
 *  To collect readings from all the sensor and package into one packet
 */
void collectData() {
  pkt.acc1[0] = rand() - rand();
  pkt.acc1[1] = rand() - rand();
  pkt.acc1[2] = rand() - rand();
  pkt.acc2[0] = rand() - rand();
  pkt.acc2[1] = rand() - rand();
  pkt.acc2[2] = rand() - rand();
  pkt.gyro[0] = rand() - rand();
  pkt.gyro[1] = rand() - rand();
  pkt.gyro[2] = rand() - rand();
  pkt.current = rand() - rand();
  pkt.voltage = rand() - rand();
}

/** 
 Change the format
 */
void changeFormat() {
  char charbuf[100];

  acc1_x = dtostrf(pkt.acc1[0], 3, 2, charbuf);
  strcat(databuf, acc1_x);
  strcat(databuf, ",");
  acc1_y = dtostrf(pkt.acc1[1], 3, 2, charbuf);
  strcat(databuf, acc1_y);
  strcat(databuf, ",");
  acc1_z = dtostrf(pkt.acc1[2], 3, 2, charbuf);
  strcat(databuf, acc1_z);
  strcat(databuf, ",");

  acc2_x = dtostrf(pkt.acc2[0], 3, 2, charbuf);
  strcat(databuf, acc2_x);
  strcat(databuf, ",");
  acc2_y = dtostrf(pkt.acc2[1], 3, 2, charbuf);
  strcat(databuf, acc2_y);
  strcat(databuf, ",");
  acc2_z = dtostrf(pkt.acc2[2], 3, 2, charbuf);
  strcat(databuf, acc2_z);
  strcat(databuf, ",");

  gyro_x = dtostrf(pkt.gyro[0], 3, 2, charbuf);
  strcat(databuf, gyro_x);
  strcat(databuf, ",");
  gyro_y = dtostrf(packet.gyro[1], 3, 2, charbuf);
  strcat(databuf, gyro_y);
  strcat(databuf, ",");
  gyro_z = dtostrf(pkt.gyro[2], 3, 2, charbuf);
  strcat(databuf, gyro_z);
  strcat(databuf, ",");

  voltage_c = dtostrf(pkt.voltage, 3, 2, charbuf);
  strcat(databuf, voltage_c);
  strcat(databuf, ",");
  current_c = dtostrf(pkt.current, 3, 2, charbuf);
  strcat(databuf, current_c);
  strcat(databuf, ",");
  power_c = dtostrf(pkt.power, 3, 2, charbuf);
  strcat(databuf, power_c);
  strcat(databuf, ",");

}

void sendToPi() {

  sendData();
  changeFormat();
  Serial.println("Send to pi");
  strcat(databuf, "\r");
  int len = strlen(databuf);

  while (Serial1.available()) {
    Serial.print("Data handshake");
    if (Serial.read() == 'A') {
      dataReady = 1;
      for (int i = 0; i < len; i++) {
        Serial.print(databuf[i]);
        Serial1.write(databuf[i]);
      }
      strcpy(databuf, "");
    }
  }

}


