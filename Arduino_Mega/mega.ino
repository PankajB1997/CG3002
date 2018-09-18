#include<Arduino_FreeRTOS.h>
#include<task.h>
#include<avr/io.h>
#include<semphr.h>
#define STACK_SIZE 200
#include<stdio.h>
#include<stdlib.h>


typedef struct DataPacket {
        int16_t GYRO_ID=0;
        int16_t gyro[3];
        int16_t ACC_ID=1;
        int16_t acc1[3];
        int16_t acc2[3];
        int16_t power;
        int16_t current;
        int16_t voltage;
        }DataPacket;   //Size of packet is 28 (14 of 2 bytes)


typedef struct Packet{
        float GYRO_ID=0;
        float gyro[3];
        float ACC_ID=1;
        float acc1[3];
        float acc2[3];
        float power;
        float current;
        float voltage;
        }Packet;   //Size of packet is 28 (14 of 2 bytes)

 char* acc1_x; char* acc1_y; char* acc1_z;
 char* acc2_x; char* acc2_y; char* acc2_z;
 char* gyro_x; char* gyro_y; char* gyro_z;
char* voltage_c; char* current_c; char* power_c;

        Packet pkt;
        DataPacket packet;
        int dataReady=0;
        char databuf[3000];

        SemaphoreHandle_t taskSemaphore=xSemaphoreCreateMutex();

        void setup()
        {
        // put your setup code here, to run once
        Serial.begin(115200);
        Serial1.begin(115200);
        handshake();
       // collectData();
       // sendToPi(
  xTaskCreate(collectData, "collectData", STACK_SIZE, (void *)NULL, 1, NULL);
  xTaskCreate(sendToPi, "sendToPi", STACK_SIZE, (void *)NULL, 2, NULL);
  vTaskStartScheduler();

        }

        void loop(){}

/**
 *  To perform handshalke to ensure that communication between Rpi and Aduino is ready
 */
        void handshake()
        {
        int h_flag=0;
        int n_flag=0;


        while(h_flag==0){
        if(Serial1.available()){
        if((Serial1.read()=='H')){
        h_flag=1;
        }
        }
        }

        while(n_flag==0){
        if(Serial1.available()){
        Serial1.write('A');
        if(Serial1.read()=='N'){
        Serial.println("Handshake done");
        n_flag=1;
        }
       }
     }

  }
/**
 *  To collect readings from all the sensor and package into one packet
 */
void collectData(void *p)
{


  static TickType_t xLastWakeTime = xTaskGetTickCount();
  while (1)
  {
    if (xSemaphoreTake(taskSemaphore, (TickType_t)portMAX_DELAY) == pdTRUE)
    {
        packet.acc1[0]=rand()-rand();
        packet.acc1[1]=rand()-rand();
        packet.acc1[2]=rand()-rand();
        packet.acc2[0]=rand()-rand();
        packet.acc2[1]=rand()-rand();
        packet.acc2[2]=rand()-rand();
        packet.gyro[0]=rand()-rand();
        packet.gyro[1]=rand()-rand();
        packet.gyro[2]=rand()-rand();
        packet.current=rand()-rand();
        packet.voltage=rand()-rand();
        Serial.println("collect data");
        xSemaphoreGive(taskSemaphore);
        }
   vTaskDelayUntil(&xLastWakeTime, 200);
  }
 }


void collectdata(void *p)
{
  static TickType_t xLastWakeTime = xTaskGetTickCount();
  while (1)
  {
    if (xSemaphoreTake(taskSemaphore, (TickType_t)portMAX_DELAY) == pdTRUE)
    {
        packet.acc1[0]=rand()-rand();
        packet.acc1[1]=rand()-rand();
        packet.acc1[2]=rand()-rand();
        packet.acc2[0]=rand()-rand();
        packet.acc2[1]=rand()-rand();
        packet.acc2[2]=rand()-rand();
        packet.gyro[0]=rand()-rand();
        packet.gyro[1]=rand()-rand();
        packet.gyro[2]=rand()-rand();
        packet.current=rand()-rand();
        packet.voltage=rand()-rand();
        Serial.println("collect data");
        xSemaphoreGive(taskSemaphore);
        }
   vTaskDelayUntil(&xLastWakeTime, 200);
  }
 }

/** 
 Change the format
 */
void changeFormat(){
char charbuf[100] ;
//
//  acc1_x = dtostrf( packet.acc1[0],3,2,charbuf);
//  strcat(databuf, acc1_x); 
//  strcat(databuf, ",");
  acc1_y = dtostrf(packet.acc1[1],3,2,charbuf);
  strcat(databuf, acc1_y); 
  strcat(databuf, ",");
  acc1_z = dtostrf(packet.acc1[2],3,2,charbuf);
  strcat(databuf, acc1_z); 
  strcat(databuf, ",");

  acc2_x = dtostrf(packet.acc2[0],3,2,charbuf);
  strcat(databuf, acc2_x); 
  strcat(databuf, ",");
  acc2_y = dtostrf(packet.acc2[1],3,2,charbuf);
  strcat(databuf, acc2_y); 
  strcat(databuf, ",");
  acc2_z = dtostrf(packet.acc2[2],3,2,charbuf);
  strcat(databuf, acc2_z); 
  strcat(databuf, ",");

  gyro_x = dtostrf( packet.gyro[0],3,2,charbuf);
  strcat(databuf, gyro_x); 
  strcat(databuf, ",");
  gyro_y = dtostrf( packet.gyro[1],3,2,charbuf);
  strcat(databuf, gyro_y); 
  strcat(databuf, ",");
  gyro_z = dtostrf( packet.gyro[2],3,2,charbuf);
  strcat(databuf, gyro_z); 
  strcat(databuf, ",");

  voltage_c = dtostrf( packet.voltage,3,2,charbuf);
  strcat(databuf,  voltage_c); 
  strcat(databuf, ",");
  current_c = dtostrf( packet.current,3,2,charbuf);
  strcat(databuf, current_c); 
  strcat(databuf, ",");
  power_c = dtostrf( packet.power,3,2,charbuf);
  strcat(databuf, power_c ); 
  strcat(databuf, ",");


  
  
}

 
/**
 *  To transmit the data Rpi3
 */
void sendToPi(void *p){

 static TickType_t xLastWakeTime = xTaskGetTickCount();
  while (1)
  {
    if (xSemaphoreTake(taskSemaphore, (TickType_t)portMAX_DELAY) == pdTRUE)
    { 
       Serial1.write('D') ;
     // while (Serial1.read == 'A' || Serial1.available()) {  
       sendData();
  //}

    xSemaphoreGive(taskSemaphore);
        }
   vTaskDelayUntil(&xLastWakeTime, 200);
  }
 }



/**
 *  To transmit the packet to Rpi3
 */

 void sendData() {
//       // int16_t pktBuf[15]; //size of buf is 30 (15 of 2 bytes)
//        serialize(pktBuf);
//
//        //Serial.println("Sending now");
//        for(int i=0;i<15;i++){
//        Serial.print(pktBuf[i]);Serial.print(" ");
//        Serial1.print(pktBuf[i]);
//        }
changeFormat();
int len = strlen(databuf);
        Serial.println("Send to pi");
        strcat(databuf, "\r");
                for(int i=0;i<len;i++){
        Serial.print(databuf[i]);Serial.print(" ");
        Serial1.print(databuf[i]);
                }
        strcpy(databuf, "");

 
}
/**
 *  To serialize the packet while ensuring data integerity via checksum
 */


 void serialize(int16_t *pktBuf){
        int16_t checksum=0;
        memcpy(pktBuf,&packet,(size_t)sizeof(packet));

        for(int i=0;i< 14;i++){
        checksum^=pktBuf[i];
        }
        pktBuf[14]=checksum;
        Serial.print("checksum");
        Serial.println(pktBuf[14]);
        }



