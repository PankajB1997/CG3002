#include<Arduino_FreeRTOS.h>
#include<task.h>
#include<avr/io.h>
#include<semphr.h>
#define STACK_SIZE 200
#include<stdio.h>
#include<stdlib.h>

#include <MPU6050.h>
#include <I2Cdev.h>
#include <ADXL345.h>
#include <Wire.h>

#define DEVICE_A_ACCEL (0x53)    //first ADXL345 device address
#define DEVICE_B_ACCEL (0x1D)    //second ADXL345 device address
#define DEVICE_C_GYRO (0x68) // MPU6050 address
#define TO_READ (6)        //num of bytes we are going to read each time
#define voltageDividerPin A0 // Arduino Analog 0 pin
#define currentSensorPin A1  // Arduino Analog 1 pin
#define RS 0.1
#define RL 10000
#define PowerBufferSize 100

ADXL345 sensorA = ADXL345(DEVICE_A_ACCEL);
ADXL345 sensorB = ADXL345(DEVICE_B_ACCEL);
MPU6050 sensorC = MPU6050(DEVICE_C_GYRO);

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

        //determining scale factor based on range set
//Since range in +-2g, range is 4mg/LSB.
//value of 1023 is used because ADC is 10 bit
int rangeAccel = (2-(-2));
const float scaleFactorAccel = rangeAccel/1023.0;

/*
 * +-250degrees/second already set initialize() function
 * value of 65535 is used due to 16 bit ADC
 */
//determining the scale factor for gyroscope
int rangeGyro = 250-(-250);
const float scaleFactorGyro = rangeGyro/65535.0;

//16 bit integer values for raw data of accelerometers
int16_t xa_raw, ya_raw, za_raw, xb_raw, yb_raw, zb_raw;

//16 bit integer values for offset data of accelerometers
int16_t xa_offset, ya_offset, za_offset, xb_offset, yb_offset, zb_offset;

//Float values for scaled factors of accelerometers
float xa, ya, za, xb, yb, zb;

//16 bit integer values for gyroscope readings
int16_t xg_raw, yg_raw, zg_raw;

//16 bit integer values for offset data of gyroscope
int16_t xg_offset, yg_offset, zg_offset;

//Float values for scaled values of gyroscopes
float xg, yg, zg;

//Function prototypes
float remapVoltage(int);
void calibrateSensors();
void getScaledReadings();
void printSensorReadings();

void setup()
{
// put your setup code here, to run once
Serial.begin(115200);
Serial1.begin(115200);
handshake();
// collectData();
// sendToPi();

// Initializing sensors
sensorA.initialize();
sensorB.initialize();
sensorC.initialize();

// Testing connection by reading device ID of each sensor
// Returns false if deviceID not found, Returns true if deviceID is found
Serial.println(sensorA.testConnection() ? "Sensor A connected successfully" : "Sensor A failed to connect");
Serial.println(sensorB.testConnection() ? "Sensor B connected successfully" : "Sensor B failed to connect");
Serial.println(sensorC.testConnection() ? "Sensor C connected successfully" : "Sensor C failed to connect");

calibrateSensors();

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


/*------------------------------------------------------------------------------HARDWARE SIDE CODE FUNCTIONS -----------------------------------------------------------*/
float getSum(float array[]){
  float result = 0;
  for(int i = 0; i < PowerBufferSize; i++)
    result += array[i];
  }
  return result;
}

void printPower(){
    //Measure and display voltage measured from voltage divider
    float voltage = analogRead(voltageDividerPin);
    //voltage divider halfs the voltage
    //times 2 here to compensate
    voltage = remapVoltage(voltage) * 2;

    //Measure voltage out from current sensor to calculate current
    float currentVoltage = analogRead(currentSensorPin);
    float current = remapVoltage(currentVoltage);

    //formula given by hookup guide
    current = (currentVoltage * 1000) / (RS * RL);

    static float powerValues[PowerBufferSize];
    static int count = 0;
    static boolean arrayFilled = false;
    static startTime = -1;

    if(startTime == -1){
      startTime = millis();
    }

    count = (count + 1) % PowerBufferSize;

    int size;

    if(arrayFilled){
      size = PowerBufferSize;
    }else{
      size = count;
    }

    float sum = getSum(powerValues);
    float averagePower = sum / size;

    float hoursPassed = (millis()-startTime) / 1000.0 / 60.0;
    float energy = hoursPassed * averagePower;

    Serial.print("current reading: ");
    Serial.println(currentReading, 9);

    Serial.print("voltage reading");
    Serial.println(voltageReading, 9);

    Serial.print("Power reading");
    Serial.println(averagePower, 9);

    Serial.print("Energy reading");
    Serial.println(energy, 9);
}

void loop()
{
  // Getting raw values at 50 Hz frequency by setting 20 ms delay
  delay(20);

  // Read values from different sensors
  getScaledReadings();
  printSensorReadings();

  //Measure and display voltage measured from voltage divider
  voltageReading = analogRead(voltageDividerPin);
  voltageReading = remapVoltage(voltageReading);
  Serial.print("voltage reading");
  Serial.println(voltageReading, 9);

  //Measure voltage out from current sensor to calculate current
  vOut = analogRead(currentSensorPin);
  vOut = remapVoltage(vOut);
  currentReading = (vOut * 1000) / (RS * RL);
  Serial.print("current reading: ");
  Serial.println(currentReading, 9);

}

float remapVoltage(int volt) {
  float analogToDigital;
  analogToDigital = (5.0/1023) * volt;  
  return analogToDigital;
}

void getScaledReadings() {
  sensorA.getAcceleration(&xa_raw, &ya_raw, &za_raw);
  xa = (xa_raw + xa_offset)*scaleFactorAccel;
  ya = (ya_raw + ya_offset)*scaleFactorAccel;
  za = (za_raw + za_offset)*scaleFactorAccel;

  sensorB.getAcceleration(&xb_raw, &yb_raw, &zb_raw);
  xb = (xb_raw + xb_offset)*scaleFactorAccel;
  yb = (yb_raw + yb_offset)*scaleFactorAccel;
  zb = (zb_raw + zb_offset)*scaleFactorAccel;

  sensorC.getRotation(&xg_raw, &yg_raw, &zg_raw);
  xg = (xg_raw + xg_offset)*scaleFactorGyro;
  yg = (yg_raw + yg_offset)*scaleFactorGyro;
  zg = (zg_raw + zg_offset)*scaleFactorGyro;
}

void printSensorReadings() {
  //Display values for different sensors
  Serial.print("accel for Sensor A:\t");
  Serial.print(xa); Serial.print("\t");
  Serial.print(ya); Serial.print("\t");
  Serial.println(za);

  Serial.print("accel for Sensor B:\t");
  Serial.print(xb); Serial.print("\t");
  Serial.print(yb); Serial.print("\t");
  Serial.println(zb);

  Serial.print("rotation for Sensor C:\t");
  Serial.print(xg); Serial.print("\t");
  Serial.print(yg); Serial.print("\t");
  Serial.println(zg);

}

float remapVoltage(int volt) {
  return map(volt, 0, 1024, 0f, 5f);

}

/*
 * Purpose of adding 255 is to account for downward default acceleration in Z axis to be 1g
 */
void calibrateSensors() {
  //Setting range of ADXL345
  sensorA.setRange(ADXL345_RANGE_2G);
  sensorB.setRange(ADXL345_RANGE_2G);

  sensorA.getAcceleration(&xa_raw, &ya_raw, &za_raw);
  sensorB.getAcceleration(&xb_raw, &yb_raw, &zb_raw);
  sensorC.getRotation(&xg_raw, &yg_raw, &zg_raw);

  xa_offset = -xa_raw;
  ya_offset = -ya_raw;
  za_offset = -za_raw+255;

  xb_offset = -xb_raw;
  yb_offset = -yb_raw;
  zb_offset = -zb_raw+255;

  xg_offset = -xg_raw;
  yg_offset = -yg_raw;
  zg_offset = -zg_raw;
}
