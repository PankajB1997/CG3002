#include <MPU6050.h>
#include <I2Cdev.h>
#include <ADXL345.h>
#include <Wire.h>

#define DEVICE_A_ACCEL (0x53)    //first ADXL345 device address
#define DEVICE_B_ACCEL (0x1D)    //second ADXL345 device address
#define DEVICE_C_GYRO (0x68) // MPU6050 address
#define TO_READ (6)        //num of bytes we are going to read each time
#define voltageDividerPin 0
#define currentSensorPin 1
#define RS 0.1
#define RL 10000

ADXL345 sensorA = ADXL345(DEVICE_A_ACCEL);
ADXL345 sensorB = ADXL345(DEVICE_B_ACCEL);
MPU6050 sensorC = MPU6050(DEVICE_C_GYRO);

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
 
//declaring variable to store value of volt and amps
    float vOut;
    float voltageReading;
    float currentReading;

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
void calibrateScale();

void setup()
{
  Wire.begin();        // join i2c bus (address optional for master)
  Serial.begin(115200);  // start serial for output
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
}
void loop()
{  
  // Getting raw values at 50 Hz frequency by setting 20 ms delay
  delay(20);

  // Read values from different sensors
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
