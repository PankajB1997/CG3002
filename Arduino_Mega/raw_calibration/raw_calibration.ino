#include <MPU6050.h>
#include <I2Cdev.h>
#include <ADXL345.h>
#include <Wire.h>

#define DEVICE_A_ACCEL (0x53)    //first ADXL345 device address
#define DEVICE_B_ACCEL (0x1D)    //second ADXL345 device address
#define DEVICE_C_GYRO (0x68) // MPU6050 address
#define TO_READ (6)        //num of bytes we are going to read each time

//Offsets for calibration of sensors
#define x1_accelRaw -55.18 //0x53 (RIGHT HAND)
#define y1_accelRaw -252.07 //0x53 (RIGHT HAND)
#define z1_accelRaw 41.42 //0x53 (RIGHT HAND)
#define x2_accelRaw 51.97 //0x1D (LEFT HAND)
#define y2_accelRaw -256.83 //0x1D (LEFT HAND)
#define z2_accelRaw 16.73//0x1D (LEFT HAND)
#define x1_gyroRaw 296.02//0x68 (LEFT HAND)
#define y1_gyroRaw -74.04//0x68 (LEFT HAND)
#define z1_gyroRaw 153.84//0x68 (LEFT HAND)

ADXL345 sensorA = ADXL345(DEVICE_A_ACCEL);
ADXL345 sensorB = ADXL345(DEVICE_B_ACCEL);
MPU6050 sensorC = MPU6050(DEVICE_C_GYRO);

//16 bit integer values for raw data of accelerometers
int16_t xa_raw, ya_raw, za_raw, xb_raw, yb_raw, zb_raw;

//16 bit integer values for gyroscope readings
int16_t xg_raw, yg_raw, zg_raw;

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

  float average_x = 0.0;
  float average_y = 0.0;
  float average_z = 0.0;

  for(int i = 0; i<100; i++) {
    sensorA.getAcceleration(&xa_raw, &ya_raw, &za_raw);
    sensorB.getAcceleration(&xb_raw, &yb_raw, &zb_raw);
    sensorC.getRotation(&xg_raw, &yg_raw, &zg_raw);

    average_x += xa_raw;
    average_y += ya_raw;
    average_z += za_raw;
    
    Serial.print("offset for Sensor A:\t");
    Serial.print(xa_raw); Serial.print("\t");
    Serial.print(ya_raw); Serial.print("\t");
    Serial.println(za_raw);
    
    Serial.print("accel for Sensor B:\t");
    Serial.print(xb_raw); Serial.print("\t");
    Serial.print(yb_raw); Serial.print("\t");
    Serial.println(zb_raw);
    
    Serial.print("rotation for Sensor C:\t");
    Serial.print(xg_raw); Serial.print("\t");
    Serial.print(yg_raw); Serial.print("\t");
    Serial.println(zg_raw);

    delay(20);
  }
  Serial.println(average_x);
  Serial.println(average_y);
  Serial.println(average_z);
  average_x = average_x / 100;
  average_y = average_y / 100;
  average_z = average_z / 100;

   Serial.print("Offset values = ");
   Serial.print(average_x); Serial.print("\t");
   Serial.print(average_y); Serial.print("\t");
   Serial.println(average_z);
}


void loop()
{

}
