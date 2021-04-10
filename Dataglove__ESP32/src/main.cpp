// DEPENDENCIES
#include <Arduino.h>
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif


// FUNCTION INITIALIZATION
void dmpDataReady();
void getFlexOffsets_median();
void getFlexOffsets_mean();
void convertUnits();
void generateDataString();
void sendData();


// DEFINITIONS
#define G               9.8
#define FLEX_1          12
#define FLEX_2          27
#define FLEX_3          25
#define FLEX_4          32
#define FLEX_5          34
#define INTERRUPT_PIN   23
#define DIV             ","
#define MOVING_AVG_BIN  5

// FLEX SENSOR GLOBAL VARIABLES
int FLEX_PINS[] = {FLEX_1, FLEX_2, FLEX_3, FLEX_4, FLEX_5};
int flex_data[] = {0, 0, 0, 0, 0};
int calibrated_values[] = {2970, 2870, 2929, 3191, 3094};

// OUTPUT BUFFER AND VARIABLE
String data;
char buffer[2056];

// MPU6050 VARIABLES
MPU6050 mpu;
bool blinkState = false;
bool dmpReady = false;
uint8_t mpuIntStatus;
uint8_t devStatus;
uint16_t packetSize;
uint16_t fifoCount;
uint8_t fifoBuffer[64];
volatile bool mpuInterrupt = false;

// MPU6050 DATA CLASSES AND VARIABLES
Quaternion q;
VectorInt16 aa;
VectorInt16 gyr;
VectorInt16 aaReal;
VectorInt16 aaWorld;
VectorFloat gravity;
float Gyr[3];
float Acc[3];
float AccReal[3];
float AccWorld[3];

float current_time, future_time, interval = 150;



//-------------------------- MAIN ROUTINE --------------------------//



void setup()
{
  // SET PINMODE OF ON_BOARD_LED, MPU6050_INTERRUPT_PIN
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH); // ON_BOARD_LED is LOW till initialization
  pinMode(INTERRUPT_PIN, INPUT);

  // INITIALIZE I2C
  #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    Wire.begin();
    Wire.setClock(400000);
  #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
    Fastwire::setup(400, true);
  #endif

  // INITIALIZE SERIAL 
  Serial.begin(230400); // BAUD_RATE = 115200
  while (!Serial);    // wait until serial is available

  // INITIALIZE MPU6050
  Serial.println(F("Initializing I2C devices..."));
  mpu.initialize();
  
  // TEST CONNECTION
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
  Serial.println(F("\nSend any character to begin DMP programming and demo: "));
  while (Serial.available() && Serial.read());

  // INITIALIZE DIGITAL MOTION PROCESSOR (DMP) OF MPU6050
  Serial.println(F("Initializing DMP..."));
  devStatus = mpu.dmpInitialize();

  // SET ACC, GYR OFFSETS << THESE VALUES SHOULD BE SET AFTER CALIBRATION FOR EACH SENSOR MODULES >>
  mpu.setXAccelOffset(123);
  mpu.setYAccelOffset(-1656);
  mpu.setZAccelOffset(1394);
  mpu.setXGyroOffset(152);
  mpu.setYGyroOffset(-8);
  mpu.setZGyroOffset(24);

  // CHECK DMP
  if (devStatus == 0) {
    mpu.setDMPEnabled(true);
    Serial.print(digitalPinToInterrupt(INTERRUPT_PIN));
    Serial.println(F(")..."));
    // DEFINE INTERRUPT PIN, FUNCTION AND MODE
    attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
    mpuIntStatus = mpu.getIntStatus();
    Serial.println(F("DMP ready! Waiting for first interrupt..."));
    dmpReady = true;
    packetSize = mpu.dmpGetFIFOPacketSize();
  } 
  else 
  {
    Serial.print(F("DMP Initialization failed (code "));
    Serial.print(devStatus);
    Serial.println(F(")"));
  }

  // SUCCESSFULL INITIALIZATION INDICATOR ~ BLINK ONCE
  digitalWrite(LED_BUILTIN, LOW);
  delay(500);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(1000);

  // INITIATE FLEX SENSOR OFFSET CALCULATION << THESE VALUES SHOULD BE SET AFTER CALIBRATION FOR EACH SENSOR MODULES >>
  // digitalWrite(LED_BUILTIN, HIGH);
  // getFlexOffsets_median();

  // SUCCESSFULL FLEX CALIBRATION INDICATOR ~ BLINK TWICE
  // digitalWrite(LED_BUILTIN, LOW);
  // delay(500);
  // digitalWrite(LED_BUILTIN, HIGH);
  // delay(500);
  // digitalWrite(LED_BUILTIN, LOW);
  // delay(500);
  // digitalWrite(LED_BUILTIN, HIGH);
}

void loop() {
  // EXIT THE LOOP IF DMP IS NOT FUNCTIONAL
  if (!dmpReady) return;
  // POLL TILL (MPU INTERRUPT IS NOT TRIGGERED) AND (DMP FIFO IS NOT FILLED)
  while (!mpuInterrupt && fifoCount < packetSize) {
    // IF (MPU INTERRUPT IS TRIGGERED) AND (DMP FIFO IS NOT FILLED) UPDATE FIFO COUNTER
    if (mpuInterrupt && fifoCount < packetSize) {
      fifoCount = mpu.getFIFOCount();
    }
  }
  // SET MPU INTERRUPT VARIABLE TO LOW
  mpuInterrupt = false;
  mpuIntStatus = mpu.getIntStatus();
  // UPDATE FIFO COUNTER
  fifoCount = mpu.getFIFOCount();
  if ((mpuIntStatus & _BV(MPU6050_INTERRUPT_FIFO_OFLOW_BIT)) || fifoCount >= 1024) 
  // CHECK IF FIFO IS OVERFLOWED OR MPU INTERNAL INTERRUPT IS TRIGGERED
  {
    // RESET FIFO AND UPDATE FIFO COUNTER
    mpu.resetFIFO();
    fifoCount = mpu.getFIFOCount();
  }
  // CHECK IF FIFO IS OVERFLOWED
  else if (mpuIntStatus & _BV(MPU6050_INTERRUPT_DMP_INT_BIT)) 
  {
    // POLL UNTILL DMP FIFO IS FILLED
    while (fifoCount < packetSize) fifoCount = mpu.getFIFOCount();
    
    // READ DMP FIFO DATA AND WRITE FIFO BUFFER
    mpu.getFIFOBytes(fifoBuffer, packetSize);
    fifoCount -= packetSize;

    // READ MPU6050 DMP DATA FROM FIFO BUFFER
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetAccel(&aa, fifoBuffer);
    mpu.dmpGetGyro(&gyr, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
    mpu.dmpGetLinearAccelInWorld(&aaWorld, &aaReal, &q);

    // READ FLEX DATA
    for (int r = 0; r < 5; r++)
    {
      flex_data[r] = analogRead(FLEX_PINS[r]) - calibrated_values[r];
    }

    // ACC, GYR UNIT CONVERTION
    convertUnits();

    // SEND SENSOR READINGS VIA SERIAL -- EITHER METHOD WORKS
    // METHOD 1
    generateDataString();
    Serial.println(data);
    // METHOD 2 
    // sendData();
  }

  // CLEAR MPU FIFO
  mpu.resetFIFO();
}




//-------------------------- FUNCTION DEFINITIONS --------------------------//



// MPU INTERRUPT DETECTION ROUTINE
void dmpDataReady()
{
  mpuInterrupt = true;
}


// FLEX OFFSET CALCULATION ROUTINE -- MEDIAN
void getFlexOffsets_median()
{
  int flex[5][100];
  for (int i = 0; i < 5; i++)
  {
    for (int j = 0; j < 100; j++)
    {
      flex[i][j] = 0;
    }
  }
  for (int i = 0; i < 100; i++)
  {
    for (int j = 0; j < 5; j++)
    {
      flex[j][i] = analogRead(FLEX_PINS[j]);
      for (int k = 0; k < i; k++)
      {
        for (int l = 1; l < 100; l++)
        {
          if (flex[j][l] > flex[j][l - 1])
          {
            int temp = flex[j][l];
            flex[j][l] = flex[j][l - 1];
            flex[j][l - 1] = temp;
          }
        }
      }
    }
    delay(70);
  }
  for (int i = 0; i < 5; i++)
  {
    calibrated_values[i] = flex[i][50];
  }
}


// FLEX OFFSET CALCULATION ROUTINE -- MEAN
void getFlexOffsets_mean()
  {
    float sum[] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 700; i++)
    {
      for (int i = 0; i < 5; i++)
      {
        sum[i] = sum[i] + analogRead(FLEX_PINS[i]);
      }
      delay(10);
    }
    for (int i = 0; i < 5; i++)
    {
      calibrated_values[i] = sum[i] / 700;
    }
  }


// ACC, GYR UNIT CONVERSION ROUTINE
void convertUnits()
{
  /*
The accelerometer is set at the sensitivity of +/-2g while the value is ranged within +/-16384.
The gyroscope is set at the sensitivity of 250 deg which can be translated as 1 degree = 131 measurement units.

ACC (in m.s^-2) = ACC (DMP measurement) / 8192 * G 
GYR (in deg.s^-1) = GYR (DMP measurement) / 131
*/

  Acc[0] = (float)aa.x / 8192 * G;
  Acc[1] = (float)aa.y / 8192 * G;
  Acc[2] = (float)aa.z / 8192 * G;
  AccReal[0] = (float)aaReal.x / 8192 * G;
  AccReal[1] = (float)aaReal.y / 8192 * G;
  AccReal[2] = (float)aaReal.z / 8192 * G;
  AccWorld[0] = (float)aaWorld.x / 8192 * G;
  AccWorld[1] = (float)aaWorld.y / 8192 * G;
  AccWorld[2] = (float)aaWorld.z / 8192 * G;
  Gyr[0] = (float)gyr.x / 131;
  Gyr[1] = (float)gyr.y / 131;
  Gyr[2] = (float)gyr.z / 131;
}


void generateDataString()
{
  sprintf(buffer, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", flex_data[0], flex_data[1], flex_data[2], flex_data[3], flex_data[4], q.w, q.x, q.y, q.z, Gyr[0], Gyr[1], Gyr[2], Acc[0], Acc[1], Acc[2], AccReal[0], AccReal[1], AccReal[2], AccWorld[0], AccWorld[1], AccWorld[2]);
  data = buffer;
}


void sendData()
{
  // SEND FLEX VALUES
  for (int r = 0; r < 5; r++)
  {
    Serial.print(flex_data[r]);
    Serial.print(DIV);
  }

  // SEND QUATERNIONS
  Serial.print(q.w, 6);
  Serial.print(DIV);
  Serial.print(q.x, 6);
  Serial.print(DIV);
  Serial.print(q.y, 6);
  Serial.print(DIV);
  Serial.print(q.z, 6);

  Serial.print(DIV);

  // SEND GYRO VALUES
  Serial.print(Gyr[0], 6);
  Serial.print(DIV);
  Serial.print(Gyr[1], 6);
  Serial.print(DIV);
  Serial.print(Gyr[2], 6);

  Serial.print(DIV);

  // SEND GYRO VALUES [UNCONVERTED]
  // Serial.print(gyr.x);
  // Serial.print(DIV);
  // Serial.print(gyr.y);
  // Serial.print(DIV);
  // Serial.print(gyr.z);

  // Serial.print(DIV);

  // SEND ACC VALUES
  Serial.print(Acc[0], 6);
  Serial.print(DIV);
  Serial.print(Acc[1], 6);
  Serial.print(DIV);
  Serial.print(Acc[2], 6);

  Serial.print(DIV);

  // SEND BODY_ACC VALUES
  Serial.print(AccReal[0], 6);
  Serial.print(DIV);
  Serial.print(AccReal[1], 6);
  Serial.print(DIV);
  Serial.print(AccReal[2], 6);

  Serial.print(DIV);

  // SEND WORLD_ACC VALUES
  Serial.print(AccWorld[0], 6);
  Serial.print(DIV);
  Serial.print(AccWorld[1], 6);
  Serial.print(DIV);
  Serial.print(AccWorld[2], 6);

  Serial.print('\n');
}
