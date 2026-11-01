#include <Arduino.h>
#include <SimpleFOC.h>
#include <Preferences.h>// 该库用于在ESP32中进行键值对数据的存储和读取，实现数据的持久化
#include "FUTABA_SBUS.h"
#include "LeggedRobot.h"
#include "MotorControl.h"

//设置工作模式

#define  Model_Selection      0   // 0://长大臂机型   1:短大臂机型
#define  SelectTelecontrol    1   // 0:SWA复位遥控    1:SWA不复位遥控
#define  AdjusParameter       0   // 0:平衡速度航向横滚调参   1:顶球


int KEY_state = 1;
// 定义按键引脚
const int buttonPin = 0;  // 使用IO0引脚（GPIO0）

// commander communication instance
Commander command = Commander(DEBUG);




ConfigurationParameter_t ConfigurationParameter;


int robot_mode = 0;//0:移动底盘模式  1：电机模式
// 融合后的速度
float fused_velocity_x = 0.0;

float Motor_KP = 0;
float Motor_KD = 0;

// Zero-drift yaw control state. Only the residual Z-axis bias is learned, so
// the proven X/Y gyro and pitch-balance calibration remain unchanged.
float YawRateBias = 0.0f;
float YawCorrectedRate = 0.0f;
float YawHoldErrorDeg = 0.0f;
bool YawHoldActive = false;
bool YawHoldResetRequested = true;
uint32_t YawBiasSamples = 0;

float TargetLegLength = 0;//高度调参切换
float LegLength = ConfigurationParameter.DefaultHeight;  //腿长
float BarycenterX = 0;    //质心X
float BodyPitching = 0;   //俯仰
float BodyRoll = 0;       //横滚
float MovementSpeed = 0;  //移动速度
float BodyTurn = 0;       //转向
float SlideStep = 0;      //滑步
float BodyX = 0;          //X位置 控制器输出
int RobotTumble = 0;      //机器摔倒

RX_MIT_Data_t RX_MIT_Data;
body_t body;
ROS_body_t ROS_body;
ROS_body_t1 ROS_body1;
RobotDiagnostic_t RobotDiagnostic = {};

uint8_t diagnostic_boot_stage = BOOT_STAGE_STARTING;
uint32_t diagnostic_boot_fault_bits = 0;


unsigned long RX_error = 0;
unsigned int RX_error_hz = 0;


ComWheelLegged_t RosComWheelLegged;
int RosMessageSign = 0;

constexpr uint16_t ROS_SET_JUMP_PROFILE = 0x0002U;

// Default jump remains the original high jump for remote control and keyboard
// /cmd_jump. Low-step values are used only while ROS sends /cmd_jump_profile.
constexpr float JumpDefaultMinHeight = 0.143331f;
constexpr float JumpDefaultMaxHeight = 0.37f;
constexpr float JumpDefaultSquattingSpeed = 0.30f;
constexpr float JumpDefaultAscendingVelocity = 25.0f;
constexpr float JumpDefaultKpPrepare = 20.0f;
constexpr float JumpDefaultKpLanding = 15.0f;
constexpr float JumpDefaultKpTakeoff = 50.0f;

constexpr float JumpLowMinHeight = 0.18f;
constexpr float JumpLowMaxHeight = 0.30f;
constexpr float JumpLowSquattingSpeed = 0.20f;
constexpr float JumpLowAscendingVelocity = 8.0f;
constexpr float JumpLowKpPrepare = 18.0f;
constexpr float JumpLowKpLanding = 12.0f;
constexpr float JumpLowKpTakeoff = 32.0f;

float bodyH = ConfigurationParameter.DefaultHeight;
float bodyRoll = 0;

//滤波
float LegLength_f = ConfigurationParameter.DefaultHeight;  //腿长
float BarycenterX_f = 0;    //质心X
float BodyPitching_f = 0;   //俯仰
float BodyRoll_f = 0;       //横滚
float MovementSpeed_f = 0;  //移动速度
float BodyTurn_f = 0;       //转向
float SlideStep_f = 0;      //滑步
float BodyX_f = 0;          //X位置
biquadFilter_t FilterLPF[15];//二阶低通滤波器
float TouchY_Pid_outputF = 0;
float TouchX_Pid_outputF = 0;

float cutoffFreq = 200;
float enableDFilter = 1;
float LpfOut[6];//

void CutoffFreq(char* cmd) {
  command.scalar(&cutoffFreq, cmd);
}

void EnableDFilter(char* cmd) {
  command.scalar(&enableDFilter, cmd);
}

const int LED_Pin = 35; //LED IO


int LED_HL = 1;
int LED_count = 0;
int LED_dt = 100;
const int analogInPin = 17; //电池电压IO
int sensorValue = 0;        // value read from the pot
biquadFilter_t VoltageFilterLPF;//二阶低通滤波器
biquadFilter_t WheelSpeedFilterLPF[2];//

uint16_t VoltageADC = 0;//电压ADC数据
float VoltageADCf = 0;//电压ADC数据
float Voltage = 0;//电压

//IMU
IMUSystem imu; // 创建全局IMU系统实例
const ImuState& imuState = imu.state();  // 全局常量引用

//创建 MahonyFilter 对象，设置比例增益和积分增益
MahonyFilter mahonyFilter(0.4f, 0.001f);
// 加速度计量程（这里设置为 ±8g）
const float accelRange = 8.0;  // 单位：g
// 陀螺仪量程（这里假设为 ±2000°/s）
const float gyroRange = 2000.0;       // 单位：°/
attitude_t attitude;
float roll_ok;//
float pitch_ok;//
  
zeroBias_t zeroBias;//零点偏移
unsigned long timestamp_prev = 0;
float IMUtime_dt = 0;

/*低通滤波参数*/
float RATE_HZ_last = 1000.0f;//采样频率
float LPF_CUTOFF_FREQ_last = 50.0f;//截止频率

float RATE_HZ = 1000.0f;//采样频率
float LPF_CUTOFF_FREQ = 50.0f;//截止频率
biquadFilter_t ImuFilterLPF[6];//二阶低通滤波器


void ImuRATE_HZ(char* cmd)      {
  command.scalar(&RATE_HZ, cmd);
}
void ImuLPF_CUTOFF_FREQ(char* cmd)      {
  command.scalar(&LPF_CUTOFF_FREQ, cmd);
}

void Target_Leg_Length(char* cmd)      {
  command.scalar(&TargetLegLength, cmd);
}



//互补滤波
float angleGyroX, angleGyroY, angleGyroZ,
      angleAccX, angleAccY;
float angleX, angleY, angleZ;
float accCoef = 0.02f;
float gyroCoef = 0.98f;



// 声明一个Preferences对象，用于后续对闪存进行数据的读写操作
Preferences preferences;
// 定义一个浮点型数组，用于存储欧拉角数据
float zeroBiasFlash[10];
// 定义一个字符串指针数组，存储横滚 俯仰角对应的键名，用户可以直接修改这些键名
// 键名用于在闪存中唯一标识数据
const char* zeroBiasKeys[10] = {
  "roll",
  "pitch",
  "gyroX",
  "gyroY",
  "gyroZ",
  "servoAngle1",
  "servoAngle2",
  "servoAngle3",
  "servoAngle4",
  "accOffsetX"
};


//校准欧拉角
const int CALL_COUNT = 100;// 设定函数调用次数
int callCounter = 0;  // 调用计数器


void zeroBias_servo1(char* cmd)      {
  command.scalar(&zeroBias.servo1, cmd);
}
void zeroBias_servo2(char* cmd)      {
  command.scalar(&zeroBias.servo2, cmd);
}
void zeroBias_servo3(char* cmd)      {
  command.scalar(&zeroBias.servo3, cmd);
}
void zeroBias_servo4(char* cmd)      {
  command.scalar(&zeroBias.servo4, cmd);
}


//遥控器
FUTABA_SBUS sBus;
int sbus_dt_ms = 0;
uint32_t sbus_last_frame_ms = 0;
int sbus_swa = 0;
int sbus_swb = 0;
int sbus_swc = 0;
int sbus_swd = 0;
float sbus_vra = 0;
float sbus_vrb = 0;
float sbus_vraf = 0;
float sbus_vrbf = 0;

#define SBUS_chMax 1792
#define SBUS_chMin 192
constexpr int16_t SBUS_CENTER = 992;
constexpr int16_t SBUS_DRIVE_DEADBAND = 12;


// 创建 PID 控制器实例
float Select = 0;//选择要打印的数据
float CalibrationSelect = 0;//保存校准数据0：校准结束  1：校准陀螺仪 2：校准欧拉角 3：校准舵机

float PidCoefficient = 1;//
PIDController AnglePid (2, 0, 7 , 0 , 0);        // 2, 0, 7 , 0 , 0.0018 (Kp, Ki, Kd ,ramp ,limit) 5, 0, 7 , 0 , 0 
PIDController SpeedPid (1.3, 1.1, 0 , 0 , 1.3);       // 1.3, 1.3, 0 , 0 , 1.3
PIDController YawPid   (0.6,  2.5, 0 , 0 , 0);        // 0.6,  2.5, 0 , 0 , 0
PIDController RollPid  (0.003, 0.03, 0.00008 , 0 , 3.5); // 0.005, 0.03, 0.00008 , 0 , 3.5
PIDController TouchXPid(0, 0, 0 , 0 , 0); //
PIDController TouchYPid(0, 0, 0 , 0 , 0); //


float PidDt = 0.01;

// 创建MyPIDController实例，设置初始参数
MyPIDController Angle_Pid(0, 0, 0, 0, 11,PidDt,0,0);//p i d iLimit outputLimit dt EnableDFilter cutoffFreq
MyPIDController Speed_Pid(0, 0, 0, 0, 0.1,PidDt,0,0);
MyPIDController Yaw_Pid  (0, 0, 0, 0, 0,PidDt,0,0);
MyPIDController Roll_Pid  (0, 0, 0, 0, 0.1,PidDt,0,0);
MyPIDController Pitching_Pid  (0, 0, 0, 0, 0,PidDt,0,0);

MyPIDController TouchX_Pid  (0, 0, 0, 0, 10,PidDt,0,0);
MyPIDController TouchY_Pid  (0, 0, 0, 0, 8,PidDt,0,0);

// RAM-only dynamic tuning state. The proven pitch balance loop remains
// read-only; only the outer speed/yaw loops and heading hold are writable.
PidTuneTelemetry_t PidTuneTelemetry = {};
PidTuneValueSet PidTuneBootValues = {};
PidTuneValueSet PidTuneRollbackValues = {};
PidTuneValueSet PidTuneTargetValues = {};
bool PidTuneInitialized = false;
bool PidTuneSessionActive = false;
float HeadingHoldKp = 1.0f;
bool HeadingHoldEnabled = true;

float TuneSessionPosition = 0.0f;
float TuneSpeedError = 0.0f;
float TuneSpeedOutput = 0.0f;
float TuneAngleErrorDeg = 0.0f;
float TuneAngleOutput = 0.0f;
float TuneYawError = 0.0f;
float TuneYawOutput = 0.0f;

void InitPidTuning(void);
void UpdatePidTuning(float dt);
void ProcessPidTuneCommand(const PidTuneCommand_t& command);
void RosSerial_PidTune_TXD(bool force = false);

void TwoKp(char* cmd)      {
  command.scalar(&mahonyFilter.twoKp, cmd);
}
void TwoKi(char* cmd)      {
  command.scalar(&mahonyFilter.twoKi, cmd);
}

void KeyScalar(char* cmd)      {
  command.scalar(&Select, cmd);
}
void KeyCalibration(char* cmd) {
  command.scalar(&CalibrationSelect, cmd);
}

#if AdjusParameter == 0
void CbAnglePid(char* cmd) {
  command.pid(&AnglePid, cmd);
}
void CbSpeedPid(char* cmd) {
  command.pid(&SpeedPid, cmd);
}
void CbYawPid(char* cmd)  {
  command.pid(&YawPid, cmd);
}

void CbRollPid(char* cmd)  {
  command.pid(&RollPid, cmd);
}

#elif AdjusParameter == 1

void CbTouchXPid(char* cmd)  {
  command.pid(&TouchXPid, cmd);
}

void CbTouchYPid(char* cmd)  {
  command.pid(&TouchYPid, cmd);
}
#endif



float  Motor1_voltage_compensation = 0;
float  Motor2_voltage_compensation = 0;
double Motor1_place_last = 0;
float  Motor1_Velocity = 0;
float  Motor1_Velocity_f = 0;

double Motor2_place_last = 0;
float  Motor2_Velocity = 0;
float  Motor2_Velocity_f = 0;

float Motor1_Target = 0;
float Motor2_Target = 0;

float time_dt = 0;
unsigned long now_us = 0;
unsigned long now_us1 = 0;
unsigned long now_us2 = 0;


int RXsbus();
int RightInverseKinematics(float x, float y, float p, float* ax);
int LeftInverseKinematics(float x, float y, float p, float* ax);
void print_data(void);
void ImuUpdate(void);
void FlashSave(int sw);
void FlashInit(void);
void PIDcontroller_angle(float dt);
void PIDcontroller_posture(float dt);

void RemoteControlFiltering(void);
void ReadVoltage(void);
void PidParameter(void);
void Robot_Tumble(void);
float rotationsPerSecondToMetersPerSecond(float rotationsPerSecond);

void command_init(void);

void SetDiagnosticBootFault(uint8_t stage, uint32_t fault_bit)
{
  diagnostic_boot_stage = stage;
  diagnostic_boot_fault_bits |= fault_bit;
}

void RosSerial_Diagnostics_TXD(bool force)
{
  static uint32_t last_tx_ms = 0;
  const uint32_t now_ms = millis();
  if (!force && (uint32_t)(now_ms - last_tx_ms) < 500U)
    return;

  RobotDiagnostic.header1 = 0x7B;
  RobotDiagnostic.header2 = 0x2E;
  RobotDiagnostic.length = sizeof(RobotDiagnostic_t);
  RobotDiagnostic.protocol_version = 2;
  RobotDiagnostic.boot_stage = diagnostic_boot_stage;
  RobotDiagnostic.robot_mode = (uint8_t)robot_mode;
  RobotDiagnostic.uptime_ms = now_ms;

  float battery_mv = Voltage * 1000.0f;
  if (battery_mv < 0.0f) battery_mv = 0.0f;
  if (battery_mv > 65535.0f) battery_mv = 65535.0f;
  RobotDiagnostic.battery_mv = (uint16_t)battery_mv;
  RobotDiagnostic.sbus_frame_period_ms =
      (uint16_t)constrain(sbus_dt_ms, 0, 65535);
  const uint32_t sbus_age = sbus_last_frame_ms == 0
      ? 65535U : (uint32_t)(now_ms - sbus_last_frame_ms);
  RobotDiagnostic.sbus_frame_age_ms =
      (uint16_t)(sbus_age > 65535U ? 65535U : sbus_age);
  RobotDiagnostic.sbus_failsafe = sBus.Failsafe();

  for (int i = 0; i < 10; ++i)
    RobotDiagnostic.sbus_channels[i] = sBus.channels[i];

  uint8_t reset_mask = 0;
  if (sbus_age < 200U && sBus.Failsafe() == SBUS_SIGNAL_OK) reset_mask |= 1U << 0;
  if (sbus_swa != 2) reset_mask |= 1U << 1;
  if (sbus_swb == 0) reset_mask |= 1U << 2;
  if (sbus_swc == 0) reset_mask |= 1U << 3;
  if (sbus_swd == 0) reset_mask |= 1U << 4;
  if (Motor_KP <= 0.05f && Motor_KD <= 0.01f) reset_mask |= 1U << 5;
  RobotDiagnostic.remote_reset_mask = reset_mask;

  uint8_t seen_mask = 0;
  uint8_t fresh_mask = 0;
  uint8_t enabled_mask = 0;
  bool has_motor_error = false;
  for (int i = 0; i < 6; ++i)
  {
    if (JointMotor[i].gReceivedFrameCount > 0) seen_mask |= 1U << i;
    if (JointMotor[i].last_tx_ms != 0 &&
        (uint32_t)(now_ms - JointMotor[i].last_tx_ms) <= 500U)
      fresh_mask |= 1U << i;
    if (JointMotor[i].MCStatusword.status.switched_on) enabled_mask |= 1U << i;
    RobotDiagnostic.motor_errors[i] = JointMotor[i].MCStatusword.errors.errors_code;
    if (RobotDiagnostic.motor_errors[i] != 0) has_motor_error = true;
  }
  RobotDiagnostic.motor_seen_mask = seen_mask;
  RobotDiagnostic.motor_fresh_mask = fresh_mask;
  RobotDiagnostic.motor_enabled_mask = enabled_mask;
  RobotDiagnostic.yaw_hold_active = YawHoldActive ? 1U : 0U;
  RobotDiagnostic.yaw_bias_millirad_s = (int16_t)constrain(
      (int)lroundf(YawRateBias * 1000.0f), -32768, 32767);
  RobotDiagnostic.corrected_yaw_rate_millirad_s = (int16_t)constrain(
      (int)lroundf(YawCorrectedRate * 1000.0f), -32768, 32767);
  RobotDiagnostic.yaw_hold_error_centideg = (int16_t)constrain(
      (int)lroundf(YawHoldErrorDeg * 100.0f), -18000, 18000);

  uint32_t faults = diagnostic_boot_fault_bits;
  if (sbus_age >= 200U || sBus.Failsafe() != SBUS_SIGNAL_OK)
    faults |= DIAG_FAULT_SBUS;
  if (diagnostic_boot_stage == BOOT_STAGE_WAIT_REMOTE_RESET && reset_mask != 0x3F)
    faults |= DIAG_FAULT_REMOTE_NOT_RESET;
  if (Voltage > 0.5f && Voltage <= 6.2f)
    faults |= DIAG_FAULT_BATTERY_LOW;
  if (diagnostic_boot_stage == BOOT_STAGE_READY && fresh_mask != 0x3F)
    faults |= DIAG_FAULT_MOTOR_OFFLINE;
  if (has_motor_error)
    faults |= DIAG_FAULT_MOTOR_ERROR;
  RobotDiagnostic.fault_bits = faults;

  if (diagnostic_boot_fault_bits != 0)
    RobotDiagnostic.overall_state = 3;
  else if (diagnostic_boot_stage == BOOT_STAGE_WAIT_REMOTE_RESET)
    RobotDiagnostic.overall_state = 1;
  else if (diagnostic_boot_stage == BOOT_STAGE_READY)
    RobotDiagnostic.overall_state = faults == 0 ? 2 : 3;
  else
    RobotDiagnostic.overall_state = 0;

  RobotDiagnostic.count++;
  RobotDiagnostic.crc = 0;
  uint8_t data[sizeof(RobotDiagnostic_t)];
  memcpy(data, &RobotDiagnostic, sizeof(data));
  RobotDiagnostic.crc = crc16(data, sizeof(data) - 2);
  memcpy(data, &RobotDiagnostic, sizeof(data));

  if (RosSerial.availableForWrite() >= (int)sizeof(data))
  {
    RosSerial.write(data, sizeof(data));
    last_tx_ms = now_ms;
  }
}

PidTuneValueSet CapturePidTuneValues(void)
{
  PidTuneValueSet values;
  values.speed_kp = SpeedPid.P;
  values.speed_ki = SpeedPid.I;
  values.speed_kd = SpeedPid.D;
  values.yaw_kp = YawPid.P;
  values.yaw_ki = YawPid.I;
  values.yaw_kd = YawPid.D;
  values.heading_hold_kp = HeadingHoldKp;
  values.heading_hold_enable = HeadingHoldEnabled;
  return values;
}

void ApplyPidTuneValuesImmediately(const PidTuneValueSet& values)
{
  SpeedPid.P = values.speed_kp;
  SpeedPid.I = values.speed_ki;
  SpeedPid.D = values.speed_kd;
  YawPid.P = values.yaw_kp;
  YawPid.I = values.yaw_ki;
  YawPid.D = values.yaw_kd;
  HeadingHoldKp = values.heading_hold_kp;
  HeadingHoldEnabled = values.heading_hold_enable;
}

float MovePidValueTowards(float current, float target, float max_step)
{
  if (current < target)
    return min(current + max_step, target);
  if (current > target)
    return max(current - max_step, target);
  return current;
}

bool PidTuneTargetsReached(void)
{
  return fabsf(SpeedPid.P - PidTuneTargetValues.speed_kp) < 0.0005f &&
         fabsf(SpeedPid.I - PidTuneTargetValues.speed_ki) < 0.0005f &&
         fabsf(SpeedPid.D - PidTuneTargetValues.speed_kd) < 0.0005f &&
         fabsf(YawPid.P - PidTuneTargetValues.yaw_kp) < 0.0005f &&
         fabsf(YawPid.I - PidTuneTargetValues.yaw_ki) < 0.0005f &&
         fabsf(YawPid.D - PidTuneTargetValues.yaw_kd) < 0.0005f &&
         fabsf(HeadingHoldKp - PidTuneTargetValues.heading_hold_kp) < 0.0005f &&
         HeadingHoldEnabled == PidTuneTargetValues.heading_hold_enable;
}

uint16_t GetPidTuneSafetyFlags(void)
{
  uint16_t flags = 0;
  if (diagnostic_boot_stage == BOOT_STAGE_READY)
    flags |= PID_TUNE_SAFE_BOOT_READY;
  if (robot_mode == 0)
    flags |= PID_TUNE_SAFE_CHASSIS_MODE;
  if (fabsf(body.TargetMovementSpeed_x) < 0.01f &&
      fabsf(body.TargetYaw) < 0.01f)
    flags |= PID_TUNE_SAFE_ZERO_COMMAND;
  if (body.JumpingStep == 0)
    flags |= PID_TUNE_SAFE_NOT_JUMPING;
  if (fabsf(pitch_ok) < 10.0f)
    flags |= PID_TUNE_SAFE_PITCH;
  const uint32_t sbus_age = sbus_last_frame_ms == 0
      ? 0xFFFFFFFFUL : (uint32_t)(millis() - sbus_last_frame_ms);
  if (sbus_age < 200U && sBus.Failsafe() == SBUS_SIGNAL_OK)
    flags |= PID_TUNE_SAFE_REMOTE;
  return flags;
}

bool PidTuneStateAllowsChange(void)
{
  const uint16_t flags = GetPidTuneSafetyFlags();
  const uint16_t always_required =
      PID_TUNE_SAFE_BOOT_READY |
      PID_TUNE_SAFE_CHASSIS_MODE |
      PID_TUNE_SAFE_NOT_JUMPING;
  if ((flags & always_required) != always_required)
    return false;

  // SWA0 has no motor output, so parameters may be prepared before balancing.
  if (sbus_swa == 0)
    return true;

  const uint16_t balance_required =
      PID_TUNE_SAFE_ZERO_COMMAND |
      PID_TUNE_SAFE_PITCH |
      PID_TUNE_SAFE_REMOTE;
  return (flags & balance_required) == balance_required;
}

void SetPidTuneStatus(const PidTuneCommand_t& tune_command, uint8_t status)
{
  PidTuneTelemetry.last_sequence = tune_command.sequence;
  PidTuneTelemetry.last_parameter_id = tune_command.parameter_id;
  PidTuneTelemetry.last_status = status;
}

bool SetPidTuneTarget(uint8_t parameter_id, float value)
{
  if (!isfinite(value))
    return false;

  switch (parameter_id)
  {
    case PID_TUNE_PARAM_SPEED_KP:
      if (value < 0.65f || value > 1.95f) return false;
      PidTuneTargetValues.speed_kp = value;
      return true;
    case PID_TUNE_PARAM_SPEED_KI:
      if (value < 0.0f || value > 1.65f) return false;
      PidTuneTargetValues.speed_ki = value;
      return true;
    case PID_TUNE_PARAM_SPEED_KD:
      if (value < 0.0f || value > 0.20f) return false;
      PidTuneTargetValues.speed_kd = value;
      return true;
    case PID_TUNE_PARAM_YAW_KP:
      if (value < 0.10f || value > 1.20f) return false;
      PidTuneTargetValues.yaw_kp = value;
      return true;
    case PID_TUNE_PARAM_YAW_KI:
      if (value < 0.0f || value > 3.0f) return false;
      PidTuneTargetValues.yaw_ki = value;
      return true;
    case PID_TUNE_PARAM_YAW_KD:
      if (value < 0.0f || value > 0.20f) return false;
      PidTuneTargetValues.yaw_kd = value;
      return true;
    case PID_TUNE_PARAM_HEADING_HOLD_KP:
      if (value < 0.0f || value > 2.0f) return false;
      PidTuneTargetValues.heading_hold_kp = value;
      return true;
    case PID_TUNE_PARAM_HEADING_HOLD_ENABLE:
      if (value != 0.0f && value != 1.0f) return false;
      PidTuneTargetValues.heading_hold_enable = value > 0.5f;
      return true;
    default:
      return false;
  }
}

void InitPidTuning(void)
{
  PidTuneBootValues = CapturePidTuneValues();
  PidTuneRollbackValues = PidTuneBootValues;
  PidTuneTargetValues = PidTuneBootValues;
  PidTuneSessionActive = false;
  PidTuneTelemetry.last_status = PID_TUNE_STATUS_OK;
  PidTuneTelemetry.last_parameter_id = PID_TUNE_PARAM_NONE;
  PidTuneInitialized = true;
}

void ProcessPidTuneCommand(const PidTuneCommand_t& tune_command)
{
  if (!PidTuneInitialized)
  {
    SetPidTuneStatus(tune_command, PID_TUNE_STATUS_UNSAFE_STATE);
    return;
  }
  if (tune_command.protocol_version != 1U)
  {
    SetPidTuneStatus(tune_command, PID_TUNE_STATUS_BAD_VERSION);
    return;
  }

  switch (tune_command.action)
  {
    case PID_TUNE_ACTION_BEGIN:
      if (!PidTuneTargetsReached())
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_TARGET_PENDING);
        return;
      }
      if (!PidTuneStateAllowsChange())
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_UNSAFE_STATE);
        return;
      }
      PidTuneRollbackValues = CapturePidTuneValues();
      PidTuneTargetValues = PidTuneRollbackValues;
      TuneSessionPosition = 0.0f;
      PidTuneSessionActive = true;
      SetPidTuneStatus(tune_command, PID_TUNE_STATUS_OK);
      return;

    case PID_TUNE_ACTION_SET:
      if (!PidTuneSessionActive)
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_NO_SESSION);
        return;
      }
      if (!PidTuneStateAllowsChange())
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_UNSAFE_STATE);
        return;
      }
      if (tune_command.parameter_id < PID_TUNE_PARAM_SPEED_KP ||
          tune_command.parameter_id > PID_TUNE_PARAM_HEADING_HOLD_ENABLE)
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_BAD_PARAMETER);
        return;
      }
      if (!SetPidTuneTarget(tune_command.parameter_id, tune_command.value))
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_OUT_OF_RANGE);
        return;
      }
      SetPidTuneStatus(tune_command, PID_TUNE_STATUS_OK);
      return;

    case PID_TUNE_ACTION_REVERT:
      if (!PidTuneSessionActive)
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_NO_SESSION);
        return;
      }
      PidTuneTargetValues = PidTuneRollbackValues;
      if (sbus_swa == 0)
        ApplyPidTuneValuesImmediately(PidTuneRollbackValues);
      PidTuneSessionActive = false;
      SetPidTuneStatus(tune_command, PID_TUNE_STATUS_OK);
      return;

    case PID_TUNE_ACTION_RESTORE_BOOT:
      if (!PidTuneSessionActive)
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_NO_SESSION);
        return;
      }
      if (!PidTuneStateAllowsChange())
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_UNSAFE_STATE);
        return;
      }
      PidTuneTargetValues = PidTuneBootValues;
      SetPidTuneStatus(tune_command, PID_TUNE_STATUS_OK);
      return;

    case PID_TUNE_ACTION_ACCEPT_RAM:
      if (!PidTuneSessionActive)
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_NO_SESSION);
        return;
      }
      if (!PidTuneTargetsReached())
      {
        SetPidTuneStatus(tune_command, PID_TUNE_STATUS_TARGET_PENDING);
        return;
      }
      PidTuneRollbackValues = CapturePidTuneValues();
      PidTuneTargetValues = PidTuneRollbackValues;
      PidTuneSessionActive = false;
      SetPidTuneStatus(tune_command, PID_TUNE_STATUS_OK);
      return;

    case PID_TUNE_ACTION_GET:
      SetPidTuneStatus(tune_command, PID_TUNE_STATUS_OK);
      return;

    default:
      SetPidTuneStatus(tune_command, PID_TUNE_STATUS_BAD_ACTION);
      return;
  }
}

void UpdatePidTuning(float dt)
{
  if (!PidTuneInitialized || dt <= 0.0f || dt > 0.1f)
    return;

  static int last_swa = 0;
  if (PidTuneSessionActive && last_swa > 0 && sbus_swa == 0)
  {
    // SWA0 is the physical emergency action: restore the session snapshot
    // immediately while all motor output is disabled.
    PidTuneTargetValues = PidTuneRollbackValues;
    ApplyPidTuneValuesImmediately(PidTuneRollbackValues);
    PidTuneSessionActive = false;
    PidTuneTelemetry.last_status = PID_TUNE_STATUS_AUTO_REVERTED;
  }
  last_swa = sbus_swa;

  SpeedPid.P = MovePidValueTowards(
      SpeedPid.P, PidTuneTargetValues.speed_kp, 0.30f * dt);
  SpeedPid.I = MovePidValueTowards(
      SpeedPid.I, PidTuneTargetValues.speed_ki, 0.30f * dt);
  SpeedPid.D = MovePidValueTowards(
      SpeedPid.D, PidTuneTargetValues.speed_kd, 0.05f * dt);
  YawPid.P = MovePidValueTowards(
      YawPid.P, PidTuneTargetValues.yaw_kp, 0.20f * dt);
  YawPid.I = MovePidValueTowards(
      YawPid.I, PidTuneTargetValues.yaw_ki, 0.50f * dt);
  YawPid.D = MovePidValueTowards(
      YawPid.D, PidTuneTargetValues.yaw_kd, 0.05f * dt);
  HeadingHoldKp = MovePidValueTowards(
      HeadingHoldKp, PidTuneTargetValues.heading_hold_kp, 0.30f * dt);
  HeadingHoldEnabled = PidTuneTargetValues.heading_hold_enable;

  if (PidTuneSessionActive)
  {
    const float measured_speed =
        (Motor1_Velocity_f + Motor2_Velocity_f) * 0.5f;
    TuneSessionPosition += measured_speed * dt;
  }
}

void RosSerial_PidTune_TXD(bool force)
{
  static uint32_t last_tx_ms = 0;
  const uint32_t now_ms = millis();
  const uint32_t interval_ms = PidTuneSessionActive ? 20U : 100U;
  if (!force && (uint32_t)(now_ms - last_tx_ms) < interval_ms)
    return;

  PidTuneTelemetry.header1 = 0x7B;
  PidTuneTelemetry.header2 = 0x30;
  PidTuneTelemetry.length = sizeof(PidTuneTelemetry_t);
  PidTuneTelemetry.protocol_version = 1;
  PidTuneTelemetry.session_active = PidTuneSessionActive ? 1U : 0U;
  PidTuneTelemetry.safety_flags = GetPidTuneSafetyFlags();
  PidTuneTelemetry.uptime_ms = now_ms;
  PidTuneTelemetry.target_speed_mps = body.TargetMovementSpeed_x_f;
  PidTuneTelemetry.measured_speed_mps =
      (Motor1_Velocity_f + Motor2_Velocity_f) * 0.5f;
  PidTuneTelemetry.session_position_m = TuneSessionPosition;
  PidTuneTelemetry.target_pitch_deg = BodyPitching_f;
  PidTuneTelemetry.measured_pitch_deg = pitch_ok;
  PidTuneTelemetry.pitch_rate_rad_s = imuState.filtered_gyro.y;
  PidTuneTelemetry.speed_error_mps = TuneSpeedError;
  PidTuneTelemetry.speed_integral = Speed_Pid.integral;
  PidTuneTelemetry.speed_output = TuneSpeedOutput;
  PidTuneTelemetry.target_leg_x_m = body.TargetPosition_x;
  PidTuneTelemetry.angle_error_deg = TuneAngleErrorDeg;
  PidTuneTelemetry.angle_output_nm = TuneAngleOutput;
  PidTuneTelemetry.target_yaw_rate_rad_s = body.TargetYaw;
  PidTuneTelemetry.measured_yaw_rate_rad_s = YawCorrectedRate;
  PidTuneTelemetry.yaw_error_rad_s = TuneYawError;
  PidTuneTelemetry.yaw_output_nm = TuneYawOutput;
  PidTuneTelemetry.yaw_hold_error_deg = YawHoldErrorDeg;
  PidTuneTelemetry.left_wheel_torque_nm = body.WheelTargetTorque[0];
  PidTuneTelemetry.right_wheel_torque_nm = body.WheelTargetTorque[1];
  PidTuneTelemetry.speed_kp = SpeedPid.P;
  PidTuneTelemetry.speed_ki = SpeedPid.I;
  PidTuneTelemetry.speed_kd = SpeedPid.D;
  PidTuneTelemetry.applied_speed_kp = Speed_Pid.Kp;
  PidTuneTelemetry.applied_speed_ki = Speed_Pid.Ki;
  PidTuneTelemetry.yaw_kp = YawPid.P;
  PidTuneTelemetry.yaw_ki = YawPid.I;
  PidTuneTelemetry.yaw_kd = YawPid.D;
  PidTuneTelemetry.applied_yaw_ki = Yaw_Pid.Ki;
  PidTuneTelemetry.heading_hold_kp = HeadingHoldKp;
  PidTuneTelemetry.configured_angle_kp = AnglePid.P;
  PidTuneTelemetry.applied_angle_kp = Angle_Pid.Kp;
  PidTuneTelemetry.angle_kd = Angle_Pid.Kd;
  PidTuneTelemetry.count++;
  PidTuneTelemetry.crc = 0;

  uint8_t data[sizeof(PidTuneTelemetry_t)];
  memcpy(data, &PidTuneTelemetry, sizeof(data));
  PidTuneTelemetry.crc = crc16(data, sizeof(data) - 2);
  memcpy(data, &PidTuneTelemetry, sizeof(data));
  if (RosSerial.availableForWrite() >= (int)sizeof(data))
  {
    RosSerial.write(data, sizeof(data));
    last_tx_ms = now_ms;
  }
}

void command_init(void)
{

  command.add('A', zeroBias_servo1, "my zeroBias_servo1"); //设置舵机1偏差
  command.add('B', zeroBias_servo2, "my zeroBias_servo2"); //
  command.add('C', zeroBias_servo3, "my zeroBias_servo3"); //
  command.add('D', zeroBias_servo4, "my zeroBias_servo4"); //

  command.add('H', ImuRATE_HZ, "my ImuRATE_HZ");
  command.add('Z', ImuLPF_CUTOFF_FREQ, "my ImuLPF_CUTOFF_FREQ");


  command.add('Q', TwoKp, "my TwoKp"); //MahonyFilter
  command.add('I', TwoKi, "my TwoKi"); //MahonyFilter

  command.add('K', KeyScalar, "my Select");
  command.add('E', KeyCalibration, "my CalibrationSelect");

  command.add('F', CutoffFreq, "my CutoffFreq");
  command.add('J', EnableDFilter, "my EnableDFilter");

#if AdjusParameter == 0
  command.add('P', CbAnglePid, "my AnglePid");
  command.add('S', CbSpeedPid, "my SpeedPid");
  command.add('Y', CbYawPid, "my YawPid");
  command.add('R', CbRollPid, "my RollPid");
  command.add('O', Target_Leg_Length, "my Target_Leg_Length");
#elif AdjusParameter == 1  
  command.add('L', CbTouchXPid, "my CbTouchXPid");
  command.add('N', CbTouchYPid, "my CbTouchYPid");
#endif

}

void ConfigurationParameterInit(void)
{
  if(Model_Selection==0)//长臂机型
  { 
    //限制
    ConfigurationParameter.LimitLinearVelocity = 0.6;         //机器最大移动速度m/s
    ConfigurationParameter.LimitAngleVelocity = 3.0;          //机器航向角速度 弧度每秒限制
    ConfigurationParameter.LimitPitchAngle = 35.0;            //俯仰角度限制 度
    ConfigurationParameter.LimitRollAngle = 15.0;             //横滚角度限制 度
    ConfigurationParameter.MaxHigh = 0.37;                    //腿最高
    ConfigurationParameter.MinLow = 0.13;                     //腿最低  
    ConfigurationParameter.PostureRestrictions_x = 0.08;      //姿态 x ± 限制
    ConfigurationParameter. DefaultHeight = 0.25;             //摇杆中间时的高度
    ConfigurationParameter.ParallelHigh = 0.143331;           //大腿与机体平行高度
    //电机
    ConfigurationParameter.JointAngleDeviation = 96.25619525; //关节角度偏差
    //机体
    ConfigurationParameter.BodyWeight = 3.0;                  //身体重量
    ConfigurationParameter.WheelTrack = 0.4118;               //轮距
    ConfigurationParameter.thigh = 0.15;                      // 大腿长度m L1 L4
    ConfigurationParameter.shank = 0.24;                      // 小腿长度m L2 L3
    ConfigurationParameter.MotorSpacing = 0.085;              // 电机间距 L5

    ConfigurationParameter.PitchControlCompensation = 85;

    ConfigurationParameter.PayloadWeight = 0;                 //kg
  }
  else if(Model_Selection==1)//短臂机型
  {
    SpeedPid.P = 1.1f;
    //限制
    ConfigurationParameter.LimitLinearVelocity = 2.0;         //机器最大移动速度m/s
    ConfigurationParameter.LimitAngleVelocity = 5.0;          //机器航向角速度 弧度每秒限制
    ConfigurationParameter.LimitPitchAngle = 35.0;            //俯仰角度限制 度
    ConfigurationParameter.LimitRollAngle = 15.0;             //横滚角度限制 度
    ConfigurationParameter.MaxHigh = 0.32;                    //腿最高 0.337
    ConfigurationParameter.MinLow = 0.17;                     //腿最低  
    ConfigurationParameter. DefaultHeight = 0.245;            //摇杆中间时的高度
    ConfigurationParameter.ParallelHigh = 0.193;              //大腿与机体平行高度
    ConfigurationParameter.PostureRestrictions_x = 0.08;      //姿态 x ± 限制
    //电机
    ConfigurationParameter.JointAngleDeviation = 97.181; //关节角度偏差
    //机体
    ConfigurationParameter.BodyWeight = 3.0;                  //身体重量 
    ConfigurationParameter.WheelTrack = 0.4118;               //轮距
    ConfigurationParameter.thigh = 0.10;                      // 大腿长度m L1 L4
    ConfigurationParameter.shank = 0.24;                      // 小腿长度m L2 L3
    ConfigurationParameter.MotorSpacing = 0.085;              // 电机间距 L5    

    ConfigurationParameter.PitchControlCompensation = 85;

    ConfigurationParameter.PayloadWeight = 0;                 //kg
  }
  
}


float mapf(long x, long in_min, long in_max, float out_min, float out_max)
{
  long divisor = (in_max - in_min);
  if (divisor == 0) {
    return -1; //AVR returns -1, SAM returns 0
  }
  return (x - in_min) * (out_max - out_min) / divisor + out_min;
}

float mapff(float x, float in_min, float in_max, float out_min, float out_max)
{
  float divisor = (in_max - in_min);
  if (divisor == 0) {
    return -1; //AVR returns -1, SAM returns 0
  }
  return (x - in_min) * (out_max - out_min) / divisor + out_min;
}

// Remove transmitter jitter around neutral without reducing the command range.
float mapSbusCentered(int16_t raw, float max_abs)
{
  raw = constrain(raw, SBUS_chMin, SBUS_chMax);
  if (abs(raw - SBUS_CENTER) <= SBUS_DRIVE_DEADBAND)
    return 0.0f;

  if (raw < SBUS_CENTER - SBUS_DRIVE_DEADBAND)
  {
    return mapff((float)raw, (float)SBUS_chMin,
                 (float)(SBUS_CENTER - SBUS_DRIVE_DEADBAND),
                 -max_abs, 0.0f);
  }

  return mapff((float)raw, (float)(SBUS_CENTER + SBUS_DRIVE_DEADBAND),
               (float)SBUS_chMax, 0.0f, max_abs);
}

void InitRosData(void)
{
  RosComWheelLegged.linear_velocity = 0;
  RosComWheelLegged.angle_velocity = 0;
  RosComWheelLegged.leg_length = ConfigurationParameter.DefaultHeight;
  RosComWheelLegged.pitch_angle = 0;
  RosComWheelLegged.roll_angle = 0;
  RosComWheelLegged.jump_min_height = JumpLowMinHeight;
  RosComWheelLegged.jump_max_height = JumpLowMaxHeight;
  RosComWheelLegged.set = 0;
}

// 自定义函数：限制float类型变量的范围
float constrainFloat(float value, float minVal, float maxVal) {
  if (value < minVal) {
    return minVal;  // 若小于最小值，返回最小值
  } else if (value > maxVal) {
    return maxVal;  // 若大于最大值，返回最大值
  } else {
    return value;   // 在范围内则返回原值
  }
}



int SWAtransition(int sw)
{
  static int count1 = 0;
  static int count2 = 0; 
  static int out = 0;//0：停止 1：开机 2：上位机通讯

  if(sw==0)
  {
    count1++;
    count2 = 0;
  }
  else if(sw==2)
  {
    count1 = 0;
    count2++;    
  }
  else if(sw==1)
  {
     if(count2>3)
     {
        out = 2;
     }
     else 
     {
        if((count1>3)&&(out==1))
        {
          out = 0;
          count1 = 0;
        }
        else if(count1>3)
        {
          out = 1;
          count1 = 0;
        }
     }
  }
  
  return out;
}



int RXsbus()
{
  static unsigned long now_ms = millis();
  static int sign = 0;
  sBus.FeedLine();
  if (sBus.toChannels == 1)
  {
    if(sign<21)
      sign++;
    sbus_dt_ms = millis()-now_ms;
    now_ms = millis();
    sbus_last_frame_ms = now_ms;
    sBus.UpdateChannels();
    sBus.toChannels = 0;
    for (int i = 0; i < 10; i++)
      ROS_body.SBUS_Channels_Data[i] = sBus.channels[i];

    sbus_swa      =  map(sBus.channels[4], SBUS_chMin, SBUS_chMax, 0, 2);
    if(SelectTelecontrol==0)
      sbus_swa = SWAtransition(sbus_swa);
    
    sbus_swb      =  map(sBus.channels[5], SBUS_chMin, SBUS_chMax, 0, 1);
    sbus_swc      =  map(sBus.channels[6], SBUS_chMin, SBUS_chMax, 0, 1);
    sbus_swd      =  map(sBus.channels[7], SBUS_chMin, SBUS_chMax, 0, 2);
    sbus_vra    =  mapf(sBus.channels[8], SBUS_chMin, SBUS_chMax, -5, 5);//修改顶球目标位置
    sbus_vrb    =  mapf(sBus.channels[9], SBUS_chMin, SBUS_chMax, -5, 5);
    Motor_KP = mapf(sBus.channels[8], SBUS_chMin, SBUS_chMax, 0, 20);
    Motor_KD = mapf(sBus.channels[8], SBUS_chMin, SBUS_chMax, 0, 0.8);
    
    ConfigurationParameter.PayloadWeight = mapf(sBus.channels[9], SBUS_chMin, SBUS_chMax, 0, 5);//kg
     
    if((sbus_swa==2)&&(RosMessageSign==1))//上位机控制&&接收到上位机数据
    {
      const int8_t slide_raw = (int8_t)((RosComWheelLegged.set >> 8) & 0xFF);
      SlideStep = constrain((slide_raw / 127.0f) * 0.08f, -0.08f, 0.08f);
      body.TargetMovementSpeed_x = constrain(RosComWheelLegged.linear_velocity, -ConfigurationParameter.LimitLinearVelocity, ConfigurationParameter.LimitLinearVelocity);
      body.TargetYaw = constrain(RosComWheelLegged.angle_velocity,-ConfigurationParameter.LimitAngleVelocity,ConfigurationParameter.LimitAngleVelocity); 
      LegLength = constrain(RosComWheelLegged.leg_length,ConfigurationParameter.MinLow, ConfigurationParameter.MaxHigh);
      BodyPitching = constrain(RosComWheelLegged.pitch_angle,-ConfigurationParameter.LimitPitchAngle, ConfigurationParameter.LimitPitchAngle);
      body.TargetRoll = constrain(RosComWheelLegged.roll_angle,-ConfigurationParameter.LimitRollAngle, ConfigurationParameter.LimitRollAngle);  

    }
    else//遥控控制
    {

      body.TargetMovementSpeed_x = mapSbusCentered(
          sBus.channels[2], ConfigurationParameter.LimitLinearVelocity);//米/s
      body.TargetYaw = -mapSbusCentered(
          sBus.channels[3], ConfigurationParameter.LimitAngleVelocity);
      if (sbus_swd == 0)//姿态控制1
      {
        SlideStep = 0;
        LegLength = mapf(sBus.channels[1], SBUS_chMin, SBUS_chMax, ConfigurationParameter.MinLow, ConfigurationParameter.MaxHigh); //双腿高

        /*
        if(sBus.channels[1]<=992)
          LegLength = mapf(sBus.channels[1], SBUS_chMin, 992, ConfigurationParameter.MinLow, ConfigurationParameter.DefaultHeight); //双腿高
        else
          LegLength = mapf(sBus.channels[1], 993, SBUS_chMax, ConfigurationParameter.DefaultHeight, ConfigurationParameter.MaxHigh); //双腿高
        */

        
        body.TargetRoll =  mapf(sBus.channels[0], SBUS_chMin, SBUS_chMax, -ConfigurationParameter.LimitRollAngle, ConfigurationParameter.LimitRollAngle); //横滚
        BodyPitching = 0;
      }
      else if (sbus_swd == 1)//姿态控制2
      {
        SlideStep = 0;
        LegLength = mapf(sBus.channels[1], SBUS_chMin, SBUS_chMax, ConfigurationParameter.MinLow, ConfigurationParameter.MaxHigh); //双腿高

        /*
        if(sBus.channels[1]<=992)
          LegLength = mapf(sBus.channels[1], SBUS_chMin, 992, ConfigurationParameter.MinLow, ConfigurationParameter.DefaultHeight); //双腿高
        else
          LegLength = mapf(sBus.channels[1], 993, SBUS_chMax, ConfigurationParameter.DefaultHeight, ConfigurationParameter.MaxHigh); //双腿高
        */
        
        body.TargetRoll =  0;
        BodyPitching = mapf(sBus.channels[0], SBUS_chMin, SBUS_chMax, -ConfigurationParameter.LimitPitchAngle, ConfigurationParameter.LimitPitchAngle); //俯仰
      }
      else if (sbus_swd == 2)//姿态控制3
      {
        LegLength = ConfigurationParameter.DefaultHeight; //默认腿高
        SlideStep = mapf(sBus.channels[1], SBUS_chMin, SBUS_chMax, -0.08, 0.08); //滑步
        body.TargetRoll =  mapf(sBus.channels[0], SBUS_chMin, SBUS_chMax, -ConfigurationParameter.LimitRollAngle, ConfigurationParameter.LimitRollAngle); //横滚
        BodyPitching = 0;

        
      }      
    }

    //修改PID系数
    if((LegLength>=ConfigurationParameter.DefaultHeight)&&(LegLength<=ConfigurationParameter.MaxHigh))
      PidCoefficient = 1;//PidCoefficient = mapff(LegLength, 0.25, ConfigurationParameter.MaxHigh, 1, 0.5);
    else
      PidCoefficient = 1;
    
    if(Voltage<=6.2)
    {
      DEBUG.print(" Voltage:");
      DEBUG.println(Voltage, 5);      
    }

  }

  return sign;
}



float ArcToAngle(float arc)//弧度转角度
{
  float angle = arc * (180 / PI);
  return angle;
}

float AngleToArc(float angle)//角度转弧度
{
  float art = angle * (PI / 180);
  return art;
}


// 运动学正解函数
void forwardKinematics(Node * LeftPlace, Node * RightPlace, JointAngles * jointAngles)
{

  float alpha, beta, theta1, theta2, selectedTheta;
  Node A, B, C;

  // 逆解控制参数
  float L1 = ConfigurationParameter.thigh;
  float L2 = ConfigurationParameter.shank;
  float L3 = ConfigurationParameter.shank;
  float L4 = ConfigurationParameter.thigh;
  float L5 = ConfigurationParameter.MotorSpacing;
   
  //右腿正解
  alpha = degreesToRadians(180-jointAngles->alphaRight);
  beta  =  degreesToRadians(jointAngles->betaRight);

  A.x = L1 * cos(alpha);
  A.y = L1 * sin(alpha);
  C.x = L5 + L4 * cos(beta);
  C.y = L4 * sin(beta);
  
  float aRight = 2 * (A.x - C.x) * L2;
  float bRight = 2 * (A.y - C.y) * L2;
  float lRight = sqrt((A.x - C.x) * (A.x - C.x) + (A.y - C.y) * (A.y - C.y));//a点与b点的距离
  float cRight = (L3 * L3 - L2 *L2 - lRight * lRight);
 
  
  
  theta1 = 2 * atan((bRight + sqrt((bRight * bRight) + (aRight * aRight) - (cRight * cRight)))/(aRight + cRight));
  theta2 = 2 * atan((bRight - sqrt((bRight * bRight) + (aRight * aRight) - (cRight * cRight)))/(aRight + cRight));
  

  // 角度解算范围限制
  theta1 = (theta1 >= 0) ? theta1 : (theta1 + 2 * PI);
  theta2 = (theta2 >= 0) ? theta2 : (theta2 + 2 * PI);

  if (theta1 >= PI / 2)
  {
    selectedTheta = theta2;
  }
  else
  {
    selectedTheta = theta1;
  }

  RightPlace->x = A.x + L2 * cos(selectedTheta) - 0.0425f;;
  RightPlace->y = A.y + L2 * sin(selectedTheta);


   //左腿正解
   alpha = degreesToRadians(180-jointAngles->alphaLeft);
   beta  =  degreesToRadians(jointAngles->betaLeft);


   A.x = L1 * cos(alpha);
   A.y = L1 * sin(alpha);
   C.x = L5 + L4 * cos(beta);
   C.y = L4 * sin(beta);
   
   float aLeft = 2 * (A.x - C.x) * L2;
   float bLeft = 2 * (A.y - C.y) * L2;
   float lLeft = sqrt((A.x - C.x) * (A.x - C.x) + (A.y - C.y) * (A.y - C.y));
   float cLeft = (L3 * L3 - L2 *L2 - lLeft * lLeft);

   theta1 = 2 * atan((bLeft + sqrt((aLeft * aLeft) + (bLeft * bLeft) - (cLeft * cLeft)))/(aLeft + cLeft));
   theta2 = 2 * atan((bLeft - sqrt((aLeft * aLeft) + (bLeft * bLeft) - (cLeft *  cLeft)))/(aLeft + cLeft));
   
 
   // 角度解算范围限制
   theta1 = (theta1 >= 0) ? theta1 : (theta1 + 2 * PI);
   theta2 = (theta2 >= 0) ? theta2 : (theta2 + 2 * PI);
 
   if (theta1 >= PI / 2)
   {
    selectedTheta = theta2;
   }
   else
   {
    selectedTheta = theta1;
   }
   
   LeftPlace->x = A.x + L2 * cos(selectedTheta) - 0.0425f;
   LeftPlace->y = A.y + L2 * sin(selectedTheta);

}





//五连杆动学逆解
int RightInverseKinematics(float x, float y, float p, float* ax)
{ 
  x = constrain(x, -ConfigurationParameter.PostureRestrictions_x, ConfigurationParameter.PostureRestrictions_x);
  y = constrain(y, ConfigurationParameter.MinLow, ConfigurationParameter.MaxHigh);

  int error = 0;//坐标设置异常
  float AB = ConfigurationParameter.thigh;// 大腿长度m AB=ED
  float BC = ConfigurationParameter.shank;// 小腿长度m BC=DC

  float OA = ConfigurationParameter.MotorSpacing/2.0f; //
  float aOCF = 0;
  float aOCF2 = 0;
  float aAOC = 0;
  float OC = 0;
  float OF = 0;
  float FC = 0;
  float AC = 0;
  float aOAC = 0;
  float aOCA = 0;
  float aBAC = 0;
  float aBAG = 0;
  float OE = OA; //
  float aEOC = 0;
  float EC = 0;
  float aOCE = 0;
  float aOEC = 0;
  float aDEC = 0;
  float aDEH = 0;

  float pitch, x1, y1;

  pitch = AngleToArc(p);  //俯仰角

  x1 = x * cosf(pitch) - y * sinf(pitch);
  y1 = x * sinf(pitch) + y * cosf(pitch);
  
  OF = x1;
  FC = y1;

  //关节1
  OC = sqrtf(pow(OF, 2) + pow(FC, 2)); //https://zh.numberempire.com/right_triangle_calculator.php
  if (AC >= (AB + BC))  //超出结构最大范围
    return error = 1;
  aOCF = asinf(OF / OC);
  aAOC = AngleToArc(90) + aOCF;

  AC = sqrtf(pow(OA, 2) + pow(OC, 2) - 2 * OA * OC * cos(aAOC)); //https://zh.numberempire.com/arbitrary_triangle_calculator.php?__cf_chl_tk=vr4WMdDsGMhU1V4wG4xVXMisQ2Q6fSgLDuKK6VlnV4o-1734502544-1.0.1.1-CoxTBWpWtqs3CDDFrz5boHxA3INdl0yYHMOzF5kZz8w
  aOCA = acosf((pow(OC, 2) + pow(AC, 2) - pow(OA, 2)) / (2 * OC * AC));
  aOAC = PI - aOCA - aAOC;
  aBAC = acos((pow(AB, 2) + pow(AC, 2) - pow(BC, 2)) / (2 * AB * AC));
  aBAG = PI - aBAC - aOAC;
  //ax[0] = ArcToAngle(aBAG);//关节1角度

  //关节2
  aOCF2 = -aOCF;
  aEOC = AngleToArc(90) + aOCF2;

  EC = sqrtf(pow(OE, 2) + pow(OC, 2) - 2 * OE * OC * cos(aEOC));
  if (EC >= (AB + BC)) //超出结构最大范围
    return error = 2;
  aOCE = acosf((pow(OC, 2) + pow(EC, 2) - pow(OE, 2)) / (2 * OC * EC));
  aOEC = PI - aOCE - aEOC;
  aDEC = acos((pow(AB, 2) + pow(EC, 2) - pow(BC, 2)) / (2 * AB * EC));
  aDEH = PI - aDEC - aOEC;
  ax[1] = ArcToAngle(aDEH);//关节1角度
  ax[0] = ArcToAngle(aBAG);//关节1角度

  return error;
}



int LeftInverseKinematics(float x, float y, float p, float* ax)
{
  x = constrain(x, -ConfigurationParameter.PostureRestrictions_x, ConfigurationParameter.PostureRestrictions_x);
  y = constrain(y, ConfigurationParameter.MinLow, ConfigurationParameter.MaxHigh);
  
  x = -x;
  p = -p;
  int error = 0;//坐标设置异常
  float AB = ConfigurationParameter.thigh;// 大腿长度m AB=ED
  float BC = ConfigurationParameter.shank;// 小腿长度m BC=DC

  float OA = ConfigurationParameter.MotorSpacing/2.0f; //
  float aOCF = 0;
  float aOCF2 = 0;
  float aAOC = 0;
  float OC = 0;
  float OF = 0;
  float FC = 0;
  float AC = 0;
  float aOAC = 0;
  float aOCA = 0;
  float aBAC = 0;
  float aBAG = 0;
  float OE = OA; //
  float aEOC = 0;
  float EC = 0;
  float aOCE = 0;
  float aOEC = 0;
  float aDEC = 0;
  float aDEH = 0;

  float pitch, x1, y1;

  pitch = AngleToArc(p);  //俯仰角

  x1 = x * cosf(pitch) - y * sinf(pitch);
  y1 = x * sinf(pitch) + y * cosf(pitch);


  OF = -x1;
  FC = y1;

  //关节1
  OC = sqrtf(pow(OF, 2) + pow(FC, 2)); //https://zh.numberempire.com/right_triangle_calculator.php
  aOCF = asinf(OF / OC);
  aAOC = AngleToArc(90) + aOCF;

  AC = sqrtf(pow(OA, 2) + pow(OC, 2) - 2 * OA * OC * cos(aAOC)); //https://zh.numberempire.com/arbitrary_triangle_calculator.php?__cf_chl_tk=vr4WMdDsGMhU1V4wG4xVXMisQ2Q6fSgLDuKK6VlnV4o-1734502544-1.0.1.1-CoxTBWpWtqs3CDDFrz5boHxA3INdl0yYHMOzF5kZz8w
  if (AC >= (AB + BC))  //超出结构最大范围
  return error = 1;
  
  aOCA = acosf((pow(OC, 2) + pow(AC, 2) - pow(OA, 2)) / (2 * OC * AC));
  aOAC = PI - aOCA - aAOC;
  aBAC = acos((pow(AB, 2) + pow(AC, 2) - pow(BC, 2)) / (2 * AB * AC));
  aBAG = PI - aBAC - aOAC;
  //ax[0] = ArcToAngle(aBAG);//关节1角度

  //关节2
  aOCF2 = -aOCF;
  aEOC = AngleToArc(90) + aOCF2;

  EC = sqrtf(pow(OE, 2) + pow(OC, 2) - 2 * OE * OC * cos(aEOC));
  if (EC >= (AB + BC)) //超出结构最大范围
    return error = 2;
    
  aOCE = acosf((pow(OC, 2) + pow(EC, 2) - pow(OE, 2)) / (2 * OC * EC));
  aOEC = PI - aOCE - aEOC;
  aDEC = acos((pow(AB, 2) + pow(EC, 2) - pow(BC, 2)) / (2 * AB * EC));
  aDEH = PI - aDEC - aOEC;
  ax[1] = ArcToAngle(aDEH);//关节1角度
  ax[0] = ArcToAngle(aBAG);//关节1角度
  
  return error;
}


void ImuUpdate(void)
{
  imu.update(); // 更新数据

  // Learn the remaining Z-gyro bias only in the safe SWA0 state while both
  // wheels are stationary. Leave SWA0 unchanged for about two seconds after
  // startup. After 500 initial samples, a very slow update tracks temperature.
  const bool yaw_bias_learning_safe =
      (sbus_swa == 0) &&
      (fabsf(Motor1_Velocity_f) < 0.03f) &&
      (fabsf(Motor2_Velocity_f) < 0.03f) &&
      (fabsf(imuState.filtered_gyro.z) < 0.08f);
  if (yaw_bias_learning_safe)
  {
    const float alpha = YawBiasSamples < 500U
        ? (1.0f / (float)(YawBiasSamples + 1U)) : 0.002f;
    YawRateBias += alpha * (imuState.filtered_gyro.z - YawRateBias);
    YawRateBias = constrain(YawRateBias, -0.05f, 0.05f);
    if (YawBiasSamples < 500U) YawBiasSamples++;
  }
  YawCorrectedRate = imuState.filtered_gyro.z - YawRateBias;

  roll_ok = imuState.roll - zeroBias.roll;
  pitch_ok = imuState.pitch - zeroBias.pitch;

  ROS_body.ImuData.q0 = imuState.q0;
  ROS_body.ImuData.q1 = imuState.q1;
  ROS_body.ImuData.q2 = imuState.q2;
  ROS_body.ImuData.q3 = imuState.q3;
  ROS_body.ImuData.gyro[0] = imuState.filtered_gyro.x;
  ROS_body.ImuData.gyro[1] = imuState.filtered_gyro.y;
  ROS_body.ImuData.gyro[2] = imuState.filtered_gyro.z;
  
  ROS_body.ImuData.accel[0] = imuState.filtered_accel.x;
  ROS_body.ImuData.accel[1] = imuState.filtered_accel.y;
  ROS_body.ImuData.accel[2] = imuState.filtered_accel.z;
  
}


void FlashInit(void)
{
  preferences.begin("preferences", false);

  //读出数据  如果读取失败（即闪存中不存在该键名对应的数据），则使用默认值0.0
  zeroBias.roll = preferences.getFloat(zeroBiasKeys[0], 0.0);
  zeroBias.pitch = preferences.getFloat(zeroBiasKeys[1], 0.0);

  //读出数据  如果读取失败（即闪存中不存在该键名对应的数据），则使用默认值0.0
  gyroBiasX = preferences.getFloat(zeroBiasKeys[2], 0.0);
  gyroBiasY = preferences.getFloat(zeroBiasKeys[3], 0.0);
  gyroBiasZ = preferences.getFloat(zeroBiasKeys[4], 0.0);

  Vector3f manualBias(gyroBiasX, gyroBiasY, gyroBiasZ);  // X/Y/Z轴偏置值
  imu.setGyroBias(manualBias);

  ConfigurationParameter.AccOffsetX = preferences.getFloat(zeroBiasKeys[9], 0.0);

  //电机角度
  zeroBias.servo1 = ConfigurationParameter.JointAngleDeviation;
  zeroBias.servo2 = ConfigurationParameter.JointAngleDeviation;
  zeroBias.servo3 = ConfigurationParameter.JointAngleDeviation;
  zeroBias.servo4 = ConfigurationParameter.JointAngleDeviation;


  // 关闭闪存访问，释放相关资源
  preferences.end();

  DEBUG.println(" ");

  DEBUG.print("  AccOffsetX: ");
  DEBUG.print(ConfigurationParameter.AccOffsetX,5);
  
  // 输出零偏
  DEBUG.print("  Roll Zero Bias: ");
  DEBUG.print(zeroBias.roll);
  DEBUG.print("  Pitch Zero Bias: ");
  DEBUG.println(zeroBias.pitch);

  // 输出零偏
  DEBUG.print("  gyroBiasX:");
  DEBUG.print(gyroBiasX);
  DEBUG.print("  gyroBiasY:");
  DEBUG.print(gyroBiasY);
  DEBUG.print("  gyroBiasZ:");
  DEBUG.println(gyroBiasZ);


  // 输出零偏
  DEBUG.print("  servo1:");
  DEBUG.print(zeroBias.servo1);
  DEBUG.print("  servo2:");
  DEBUG.print(zeroBias.servo2);
  DEBUG.print("  servo3:");
  DEBUG.print(zeroBias.servo3);
  DEBUG.print("  servo4:");
  DEBUG.println(zeroBias.servo4);



}


// 计算欧拉角零偏的函数
void calculateZeroBias() {
  // 初始化累加器
  static float rollSum = 0;
  static float pitchSum = 0;

  // 累加欧拉角
  rollSum += imuState.roll;
  pitchSum += imuState.pitch;

  // 增加调用计数器
  callCounter++;

  if (callCounter >= CALL_COUNT)
  {
    // 计算平均值，得到零偏
    zeroBias.roll = rollSum / CALL_COUNT;
    zeroBias.pitch = pitchSum / CALL_COUNT;

    // 初始化闪存访问，打开名为 "preferences" 的命名空间
    // 第二个参数为false，表示以可写模式打开命名空间
    preferences.begin("preferences", false);

    //写入数据
    zeroBiasFlash[0] = zeroBias.roll;
    preferences.putFloat(zeroBiasKeys[0], zeroBiasFlash[0]);
    zeroBiasFlash[1] = zeroBias.pitch;
    preferences.putFloat(zeroBiasKeys[1], zeroBiasFlash[1]);
    //读出数据  如果读取失败（即闪存中不存在该键名对应的数据），则使用默认值0.0
    zeroBias.roll = preferences.getFloat(zeroBiasKeys[0], 0.0);
    zeroBias.pitch = preferences.getFloat(zeroBiasKeys[1], 0.0);

    // 关闭闪存访问，释放相关资源
    preferences.end();

    // 输出零偏
    DEBUG.print("  Roll Zero Bias: ");
    DEBUG.print(zeroBias.roll);
    DEBUG.print("  Pitch Zero Bias: ");
    DEBUG.println(zeroBias.pitch);
    rollSum = 0;
    pitchSum = 0;
    callCounter = 0;//清零下次用
    CalibrationSelect = 0;//校准完整退出校准

  }

}


// 计算欧拉角零偏的函数
void calculateAccBias() {
  // 初始化累加器
  static float AccSumX = 0;

  // 累加
  AccSumX += imuState.filtered_accel.x;

  // 增加调用计数器
  callCounter++;

  if (callCounter >= CALL_COUNT)
  {
    // 计算平均值，得到零偏
    ConfigurationParameter.AccOffsetX = (float)AccSumX / CALL_COUNT;

    // 初始化闪存访问，打开名为 "preferences" 的命名空间
    // 第二个参数为false，表示以可写模式打开命名空间
    preferences.begin("preferences", false);

    DEBUG.print("  写入数据 AccOffsetX: ");
    DEBUG.println(ConfigurationParameter.AccOffsetX,5);
    //写入数据
    
    preferences.putFloat(zeroBiasKeys[9], ConfigurationParameter.AccOffsetX);
    //读出数据  如果读取失败（即闪存中不存在该键名对应的数据），则使用默认值0.0
    ConfigurationParameter.AccOffsetX = preferences.getFloat(zeroBiasKeys[9], 0.0);

    // 关闭闪存访问，释放相关资源
    preferences.end();

    // 输出零偏
    DEBUG.print("  读出数据 AccOffsetX: ");
    DEBUG.println(ConfigurationParameter.AccOffsetX,5);
    AccSumX = 0;
    callCounter = 0;//清零下次用
    CalibrationSelect = 0;//校准完整退出校准

  }

}



void FlashSave(int sw)
{
  Vector3f currentBias;
  static float servo1_last = zeroBias.servo1;//上一次偏差
  static float servo2_last = zeroBias.servo2;
  static float servo3_last = zeroBias.servo3;
  static float servo4_last = zeroBias.servo4;

  switch (sw)
  {
    case 1:
      // 进行陀螺仪校准
      imu.calibrateGyro(); // 执行陀螺校准

      // 读取并打印当前偏置值
      currentBias = imu.getGyroBias();

      gyroBiasX = currentBias.x;
      gyroBiasY = currentBias.y;
      gyroBiasZ = currentBias.z;
      
      preferences.begin("preferences", false);

      //写入数据
      preferences.putFloat(zeroBiasKeys[2], gyroBiasX);
      preferences.putFloat(zeroBiasKeys[3], gyroBiasY);
      preferences.putFloat(zeroBiasKeys[4], gyroBiasZ);

      //读出数据  如果读取失败（即闪存中不存在该键名对应的数据），则使用默认值0.0
      gyroBiasX = preferences.getFloat(zeroBiasKeys[2], 0.0);
      gyroBiasY = preferences.getFloat(zeroBiasKeys[3], 0.0);
      gyroBiasZ = preferences.getFloat(zeroBiasKeys[4], 0.0);

      // 关闭闪存访问，释放相关资源
      preferences.end();

      // 输出零偏
      DEBUG.print("  gyroBiasX:");
      DEBUG.print(gyroBiasX);
      DEBUG.print("  gyroBiasY:");
      DEBUG.print(gyroBiasY);
      DEBUG.print("  gyroBiasZ:");
      DEBUG.println(gyroBiasZ);


      CalibrationSelect = 0;//校准结束
      break;

    case 2:
      // 计算欧拉角零偏的函数
      calculateZeroBias();
      break;

    case 3:

      preferences.begin("preferences", false);

      if (zeroBias.servo1 != servo1_last) //参数调整过就保存
      {
        servo1_last = zeroBias.servo1;//
        //写入数据
        preferences.putFloat(zeroBiasKeys[5], zeroBias.servo1);
        //读出舵机角度
        zeroBias.servo1 = preferences.getFloat(zeroBiasKeys[5], 0.0);
        //打印数据
        DEBUG.print("  zeroBias.servo1:");
        DEBUG.println(zeroBias.servo1);

      }

      if (zeroBias.servo2 != servo2_last) //参数调整过就保存
      {
        servo2_last = zeroBias.servo2;//
        //写入数据
        preferences.putFloat(zeroBiasKeys[6], zeroBias.servo2);
        //读出舵机角度
        zeroBias.servo2 = preferences.getFloat(zeroBiasKeys[6], 0.0);
        //打印数据
        DEBUG.print("  zeroBias.servo2:");
        DEBUG.println(zeroBias.servo2);

      }

      if (zeroBias.servo3 != servo3_last) //参数调整过就保存
      {
        servo3_last = zeroBias.servo3;//
        //写入数据
        preferences.putFloat(zeroBiasKeys[7], zeroBias.servo3);
        //读出舵机角度
        zeroBias.servo3 = preferences.getFloat(zeroBiasKeys[7], 0.0);
        //打印数据
        DEBUG.print("  zeroBias.servo3:");
        DEBUG.println(zeroBias.servo3);

      }

      if (zeroBias.servo4 != servo4_last) //参数调整过就保存
      {
        servo4_last = zeroBias.servo4;//
        //写入数据
        preferences.putFloat(zeroBiasKeys[8], zeroBias.servo4);
        //读出舵机角度
        zeroBias.servo4 = preferences.getFloat(zeroBiasKeys[8], 0.0);
        //打印数据
        DEBUG.print("  zeroBias.servo4:");
        DEBUG.println(zeroBias.servo4);
      }


      // 关闭闪存访问，释放相关资源
      preferences.end();
      //CalibrationSelect = 0;//手动设置结束校准

      break;

    case 4:
      // 计算ACC零偏
      calculateAccBias();
      break;

    default:

      break;
  }
}


void print_data(void)
{
  switch ((int)Select)
  {
    case 1:
          // 输出欧拉角
          DEBUG.print("dt:");
          DEBUG.print(time_dt,6);
          DEBUG.print(" Roll:");
          DEBUG.print(imuState.roll);
          DEBUG.print(" Pitch:");
          DEBUG.print(imuState.pitch);
          DEBUG.print(" Yaw:");
          DEBUG.println(imuState.yaw);
    
          break;

    case 2:
          // 输出acc
          DEBUG.print("dt:");
          DEBUG.print(time_dt,6);
          DEBUG.print(" accx:");
          DEBUG.print(imuState.filtered_accel.x);
          DEBUG.print(" accy:");
          DEBUG.print(imuState.filtered_accel.y);
          DEBUG.print(" accz:");
          DEBUG.println(imuState.filtered_accel.z);
    
          break;

    case 3:
          // 输出
          DEBUG.print("dt:");
          DEBUG.print(time_dt,6);
          DEBUG.print(" rad x:");
          DEBUG.print(imuState.filtered_gyro.x,4);
          DEBUG.print(" rad y:");
          DEBUG.print(imuState.filtered_gyro.y,4);
          DEBUG.print(" rad z:");
          DEBUG.println(imuState.filtered_gyro.z,4);
    
          break;

    case 4:
          // 输出欧拉角
          DEBUG.print("dt:");
          DEBUG.print(time_dt,6);
          DEBUG.print(" Roll:");
          DEBUG.print(imuState.roll - zeroBias.roll);
          DEBUG.print(" Pitch:");
          DEBUG.print(imuState.pitch - zeroBias.pitch);
          DEBUG.print(" Yaw:");
          DEBUG.println(imuState.yaw - zeroBias.yaw);
    
          break;

    case 5:
          //
          DEBUG.print("dt:");
          DEBUG.print(time_dt,6);
          DEBUG.print(" eRoll:");
          DEBUG.print(zeroBias.roll);
          DEBUG.print(" ePitch:");
          DEBUG.print(zeroBias.pitch);
          DEBUG.print(" eYaw:");
          DEBUG.println(zeroBias.yaw);
    
          break;

    case 6:
          //
          DEBUG.print("dt:");
          DEBUG.print(time_dt,6);
          DEBUG.print(" q0:");
          DEBUG.print(imuState.q0);
          DEBUG.print(" q1:");
          DEBUG.print(imuState.q1);
          DEBUG.print(" q2:");
          DEBUG.print(imuState.q2);
          DEBUG.print(" q3:");
          DEBUG.println(imuState.q3);
   
          break;

    case 7:
          //
          DEBUG.print("dt:");
          DEBUG.print(time_dt,6);
          DEBUG.print(" deg x:");
          DEBUG.print(imuState.filtered_gyro_deg.x,4);
          DEBUG.print(" deg y:");
          DEBUG.print(imuState.filtered_gyro_deg.y,4);
          DEBUG.print(" deg z:");
          DEBUG.println(imuState.filtered_gyro_deg.z,4);
          break;

    case 8:
          //
          for (int i = 0; i < 10; i++)
          {
            DEBUG.print(" ch:"); DEBUG.print(sBus.channels[i]);
          }
    
          DEBUG.print(" sbus_dt_ms:");
          DEBUG.print(sbus_dt_ms);
          DEBUG.println(" ");
          break;

    case 9:
          //
          DEBUG.print(" PP:");
          DEBUG.print(Angle_Pid.Kp);
          DEBUG.print(" PI:");
          DEBUG.print(Angle_Pid.Ki);
          DEBUG.print(" PD:");
          DEBUG.print(Angle_Pid.Kd);
    
          DEBUG.print(" SP:");
          DEBUG.print(Speed_Pid.Kp);
          DEBUG.print(" SI:");
          DEBUG.print(Speed_Pid.Ki);
          DEBUG.print(" SD:");
          DEBUG.print(Speed_Pid.Kd);
    
          DEBUG.print(" YP:");
          DEBUG.print(Yaw_Pid.Kp);
          DEBUG.print(" YI:");
          DEBUG.print(Yaw_Pid.Ki);
          DEBUG.print(" YD:");
          DEBUG.print(Yaw_Pid.Kd);
    
          DEBUG.print("dt:");
          DEBUG.println(time_dt,6);
    
          break;


    case 10:
            //
            DEBUG.print(" twoKp:");
            DEBUG.print(mahonyFilter.twoKp);
            DEBUG.print(" twoKi:");
            DEBUG.print(mahonyFilter.twoKi);
      
            DEBUG.print(" Roll:");
            DEBUG.print(imuState.roll);
            DEBUG.print(" Pitch:");
            DEBUG.print(imuState.pitch);
      
            DEBUG.print("IMUdt:");
            DEBUG.println(IMUtime_dt, 6);
            break;

    case 11:
            //
      
            DEBUG.print(" x:");
            DEBUG.print( angleX);
            DEBUG.print(" y:");
            DEBUG.print( angleY);
            DEBUG.print(" z:");
            DEBUG.print( angleZ);
      
      
            DEBUG.print(" gx:");
            DEBUG.print( angleGyroX);
            DEBUG.print(" gy:");
            DEBUG.print( angleGyroY);
            DEBUG.print(" gz:");
            DEBUG.print( angleGyroZ);
      
      
            DEBUG.print(" IMUdt:");
            DEBUG.println( IMUtime_dt, 6);
      
            break;

    case 12:
            //
      
            DEBUG.print(" x:");
            DEBUG.print( angleX);
            //DEBUG.print(" y:");
            //DEBUG.print( angleY);
      
            DEBUG.print(" Roll:");
            DEBUG.println( imuState.roll);
            //DEBUG.print(" Pitch:");
            //DEBUG.print( imuState.pitch);
      
      
            //DEBUG.print(" IMUdt:");
            //DEBUG.println(IMUtime_dt,6);
      
            break;


    case 13:
            // 输出
            DEBUG.print("dt:");
            DEBUG.print(time_dt,6);
            DEBUG.print(" gyroxf:");
            DEBUG.print(attitude.gyrof.x);
            DEBUG.print(" gyroyf:");
            DEBUG.print(attitude.gyrof.y);
            DEBUG.print(" gyrozf:");
            DEBUG.println(attitude.gyrof.z);
            break;


    case 14:
            // 输出acc
            DEBUG.print("dt:");
            DEBUG.print(time_dt,6);
            DEBUG.print(" accx:");
            DEBUG.print(attitude.accf.x);
            DEBUG.print(" accy:");
            DEBUG.print(attitude.accf.y);
            DEBUG.print(" accz:");
            DEBUG.println(attitude.accf.z);
            break;


    case 15:
            // 输出acc
            //DEBUG.print("dt:");
            //DEBUG.print(time_dt,6);
            DEBUG.print(" accy:");
            DEBUG.print(attitude.acc.y);
            DEBUG.print(" accyf:");
            DEBUG.println(attitude.accf.y);
            break;

    case 16:
            // 输出acc
            //DEBUG.print("dt:");
            //DEBUG.print(time_dt,6);
            DEBUG.print(" gyro:");
            DEBUG.print(attitude.gyro.y);
            DEBUG.print(" gyrof:");
            DEBUG.println(attitude.gyrof.y);
            break;


    case 17:
            
            DEBUG.print(" x:");
            DEBUG.println(body.TargetPosition_x);
            break;

    case 18:
            DEBUG.print(" M5:");
            DEBUG.print(JointMotor[4].Turn_s,4);

            DEBUG.print(" s1:");
            DEBUG.println(body.WheelSpeed_f[0],4);

  
            break;

    case 19:
            //
            DEBUG.print(" servo1:");
            DEBUG.print(zeroBias.servo1);
            DEBUG.print(" servo2:");
            DEBUG.print(zeroBias.servo2);
            DEBUG.print(" servo3:");
            DEBUG.print(zeroBias.servo3);
            DEBUG.print(" servo4:");
            DEBUG.println(zeroBias.servo4);
            break;

    case 20:
    
            DEBUG.print(" vra:");
            DEBUG.print(sbus_vra, 6);
            DEBUG.print(" body.TargetRoll:");
            DEBUG.print(body.TargetRoll, 6);
            DEBUG.print(" LegLength:");
            DEBUG.print(LegLength, 6);
            DEBUG.print(" sc:");
            DEBUG.println(PidCoefficient, 6);
            break;

    case 21:
            DEBUG.print(" roll_ok:");
            DEBUG.print(roll_ok, 6);
            DEBUG.print(" BodyPitching:");
            DEBUG.println(BodyPitching, 6);
            break;


    case 22:
            /*
            DEBUG.print(" PP:");
            DEBUG.print(Angle_Pid.Kp, 5);
            DEBUG.print(" PI:");
            DEBUG.print(Angle_Pid.Ki, 5);
            DEBUG.print(" PD:");
            DEBUG.print(Angle_Pid.Kd, 5);   
            */  
            DEBUG.print(" E:");
            DEBUG.print(Angle_Pid.error*RAD_TO_DEG, 6);
            DEBUG.print(" it:");
            DEBUG.print(Angle_Pid.iLimit, 5);
            DEBUG.print(" il:");
            DEBUG.print(Angle_Pid.integral, 5);      
            DEBUG.print(" oI:");
            DEBUG.print(Angle_Pid.outI, 5);
            DEBUG.print(" out:");
            DEBUG.println(Angle_Pid.output, 5);
            break;


    case 23:
        
            
            DEBUG.print(" T:");
            DEBUG.print(body.TargetMovementSpeed_x_f, 5);
            DEBUG.print(" E:");
            DEBUG.print(Speed_Pid.error, 6);
            DEBUG.print(" it:");
            DEBUG.print(Speed_Pid.iLimit, 5);
            DEBUG.print(" il:");
            DEBUG.print(Speed_Pid.integral, 5);         
            DEBUG.print(" oI:");
            DEBUG.print(Speed_Pid.outI, 5);
            DEBUG.print(" out:");
            DEBUG.println(body.TargetPosition_x, 5);
            break;

    case 24:
            DEBUG.print(" EN:");
            DEBUG.print(enableDFilter);
            DEBUG.print(" HZ:");
            DEBUG.println(cutoffFreq, 5);
            break;

    case 25:
            DEBUG.print(" LpfOut:");
            DEBUG.print(BodyPitching_f, 6);
            DEBUG.print(" BodyPitching:");
            DEBUG.println(BodyPitching, 6);
            break;

    case 26:

            if(Touch.state == 1)
            {
              DEBUG.print("  aX:");
              DEBUG.print(Touch.XPdat); 
              DEBUG.print("  aY:");
              DEBUG.println(Touch.YPdat);          
            }
            else if(Touch.state == 0)
            {
              DEBUG.print("  tX:");
              DEBUG.print(Touch.XLdat); 
              DEBUG.print("  tY:");
              DEBUG.println(Touch.YLdat);           
            }
            break;


    case 27:

            DEBUG.print("  aX:");
            DEBUG.print(Touch.XPdat); 
            DEBUG.print("  aY:");
            DEBUG.print(Touch.YPdat);          
            DEBUG.print("  aXF:");
            DEBUG.print(Touch.XPdatF); 
            DEBUG.print("  aYF:");
            DEBUG.println(Touch.YPdatF);  
            break;

    case 28:

            DEBUG.print("  P:");
            DEBUG.print(BodyPitching_f); 
            DEBUG.print("  R:");
            DEBUG.print(body.TargetRoll_f,5);          
            DEBUG.print("  H:");
            DEBUG.print(LegLength_f,5); 
            DEBUG.print("  S:");
            DEBUG.print(SlideStep_f);  
            DEBUG.print("  vra:");
            DEBUG.print(sbus_vra);
            DEBUG.print("  vra:");
            DEBUG.println(sbus_vrb);
            break;

    case 29:
            DEBUG.print(" Kp:");
            DEBUG.print(TouchY_Pid.Kp,6);
            DEBUG.print(" Ki:");
            DEBUG.print(TouchY_Pid.Ki,6);      
            DEBUG.print(" Kd:");
            DEBUG.print(TouchY_Pid.Kd,6);

            DEBUG.print(" deriv:");
            DEBUG.print(TouchY_Pid.deriv);
            DEBUG.print(" out:");
            DEBUG.println(TouchY_Pid.output);
            break;

    case 30:
            DEBUG.print(" deriv:");
            DEBUG.println(TouchY_Pid.deriv);
            break;

    case 31:
            DEBUG.print(" E:");
            DEBUG.print(Roll_Pid.error, 6);
            DEBUG.print(" it:");
            DEBUG.print(Roll_Pid.iLimit, 5);
            DEBUG.print(" il:");
            DEBUG.print(Roll_Pid.integral, 5);         
            DEBUG.print(" oI:");
            DEBUG.print(Roll_Pid.outI, 5);
            DEBUG.print(" out:");
            DEBUG.println(Roll_Pid.output, 5);
            break;

    case 32:

            DEBUG.print(" RP:");
            DEBUG.print(Roll_Pid.Kp, 6);
            DEBUG.print(" RI:");
            DEBUG.print(Roll_Pid.Ki, 6);
            DEBUG.print(" RD:");
            DEBUG.println(Roll_Pid.Kd, 6);   
            break;
            
    case 33:

            DEBUG.print(" T:");
            DEBUG.print( body.TargetYaw  , 5); 
            DEBUG.print(" E:");
            DEBUG.print(Yaw_Pid.error, 6);
            DEBUG.print(" it:");
            DEBUG.print(Yaw_Pid.iLimit, 5);
            DEBUG.print(" il:");
            DEBUG.print(Yaw_Pid.integral, 5);         
            DEBUG.print(" oI:");
            DEBUG.print(Yaw_Pid.outI, 5);
            DEBUG.print(" out:");
            DEBUG.println(Yaw_Pid.output, 5);
            break;

    case 34:
            DEBUG.print(" Kp:");
            DEBUG.print(TouchX_Pid.Kp,6);
            DEBUG.print(" Ki:");
            DEBUG.print(TouchX_Pid.Ki,6);      
            DEBUG.print(" Kd:");
            DEBUG.print(TouchX_Pid.Kd,6);

            DEBUG.print(" deriv:");
            DEBUG.print(TouchX_Pid.deriv);
            DEBUG.print(" out:");
            DEBUG.println(TouchX_Pid.output);
            break;

    case 35:
            DEBUG.print(" deriv:");
            DEBUG.println(TouchX_Pid.deriv);
            break;

    case 36:
            DEBUG.print(" state:");
            DEBUG.print(Touch.state);
            DEBUG.print(" start:");
            DEBUG.println(Touch.start);            
            break;

    case 37:
            DEBUG.print(" sbus_vra:");
            DEBUG.print(sbus_vra);
            DEBUG.print(" sbus_vraf:");
            DEBUG.print(sbus_vraf);      
            DEBUG.print(" sbus_vrb:");
            DEBUG.print(sbus_vrb);
            DEBUG.print(" sbus_vrbf:");
            DEBUG.println(sbus_vrbf);            
            break;

    case 38:
            DEBUG.print(" X OUT:");
            DEBUG.print(BodyPitching);
            DEBUG.print(" Y OUT:");
            DEBUG.println(TouchY_Pid.output);            
            break;            


    case 39:
            DEBUG.print(" it:");
            DEBUG.print(TouchY_Pid.iLimit, 5);
            DEBUG.print(" il:");
            DEBUG.print(TouchY_Pid.integral, 5);         
            DEBUG.print(" oI:");
            DEBUG.print(TouchY_Pid.outI, 5);
            DEBUG.print(" out:");
            DEBUG.println(TouchY_Pid.output, 5);
            break;
            

    case 40:

            if(Touch.state == 1)
            {
              DEBUG.print("  aX:");
              DEBUG.print(Touch.XPressDat); 
              DEBUG.print("  aY:");
              DEBUG.println(Touch.YPressDat);          
            }
            else if(Touch.state == 0)
            {
              DEBUG.print("  tX:");
              DEBUG.print(Touch.XPressDat); 
              DEBUG.print("  tX:");
              DEBUG.println(Touch.YPressDat);           
            }
            break;

    case 41:

            DEBUG.print(" roll_ok:");
            DEBUG.print(roll_ok, 5);            
            DEBUG.print(" pa:");
            DEBUG.print(BodyPitching, 5);
            DEBUG.print(" out:");
            DEBUG.println(Speed_Pid.output, 5);
            break;

    case 42:

            DEBUG.print(" P:");
            DEBUG.print(roll_ok, 5);            
            DEBUG.print(" P1:");
            DEBUG.print(BodyPitching, 5);
            DEBUG.print(" P3:");
            DEBUG.println(BodyPitchingCorrect(BodyPitching_f), 5);
            break;

    case 43:

            DEBUG.print(" Vdat:");
            DEBUG.print(VoltageADC);            
            DEBUG.print(" Vdatf:");
            DEBUG.print(VoltageADCf);
            DEBUG.print(" V:");
            DEBUG.println(Voltage, 5);
            break;

    case 44:
            DEBUG.print(" TargetLegLength:");
            DEBUG.println(TargetLegLength, 6);
            
            break;

    case 45:

            DEBUG.print(" RobotTumble:");
            DEBUG.print(RobotTumble);   
            DEBUG.print(" roll_ok:");
            DEBUG.print(roll_ok, 6);    
            DEBUG.print(" Angle_Pid.error:");
            DEBUG.println(Angle_Pid.error, 6);
   
            break;

    case 46:
            DEBUG.print(" WT1:");
            DEBUG.print(body.WheelTargetTorque[0], 3);    
            DEBUG.print(" WT2:");
            DEBUG.println(body.WheelTargetTorque[1], 3);   
            break;

    case 47: 
            DEBUG.print(" E:");
            DEBUG.println(Angle_Pid.error*RAD_TO_DEG, 6);
   
            break;    

    case 48: 
            DEBUG.print(" s:");
            DEBUG.print(ROS_body.milemeter.Speed, 5);    
            DEBUG.print(" w:");
            DEBUG.println(ROS_body.milemeter.AngularVelocity, 5);    
            break;  

    case 49: 
            DEBUG.print(" z:");
            DEBUG.print(imuState.filtered_gyro.z*RAD_TO_DEG, 5);    
            DEBUG.print(" w:");
            DEBUG.println(ROS_body.milemeter.AngularVelocity*RAD_TO_DEG, 5);  
            break;   

    case 50:    

            DEBUG.print(" q0:");
            DEBUG.print(ROS_body.ImuData.q0,3); 
            DEBUG.print(" q1:");
            DEBUG.print(ROS_body.ImuData.q1,3); 
            DEBUG.print(" q2:");
            DEBUG.print(ROS_body.ImuData.q2,3); 
            DEBUG.print(" q3:");
            DEBUG.print(ROS_body.ImuData.q3,3); 

            DEBUG.print(" gyroX:");
            DEBUG.print(ROS_body.ImuData.gyro[0],3); 
            DEBUG.print(" gyroY:");
            DEBUG.print(ROS_body.ImuData.gyro[1],3); 
            DEBUG.print(" gyroZ:");
            DEBUG.print(ROS_body.ImuData.gyro[2],3); 

            DEBUG.print(" accelX:");
            DEBUG.print(ROS_body.ImuData.accel[0],3); 
            DEBUG.print(" accelY:");
            DEBUG.print(ROS_body.ImuData.accel[1],3); 
            DEBUG.print(" accelZ:");
            DEBUG.println(ROS_body.ImuData.accel[2],3); 
            break;   

    case 51: 
            DEBUG.print(" mv1:");
            DEBUG.print(body.MotorVelocityF[0],3); 
            DEBUG.print(" mv2:");
            DEBUG.print(body.MotorVelocityF[1],3); 
            DEBUG.print(" mv3:");
            DEBUG.print(body.MotorVelocityF[2],3); 
            DEBUG.print(" mv4:");
            DEBUG.println(body.MotorVelocityF[3],3); 
            break;                                    
      
    case 52: 
    
            DEBUG.print(" E:");
            DEBUG.print(Speed_Pid.error*55, 6);       
            DEBUG.print(" pd:");
            DEBUG.println(body.TargetPitching_differential,5); 
            
            break;   

    case 53: 
            DEBUG.print(" E:");
            DEBUG.print(Speed_Pid.error, 6);       
            DEBUG.print(" pd:");
            DEBUG.print(body.TargetPitching_differential/55,5);     
            DEBUG.print(" T:");
            DEBUG.println((Speed_Pid.error+body.TargetPitching_differential/55),5);     
            

            break;  

    case 54:
            DEBUG.print(" E:");
            DEBUG.print(Pitching_Pid.error, 6);
            DEBUG.print(" it:");
            DEBUG.print(Pitching_Pid.iLimit, 5);
            DEBUG.print(" il:");
            DEBUG.print(Pitching_Pid.integral, 5);         
            DEBUG.print(" oI:");
            DEBUG.print(Pitching_Pid.outI, 5);
            DEBUG.print(" out:");
            DEBUG.println(Pitching_Pid.output, 5);
            break;

    case 55:
            //DEBUG.print("   body.TargetMovementSpeed_x:");
            //DEBUG.print(body.TargetMovementSpeed_x, 3);    
            DEBUG.print("   roll_ok:");
            DEBUG.println(roll_ok, 3);    
            break;

    case 56:
            DEBUG.print(" pp:");
            DEBUG.print(Angle_Pid.Kp, 6);    
            DEBUG.print(" pi:");
            DEBUG.print(Angle_Pid.Ki, 6);    
            DEBUG.print(" pd:");
            DEBUG.print(Angle_Pid.Kd, 6);   
            DEBUG.print(" output:");
            DEBUG.println(Angle_Pid.output, 6);             
            break;

    case 57:
            DEBUG.print (" a1:") ;
            DEBUG.print (body.LeftLegJointTargetAngle[0],3) ;   
            DEBUG.print (" a2:") ;
            DEBUG.print (body.LeftLegJointTargetAngle[1],3) ;     
            DEBUG.print (" a3:") ;
            DEBUG.print (body.RightLegJointTargetAngle[0],3) ;   
            DEBUG.print (" a4:") ;
            DEBUG.println (body.RightLegJointTargetAngle[1],3) ;  
            break;

    case 58:
            DEBUG.print (" p: ") ;
            DEBUG.print (JointMotor[0].Angle,3) ;   
            DEBUG.print (" ts: ") ;
            DEBUG.print (JointMotor[0].Turn_s,3) ; 
            DEBUG.print (" v: ") ;
            DEBUG.print (JointMotor[0].Speed,3) ; 
            DEBUG.print (" t: ") ;
            DEBUG.print (JointMotor[0].Torque,3) ; 
            DEBUG.print (" T: ") ;
            DEBUG.println (JointMotor[0].Temp,3) ;            
            break;
            
    case 59:
            DEBUG.print (" Motor_KP:") ;
            DEBUG.print (Motor_KP,3) ; 
            DEBUG.print (" Motor_KD:") ;
            DEBUG.println (Motor_KD) ;            
            break;

    case 60:
            DEBUG.print ("电机开启状态位 ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].MCStatusword.status.switched_on);
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].MCStatusword.status.switched_on);  
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].MCStatusword.status.switched_on);  
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].MCStatusword.status.switched_on);  
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].MCStatusword.status.switched_on);  
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].MCStatusword.status.switched_on);     
            
            break;

    case 61:
            DEBUG.print ("过压错误 ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].MCStatusword.errors.over_voltage);
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].MCStatusword.errors.over_voltage);  
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].MCStatusword.errors.over_voltage);  
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].MCStatusword.errors.over_voltage);  
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].MCStatusword.errors.over_voltage);  
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].MCStatusword.errors.over_voltage);                 
            break;
    case 62:
            DEBUG.print ("欠压错误 ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].MCStatusword.errors.under_voltage);
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].MCStatusword.errors.under_voltage);  
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].MCStatusword.errors.under_voltage);  
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].MCStatusword.errors.under_voltage);  
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].MCStatusword.errors.under_voltage);  
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].MCStatusword.errors.under_voltage);                 
            break;

    case 63:
            DEBUG.print ("过流错误 ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].MCStatusword.errors.over_current);
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].MCStatusword.errors.over_current);  
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].MCStatusword.errors.over_current);  
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].MCStatusword.errors.over_current);  
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].MCStatusword.errors.over_current);  
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].MCStatusword.errors.over_current);                 
            break;

    case 64:
            DEBUG.print ("ADC自测试失败 ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].MCStatusword.errors.adc_selftest_fatal);
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].MCStatusword.errors.adc_selftest_fatal);  
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].MCStatusword.errors.adc_selftest_fatal);  
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].MCStatusword.errors.adc_selftest_fatal);  
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].MCStatusword.errors.adc_selftest_fatal);  
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].MCStatusword.errors.adc_selftest_fatal);                 
            break;

    case 65:
            DEBUG.print ("编码器离线 ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].MCStatusword.errors.encoder_offline);
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].MCStatusword.errors.encoder_offline);  
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].MCStatusword.errors.encoder_offline);  
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].MCStatusword.errors.encoder_offline);  
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].MCStatusword.errors.encoder_offline);  
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].MCStatusword.errors.encoder_offline);                 
            break;

    case 66:
            DEBUG.print ("电流限制激活标志位 ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].MCStatusword.status.current_limit_active);
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].MCStatusword.status.current_limit_active);  
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].MCStatusword.status.current_limit_active);  
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].MCStatusword.status.current_limit_active);  
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].MCStatusword.status.current_limit_active);  
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].MCStatusword.status.current_limit_active);                 
            break;
    case 67:
            DEBUG.print ("status ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].MCStatusword.status.status_code);
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].MCStatusword.status.status_code);  
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].MCStatusword.status.status_code);  
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].MCStatusword.status.status_code);  
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].MCStatusword.status.status_code);  
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].MCStatusword.status.status_code);                 
            break;
    case 68:
            DEBUG.print ("errors ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].MCStatusword.errors.errors_code);
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].MCStatusword.errors.errors_code);  
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].MCStatusword.errors.errors_code);  
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].MCStatusword.errors.errors_code);  
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].MCStatusword.errors.errors_code);  
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].MCStatusword.errors.errors_code);                 
            break;

    case 69:
            DEBUG.print ("Angle ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].Angle,3) ;  
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].Angle,3) ;    
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].Angle,3) ;    
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].Angle,3) ;    
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].Angle,3) ;    
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].Angle,3) ;            
            break;
    case 70:
            DEBUG.print ("Speed ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].Speed,3) ;  
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].Speed,3) ;    
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].Speed,3) ;    
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].Speed,3) ;    
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].Speed,3) ;    
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].Speed,3) ;            
            break;
    case 71:
            DEBUG.print ("Torque ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].Torque,3) ;  
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].Torque,3) ;    
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].Torque,3) ;    
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].Torque,3) ;    
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].Torque,3) ;    
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].Torque,3) ;            
            break;
    case 72:
            DEBUG.print ("Temp ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].Temp,3) ;  
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].Temp,3) ;    
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].Temp,3) ;    
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].Temp,3) ;    
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].Temp,3) ;    
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].Temp,3) ;            
            break;

    case 73:
            DEBUG.print(" bodyH:");
            DEBUG.print(bodyH,4);  
            DEBUG.print(" body.TargetPosition_x:");
            DEBUG.print(body.TargetPosition_x,4);  
            
            DEBUG.print (" LefX:") ;
            DEBUG.print (body.LeftPlace.x,3) ;  
            DEBUG.print (" LefY:") ;
            DEBUG.print (body.LeftPlace.y,3) ;    
            DEBUG.print (" RightX:") ;
            DEBUG.print (body.RightPlace.x,3) ;    
            DEBUG.print (" RightY:") ;
            DEBUG.print (body.RightPlace.y,3) ;

            DEBUG.print (" LefXp:") ;
            DEBUG.print (body.LeftPlace_p.x,3) ;  
            DEBUG.print (" LefYp:") ;
            DEBUG.print (body.LeftPlace_p.y,3) ;    
            DEBUG.print (" RightXp:") ;
            DEBUG.print (body.RightPlace_p.x,3) ;    
            DEBUG.print (" RightYp:") ;
            DEBUG.println (body.RightPlace_p.y,3) ;      
            break;
            
    case 74:
            DEBUG.print ("Turn_s ") ;
            DEBUG.print (" m1:") ;
            DEBUG.print (JointMotor[0].Turn_s,3) ;  
            DEBUG.print (" m2:") ;
            DEBUG.print (JointMotor[1].Turn_s,3) ;    
            DEBUG.print (" m3:") ;
            DEBUG.print (JointMotor[2].Turn_s,3) ;    
            DEBUG.print (" m4:") ;
            DEBUG.print (JointMotor[3].Turn_s,3) ;    
            DEBUG.print (" m5:") ;
            DEBUG.print (JointMotor[4].Turn_s,3) ;    
            DEBUG.print (" m6:") ;
            DEBUG.println (JointMotor[5].Turn_s,3) ;            
            break;

    case 75:
            DEBUG.print ("mm/s ") ;  
            DEBUG.print (" Motor1_Velocity:") ;
            DEBUG.print (Motor1_Velocity*1000,5) ;    
            DEBUG.print (" Motor1_Velocity_f:") ;
            DEBUG.print (Motor1_Velocity_f*1000,5) ;    
            DEBUG.print (" Motor2_Velocity:") ;
            DEBUG.print (Motor2_Velocity*1000,5) ;    
            DEBUG.print (" Motor2_Velocity_f:") ;
            DEBUG.println (Motor2_Velocity_f*1000,5) ;            
            break;
    case 76:
            DEBUG.print ("mm/s ") ;  
            DEBUG.print (" Motor1_Velocity:") ;
            DEBUG.print (Motor1_Velocity*1000,5);    
            DEBUG.print (" Motor1_Velocity_f:") ;
            DEBUG.println (Motor1_Velocity_f*1000,5) ;             
            break;
            
    case 77:
            DEBUG.print(" 速度: ");
            DEBUG.print(RosComWheelLegged.linear_velocity);
            DEBUG.print(" 角速度: ");
            DEBUG.print(RosComWheelLegged.angle_velocity);
            DEBUG.print(" 俯仰: ");
            DEBUG.print(RosComWheelLegged.pitch_angle);
            DEBUG.print(" 横滚: ");
            DEBUG.print(RosComWheelLegged.roll_angle);
            DEBUG.print(" 腿高: ");
            DEBUG.print(RosComWheelLegged.leg_length);
            DEBUG.print(" count: ");
            DEBUG.print(RosComWheelLegged.count);
            DEBUG.print(" Sign: ");
            DEBUG.println(RosMessageSign);
            
            break;

    case 78:
            DEBUG.print(" m1: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[0].position,5);
            DEBUG.print(" m2: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[1].position,5);
            DEBUG.print(" m3: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[2].position,5);
            DEBUG.print(" m4: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[3].position,5);
            DEBUG.print(" m5: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[4].position,5);
            DEBUG.print(" m6: ");
            DEBUG.println(ROS_body.MIT_Feedback_Data[5].position,5);
        
            break;

    case 79:
            DEBUG.print(" m1: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[0].velocity,6);
            DEBUG.print(" m2: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[1].velocity,6);
            DEBUG.print(" m3: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[2].velocity,6);
            DEBUG.print(" m4: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[3].velocity,5);
            DEBUG.print(" m5: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[4].velocity,6);
            DEBUG.print(" m6: ");
            DEBUG.println(ROS_body.MIT_Feedback_Data[5].velocity,6);
           
            break;

    case 80:
            DEBUG.print(" m1: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[0].torque,5);
            DEBUG.print(" m2: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[1].torque,5);
            DEBUG.print(" m3: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[2].torque,5);
            DEBUG.print(" m4: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[3].torque,5);
            DEBUG.print(" m5: ");
            DEBUG.print(ROS_body.MIT_Feedback_Data[4].torque,5);
            DEBUG.print(" m6: ");
            DEBUG.println(ROS_body.MIT_Feedback_Data[5].torque,5);
        
            break;

    case 81:
             for (int i = 0; i < 10; i++)
             {
              DEBUG.print(" ");
              DEBUG.print(ROS_body.SBUS_Channels_Data[i]);              
             }
             DEBUG.println(" ");
        
            break;

    case 82:
            DEBUG.print(" ROS_body_t: ");
            DEBUG.print(sizeof(ROS_body_t));   
            DEBUG.print(" ROS_body_t1: ");
            DEBUG.println(sizeof(ROS_body_t1));  
            break;

    case 83:
            DEBUG.print(" m1: ");
            DEBUG.print(RX_MIT_Data.MIT_Command_Data[0].position,5);
            DEBUG.print(" m2: ");
            DEBUG.print(RX_MIT_Data.MIT_Command_Data[1].position,5);
            DEBUG.print(" m3: ");
            DEBUG.print(RX_MIT_Data.MIT_Command_Data[2].position,5);
            DEBUG.print(" m4: ");
            DEBUG.print(RX_MIT_Data.MIT_Command_Data[3].position,5);
            DEBUG.print(" m5: ");
            DEBUG.print(RX_MIT_Data.MIT_Command_Data[4].position,5);
            DEBUG.print(" m6: ");
            DEBUG.println(RX_MIT_Data.MIT_Command_Data[5].position,5);
        
            break;


    case 84:
            DEBUG.print(" sF:");
            DEBUG.print((Motor1_Velocity_f + Motor2_Velocity_f) * 0.5 , 6);
            DEBUG.print(" LSF:");
            DEBUG.print(Motor1_Velocity_f, 6);
            DEBUG.print(" RSF:");
            DEBUG.print(Motor2_Velocity_f, 6);

            DEBUG.print(" LS:");
            DEBUG.print(Motor1_Velocity, 6);
            DEBUG.print(" RS:");
            DEBUG.println(Motor2_Velocity, 6);   
            break;

    case 85:
            DEBUG.print(" PayloadWeight:");
            DEBUG.println(ConfigurationParameter.PayloadWeight, 6);
         
            break;


    default:

          
      break;
  }
}


float BodyPitchingCorrect(float x)//俯仰角校正
{
  float y = 0.000004*x*x + 0.0004*x - 0.0008;//y = 4E-06x2 + 0.0004x - 0.0008    y = -2E-07x2 + 0.0002x - 0.0029
  return y;
}

float WrapAngleRad(float angle)
{
  while (angle > PI) angle -= 2.0f * PI;
  while (angle < -PI) angle += 2.0f * PI;
  return angle;
}


void PIDcontroller_posture(float dt)
{
  static int sign = 0;
  if(sign<3)
    sign++;
  
  //触摸屏
  TouchX_Pid.Kp = TouchXPid.P/100;
  TouchX_Pid.Ki = TouchXPid.I/100;
  TouchX_Pid.Kd = TouchXPid.D/100;
  TouchX_Pid.iLimit = TouchXPid.limit;//积分限幅
  
  TouchY_Pid.Kp = TouchYPid.P/100;
  TouchY_Pid.Ki = TouchYPid.I/100;
  TouchY_Pid.Kd = TouchYPid.D/100;
  TouchY_Pid.iLimit = TouchYPid.limit;//积分限幅
  
  float touchXError = Touch.XPdatF;  
  float touchYError = Touch.YPdatF;  
  static float TouchX_kd = 0;
  static float TouchY_kd = 0;

  TouchX_Pid.compute(touchXError, dt); 
  TouchY_Pid.compute(touchYError, dt);   

  TouchX_Pid.deriv = constrain(TouchX_Pid.deriv, -11000, 11000);
  TouchX_Pid.outD = TouchX_kd * TouchX_Pid.deriv;  
   
  if((sbus_swd == 2)||(sbus_swc == 1))//顶球模式 
  {
    BodyPitching = TouchX_Pid.outP + TouchX_Pid.outI + TouchX_Pid.outD + sbus_vraf;
    BodyPitching = -BodyPitching;    
  }  

  TouchY_Pid.deriv = constrain(TouchY_Pid.deriv, -11000, 11000);
  TouchY_Pid.outD = TouchY_kd * TouchY_Pid.deriv;  
  TouchY_Pid.output = TouchY_Pid.outP + TouchY_Pid.outI + TouchY_Pid.outD + sbus_vrbf;
  if((int)enableDFilter==1)
  {
    TouchY_Pid_outputF = biquadFilterApply(&FilterLPF[10], TouchY_Pid.output);    
    TouchX_Pid_outputF = biquadFilterApply(&FilterLPF[11], TouchX_Pid.output);  
  }
  else
  {
    TouchY_Pid_outputF = TouchY_Pid.output;    
    TouchX_Pid_outputF = TouchX_Pid.output;   
  }

  if((Touch.state==1)&&(Touch.P_count<4))
  {
    Touch.P_count++;
    TouchX_Pid.integral = 0;
    //TouchX_Pid.output = 0;
    //BodyPitching = 0;
    TouchY_Pid.integral = 0;
    //TouchY_Pid_outputF = 0;  
  }
    
  if(Touch.P_count>=4)
  {
    if(TouchX_kd<TouchX_Pid.Kd)
    {
      TouchX_kd = TouchX_kd + (TouchX_Pid.Kd/22);
    }
    if(TouchY_kd<TouchY_Pid.Kd)
    {
      TouchY_kd = TouchY_kd + (TouchY_Pid.Kd/22);
    }  
    Touch.start = 2;
  } 

  if((Touch.state==0)&&(Touch.L_count<44))
    Touch.L_count++;
  else
    Touch.L_count = 0;
      
  if(Touch.L_count>=44)
  {
    TouchX_Pid.integral = 0;
    //TouchX_Pid.output = 0;
    //BodyPitching = 0;
    TouchY_Pid.integral = 0;
    //TouchY_Pid_outputF = 0;  
    Touch.P_count = 0;
    TouchX_kd = 0;   
    TouchY_kd = 0; 
    Touch.start = -2;    
  }  
  
  if((sbus_swd != 2)||(sbus_swc != 1))//非顶球模式 
  {
    TouchX_Pid.integral = 0;
    TouchX_Pid.output = 0;
    //BodyPitching = 0;
    TouchY_Pid.integral = 0;
    TouchY_Pid.output = 0;
    TouchY_Pid_outputF = 0; 
    TouchX_kd = 0;   
    TouchY_kd = 0; 
  }



  //横滚角
  Roll_Pid.Kp = RollPid.P;
  Roll_Pid.Ki = RollPid.I;
  Roll_Pid.Kd = RollPid.D;
  Roll_Pid.iLimit = RollPid.limit;//积分限幅

  float TargetBodyRoll = body.TargetRoll_f; //横滚
  if(sbus_swd==2)//顶球禁止手动横滚
    TargetBodyRoll = 0;
  float RollError = -roll_ok + TargetBodyRoll;// - TouchY_Pid_outputF;  
  if(sbus_swc == 1)//横滚调平
  {
      Roll_Pid.compute(RollError, dt);    
  }
  else
  {
      Roll_Pid.output = 0;
      Roll_Pid.integral = 0;
  }


  //速度环
  Speed_Pid.Kp = SpeedPid.P*PidCoefficient;
  Speed_Pid.Ki = SpeedPid.I*PidCoefficient;
  Speed_Pid.Kd = SpeedPid.D;
  Speed_Pid.iLimit = SpeedPid.limit;//积分限幅

  if(Speed_Pid.Kp>SpeedPid.P)
    Speed_Pid.Kp = SpeedPid.P;

  if(Speed_Pid.Ki>SpeedPid.I)
    Speed_Pid.Ki = SpeedPid.I;
    
  //fused_velocity_x (Motor1_Velocity_f + Motor2_Velocity_f) * 0.5
  float speedError = (Motor1_Velocity_f + Motor2_Velocity_f) * 0.5 - body.TargetMovementSpeed_x_f + (body.TargetPitching_differential/ConfigurationParameter.PitchControlCompensation); //测量值减去目标值
  TuneSpeedError = speedError;
  TuneSpeedOutput = 0.0f;
  if(ConfigurationParameter.JumpingState==1)
  {
    float speedAcc = Speed_Pid.compute(-speedError, dt);// + BodyPitchingCorrect(BodyPitching_f);//
    TuneSpeedOutput = speedAcc;
    body.TargetPosition_x = SpeedController(speedAcc);
  }
    
   
  //平衡环
  float Speed = (Motor1_Velocity_f + Motor2_Velocity_f) * 0.5;
  if((abs(Speed)<0.1)&&(abs(body.TargetMovementSpeed_x_f)<0.001)&&(abs(body.TargetYaw)<0.1)&&(LegLength<(ConfigurationParameter.DefaultHeight+0.01))&&(abs(BodyPitching)<0.1))
  {
    AnglePid.P = 4;
  }
  else
  {
    AnglePid.P = 2;    
  }
  

  
  Angle_Pid.Kp = AnglePid.P*PidCoefficient;
  if(Angle_Pid.Kp>AnglePid.P)
    Angle_Pid.Kp = AnglePid.P;

  
  Angle_Pid.iLimit = AnglePid.limit;//积分限幅

  if(sign>2)
  {
    Angle_Pid.Ki = AnglePid.I*PidCoefficient;
    Angle_Pid.Kd = AnglePid.D*PidCoefficient;
    
    if(Angle_Pid.Ki>AnglePid.I)
      Angle_Pid.Ki = AnglePid.I;
      
    if(Angle_Pid.Kd>AnglePid.D)
      Angle_Pid.Kd = AnglePid.D;
       
  }
  else
  {
    Angle_Pid.Ki = 0;
    Angle_Pid.Kd = 0; 
  }

  //

  float angleError = pitch_ok - BodyPitching_f; //测量值减去目标值

  float angleOutput = BalanceController(angleError,BodyPitching_f,dt);//Angle_Pid.compute(angleError, dt);
  TuneAngleErrorDeg = angleError;
  TuneAngleOutput = angleOutput;

  //转向环
  Yaw_Pid.Kp = YawPid.P;
  Yaw_Pid.Kd = YawPid.D;
  Yaw_Pid.iLimit = YawPid.limit;

  // Integrate the bias-corrected yaw rate only while balance control is active.
  // This is a relative heading reference; no magnetometer is required.
  static float yaw_heading = 0.0f;
  static float yaw_target_heading = 0.0f;
  if (YawHoldResetRequested)
  {
    yaw_heading = 0.0f;
    yaw_target_heading = 0.0f;
    YawHoldActive = false;
    YawHoldResetRequested = false;
  }
  yaw_heading = WrapAngleRad(yaw_heading + YawCorrectedRate * dt);

  float yaw_target_rate = body.TargetYaw;
  const bool zero_yaw_command = fabsf(body.TargetYaw) < 0.01f;
  const bool can_hold_heading = HeadingHoldEnabled && zero_yaw_command &&
                                body.JumpingStep == 0;
  if (can_hold_heading)
  {
    if (!YawHoldActive)
      yaw_target_heading = yaw_heading;
    YawHoldActive = true;
    const float heading_error = WrapAngleRad(yaw_target_heading - yaw_heading);
    YawHoldErrorDeg = heading_error * RAD_TO_DEG;
    yaw_target_rate = constrain(HeadingHoldKp * heading_error, -0.30f, 0.30f);

    // A zero-rate gyro bias must never wind up into continuous wheel torque.
    Yaw_Pid.Ki = 0.0f;
    Yaw_Pid.EnablePoints = 0;
    Yaw_Pid.integral = 0.0f;
  }
  else
  {
    YawHoldActive = false;
    YawHoldErrorDeg = 0.0f;
    Yaw_Pid.Ki = YawPid.I;
    Yaw_Pid.EnablePoints = 1;
  }

  float yawError = YawCorrectedRate - yaw_target_rate;//测量值减去目标值 弧度/s
  float yawOutput = Yaw_Pid.compute(yawError, dt);
  TuneYawError = yawError;
  TuneYawOutput = yawOutput;

  body.WheelTargetTorque[0] = angleOutput + yawOutput;
  body.WheelTargetTorque[1] = angleOutput - yawOutput;

  //motor1.target = target1;
  //motor2.target = target2;
  //SetWheelTorque(target1,target2) 
}




void RemoteControlFiltering(void)//遥控器滤波
{
  static int enableDFilter_last = (int)enableDFilter;
  static int cutoffFreq_last = (int)cutoffFreq;

  sbus_vraf = biquadFilterApply(&FilterLPF[8], sbus_vra);
  sbus_vrbf = biquadFilterApply(&FilterLPF[9], sbus_vrb); 

  if((int)enableDFilter==1)
  {
    //BodyPitching_f = updateVelocitySetpoint(&body.PitchAngleSlope,BodyPitching,time_dt);
    BodyPitching_f = biquadFilterApply(&FilterLPF[0], BodyPitching);// 
           
    body.TargetRoll_f = biquadFilterApply(&FilterLPF[1], body.TargetRoll);
    LegLength_f = biquadFilterApply(&FilterLPF[2], LegLength);
    SlideStep_f = biquadFilterApply(&FilterLPF[3], SlideStep);
    body.TargetMovementSpeed_x_f = biquadFilterApply(&FilterLPF[4], body.TargetMovementSpeed_x);
    
    body.TargetPitching_differential = calculateAngularVelocity(BodyPitching_f,time_dt);
  }
  else
  {
    BodyPitching_f = BodyPitching;
    body.TargetRoll_f = body.TargetRoll;
    LegLength_f = LegLength;
    SlideStep_f = SlideStep;    
  }
  

  if(((int)enableDFilter!=enableDFilter_last)||((int)cutoffFreq!=cutoffFreq_last))
  {
    for (int i = 0; i < 6; i++)
    {
      biquadFilterInitLPF(&FilterLPF[i], 100, (unsigned int)cutoffFreq);//遥控器滤波器
      //TouchscreenInit((unsigned int)cutoffFreq);  
    }

    enableDFilter_last = (int)enableDFilter;
    cutoffFreq_last = (int)cutoffFreq;   
    DEBUG.println(" "); 
    DEBUG.println(" ok "); 
  } 
    
}

void ReadVoltage(void)
{
  VoltageADC = analogRead(analogInPin);
  VoltageADCf = biquadFilterApply(&VoltageFilterLPF, VoltageADC);
  Voltage = (float)7.77/813.43*VoltageADCf;
}


void Robot_Tumble(void)
{
    static int x = 0;
    if((abs(pitch_ok)>=35)||(abs(roll_ok)>=35))
    {
      x++;
      if(x>=20)
      {
        x=20;
        RobotTumble=1;//机器摔倒
      }      
    }  
    else
    {
      if((RobotTumble==1)&&(abs(pitch_ok)<=5)&&(abs(roll_ok)<=5))//机器摔倒后扶起
      {
        x--;
        if(x<=0)
        {
          x = 0;
          RobotTumble = 0;  
        }
      }
    }

    if(RobotTumble == 1)//机器摔倒
    {
      bodyH = ConfigurationParameter.DefaultHeight;
      body.TargetRoll = 0;
      body.TargetPosition_x = 0;
      body.TargetRoll = 0;
      BodyPitching_f = 0;
    } 
}




void voltage_Indicator_Light(void)//电压指示灯
{
  LED_count++;
  if(LED_count>=LED_dt)
  {
    LED_count = 0 ;
    if(LED_HL==1)
    {
      digitalWrite(LED_Pin, LOW);   //亮 
      LED_HL = 0;
    }
    else
    {
      digitalWrite(LED_Pin, HIGH);  //不亮 
      LED_HL = 1;
    }    
  }
  
  if(Voltage<=7.4)
    LED_dt = 20;
  else
    LED_dt = 100;
    
    
}

void LPF_Parameter_Init(void)//滤波器调参初始化
{
  if (RATE_HZ !=  RATE_HZ_last)
  {
    //初始化二阶低通滤波
    for (int axis = 0; axis < 6; axis++)
    {
      biquadFilterInitLPF(&ImuFilterLPF[axis], (unsigned int)LPF_CUTOFF_FREQ, (unsigned int)RATE_HZ);
    }
    DEBUG.print(" RATE_HZ:");
    DEBUG.print(RATE_HZ);
    RATE_HZ_last = RATE_HZ;
  }
  if (LPF_CUTOFF_FREQ !=  LPF_CUTOFF_FREQ_last)
  {
    //初始化二阶低通滤波
    for (int axis = 0; axis < 6; axis++)
    {
      biquadFilterInitLPF(&ImuFilterLPF[axis], (unsigned int)LPF_CUTOFF_FREQ, (unsigned int)RATE_HZ);
    }
    DEBUG.print(" LPF_CUTOFF_FREQ:");
    DEBUG.print(LPF_CUTOFF_FREQ);
    LPF_CUTOFF_FREQ_last = LPF_CUTOFF_FREQ;
  }  
}


void Robot_Stop(void)
{
  bodyH = ConfigurationParameter.DefaultHeight;
  body.TargetRoll = 0;
  body.TargetPosition_x = 0;
  body.TargetRoll = 0;
  BodyPitching_f = 0;

  Roll_Pid.output = 0;
  Roll_Pid.integral = 0;

  Angle_Pid.integral = 0;
  Speed_Pid.integral = 0;
  Yaw_Pid.integral = 0;
  YawHoldActive = false;
  YawHoldErrorDeg = 0.0f;
  YawHoldResetRequested = true;
}





void inverse_kinematics_task(void)//运动学逆解任务
{
    
  if(sbus_swc == 1)//手自横滚
    body.SetRoll = Roll_Pid.output;
  else
    body.SetRoll = body.TargetRoll_f*0.0034;//


  body.TargetPitchingAngle = BodyPitching_f;

  if (RightInverseKinematics(body.TargetPosition_x+SlideStep_f, bodyH - body.SetRoll, body.TargetPitchingAngle, body.RightLegJointTargetAngle))
    DEBUG.println("右腿 运动学逆解出错");

  if (LeftInverseKinematics(body.TargetPosition_x-SlideStep_f, bodyH + body.SetRoll, body.TargetPitchingAngle, body.LeftLegJointTargetAngle))
    DEBUG.println("左腿 运动学逆解出错");  

    
}



void Joint_task(void)//电机任务
{
  inverse_kinematics_task();//运动学逆解任务

  if(Motor_KD<0.1)
    Motor_KD = 0.1;
  // 将四个电机的角度设置为....
  SetJointAngles(body.LeftLegJointTargetAngle[0] - ConfigurationParameter.JointAngleDeviation,body.LeftLegJointTargetAngle[1] - ConfigurationParameter.JointAngleDeviation,body.RightLegJointTargetAngle[0] - ConfigurationParameter.JointAngleDeviation,body.RightLegJointTargetAngle[1] - ConfigurationParameter.JointAngleDeviation,Motor_KP,Motor_KD);

}


// 转速(转/秒)转线速度(米/秒)的函数，轮子直径为19厘米
float rotationsPerSecondToMetersPerSecond(float rotationsPerSecond) {
    
    const float wheelDiameter = 0.19;  // 轮子直径(米)
    const float wheelCircumference = wheelDiameter * PI;  // 轮子周长(米)
    
    return rotationsPerSecond * wheelCircumference / 8.0f;  // 线速度 = 转速 × 周长   电机减速比8
}


void Update_Motor_Speed(void)//更新电机速度
{
  body.WheelSpeed_f[0] = biquadFilterApply(&WheelSpeedFilterLPF[0], body.WheelSpeed[0]);//之态补偿后的速度 
  body.WheelSpeed_f[1] = biquadFilterApply(&WheelSpeedFilterLPF[1], body.WheelSpeed[1]); 
  
  Motor1_Velocity = rotationsPerSecondToMetersPerSecond(JointMotor[4].Turn_s);// 转速(转/秒)转线速度(米/秒)的函数，轮子直径为19厘米
  Motor2_Velocity = rotationsPerSecondToMetersPerSecond(JointMotor[5].Turn_s);

  if(abs(Motor1_Velocity)<0.001)
    Motor1_Velocity = 0;

  if(abs(Motor2_Velocity)<0.001)
    Motor2_Velocity = 0;
    


  
  Motor1_Velocity_f = rotationsPerSecondToMetersPerSecond(body.WheelSpeed_f[0]);// 转速(转/秒)转线速度(米/秒)的函数，轮子直径为19厘米
  Motor2_Velocity_f = rotationsPerSecondToMetersPerSecond(body.WheelSpeed_f[1]);

  if(abs(Motor1_Velocity_f)<0.001)
    Motor1_Velocity_f = 0;

  if(abs(Motor2_Velocity_f)<0.001)
    Motor2_Velocity_f = 0;
    

  MilemeterCalculate();//里程计数据
}

float SpeedController(float TargetAcc)//
{
  float x1,y1,x2,y2,TargetX;
  float g = 9.8;

  //TargetAcc =  mapf(sBus.channels[2], SBUS_chMin, SBUS_chMax, -g, g);//加速度

  if(TargetAcc>7)//
    TargetAcc = 7;
  if(TargetAcc<-7)
    TargetAcc = -7;    
  
  //左腿
  x1 = body.LeftPlace_p.x;
  y1 = body.LeftPlace_p.y;

  float Arc = asin(TargetAcc/g);

  if(Arc>0.785)//
    Arc = 0.785;
  if(Arc<-0.785)
    Arc = -0.785;    
  
  float a = ArcToAngle(Arc); 

  x1 = y1*tan(Arc);

  //右腿
  x2 = body.RightPlace_p.x;
  y2 = body.RightPlace_p.y;

  Arc = asin(TargetAcc/g);

  if(Arc>0.785)//限制在±40度以内防止出错
    Arc = 0.785;
  if(Arc<-0.785)
    Arc = -0.785;    
  
  a = ArcToAngle(Arc); 

  x2 = y2*tan(Arc);

  TargetX = (x1 + x2)*0.5;

  if(TargetX>ConfigurationParameter.PostureRestrictions_x)//
    TargetX = ConfigurationParameter.PostureRestrictions_x;
  if(TargetX<-ConfigurationParameter.PostureRestrictions_x)
    TargetX = -ConfigurationParameter.PostureRestrictions_x;   
    
  return TargetX;   
}


//平衡控制器
float BalanceController(float error,float p ,float dt)
{

  float M = ConfigurationParameter.BodyWeight + ConfigurationParameter.PayloadWeight*0.4;//kg
  float g = 9.8;
  float pitch, x, y, x1, y1, x2, y2;//旋转弧度 与旋转后的坐标

  //加入俯仰角正解后的坐标
  float pArc = AngleToArc(p);  //俯仰角
  x = body.LeftPlace.x * cosf(pArc) - body.LeftPlace.y * sinf(pArc);
  y = body.LeftPlace.x * sinf(pArc) + body.LeftPlace.y * cosf(pArc); 
  body.LeftPlace_p.x = x;
  body.LeftPlace_p.y = y;
   
  //左腿杆长lop
  pitch = -AngleToArc(error);  //俯仰角
  x1 = x * cosf(pitch) - y * sinf(pitch);
  y1 = x * sinf(pitch) + y * cosf(pitch);
  float xx = x1 * x1 + y1 * y1;
  float lop = sqrt(xx);

  //加入俯仰角正解后的坐标
  x = body.RightPlace.x * cosf(pArc) - body.RightPlace.y * sinf(pArc);
  y = body.RightPlace.x * sinf(pArc) + body.RightPlace.y * cosf(pArc);
  body.RightPlace_p.x = x;
  body.RightPlace_p.y = y;
    
  //右腿杆长rop
  x2 = x * cosf(pitch) - y * sinf(pitch);
  y2 = x * sinf(pitch) + y * cosf(pitch);
  xx = x2 * x2 + y2 * y2;
  float rop = sqrt(xx);

  float r = (lop + rop) * 0.5;//旋转半径
  float l = (x1 + x2) * 0.5;//力臂
  float J = M * r * r;

  Angle_Pid.compute(-pitch, dt);//计算(弧度)角速度以及补偿和限幅

  Angle_Pid.outP = Angle_Pid.Kp * M * g * l;
  Angle_Pid.outD = Angle_Pid.Kd * Angle_Pid.deriv * J;
  Angle_Pid.outI = Angle_Pid.Ki * Angle_Pid.integral;   
/*
  float v = (Motor1_Velocity_f + Motor2_Velocity_f) * 0.5;
  
  if((abs(body.TargetMovementSpeed_x)<0.01)&&(abs(v)<0.1))//消除控制死区
  {
    Angle_Pid.EnablePoints = 1;//使能积分
    Angle_Pid.outI = Angle_Pid.Ki * Angle_Pid.integral;    
  }
  else
    Angle_Pid.EnablePoints = 0;//停止积分
*/    
    
  // 电机力矩 = 重力力矩 + 角动量 + 补偿
  Angle_Pid.output = Angle_Pid.outP + Angle_Pid.outD + Angle_Pid.outI;

  //输出限幅
  if (Angle_Pid.outputLimit != 0)
  {
    Angle_Pid.output = constrain(Angle_Pid.output, -Angle_Pid.outputLimit, Angle_Pid.outputLimit);
  }

  return Angle_Pid.output;   
}


void LegOffGroundDetection(float TorqueThreshold) 
{

  float t = (JointMotor[0].Torque + JointMotor[1].Torque)*0.5;
  if(t>TorqueThreshold)
    body.LegOffGround[0] = 1;
  else
    body.LegOffGround[0] = 0;

  t = (JointMotor[2].Torque + JointMotor[3].Torque)*0.5;
  if(t>TorqueThreshold)
    body.LegOffGround[1] = 1;
  else
    body.LegOffGround[1] = 0;     

  if((body.LegOffGround[0]==1)||(body.LegOffGround[1]==1))
    body.LegOffGround[2] = 1;
  else if((body.LegOffGround[0]==0)&&(body.LegOffGround[1]==0))
    body.LegOffGround[2] = 0; 
       
    //body.LegOffGround[0] = 1;      
    //body.LegOffGround[1] = 1;
    //body.LegOffGround[2] = 1;  
}

// 将弧度每秒转换为转每秒
float radPerSecToRps(float radPerSec) {
  return radPerSec / (2 * PI);
}


int SpeedCalculation(float dt)
{  
  int error = 0;//坐标设置异常
  float AB = ConfigurationParameter.thigh;// 大腿长度m AB=ED
  float BC = ConfigurationParameter.shank;// 小腿长度m BC=DC

  static float L_aIBC_last = 0;
  static float R_aIBC_last = 0;

  float OA = ConfigurationParameter.MotorSpacing/2.0f; //
  float aOCF = 0;
  float aOCF2 = 0;
  float aAOC = 0;
  float OC = 0;
  float OF = 0;
  float FC = 0;
  float AC = 0;
  float aOAC = 0;
  float aOCA = 0;
  float aBAC = 0;
  float aBAG = 0;
  float OE = OA; //
  float aEOC = 0;
  float EC = 0;
  float aOCE = 0;
  float aOEC = 0;
  float aDEC = 0;
  float aDEH = 0;

//////////////////左腿电机速度姿态修正/////////////////////////////////////////////////////////        
  OF = body.LeftPlace.x;
  FC = body.LeftPlace.y;

  //关节1
  OC = sqrtf(pow(OF, 2) + pow(FC, 2)); //https://zh.numberempire.com/right_triangle_calculator.php

  aOCF = asinf(OF / OC);
  aAOC = AngleToArc(90) + aOCF;

  AC = sqrtf(pow(OA, 2) + pow(OC, 2) - 2 * OA * OC * cos(aAOC)); //https://zh.numberempire.com/arbitrary_triangle_calculator.php?__cf_chl_tk=vr4WMdDsGMhU1V4wG4xVXMisQ2Q6fSgLDuKK6VlnV4o-1734502544-1.0.1.1-CoxTBWpWtqs3CDDFrz5boHxA3INdl0yYHMOzF5kZz8w
  if (AC >= (AB + BC))  //超出结构最大范围
    return error = 1;  

  aOCA = acosf((pow(OC, 2) + pow(AC, 2) - pow(OA, 2)) / (2 * OC * AC));
  aOAC = PI - aOCA - aAOC;
  aBAC = acos((pow(AB, 2) + pow(AC, 2) - pow(BC, 2)) / (2 * AB * AC));
  aBAG = PI - aBAC - aOAC;

  //关节2
  aOCF2 = -aOCF;
  aEOC = AngleToArc(90) + aOCF2;

  EC = sqrtf(pow(OE, 2) + pow(OC, 2) - 2 * OE * OC * cos(aEOC));
  if (EC >= (AB + BC)) //超出结构最大范围
    return error = 2;


  float aABC = acos((pow(AB, 2) + pow(BC, 2) - pow(AC, 2)) / (2 * AB * BC));

  float aIBC = PI -( aABC - aBAG);


  float L_aIBC_rad = (aIBC - L_aIBC_last) / dt;
  L_aIBC_last = aIBC;
  L_aIBC_rad = radPerSecToRps(L_aIBC_rad)*8;//减速比8

  if((error==0)&&(abs(L_aIBC_rad)<5000))
  {
    body.WheelSpeed[0] = JointMotor[4].Turn_s - L_aIBC_rad;  
  }
  else
    return error = 3;


//////////////////右腿电机速度姿态修正/////////////////////////////////////////////////////////           
  OF = body.RightPlace.x;
  FC = body.RightPlace.y;

  //关节1
  OC = sqrtf(pow(OF, 2) + pow(FC, 2)); //https://zh.numberempire.com/right_triangle_calculator.php

  aOCF = asinf(OF / OC);
  aAOC = AngleToArc(90) + aOCF;

  AC = sqrtf(pow(OA, 2) + pow(OC, 2) - 2 * OA * OC * cos(aAOC)); //https://zh.numberempire.com/arbitrary_triangle_calculator.php?__cf_chl_tk=vr4WMdDsGMhU1V4wG4xVXMisQ2Q6fSgLDuKK6VlnV4o-1734502544-1.0.1.1-CoxTBWpWtqs3CDDFrz5boHxA3INdl0yYHMOzF5kZz8w
  if (AC >= (AB + BC))  //超出结构最大范围
    return error = 4;  

  aOCA = acosf((pow(OC, 2) + pow(AC, 2) - pow(OA, 2)) / (2 * OC * AC));
  aOAC = PI - aOCA - aAOC;
  aBAC = acos((pow(AB, 2) + pow(AC, 2) - pow(BC, 2)) / (2 * AB * AC));
  aBAG = PI - aBAC - aOAC;

  //关节2
  aOCF2 = -aOCF;
  aEOC = AngleToArc(90) + aOCF2;

  EC = sqrtf(pow(OE, 2) + pow(OC, 2) - 2 * OE * OC * cos(aEOC));
  if (EC >= (AB + BC)) //超出结构最大范围
    return error = 5;

  aABC = acos((pow(AB, 2) + pow(BC, 2) - pow(AC, 2)) / (2 * AB * BC));

  aIBC = PI -( aABC - aBAG);

  float R_aIBC_rad = (aIBC - R_aIBC_last) / dt;
  R_aIBC_last = aIBC;
  R_aIBC_rad = radPerSecToRps(R_aIBC_rad)*8;//减速比8

  if((error==0)&&(abs(R_aIBC_rad)<5000))
  {
    body.WheelSpeed[1] = JointMotor[5].Turn_s - R_aIBC_rad;
        
  }
  else
    return error = 6;
/*
  DEBUG.print (" m5:") ;
  DEBUG.print (JointMotor[4].Turn_s,4) ;  

  DEBUG.print (" m6:") ;
  DEBUG.print (JointMotor[5].Turn_s,4) ;  

  DEBUG.print (" s1:") ;
  DEBUG.print (body.WheelSpeed_f[0],4) ;  

  DEBUG.print (" s2:") ;
  DEBUG.println (body.WheelSpeed_f[1],4) ;  
*/
  return error;
}

int JumpingButton(int Button)
{
  static int sign = 0;
  int key = 0;
  if((Button==0)&&(sign==0))
    sign = 1;
  else if((Button==1)&&(sign==1))
    sign = 2;
  else if((Button==0)&&(sign==2))
  {
    sign = 0; 
    key = 1;
  }
 
  return key;    
}


float RobotJumping(float min_h,float max_h,float SquattingSpeed,float AscendingVelocity,float dt)
{
  static float set_h = ConfigurationParameter.DefaultHeight;
  static int js = 0;
  Node  LeftCoordinates;
  Node  RightCoordinates; 

  LeftCoordinates.x = body.LeftPlace_p.x;
  LeftCoordinates.y = body.LeftPlace_p.y;

  RightCoordinates.x = body.RightPlace_p.x;
  RightCoordinates.y = body.RightPlace_p.y;

  float h = (LeftCoordinates.y + RightCoordinates.y)*0.5;
    
  switch (body.JumpingStep) 
  {
      case 0: {
        set_h = ConfigurationParameter.DefaultHeight;//
        js = 0;

      } break;
      //蹲下
      case 1: {
        // 计算最大步长（受斜坡率限制）
        float max_step_size = abs(dt * SquattingSpeed);
        float full_step     = min_h - set_h;
        // 限制步长在允许范围内
        float step1          = CLAMP(full_step, -max_step_size, max_step_size);
        // 更新高度设定点
        set_h += step1;    
        
        if(abs(h-min_h)<0.005)
          body.JumpingStep = 2;
       
      } break;
      //起跳
      case 2: {
        float max_step_size = abs(dt * AscendingVelocity);
        float full_step     = max_h - set_h;
        float step1          = CLAMP(full_step, -max_step_size, max_step_size);
        set_h += step1;    
        
        if((abs(h-max_h)<0.005)||(body.LegOffGround[2]==0))
          body.JumpingStep = 3;
          
      } break;
      //缩腿
      case 3: {
        /*
        float max_step_size = abs(dt * AscendingVelocity);
        float full_step     = min_h - set_h;
        float step1          = CLAMP(full_step, -max_step_size, max_step_size);
        set_h += step1;   
        */
        set_h = min_h;
        if((abs(h-min_h)<0.005)||(body.LegOffGround[2]==1))
          body.JumpingStep = 4;  
       
      } break;
      //落地
      case 4: {
        
        if(body.LegOffGround[2]==1)   
        {
          js++;  
          if(js>30)
          {
            //digitalWrite(Buzzer_Pin, HIGH);   //响
            body.JumpingStep = 5;    
          }
        }
        else
        {
          js = 0; 
        }
          
       
      } break;   
      
      //伸腿到遥控中值
      case 5: {
        float max_step_size = abs(dt * SquattingSpeed);
        float full_step     = ConfigurationParameter.DefaultHeight - set_h;
        float step1          = CLAMP(full_step, -max_step_size, max_step_size);
        set_h += step1; 
        
        if(abs(h-ConfigurationParameter.DefaultHeight)<0.015)
        {
           //digitalWrite(Buzzer_Pin, LOW);
           body.JumpingStep = 0;  
        }
          
       
      } break;   

         
      default:
        break;
  }
  if(set_h>ConfigurationParameter.MaxHigh)
    set_h = ConfigurationParameter.MaxHigh;
  if(set_h<ConfigurationParameter.MinLow)
    set_h = ConfigurationParameter.MinLow;  
    
  return set_h;  
}


void WheelTask(void)
{
    if((body.LegOffGround[0]==1)&&(body.LegOffGround[1]==1))
      SetWheelTorque(body.WheelTargetTorque[0],body.WheelTargetTorque[1]);//车轮力矩
    else if((body.LegOffGround[0]==1)&&(body.LegOffGround[1]==0))
      SetWheelTorque(body.WheelTargetTorque[0],0);
    else if((body.LegOffGround[0]==0)&&(body.LegOffGround[1]==1))
      SetWheelTorque(0,body.WheelTargetTorque[1]);
    else
      SetWheelTorque(0,0);  
}


unsigned long LandingTimeMs()
{
  static unsigned long landing_time_ms = millis();
  unsigned long ms = 0;
  if(body.LegOffGround[2]==0)
  {
    landing_time_ms = millis();
  }
  else
  {
    ms = millis()-landing_time_ms;
  }
  return ms;
}


/**
 * 更新速度设定点（使用单个结构体参数）
 * @param controller 指向VelocityController结构体的指针
 * @param measure_period 测量周期
 */
float updateVelocitySetpoint(accelerated_speed* controller,float input, float measure_period) {

    controller->input_velocity = input;
    // 计算速度最大步长
    float max_step_size = abs(measure_period * controller->velocity_ramp_rate);
    float full_step     = controller->input_velocity - controller->vel_setpoint;
    float step          = CLAMP(full_step, -max_step_size, max_step_size);
    
    // 更新速度设定点
    controller->vel_setpoint += step;
    return controller->vel_setpoint; 
}

float calculateAngularVelocity(float currentAngle, float dt) 
{
  static float previousAngle = currentAngle;
  // 防止除以零错误
  if (dt <= 0) {
    return 0.0;
  }
  
  // 计算角度变化量
  float angleDelta = currentAngle - previousAngle;
  
  // 计算角速度（角度变化量 / 时间变化量）
  float angularVelocity = angleDelta / dt;

  previousAngle = currentAngle;
  return angularVelocity;
}


void MilemeterCalculate(void)
{
  ROS_body.milemeter.LeftWheelSpeed = Motor1_Velocity_f;
  ROS_body.milemeter.RightWheelSpeed = Motor2_Velocity_f;
  ROS_body.milemeter.Speed = (ROS_body.milemeter.LeftWheelSpeed + ROS_body.milemeter.RightWheelSpeed) * 0.5;
  ROS_body.milemeter.AngularVelocity = (ROS_body.milemeter.RightWheelSpeed - ROS_body.milemeter.LeftWheelSpeed)/ConfigurationParameter.WheelTrack; 


  ROS_body.MIT_Feedback_Data[0].position = JointMotor[0].Angle;//角度 ±720
  ROS_body.MIT_Feedback_Data[1].position = JointMotor[1].Angle; 
  ROS_body.MIT_Feedback_Data[2].position = JointMotor[2].Angle; 
  ROS_body.MIT_Feedback_Data[3].position = JointMotor[3].Angle;  
  ROS_body.MIT_Feedback_Data[4].position = JointMotor[4].Angle; 
  ROS_body.MIT_Feedback_Data[5].position = JointMotor[5].Angle; 

  ROS_body.MIT_Feedback_Data[0].velocity = JointMotor[0].Speed;//角度每秒(deg/s)
  ROS_body.MIT_Feedback_Data[1].velocity = JointMotor[1].Speed; 
  ROS_body.MIT_Feedback_Data[2].velocity = JointMotor[2].Speed; 
  ROS_body.MIT_Feedback_Data[3].velocity = JointMotor[3].Speed;  
  ROS_body.MIT_Feedback_Data[4].velocity = JointMotor[4].Speed; 
  ROS_body.MIT_Feedback_Data[5].velocity = JointMotor[5].Speed; 

  ROS_body.MIT_Feedback_Data[0].torque = JointMotor[0].Torque;//牛米 Nm
  ROS_body.MIT_Feedback_Data[1].torque = JointMotor[1].Torque; 
  ROS_body.MIT_Feedback_Data[2].torque = JointMotor[2].Torque; 
  ROS_body.MIT_Feedback_Data[3].torque = JointMotor[3].Torque;  
  ROS_body.MIT_Feedback_Data[4].torque = JointMotor[4].Torque; 
  ROS_body.MIT_Feedback_Data[5].torque = JointMotor[5].Torque; 

  ROS_body.MIT_Feedback_Data[0].Temp = JointMotor[0].Temp;//温度℃
  ROS_body.MIT_Feedback_Data[1].Temp = JointMotor[1].Temp; 
  ROS_body.MIT_Feedback_Data[2].Temp = JointMotor[2].Temp; 
  ROS_body.MIT_Feedback_Data[3].Temp = JointMotor[3].Temp;  
  ROS_body.MIT_Feedback_Data[4].Temp = JointMotor[4].Temp; 
  ROS_body.MIT_Feedback_Data[5].Temp = JointMotor[5].Temp; 
  
}

// CRC16校验函数 (Modbus协议常用)
// 参数：data-数据数组，length-数据长度
// 返回值：16位CRC校验值
uint16_t crc16(const uint8_t* data, uint32_t length) 
{
  uint16_t crc = 0xFFFF;  // 初始值
  for (uint32_t i = 0; i < length; i++) {
    crc ^= data[i];       // 与当前字节异或
    // 按位处理
    for (uint8_t j = 0; j < 8; j++) {
      if (crc & 0x0001) { // 最低位为1
        crc >>= 1;
        crc ^= 0xA001;    // 多项式反转值 (0x8005)
      } else {
        crc >>= 1;
      }
    }
  }
  return crc;
}


// 函数：将结构体数据复制到字节数组
// 参数：
//   structData: 源结构体
//   array: 目标字节数组（需提前分配足够空间）
//   arraySize: 数组大小（用于安全检查）
// 返回值：成功复制的字节数，失败返回0
size_t structToArr(const ROS_body_t& structData, uint8_t* array, size_t arraySize) {
  const size_t structSize = sizeof(ROS_body_t);
  
  // 安全检查：数组不为空且大小足够
  if (array == nullptr || arraySize < structSize) {
    return 0;
  }
  
  // 将结构体指针转换为字节指针，逐个复制
  const uint8_t* structBytes = (const uint8_t*)&structData;
  for (size_t i = 0; i < structSize; i++) {
    array[i] = structBytes[i];
  }
  
  return structSize;
}





size_t structToArr1(const ROS_body_t1& structData, uint8_t* array, size_t arraySize) {
  const size_t structSize = sizeof(ROS_body_t1);
  
  // 安全检查：数组不为空且大小足够
  if (array == nullptr || arraySize < structSize) {
    return 0;
  }
  
  // 将结构体指针转换为字节指针，逐个复制
  const uint8_t* structBytes = (const uint8_t*)&structData;
  for (size_t i = 0; i < structSize; i++) {
    array[i] = structBytes[i];
  }
  
  return structSize;
}


// 函数：将字节数组数据复制到结构体
// 参数：
//   array: 源字节数组
//   arraySize: 数组实际大小
//   structData: 目标结构体
// 返回值：成功复制的字节数，失败返回0
size_t arrToStruct(const uint8_t* array, size_t arraySize, ROS_body_t& structData) {
  const size_t structSize = sizeof(ROS_body_t);
  
  // 安全检查：数组不为空且大小足够
  if (array == nullptr || arraySize < structSize) {
    return 0;
  }
  
  // 将数组字节逐个复制到结构体
  uint8_t* structBytes = (uint8_t*)&structData;
  for (size_t i = 0; i < structSize; i++) {
    structBytes[i] = array[i];
  }
  
  return structSize;
}

size_t CodArrToStruct(const uint8_t* array, size_t arraySize, ComWheelLegged_t& structData) {
  const size_t structSize = sizeof(ComWheelLegged_t);
  
  // 安全检查：数组不为空且大小足够
  if (array == nullptr || arraySize < structSize) {
    return 0;
  }
  
  // 将数组字节逐个复制到结构体
  uint8_t* structBytes = (uint8_t*)&structData;
  for (size_t i = 0; i < structSize; i++) {
    structBytes[i] = array[i];
  }
  
  return structSize;
}

size_t JointCodArrToStruct(const uint8_t* array, size_t arraySize, RX_MIT_Data_t& structData) {
  const size_t structSize = sizeof(RX_MIT_Data_t);
  
  // 安全检查：数组不为空且大小足够
  if (array == nullptr || arraySize < structSize) {
    return 0;
  }
  
  // 将数组字节逐个复制到结构体
  uint8_t* structBytes = (uint8_t*)&structData;
  for (size_t i = 0; i < structSize; i++) {
    structBytes[i] = array[i];
  }
  
  return structSize;
}


void RosSerial_TXD(void)
{ 
  const size_t dataSize = sizeof(ROS_body_t);

  uint8_t dataArray[dataSize];  // 用于存储转换后的字节数据
  
  ROS_body.header1 = 123;
  ROS_body.header2 = 45;
  ROS_body.length = dataSize;
  ROS_body.count++;
  if(ROS_body.count>255)
    ROS_body.count=0;
    
  size_t copied = structToArr(ROS_body, dataArray, dataSize);//将结构体数据复制到字节数组
  if (copied > 0) 
  {
    ROS_body.crc = crc16(dataArray, dataSize-2);
    
    //DEBUG.print (" crc 0x") ;
    //DEBUG.println (ROS_body.crc, HEX) ; 
    
    copied = structToArr(ROS_body, dataArray, dataSize);//将结构体数据复制到字节数组

    if (copied > 0) 
    {
      // UART0 runs at only 230400 baud through the CH340K. Never wait for
      // UART transmission inside the 100 Hz balance-control loop. If the
      // asynchronous TX ring buffer is temporarily full, drop this telemetry
      // frame; the next 50 Hz frame will contain fresh state.
      if (RosSerial.availableForWrite() >= (int)dataSize)
      {
        RosSerial.write(dataArray, dataSize);//非阻塞发送数据
      }

    }
    else
    {
      DEBUG.println("结构体转数组失败！");

    }
  } 
  else 
  {
    DEBUG.println("结构体转数组失败！");

  }
}


void RosSerial_TXD1(void)
{ 
  const size_t dataSize = sizeof(ROS_body_t1);

  uint8_t dataArray[dataSize];  // 用于存储转换后的字节数据
  
  ROS_body1.header1 = 123;
  ROS_body1.header2 = 45;
  ROS_body1.length = dataSize;
  ROS_body1.count++;
  if(ROS_body1.count>255)
    ROS_body1.count=0;
    
  size_t copied = structToArr1(ROS_body1, dataArray, dataSize);//将结构体数据复制到字节数组
  if (copied > 0) 
  {
    ROS_body1.crc = crc16(dataArray, dataSize-2);
    
    //DEBUG.print (" crc1 0x") ;
    //DEBUG.println (ROS_body1.crc, HEX) ; 
    
    copied = structToArr1(ROS_body1, dataArray, dataSize);//将结构体数据复制到字节数组

    if (copied > 0) 
    {
      // Keep the alternate telemetry path non-blocking as well.
      if (RosSerial.availableForWrite() >= (int)dataSize)
      {
        RosSerial.write(dataArray, dataSize);//非阻塞发送数据
      }

    }
    else
    {
      DEBUG.println("1结构体转数组失败！");

    }
  } 
  else 
  {
    DEBUG.println("1结构体转数组失败！");

  }
}


void Ros_MobileChassis_RXD(void) //读串口1数据
{
  constexpr size_t kMaxFrameSize = 64;
  static uint8_t dataArray[kMaxFrameSize] = {};
  static size_t count = 0;
  static uint16_t frame_length = 0;
  static uint8_t last_control_count = 0;
  static uint8_t repeated_control_count = 0;

  auto reset_parser = [&]() {
    count = 0;
    frame_length = 0;
  };

  while (RosSerial.available())
  {
    const uint8_t dat = (uint8_t)RosSerial.read();
    if (count == 0)
    {
      if (dat == 0x7B)
        dataArray[count++] = dat;
      continue;
    }
    if (count == 1)
    {
      if (dat == 0x2D || dat == 0x2F)
        dataArray[count++] = dat;
      else if (dat == 0x7B)
        dataArray[0] = dat;
      else
        reset_parser();
      continue;
    }
    if (count >= kMaxFrameSize)
    {
      reset_parser();
      continue;
    }

    dataArray[count++] = dat;
    if (count == 4)
    {
      frame_length = (uint16_t)dataArray[2] |
                     ((uint16_t)dataArray[3] << 8);
      const bool known_frame =
          (dataArray[1] == 0x2D &&
           frame_length == sizeof(ComWheelLegged_t)) ||
          (dataArray[1] == 0x2F &&
           frame_length == sizeof(PidTuneCommand_t));
      if (!known_frame || frame_length < 6 || frame_length > kMaxFrameSize)
      {
        reset_parser();
        continue;
      }
    }
    if (frame_length == 0 || count < frame_length)
      continue;

    uint16_t received_crc = 0;
    memcpy(&received_crc, dataArray + frame_length - 2, sizeof(received_crc));
    const uint16_t calculated_crc = crc16(dataArray, frame_length - 2);
    if (received_crc != calculated_crc)
    {
      RX_error++;
      reset_parser();
      continue;
    }

    if (dataArray[1] == 0x2F)
    {
      PidTuneCommand_t tune_command;
      memcpy(&tune_command, dataArray, sizeof(tune_command));
      ProcessPidTuneCommand(tune_command);
    }
    else
    {
      ComWheelLegged_t control_command;
      memcpy(&control_command, dataArray, sizeof(control_command));
      if (control_command.count == last_control_count)
      {
        if (repeated_control_count < 100)
          repeated_control_count++;
      }
      else
      {
        repeated_control_count = 0;
      }
      last_control_count = control_command.count;

      if (repeated_control_count < 100)
      {
        RosComWheelLegged = control_command;
        RosComWheelLegged.linear_velocity = constrainFloat(
            RosComWheelLegged.linear_velocity,
            -ConfigurationParameter.LimitLinearVelocity,
            ConfigurationParameter.LimitLinearVelocity);
        RosComWheelLegged.angle_velocity = constrainFloat(
            RosComWheelLegged.angle_velocity,
            -ConfigurationParameter.LimitAngleVelocity,
            ConfigurationParameter.LimitAngleVelocity);
        RosComWheelLegged.leg_length = constrainFloat(
            RosComWheelLegged.leg_length,
            ConfigurationParameter.MinLow,
            ConfigurationParameter.MaxHigh);
        RosComWheelLegged.pitch_angle = constrainFloat(
            RosComWheelLegged.pitch_angle,
            -ConfigurationParameter.LimitPitchAngle,
            ConfigurationParameter.LimitPitchAngle);
        RosComWheelLegged.roll_angle = constrainFloat(
            RosComWheelLegged.roll_angle,
            -ConfigurationParameter.LimitRollAngle,
            ConfigurationParameter.LimitRollAngle);
        RosComWheelLegged.jump_min_height = constrainFloat(
            RosComWheelLegged.jump_min_height,
            0.16f,
            0.24f);
        RosComWheelLegged.jump_max_height = constrainFloat(
            RosComWheelLegged.jump_max_height,
            0.24f,
            0.34f);
        if (RosComWheelLegged.jump_max_height <
            RosComWheelLegged.jump_min_height + 0.04f)
        {
          RosComWheelLegged.jump_max_height =
              RosComWheelLegged.jump_min_height + 0.04f;
        }
        RosMessageSign = 1;
      }
      else
      {
        InitRosData();
        RosMessageSign = 0;
      }
    }
    reset_parser();
  }
}

void RxtErrorHz(void)
{
    static unsigned long ms = millis();
    unsigned long tt = millis() - ms;
    if(tt>1000)
    {
        ms = millis();   
        RX_error_hz = RX_error;
        RX_error = 0;
    }
}

void Ros_JointData_RXD(void) //读串口1数据
{
  static unsigned long ms = millis();
  
  static unsigned char count = 0;       // 接收字节计数
  static unsigned char recstatu = 0;    // 接收状态标志
  
  const size_t dataSize = sizeof(RX_MIT_Data_t);
  static uint8_t dataArray[dataSize];   // 接收数据缓冲区

  RxtErrorHz();  
  while(RosSerial.available())
  { 
    unsigned char dat = RosSerial.read();
    //DEBUG.println(dat);
    if ((count == 0) && (dat == 123))
    {
      dataArray[count] = dat;
      count = 1;
    }
    else if ((count == 1) && (dat == 45))
    {
      dataArray[count] = dat;
      recstatu = 1;
      count = 2;
    }
    else if (recstatu == 1) //头字节正确
    {
      dataArray[count] = dat;
      count++;
      if (count >= dataArray[2])
      {
        uint16_t crc = crc16(dataArray, dataSize-2);//CRC校验数组
        RX_MIT_Data_t ComRX_MIT_Data;
        size_t copied = JointCodArrToStruct(dataArray, dataSize, ComRX_MIT_Data);//数组转结构体
        if (copied > 0) 
        {
          if (crc == ComRX_MIT_Data.crc)//校验值与接收值对比
          {
            //数据接收成功
            unsigned long tt = millis()-ms;
            ms = millis();
            RX_MIT_Data = ComRX_MIT_Data;

            //DEBUG.print(" RX_error: ");
            //DEBUG.print(RX_error);  
            //DEBUG.print(" RX_error_hz: ");
            //DEBUG.print(RX_error_hz);  
            //DEBUG.print(" ms: ");
            //DEBUG.println(tt);   
          }
          else
          {
            RX_error++;
            //DEBUG.println("CRC校验失败！");
          }

        } 
        else 
        {
          RX_error++;
          DEBUG.println("数组转结构体失败！");
        }


        dataArray[0] = 0;
        dataArray[1] = 0;
        dataArray[2] = 0;
        recstatu = 0;
        count = 0;
       
      }
    }
    else
    {
      dataArray[0] = 0;
      dataArray[1] = 0;
      dataArray[2] = 0;
      recstatu = 0;
      count = 0;
      dat = 0;
    }
  }
}




void MobileChassisTask(void)
{
  //body.LegOffGround[2]=1;
  if((sbus_swa == 0)||(RobotTumble == 1)||(body.LegOffGround[2]==0))//遥控关闭或者机器摔倒
  {
    ROS_body.Status.status.switched_on = 0;
    Robot_Stop();//机器人停止
    body.WheelTargetTorque[0] = 0;
    body.WheelTargetTorque[1] = 0;
  }
  else if((sbus_swa >= 1)&&(RobotTumble == 0)&&(Motor_KP>=5))//遥控启动且机器没摔倒
  {

    ROS_body.Status.status.switched_on = 1;
    PIDcontroller_posture(time_dt);//PID控制器
  
  }      

  WheelTask();//
  
  // The ROS jump request uses bit 0 as a rising-edge, one-shot command.  Track
  // the edge even outside ROS mode so switching SWA cannot replay an old bit.
  static bool last_ros_jump_request = false;
  const bool ros_jump_request = (RosComWheelLegged.set & 0x0001U) != 0;
  const bool ros_jump_rising = ros_jump_request && !last_ros_jump_request;
  last_ros_jump_request = ros_jump_request;

  if((sbus_swa==1)&&(body.LegOffGround[2]==1)&&(Model_Selection==0))//工作在遥控模式&&机器双脚在地面&&跳跃版机型 
  {
    if(JumpingButton(sbus_swb)==1)//检测起跳
    {
      body.JumpingStep = 1;//起跳开始
        
    }      
  }
  else if((sbus_swa==2) && ros_jump_rising &&
          (body.LegOffGround[2]==1) && (Model_Selection==0) &&
          (body.JumpingStep==0))
  {
    // ROS 带速跳跃：与遥控一样，只要双轮着地且当前没在跳，就接受 /cmd_jump 上升沿。
    // 不再要求线速度/角速度接近零，否则只能原地跳，上不了越障台阶。
    body.JumpingStep = 1;
  }
  
  const bool use_ros_low_jump =
      (sbus_swa == 2) &&
      (RosMessageSign == 1) &&
      ((RosComWheelLegged.set & ROS_SET_JUMP_PROFILE) != 0);

  float jump_min_h = use_ros_low_jump
      ? RosComWheelLegged.jump_min_height
      : JumpDefaultMinHeight;
  float jump_max_h = use_ros_low_jump
      ? RosComWheelLegged.jump_max_height
      : JumpDefaultMaxHeight;
  float jump_squat_speed = use_ros_low_jump
      ? JumpLowSquattingSpeed
      : JumpDefaultSquattingSpeed;
  float jump_ascending_velocity = use_ros_low_jump
      ? JumpLowAscendingVelocity
      : JumpDefaultAscendingVelocity;
  float jump_kp_prepare = use_ros_low_jump
      ? JumpLowKpPrepare
      : JumpDefaultKpPrepare;
  float jump_kp_landing = use_ros_low_jump
      ? JumpLowKpLanding
      : JumpDefaultKpLanding;
  float jump_kp_takeoff = use_ros_low_jump
      ? JumpLowKpTakeoff
      : JumpDefaultKpTakeoff;

  if (!use_ros_low_jump && Model_Selection == 0)
  {
    jump_min_h = ConfigurationParameter.ParallelHigh;
    jump_max_h = ConfigurationParameter.MaxHigh;
  }

  float th = RobotJumping(
      jump_min_h,
      jump_max_h,
      jump_squat_speed,
      jump_ascending_velocity,
      time_dt);

  if(body.JumpingStep==0)
  {
    if(TargetLegLength==0)
      bodyH = LegLength_f;//遥控设置腿长
    else
      bodyH = TargetLegLength;//串口设腿长目标值        
  }
  else
  {   if((body.JumpingStep==4)||(body.JumpingStep==5)||(body.JumpingStep==1))
      {
        Motor_KP = jump_kp_prepare; //下蹲与起来
        if(body.JumpingStep==4) 
          Motor_KP = jump_kp_landing;//落地
      }
      else
        Motor_KP = jump_kp_takeoff;//起跳与缩腿
      bodyH = th;
  }
  
  if((body.JumpingStep == 0)||(body.JumpingStep==5)||(body.JumpingStep==1))
  {
      Speed_Pid.EnablePoints = 1;//继续积分
      ConfigurationParameter.JumpingState = 1;
  }  
  else 
  {
      Speed_Pid.EnablePoints = 0;//停止积分
      ConfigurationParameter.JumpingState = 0;       
  }

  
  Joint_task();//关节任务
  LegOffGroundDetection(0);
  
  JointAngles jointAngles;
  jointAngles.alphaLeft = JointMotor[1].Angle;   // 左腿α角度
  jointAngles.betaLeft = JointMotor[0].Angle;     // 左腿β角度
  jointAngles.alphaRight = JointMotor[3].Angle;  // 右腿α角度
  jointAngles.betaRight = JointMotor[2].Angle;   // 右腿β角度
  forwardKinematics(&body.LeftPlace,&body.RightPlace,&jointAngles);//运动学正解函数
    
}

void JointTask(void)
{
  if(sbus_swa==0)
  {
    for (int i = 0; i < 6; i++)
    {
      JointMotor[i].kp = 0;
      JointMotor[i].kd = 0.5;
      JointMotor[i].in_p_des = 0;
      JointMotor[i].in_v_des = 0;
      JointMotor[i].in_torque = 0;
    } 
    ROS_body.Status.status.switched_on = 0;   
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      JointMotor[i].kp = RX_MIT_Data.MIT_Command_Data[i].kp;
      JointMotor[i].kd = RX_MIT_Data.MIT_Command_Data[i].kd;
      JointMotor[i].in_p_des = RX_MIT_Data.MIT_Command_Data[i].position-ConfigurationParameter.JointAngleDeviation;
      JointMotor[i].in_v_des = RX_MIT_Data.MIT_Command_Data[i].velocity;
      JointMotor[i].in_torque = RX_MIT_Data.MIT_Command_Data[i].torque_ff;
    }
    for (int i = 4; i < 6; i++)
    {
      JointMotor[i].kp = RX_MIT_Data.MIT_Command_Data[i].kp;
      JointMotor[i].kd = RX_MIT_Data.MIT_Command_Data[i].kd;
      JointMotor[i].in_p_des = RX_MIT_Data.MIT_Command_Data[i].position;
      JointMotor[i].in_v_des = RX_MIT_Data.MIT_Command_Data[i].velocity;
      JointMotor[i].in_torque = RX_MIT_Data.MIT_Command_Data[i].torque_ff;
    }
    
    ROS_body.Status.status.switched_on = 1;      
  }
  
 
  MitSetJointAngles();
}
    

void DistanceFiltering(float dt)
{
  // 卡尔曼滤波器参数（针对X方向速度）
  static float velocity_x = 0.0;      // X方向速度估计
  static float bias_accel_x = 0.0;    // X加速度计零偏
  static float P_00 = 1.0, P_01 = 0.0, P_10 = 0.0, P_11 = 1.0; // 协方差矩阵
  static float Q_velocity = 0.001;    // 过程噪声协方差（速度）
  static float Q_bias = 0.003;        // 过程噪声协方差（零偏）
  static float R_velocity = 0.1;      // 测量噪声协方差

  
  static float x_v = 0;
  
  float g = 9.81;
  float roll = imuState.roll - zeroBias.roll;
  float pitch = imuState.pitch - zeroBias.pitch;
  
  float g_x = g * sinf(AngleToArc(pitch)) * cosf(AngleToArc(roll)) - ConfigurationParameter.AccOffsetX;

  float a_x = (imuState.filtered_accel.x-ConfigurationParameter.AccOffsetX)*g + g_x;

  body.barycenter_p.x = (body.LeftPlace_p.x + body.RightPlace_p.x)*0.5;
  body.barycenter_p.y = (body.LeftPlace_p.y + body.RightPlace_p.y)*0.5;

  body.barycenter_p_v.x = (body.barycenter_p.x - body.barycenter_last.x)/dt;
  body.barycenter_p_v.y = (body.barycenter_p.y - body.barycenter_last.y)/dt;

  body.barycenter_last.x = body.barycenter_p.x;
  body.barycenter_last.y = body.barycenter_p.y;

  
  x_v = x_v + a_x*dt;

  if(sbus_swd==1)
    x_v=0;


  // 预测步骤 (使用IMU加速度数据)
  velocity_x += dt * (a_x - bias_accel_x);

  
  // 更新协方差矩阵
  P_00 += dt * (dt * P_11 - P_01 - P_10 + Q_velocity);
  P_01 -= dt * P_11;
  P_10 -= dt * P_11;
  P_11 += Q_bias * dt;
  
  // 更新步骤 (使用里程计速度测量)
  float innovation = ROS_body.milemeter.Speed - velocity_x; // 测量残差
  float S = P_00 + R_velocity; // 创新协方差
  float K_0 = P_00 / S; // 卡尔曼增益 (速度)
  float K_1 = P_10 / S; // 卡尔曼增益 (零偏)
  
  // 更新状态估计
  velocity_x += K_0 * innovation;
  bias_accel_x += K_1 * innovation;
  
  // 更新估计协方差
  P_00 -= K_0 * P_00;
  P_01 -= K_0 * P_01;
  P_10 -= K_1 * P_00;
  P_11 -= K_1 * P_01;
  
  // 更新融合后的速度
  fused_velocity_x = velocity_x;

  
  /*
  DEBUG.print (" LefXp:") ;
  DEBUG.print (body.LeftPlace_p.x,3) ;  
  DEBUG.print (" LefYp:") ;
  DEBUG.print (body.LeftPlace_p.y,3) ;    
  DEBUG.print (" Xp:") ;
  DEBUG.print (body.barycenter_p.x,3) ;    
  DEBUG.print (" Yp:") ;
  DEBUG.print (body.barycenter_p.y,3) ; 
*/
/*
  DEBUG.print(" v:");
  DEBUG.print(ROS_body.milemeter.Speed, 5);   
  
  DEBUG.print(" K_x:");
  DEBUG.println(fused_velocity_x,5);
*/
  //DEBUG.print(" a_x:");
  //DEBUG.print(a_x,5);
  
  //DEBUG.print(" AccX:");
  //DEBUG.print((imuState.filtered_accel.x-ConfigurationParameter.AccOffsetX)*g,5);

  //DEBUG.print(" g_x:");
  //DEBUG.println(g_x,5);    
}


void TemperatureAlarm(float Temp)
{
  static int t = 100;
  static int i = 0;
  static int x = 0;
  if((JointMotor[0].Temp>Temp)||(JointMotor[1].Temp>Temp)||(JointMotor[2].Temp>Temp)||(JointMotor[3].Temp>Temp)||(JointMotor[4].Temp>Temp)||(JointMotor[5].Temp>Temp))
  {
    x++;
    if(x>t)
    {
      if(i==0)
      {
        digitalWrite(Buzzer_Pin, HIGH);   //响
        i=1;
        t=10;
      }
      else
      {
        digitalWrite(Buzzer_Pin, LOW);  
        i=0;
        t=100;
      } 
      x = 0;   
    }    
  }
  else
  {
    digitalWrite(Buzzer_Pin, LOW);  
    i = 0;    
    x = 0;
  }

}


void setup() 
{
  // UART0 uses its default pins (GPIO43/44) connected to the Type-C CH340K.
  // CH340K supports up to 230400 baud.
  // HardwareSerial has no TX ring buffer by default. A 512-byte buffer makes
  // the 182-byte telemetry write asynchronous so it cannot stall balance PID.
  RosSerial.setTxBufferSize(512);
  RosSerial.begin(230400);
  // Keep tuning and diagnostic output on the former ROS port (GPIO4/5).
  DEBUG.begin(1000000, SERIAL_8N1, RXD1, TXD1);
  diagnostic_boot_stage = BOOT_STAGE_STARTING;
  RosSerial_Diagnostics_TXD(true);
  FlashInit();//读取闪存数据
  pinMode(LED_Pin, OUTPUT);
  digitalWrite(LED_Pin, LOW);   //亮
  pinMode(Buzzer_Pin, OUTPUT);
  digitalWrite(Buzzer_Pin, LOW);   //不响
  pinMode(buttonPin, INPUT_PULLUP);

  body.JumpingStep = 0;//跳跃空闲
  body.PitchAngleSlope.velocity_ramp_rate=30;

  
  //初始化二阶低通滤波
  for (int axis = 0; axis < 6; axis++)
  {
    biquadFilterInitLPF(&ImuFilterLPF[axis], (unsigned int)LPF_CUTOFF_FREQ, (unsigned int)RATE_HZ);
  }
  
  biquadFilterInitLPF(&VoltageFilterLPF,50.0f, 100);//电压滤波函数初始化
  biquadFilterInitLPF(&WheelSpeedFilterLPF[0],50.0f, 1000);
  biquadFilterInitLPF(&WheelSpeedFilterLPF[1],50.0f, 1000);

  diagnostic_boot_stage = BOOT_STAGE_MOTOR_INIT;
  RosSerial_Diagnostics_TXD(true);
  MotorInit();//电机初始化
  GetAllMotorStatusWord();//上电时读取一次六个电机状态字

  ConfigurationParameterInit();//配置参数初始化
  InitPidTuning();//捕获本次开机的稳定参数，调参仅在RAM中生效
 

  diagnostic_boot_stage = BOOT_STAGE_IMU_INIT;
  RosSerial_Diagnostics_TXD(true);
  while (!imu.begin()) { // 初始化IMU
      SetDiagnosticBootFault(BOOT_STAGE_IMU_FAULT, DIAG_FAULT_IMU);
      RosSerial_Diagnostics_TXD();
      DEBUG.println("Sensor Initialization Failed, Retrying...");
      digitalWrite(Buzzer_Pin, HIGH);   //响
      delay(200);
      digitalWrite(Buzzer_Pin, LOW);  
      delay(200);   
  }
  // A retry may recover after the cable/power settles; only keep faults that
  // are still present rather than reporting a transient IMU retry forever.
  diagnostic_boot_fault_bits &= ~DIAG_FAULT_IMU;
  imu.setMountingZ90();  // 若IMU绕Z轴旋转90度安装
  //imu.calibrateGyro();                  // 执行陀螺校准
  //遥控器
  diagnostic_boot_stage = BOOT_STAGE_REMOTE_INIT;
  sBus.begin();
  InitRosData();   
  for (int i = 0; i < 6; i++)
    biquadFilterInitLPF(&FilterLPF[i], 100, (unsigned int)cutoffFreq);//遥控器滤波器

 
  biquadFilterInitLPF(&FilterLPF[0], 100, (unsigned int)cutoffFreq);//遥控器滤波器 
  biquadFilterInitLPF(&FilterLPF[8], 50, (unsigned int)cutoffFreq);//遥控器滤波器
  biquadFilterInitLPF(&FilterLPF[9], 50, (unsigned int)cutoffFreq);//遥控器滤波器 
  biquadFilterInitLPF(&FilterLPF[10], 200, (unsigned int)400);//
  biquadFilterInitLPF(&FilterLPF[11], 200, (unsigned int)400);//
  biquadFilterInitLPF(&FilterLPF[12], 50, (unsigned int)cutoffFreq);//遥控器滤波器 
  


  command_init();//调试命令初始化

  for (int i = 0; i < 200; i++) 
  {
    ReadVoltage();  // 调用函数读取电压
    delay(1);  // 
   }
   
  diagnostic_boot_stage = BOOT_STAGE_WAIT_REMOTE_RESET;
  while(1)
  {
    int sign = RXsbus(); //更新遥控器数据
    delay(10);
    RosSerial_Diagnostics_TXD();
    DEBUG.println (" 等待遥控器复位 ") ;
    sbus_swa      =  map(sBus.channels[4], SBUS_chMin, SBUS_chMax, 0, 2);
    if((sbus_swa!=2)&&(sbus_swb==0)&&(sbus_swc==0)&&(sbus_swd==0)&&(Motor_KP==0)&&(Motor_KD==0)&&(sign>20))
    {
      robot_mode = 0; 
      ROS_body.Status.status.model = 0;
      DEBUG.println (" 移动底盘模式 ") ;
      break; 
    }
    else
    {
      if((sbus_swa==2)&&(sbus_swb==0)&&(sbus_swc==0)&&(sbus_swd==0)&&(Motor_KP==0)&&(Motor_KD==0)&&(sign>20))
      {
        robot_mode = 1; 
        ROS_body.Status.status.model = 1;
        DEBUG.println (" 电机模式 ") ;
        break; 
      } 
      else if(sign>20)   
      {
        digitalWrite(Buzzer_Pin, HIGH);   //响
        delay(100);
        digitalWrite(Buzzer_Pin, LOW);  
        delay(900);       
      }
        
    } 

   
  }

  digitalWrite(Buzzer_Pin, HIGH);   //响
  delay(555);
  digitalWrite(Buzzer_Pin, LOW);  
  DEBUG.println("初始化完成");
  diagnostic_boot_stage = BOOT_STAGE_READY;
  RosSerial_Diagnostics_TXD(true);
  RosSerial_PidTune_TXD(true);
  now_us1 = micros();
}


void loop() {

  now_us = micros();
  command.run();//通讯调参
  ImuUpdate();//更新IMU数据   
  
  RXsbus(); //更新遥控器数据

  if(robot_mode==0)
  {
    Ros_MobileChassis_RXD();   
  }
  else if(robot_mode==1)
  {
    Ros_JointData_RXD();
  }
  time_dt = (now_us - now_us1) / 1000000.0f;
  if (time_dt >= 0.01f)
  {
    static int key = 0;
    static unsigned long ws = micros();
    ws = micros();
    UpdatePidTuning(time_dt);
    //TemperatureAlarm(60.5);//电机温度60.5
    if(key==0)
    {
      // The control loop remains at 100 Hz, while the 182-byte telemetry
      // packet is sent at 50 Hz to leave enough margin at 230400 baud.
      RosSerial_TXD();
      RosSerial_Diagnostics_TXD();
      RosSerial_PidTune_TXD();
      key = 1;
    }
    else
    {
      key = 0;
    }

    DistanceFiltering(time_dt);//里程计与IMU数据融合
    RemoteControlFiltering();//遥控信号滤波
    ReadVoltage();//电池
    print_data();//串口打印数据
    Robot_Tumble();//机器摔倒检测
    voltage_Indicator_Light();//电压指示灯  
    LPF_Parameter_Init();//滤波器调参初始化
    FlashSave((int)CalibrationSelect);//校准保存数据
    Update_Motor_Speed();//更新电机速度
    SpeedCalculation(time_dt);//速度估计
    if(robot_mode==0)//遥控器或者上位机控制移动底盘
    {
      MobileChassisTask();
    }
    else if(robot_mode==1)//上位机控制关节
    {
      JointTask();
    }

    
    //GetAllMotorStatusWord();//获取所有电机状态

    //DEBUG.print (" 微秒：") ;
    //DEBUG.println (micros()-ws); 
    now_us1 = now_us;
  }
}
