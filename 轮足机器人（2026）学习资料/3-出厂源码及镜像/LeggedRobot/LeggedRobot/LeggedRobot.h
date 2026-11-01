#ifndef LeggedRobot_h
#define LeggedRobot_h

#include <Arduino.h>
#include "filter.h"
#include "touchscreen.h"
#include "IMUSystem.h"

#define CLAMP(x, lower, upper)                        (min(upper, max(x, lower)))
const int Buzzer_Pin = 11;


typedef struct {
  //限制
  float LimitLinearVelocity = 2.0;         //机器最大移动速度m/s
  float LimitAngleVelocity = 5.0;          //机器航向角速度 弧度每秒限制
  float LimitPitchAngle = 35.0;            //俯仰角度限制 度
  float LimitRollAngle = 15.0;             //横滚角度限制 度
  float MaxHigh = 0.37;                    //腿最高
  float MinLow = 0.13;                     //腿最低  
  float PostureRestrictions_x = 0.08;      //姿态 x ± 限制
  float  DefaultHeight = 0.25;             //摇杆中间时的高度
  float ParallelHigh = 0.143331;           //大腿与机体平行高度
  //电机
  float JointAngleDeviation = 96.25619525; //关节角度偏差
  //机体
  float BodyWeight = 3.0;                  //身体重量
  float WheelTrack = 0.4118;               //轮距
  float thigh = 0.15;                      // 大腿长度m L1 L4
  float shank = 0.24;                      // 小腿长度m L2 L3
  float MotorSpacing = 0.085;              // 电机间距 L5

  float PayloadWeight = 0;                 //kg

  float PitchControlCompensation = 85;     //

  float AccOffsetX = 0;

  int JumpingState = 0;

} ConfigurationParameter_t;//配置参数


enum MotorID : uint8_t  
{
    LEFT_FRONT = 0,    // 左前电机
    LEFT_REAR = 1,     // 左后电机
    RIGHT_FRONT = 2,   // 右前电机
    RIGHT_REAR = 3,    // 右后电机
    LEFT_WHEEL = 4,    // 左轮电机
    RIGHT_WHEEL = 5,   // 右轮电机

    TOTAL_COUNT        // 电机总数（用于数组大小定义、遍历等）
};


struct RobotStatus {
    // 状态码联合体（你的定义）
    union {
        uint16_t status_code;  // 32位状态码整体
        struct {
            uint16_t switched_on : 1;           // 机器开启状态(1=开启，0=关闭)
            uint16_t model : 1;                 // 机器工作模式(1=控制电机模式，0=控制移动底盘模式)
            uint16_t PADDING : 14;              // 保留位（必须为0）
        };
    } status;
};


// ==================== 新增结构体定义 ====================
#pragma pack(push, 1)  // 确保1字节对齐，避免填充字节


// ROS通信数据结构体
typedef struct {
    uint8_t header1;          // 帧头1 (123)
    uint8_t header2;          // 帧头2 (45)
    uint16_t length;          // 结构体长度
    float linear_velocity;
    float angle_velocity;
    float leg_length;
    float pitch_angle;
    float roll_angle;
    float jump_min_height;
    float jump_max_height;
    uint16_t set;             // 设置
    uint8_t count;
    uint16_t crc;             // CRC16校验
    
} ComWheelLegged_t;


// 里程计数据结构
typedef struct {
    float LeftWheelSpeed;     // 左轮速度 (m/s)
    float RightWheelSpeed;    // 右轮速度 (m/s)
    float Speed;              // 前进速度 (m/s)
    float AngularVelocity;    // 角速度 (rad/s)
} milemeter_t;

// IMU传感器数据结构
typedef struct {
    float accel[3];           // 加速度计数据 [x, y, z] (m/s²)
    float gyro[3];            // 陀螺仪数据 [x, y, z] (rad/s)
    float q0 = 1.0f;          // 四元数分量
    float q1 = 0.0f;          // 四元数分量
    float q2 = 0.0f;          // 四元数分量
    float q3 = 0.0f;          // 四元数分量
} ImuData_t;


// 电机控制参数
typedef struct {
    float position; // 电机位置信息 减速后的角度
    float velocity; // 电机速度信息 减速后的速度 度每秒
    float kp;           // 比例增益
    float kd;           // 微分增益
    float torque_ff;    // 前馈转矩
} MIT_Command_Data_t;


typedef struct {
    uint8_t header1;          // 帧头1 (123)
    uint8_t header2;          // 帧头2 (45)
    uint16_t length;          // 数据包长度
    MIT_Command_Data_t MIT_Command_Data[6];
    uint16_t count;            // 计数
    uint16_t crc;             // CRC16校验    
} RX_MIT_Data_t;


typedef struct {
    float position; // 电机位置信息 减速后的角度
    float velocity; // 电机速度信息 减速后的速度 度每秒
    float torque;   // 力矩 牛米
    float Temp;           //回传温度
} MIT_Feedback_Data_t;


// ROS通信数据结构体
typedef struct {
    uint8_t header1;          // 帧头1 (123)
    uint8_t header2;          // 帧头2 (45)
    uint16_t length;          // 数据包长度
    milemeter_t milemeter;    // 里程计数据
    ImuData_t ImuData;        // IMU数据
    MIT_Feedback_Data_t MIT_Feedback_Data[6];//电机反馈
    int16_t SBUS_Channels_Data[10];
    RobotStatus Status;          // 状态标志
    uint16_t count;            // 计数
    uint16_t crc;             // CRC16校验
} ROS_body_t;


// ROS通信数据结构体
typedef struct {
    uint8_t header1;          // 帧头1 (123)
    uint8_t header2;          // 帧头2 (45)
    uint16_t length;          // 数据包长度
    MIT_Feedback_Data_t MIT_Feedback_Data[6];//电机反馈
    int16_t SBUS_Channels_Data[10];
    uint16_t count;            // 计数
    uint16_t crc;             // CRC16校验
} ROS_body_t1;

// Type-C structured diagnostic frame (ESP32 -> OrangePi).
// It uses a different second header byte from normal telemetry so binary
// diagnostics can share UART0 without mixing human-readable text into ROS data.
typedef struct {
    uint8_t header1;              // 0x7B
    uint8_t header2;              // 0x2E (diagnostic frame)
    uint16_t length;
    uint8_t protocol_version;     // diagnostic protocol version
    uint8_t boot_stage;           // BootStage value
    uint8_t overall_state;        // 0=starting, 1=waiting, 2=ready, 3=fault
    uint8_t robot_mode;           // 0=chassis, 1=joint mode
    uint32_t fault_bits;          // DiagnosticFaultBits
    uint32_t uptime_ms;
    uint16_t battery_mv;
    uint16_t sbus_frame_period_ms;
    uint16_t sbus_frame_age_ms;
    int16_t sbus_channels[10];
    uint8_t sbus_failsafe;        // 0=OK, 1=lost, 3=failsafe
    uint8_t remote_reset_mask;    // bits 0..5: signal/SWA/SWB/SWC/SWD/VRA
    uint8_t motor_seen_mask;      // motors that have replied since boot
    uint8_t motor_fresh_mask;     // motors that replied in the last 500 ms
    uint8_t motor_enabled_mask;
    uint8_t yaw_hold_active;
    int16_t yaw_bias_millirad_s;
    int16_t corrected_yaw_rate_millirad_s;
    int16_t yaw_hold_error_centideg;
    uint32_t motor_errors[6];
    uint16_t count;
    uint16_t crc;
} RobotDiagnostic_t;

static_assert(sizeof(RobotDiagnostic_t) == 82,
              "RobotDiagnostic_t protocol size must be 82 bytes");

// Safe RAM-only PID tuning command (OrangePi -> ESP32).
typedef struct {
    uint8_t header1;              // 0x7B
    uint8_t header2;              // 0x2F
    uint16_t length;
    uint8_t protocol_version;
    uint8_t action;
    uint8_t parameter_id;
    uint8_t reserved;
    float value;
    uint16_t sequence;
    uint16_t crc;
} PidTuneCommand_t;

static_assert(sizeof(PidTuneCommand_t) == 16,
              "PidTuneCommand_t protocol size must be 16 bytes");

// High-rate tuning telemetry (ESP32 -> OrangePi).
typedef struct {
    uint8_t header1;              // 0x7B
    uint8_t header2;              // 0x30
    uint16_t length;
    uint8_t protocol_version;
    uint8_t session_active;
    uint8_t last_status;
    uint8_t last_parameter_id;
    uint16_t last_sequence;
    uint16_t safety_flags;
    uint32_t uptime_ms;
    float target_speed_mps;
    float measured_speed_mps;
    float session_position_m;
    float target_pitch_deg;
    float measured_pitch_deg;
    float pitch_rate_rad_s;
    float speed_error_mps;
    float speed_integral;
    float speed_output;
    float target_leg_x_m;
    float angle_error_deg;
    float angle_output_nm;
    float target_yaw_rate_rad_s;
    float measured_yaw_rate_rad_s;
    float yaw_error_rad_s;
    float yaw_output_nm;
    float yaw_hold_error_deg;
    float left_wheel_torque_nm;
    float right_wheel_torque_nm;
    float speed_kp;
    float speed_ki;
    float speed_kd;
    float applied_speed_kp;
    float applied_speed_ki;
    float yaw_kp;
    float yaw_ki;
    float yaw_kd;
    float applied_yaw_ki;
    float heading_hold_kp;
    float configured_angle_kp;
    float applied_angle_kp;
    float angle_kd;
    uint16_t count;
    uint16_t crc;
} PidTuneTelemetry_t;

static_assert(sizeof(PidTuneTelemetry_t) == 148,
              "PidTuneTelemetry_t protocol size must be 148 bytes");


#pragma pack(pop)  // 恢复默认对齐方式

enum PidTuneAction : uint8_t {
    PID_TUNE_ACTION_BEGIN = 1,
    PID_TUNE_ACTION_SET = 2,
    PID_TUNE_ACTION_REVERT = 3,
    PID_TUNE_ACTION_RESTORE_BOOT = 4,
    PID_TUNE_ACTION_ACCEPT_RAM = 5,
    PID_TUNE_ACTION_GET = 6,
};

enum PidTuneParameter : uint8_t {
    PID_TUNE_PARAM_NONE = 0,
    PID_TUNE_PARAM_SPEED_KP = 1,
    PID_TUNE_PARAM_SPEED_KI = 2,
    PID_TUNE_PARAM_SPEED_KD = 3,
    PID_TUNE_PARAM_YAW_KP = 4,
    PID_TUNE_PARAM_YAW_KI = 5,
    PID_TUNE_PARAM_YAW_KD = 6,
    PID_TUNE_PARAM_HEADING_HOLD_KP = 7,
    PID_TUNE_PARAM_HEADING_HOLD_ENABLE = 8,
};

enum PidTuneStatus : uint8_t {
    PID_TUNE_STATUS_OK = 0,
    PID_TUNE_STATUS_NO_SESSION = 1,
    PID_TUNE_STATUS_UNSAFE_STATE = 2,
    PID_TUNE_STATUS_BAD_PARAMETER = 3,
    PID_TUNE_STATUS_OUT_OF_RANGE = 4,
    PID_TUNE_STATUS_BAD_ACTION = 5,
    PID_TUNE_STATUS_BAD_VERSION = 6,
    PID_TUNE_STATUS_AUTO_REVERTED = 7,
    PID_TUNE_STATUS_TARGET_PENDING = 8,
};

enum PidTuneSafetyFlags : uint16_t {
    PID_TUNE_SAFE_BOOT_READY = 1U << 0,
    PID_TUNE_SAFE_CHASSIS_MODE = 1U << 1,
    PID_TUNE_SAFE_ZERO_COMMAND = 1U << 2,
    PID_TUNE_SAFE_NOT_JUMPING = 1U << 3,
    PID_TUNE_SAFE_PITCH = 1U << 4,
    PID_TUNE_SAFE_REMOTE = 1U << 5,
};

struct PidTuneValueSet {
    float speed_kp;
    float speed_ki;
    float speed_kd;
    float yaw_kp;
    float yaw_ki;
    float yaw_kd;
    float heading_hold_kp;
    bool heading_hold_enable;
};

enum BootStage : uint8_t {
    BOOT_STAGE_STARTING = 0,
    BOOT_STAGE_MOTOR_INIT = 1,
    BOOT_STAGE_IMU_INIT = 2,
    BOOT_STAGE_REMOTE_INIT = 3,
    BOOT_STAGE_WAIT_REMOTE_RESET = 4,
    BOOT_STAGE_READY = 5,
    BOOT_STAGE_CAN_FAULT = 100,
    BOOT_STAGE_MOTOR_STOP_FAULT = 101,
    BOOT_STAGE_MOTOR_HOME_FAULT = 102,
    BOOT_STAGE_MOTOR_DIRECTION_FAULT = 103,
    BOOT_STAGE_MOTOR_MODE_FAULT = 104,
    BOOT_STAGE_MOTOR_ENABLE_FAULT = 105,
    BOOT_STAGE_IMU_FAULT = 106,
};

enum DiagnosticFaultBits : uint32_t {
    DIAG_FAULT_CAN_INIT = 1UL << 0,
    DIAG_FAULT_MOTOR_STOP = 1UL << 1,
    DIAG_FAULT_MOTOR_HOME = 1UL << 2,
    DIAG_FAULT_MOTOR_DIRECTION = 1UL << 3,
    DIAG_FAULT_MOTOR_MODE = 1UL << 4,
    DIAG_FAULT_MOTOR_ENABLE = 1UL << 5,
    DIAG_FAULT_IMU = 1UL << 6,
    DIAG_FAULT_SBUS = 1UL << 7,
    DIAG_FAULT_REMOTE_NOT_RESET = 1UL << 8,
    DIAG_FAULT_BATTERY_LOW = 1UL << 9,
    DIAG_FAULT_MOTOR_OFFLINE = 1UL << 10,
    DIAG_FAULT_MOTOR_ERROR = 1UL << 11,
};

// Implemented by the main sketch and also called from MotorControl.cpp when a
// blocking startup fault occurs.
void SetDiagnosticBootFault(uint8_t stage, uint32_t fault_bit);
void RosSerial_Diagnostics_TXD(bool force = false);

typedef struct {
    // 控制器状态
    float input_velocity;     // 输入速度
    float vel_setpoint;       // 速度设定点
    
    // 配置参数
    float velocity_ramp_rate; // 速度斜坡率
} accelerated_speed;

typedef struct {
    float x;
    float y;
} Node;


typedef struct 
{
    float alphaLeft;   // 左腿α角度
    float betaLeft;     // 左腿β角度
    float alphaRight;  // 右腿α角度
    float betaRight;   // 右腿β角度
}JointAngles;



typedef struct
{
  unsigned char rxbuf[30];//接收数据的缓冲区
  unsigned char txbuf[30];
  unsigned char recstatu;//表示是否处于一个正在接收数据包的状态
  unsigned char count;//计数
  unsigned char packerflag;//是否接收到一个完整的数据包标志
  unsigned char dat;     
} Serial_t;


typedef struct
{

  float MotorVelocityF[4];   //电机速度


  uint8_t MotorMode;   //电机工作模式

  int Serial1HZ;
  int Serial1count;

  //目标值
  float WheelTargetTorque[2];//设置车轮力矩
  float LeftLegJointTargetAngle[2];
  float RightLegJointTargetAngle[2];
  float TargetPitchingAngle;
  float TargetRollAngle;
  float TargetPosition_x;
  float TargetMovementSpeed_x;
  float TargetMovementSpeed_x_f;
  float TargetRoll;
  float TargetRoll_f;
  float TargetPitching_differential;
  float TargetYaw;  
  float SetRoll;
 
  Node LeftPlace;//两电机角度正解的坐标
  Node RightPlace;

  Node LeftPlace_p;//加入俯仰角正解后的坐标
  Node RightPlace_p;  

  Node barycenter_p;  
  Node barycenter_last;//
  Node barycenter_p_v;//


  int LegOffGround[3];//脚离地标志

  float WheelSpeed[2];//补偿之态运动后的转速
  float WheelSpeed_f[2];//

  int JumpingStep;//跳跃步骤

  accelerated_speed PitchAngleSlope ; 

  milemeter_t milemeter;//里程计
  ImuData_t ImuData;
 
} body_t;




typedef union 
{
  struct 
  {
    float x;
    float y;
    float z;
  };
  float axis[3];
} Axis3f;

//姿态数据结构
typedef struct  
{
  Axis3f accf;       //滤波 加速度（G）
  Axis3f gyrof;      //滤波 陀螺仪（deg/s）  
  Axis3f acc;       //加速度（G）
  Axis3f gyro;      //陀螺仪（deg/s）  
  float roll;
  float pitch;
  float yaw;
  float temp;
} attitude_t;


typedef struct  
{
  Axis3f acc;       //加速度（G）
  Axis3f gyro;      //陀螺仪（deg/s） 
  float roll;
  float pitch;
  float yaw;

  float servo1;
  float servo2;
  float servo3;
  float servo4;

  
} zeroBias_t;

class MyPIDController {
  private:


  public:
  
    float Kp;  // 比例系数
    float Ki;  // 积分系数
    float Kd;  // 微分系数
    float deriv;
    float integral;  // 误差积分
    float previousError;  // 上一次的误差
    float iLimit;
    float outputLimit;
    float outP;
    float outI;
    float outD;    
    float output;
    float error;
    float enableDFilter;
    int EnablePoints;
    biquadFilter_t dFilter;  //
    float cutoffFreq;
  
    // 构造函数，初始化PID参数
    MyPIDController(float p, float i, float d, float iLimit, float outputLimit,float dt, float EnableDFilter, float CutoffFreq) {
      Kp = p;
      Ki = i;
      Kd = d;
      EnablePoints = 1;
      enableDFilter = EnableDFilter;
      integral = iLimit;
      previousError = outputLimit;//输出限幅
      cutoffFreq = CutoffFreq;
      
      if ((int)enableDFilter==1)
      {
        biquadFilterInitLPF(&dFilter, (1.0f/dt), (unsigned int)cutoffFreq);
      }      
      
    }

    // 计算PID输出的函数，参数包含误差和时间间隔dt
    float compute(float Error, float dt) {

      error = Error;
      // 计算误差积分
      if(EnablePoints==1)
        integral += error * dt;

      //积分限幅
      if (iLimit != 0)
      {
        if(integral>iLimit)
          integral = iLimit;
        if(integral<(-iLimit))
          integral = -iLimit;
      }

      // 计算误差微分
      deriv = (error - previousError) / dt;
      if (enableDFilter==1)
      {
        deriv = biquadFilterApply(&dFilter, deriv);
      }

      
      outP = Kp * error;
      outI = Ki * integral;
      outD = Kd * deriv;


      // 计算PID输出
      output = outP + outI + outD;

      // 更新上一次的误差
      previousError = error;

      //输出限幅
      if (outputLimit != 0)
      {
        output = constrain(output, -outputLimit, outputLimit);
      }
  

      return output;
    }

    // 设置PID系数的函数
    void setPID(float p, float i, float d, float iLimit, float outputLimit,float dt, float EnableDFilter, float CutoffFreq) {
      Kp = p;
      Ki = i;
      Kd = d;
      enableDFilter = EnableDFilter;
      integral = iLimit;
      previousError = outputLimit;//输出限幅
      cutoffFreq = CutoffFreq;
      
      if ((int)enableDFilter)
      {
        biquadFilterInitLPF(&dFilter, (1.0f/dt), (unsigned int)cutoffFreq);
      }    
    }
};

extern ConfigurationParameter_t ConfigurationParameter;


#endif
