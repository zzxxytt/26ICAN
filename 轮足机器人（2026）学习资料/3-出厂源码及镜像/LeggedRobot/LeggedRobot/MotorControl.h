#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <Arduino.h>
#include <ACAN_ESP32.h> // 引入ACAN_ESP32库

#define OFFSET_LUT_NUM   128U

// CAN ID 11 bit high to low
#define ID_ECHO_BIT         0x400 // 1 bit
#define ID_NODE_BIT         0x3E0 // 5 bit
#define ID_CMD_BIT          0x01F // 5 bit

#define IS_ECHO(can_id)     (can_id & ID_ECHO_BIT)
#define GET_NODE_ID(can_id) ((can_id & ID_NODE_BIT) >> 5)
#define GET_CMD(can_id)     (can_id & ID_CMD_BIT)


//控制参数最值，谨慎更改
#define P_MIN -16.0f
#define P_MAX 16.0f
#define V_MIN -50.0f //转/s
#define V_MAX 50.0f
#define KP_MIN 0.0f
#define KP_MAX 500.0f
#define KD_MIN 0.0f
#define KD_MAX 5.0f
#define T_MIN -12.0f //牛米
#define T_MAX 12.0f
#define MAX_P 16.0f
#define MIN_P -16.0f  //转

#define MAX_T 200.0f
#define MIN_T -20.0f  //温度


#define Speed_Conversion_Deviation 0.00076294  //速度转换偏差

// Type-C is connected to ESP32-S3 UART0 through the on-board CH340K.
// Keep binary ROS communication and human-readable debug output on separate
// UARTs so debug text can never corrupt a protocol frame.
#define  RosSerial Serial
#define  DEBUG Serial1


typedef enum {
    CONTROL_MODE_TORQUE_RAMP      = 0,
    CONTROL_MODE_VELOCITY_RAMP    = 1,
    CONTROL_MODE_POSITION_FILTER  = 2,
    CONTROL_MODE_POSITION_PROFILE = 3,
   CONTROL_MODE_MOTION = 4, //MIT
} tControlMode;

/**
 * 用户配置参数结构体
 * 功能：存储系统运行所需的所有可配置参数
 * 分类：包含电机参数、校准参数、控制参数、保护参数、通信参数和编码器参数
 */
typedef struct sUsrConfig
{
    // ==================== 电机参数 ====================
    int32_t invert_motor_dir;       // 电机方向反转：0=正常 1=反转
    float   inertia;                // 转动惯量 [A/(turn/s²)]       (范围: 0~100)
    float   torque_constant;        // 转矩常数 [Nm/A]             (范围: 0~10)
    int32_t motor_pole_pairs;       // 电机极对数 [PP]             (范围: 2~30)
    float   motor_phase_resistance; // 相电阻 [Ω]                  (范围: 0~10)
    float   motor_phase_inductance; // 相电感 [H]                  (范围: 0~10)
    float   current_limit;          // 电流限制 [A]                (范围: 0~45)
    float   velocity_limit;         // 速度限制 [turn/s]           (范围: 0~100)

    // ==================== 校准参数 ====================
    float calib_current;            // 校准电流 [A]                (范围: 0~45)
    float calib_voltage;            // 校准电压 [V]                (范围: 0~50)

    // ==================== 控制器参数 ====================
    int32_t control_mode;           // 控制模式 (如位置/速度/转矩控制) 0：转矩爬升 1：转速爬升 2：位置过滤波 3：轮廓位置 4：MIT模式
    float   pos_gain;               // 位置环增益                  (范围: 0~1000)
    float   vel_gain;               // 速度环增益                  (范围: 0~1000)
    float   vel_integrator_gain;    // 速度环积分增益              (范围: 0~1000)
    float   current_ctrl_bw;        // 电流环带宽 [Hz]             (范围: 2~60000)
    int32_t anticogging_enable;     // 齿槽效应补偿：0=禁用 1=启用
    int32_t sync_target_enable;     // 同步目标：0=禁用 1=启用
    float   target_velcity_window;  // 目标速度窗口 [turn/s]       (范围: 0~100)
    float   target_position_window; // 目标位置窗口 [turn]         (范围: 0~100)
    float   torque_ramp_rate;       // 转矩斜坡率 [Nm/s]           (范围: 0~100)
    float   velocity_ramp_rate;     // 速度斜坡率 [(turn/s)/s]     (范围: 0~1000)
    float   position_filter_bw;     // 位置滤波器带宽 [Hz]         (范围: 2~5000)
    float   profile_velocity;       // 轮廓速度 [turn/s]           (范围: 0~100)
    float   profile_accel;          // 轮廓加速度 [(turn/s)/s]     (范围: 0~1000)
    float   profile_decel;          // 轮廓减速度 [(turn/s)/s]     (范围: 0~1000)

    // ==================== 保护参数 ====================
    float protect_under_voltage;    // 欠压保护阈值 [V]            (范围: 0~50)
    float protect_over_voltage;     // 过压保护阈值 [V]            (范围: 0~50)
    float protect_over_current;     // 过流保护阈值 [A]            (范围: 0~45)
    float protect_i_bus_max;        // 总线电流最大值 [A]          (范围: 0~10)

    // ==================== CAN通信参数 ====================
    int32_t node_id;                // CAN节点ID                  (范围: 1~31)
    int32_t can_baudrate;           // CAN波特率 (参考tCanBaudrate枚举)
    int32_t heartbeat_consumer_ms;  // 接收心跳超时时间 [ms]：0=禁用 (范围: 0~600000)
    int32_t heartbeat_producer_ms;  // 发送心跳间隔时间 [ms]：0=禁用 (范围: 0~600000)

    // ==================== 编码器参数 ====================
    int32_t calib_valid;                // 校准有效性标志 (自动生成)
    int32_t encoder_dir;                // 编码器方向 (自动生成)
    int32_t encoder_offset;             // 编码器偏移量 (自动生成)
    int32_t offset_lut[OFFSET_LUT_NUM]; // 编码器偏移查找表 (自动生成)

    uint32_t crc;                       // 配置参数CRC校验值
} tUsrConfig;

/**
 * @brief 电机控制器状态字结构体
 * 
 * 该结构体用于存储电机控制器的状态信息和错误代码
 * 采用位域和联合体的方式实现对状态码和错误码的位操作
 */
typedef struct sMCStatusword
{
    // 状态码联合体 - 允许按位访问或作为32位整数访问
    union {
        uint32_t status_code;  /**< 32位状态码值 */
        struct
        {
            uint32_t switched_on : 1;           /**< 电机开启状态位(1=开启，0=关闭) */
            uint32_t target_reached : 1;        /**< 目标值到达标志位 */
            uint32_t current_limit_active : 1;  /**< 电流限制激活标志位 */
            uint32_t PADDING : 29;              /**< 保留位，必须填充为0 */
        };
    } status;

    // 错误码联合体 - 允许按位访问或作为32位整数访问
    union {
        uint32_t errors_code;  /**< 32位错误码值 */
        struct
        {
            // FATAL错误区域(低16位)
            uint32_t adc_selftest_fatal : 1;  /**< ADC自测试失败(1<<0) */
            uint32_t encoder_offline : 1;     /**< 编码器离线(1<<1) */
            uint32_t PADDING_1 : 14;          /**< 保留位 */
            
            // ERROR错误区域(高16位)
            uint32_t over_voltage : 1;        /**< 过压错误(1<<16) */
            uint32_t under_voltage : 1;       /**< 欠压错误(1<<17) */
            uint32_t over_current : 1;        /**< 过流错误(1<<18) */
            uint32_t PADDING_2 : 13;          /**< 保留位 */
        };
    } errors;
} tMCStatusword;


typedef struct{         //

  int HomeOk;//电机零点是否设置成功
  tUsrConfig UsrConfig;
  tMCStatusword MCStatusword;
  //uint8_t CAN_ID;       //CAN ID
  uint8_t dlc;
  uint8_t data[8];
  float kp;
  float kd;
  float in_p_des;//输入角度
  float in_v_des;//输入角速度
  float in_torque;//输入力矩
  float Angle;          //回传角度 -720~720
  float Turn;
  float Turn_f;
  float Turn_s;
  float Speed;          // 度每秒
  float Torque;         //回传力矩（-12Nm~12Nm）  
  float Temp;           //回传温度
  uint16_t set_current;
  uint16_t set_speed;
  uint16_t set_position;
  
  uint8_t error_code;
  
  uint16_t HZ;
  
  float Angle_Bias;
  uint32_t last_tx_ms;//上一次发送时间

  uint32_t gReceivedFrameCount;
  
}MI_Motor;



typedef enum sUsrConfigCmd
{
    INVERT_MOTOR_DIR_ENUM = 1,//电机方向   
    INERTIA_ENUM,                
    TORQUE_CONSTANT_ENUM,        
    MOTOR_POLE_PAIRS_ENUM,       
    MOTOR_PHASE_RESISTANCE_ENUM, 
    MOTOR_PHASE_INDUCTANCE_ENUM, 
    CURRENT_LIMIT_ENUM,          
    VELOCITY_LIMIT_ENUM,         
    CALIB_CURRENT_ENUM,          
    CALIB_VOLTAGE_ENUM,          
    CONTROL_MODE_ENUM, //11 控制模式          
    POS_GAIN_ENUM,//12               
    VEL_GAIN_ENUM,//13               
    VEL_INTEGRATOR_GAIN_ENUM,//14    
    CURRENT_CTRL_BW_ENUM,//15        
    ANTICOGGING_ENABLE_ENUM,//16     
    SYNC_TARGET_ENABLE_ENUM,//17     
    TARGET_VELOCITY_WINDOW_ENUM,//18  
    TARGET_POSITION_WINDOW_ENUM,//19 
    TORQUE_RAMP_RATE_ENUM,//20       
    VELOCITY_RAMP_RATE_ENUM,//21     
    POSITION_FILTER_BW_ENUM,//22     
    PROFILE_VELOCITY_ENUM,//23       
    PROFILE_ACCEL_ENUM,//24          
    PROFILE_DECEL_ENUM,//25          
    PROTECT_UNDER_VOLTAGE_ENUM,//26    
    PROTECT_OVER_VOLTAGE_ENUM,//27     
    PROTECT_OVER_CURRENT_ENUM,//28     
    PROTECT_I_BUS_MAX_ENUM,//29        
    NODE_ID_ENUM,//30                
    CAN_BAUDRATE_ENUM,//31           
    HEARTBEAT_CONSUMER_MS_ENUM,//32  
    HEARTBEAT_PRODUCER_MS_ENUM,  
    CALIB_VALID_ENUM,                
    ENCODER_DIR_ENUM,                
    ENCODER_OFFSET_ENUM,             
    OFFSET_LUT_ENUM, 
    CRC_ENUM,                       
} tUsrConfigCmd;

typedef enum eCanCmd {
    CAN_CMD_MOTOR_DISABLE = 0,    /**< 0禁用电机，停止所有控制输出 */
    CAN_CMD_MOTOR_ENABLE,         /**< 1启用电机，允许控制输出 */

    CAN_CMD_SET_TORQUE,           /**< 2设置电机转矩控制模式及目标转矩值 */
    CAN_CMD_SET_VELOCITY,         /**< 3设置电机速度控制模式及目标速度值 */
    CAN_CMD_SET_POSITION,         /**< 4设置电机位置控制模式及目标位置值 */

    CAN_CMD_CALIB_START,          /**< 5启动电机校准过程 */
    CAN_CMD_CALIB_REPORT,         /**< 6请求校准结果报告 */
    CAN_CMD_CALIB_ABORT,          /**< 7中止正在进行的校准过程 */

    CAN_CMD_ANTICOGGING_START,    /**< 8启动齿槽效应补偿学习过程 */
    CAN_CMD_ANTICOGGING_REPORT,   /**< 9请求齿槽效应补偿数据报告 */
    CAN_CMD_ANTICOGGING_ABORT,    /**< 10中止正在进行的齿槽效应补偿学习 */

    CAN_CMD_SET_HOME,             /**< 11设置当前位置为零点位置 */
    CAN_CMD_ERROR_RESET,          /**< 12清除错误状态，恢复正常运行 */
    CAN_CMD_GET_STATUSWORD,       /**< 13请求获取当前状态字 */
    CAN_CMD_STATUSWORD_REPORT,    /**< 14状态字报告响应 */

    CAN_CMD_GET_VALUE_1,          /**< 15获取第一组参数值(见fill_value函数) */
    CAN_CMD_GET_VALUE_2,          /**< 16获取第二组参数值(见fill_value函数) */

    CAN_CMD_SET_CONFIG,           /**< 17设置特定配置参数 */
    CAN_CMD_GET_CONFIG,           /**< 18请求获取特定配置参数 */
    CAN_CMD_SAVE_ALL_CONFIG,      /**< 19将当前配置保存到非易失存储器 */
    CAN_CMD_RESET_ALL_CONFIG,     /**< 20重置所有配置为默认值 */

    CAN_CMD_SYNC,                 /**< 21同步命令，用于多电机同步控制 */
    CAN_CMD_HEARTBEAT,            /**< 22心跳命令，用于保持通信连接 */
    CAN_CMD_SET_TORQUE_MIT,        /**< 23MIT控制模式 */

    CAN_CMD_GET_FW_VERSION = 28,  /**< 请求获取固件版本信息 */

    CAN_CMD_DFU_START,            /**< 启动固件更新模式 */
    CAN_CMD_DFU_DATA,             /**< 传输固件数据块 */
    CAN_CMD_DFU_END,              /**< 结束固件更新过程 */
   
} tCanCmd;


extern MI_Motor JointMotor[6];

void SetAllJointMode(int32_t Mode);
void EnableAllMotors(void);
void StopAllMotors(void);
void CanInit(void);
void MotorInit(void); // 电机初始化
void SetJointAngles(float a1,float a2,float a3,float a4,float kp,float kd);
void SetWheelTorque(float left,float right);
float angleToRevolutions(int angle) ;
float fullTurnToAngle(float turn);
float rpsToDegreesPerSecond(float TurnPerSecond) ;
float degreesPerSecondToRps(float degreesPerSecond) ;

void MotorMit(MI_Motor *Motor,float p_des, float v_des, float kp, float kd, float t_ff);
int ConfigureMotorMode(MI_Motor *Motor,int32_t Mode);
int EnableMotor(MI_Motor *Motor);
int StopMotors(MI_Motor *Motor);
int ReverseMotorDirection(MI_Motor *Motor,int32_t Dir);
void SaveAllConfigurations(MI_Motor *Motor);
void GetAllMotorStatusWord(void);//获取所有电机当前状态字
float angleToTurn(int angle);
double degreesToRadians(double degrees);
int GetAllMotorStop(int n);
void MitSetJointAngles(void);

#endif
