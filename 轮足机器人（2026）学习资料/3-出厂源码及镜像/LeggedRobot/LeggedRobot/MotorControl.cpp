#include "MotorControl.h"
#include "LeggedRobot.h"
#include "filter.h"  




#define Motor_Reduction_Ratio 8.0f



//电机
MI_Motor JointMotor[6] = {0};
biquadFilter_t MotorFilterLPF[6];//二阶低通滤波器

// 设置CAN波特率为1000kbps
static const uint32_t DESIRED_BIT_RATE = 1000UL * 1000UL ;

// 定义引脚 (根据实际硬件调整)
static const gpio_num_t CAN_TX_PIN = GPIO_NUM_20; // CAN发送引脚
static const gpio_num_t CAN_RX_PIN = GPIO_NUM_19; // CAN接收引脚

// 全局变量
static volatile uint32_t gReceivedFrameCount = 0 ; // 接收帧计数器
static volatile uint32_t gSentFrameCount = 0 ;    // 发送帧计数器
static uint32_t gBlinkLedDate = 0 ;               // LED闪烁计时器


//——————————————————————————————————————————————————————————————————————————————
//   CAN发送函数实现
//——————————————————————————————————————————————————————————————————————————————
bool sendCANMessage(uint32_t id, const uint8_t* data, uint8_t len, bool ext) {
  CANMessage frame;
  frame.id = id;      // 设置消息ID
  frame.ext = ext;    // 帧格式: false=标准帧, true=扩展帧
  frame.len = len;    // 数据长度(0-8)
  
  // 复制数据到帧
  for (int i = 0; i < len && i < 8; i++) {
    frame.data[i] = data[i];
  }
  
  // 尝试发送
  const bool success = ACAN_ESP32::can.tryToSend(frame);
  
  if (success) {
    gSentFrameCount++; // 更新发送计数器
    //DEBUG.print("ID 0x");
    //DEBUG.print(id, HEX);
    //DEBUG.println(" 消息已发送");
  } else {
    DEBUG.println("发送失败: 队列已满");
  }

  return success;
}

/**
  * @brief          小米电机回文16位数据转浮点
  * @param[in]      x:16位回文
  * @param[in]      x_min:对应参数下限
  * @param[in]      x_max:对应参数上限
  * @param[in]      bits:参数位数
  * @retval         返回浮点值
  */
static float uint16_to_float(uint16_t x,float x_min,float x_max,int bits)
{
    uint32_t span = (1 << bits) - 1;
    float offset = x_max - x_min;
    return offset * x / span + x_min;
}


/**
  * @brief          小米电机发送浮点转16位数据
  * @param[in]      x:浮点
  * @param[in]      x_min:对应参数下限
  * @param[in]      x_max:对应参数上限
  * @param[in]      bits:参数位数
  * @retval         返回浮点值
  */
static int float_to_uint(float x, float x_min, float x_max, int bits)
{
  float span = x_max - x_min;
  float offset = x_min;
  if(x > x_max) x=x_max;
  else if(x < x_min) x= x_min;
  return (int) ((x-offset)*((float)((1<<bits)-1))/span);
}

int32_t data_to_int32(uint8_t *data)
{
    int32_t tmp;
    *(((uint8_t *) (&tmp)) + 0) = data[0];
    *(((uint8_t *) (&tmp)) + 1) = data[1];
    *(((uint8_t *) (&tmp)) + 2) = data[2];
    *(((uint8_t *) (&tmp)) + 3) = data[3];
    return tmp;
}

void float_to_data(float value, uint8_t *data) {
    uint8_t *ptr = (uint8_t*)&value;
    data[0] = ptr[0];  // 最低有效字节 (LSB)
    data[1] = ptr[1];
    data[2] = ptr[2];
    data[3] = ptr[3];  // 最高有效字节 (MSB)
}


float data_to_float(uint8_t *data)
{
    float tmp_float;
    *(((uint8_t *) (&tmp_float)) + 0) = data[0];
    *(((uint8_t *) (&tmp_float)) + 1) = data[1];
    *(((uint8_t *) (&tmp_float)) + 2) = data[2];
    *(((uint8_t *) (&tmp_float)) + 3) = data[3];
    return tmp_float;
}

int int32_to_data(int32_t val, uint8_t *data)
{
    data[0] = *(((uint8_t *) (&val)) + 0);
    data[1] = *(((uint8_t *) (&val)) + 1);
    data[2] = *(((uint8_t *) (&val)) + 2);
    data[3] = *(((uint8_t *) (&val)) + 3);
    return 4;
}



uint32_t data_to_uint32(uint8_t *data)
{
    uint32_t tmp;
    *(((uint8_t *) (&tmp)) + 0) = data[0];
    *(((uint8_t *) (&tmp)) + 1) = data[1];
    *(((uint8_t *) (&tmp)) + 2) = data[2];
    *(((uint8_t *) (&tmp)) + 3) = data[3];
    return tmp;
}


int ReadMotorData(MI_Motor *Motor)
{
  int ok=1;
  // 检查并处理接收到的CAN帧
  CANMessage frame;
  while(ACAN_ESP32::can.receive (frame)) 
  {
    int32_t ccgv1 = 0; 
    int32_t data_in32 = 0; 
    float ccgv2 = 0; 
    
    uint8_t node_id = GET_NODE_ID(frame.id);  // 提取帧中的节点ID
    uint8_t cmd     = GET_CMD(frame.id);      // 提取帧中的命令码
    //DEBUG.print (" ID: ") ;
    //DEBUG.print (node_id) ;     
    // 筛选ID
    if(node_id <= 6) 
    {
      // Track the responder by its actual CAN node ID. This is diagnostic-only
      // metadata and does not alter the existing motor-control data path.
      MI_Motor *responseMotor = &JointMotor[node_id - 1];
      responseMotor->gReceivedFrameCount++;
      responseMotor->last_tx_ms = millis();
      switch (cmd) 
      {
        case CAN_CMD_MOTOR_DISABLE:  /**< 0禁用电机，停止所有控制输出 */
            ccgv1 = data_to_int32(frame.data); 
            if(ccgv1==0)
            {
              Motor->MCStatusword.status.switched_on = 0;  // 关闭电源标志
              responseMotor->MCStatusword.status.switched_on = 0;
              //DEBUG.println (" 电机停止所有控制输出 ") ;
            }
            else
            {
              DEBUG.print (" 电机停止出错 错误码：") ;
              DEBUG.println (ccgv1);   
            }                         
            break;
    
        case CAN_CMD_MOTOR_ENABLE:  /**< 1启用电机，允许控制输出 */
            ccgv1 = data_to_int32(frame.data); 
            if(ccgv1==0)
            {
              Motor->MCStatusword.status.switched_on = 1;  // 设备已接通电源
              responseMotor->MCStatusword.status.switched_on = 1;
              //DEBUG.println (" 允许电机控制输出 ") ;
            }
            else
            {
              DEBUG.print (" 电机启用出错 错误码：") ;
              DEBUG.println (ccgv1);   
            }                         
            break;
        case CAN_CMD_GET_STATUSWORD: //13获取状态字与错误码
            Motor->MCStatusword.status.status_code = data_to_uint32(frame.data); 
            Motor->MCStatusword.errors.errors_code = data_to_uint32(&frame.data[4]); 
            responseMotor->MCStatusword.status.status_code = data_to_uint32(frame.data);
            responseMotor->MCStatusword.errors.errors_code = data_to_uint32(&frame.data[4]);
            ok=0;
            break;
    
        case CAN_CMD_STATUSWORD_REPORT:  /**< 14状态字报告响应 */
            Motor->MCStatusword.status.status_code = data_to_uint32(frame.data); 
            Motor->MCStatusword.errors.errors_code = data_to_uint32(&frame.data[4]); 
            responseMotor->MCStatusword.status.status_code = data_to_uint32(frame.data);
            responseMotor->MCStatusword.errors.errors_code = data_to_uint32(&frame.data[4]);
            /*
            DEBUG.print (" 电机状态字报告响应 ID") ;
            DEBUG.print (node_id) ; 
            DEBUG.print (" status:") ;
            DEBUG.print (Motor->MCStatusword.status.status_code, BIN) ; 
            DEBUG.print (" errors:") ;
            DEBUG.println (Motor->MCStatusword.errors.errors_code, BIN) ; 
            */
            break;

            
        case CAN_CMD_GET_VALUE_1:  /**< 15获取第一组参数值(见fill_value函数) */
            Motor->Turn = uint16_to_float(frame.data[0]<<8|frame.data[1],MIN_P,MAX_P,16);//转     
            if(node_id<5)
              Motor->Angle = fullTurnToAngle(Motor->Turn) + ConfigurationParameter.JointAngleDeviation;
            else 
              Motor->Angle = fullTurnToAngle(Motor->Turn);
              
            Motor->Turn_s = uint16_to_float(frame.data[2]<<8|frame.data[3],V_MIN,V_MAX,16) + Speed_Conversion_Deviation;//-50转/s~50转/s) 
            Motor->Speed = rpsToDegreesPerSecond(Motor->Turn_s);//将圈每秒(rps)转换为角度每秒(deg/s)
            Motor->Torque = uint16_to_float(frame.data[4]<<8|frame.data[5],T_MIN,T_MAX,16);//[0~65535]对应（-12Nm~12Nm）      
            Motor->Temp = uint16_to_float((frame.data[6]<<8|frame.data[7]),MIN_T,MAX_T,16);//摄氏度
            DEBUG.print (" p: ") ;
            DEBUG.print (Motor->Angle,3) ;       
            DEBUG.print (" v: ") ;
            DEBUG.print (Motor->Speed,3) ; 
            DEBUG.print (" t: ") ;
            DEBUG.print (Motor->Torque,3) ; 
            DEBUG.print (" T: ") ;
            DEBUG.println (Motor->Temp,3) ;                            
            break;
            
        case CAN_CMD_GET_VALUE_2:  /**< 16获取第二组参数值(见fill_value函数) */
            ccgv2 = data_to_float(frame.data); 
            DEBUG.print ("请求变量: ") ;
            DEBUG.println (ccgv2,5) ;           
            break;

        case CAN_CMD_SET_CONFIG:  /**< 17设置特定配置参数 */
            ccgv1 = data_to_int32(frame.data); 
            data_in32 = data_to_int32(&frame.data[4]);
            switch (ccgv1) 
            {
              case CONTROL_MODE_ENUM: //电机工作模式
                  DEBUG.print (" ID: ") ;
                  DEBUG.print (node_id) ; 
                  if(data_in32==0)
                  {
                    Motor->UsrConfig.control_mode = data_in32;
                    DEBUG.print (" 转矩爬升模式：") ;
                    DEBUG.println (Motor->UsrConfig.control_mode) ; 
                  }
                  else if(data_in32==1)
                  {
                    Motor->UsrConfig.control_mode = data_in32;
                    DEBUG.print (" 转速爬升模式：") ;
                    DEBUG.println (Motor->UsrConfig.control_mode) ; 
                  }      
                  else if(data_in32==2)
                  {
                    Motor->UsrConfig.control_mode = data_in32;
                    DEBUG.print (" 位置过滤波模式：") ;
                    DEBUG.println (Motor->UsrConfig.control_mode) ; 
                  } 
                  else if(data_in32==3)
                  {
                    Motor->UsrConfig.control_mode = data_in32;
                    DEBUG.print (" 轮廓位置模式：") ;
                    DEBUG.println (Motor->UsrConfig.control_mode) ; 
                  }      
                  else if(data_in32==4)
                  {
                    Motor->UsrConfig.control_mode = data_in32;
                    DEBUG.print (" MIT模式：") ;
                    DEBUG.println (Motor->UsrConfig.control_mode) ; 
                  }                
                  else
                  {
                    DEBUG.print (" 工作模式配置错误：") ;
                    DEBUG.println (data_in32) ;                     
                  }                          
                  break;
                  
              case INVERT_MOTOR_DIR_ENUM: //电机方向
                  DEBUG.print (" ID: ") ;
                  DEBUG.print (node_id) ; 
                  if(data_in32==0)
                  {
                    Motor->UsrConfig.invert_motor_dir = data_in32;
                    DEBUG.print (" 电机方向 正常：") ;
                    DEBUG.println (Motor->UsrConfig.invert_motor_dir) ; 
                  }
                  else if(data_in32==1)
                  {
                    Motor->UsrConfig.invert_motor_dir = data_in32;
                    DEBUG.print (" 电机方向 反转：") ;
                    DEBUG.println (Motor->UsrConfig.invert_motor_dir) ; 
                  }    
                  break;
              
              default:
                DEBUG.print (" 其它设置：") ;
                DEBUG.print (" ID: ") ;
                DEBUG.print (node_id) ;  
                DEBUG.println (ccgv1) ;   
                break;
                  
            }         
            break;

    
        case CAN_CMD_SET_TORQUE_MIT:  /**< 23MIT控制模式 */
            Motor->Turn = uint16_to_float(frame.data[0]<<8|frame.data[1],MIN_P,MAX_P,16);//转
            if(node_id<5)
              Motor->Angle = fullTurnToAngle(Motor->Turn) + ConfigurationParameter.JointAngleDeviation;
            else 
              Motor->Angle = fullTurnToAngle(Motor->Turn);
              
            Motor->Turn_s = uint16_to_float(frame.data[2]<<8|frame.data[3],V_MIN,V_MAX,16) + Speed_Conversion_Deviation;//-50转/s~50转/s) 
            Motor->Speed = rpsToDegreesPerSecond(Motor->Turn_s);
            Motor->Torque = uint16_to_float(frame.data[4]<<8|frame.data[5],T_MIN,T_MAX,16);//[0~65535]对应（-12Nm~12Nm）      
            Motor->Temp = uint16_to_float((frame.data[6]<<8|frame.data[7]),MIN_T,MAX_T,16);//摄氏度
            /*
            DEBUG.print (" p: ") ;
            DEBUG.print (Motor->Angle,3) ;   
            DEBUG.print (" ts: ") ;
            DEBUG.print (Motor->Turn_s,3) ; 
            DEBUG.print (" v: ") ;
            DEBUG.print (Motor->Speed,3) ; 
            DEBUG.print (" t: ") ;
            DEBUG.print (Motor->Torque,3) ; 
            DEBUG.print (" T: ") ;
            DEBUG.println (Motor->Temp,3) ;          
            */                  
            break;
    
        case CAN_CMD_CALIB_REPORT:   /**< 6请求校准结果报告 */
            ccgv1 = data_to_int32(frame.data); 
            ccgv2 = data_to_float(&frame.data[4]); 
            DEBUG.print ("步骤: ") ;
            DEBUG.println (ccgv1) ;    
            DEBUG.print ("数据: ") ;
            DEBUG.println (ccgv2,5) ;                         
            break;

        case CAN_CMD_SET_HOME:   /**< 11设置当前位置为零点位置 */
                  DEBUG.print (" ID: ") ;
                  DEBUG.print (node_id) ; 
                  if(data_in32==0)
                  {
                    DEBUG.println (" 设置当前位置为零点位置 ") ;
                    Motor->HomeOk = 1;
                  }
                  else
                  {
                    DEBUG.print (" 设置零点出错：") ;
                    DEBUG.println (data_in32) ; 
                    Motor->HomeOk = 0;
                  }                      
            break;
            
    
        default:
        // 打印接收到的消息
    
            DEBUG.print ("其它命令 cmd: ") ;
            DEBUG.print (cmd) ;     
            DEBUG.print (" frame.len: ") ;
            DEBUG.print (frame.len) ;         
            DEBUG.print (" 收到数据: ") ;
            for (int i = 0; i < frame.len; ++i) {
              if (frame.data[i] < 0x10) DEBUG.print("0"); // 补零
              DEBUG.print (frame.data[i], HEX) ;
              DEBUG.print (" ") ;
            }
            DEBUG.println() ;
      
        
            break;
      }      
    
      
    }
    else
    {
      // 打印接收到的消息
      DEBUG.print ("错误ID: ") ;
      DEBUG.print (node_id) ;
      DEBUG.print (" cmd: ") ;
      DEBUG.print (cmd) ;      
      DEBUG.print (" 收到数据: ") ;
      for (int i = 0; i < frame.len; ++i) {
        if (frame.data[i] < 0x10) DEBUG.print("0"); // 补零
        DEBUG.print (frame.data[i], HEX) ;
        DEBUG.print (" ") ;
      }
      DEBUG.println() ;  
    } 
    //DEBUG.println() ;          
  }
  return ok;
 
}





/// CAN Command Packet Structure ///
/// 16 bit position command, between -4*pi and 4*pi
/// 12 bit velocity command, between -30 and + 30 rad/s
/// 12 bit kp, between 0 and 500 N-m/rad
/// 12 bit kd, between 0 and 100 N-m*s/rad
/// 12 bit feed forward torque, between -18 and 18 N-m
/// CAN Packet is 8 8-bit words
/// Formatted as follows.  For each quantity, bit 0 is LSB
/// 0: [position[15-8]]
/// 1: [position[7-0]] 
/// 2: [velocity[11-4]]
/// 3: [velocity[3-0], kp[11-8]]
/// 4: [kp[7-0]]
/// 5: [kd[11-4]]
/// 6: [kd[3-0], torque[11-8]]
/// 7: [torque[7-0]]

void pack_cmd(uint8_t* msg, float p_des, float v_des, float kp, float kd, float t_ff)
{
     /// limit data to be within bounds ///
     p_des = fminf(fmaxf(P_MIN, p_des), P_MAX);                    
     v_des = fminf(fmaxf(V_MIN, v_des), V_MAX);
     kp    = fminf(fmaxf(KP_MIN, kp), KP_MAX);
     kd    = fminf(fmaxf(KD_MIN, kd), KD_MAX);
     t_ff  = fminf(fmaxf(T_MIN, t_ff), T_MAX);
     /// convert floats to unsigned ints ///
     int p_int = float_to_uint(p_des, P_MIN, P_MAX, 16);            
     int v_int = float_to_uint(v_des, V_MIN, V_MAX, 12);
     int kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12);
     int kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 12);
     int t_int = float_to_uint(t_ff, T_MIN, T_MAX, 12);
     /// pack ints into the can buffer ///
     msg[0] = p_int>>8;                                       
     msg[1] = p_int&0xFF;
     msg[2] = v_int>>4;
     msg[3] = ((v_int&0xF)<<4)|(kp_int>>8);
     msg[4] = kp_int&0xFF;
     msg[5] = kd_int>>4;
     msg[6] = ((kd_int&0xF)<<4)|(t_int>>8);
     msg[7] = t_int&0xff;
}


void MotorSet(MI_Motor *Motor,uint8_t cmd_id,float value,float value2)
{
  uint32_t id = (Motor->UsrConfig.node_id<<5)& 0x7FF ;

  switch (cmd_id) {
    case CAN_CMD_MOTOR_DISABLE:  // 0禁用电机 待机模式
        Motor->dlc = 0;
        break;

    case CAN_CMD_MOTOR_ENABLE: // 1启用电机
        Motor->dlc = 0;
        id = id + CAN_CMD_MOTOR_ENABLE;
        break;

   //  ---------------------- 控制参数设置类命令 ----------------------    
    case CAN_CMD_SET_TORQUE:   // 2设置扭矩指令
        Motor->dlc = 4;
        id = id + CAN_CMD_SET_TORQUE;
        float_to_data(value,&Motor->data[0]);
        break;


    case CAN_CMD_SET_VELOCITY: // 3设置速度指令
        Motor->dlc = 4;
        id = id + CAN_CMD_SET_VELOCITY;
        float_to_data(value,&Motor->data[0]);
        break;

    case CAN_CMD_SET_POSITION: // 4设置位置指令
        Motor->dlc = 4;
        id = id + CAN_CMD_SET_POSITION;
        float_to_data(value,&Motor->data[0]);
        break;
    // ---------------------- 校准与特殊功能命令 ----------------------
    case CAN_CMD_CALIB_START: // 5启动校准
        Motor->dlc = 0;
        id = id + CAN_CMD_CALIB_START;
        break;

    case CAN_CMD_CALIB_ABORT: //6终止校准
        Motor->dlc = 0;
        id = id + CAN_CMD_CALIB_ABORT;
        break;

    case CAN_CMD_ANTICOGGING_START: //启动齿槽效应补偿
         Motor->dlc = 0;
         id = id + CAN_CMD_ANTICOGGING_START;
         break;

    case CAN_CMD_ANTICOGGING_ABORT: //终止齿槽效应补偿
         Motor->dlc = 0;
         id = id + CAN_CMD_ANTICOGGING_ABORT;
         break;

    case CAN_CMD_SET_HOME: //设置机械原点
        Motor->dlc = 0;
        id = id + CAN_CMD_SET_HOME;
        break;

    case CAN_CMD_ERROR_RESET: //重置错误状态
        Motor->dlc = 0;
        id = id + CAN_CMD_ERROR_RESET;
        break;

    case CAN_CMD_GET_STATUSWORD: //获取状态字与错误码
        Motor->dlc = 0;
        id = id + CAN_CMD_GET_STATUSWORD;
        break;

    case CAN_CMD_GET_VALUE_1:  //15获取自定义数值
        Motor->dlc = 1;
        id = id + CAN_CMD_GET_VALUE_1;
        Motor->data[0] = (uint8_t)value;
        break;
        
    case CAN_CMD_GET_VALUE_2:
        Motor->dlc = 1;
        id = id + CAN_CMD_GET_VALUE_2;
        Motor->data[0] = (uint8_t)value;
        break;

    case CAN_CMD_SET_CONFIG:   //17设置设备配置
        Motor->dlc = 8;
        id = id + CAN_CMD_SET_CONFIG;
        int32_to_data((int32_t)value,&Motor->data[0]);
        int32_to_data((int32_t)value2,&Motor->data[4]);//0:转矩 1:转速 2：位置过滤波  3：轮廓位置 4：运控模式
        break;

    case CAN_CMD_GET_CONFIG:   //18获取设备配置

        break;

    case CAN_CMD_SAVE_ALL_CONFIG: //保存所有配置到 Flash
         Motor->dlc = 0;
         id = id + CAN_CMD_SAVE_ALL_CONFIG;
         break;

    case CAN_CMD_RESET_ALL_CONFIG: //恢复出厂配置

        break;

    case CAN_CMD_SYNC: //触发参数同步

        break;

    case CAN_CMD_GET_FW_VERSION: //获取固件版本

        break;

    case CAN_CMD_DFU_START:  //启动固件升级（DFU）

        break;

    case CAN_CMD_DFU_DATA:  //写入升级数据

      break;

   case CAN_CMD_DFU_END:  //结束升级并跳转引导程序
     
      break;

    case CAN_CMD_SET_TORQUE_MIT:   // 32 MIT指令
        Motor->dlc = 8;
        id = id + CAN_CMD_SET_TORQUE_MIT;
        pack_cmd(&Motor->data[0],0, 0,0,0,value);
 
        //DEBUG.print ("value数据tx：") ;
        //DEBUG.println (value,5) ;
        break;

    default:
        break;
    }
  
  sendCANMessage(id, Motor->data, Motor->dlc,false);  
}



void MotorMit(MI_Motor *Motor,float p_des, float v_des, float kp, float kd, float t_ff)
{
  int txTime = 333;
  uint32_t id = (Motor->UsrConfig.node_id<<5)& 0x7FF ;

  Motor->dlc = 8;
  id = id + CAN_CMD_SET_TORQUE_MIT;
  pack_cmd(&Motor->data[0],p_des,v_des,kp,kd,t_ff);

  sendCANMessage(id, Motor->data, Motor->dlc,false);  
  delayMicroseconds(txTime);
  ReadMotorData(Motor); //读数据 
}

int StopMotors(MI_Motor *Motor)//返回0 电机停止成功
{
  int txTime = 333;  
  int ok = 1;
  MotorSet(Motor,CAN_CMD_MOTOR_DISABLE,0,0);//停止电机1
  delayMicroseconds(txTime);
  ReadMotorData(Motor); //读数据 
  if(Motor->MCStatusword.status.switched_on==0)
     ok = 0;

  return ok;
}

int EnableMotor(MI_Motor *Motor)//返回0 使能电机成功
{
  int txTime = 2222;  
  int ok = 1;
  MotorSet(Motor,CAN_CMD_MOTOR_ENABLE,0,0);
  delayMicroseconds(txTime);
  ReadMotorData(Motor); //读数据 
  if(Motor->MCStatusword.status.switched_on==1)
     ok = 0;

  return ok;
}

int ConfigureMotorMode(MI_Motor *Motor,int32_t Mode)//配置电机模式 返回0 成功  Mode = 0:转矩 1:转速 2：位置过滤波  3：轮廓位置 4：运控模式
{
  int txTime = 333;  
  int ok = 0;
  //工作模式
  MotorSet(Motor,CAN_CMD_SET_CONFIG,CONTROL_MODE_ENUM,Mode);//
  delayMicroseconds(txTime);
  ReadMotorData(Motor); //读数据 
  if(Motor->UsrConfig.control_mode != Mode)
     ok = 1;
     
  return ok;
}

int GetMotorStatusWord(MI_Motor *Motor)//获取电机当前状态字
{
  int txTime = 333;  
  int ok = 0;
  MotorSet(Motor,CAN_CMD_GET_STATUSWORD,0,0);//请求获取当前状态字
  delayMicroseconds(txTime);
  ok = ReadMotorData(Motor); //读数据 
     
  return ok;
}

void GetAllMotorStatusWord(void)//获取所有电机当前状态字
{
  int ok = 0;
  if(GetMotorStatusWord(&JointMotor[0]))
    DEBUG.println (" 获取电机1当前状态出错 ") ;
  else 
    ok++;     
  
  if(GetMotorStatusWord(&JointMotor[1]))
    DEBUG.println (" 获取电机2当前状态出错 ") ;
  else 
    ok++;    
    
  if(GetMotorStatusWord(&JointMotor[2]))
    DEBUG.println (" 获取电机3当前状态出错 ") ;
  else 
    ok++;    
    
  if(GetMotorStatusWord(&JointMotor[3]))
    DEBUG.println (" 获取电机4当前状态出错 ") ;
  else 
    ok++;    
    
  if(GetMotorStatusWord(&JointMotor[4]))
    DEBUG.println (" 获取电机5当前状态出错 ") ;
  else 
    ok++;    
    
  if(GetMotorStatusWord(&JointMotor[5]))
    DEBUG.println (" 获取电机6当前状态出错 ") ;
  else 
    ok++;    
/*
  if(ok==6)
    DEBUG.println (" 已经获取6个电机当前状态 ") ;
  else
  {
    DEBUG.print(" 获取6个电机当前状态成功:");
    DEBUG.println(ok);//   
  }
*/    
}


void SetAllJointMode(int32_t Mode)
{ 
  for (int x = 0; x < 6; x++)
    JointMotor[x].UsrConfig.control_mode = 0;
    
  if(ConfigureMotorMode(&JointMotor[0],Mode)==1)
    DEBUG.println (" 电机1设置工作模式出错 ") ;
  if(ConfigureMotorMode(&JointMotor[1],Mode)==1)
    DEBUG.println (" 电机2设置工作模式出错 ") ;
  if(ConfigureMotorMode(&JointMotor[2],Mode)==1)
    DEBUG.println (" 电机3设置工作模式出错 ") ;
  if(ConfigureMotorMode(&JointMotor[3],Mode)==1)
    DEBUG.println (" 电机4设置工作模式出错 ") ;  
  if(ConfigureMotorMode(&JointMotor[4],Mode)==1)
    DEBUG.println (" 电机5设置工作模式出错 ") ;
  if(ConfigureMotorMode(&JointMotor[5],Mode)==1)
    DEBUG.println (" 电机6设置工作模式出错 ") ;
}


int SetAllJointMode_n(int n)
{
  int ok = 1;
  SetAllJointMode(CONTROL_MODE_MOTION);
  DEBUG.println (" ") ;
  for (int axis = 0; axis < n; axis++)
  {
    if((JointMotor[0].UsrConfig.control_mode==4)&&(JointMotor[1].UsrConfig.control_mode==4)&&(JointMotor[2].UsrConfig.control_mode==4)
      &&(JointMotor[3].UsrConfig.control_mode==4)&&(JointMotor[4].UsrConfig.control_mode==4)&&(JointMotor[5].UsrConfig.control_mode==4))
    {
      DEBUG.println ("所有电机工作模式设置成功") ;
      ok = 0;
      return ok;
      
    }
    else
    {
      SetAllJointMode(CONTROL_MODE_MOTION);
      DEBUG.println ("有电机工作模式设置失败 重试") ;
    }    
  }
  return ok;
}




void EnableAllMotors(void)
{
  DEBUG.println (" ") ;
  if(EnableMotor(&JointMotor[0])==1)
    DEBUG.println (" 电机1启动出错 ") ;

  if(EnableMotor(&JointMotor[1])==1)
    DEBUG.println (" 电机2启动出错 ") ;

  if(EnableMotor(&JointMotor[2])==1)
    DEBUG.println (" 电机3启动出错 ") ;

  if(EnableMotor(&JointMotor[3])==1)
    DEBUG.println (" 电机4启动出错 ") ;  
    
  if(EnableMotor(&JointMotor[4])==1)
    DEBUG.println (" 电机5启动出错 ") ;

  if(EnableMotor(&JointMotor[5])==1)
    DEBUG.println (" 电机6启动出错 ") ;

}



int EnableAllMotors_n(int n)
{
  int ok = 1;
  EnableAllMotors();
  for (int axis = 0; axis < n; axis++)
  {
    if((JointMotor[0].MCStatusword.status.switched_on==1)&&(JointMotor[1].MCStatusword.status.switched_on==1)&&(JointMotor[2].MCStatusword.status.switched_on==1)
      &&(JointMotor[3].MCStatusword.status.switched_on==1)&&(JointMotor[4].MCStatusword.status.switched_on==1)&&(JointMotor[5].MCStatusword.status.switched_on==1))
    {
      DEBUG.println ("使能所有电机成功") ;
      ok = 0;
      return ok;
      
    }
    else
    {
      EnableAllMotors();  
      DEBUG.println ("有电机使能失败 重试") ;
    }    
  }
  return ok;
}




int GetMotorStatusSord(MI_Motor *Motor)// /**< 13请求获取当前状态字 */
{
  int txTime = 333;  
  int ok = 0;
  MotorSet(Motor,CAN_CMD_GET_STATUSWORD,0,0);//请求获取当前状态字
  delayMicroseconds(txTime);
  if(ReadMotorData(Motor)==1)
     ok = 1;
  return ok;
}



void StopAllMotors(void)
{
  if(StopMotors(&JointMotor[0])==1)
    DEBUG.println (" 电机1停止出错 ") ;

  if(StopMotors(&JointMotor[1])==1)
    DEBUG.println (" 电机2停止出错 ") ;

  if(StopMotors(&JointMotor[2])==1)
    DEBUG.println (" 电机3停止出错 ") ;

  if(StopMotors(&JointMotor[3])==1)
    DEBUG.println (" 电机4停止出错 ") ;  
    
  if(StopMotors(&JointMotor[4])==1)
    DEBUG.println (" 电机5停止出错 ") ;

  if(StopMotors(&JointMotor[5])==1)
    DEBUG.println (" 电机6停止出错 ") ;

}

int StopAllMotorStop_n(int n)
{
  int ok = 1;
  StopAllMotors();
  DEBUG.println (" ") ;
  for (int axis = 0; axis < n; axis++)
  {
    if((JointMotor[0].MCStatusword.status.switched_on==0)&&(JointMotor[1].MCStatusword.status.switched_on==0)&&(JointMotor[2].MCStatusword.status.switched_on==0)
      &&(JointMotor[3].MCStatusword.status.switched_on==0)&&(JointMotor[4].MCStatusword.status.switched_on==0)&&(JointMotor[5].MCStatusword.status.switched_on==0))
    {
      DEBUG.println ("停止所有电机") ;
      ok = 0;
      return ok;
      
    }
    else
    {
      StopAllMotors();  
      DEBUG.println ("有电机停止失败 重试") ;
    }    
  }
  return ok;
}



void CanInit(void)
{
  // 配置CAN总线参数
  DEBUG.println ("配置ESP32 CAN...") ;
  ACAN_ESP32_Settings settings (DESIRED_BIT_RATE);
  settings.mRequestedCANMode = ACAN_ESP32_Settings::NormalMode ; // 标准模式(非回环)
  settings.mTxPin = CAN_TX_PIN; // 设置发送引脚
  settings.mRxPin = CAN_RX_PIN; // 设置接收引脚

  // 初始化CAN控制器
  const uint32_t errorCode = ACAN_ESP32::can.begin (settings) ;
  
  // 检查初始化结果
  if (errorCode == 0) {
    DEBUG.println ("CAN初始化成功！") ;
    DEBUG.print ("实际波特率: ") ;
    DEBUG.print (settings.actualBitRate ()) ;
    DEBUG.println (" bit/s") ;
  } else {
    DEBUG.print ("CAN初始化失败，错误代码: 0x") ;
    DEBUG.println (errorCode, HEX) ;
    SetDiagnosticBootFault(BOOT_STAGE_CAN_FAULT, DIAG_FAULT_CAN_INIT);
    while (1) {
      RosSerial_Diagnostics_TXD();
      delay(500);
    }
  }  
}


int ReverseMotorDirection(MI_Motor *Motor,int32_t Dir)// 电机方向反转使能(0=不反转,1=反转)
{
  int txTime = 333;  
  int ok = 0;
  //工作模式
  MotorSet(Motor,CAN_CMD_SET_CONFIG,INVERT_MOTOR_DIR_ENUM,Dir);//
  delayMicroseconds(txTime);
  ReadMotorData(Motor); //读数据 
  if(Motor->UsrConfig.invert_motor_dir != Dir)
     ok = 1;
     
  return ok;
}

void SaveAllConfigurations(MI_Motor *Motor)//保存所有配置到 Flash
{
  int txTime = 111;  
  int ok = 1;
  MotorSet(Motor,CAN_CMD_SAVE_ALL_CONFIG,0,0);
  delay(txTime);
  ReadMotorData(Motor); //读数据 
}

int SetMotorHome(MI_Motor *Motor)//配置电机零点
{
  int txTime = 666;  
  int ok = 0;
  MotorSet(Motor,CAN_CMD_SET_HOME,0,0);//设置当前位置为零点位置
  delayMicroseconds(txTime);
  ReadMotorData(Motor); //读数据 
  //if(Motor->UsrConfig.control_mode != Mode)
     //ok = 1;   
  return ok;
}


void SetAllMotorHome(void)
{
  for (int axis = 0; axis < 6; axis++)
    JointMotor[axis].HomeOk = 0;      
  
  SetMotorHome(&JointMotor[0]);//配置电机零点
  SetMotorHome(&JointMotor[1]);//配置电机零点
  SetMotorHome(&JointMotor[2]);//配置电机零点
  SetMotorHome(&JointMotor[3]);//配置电机零点
  SetMotorHome(&JointMotor[4]);//配置电机零点
  SetMotorHome(&JointMotor[5]);//配置电机零点
}



int SetMotorDirection_n(int n)
{
  int ok = 1;
  DEBUG.println (" ") ;
  ReverseMotorDirection(&JointMotor[0],1);//电机1 方向反转：0=正常 1=反转  
  ReverseMotorDirection(&JointMotor[3],1);//电机3 方向反转：0=正常 1=反转  
  ReverseMotorDirection(&JointMotor[4],1);//电机5 方向反转：0=正常 1=反转  

  for (int axis = 0; axis < n; axis++)
  {

    if((JointMotor[0].UsrConfig.invert_motor_dir==1)&&(JointMotor[0].UsrConfig.invert_motor_dir==1)&&(JointMotor[0].UsrConfig.invert_motor_dir==1))
    {
      DEBUG.println ("电机设置反转成功") ;
      ok = 0;
      return ok;
      
    }
    else
    {
      ReverseMotorDirection(&JointMotor[0],1);//电机1 方向反转：0=正常 1=反转  
      ReverseMotorDirection(&JointMotor[3],1);//电机3 方向反转：0=正常 1=反转  
      ReverseMotorDirection(&JointMotor[4],1);//电机5 方向反转：0=正常 1=反转  
      DEBUG.println ("有电机设置反转失败 ") ;
    }    
  }
  return ok;
} 

int SetAllMotorHome_n(int n)
{
  int ok = 1;
  DEBUG.println (" ") ;
  SetAllMotorHome();

  for (int axis = 0; axis < n; axis++)
  {
    for (int x = 0; x < 6; x++)
    {
        DEBUG.print ("  ID") ;
        DEBUG.print (x+1) ;   
        DEBUG.print ("HomeOk:  ") ;
        DEBUG.print (JointMotor[x].HomeOk) ;         
    }
    DEBUG.println (" ") ;
    
    if((JointMotor[0].HomeOk==1)&&(JointMotor[1].HomeOk==1)&&(JointMotor[2].HomeOk==1)&&(JointMotor[3].HomeOk==1)&&(JointMotor[4].HomeOk==1)&&(JointMotor[5].HomeOk==1))
    {
      DEBUG.println ("所有电机设置零点成功") ;
      ok = 0;
      return ok;
      
    }
    else
    {
      SetAllMotorHome();  
      DEBUG.println ("有电机设置零点失败 ") ;
    }    
  }
  return ok;
}




void MotorInit(void) // 电机初始化
{
  pinMode(Buzzer_Pin, OUTPUT);
  digitalWrite(Buzzer_Pin, LOW);   //不响

  //初始化二阶低通滤波
  for (int axis = 0; axis < 6; axis++)
  {
    biquadFilterInitLPF(&MotorFilterLPF[axis], 50, 1000);
  }

  
  for (int axis = 0; axis < 6; axis++)
    JointMotor[axis].HomeOk = 0;
  
  JointMotor[0].UsrConfig.node_id = 1;
  JointMotor[1].UsrConfig.node_id = 2;
  JointMotor[2].UsrConfig.node_id = 3;
  JointMotor[3].UsrConfig.node_id = 4;
  JointMotor[4].UsrConfig.node_id = 5;
  JointMotor[5].UsrConfig.node_id = 6;

  for (int i = 0; i < 6; i++)
  {
    JointMotor[i].kp = 0;
    JointMotor[i].kd = 0;
    JointMotor[i].in_p_des = 0;
    JointMotor[i].in_v_des = 0;
    JointMotor[i].in_torque = 0;
  }
  
  CanInit();
  delay (1000) ;//等待电机上电稳定

  if(StopAllMotorStop_n(1)==1)//停止电机
  {
    SetDiagnosticBootFault(BOOT_STAGE_MOTOR_STOP_FAULT, DIAG_FAULT_MOTOR_STOP);
    DEBUG.println ("电机停止异常 1.尝试重新上 2.电检查供电与CAN接线是否正常");
    while(1)
    {
      RosSerial_Diagnostics_TXD();
      digitalWrite(Buzzer_Pin, HIGH);   //响
      delay(100);
      digitalWrite(Buzzer_Pin, LOW);   //响
      delay(100);   
      DEBUG.println ("电机停止异常 1.尝试重新上 2.电检查供电与CAN接线是否正常");         
    }
    
  }
  

  delay (1000) ;//等待电机稳定
  if(SetAllMotorHome_n(1)==1)//电机零点
  {
    SetDiagnosticBootFault(BOOT_STAGE_MOTOR_HOME_FAULT, DIAG_FAULT_MOTOR_HOME);
    DEBUG.println ("电机设置零点异常 1.尝试重新上 2.电检查供电与CAN接线是否正常");
    while(1)
    {
      RosSerial_Diagnostics_TXD();
      digitalWrite(Buzzer_Pin, HIGH);   //响
      delay(200);
      digitalWrite(Buzzer_Pin, LOW);   //响
      delay(200);  
      
      DEBUG.println ("电机设置零点异常 1.尝试重新上 2.电检查供电与CAN接线是否正常");          
    }
  }
  
    
  if(SetMotorDirection_n(1)==1)//反转
  {
    SetDiagnosticBootFault(BOOT_STAGE_MOTOR_DIRECTION_FAULT, DIAG_FAULT_MOTOR_DIRECTION);
    DEBUG.println ("电机设置反转异常 1.尝试重新上 2.电检查供电与CAN接线是否正常");
    while(1)
    {
      RosSerial_Diagnostics_TXD();
      digitalWrite(Buzzer_Pin, HIGH);   //响
      delay(300);
      digitalWrite(Buzzer_Pin, LOW);   //响
      delay(300);   
      DEBUG.println ("电机设置反转异常 1.尝试重新上 2.电检查供电与CAN接线是否正常");         
    }  
  }

  if(SetAllJointMode_n(1)==1)//工作模式
  {
    SetDiagnosticBootFault(BOOT_STAGE_MOTOR_MODE_FAULT, DIAG_FAULT_MOTOR_MODE);
    DEBUG.println ("电机工作模式异常 1.尝试重新上 2.电检查供电与CAN接线是否正常");
    while(1)
    {
      RosSerial_Diagnostics_TXD();
      digitalWrite(Buzzer_Pin, HIGH);   //响
      delay(400);
      digitalWrite(Buzzer_Pin, LOW);   //响
      delay(400);  
      DEBUG.println ("电机工作模式异常 1.尝试重新上 2.电检查供电与CAN接线是否正常");          
    }        
  }
  
  if(EnableAllMotors_n(1)==1)//启动电机
  {
    SetDiagnosticBootFault(BOOT_STAGE_MOTOR_ENABLE_FAULT, DIAG_FAULT_MOTOR_ENABLE);
    DEBUG.println ("有电机启动电机异常 1.尝试重新上 2.电检查供电与CAN接线是否正常");
    while(1)
    {
      RosSerial_Diagnostics_TXD();
      digitalWrite(Buzzer_Pin, HIGH);   //响
      delay(500);
      digitalWrite(Buzzer_Pin, LOW);   //响
      delay(500); 
      DEBUG.println ("有电机启动电机异常 1.尝试重新上 2.电检查供电与CAN接线是否正常");           
    }
  }
  
  SetJointAngles(0,0,0,0,0.1,0.1);
  SetWheelTorque(0,0);
}


/**
 * 将圈每秒(rps)转换为角度每秒(deg/s)
 * @param TurnPerSecond 圈每秒的转速
 * @return 对应的角度每秒的转速
 */
float rpsToDegreesPerSecond(float TurnPerSecond) 
{
    return TurnPerSecond * 360.0/Motor_Reduction_Ratio;
}

/**
 * 将角度每秒(deg/s)转换为圈每秒(rps)
 * @param degreesPerSecond 角度每秒的转速
 * @return 对应的圈每秒的转速
 */
float degreesPerSecondToRps(float degreesPerSecond) 
{
    return degreesPerSecond / 360.0*Motor_Reduction_Ratio;
}


/**
 * 将角度值转换为对应的旋转圈数
 * @param angle 输入的角度值
 * @return 对应的圈数，浮点数表示
 */
float angleToTurn(float angle) 
{
    float Turn = angle / 360;   
    return Turn*Motor_Reduction_Ratio;//减速比8：1
}


/**
 * 将多圈数值转换为对应的完整角度值
 * @param Turn 输入的圈数值（可以为任意浮点数，正或负）
 * @return 对应的角度值（例如2.5圈返回900度，-1.25圈返回-450度）
 */
float fullTurnToAngle(float turn)
{
    // 直接计算总角度（圈数乘以360度）
    return turn * 360.0/Motor_Reduction_Ratio;
}


// 角度转弧度的函数
double degreesToRadians(double degrees) {
    return degrees * M_PI / 180.0;
}


void SetJointAngles(float a1,float a2,float a3,float a4,float kp,float kd)
{
  if(kp<0)
    kp=0;

  if(kd<0)
    kd=0;
  
  MotorMit(&JointMotor[0],angleToTurn(a1), 0, kp, kd, 0);
  MotorMit(&JointMotor[1],angleToTurn(a2), 0, kp, kd, 0);
  MotorMit(&JointMotor[2],angleToTurn(a3), 0, kp, kd, 0);
  MotorMit(&JointMotor[3],angleToTurn(a4), 0, kp, kd, 0);  
}


void MitSetJointAngles(void)
{
  
  MotorMit(&JointMotor[0],angleToTurn(JointMotor[0].in_p_des), JointMotor[0].in_v_des, JointMotor[0].kp, JointMotor[0].kd, JointMotor[0].in_torque);
  MotorMit(&JointMotor[1],angleToTurn(JointMotor[1].in_p_des), JointMotor[1].in_v_des, JointMotor[1].kp, JointMotor[1].kd, JointMotor[1].in_torque);
  MotorMit(&JointMotor[2],angleToTurn(JointMotor[2].in_p_des), JointMotor[2].in_v_des, JointMotor[2].kp, JointMotor[2].kd, JointMotor[2].in_torque);
  MotorMit(&JointMotor[3],angleToTurn(JointMotor[3].in_p_des), JointMotor[3].in_v_des, JointMotor[3].kp, JointMotor[3].kd, JointMotor[3].in_torque); 
  MotorMit(&JointMotor[4],angleToTurn(JointMotor[4].in_p_des), JointMotor[4].in_v_des, JointMotor[4].kp, JointMotor[4].kd, JointMotor[4].in_torque);
  MotorMit(&JointMotor[5],angleToTurn(JointMotor[5].in_p_des), JointMotor[5].in_v_des, JointMotor[5].kp, JointMotor[5].kd, JointMotor[5].in_torque); 
}


void SetWheelTorque(float left,float right)
{
  MotorMit(&JointMotor[4],0, 0, 0, 0, left);
  MotorMit(&JointMotor[5],0, 0, 0, 0, right);  
}
