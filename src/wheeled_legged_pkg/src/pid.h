/**
 * @file    pid.h
 * @brief   最简 PID 控制器（只有结构体，头文件即全部实现）
 *
 * 特点：
 *   1. 一个结构体 + 一个 update()，没有类、没有虚函数、没有动态内存
 *   2. 带积分抗饱和（iMax）和输出限幅（outMax）—— 这两个不能省，
 *      少了 iMax，电机被堵转时积分会无限累积，一松开就满速冲出去
 *
 * 用法：
 *      Pid pid;
 *      pid.kp = 30.0f;
 *      float out = pid.update(target, measure);
 */

#ifndef PID_H
#define PID_H

struct Pid
{
    // ---------------------------------------------------------- 参数（你调的）
    float kp     = 0.0f;        // 比例
    float ki     = 0.0f;        // 积分
    float kd     = 0.0f;        // 微分
    float outMax = 5.0f;    // 输出限幅（对称，正负同值）
    float iMax   = 0.5f;     // 积分限幅，抗饱和
    int reverse = 0;            //反转
    
    // ---------------------------------------------------------- 内部状态（别手改）
    float integral  = 0.0f;     // 误差积分累计
    float prevError = 0.0f;     // 上一次的误差（用来算微分）

    /** @brief 清空内部状态。切换控制模式 / 重新使能 / 失控后恢复时调用 */
    void reset()
    {
        integral  = 0.0f;
        prevError = 0.0f;
    }

    float normalize_PI(float degree)
    {
        while (degree > 3.14159265358979323846f) degree -= 2*3.14159265358979323846f;
        while (degree < -3.14159265358979323846f) degree += 2*3.14159265358979323846f;
        return degree;
    }

    /**
     * @brief 跑一次 PID
     * @param target   目标值
     * @param measure  当前测量值（必须和 target 同单位）
     * @return         控制输出，已限幅在 ±outMax
     */
    float update(float target, float measure)
    {
        float error = normalize_PI(target - measure);
        // ---- 积分项 + 抗饱和 ----
        integral += error;
        if (integral >  iMax) integral =  iMax;
        if (integral < -iMax) integral = -iMax;
        // ---- 微分项 ----
        float deriv = normalize_PI(error - prevError);
        prevError = error;
        // ---- 三项求和 ----
        float out = kp * error + ki * integral + kd * deriv;
        // ---- 输出限幅 ----
        if (out >  outMax) out =  outMax;
        if (out < -outMax) out = -outMax;
        if(reverse)
            out = -out;
        return out;
    }
};

#endif  // PID_H
