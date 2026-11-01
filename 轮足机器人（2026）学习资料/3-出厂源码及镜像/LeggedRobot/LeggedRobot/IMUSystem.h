#ifndef IMUSYSTEM_H
#define IMUSYSTEM_H

#include <Arduino.h>         
#include "ICM42688.h"        
#include "MahonyFilter.h"    
#include "filter.h"         

#define RAD_TO_DEG 57.29577951f
#define DEG_TO_RAD 0.0174532925f  // 角度转弧度系数
 

namespace ImuConstants {
    constexpr float ACCEL_RANGE      = 8.0f;     
    constexpr float GYRO_RANGE       = 2000.0f;  
    constexpr float INT16_SCALE      = 32768.0f; 
    constexpr float TEMP_SCALE       = 132.48f;  
    constexpr float TEMP_OFFSET      = 25.0f;    
    constexpr uint16_t CALIB_SAMPLES = 1000;     
    constexpr uint16_t GYRO_CAL_THRESHOLD = 40; // 陀螺仪校准阈值，可根据实际情况调整 
}

struct Vector3f {
    union {
        struct { float x, y, z; };
        float data[3];
    };

    Vector3f() : x(0), y(0), z(0) {}
    Vector3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vector3f operator-(const Vector3f& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vector3f operator+(const Vector3f& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vector3f& operator+=(const Vector3f& other) {
        x += other.x; y += other.y; z += other.z; return *this;
    }

    Vector3f operator/(float scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }

    bool isBelow(float threshold) const {
        return (fabs(x) < threshold) && (fabs(y) < threshold) && (fabs(z) < threshold);
    }
};

struct ImuState {
    Vector3f raw_accel;        
    Vector3f raw_gyro;         
    Vector3f calibrated_gyro;  
    Vector3f filtered_accel;   
    Vector3f filtered_gyro;   
    Vector3f filtered_gyro_deg;  
    Vector3f gyro_bias;        
    float roll  = 0.0f;        
    float pitch = 0.0f;        
    float yaw   = 0.0f;        
    float temp  = 0.0f;        
    float q0 = 1.0f;           
    float q1 = 0.0f;           
    float q2 = 0.0f;           
    float q3 = 0.0f;           
};


class IMUSystem {
public:
    // ================== IMU安装位置配置接口 ==================
    /**
     * @brief 设置IMU安装姿态转换矩阵及偏航补偿
     * @param matrix 3x3旋转矩阵
     * @param yaw_offset 偏航角补偿值（度），用于抵消安装带来的初始偏移
     */
    void setMountingMatrix(const float matrix[3][3], float yaw_offset = 0.0f) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mountingMatrix_[i][j] = matrix[i][j];
            }
        }
        matrixToQuaternion(mountingMatrix_, mountQuaternion_);
        yaw_compensation_ = yaw_offset;  // 保存补偿值
    }

    /**
     * @brief 预设：IMU绕Z轴旋转90度安装
     * 补偿值：+90度（抵消安装带来的-90度偏移）
     */
    void setMountingZ90() {
        const float matrix[3][3] = {
            {0, 1, 0},
            {-1, 0, 0},
            {0, 0, 1}
        };
        setMountingMatrix(matrix, 90.0f);  // 显式指定补偿值
    }

    /**
     * @brief 预设：IMU绕X轴旋转180度安装
     * 补偿值：0度（X轴旋转不影响偏航角初始值）
     */
    void setMountingX180() {
        const float matrix[3][3] = {
            {1, 0, 0},
            {0, -1, 0},
            {0, 0, -1}
        };
        setMountingMatrix(matrix, 0.0f);  // X轴旋转无偏航偏移
    }

    /**
     * @brief 预设：IMU绕Y轴旋转90度安装
     * 补偿值：+90度（抵消安装带来的-90度偏移）
     */
    void setMountingY90() {
        const float matrix[3][3] = {
            {0, 0, -1},
            {0, 1, 0},
            {1, 0, 0}
        };
        setMountingMatrix(matrix, 90.0f);  // 指定补偿值
    }

    // ================== 陀螺仪偏置接口 ==================
    void setGyroBias(const Vector3f& bias) {
        gyro_bias_ = bias;
        state_.gyro_bias = bias;
    }

    const Vector3f& getGyroBias() const {
        return gyro_bias_;
    }

    // ================== 公共接口 ==================
    void calibrateGyro() {
        Serial1.println("Starting Gyro Calibration...");
        Vector3f sum;
        uint16_t valid_samples = 0;

        while (valid_samples < ImuConstants::CALIB_SAMPLES) {
            int16_t accel[3], gyro[3], temp;
            readIMUData(accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2], temp);

            Vector3f current_gyro(gyro[0], gyro[1], gyro[2]);
            
            if (current_gyro.isBelow(ImuConstants::GYRO_CAL_THRESHOLD)) {
                sum += current_gyro;
                valid_samples++;
            } else {
                handleCalibrationError(current_gyro);
                sum = Vector3f();
                valid_samples = 0;
            }
            delay(3);
        }

        gyro_bias_ = sum / ImuConstants::CALIB_SAMPLES;
        state_.gyro_bias = gyro_bias_;
        Serial1.println("Gyro Calibration Complete.");
    }

    IMUSystem(float lpf_cutoff = 50.0f, float sample_rate = 1000.0f)
        : lpf_cutoff_(lpf_cutoff), sample_rate_(sample_rate), yaw_compensation_(0.0f) {
        // 初始化安装矩阵为单位矩阵（默认无旋转）
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mountingMatrix_[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        // 初始化安装姿态四元数为单位四元数
        mountQuaternion_[0] = 1.0f;
        mountQuaternion_[1] = mountQuaternion_[2] = mountQuaternion_[3] = 0.0f;
    }

    bool begin() {
        if (!initICM42688()) return false;
        initializeFilters();
        return true;
    }

    void update() {
        const uint32_t now = micros();
        const float dt = (now - prev_timestamp_) * 1e-6f;
        
        readRawData();
        processData();
        updateOrientation(dt);
        prev_timestamp_ = now;
    }

    const ImuState& state() const { return state_; }

private:
    // ================== 安装位置相关成员 ==================
    float mountingMatrix_[3][3];  // 安装姿态转换矩阵
    float mountQuaternion_[4];    // 安装姿态对应的四元数
    float yaw_compensation_;      // 偏航角补偿值（新增：解决初始偏移）

    // ================== 原有私有成员 ==================
    static constexpr float GYRO_SCALE = 
        ImuConstants::GYRO_RANGE * DEG_TO_RAD / ImuConstants::INT16_SCALE;

    MahonyFilter mahony_filter_{0.5f, 0.003f};
    biquadFilter_t filters_[6];
    ImuState state_;
    Vector3f gyro_bias_;
    uint32_t prev_timestamp_ = 0;
    float lpf_cutoff_;
    float sample_rate_;

    // ================== 坐标转换工具函数 ==================
    /**
     * @brief 将旋转矩阵转换为四元数
     */
    void matrixToQuaternion(const float matrix[3][3], float* quat) {
        float tr = matrix[0][0] + matrix[1][1] + matrix[2][2];
        if (tr > 0) {
            float S = sqrt(tr + 1.0f) * 2.0f;
            quat[0] = 0.25f * S;
            quat[1] = (matrix[2][1] - matrix[1][2]) / S;
            quat[2] = (matrix[0][2] - matrix[2][0]) / S;
            quat[3] = (matrix[1][0] - matrix[0][1]) / S;
        } else if ((matrix[0][0] > matrix[1][1]) && (matrix[0][0] > matrix[2][2])) {
            float S = sqrt(1.0f + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0f;
            quat[0] = (matrix[2][1] - matrix[1][2]) / S;
            quat[1] = 0.25f * S;
            quat[2] = (matrix[0][1] + matrix[1][0]) / S;
            quat[3] = (matrix[0][2] + matrix[2][0]) / S;
        } else if (matrix[1][1] > matrix[2][2]) {
            float S = sqrt(1.0f + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0f;
            quat[0] = (matrix[0][2] - matrix[2][0]) / S;
            quat[1] = (matrix[0][1] + matrix[1][0]) / S;
            quat[2] = 0.25f * S;
            quat[3] = (matrix[1][2] + matrix[2][1]) / S;
        } else {
            float S = sqrt(1.0f + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0f;
            quat[0] = (matrix[1][0] - matrix[0][1]) / S;
            quat[1] = (matrix[0][2] + matrix[2][0]) / S;
            quat[2] = (matrix[1][2] + matrix[2][1]) / S;
            quat[3] = 0.25f * S;
        }
    }

    /**
     * @brief 应用安装矩阵转换向量
     */
    Vector3f applyMountingTransform(const Vector3f& in) const {
        Vector3f out;
        out.x = mountingMatrix_[0][0] * in.x + mountingMatrix_[0][1] * in.y + mountingMatrix_[0][2] * in.z;
        out.y = mountingMatrix_[1][0] * in.x + mountingMatrix_[1][1] * in.y + mountingMatrix_[1][2] * in.z;
        out.z = mountingMatrix_[2][0] * in.x + mountingMatrix_[2][1] * in.y + mountingMatrix_[2][2] * in.z;
        return out;
    }

    /**
     * @brief 四元数乘法
     */
    void quaternionMultiply(const float* q1, const float* q2, float* result) {
        result[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
        result[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2];
        result[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1];
        result[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0];
    }

    /**
     * @brief 四元数转欧拉角（带补偿）
     */
    void quaternionToEuler1(float q0, float q1, float q2, float q3,
                          float* roll, float* pitch, float* yaw) {
        // 计算原始欧拉角（弧度）
        *roll = atan2(2.0f * (q0 * q1 + q2 * q3), 1.0f - 2.0f * (q1 * q1 + q2 * q2));
        *pitch = asin(2.0f * (q0 * q2 - q3 * q1));
        *yaw = atan2(2.0f * (q0 * q3 + q1 * q2), 1.0f - 2.0f * (q2 * q2 + q3 * q3));

        // 转换为角度
        *roll *= RAD_TO_DEG;
        *pitch *= RAD_TO_DEG;
        *yaw *= RAD_TO_DEG;

        // 应用偏航角补偿（核心修改）
        *yaw += yaw_compensation_;

        // 确保角度在[-180, 180]范围内
        normalizeAngle(yaw);
    }

    /**
     * @brief 将角度归一化到[-180, 180]度
     */
    void normalizeAngle(float* angle) {
        while (*angle > 180.0f) {
            *angle -= 360.0f;
        }
        while (*angle < -180.0f) {
            *angle += 360.0f;
        }
    }

    // ================== 原有私有方法 ==================
    void initializeFilters() {
        for (auto& filter : filters_) {
            biquadFilterInitLPF(&filter, lpf_cutoff_, sample_rate_);
        }
    }

    void readRawData() {
        int16_t accel[3], gyro[3], temp;
        readIMUData(accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2], temp);
        
        state_.raw_accel = {static_cast<float>(accel[0]), 
                           static_cast<float>(accel[1]), 
                           static_cast<float>(accel[2])};
        state_.raw_gyro  = {static_cast<float>(gyro[0]), 
                           static_cast<float>(gyro[1]), 
                           static_cast<float>(gyro[2])};
        state_.temp      = temp;
    }

    void processData() {
        applyCalibration();
        convertUnits();
        applyFiltering();
        applyMountingTransform();
        processTemperature();
    }

    void applyCalibration() {
        state_.raw_gyro = state_.raw_gyro - gyro_bias_;
    }

    void convertUnits() {
        constexpr float accel_scale = ImuConstants::ACCEL_RANGE / ImuConstants::INT16_SCALE;
        state_.filtered_accel = {
            state_.raw_accel.x * accel_scale,
            state_.raw_accel.y * accel_scale,
            state_.raw_accel.z * accel_scale
        };

        state_.calibrated_gyro = {
            state_.raw_gyro.x * GYRO_SCALE,
            state_.raw_gyro.y * GYRO_SCALE,
            state_.raw_gyro.z * GYRO_SCALE
        };
    }

    void applyFiltering() {
        state_.filtered_gyro.x = biquadFilterApply(&filters_[0], state_.calibrated_gyro.x);
        state_.filtered_gyro.y = biquadFilterApply(&filters_[1], state_.calibrated_gyro.y);
        state_.filtered_gyro.z = biquadFilterApply(&filters_[2], state_.calibrated_gyro.z);
        
        state_.filtered_gyro_deg.x = state_.filtered_gyro.x * RAD_TO_DEG;
        state_.filtered_gyro_deg.y = state_.filtered_gyro.y * RAD_TO_DEG;
        state_.filtered_gyro_deg.z = state_.filtered_gyro.z * RAD_TO_DEG;


        state_.filtered_accel.x = biquadFilterApply(&filters_[3], state_.filtered_accel.x);
        state_.filtered_accel.y = biquadFilterApply(&filters_[4], state_.filtered_accel.y);
        state_.filtered_accel.z = biquadFilterApply(&filters_[5], state_.filtered_accel.z);
    }

    void applyMountingTransform() {
        state_.filtered_accel = applyMountingTransform(state_.filtered_accel);
        state_.filtered_gyro = applyMountingTransform(state_.filtered_gyro);
        state_.filtered_gyro_deg = applyMountingTransform(state_.filtered_gyro_deg);
    }

    void processTemperature() {
        state_.temp = state_.temp / ImuConstants::TEMP_SCALE + ImuConstants::TEMP_OFFSET;
    }

    void updateOrientation(float dt) {
        mahony_filter_.update(
            state_.filtered_gyro.x,
            state_.filtered_gyro.y,
            state_.filtered_gyro.z,
            state_.filtered_accel.x,
            state_.filtered_accel.y,
            state_.filtered_accel.z,
            dt
        );

        // 获取IMU原始四元数
        float imuQuat[4];
        mahony_filter_.getQuaternion(imuQuat[0], imuQuat[1], imuQuat[2], imuQuat[3]);
        
        // 融合安装姿态四元数
        float totalQuat[4];
        quaternionMultiply(mountQuaternion_, imuQuat, totalQuat);
        
        // 更新状态四元数
        state_.q0 = totalQuat[0];
        state_.q1 = totalQuat[1];
        state_.q2 = totalQuat[2];
        state_.q3 = totalQuat[3];
        
        // 转换为欧拉角（使用类内部带补偿的转换函数）
        quaternionToEuler1(
            state_.q0, 
            state_.q1, 
            state_.q2, 
            state_.q3, 
            &state_.roll, 
            &state_.pitch, 
            &state_.yaw
        );
    }

    void handleCalibrationError(const Vector3f& gyro) {
        Serial1.print("Calibration Error: ");
        Serial1.print(gyro.x);
        Serial1.print(", ");
        Serial1.print(gyro.y);
        Serial1.print(", ");
        Serial1.println(gyro.z);
    }
};

#endif // IMUSYSTEM_H
