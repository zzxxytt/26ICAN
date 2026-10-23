// VIPLite 网络封装：把 17 步生命周期收敛成 RAII。
//
// 这里的 API 调用顺序来自全志 Tina SDK 的《VIPLite API 说明》：
//   vip_init -> vip_create_network -> vip_query_input/output
//   -> vip_create_buffer -> vip_prepare_network -> vip_map_buffer
//   -> vip_set_input/output -> vip_run_network -> vip_flush_buffer
//   -> vip_finish_network -> vip_destroy_network/buffer -> vip_destroy
//
// 目标平台：Allwinner A733 / Vivante VIP9000 (VIP9000NANODI_PLUS_PID0X1000003B)

#ifndef RECOGNIZE_YOLO_VIP_RUNNER_H
#define RECOGNIZE_YOLO_VIP_RUNNER_H

#include <cstdint>
#include <string>
#include <vector>

#include "vip_lite.h"

namespace recognize_yolo {

//: 单个输入/输出张量的描述，全部来自 vip_query_*，不靠猜。
struct TensorInfo {
  uint32_t index = 0;
  std::string name;
  std::vector<uint32_t> dims;   // 原始维度顺序，例如 {80, 80, 85, 3}
  uint32_t element_count = 0;   // 元素总数
  uint32_t bytes = 0;           // 字节数
  uint32_t data_format = 0;     // VIP_BUFFER_FORMAT_*
  uint32_t quant_format = 0;    // VIP_BUFFER_QUANTIZE_*
  int32_t fixed_point_pos = 0;  // 定点位置，int8/int16 量化时有效
  float tf_scale = 1.0f;        // TF 风格非对称量化比例
  int32_t tf_zero_point = 0;    // TF 风格非对称量化零点

  //: 打印成一行，probe 模式和启动日志都用它。
  std::string Describe() const;
};

//: 输出张量的实际数据，交给上层序列化。
struct OutputTensor {
  TensorInfo info;
  std::vector<uint8_t> data;
};

//: 输入张量的描述与缓冲。
struct InputTensor {
  TensorInfo info;
  vip_buffer buffer = VIP_NULL;
  void* mapped = nullptr;
};

//: VIPLite 网络封装。
//
// 构造即初始化 NPU 并加载 NBG；任何一步失败都抛 std::runtime_error，
// 析构时按相反顺序释放。驱动要求 init/destroy 严格配对，RAII 能保证这一点。
class VipRunner {
 public:
  VipRunner() = default;
  ~VipRunner();

  VipRunner(const VipRunner&) = delete;
  VipRunner& operator=(const VipRunner&) = delete;

  //: 初始化 NPU 并加载 .nb 模型。
  void Load(const std::string& model_path);

  //: 跑一次推理。input 必须和 input_info().bytes 一致。
  //:
  //: 注意：这里每次都重新 attach 输入缓冲并 flush 缓存。A733 的 NPU 与 CPU
  //: 不共享缓存，漏掉 vip_flush_buffer 会读到上一帧的旧数据（表现为检测框
  //: 抖动或恒定不变），所以宁可多刷一次。
  void Run(const uint8_t* input, uint32_t input_bytes,
           std::vector<OutputTensor>* outputs);

  //: 网络输入张量信息，决定预处理输出多大。
  const std::vector<InputTensor>& inputs() const { return inputs_; }

  //: 网络输出张量信息；probe 模式打印它来核对布局。
  const std::vector<TensorInfo>& output_info() const { return output_info_; }

  //: 是否已成功加载。
  bool loaded() const { return network_ != VIP_NULL; }

 private:
  void QueryTensors();
  void CreateBuffers();
  void ReleaseAll();

  vip_network network_ = VIP_NULL;
  bool initialized_ = false;
  std::vector<InputTensor> inputs_;
  std::vector<TensorInfo> output_info_;
  std::vector<vip_buffer> output_buffers_;
  std::vector<void*> output_mapped_;
};

//: 把 vip_status_e 转成可读文本。
std::string VipStatusToString(vip_status_e status);

//: 检查返回值，非 VIP_SUCCESS 就抛异常。
void CheckVip(vip_status_e status, const std::string& what);

}  // namespace recognize_yolo

#endif  // RECOGNIZE_YOLO_VIP_RUNNER_H
