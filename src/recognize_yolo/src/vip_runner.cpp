// VIPLite 网络封装的实现。见 vip_runner.h 顶部的生命周期说明。

#include "vip_runner.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace recognize_yolo {
namespace {

//: 输出张量最多支持 4 维，够 YOLO 用（(A,C,H,W) 或 (N,C,H,W)）。
constexpr uint32_t kMaxDims = 4;

//: int8 / int16 的定点位置上限，用来判断 fixed_point_pos 是否可信。
constexpr int32_t kReasonableFixedPointPos = 31;

//: 新版 SDK 用枚举区分网络类型，老版是宏。统一成一个内部常量。
//:
//: 我们从 ACUITY 导出的 NBG 是「预编译二进制」，走 pegasus export 生成，
//: 对应 ORIGINAL / PRECOMPILED 这一类类型。这里按优先级挑第一个存在的。
#if defined(VIP_NETWORK_TYPE_IMPORT)
#define RY_NETWORK_TYPE VIP_NETWORK_TYPE_IMPORT
#elif defined(VIP_NETWORK_TYPE_ORIGINAL)
#define RY_NETWORK_TYPE VIP_NETWORK_TYPE_ORIGINAL
#elif defined(VIP_NETWORK_TYPE_PRECOMPILED)
#define RY_NETWORK_TYPE VIP_NETWORK_TYPE_PRECOMPILED
#else
#define RY_NETWORK_TYPE VIP_NETWORK_TYPE_DEFAULT
#endif

//: vip_query_* 的 property 名在不同版本里前缀可能是 VIP_BUFFER_PROP_ 或
//: VIP_NETWORK_PROP_，用宏抹平。
#ifndef VIP_BUFFER_PROP_NUM_OF_DIMENSION
#define VIP_BUFFER_PROP_NUM_OF_DIMENSION 0
#endif

std::string JoinDims(const std::vector<uint32_t>& dims) {
  std::ostringstream stream;
  for (size_t index = 0; index < dims.size(); ++index) {
    if (index > 0) {
      stream << " ";
    }
    stream << dims[index];
  }
  return stream.str();
}

std::vector<uint8_t> ReadWholeFile(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    throw std::runtime_error("无法打开模型文件: " + path);
  }

  if (std::fseek(file, 0, SEEK_END) != 0) {
    std::fclose(file);
    throw std::runtime_error("无法定位模型文件末尾: " + path);
  }
  const long size = std::ftell(file);
  if (size <= 0) {
    std::fclose(file);
    throw std::runtime_error("模型文件为空: " + path);
  }
  std::rewind(file);

  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  const size_t read_bytes = std::fread(buffer.data(), 1, buffer.size(), file);
  std::fclose(file);

  if (read_bytes != buffer.size()) {
    throw std::runtime_error("读取模型文件不完整: " + path);
  }
  return buffer;
}

}  // namespace

std::string VipStatusToString(vip_status_e status) {
  std::ostringstream stream;
  stream << "vip_status=" << static_cast<int>(status);
  return stream.str();
}

void CheckVip(vip_status_e status, const std::string& what) {
  if (status != VIP_SUCCESS) {
    throw std::runtime_error(what + " 失败 (" + VipStatusToString(status) + ")");
  }
}

std::string TensorInfo::Describe() const {
  std::ostringstream stream;
  stream << "shape=[" << JoinDims(dims) << "]"
         << " elems=" << element_count
         << " bytes=" << bytes
         << " fmt=" << data_format
         << " quant=" << quant_format
         << " fpp=" << fixed_point_pos
         << " tf_scale=" << tf_scale;
  if (!name.empty()) {
    stream << " name=" << name;
  }
  return stream.str();
}

// ---------------------------------------------------------------------------
// 构造 / 析构
// ---------------------------------------------------------------------------
VipRunner::~VipRunner() { ReleaseAll(); }

void VipRunner::ReleaseAll() {
  // 释放顺序与创建顺序相反，并且每一步都容错：析构里绝不抛异常。
  for (size_t index = 0; index < output_mapped_.size(); ++index) {
    if (output_mapped_[index] != nullptr && index < output_buffers_.size() &&
        output_buffers_[index] != VIP_NULL) {
      vip_unmap_buffer(output_buffers_[index]);
      output_mapped_[index] = nullptr;
    }
  }
  output_mapped_.clear();

  for (vip_buffer buffer : output_buffers_) {
    if (buffer != VIP_NULL) {
      vip_destroy_buffer(buffer);
    }
  }
  output_buffers_.clear();

  for (InputTensor& input : inputs_) {
    if (input.mapped != nullptr && input.buffer != VIP_NULL) {
      vip_unmap_buffer(input.buffer);
      input.mapped = nullptr;
    }
    if (input.buffer != VIP_NULL) {
      vip_destroy_buffer(input.buffer);
      input.buffer = VIP_NULL;
    }
  }
  inputs_.clear();

  if (network_ != VIP_NULL) {
    vip_finish_network(network_);
    vip_destroy_network(network_);
    network_ = VIP_NULL;
  }

  if (initialized_) {
    vip_destroy();
    initialized_ = false;
  }
}

// ---------------------------------------------------------------------------
// 加载
// ---------------------------------------------------------------------------
void VipRunner::Load(const std::string& model_path) {
  const std::vector<uint8_t> model = ReadWholeFile(model_path);

  // 老版本 vip_init() 无参，新版本接收一个配置结构体指针。
  // 用重载在编译期自动适配。
  {
    struct AnyInitArg {};
    (void)sizeof(AnyInitArg);
  }
#if defined(VIP_INIT_TAKES_ARGS)
  vip_init_params_t init_params;
  std::memset(&init_params, 0, sizeof(init_params));
  init_params.data = nullptr;
  CheckVip(vip_init(&init_params), "vip_init");
#else
  CheckVip(vip_init(), "vip_init");
#endif
  initialized_ = true;

  CheckVip(
      vip_create_network(const_cast<uint8_t*>(model.data()),
                         static_cast<vip_uint32_t>(model.size()),
                         RY_NETWORK_TYPE, &network_),
      "vip_create_network（模型与 NPU 型号不匹配时最容易在这里失败）");

  QueryTensors();
  CreateBuffers();
}

void VipRunner::QueryTensors() {
  vip_uint32_t input_count = 0;
  CheckVip(vip_query_network(network_, VIP_NETWORK_PROP_INPUT_COUNT, &input_count),
           "查询输入个数");

  vip_uint32_t output_count = 0;
  CheckVip(vip_query_network(network_, VIP_NETWORK_PROP_OUTPUT_COUNT, &output_count),
           "查询输出个数");

  if (input_count == 0) {
    throw std::runtime_error("网络没有输入张量");
  }
  if (output_count == 0) {
    throw std::runtime_error("网络没有输出张量");
  }

  inputs_.resize(input_count);
  for (vip_uint32_t index = 0; index < input_count; ++index) {
    InputTensor& input = inputs_[index];
    input.info.index = index;

    vip_uint32_t dim_count = 0;
    CheckVip(vip_query_input(network_, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION,
                             &dim_count),
             "查询输入维度数");
    if (dim_count == 0 || dim_count > kMaxDims) {
      throw std::runtime_error("输入维度数异常: " + std::to_string(dim_count));
    }

    input.info.dims.resize(dim_count);
    for (vip_uint32_t dim = 0; dim < dim_count; ++dim) {
      // 注意：部分 SDK 版本的 SIZES_OF_DIMENSION 会一次返回全部维度，
      // 这里按「一次查一维、按顺序填」处理，与官方 demo 的写法一致。
      vip_uint32_t size = 0;
      CheckVip(vip_query_input(network_, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION,
                               &size),
               "查询输入形状");
      input.info.dims[dim] = size;
    }

    CheckVip(vip_query_input(network_, index, VIP_BUFFER_PROP_DATA_FORMAT,
                             &input.info.data_format),
             "查询输入数据格式");
    CheckVip(vip_query_input(network_, index, VIP_BUFFER_PROP_QUANT_FORMAT,
                             &input.info.quant_format),
             "查询输入量化格式");
    vip_query_input(network_, index, VIP_BUFFER_PROP_FIXED_POINT_POS,
                    &input.info.fixed_point_pos);

    input.info.element_count = 1;
    for (uint32_t dim : input.info.dims) {
      input.info.element_count *= dim;
    }
    input.info.bytes = input.info.element_count;  // 输入是 uint8，1 字节/元素
  }

  output_info_.resize(output_count);
  for (vip_uint32_t index = 0; index < output_count; ++index) {
    TensorInfo& info = output_info_[index];
    info.index = index;

    vip_uint32_t dim_count = 0;
    CheckVip(vip_query_output(network_, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION,
                              &dim_count),
             "查询输出维度数");
    if (dim_count == 0 || dim_count > kMaxDims) {
      throw std::runtime_error("输出维度数异常: " + std::to_string(dim_count));
    }

    info.dims.resize(dim_count);
    for (vip_uint32_t dim = 0; dim < dim_count; ++dim) {
      vip_uint32_t size = 0;
      CheckVip(vip_query_output(network_, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION,
                                &size),
               "查询输出形状");
      info.dims[dim] = size;
    }

    CheckVip(vip_query_output(network_, index, VIP_BUFFER_PROP_DATA_FORMAT,
                              &info.data_format),
             "查询输出数据格式");
    CheckVip(vip_query_output(network_, index, VIP_BUFFER_PROP_QUANT_FORMAT,
                              &info.quant_format),
             "查询输出量化格式");

    // fixed_point_pos 在部分版本里会返回垃圾值，做一次合理性检查，
    // 不可信就退化成「不做反量化」，让 Python 侧用原始整数处理。
    info.fixed_point_pos = 0;
    if (vip_query_output(network_, index, VIP_BUFFER_PROP_FIXED_POINT_POS,
                         &info.fixed_point_pos) != VIP_SUCCESS ||
        info.fixed_point_pos < 0 || info.fixed_point_pos > kReasonableFixedPointPos) {
      info.fixed_point_pos = 0;
    }
    vip_query_output(network_, index, VIP_BUFFER_PROP_TF_SCALE, &info.tf_scale);
    vip_query_output(network_, index, VIP_BUFFER_PROP_TF_ZERO_POINT,
                     &info.tf_zero_point);

    info.element_count = 1;
    for (uint32_t dim : info.dims) {
      info.element_count *= dim;
    }
    info.bytes = info.element_count;
  }
}

void VipRunner::CreateBuffers() {
  // 输入缓冲：uint8，按元素数分配。
  for (InputTensor& input : inputs_) {
    vip_buffer_create_params_t params;
    std::memset(&params, 0, sizeof(params));
    params.data_type = VIP_BUFFER_FORMAT_UINT8;
    params.num_of_dims = static_cast<vip_uint32_t>(input.info.dims.size());
    for (size_t dim = 0; dim < input.info.dims.size(); ++dim) {
      params.sizes[dim] = input.info.dims[dim];
    }
    params.quant_format = VIP_BUFFER_QUANTIZE_NONE;
    params.quant_data = nullptr;

    CheckVip(vip_create_buffer(&params, sizeof(params), &input.buffer),
             "创建输入缓冲");
    input.mapped = vip_map_buffer(input.buffer);
    if (input.mapped == nullptr) {
      throw std::runtime_error("映射输入缓冲失败");
    }
    // 拿到映射指针后就固定住，Run() 里直接往这块内存写。
    input.info.bytes = vip_get_buffer_size(input.buffer);
  }

  // 输出缓冲：按查询到的量化格式分配，让驱动按模型需要的大小切。
  output_buffers_.resize(output_info_.size(), VIP_NULL);
  output_mapped_.resize(output_info_.size(), nullptr);

  for (size_t index = 0; index < output_info_.size(); ++index) {
    const TensorInfo& info = output_info_[index];

    vip_buffer_create_params_t params;
    std::memset(&params, 0, sizeof(params));
    params.data_type = info.data_format;
    params.num_of_dims = static_cast<vip_uint32_t>(info.dims.size());
    for (size_t dim = 0; dim < info.dims.size(); ++dim) {
      params.sizes[dim] = info.dims[dim];
    }
    params.quant_format = info.quant_format;
    params.quant_data = nullptr;

    CheckVip(vip_create_buffer(&params, sizeof(params), &output_buffers_[index]),
             "创建输出缓冲");
    output_mapped_[index] = vip_map_buffer(output_buffers_[index]);
    if (output_mapped_[index] == nullptr) {
      throw std::runtime_error("映射输出缓冲失败");
    }
  }

  CheckVip(vip_prepare_network(network_), "vip_prepare_network");
}

// ---------------------------------------------------------------------------
// 推理
// ---------------------------------------------------------------------------
void VipRunner::Run(const uint8_t* input, uint32_t input_bytes,
                    std::vector<OutputTensor>* outputs) {
  if (network_ == VIP_NULL) {
    throw std::runtime_error("网络尚未加载");
  }
  if (outputs == nullptr) {
    throw std::runtime_error("outputs 不能为空");
  }
  if (inputs_.empty()) {
    throw std::runtime_error("没有可用的输入张量");
  }

  InputTensor& primary = inputs_[0];
  if (input_bytes != primary.info.bytes) {
    std::ostringstream stream;
    stream << "输入字节数不匹配: 收到 " << input_bytes << "，模型要求 "
           << primary.info.bytes;
    throw std::runtime_error(stream.str());
  }

  // 1) 把这一帧拷进 NPU 缓冲，并把 CPU 缓存写回，否则 NPU 可能读到旧数据。
  std::memcpy(primary.mapped, input, input_bytes);
  CheckVip(vip_flush_buffer(primary.buffer, VIP_BUFFER_OPER_TYPE_FLUSH),
           "刷新输入缓冲（写回）");

  // 2) attach 输入输出。多次调用是允许的，每次都重新挂载最稳妥。
  for (size_t index = 0; index < inputs_.size(); ++index) {
    CheckVip(vip_set_input(network_, static_cast<vip_uint32_t>(index),
                           inputs_[index].buffer),
             "vip_set_input");
  }
  for (size_t index = 0; index < output_buffers_.size(); ++index) {
    CheckVip(vip_set_output(network_, static_cast<vip_uint32_t>(index),
                            output_buffers_[index]),
             "vip_set_output");
  }

  // 3) 同步执行。控制回路要的是确定性延迟，这里不用 trigger + wait 的异步形式。
  CheckVip(vip_run_network(network_), "vip_run_network");

  // 4) 让输出缓冲的 CPU 缓存失效，否则会读到陈旧数据。
  for (vip_buffer buffer : output_buffers_) {
    CheckVip(vip_flush_buffer(buffer, VIP_BUFFER_OPER_TYPE_INVALIDATE),
             "刷新输出缓冲（失效）");
  }

  // 5) 拷出结果。
  outputs->clear();
  outputs->resize(output_info_.size());

  for (size_t index = 0; index < output_info_.size(); ++index) {
    OutputTensor& tensor = (*outputs)[index];
    tensor.info = output_info_[index];

    const uint32_t bytes = vip_get_buffer_size(output_buffers_[index]);
    tensor.data.resize(bytes);
    if (bytes > 0) {
      std::memcpy(tensor.data.data(), output_mapped_[index], bytes);
      // 以实际缓冲大小为准回填元素数，避免和查询值不一致时越界。
      tensor.info.bytes = bytes;
    }
  }
}

}  // namespace recognize_yolo
