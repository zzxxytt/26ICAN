// yolo_npu_worker —— 在 A733 的 NPU 上跑 NBG 模型，把原始输出张量交回 Python。
//
// 为什么单独做成一个进程：
//   * VIPLite 的 init/destroy 必须严格配对，进程崩溃时内核能自动回收 NPU 资源
//   * 前向过程不占 Python GIL，不会拖慢 ROS 回调
//   * C++ 只需要负责「把张量搬出来」，解码留在 Python，改参数不用重编
//
// 用法：
//   yolo_npu_worker --model /path/to/model.nb            # 正常服务模式
//   yolo_npu_worker --model /path/to/model.nb --probe    # 只打印输入输出信息后退出
//
// 协议见 recognize_yolo/backend_npu.py 的模块注释。

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "postprocess.h"
#include "vip_runner.h"

namespace {

// 与 Python 侧约定的 magic
constexpr uint32_t kMagicReady = 0x594F4C4F;     // "YOLO"
constexpr uint32_t kMagicRequest = 0x52455131;   // "REQ1"
constexpr uint32_t kMagicResponse = 0x52455331;  // "RES1"

// 图像格式
constexpr uint32_t kFormatBgr = 0;
constexpr uint32_t kFormatRgb = 1;
constexpr uint32_t kFormatGray = 2;

// 线上固定用 float32，量化还原在 worker 内完成。
constexpr uint32_t kDtypeFloat32 = 0;

constexpr uint32_t kMaxImageBytes = 64u * 1024u * 1024u;  // 64MB 上限，防协议错位

volatile std::sig_atomic_t g_interrupted = 0;

void HandleSignal(int /*signal_number*/) { g_interrupted = 1; }

// 稳健读取：管道被关闭或读不够就返回 false，而不是死循环。
bool ReadExact(void* destination, size_t bytes) {
  auto* cursor = static_cast<uint8_t*>(destination);
  size_t remaining = bytes;
  while (remaining > 0) {
    const size_t got = std::fread(cursor, 1, remaining, stdin);
    if (got == 0) {
      return false;
    }
    cursor += got;
    remaining -= got;
  }
  return true;
}

bool WriteExact(const void* source, size_t bytes) {
  const auto* cursor = static_cast<const uint8_t*>(source);
  size_t remaining = bytes;
  while (remaining > 0) {
    const size_t put = std::fwrite(cursor, 1, remaining, stdout);
    if (put == 0) {
      return false;
    }
    cursor += put;
    remaining -= put;
  }
  return true;
}

bool WriteUint32(uint32_t value) { return WriteExact(&value, sizeof(value)); }
bool WriteInt32(int32_t value) { return WriteExact(&value, sizeof(value)); }
bool WriteFloat(float value) { return WriteExact(&value, sizeof(value)); }

// 把图像字节按目标通道顺序重排。
//   * A733 的 NPU 输入是 RGB，而 OpenCV 给的是 BGR，必须在这里翻过来，
//     否则会出现「框的位置对但类别全错」这种很难查的现象。
//   * 灰度图按格式直接透传，不复制。
std::vector<uint8_t> NormalizeChannels(const std::vector<uint8_t>& source,
                                       uint32_t format, uint32_t pixel_count) {
  if (format == kFormatGray) {
    return source;
  }

  const bool want_rgb = (format == kFormatRgb);
  const bool want_bgr = (format == kFormatBgr);
  if (!want_rgb && !want_bgr) {
    throw std::runtime_error("未知的图像格式编码: " + std::to_string(format));
  }

  std::vector<uint8_t> output(source.size());
  for (uint32_t index = 0; index < pixel_count; ++index) {
    const size_t base = static_cast<size_t>(index) * 3;
    // 输入按 BGR 解释（OpenCV 的默认顺序），按需求输出。
    const uint8_t blue = source[base + 0];
    const uint8_t green = source[base + 1];
    const uint8_t red = source[base + 2];
    if (want_rgb) {
      output[base + 0] = red;
      output[base + 1] = green;
      output[base + 2] = blue;
    } else {
      output[base + 0] = blue;
      output[base + 1] = green;
      output[base + 2] = red;
    }
  }
  return output;
}

bool WriteError(int32_t status, const std::string& message) {
  if (!WriteUint32(kMagicResponse) || !WriteInt32(status)) {
    return false;
  }
  const uint32_t length = static_cast<uint32_t>(message.size());
  if (!WriteUint32(length)) {
    return false;
  }
  return length == 0 || WriteExact(message.data(), length);
}

// 单个张量的元数据 28 字节，与 Python 的 struct.unpack("<IIIIIfi", ...) 对齐。
// 注意：不要往这个结构里加字段，否则 Python 侧的解析会错位。
bool WriteTensorHeader(uint32_t ndim, const uint32_t* dims, uint32_t dtype,
                       float scale, int32_t zero_point, uint32_t bytes) {
  if (!WriteUint32(ndim)) return false;
  for (uint32_t index = 0; index < 4; ++index) {
    if (!WriteUint32(index < ndim ? dims[index] : 0u)) return false;
  }
  if (!WriteUint32(dtype)) return false;
  if (!WriteFloat(scale)) return false;
  if (!WriteInt32(zero_point)) return false;
  if (!WriteUint32(bytes)) return false;
  return true;
}

// probe 模式：只加载模型并打印输入输出信息，用来核对模型与预处理是否匹配。
int ProbeModel(const std::string& model_path) {
  recognize_yolo::VipRunner runner;
  runner.Load(model_path);

  std::printf("模型加载成功: %s\n", model_path.c_str());
  std::printf("--- 输入张量 ---\n");
  for (const auto& input : runner.inputs()) {
    std::printf("  input[%u] %s\n", input.info.index, input.info.Describe().c_str());
  }
  std::printf("--- 输出张量 ---\n");
  for (const auto& info : runner.output_info()) {
    std::printf("  output[%u] %s (%s)\n", info.index, info.Describe().c_str(),
                recognize_yolo::FormatName(info.data_format).c_str());
  }

  std::printf(
      "\n核对要点：\n"
      "  1. 输入边长应等于 ROS 参数 input_size（例如 3 640 640 1 或 640 640 3）。\n"
      "  2. YOLOv5 带 --grid 导出时，输出常见形状为 (3, 85, 80, 80)/(3, 85, 40, 40)/\n"
      "     (3, 85, 20, 20)，通道数 = 5 + 类别数。\n"
      "  3. 若输出是 int8/uint8，确认 tf_scale 不为 0，否则量化还原不出来。\n");
  return 0;
}

void PrintUsage(const char* program) {
  std::fprintf(stderr,
               "用法: %s --model <model.nb> [--probe]\n"
               "  --model   由 ACUITY/pegasus export 生成的 NBG 模型\n"
               "  --probe   只打印输入输出张量信息后退出，用于核对模型\n",
               program);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  bool probe = false;

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--model" && index + 1 < argc) {
      model_path = argv[++index];
    } else if (argument == "--probe") {
      probe = true;
    } else if (argument == "-h" || argument == "--help") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "未知参数: %s\n", argument.c_str());
      PrintUsage(argv[0]);
      return 2;
    }
  }

  if (model_path.empty()) {
    PrintUsage(argv[0]);
    return 2;
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);
  // 忽略 SIGPIPE：Python 侧先退出时我们要能正常收尾，而不是被信号杀掉。
  std::signal(SIGPIPE, SIG_IGN);

  if (probe) {
    try {
      return ProbeModel(model_path);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "probe 失败: %s\n", error.what());
      return 1;
    }
  }

  try {
    recognize_yolo::VipRunner runner;
    runner.Load(model_path);

    // 日志走 stderr，stdout 只跑二进制协议，两者绝不能混。
    std::fprintf(stderr, "yolo_npu_worker: 模型已加载 %s\n", model_path.c_str());
    for (const auto& input : runner.inputs()) {
      std::fprintf(stderr, "yolo_npu_worker: 输入 %s\n", input.info.Describe().c_str());
    }
    for (const auto& info : runner.output_info()) {
      std::fprintf(stderr, "yolo_npu_worker: 输出 %s (%s)\n", info.Describe().c_str(),
                   recognize_yolo::FormatName(info.data_format).c_str());
    }

    if (runner.inputs().empty()) {
      std::fprintf(stderr, "yolo_npu_worker: 模型没有输入张量\n");
      return 1;
    }
    const uint32_t expected_bytes = runner.inputs()[0].info.bytes;

    // 握手：先告诉 Python 侧 NPU 已就绪。
    std::fflush(stderr);
    if (!WriteUint32(kMagicReady)) {
      std::fprintf(stderr, "yolo_npu_worker: 握手写失败\n");
      return 1;
    }
    std::fflush(stdout);

    std::vector<uint8_t> image_buffer;
    std::vector<recognize_yolo::OutputTensor> outputs;

    while (!g_interrupted) {
      uint32_t magic = 0;
      uint32_t width = 0;
      uint32_t height = 0;
      uint32_t format = 0;
      uint32_t length = 0;

      if (!ReadExact(&magic, sizeof(magic))) {
        break;  // stdin 关闭，Python 侧正常退出
      }
      if (magic != kMagicRequest) {
        WriteError(-1, "协议 magic 错误，期望 REQ1");
        break;
      }
      if (!ReadExact(&width, sizeof(width)) || !ReadExact(&height, sizeof(height)) ||
          !ReadExact(&format, sizeof(format)) || !ReadExact(&length, sizeof(length))) {
        break;
      }

      if (length == 0 || length > kMaxImageBytes) {
        WriteError(-2, "图像字节数非法: " + std::to_string(length));
        break;
      }
      if (width == 0 || height == 0) {
        WriteError(-3, "图像尺寸非法");
        break;
      }

      image_buffer.resize(length);
      if (!ReadExact(image_buffer.data(), length)) {
        break;
      }

      const uint32_t channels = (format == kFormatGray) ? 1u : 3u;
      if (static_cast<uint64_t>(width) * height * channels != length) {
        WriteError(-4, "图像尺寸与字节数不一致: " + std::to_string(width) + "x" +
                           std::to_string(height) + "x" + std::to_string(channels) +
                           " != " + std::to_string(length));
        continue;
      }

      try {
        const uint32_t pixel_count = width * height;
        const std::vector<uint8_t> prepared =
            NormalizeChannels(image_buffer, format, pixel_count);

        if (prepared.size() != expected_bytes) {
          WriteError(-5, "输入尺寸与模型不匹配: 图像 " + std::to_string(prepared.size()) +
                             " 字节，模型需要 " + std::to_string(expected_bytes) + " 字节");
          continue;
        }

        runner.Run(prepared.data(), static_cast<uint32_t>(prepared.size()), &outputs);

        // 写出应答
        bool ok = WriteUint32(kMagicResponse) && WriteInt32(0) &&
                  WriteUint32(static_cast<uint32_t>(outputs.size()));

        for (const auto& tensor : outputs) {
          if (!ok) break;

          // 统一转成 float32 再上线，量化还原在这里一次做完。
          const std::vector<float> values = recognize_yolo::Dequantize(
              tensor.data, tensor.info.element_count, tensor.info.data_format,
              tensor.info.tf_scale, tensor.info.tf_zero_point);

          uint32_t dims[4] = {0, 0, 0, 0};
          const uint32_t ndim = static_cast<uint32_t>(
              tensor.info.dims.size() < 4 ? tensor.info.dims.size() : 4);
          for (uint32_t dim = 0; dim < ndim; ++dim) {
            dims[dim] = tensor.info.dims[dim];
          }

          const uint32_t bytes = static_cast<uint32_t>(values.size() * sizeof(float));
          ok = WriteTensorHeader(ndim, dims, kDtypeFloat32, 1.0f, 0, bytes) &&
               (bytes == 0 || WriteExact(values.data(), bytes));
        }

        if (!ok) {
          std::fprintf(stderr, "yolo_npu_worker: 应答写失败，退出\n");
          return 1;
        }
        std::fflush(stdout);
      } catch (const std::exception& error) {
        // 单帧失败不退出：继续服务，让 ROS 侧自己决定是否告警。
        std::fprintf(stderr, "yolo_npu_worker: 推理失败: %s\n", error.what());
        if (!WriteError(-6, error.what())) {
          return 1;
        }
        std::fflush(stdout);
      }
    }

    std::fprintf(stderr, "yolo_npu_worker: 正常退出\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "yolo_npu_worker: 初始化失败: %s\n", error.what());
    std::fprintf(stderr,
                 "排查建议：先运行 scripts/check_npu_env.py 确认驱动、"
                 "libvip-lite.so 与模型型号是否匹配。\n");
    return 1;
  }
}
