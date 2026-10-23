// 输出后处理的公共部分。
//
// 解码本身（YOLO 的框回归 + NMS）放在 Python 侧做，理由是：
//   * NMS 是纯 CPU 活儿，numpy 已经足够快，改阈值不用重新编译
//   * 类别数、anchor 布局这些现场要调的参数不用动 C++
//   * C++ 侧只负责它唯一不可替代的事：把 NPU 输出搬出来
//
// 这个文件只保留「量化输出 -> float」这类通用工具，方便 C++ 侧自测。

#ifndef RECOGNIZE_YOLO_POSTPROCESS_H
#define RECOGNIZE_YOLO_POSTPROCESS_H

#include <cstdint>
#include <string>
#include <vector>

namespace recognize_yolo {

//: 把量化输出按 scale / zero_point 还原成 float。
//:
//: 为什么要显式写这一步：A733 的 NPU 输出缓冲经常是 int8（VIP_BUFFER_FORMAT_INT8），
//: 如果直接按 float 解释，会得到一堆 NaN 或天文数字，而不是「没检测到」。
//: 这也是新手最常见的「模型跑通了但没有结果」的原因。
std::vector<float> Dequantize(const std::vector<uint8_t>& raw,
                              uint32_t element_count,
                              uint32_t data_format,
                              float tf_scale,
                              int32_t tf_zero_point);

//: 数据格式的字节宽度。
uint32_t BytesPerElement(uint32_t data_format);

//: 数据格式是否带符号。
bool IsSignedFormat(uint32_t data_format);

//: 把数据格式转成可读名字，probe 模式打印用。
std::string FormatName(uint32_t data_format);

}  // namespace recognize_yolo

#endif  // RECOGNIZE_YOLO_POSTPROCESS_H
