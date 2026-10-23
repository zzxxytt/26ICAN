#include "postprocess.h"

#include <cstring>

#include "vip_lite.h"

namespace recognize_yolo {

uint32_t BytesPerElement(uint32_t data_format) {
  switch (data_format) {
    case VIP_BUFFER_FORMAT_INT8:
    case VIP_BUFFER_FORMAT_UINT8:
      return 1;
    case VIP_BUFFER_FORMAT_INT16:
    case VIP_BUFFER_FORMAT_UINT16:
    case VIP_BUFFER_FORMAT_FP16:
      return 2;
    case VIP_BUFFER_FORMAT_INT32:
    case VIP_BUFFER_FORMAT_UINT32:
    case VIP_BUFFER_FORMAT_FP32:
      return 4;
    default:
      return 1;
  }
}

bool IsSignedFormat(uint32_t data_format) {
  switch (data_format) {
    case VIP_BUFFER_FORMAT_INT8:
    case VIP_BUFFER_FORMAT_INT16:
    case VIP_BUFFER_FORMAT_INT32:
      return true;
    default:
      return false;
  }
}

std::string FormatName(uint32_t data_format) {
  switch (data_format) {
    case VIP_BUFFER_FORMAT_INT8:
      return "int8";
    case VIP_BUFFER_FORMAT_UINT8:
      return "uint8";
    case VIP_BUFFER_FORMAT_INT16:
      return "int16";
    case VIP_BUFFER_FORMAT_UINT16:
      return "uint16";
    case VIP_BUFFER_FORMAT_FP16:
      return "fp16";
    case VIP_BUFFER_FORMAT_INT32:
      return "int32";
    case VIP_BUFFER_FORMAT_UINT32:
      return "uint32";
    case VIP_BUFFER_FORMAT_FP32:
      return "float32";
    default:
      return "unknown(" + std::to_string(data_format) + ")";
  }
}

std::vector<float> Dequantize(const std::vector<uint8_t>& raw,
                              uint32_t element_count,
                              uint32_t data_format,
                              float tf_scale,
                              int32_t tf_zero_point) {
  std::vector<float> result(element_count, 0.0f);
  if (element_count == 0) {
    return result;
  }

  const uint32_t width = BytesPerElement(data_format);
  const uint32_t available = static_cast<uint32_t>(raw.size()) / width;
  const uint32_t count = element_count < available ? element_count : available;

  // tf_scale <= 0 说明查询失败，退化成「只做类型转换」，不做数值缩放。
  const bool has_scale = tf_scale > 0.0f;

  for (uint32_t index = 0; index < count; ++index) {
    const uint8_t* slot = raw.data() + static_cast<size_t>(index) * width;
    float value = 0.0f;

    switch (data_format) {
      case VIP_BUFFER_FORMAT_FP32: {
        float temp = 0.0f;
        std::memcpy(&temp, slot, sizeof(float));
        value = temp;
        break;
      }
      case VIP_BUFFER_FORMAT_INT8: {
        int8_t temp = 0;
        std::memcpy(&temp, slot, sizeof(int8_t));
        value = has_scale ? (static_cast<float>(temp) - static_cast<float>(tf_zero_point)) * tf_scale
                          : static_cast<float>(temp);
        break;
      }
      case VIP_BUFFER_FORMAT_UINT8: {
        uint8_t temp = 0;
        std::memcpy(&temp, slot, sizeof(uint8_t));
        value = has_scale ? (static_cast<float>(temp) - static_cast<float>(tf_zero_point)) * tf_scale
                          : static_cast<float>(temp);
        break;
      }
      case VIP_BUFFER_FORMAT_INT16: {
        int16_t temp = 0;
        std::memcpy(&temp, slot, sizeof(int16_t));
        value = has_scale ? (static_cast<float>(temp) - static_cast<float>(tf_zero_point)) * tf_scale
                          : static_cast<float>(temp);
        break;
      }
      case VIP_BUFFER_FORMAT_UINT16: {
        uint16_t temp = 0;
        std::memcpy(&temp, slot, sizeof(uint16_t));
        value = has_scale ? (static_cast<float>(temp) - static_cast<float>(tf_zero_point)) * tf_scale
                          : static_cast<float>(temp);
        break;
      }
      default:
        value = static_cast<float>(slot[0]);
        break;
    }

    result[index] = value;
  }

  return result;
}

}  // namespace recognize_yolo
