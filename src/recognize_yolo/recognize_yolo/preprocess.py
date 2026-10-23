"""图像预处理：letterbox、颜色通道与归一化。

两条推理链路的预处理分工不同，务必区分清楚：

* NPU (VIPLite) 链路 —— ACUITY 转换时已经把 mean / scale 写进了模型的
  inputmeta，NPU 自己完成归一化。因此 Python 只需要送 **uint8 的
  letterbox 后的 BGR 或 RGB 图**，不要再除 255。
* ONNX Runtime 链路 —— 归一化必须由 Python 完成，输出 float32 NCHW。

两条链路都使用同一个 letterbox，保证检测框映射回原图时用同一套
ratio / pad 参数。
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .normalize import LetterboxInfo, to_nchw_float, to_uint8_hwc

# 归一化与几何参数放在 normalize.py（不依赖 OpenCV），这里再导出一次，
# 调用方既可以 from preprocess import to_nchw_float，也可以从 normalize 导入。
__all__ = [
    "LetterboxInfo",
    "letterbox",
    "bgr_to_rgb",
    "to_nchw_float",
    "to_uint8_hwc",
]


def letterbox(
    image: np.ndarray,
    input_size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
    scale_up: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, LetterboxInfo]:
    """保持宽高比缩放并居中填充到 input_size × input_size 的正方形。

    参数
    ----
    image:
        BGR 或 RGB 的 uint8 图，通道顺序不影响本函数（只做几何变换）。
    input_size:
        目标边长，例如 640。
    color:
        填充像素值，YOLOv5/v8 官方默认 114 灰。
    scale_up:
        False 时只缩小不放大，避免小图被插值糊掉。
    stride:
        长宽都对齐到该值。NPU 对非 32 倍数尺寸容易报错，所以默认 32。

    返回
    ----
    (letterboxed_image, info)

    实现要点：**宽和高必须用同一个缩放比例**。曾经的做法是先算出
    new_width / new_height 再各自对齐到 stride，对齐之后两个方向的实际
    比例就不相等了；此时若 info.ratio 只记录其中一个方向，把检测框映射回
    原图时水平方向会整体偏移（宽高比越极端偏得越多）。这里改为先按统一
    比例算出对齐后的边长，再把实际比例一并写进 info。
    """
    if image is None or image.size == 0:
        raise ValueError("letterbox 收到空图像")

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"非法图像尺寸 {width}x{height}")

    ratio = min(input_size / width, input_size / height)
    if not scale_up:
        ratio = min(ratio, 1.0)

    if stride > 1:
        # 用统一比例算出目标边长，再对齐到 stride，保证两个方向同比例。
        aligned_width = int(round(width * ratio / stride)) * stride
        aligned_height = int(round(height * ratio / stride)) * stride
        # 对齐可能把边长推到 0 或超过 input_size，两端都夹一下。
        new_width = max(stride, min(aligned_width, input_size))
        new_height = max(stride, min(aligned_height, input_size))
    else:
        new_width = max(1, min(int(round(width * ratio)), input_size))
        new_height = max(1, min(int(round(height * ratio)), input_size))

    # 对齐带来的微小取整误差靠这一点点插值吸收，几何上仍然一致。
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_width = input_size - new_width
    pad_height = input_size - new_height
    # 左/上取 floor，右/下取剩余，保证总填充量精确，不丢列。
    pad_left = pad_width // 2
    pad_top = pad_height // 2
    pad_right = pad_width - pad_left
    pad_bottom = pad_height - pad_top

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=color,
    )

    info = LetterboxInfo(
        ratio=float(ratio),
        pad_x=float(pad_left),
        pad_y=float(pad_top),
        input_size=int(input_size),
        original_width=int(width),
        original_height=int(height),
        unpad_top=int(pad_top),
        unpad_left=int(pad_left),
    )
    return padded, info


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR -> RGB。OpenCV 读进来是 BGR，YOLO 训练时用 RGB。"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
