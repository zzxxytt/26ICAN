"""归一化、几何参数与内存布局转换，只依赖 numpy。

单独拆出来是为了让解码链路（decode.py）不必依赖 OpenCV：
板子上没装 cv2 时，解码与 NMS 的逻辑依然可以被导入和测试。
LetterboxInfo 定义在这里，因为它是解码阶段把框映射回原图所必需的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LetterboxInfo:
    """letterbox 的几何信息，用于把检测框还原回原图坐标。

    ratio 是**统一**的缩放比例，宽高共用；pad_x / pad_y 是左侧、上侧的填充量。
    还原公式：原图坐标 = (输入坐标 - pad) / ratio。
    """

    ratio: float
    pad_x: float
    pad_y: float
    input_size: int
    original_width: int
    original_height: int
    # 去掉上下黑边需要裁剪的行列范围，等价于 pad 的反向操作
    unpad_top: int = 0
    unpad_left: int = 0


def to_nchw_float(
    image: np.ndarray,
    scale: float = 1.0 / 255.0,
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    rgb: bool = True,
) -> np.ndarray:
    """HWC uint8 -> NCHW float32，供 ONNX Runtime 使用。

    等价于 (pixel / 255 - mean) / std。默认 mean=0、std=1 就是 YOLOv5/v8 的
    标准预处理 (x/255)。

    通道顺序：入参按 OpenCV 约定是 BGR，rgb=True 时翻成 RGB —— YOLO 训练时
    用的是 RGB，这一步漏掉会导致「框的位置对但类别全错」。
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 HWC 三通道图像，收到 {image.shape}")

    array = image.astype(np.float32) * float(scale)
    if rgb:
        array = array[:, :, ::-1]

    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    array = (array - mean_array) / std_array

    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def to_uint8_hwc(image: np.ndarray, rgb: bool = False) -> np.ndarray:
    """整理成 NPU 需要的 uint8 HWC 连续内存，可按需转 RGB。

    注意：NPU 链路不需要除 255 —— ACUITY 转换时已把 mean/scale 写进模型的
    inputmeta，NPU 自己完成归一化。这里再做一次会把输入压成接近全 0。
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 HWC 三通道图像，收到 {image.shape}")

    if rgb:
        return np.ascontiguousarray(image[:, :, ::-1].astype(np.uint8))
    return np.ascontiguousarray(image.astype(np.uint8))
