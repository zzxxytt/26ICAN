"""假后端：在没有 NPU / 模型文件时验证整条 ROS 链路。

用途很明确 —— 电脑上先跑通话题、日志、可视化、JSON 输出，
再去调真正的模型。它不认识图像内容，只按固定规则画几个框。
"""

from __future__ import annotations

import math
import time
from typing import List, Sequence

import numpy as np

from .backend import BaseBackend, InferenceResult
from .decode import DecodeConfig, Detection


class MockBackend(BaseBackend):
    """产生周期性运动的假检测框。"""

    name = "mock"

    def __init__(
        self,
        decode_config: DecodeConfig,
        letterbox_info=None,
        num_boxes: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(decode_config, letterbox_info)
        self.num_boxes = max(1, int(num_boxes))
        self._frame_index = 0

    def _forward(self, input_image: np.ndarray) -> Sequence[np.ndarray]:
        # 模拟一点前向耗时，让性能统计看起来接近真实。
        time.sleep(0.005)
        self._frame_index += 1

        height, width = input_image.shape[:2]
        phase = self._frame_index * 0.05
        center_x = width * (0.5 + 0.25 * math.sin(phase))
        center_y = height * (0.5 + 0.15 * math.cos(phase * 1.3))
        box_w = width * 0.18
        box_h = height * 0.35

        # 直接构造标准行向量（4 + 1 + num_classes），走一遍真实解码路径，
        # 这样连 NMS 和阈值逻辑都能被测到。
        classes = self.decode_config.num_classes
        rows = np.zeros((self.num_boxes, 5 + classes), dtype=np.float32)
        for index in range(self.num_boxes):
            offset = index * width * 0.12
            rows[index, 0] = center_x + offset
            rows[index, 1] = center_y
            rows[index, 2] = box_w
            rows[index, 3] = box_h
            rows[index, 4] = 0.9 - index * 0.15
            class_id = index % max(1, min(classes, 3))
            rows[index, 5 + class_id] = 0.85 - index * 0.2

        return [rows]

    def describe(self) -> str:
        return f"mock(num_boxes={self.num_boxes}, 仅供链路验证)"
