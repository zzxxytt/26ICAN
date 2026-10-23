"""推理后端公共接口。

设计目标：让 detector_node 完全不关心底层是 NPU、ONNX 还是假数据，
只负责「采图 -> letterbox -> backend.infer -> 映射回原图 -> 发布」。

后端拿到的是 **已经 letterbox 好的输入图**（边长 = input_size），
返回的两个东西分开：

* detections —— 已经映射回 letterbox 之前那张图的坐标系
* raw_tensors —— 未解码的原始输出，用于现场核对模型输出布局
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .decode import DecodeConfig, Detection, decode_yolo_outputs


@dataclass
class InferenceResult:
    """一次推理的完整返回。"""

    detections: List[Detection] = field(default_factory=list)
    # 纯 NPU / GPU 前向耗时，毫秒
    forward_ms: float = 0.0
    # 预处理 + 前向 + 后处理，毫秒
    total_ms: float = 0.0
    # 未解码的原始输出，按输出头顺序
    raw_tensors: Sequence[np.ndarray] = field(default_factory=list)
    backend: str = "unknown"
    # 后端自述信息，用于日志与诊断，例如 NPU 固件版本
    info: str = ""


class BaseBackend(abc.ABC):
    """所有推理后端的基类。"""

    #: 后端短名，会出现在 DetectionArray.backend 里
    name = "base"

    def __init__(self, decode_config: DecodeConfig, letterbox_info=None) -> None:
        self.decode_config = decode_config
        self.letterbox_info = letterbox_info

    @abc.abstractmethod
    def _forward(self, input_image: np.ndarray) -> Sequence[np.ndarray]:
        """执行一次前向，返回原始输出张量列表。"""

    def infer(self, input_image: np.ndarray) -> InferenceResult:
        """模板方法：前向 + 解码 + 计时 + 坐标映射。"""
        started = time.perf_counter()
        forward_started = time.perf_counter()
        raw_tensors = self._forward(input_image)
        forward_ms = (time.perf_counter() - forward_started) * 1000.0

        detections = self.decode(raw_tensors)
        total_ms = (time.perf_counter() - started) * 1000.0

        return InferenceResult(
            detections=detections,
            forward_ms=round(forward_ms, 3),
            total_ms=round(total_ms, 3),
            raw_tensors=raw_tensors,
            backend=self.name,
            info=self.describe(),
        )

    def decode(self, raw_tensors: Sequence[np.ndarray]) -> List[Detection]:
        """默认解码：把 letterbox 坐标系还原回输入图坐标系。"""
        return decode_yolo_outputs(
            raw_tensors, self.decode_config, letterbox_info=self.letterbox_info
        )

    def describe(self) -> str:
        """后端自述，子类可覆盖。"""
        return self.name

    def close(self) -> None:
        """释放资源，子类可覆盖。"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class BackendUnavailable(RuntimeError):
    """后端初始化失败，通常意味着依赖缺失或硬件不支持。"""


def create_backend(
    backend_name: str,
    decode_config: DecodeConfig,
    letterbox_info=None,
    **kwargs,
) -> BaseBackend:
    """按名字创建后端。

    延迟导入各个后端模块，这样在没装 onnxruntime 的机器上也能 import 本包。
    """
    normalized = (backend_name or "").strip().lower()

    if normalized in ("npu", "vip", "viplite", "vip_lite"):
        from .backend_npu import NpuBackend

        return NpuBackend(decode_config, letterbox_info, **kwargs)

    if normalized in ("onnx", "onnxruntime", "cpu"):
        from .backend_onnx import OnnxBackend

        return OnnxBackend(decode_config, letterbox_info, **kwargs)

    if normalized in ("mock", "fake", "dummy"):
        from .backend_mock import MockBackend

        return MockBackend(decode_config, letterbox_info, **kwargs)

    raise BackendUnavailable(
        f"未知后端 '{backend_name}'，可选值：npu / onnx / mock"
    )


def summarize_tensors(tensors: Sequence[np.ndarray], max_items: int = 8) -> str:
    """把输出张量形状拼成一行，方便打日志。"""
    if not tensors:
        return "(无输出)"
    parts: List[str] = []
    for index, tensor in enumerate(tensors[:max_items]):
        array = np.asarray(tensor)
        parts.append(f"out{index}={tuple(int(d) for d in array.shape)}/{array.dtype}")
    if len(tensors) > max_items:
        parts.append(f"...共 {len(tensors)} 个")
    return " ".join(parts)


def letterbox_placeholder() -> Optional[None]:
    """类型占位，避免上层模块因为循环导入而写死 None。"""
    return None
