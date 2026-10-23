"""ONNX Runtime 后端：CPU 兜底与「精度对照基准」。

在香橙派 4 Pro 上它的作用有两个：

1. NPU 链路还没调通时，先用它把整条 ROS 流程跑起来（YOLOv8n @ 320 大约
   能做到每秒几帧，做静态验证足够）。
2. 作为精度基准：同一张图分别过 NPU 和 ONNX，比较检测框差异，用来判断
   是量化掉精度还是后处理写错了。
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

from .backend import BackendUnavailable, BaseBackend
from .decode import DecodeConfig
from .preprocess import to_nchw_float

LOGGER = logging.getLogger(__name__)


class OnnxBackend(BaseBackend):
    """用 onnxruntime 跑 YOLO ONNX 模型。"""

    name = "onnx"

    def __init__(
        self,
        decode_config: DecodeConfig,
        letterbox_info=None,
        model_path: str = "",
        input_name: str = "",
        providers: str = "CPUExecutionProvider",
        intra_op_threads: int = 0,
        scale: float = 1.0 / 255.0,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs,
    ) -> None:
        super().__init__(decode_config, letterbox_info)

        if not model_path:
            raise BackendUnavailable("ONNX 后端需要 model_path 参数")

        try:
            import onnxruntime as ort
        except ImportError as error:
            raise BackendUnavailable(
                "未安装 onnxruntime。板上安装：sudo apt install python3-onnxruntime，"
                "或 pip3 install onnxruntime"
            ) from error

        self._ort = ort
        self.model_path = str(model_path)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            options.intra_op_num_threads = int(intra_op_threads)
        # 日志等级调低，否则每帧都会刷 provider 信息。
        options.log_severity_level = 3

        available = ort.get_available_providers()
        requested = [item.strip() for item in str(providers).split(",") if item.strip()]
        usable = [item for item in requested if item in available]
        if not usable:
            usable = ["CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(
                self.model_path, sess_options=options, providers=usable
            )
        except Exception as error:
            raise BackendUnavailable(
                f"加载 ONNX 模型失败: {self.model_path} ({error})"
            ) from error

        self.providers = usable
        self.scale = float(scale)
        self.mean = tuple(float(item) for item in mean)
        self.std = tuple(float(item) for item in std)

        model_inputs = self.session.get_inputs()
        if not model_inputs:
            raise BackendUnavailable("ONNX 模型没有输入节点")
        self.input_name = input_name or model_inputs[0].name
        self._input_shape = model_inputs[0].shape

        LOGGER.info(
            "ONNX 模型已加载: %s, provider=%s, input=%s %s",
            self.model_path, usable, self.input_name, self._input_shape,
        )

    def _forward(self, input_image: np.ndarray) -> Sequence[np.ndarray]:
        blob = to_nchw_float(
            input_image, scale=self.scale, mean=self.mean, std=self.std, rgb=True
        )

        expected = self._input_shape
        if len(expected) == 4 and isinstance(expected[2], int) and isinstance(expected[3], int):
            if blob.shape[2] != expected[2] or blob.shape[3] != expected[3]:
                raise RuntimeError(
                    f"输入尺寸不匹配：模型要求 {expected[2]}x{expected[3]}，"
                    f"实际 {blob.shape[2]}x{blob.shape[3]}。请把 input_size 参数改成 "
                    f"{expected[2]}"
                )

        outputs = self.session.run(None, {self.input_name: blob})
        return [np.asarray(item) for item in outputs]

    def describe(self) -> str:
        return f"onnx(model={self.model_path}, providers={','.join(self.providers)})"

    def close(self) -> None:
        self.session = None
