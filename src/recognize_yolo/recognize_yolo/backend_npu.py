"""NPU 后端：通过子进程调用 C++ 的 VIPLite worker。

为什么用「子进程 + 管道」而不是 ctypes 直接调 libvip-lite.so：

1. VIPLite 驱动对初始化 / 释放的配对很敏感，一旦 Python 侧异常退出就可能
   留下没释放的网络资源，需要重启系统才能恢复。放到子进程里，进程死掉
   内核会自动回收。
2. 前向过程完全不碰 GIL，不会拖慢 ROS 的其它回调。
3. 协议只有几十行，比维护一份 ctypes 结构体映射更不容易出错。

协议（全部小端）
----------------
worker 启动就绪后先发 magic 0x594F4C4F (``YOLO``)，表示 NPU 初始化完成。

请求：
    magic   uint32   0x52455131 (``REQ1``)
    width   uint32
    height  uint32
    fmt     uint32   0=BGR, 1=RGB, 2=GRAY
    length  uint32   图像字节数
    data    uint8[length]

应答：
    magic   uint32   0x52455331 (``RES1``)
    status  int32    0 成功
    count   uint32   输出张量个数
    每个张量：
        ndim    uint32
        dims    uint32[ndim]
        dtype   uint32   0=float32, 1=int8, 2=uint8, 3=int16
        scale   float32
        zero    int32
        nbytes  uint32
        data    uint8[nbytes]

错误应答：status 非 0 时，紧跟一个 uint32 长度和 UTF-8 错误信息。
"""

from __future__ import annotations

import logging
import struct
import subprocess
import threading
from typing import List, Optional, Sequence

import numpy as np

from .backend import BackendUnavailable, BaseBackend
from .decode import DecodeConfig

LOGGER = logging.getLogger(__name__)

MAGIC_READY = 0x594F4C4F
MAGIC_REQUEST = 0x52455131
MAGIC_RESPONSE = 0x52455331

FORMAT_CODES = {"bgr": 0, "rgb": 1, "gray": 2}
DTYPE_CODES = {0: np.float32, 1: np.int8, 2: np.uint8, 3: np.int16}


class NpuBackend(BaseBackend):
    """把 letterbox 后的 uint8 图像交给 VIPLite worker，取回原始张量。"""

    name = "npu"

    def __init__(
        self,
        decode_config: DecodeConfig,
        letterbox_info=None,
        worker_path: str = "",
        model_path: str = "",
        input_format: str = "rgb",
        startup_timeout: float = 30.0,
        request_timeout: float = 10.0,
        log_stderr: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(decode_config, letterbox_info)

        if not worker_path:
            raise BackendUnavailable("NPU 后端需要 worker_path 参数指向 yolo_npu_worker")
        if not model_path:
            raise BackendUnavailable("NPU 后端需要 model_path 参数指向 .nb 模型")

        self.worker_path = str(worker_path)
        self.model_path = str(model_path)
        self.input_format = str(input_format).strip().lower()
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)

        if self.input_format not in FORMAT_CODES:
            raise BackendUnavailable(
                f"input_format 只支持 bgr / rgb / gray，收到 '{input_format}'"
            )

        self._lock = threading.Lock()
        self._closed = False
        self._stderr_lines: List[str] = []

        self._process = subprocess.Popen(
            [self.worker_path, "--model", self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        if log_stderr:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, name="npu-stderr", daemon=True
            )
            self._stderr_thread.start()

        self._wait_for_ready()
        LOGGER.info("NPU worker 已就绪: %s", self.worker_path)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def _drain_stderr(self) -> None:
        """把 worker 的日志转到 Python logging，便于 ros2 launch 里看到。"""
        assert self._process.stderr is not None
        for raw_line in iter(self._process.stderr.readline, b""):
            text = raw_line.decode("utf-8", errors="replace").rstrip()
            if text:
                self._stderr_lines.append(text)
                del self._stderr_lines[:-50]
                LOGGER.info("[npu_worker] %s", text)

    def _wait_for_ready(self) -> None:
        """等待 worker 打印 magic，超时或进程退出就报错并带上 stderr 摘要。"""
        assert self._process.stdout is not None
        holder: dict = {}

        def _read() -> None:
            try:
                holder["data"] = self._process.stdout.read(4)
            except Exception as error:  # pragma: no cover - 异常路径
                holder["error"] = error

        reader = threading.Thread(target=_read, name="npu-ready", daemon=True)
        reader.start()
        reader.join(self.startup_timeout)

        if reader.is_alive():
            self.close()
            raise BackendUnavailable(
                f"NPU worker 启动超时（{self.startup_timeout:.0f}s）。"
                f"Worker 日志：{self._stderr_tail()}"
            )

        if "error" in holder:
            self.close()
            raise BackendUnavailable(f"读取 NPU worker 握手失败：{holder['error']}")

        data = holder.get("data", b"")
        if len(data) != 4:
            tail = self._stderr_tail()
            self.close()
            raise BackendUnavailable(
                "NPU worker 未正常握手就退出。"
                f"常见原因：模型文件不存在、VIPLite 驱动未加载、模型与 NPU 不匹配。"
                f"Worker 日志：{tail}"
            )

        magic = struct.unpack("<I", data)[0]
        if magic != MAGIC_READY:
            self.close()
            raise BackendUnavailable(
                f"NPU worker 握手 magic 异常: 0x{magic:08X}，期望 0x{MAGIC_READY:08X}"
            )

    def _stderr_tail(self) -> str:
        if not self._stderr_lines:
            return "(无输出)"
        return " | ".join(self._stderr_lines[-5:])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        process = self._process
        try:
            if process.stdin:
                process.stdin.close()
        except Exception:
            pass
        try:
            process.wait(timeout=3.0)
        except Exception:
            process.kill()

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def _forward(self, input_image: np.ndarray) -> Sequence[np.ndarray]:
        with self._lock:
            if self._closed:
                raise RuntimeError("NPU 后端已关闭")
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"NPU worker 已退出（code={self._process.returncode}）。"
                    f"日志：{self._stderr_tail()}"
                )
            return self._exchange(input_image)

    def _exchange(self, input_image: np.ndarray) -> List[np.ndarray]:
        assert self._process.stdin is not None and self._process.stdout is not None

        image = np.ascontiguousarray(input_image)
        if image.dtype != np.uint8:
            # NPU 输入是量化过的 uint8，这里做一次显式转换而不是静默截断。
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            height, width = image.shape
            fmt = FORMAT_CODES["gray"]
        else:
            height, width = image.shape[:2]
            fmt = FORMAT_CODES[self.input_format]

        payload = image.tobytes()
        header = struct.pack(
            "<IIIII", MAGIC_REQUEST, width, height, fmt, len(payload)
        )
        self._process.stdin.write(header)
        self._process.stdin.write(payload)
        self._process.stdin.flush()

        return self._read_response()

    def _read_exact(self, count: int) -> bytes:
        assert self._process.stdout is not None
        buffer = bytearray()
        while len(buffer) < count:
            chunk = self._process.stdout.read(count - len(buffer))
            if not chunk:
                raise RuntimeError(
                    "NPU worker 连接中断（读取应答时 EOF）。"
                    f"日志：{self._stderr_tail()}"
                )
            buffer.extend(chunk)
        return bytes(buffer)

    def _read_response(self) -> List[np.ndarray]:
        magic, status = struct.unpack("<Ii", self._read_exact(8))
        if magic != MAGIC_RESPONSE:
            raise RuntimeError(
                f"NPU worker 应答 magic 异常: 0x{magic:08X}，协议错位"
            )

        if status != 0:
            (message_length,) = struct.unpack("<I", self._read_exact(4))
            message = self._read_exact(message_length).decode("utf-8", errors="replace")
            raise RuntimeError(f"NPU worker 推理失败 (status={status}): {message}")

        (tensor_count,) = struct.unpack("<I", self._read_exact(4))
        tensors: List[np.ndarray] = []

        for index in range(tensor_count):
            raw_header = self._read_exact(28)
            ndim, d0, d1, d2, d3, dtype_code, scale, zero = struct.unpack(
                "<IIIIIfi", raw_header
            )
            if ndim > 4:
                raise RuntimeError(f"输出张量 {index} 维度 {ndim} 超出支持范围")

            dims = [d0, d1, d2, d3][:ndim]
            (nbytes,) = struct.unpack("<I", self._read_exact(4))
            data = self._read_exact(nbytes)

            dtype = DTYPE_CODES.get(dtype_code)
            if dtype is None:
                raise RuntimeError(f"输出张量 {index} 的 dtype 编码 {dtype_code} 未知")

            array = np.frombuffer(data, dtype=dtype)

            expected = int(np.prod(dims)) if dims else 0
            if array.size != expected:
                raise RuntimeError(
                    f"输出张量 {index} 字节数与形状不符: dims={dims} "
                    f"期望 {expected} 个元素，实际 {array.size} 个"
                )

            array = array.reshape(dims).copy()

            # 量化输出还原成浮点。scale/zero 由 worker 从 vip_query_output
            # 的 FIXED_POINT_POS / TF_SCALE 里取，float 模型时 scale 为 1。
            if dtype_code != 0 and scale not in (0.0, 1.0):
                array = (array.astype(np.float32) - float(zero)) * float(scale)
            elif dtype_code != 0:
                array = array.astype(np.float32)

            tensors.append(array)

        return tensors

    def describe(self) -> str:
        return f"npu(model={self.model_path}, fmt={self.input_format})"
