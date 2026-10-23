"""摄像头采集：封装 cv2.VideoCapture，读失败时自动重连。

## 后端选择为什么重要（实测结论）

在这台板子上（OpenCV 4.5.4 / Orange Pi 4 Pro）实测了三种打开方式，
结果是：

| 后端 | set(FOURCC/WIDTH/FPS) | 读回格式 | 实际分辨率 |
| --- | --- | --- | --- |
| 自动选择（落到 GStreamer） | 全部返回 False | 0，未报告 | 1280x720 |
| **CAP_V4L2** | 全部返回 True | **MJPG** | **640x480** |
| CAP_GSTREAMER | 全部返回 False | 0，未报告 | 1280x720 |

也就是说：

* **OpenCV 的自动选择会落到 GStreamer**，而 GStreamer 后端**不支持
  CAP_PROP_* 属性**（日志里会打印 `GStreamer: unhandled property`），
  于是 pixel_format / width / height / frame_rate **全部被静默忽略**，
  你会拿到摄像头默认的 1280x720 @ 10fps（YUYV）。
* **只有 CAP_V4L2 能真正设置像素格式**，MJPG 写进去后能读回，
  640x480 @ 30fps 全部生效。

所以规则是：**一旦指定了 pixel_format（不是 AUTO），就必须用 V4L2 后端**，
否则参数等于没设。pixel_format 保持 AUTO 时沿用自动选择，
以兼容其它板子的行为（有的板子 GStreamer 反而更稳）。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

VALID_ROTATIONS = (0, 90, 180, 270)
VALID_MIRROR_MODES = ("none", "horizontal", "vertical", "both")

#: 可选的采集后端。
#: auto  = 让 OpenCV 自己选（GStreamer 优先，但 CAP_PROP_* 会失效）
#: v4l2  = 强制 V4L2，能真正设置像素格式 / 分辨率 / 帧率
#: gstreamer = 强制 GStreamer
VALID_CAPTURE_BACKENDS = ("auto", "v4l2", "gstreamer")


class CameraSource:
    """封装 cv2.VideoCapture，读失败时自动重连。"""

    def __init__(
        self,
        camera_id: int = 0,
        camera_device: str = "",
        width: int = 640,
        height: int = 480,
        frame_rate: float = 0.0,
        pixel_format: str = "AUTO",
        rotation_degrees: int = 0,
        mirror_mode: str = "none",
        max_read_failures: int = 30,
        capture_backend: str = "auto",
    ) -> None:
        if rotation_degrees not in VALID_ROTATIONS:
            raise ValueError(f"rotation_degrees 必须是 {VALID_ROTATIONS} 之一")
        mirror = str(mirror_mode).strip().lower()
        if mirror not in VALID_MIRROR_MODES:
            raise ValueError(f"mirror_mode 必须是 {VALID_MIRROR_MODES} 之一")
        backend = str(capture_backend).strip().lower()
        if backend not in VALID_CAPTURE_BACKENDS:
            raise ValueError(
                f"capture_backend 必须是 {VALID_CAPTURE_BACKENDS} 之一，收到 {capture_backend!r}"
            )

        self.camera_id = int(camera_id)
        self.camera_device = str(camera_device)
        self.width = int(width)
        self.height = int(height)
        self.frame_rate = float(frame_rate)
        self.pixel_format = str(pixel_format).upper()
        self.rotation_degrees = int(rotation_degrees)
        self.mirror_mode = mirror
        self.max_read_failures = max(1, int(max_read_failures))
        self.capture_backend = backend

        self._capture: Optional[cv2.VideoCapture] = None
        self._read_failures = 0
        self._reported_set_results: List = []
        self.open()

    def _resolve_backend(self):
        """决定用哪个 OpenCV 后端，返回 (cap_argument, 说明文本)。

        规则（依据本文件顶部的实测结论）：
          * capture_backend 显式给了 v4l2 / gstreamer -> 用它
          * auto：如果指定了 pixel_format（不是 AUTO），
            **必须**用 V4L2，否则 GStreamer 后端会把参数全部忽略；
            否则沿用 OpenCV 自动选择，兼容其它板子。
        """
        if self.capture_backend == "v4l2":
            return cv2.CAP_V4L2, "V4L2（显式指定）"

        if self.capture_backend == "gstreamer":
            return cv2.CAP_GSTREAMER, "GStreamer（显式指定）"

        # auto
        if self.pixel_format != "AUTO":
            return (
                cv2.CAP_V4L2,
                "V4L2（因为指定了 pixel_format，自动改用 V4L2 —— "
                "GStreamer 后端不支持 CAP_PROP_*，参数会被静默忽略）",
            )

        return cv2.CAP_ANY, "自动选择"

    def open(self) -> None:
        """打开摄像头，失败抛 RuntimeError。"""
        source = self.camera_device if self.camera_device else self.camera_id
        backend, backend_note = self._resolve_backend()

        self._capture = (
            cv2.VideoCapture(source)
            if backend == cv2.CAP_ANY
            else cv2.VideoCapture(source, backend)
        )
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"无法打开摄像头 {source}")

        LOGGER.info("采集后端: %s", backend_note)

        # ------------------------------------------------------------------
        # 参数设置顺序很关键：FOURCC -> 分辨率 -> 帧率
        #
        # V4L2 的格式协商是「先定像素格式，再在它支持的能力表里挑分辨率和帧率」。
        # 驱动默认用 YUYV（未压缩），此时 1280x720 只能给到 10fps；
        # 如果先把分辨率设成 1280x720、再想切成 MJPG，分辨率与帧率已经被
        # YUYV 的组合锁死，切格式也救不回来。
        # 先设 MJPG 再设分辨率，同样的 1280x720 就能拿到 30fps。
        # ------------------------------------------------------------------
        self._reported_set_results = []
        if self.pixel_format != "AUTO":
            fourcc = cv2.VideoWriter_fourcc(*self.pixel_format)
            self._reported_set_results.append(
                ("FOURCC=" + self.pixel_format,
                 bool(self._capture.set(cv2.CAP_PROP_FOURCC, fourcc)))
            )

        self._reported_set_results.append(
            ("WIDTH", bool(self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)))
        )
        self._reported_set_results.append(
            ("HEIGHT", bool(self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)))
        )

        if self.frame_rate > 0.0:
            self._reported_set_results.append(
                ("FPS", bool(self._capture.set(cv2.CAP_PROP_FPS, self.frame_rate)))
            )

        # ------------------------------------------------------------------
        # 读回实际协商结果。摄像头不一定答应你请求的参数，
        # 静默降级（要 30fps 却只给 10fps、要 640x480 却给 1280x720）
        # 会让后面的数据集和训练输入尺寸对不上，所以这里显式报出来。
        # ------------------------------------------------------------------
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        actual_format = "".join(
            chr((actual_fourcc >> (8 * index)) & 0xFF) for index in range(4)
        ).strip("\x00 ")

        LOGGER.info(
            "摄像头已打开: %s, 后端=%s, %dx%d @ %.1f fps, 格式=%s",
            source, self._capture.getBackendName(),
            actual_width, actual_height, actual_fps,
            actual_format or "(未报告)",
        )

        self.actual_width = actual_width
        self.actual_height = actual_height
        self.actual_fps = actual_fps
        self.actual_format = actual_format

        warnings: List[str] = []

        # set() 返回 False 是最早的信号：说明这个后端根本不认这些属性。
        # 实测：GStreamer 后端下 FOURCC / WIDTH / FPS 全部返回 False，
        # 参数被静默忽略，于是拿到摄像头默认的 1280x720 @ 10fps。
        failed_sets = [name for name, ok in self._reported_set_results if not ok]
        if failed_sets:
            warnings.append(
                f"这些属性设置失败（后端直接拒绝）: {', '.join(failed_sets)}"
            )

        if actual_width != self.width or actual_height != self.height:
            warnings.append(
                f"分辨率被驱动改为 {actual_width}x{actual_height}"
                f"（请求 {self.width}x{self.height}）"
            )
        if self.frame_rate > 0.0 and actual_fps > 0 and abs(actual_fps - self.frame_rate) > 0.5:
            warnings.append(
                f"帧率被驱动改为 {actual_fps:.1f}（请求 {self.frame_rate:.1f}）"
            )
        if self.pixel_format != "AUTO" and actual_format and actual_format != self.pixel_format:
            warnings.append(
                f"像素格式被驱动改为 {actual_format}（请求 {self.pixel_format}）"
            )

        for message in warnings:
            LOGGER.warning("  %s", message)

        if warnings:
            LOGGER.warning(
                "  摄像头没有完全接受请求的参数。两个最常见的原因：\n"
                "    1) 后端不对 —— GStreamer 后端不支持 CAP_PROP_*，参数会被全部忽略。\n"
                "       本类在指定 pixel_format 时已自动改用 V4L2；仍失败可显式设\n"
                "       capture_backend='v4l2'（或用 probe_camera.py 实测各后端）\n"
                "    2) 像素格式不对 —— 未压缩的 YUYV 在高分辨率下帧率很低，改用 MJPG"
            )

    def _apply_orientation(self, frame: np.ndarray) -> np.ndarray:
        """按参数做旋转与镜像，保证送进模型的方向和实际安装方向一致。"""
        if self.rotation_degrees == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_degrees == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_degrees == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.mirror_mode == "horizontal":
            frame = cv2.flip(frame, 1)
        elif self.mirror_mode == "vertical":
            frame = cv2.flip(frame, 0)
        elif self.mirror_mode == "both":
            frame = cv2.flip(frame, -1)
        return frame

    def read(self) -> Optional[np.ndarray]:
        """读一帧 BGR 图；连续失败时尝试重连，仍失败返回 None。"""
        if self._capture is None:
            try:
                self.open()
            except RuntimeError:
                return None

        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._read_failures += 1
            if self._read_failures >= self.max_read_failures:
                LOGGER.warning(
                    "连续 %d 次读帧失败，尝试重新打开摄像头", self._read_failures
                )
                self._read_failures = 0
                self.release()
                try:
                    self.open()
                except RuntimeError as error:
                    LOGGER.error("重开摄像头失败: %s", error)
            return None

        self._read_failures = 0
        return self._apply_orientation(frame)

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False
