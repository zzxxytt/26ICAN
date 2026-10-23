#!/usr/bin/env python3
"""Display the robot camera without publishing any control commands."""

import time

import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

class CameraViewer(Node):
    """Small camera viewer used to verify mounting orientation."""

    VALID_ROTATIONS = (0, 90, 180, 270)
    VALID_MIRROR_MODES = ("none", "horizontal", "vertical", "both")

    def __init__(self) -> None:
        super().__init__("camera_viewer")

        self.declare_parameter("camera_id", 0)
        self.declare_parameter("camera_device", "")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("frame_rate", 0.0)
        self.declare_parameter("pixel_format", "AUTO")
        self.declare_parameter("display_rate", 30.0)
        self.declare_parameter("rotation_degrees", 0)
        self.declare_parameter("mirror_mode", "none")
        self.declare_parameter("window_name", "Wheel Robot Camera (Q/Esc to quit)")
        self.declare_parameter("publish_image", True)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("show_window", True)
        self.declare_parameter("capture_backend", "auto")
        self.declare_parameter("frame_id", "camera_link")

        self.camera_id = int(self.get_parameter("camera_id").value)
        self.camera_device = str(self.get_parameter("camera_device").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.frame_rate = float(self.get_parameter("frame_rate").value)
        self.pixel_format = str(self.get_parameter("pixel_format").value).upper()
        self.display_rate = float(self.get_parameter("display_rate").value)
        self.rotation_degrees = int(self.get_parameter("rotation_degrees").value)
        self.mirror_mode = str(self.get_parameter("mirror_mode").value).lower()
        self.window_name = str(self.get_parameter("window_name").value)
        self.publish_image = bool(self.get_parameter("publish_image").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.capture_backend = str(self.get_parameter("capture_backend").value).lower()
        self.frame_id = str(self.get_parameter("frame_id").value)

        self._validate_parameters()
        
        # capture_backend=auto 沿用 OpenCV 自动选择（这台板子会落到 GStreamer，
        # CAP_PROP_* 全部失效，实际拿到 1280x720）。要真正生效像素格式/分辨率/
        # 帧率，必须用 v4l2。参考 recognize_yolo/recognize_yolo/camera_source.py。
        camera_source = self.camera_device if self.camera_device else self.camera_id
        backend_flags = {
            "auto": cv2.CAP_ANY,
            "v4l2": cv2.CAP_V4L2,
            "gstreamer": cv2.CAP_GSTREAMER,
        }
        if self.capture_backend not in backend_flags:
            raise ValueError("capture_backend 只能是 auto、v4l2 或 gstreamer")
        self.capture = cv2.VideoCapture(
            camera_source, backend_flags[self.capture_backend]
        )
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"无法打开摄像头 {camera_source}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        if self.pixel_format != "AUTO":
            fourcc = cv2.VideoWriter_fourcc(*self.pixel_format)
            self.capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        if self.frame_rate > 0.0:
            self.capture.set(cv2.CAP_PROP_FPS, self.frame_rate)

        if self.show_window:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            except cv2.error as error:
                self.capture.release()
                raise RuntimeError(
                    "无法创建显示窗口；请在有桌面的终端运行，"
                    "或设 show_window:=false"
                ) from error

        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        backend_name = self.capture.getBackendName()
        self.get_logger().info(
            f"摄像头已打开: {camera_source}, 后端={backend_name}, "
            f"{actual_width}x{actual_height} @ {actual_fps:.1f} fps, "
            f"旋转={self.rotation_degrees}°, 镜像={self.mirror_mode}"
        )
        self.get_logger().info(
            "本节点只发布图像，不发布 /cmd_vel 或其他控制指令"
        )

        self._image_publisher = None
        if self.publish_image:
            qos = QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self._image_publisher = self.create_publisher(
                Image, self.image_topic, qos
            )
            self.get_logger().info(
                f"发布图像到 {self.image_topic} (bgr8, BEST_EFFORT, depth 1)"
            )

        self._last_frame_time = time.monotonic()
        self._display_fps = 0.0
        self._consecutive_read_failures = 0
        self._first_frame_reported = False
        timer_rate = max(1.0, min(self.display_rate, 120.0))
        self.timer = self.create_timer(1.0 / timer_rate, self._show_next_frame)

    def _validate_parameters(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width 和 image_height 必须大于 0")
        if self.camera_id < 0 and not self.camera_device:
            raise ValueError("camera_id 必须大于等于 0，或设置 camera_device")
        if self.frame_rate < 0.0:
            raise ValueError("frame_rate 必须大于等于 0；0 表示自动协商")
        if self.pixel_format != "AUTO" and len(self.pixel_format) != 4:
            raise ValueError(
                "pixel_format 必须是 AUTO 或四字符格式，例如 YUYV、MJPG"
            )
        if self.display_rate <= 0.0:
            raise ValueError("display_rate 必须大于 0")
        if self.rotation_degrees not in self.VALID_ROTATIONS:
            raise ValueError("rotation_degrees 只能是 0、90、180 或 270")
        if self.mirror_mode not in self.VALID_MIRROR_MODES:
            raise ValueError(
                "mirror_mode 只能是 none、horizontal、vertical 或 both"
            )

    def _transform_frame(self, frame):
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

    def _publish_frame(self, frame) -> None:
        height, width = frame.shape[:2]
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.height = int(height)
        message.width = int(width)
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = int(width * 3)
        message.data = np.ascontiguousarray(frame).tobytes()
        self._image_publisher.publish(message)

    def _show_next_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self._consecutive_read_failures += 1
            if self._consecutive_read_failures == 1 or (
                self._consecutive_read_failures % 30 == 0
            ):
                self.get_logger().error(
                    f"摄像头读取失败（连续 {self._consecutive_read_failures} 帧）"
                )
            return

        self._consecutive_read_failures = 0
        if not self._first_frame_reported:
            height, width = frame.shape[:2]
            self.get_logger().info(
                f"收到首帧: {width}x{height}, 平均亮度={frame.mean():.1f}"
            )
            self._first_frame_reported = True

        now = time.monotonic()
        elapsed = now - self._last_frame_time
        if elapsed > 0.0:
            instant_fps = 1.0 / elapsed
            if self._display_fps == 0.0:
                self._display_fps = instant_fps
            else:
                self._display_fps = 0.9 * self._display_fps + 0.1 * instant_fps
        self._last_frame_time = now

        frame = self._transform_frame(frame)

        # 发布必须在画 FPS/TOP/LEFT 叠加之前，下游拿到的才是干净原图
        if self._image_publisher is not None:
            self._publish_frame(frame)

        if not self.show_window:
            return

        height, width = frame.shape[:2]
        cv2.putText(
            frame,
            f"TOP  {width}x{height}  {self._display_fps:.1f} fps",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "LEFT",
            (12, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        # WND_PROP_VISIBLE incorrectly reports zero in some X11/remote desktop
        # environments even while the window is open. Do not use it as an exit
        # condition, otherwise the first queued frame is destroyed before it is
        # painted and the user sees only a black window.
        if key in (ord("q"), ord("Q"), 27):
            self.get_logger().info("关闭摄像头查看窗口")
            rclpy.shutdown()

    def close(self) -> None:
        if hasattr(self, "timer"):
            self.timer.cancel()
        if hasattr(self, "capture"):
            self.capture.release()
        if getattr(self, "show_window", True):
            cv2.destroyAllWindows()

def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = CameraViewer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError) as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f"摄像头查看节点启动失败: {error}")
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
