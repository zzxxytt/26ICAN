#!/usr/bin/env python3
"""YOLO 识别主节点：采图 -> 推理 -> 发布识别信息。

发布的话题（都在 ~ 私有命名空间下，默认节点名 recognize_yolo）：

    ~/detections       recognize_yolo_msgs/msg/DetectionArray   自定义消息，推荐
    ~/detections_json  std_msgs/msg/String                 JSON 字符串，免自定义消息
    ~/debug_image      sensor_msgs/msg/Image               带框的调试图，可用参数关掉

本节点只做感知，不发布 /cmd_vel 或任何控制指令，控制逻辑放到跟随节点里。
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String

from recognize_yolo.backend import BackendUnavailable, create_backend, summarize_tensors
from recognize_yolo.camera_source import CameraSource
from recognize_yolo.decode import DecodeConfig, Detection, load_class_names
from recognize_yolo.drawing import draw_detections, draw_status_bar, summarize_detections
from recognize_yolo.preprocess import letterbox, to_uint8_hwc
from recognize_yolo.serialize import detection_array_to_json

try:
    from recognize_yolo_msgs.msg import Detection as DetectionMsg
    from recognize_yolo_msgs.msg import DetectionArray as DetectionArrayMsg

    MESSAGES_AVAILABLE = True
except ImportError:  # pragma: no cover - 只在消息未编译时触发
    MESSAGES_AVAILABLE = False
    DetectionMsg = None
    DetectionArrayMsg = None

try:
    from cv_bridge import CvBridge

    CV_BRIDGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV_BRIDGE_AVAILABLE = False
    CvBridge = None


# 传感器数据用 BEST_EFFORT：丢帧比阻塞好，控制回路要的是低延迟而不是完整。
SENSOR_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class YoloDetectorNode(Node):
    """把 YOLO 推理包装成 ROS 2 节点。"""

    def __init__(self) -> None:
        super().__init__("recognize_yolo")
        self._declare_parameters()

        self.backend_name = str(self.get_parameter("backend").value).strip().lower()
        self.input_size = int(self.get_parameter("input_size").value)
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self.publish_json = bool(self.get_parameter("publish_json").value)
        self.log_detections = bool(self.get_parameter("log_detections").value)
        self.process_rate = float(self.get_parameter("process_rate").value)
        self.source_mode = str(self.get_parameter("source").value).strip().lower()

        if self.input_size <= 0 or self.input_size % 32 != 0:
            raise ValueError(
                f"input_size 必须是 32 的正整数倍，收到 {self.input_size}"
            )

        # 类别名：优先用文件，其次用参数里的逗号分隔列表。
        class_file = str(self.get_parameter("class_names_file").value).strip()
        if class_file:
            self.class_names = load_class_names(class_file)
            if not self.class_names:
                raise ValueError(f"类别文件为空或不存在: {class_file}")
        else:
            raw_names = str(self.get_parameter("class_names").value).strip()
            self.class_names = (
                [item.strip() for item in raw_names.split(",") if item.strip()]
                if raw_names
                else []
            )

        declared_classes = int(self.get_parameter("num_classes").value)
        if self.class_names and declared_classes != len(self.class_names):
            self.get_logger().warning(
                f"num_classes={declared_classes} 与类别名数量 {len(self.class_names)} 不一致，"
                f"以 num_classes 为准"
            )

        self.decode_config = DecodeConfig(
            num_classes=declared_classes,
            input_size=self.input_size,
            conf_threshold=float(self.get_parameter("conf_threshold").value),
            iou_threshold=float(self.get_parameter("iou_threshold").value),
            max_detections=int(self.get_parameter("max_detections").value),
            min_box_area=float(self.get_parameter("min_box_area").value),
            apply_sigmoid=bool(self.get_parameter("apply_sigmoid").value),
            class_names=self.class_names,
        )

        self.rgb_input = str(self.get_parameter("input_color").value).strip().lower() == "rgb"

        # ------------------------------------------------------------------
        # 发布者
        # ------------------------------------------------------------------
        if MESSAGES_AVAILABLE:
            self.detection_publisher = self.create_publisher(
                DetectionArrayMsg, "~/detections", 10
            )
        else:
            self.detection_publisher = None
            self.get_logger().error(
                "recognize_yolo 消息未编译，只有 JSON 话题可用。"
                "请在板上重新 colcon build --packages-select recognize_yolo"
            )

        self.json_publisher = (
            self.create_publisher(String, "~/detections_json", 10)
            if self.publish_json
            else None
        )
        self.debug_publisher = (
            self.create_publisher(RosImage, "~/debug_image", SENSOR_QOS)
            if self.publish_debug_image
            else None
        )

        # ------------------------------------------------------------------
        # 图像来源
        # ------------------------------------------------------------------
        self.camera: Optional[CameraSource] = None
        self.bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
        self._last_topic_image: Optional[np.ndarray] = None
        self._last_topic_stamp = 0.0
        self._last_topic_wallclock = 0.0
        self.image_subscription = None

        self._setup_source()

        # ------------------------------------------------------------------
        # 后端
        # ------------------------------------------------------------------
        self.backend = None
        self._warned_no_frame = False

        # ------------------------------------------------------------------
        # 计时
        # ------------------------------------------------------------------
        self._frame_count = 0
        self._setup_time = time.monotonic()
        self._last_fps_time = self._setup_time
        self._fps = 0.0
        self._last_error_log_time = 0.0

        period = 1.0 / self.process_rate if self.process_rate > 0.0 else 0.0
        if period > 0.0:
            self.timer = self.create_timer(period, self._on_timer)
        else:
            self.timer = None
            # 订阅模式下靠回调驱动，避免定时器空转。
            self.get_logger().info("process_rate=0，改为每收到一帧图像推理一次")

        self.get_logger().info(
            f"识别参数: backend={self.backend_name}, input_size={self.input_size}, "
            f"num_classes={self.decode_config.num_classes}, "
            f"conf={self.decode_config.conf_threshold}, "
            f"iou={self.decode_config.iou_threshold}, source={self.source_mode}"
        )

    # ------------------------------------------------------------------
    # 参数
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        # 图像来源
        self.declare_parameter("source", "camera")  # camera / image_topic
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_id", 0)
        self.declare_parameter("camera_device", "")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("frame_rate", 0.0)
        self.declare_parameter("pixel_format", "AUTO")
        self.declare_parameter("rotation_degrees", 0)
        self.declare_parameter("mirror_mode", "none")
        self.declare_parameter("fallback_to_camera", True)
        self.declare_parameter("topic_wait_timeout", 5.0)

        # 模型与后端
        self.declare_parameter("backend", "npu")  # npu / onnx / mock
        self.declare_parameter("input_size", 640)
        self.declare_parameter("input_color", "rgb")  # bgr / rgb
        self.declare_parameter("num_classes", 80)
        self.declare_parameter("class_names", "")
        self.declare_parameter("class_names_file", "")

        # NPU 后端
        self.declare_parameter("npu_worker_path", "")
        self.declare_parameter("npu_model_path", "")
        self.declare_parameter("npu_startup_timeout", 30.0)

        # ONNX 后端
        self.declare_parameter("onnx_model_path", "")
        self.declare_parameter("onnx_input_name", "")
        self.declare_parameter("onnx_providers", "CPUExecutionProvider")
        self.declare_parameter("onnx_intra_op_threads", 0)
        self.declare_parameter("onnx_scale", 1.0 / 255.0)
        self.declare_parameter("onnx_mean", [0.0, 0.0, 0.0])
        self.declare_parameter("onnx_std", [1.0, 1.0, 1.0])

        # 解码
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("max_detections", 100)
        self.declare_parameter("min_box_area", 0.0)
        self.declare_parameter("apply_sigmoid", False)

        # 运行
        self.declare_parameter("process_rate", 10.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("publish_json", True)
        self.declare_parameter("log_detections", True)
        self.declare_parameter("log_tensor_shapes", False)

    # ------------------------------------------------------------------
    # 图像来源
    # ------------------------------------------------------------------
    def _setup_source(self) -> None:
        if self.source_mode == "image_topic":
            if not CV_BRIDGE_AVAILABLE:
                raise RuntimeError(
                    "source=image_topic 需要 cv_bridge：sudo apt install ros-$ROS_DISTRO-cv-bridge"
                )
            topic = str(self.get_parameter("image_topic").value)
            self.image_subscription = self.create_subscription(
                RosImage, topic, self._on_image, SENSOR_QOS
            )
            self.get_logger().info(f"订阅图像话题: {topic}")
            return

        if self.source_mode != "camera":
            raise ValueError(f"source 只支持 camera / image_topic，收到 '{self.source_mode}'")

        self.camera = CameraSource(
            camera_id=int(self.get_parameter("camera_id").value),
            camera_device=str(self.get_parameter("camera_device").value),
            width=int(self.get_parameter("image_width").value),
            height=int(self.get_parameter("image_height").value),
            frame_rate=float(self.get_parameter("frame_rate").value),
            pixel_format=str(self.get_parameter("pixel_format").value),
            rotation_degrees=int(self.get_parameter("rotation_degrees").value),
            mirror_mode=str(self.get_parameter("mirror_mode").value),
        )
        self.get_logger().info("本节点只发布识别结果，不发布 /cmd_vel 或其他控制指令")

    def _on_image(self, message: RosImage) -> None:
        """图像话题回调。"""
        if self.bridge is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as error:
            self.get_logger().error(f"图像转换失败: {error}", throttle_duration_sec=5.0)
            return

        self._last_topic_image = frame
        self._last_topic_stamp = (
            float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        )
        self._last_topic_wallclock = time.monotonic()

        if self.timer is None:
            self._process_frame(frame, self._last_topic_stamp)

    def _grab_frame(self):
        """按当前模式取一帧，返回 (frame, stamp, source_name) 或 None。"""
        if self.source_mode == "image_topic":
            if self._last_topic_image is None:
                return None
            return self._last_topic_image, self._last_topic_stamp, "image_topic"

        if self.camera is None:
            return None
        frame = self.camera.read()
        if frame is None:
            return None
        return frame, self.get_clock().now().nanoseconds * 1e-9, "camera"

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        if self.source_mode == "image_topic":
            self._maybe_fallback_to_camera()

        grabbed = self._grab_frame()
        if grabbed is None:
            if not self._warned_no_frame:
                self.get_logger().warning("暂时取不到图像帧，等待中……")
                self._warned_no_frame = True
            return
        self._warned_no_frame = False

        frame, stamp, source_name = grabbed
        self._process_frame(frame, stamp, source_name)

    def _maybe_fallback_to_camera(self) -> None:
        """图像话题迟迟没有数据时，可选地退回直接开摄像头。"""
        if not bool(self.get_parameter("fallback_to_camera").value):
            return
        if self.image_subscription is not None:
            return

        timeout = float(self.get_parameter("topic_wait_timeout").value)
        if timeout <= 0.0:
            return

        # 从节点启动算起，超时仍没收到任何一帧就回退。
        if time.monotonic() - self._setup_time < timeout:
            return
        if self._last_topic_image is not None:
            return

        self.get_logger().warning(
            f"图像话题 {self.get_parameter('image_topic').value} 在 {timeout:.0f}s 内没有数据，"
            f"回退到直接打开摄像头"
        )
        self.destroy_subscription(self.image_subscription)
        self.image_subscription = None
        self.source_mode = "camera"
        self.camera = CameraSource(
            camera_id=int(self.get_parameter("camera_id").value),
            camera_device=str(self.get_parameter("camera_device").value),
            width=int(self.get_parameter("image_width").value),
            height=int(self.get_parameter("image_height").value),
            frame_rate=float(self.get_parameter("frame_rate").value),
            pixel_format=str(self.get_parameter("pixel_format").value),
            rotation_degrees=int(self.get_parameter("rotation_degrees").value),
            mirror_mode=str(self.get_parameter("mirror_mode").value),
        )

    def _process_frame(self, frame: np.ndarray, stamp: float, source_name: str = "camera") -> None:
        started = time.perf_counter()

        # letterbox 只做一次：后端拿到的是已经缩放好的图，解码时再把框映射回原图。
        padded, letterbox_info = letterbox(frame, self.input_size)

        if self.backend is None:
            try:
                self.backend = self._create_backend(letterbox_info)
            except BackendUnavailable as error:
                self.get_logger().error(
                    f"后端初始化失败: {error}。"
                    f"如果是硬件/模型还没准备好，可先用 -p backend:=mock 验证链路",
                    throttle_duration_sec=10.0,
                )
                return

        # letterbox 参数每帧可能不同（图像尺寸变化），推理前刷新一次。
        self.backend.letterbox_info = letterbox_info

        try:
            result = self.backend.infer(padded)
        except Exception as error:
            self._log_throttled_error(f"推理失败: {error}")
            return

        if bool(self.get_parameter("log_tensor_shapes").value):
            self.get_logger().info(f"模型输出: {summarize_tensors(result.raw_tensors)}")

        height, width = frame.shape[:2]
        detections = result.detections

        self._publish(
            detections=detections,
            frame=frame,
            stamp=stamp,
            source_name=source_name,
            inference_ms=result.total_ms,
        )

        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now
            self.get_logger().info(
                f"{self._fps:.1f} fps | 前向 {result.forward_ms:.1f}ms | "
                f"整帧 {result.total_ms:.1f}ms | {len(detections)} 个目标 | "
                f"{summarize_detections(detections)}"
            )

        if self.log_detections and detections and not self._fps:
            # fps 统计每秒才刷一次，这里补一条即时日志方便调参。
            self.get_logger().debug(
                f"检测到 {len(detections)} 个目标: {summarize_detections(detections)}"
            )

    def _create_backend(self, letterbox_info):
        backend_kwargs = {}

        if self.backend_name in ("npu", "vip", "viplite", "vip_lite"):
            worker_path = str(self.get_parameter("npu_worker_path").value).strip()
            model_path = str(self.get_parameter("npu_model_path").value).strip()
            if not worker_path:
                raise BackendUnavailable(
                    "backend=npu 需要设置参数 npu_worker_path，"
                    "例如 install/recognize_yolo/lib/recognize_yolo/yolo_npu_worker"
                )
            if not model_path:
                raise BackendUnavailable(
                    "backend=npu 需要设置参数 npu_model_path，例如 "
                    "~/models/yolov5s_rt_uint8_a733.nb"
                )
            backend_kwargs = {
                "worker_path": worker_path,
                "model_path": model_path,
                "input_format": "rgb" if self.rgb_input else "bgr",
                "startup_timeout": float(
                    self.get_parameter("npu_startup_timeout").value
                ),
            }

        elif self.backend_name in ("onnx", "onnxruntime", "cpu"):
            model_path = str(self.get_parameter("onnx_model_path").value).strip()
            if not model_path:
                raise BackendUnavailable(
                    "backend=onnx 需要设置参数 onnx_model_path，例如 "
                    "~/models/yolov8n.onnx"
                )
            backend_kwargs = {
                "model_path": model_path,
                "input_name": str(self.get_parameter("onnx_input_name").value),
                "providers": str(self.get_parameter("onnx_providers").value),
                "intra_op_threads": int(
                    self.get_parameter("onnx_intra_op_threads").value
                ),
                "scale": float(self.get_parameter("onnx_scale").value),
                "mean": tuple(self.get_parameter("onnx_mean").value),
                "std": tuple(self.get_parameter("onnx_std").value),
            }

        backend = create_backend(
            self.backend_name, self.decode_config, letterbox_info, **backend_kwargs
        )
        self.get_logger().info(f"推理后端已就绪: {backend.describe()}")
        return backend

    def _log_throttled_error(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_error_log_time >= 5.0:
            self._last_error_log_time = now
            self.get_logger().error(message)

    # ------------------------------------------------------------------
    # 发布
    # ------------------------------------------------------------------
    def _publish(
        self,
        detections,
        frame: np.ndarray,
        stamp: float,
        source_name: str,
        inference_ms: float,
    ) -> None:
        height, width = frame.shape[:2]
        sec = int(stamp)
        nanosec = int((stamp - sec) * 1e9)

        if self.detection_publisher is not None:
            message = DetectionArrayMsg()
            message.header.stamp.sec = sec
            message.header.stamp.nanosec = nanosec
            message.header.frame_id = "camera"
            message.source = source_name
            message.image_width = int(width)
            message.image_height = int(height)
            message.inference_ms = float(inference_ms)
            message.input_size = int(self.input_size)
            message.backend = str(self.backend_name)

            for item in detections:
                entry = DetectionMsg()
                entry.class_id = int(item.class_id)
                entry.class_name = item.class_name or f"class_{item.class_id}"
                entry.score = float(item.score)
                entry.xmin = float(item.xmin)
                entry.ymin = float(item.ymin)
                entry.xmax = float(item.xmax)
                entry.ymax = float(item.ymax)
                entry.center_x = float(item.center_x)
                entry.center_y = float(item.center_y)
                entry.norm_center_x = float(item.center_x) / float(width) if width else 0.0
                entry.norm_center_y = float(item.center_y) / float(height) if height else 0.0
                entry.track_id = int(item.track_id)
                message.detections.append(entry)

            self.detection_publisher.publish(message)

        if self.json_publisher is not None:
            payload = detection_array_to_json(
                detections,
                image_width=width,
                image_height=height,
                stamp_sec=stamp,
                frame_id="camera",
                inference_ms=inference_ms,
                backend=self.backend_name,
                source=source_name,
                input_size=self.input_size,
            )
            json_message = String()
            json_message.data = payload
            self.json_publisher.publish(json_message)

        if self.debug_publisher is not None and self.bridge is not None:
            annotated = draw_detections(frame, detections)
            annotated = draw_status_bar(
                annotated,
                backend=self.backend_name,
                inference_ms=inference_ms,
                detection_count=len(detections),
                fps=self._fps,
            )
            try:
                image_message = self.bridge.cv2_to_imgmsg(
                    annotated, encoding="bgr8"
                )
                image_message.header.stamp.sec = sec
                image_message.header.stamp.nanosec = nanosec
                image_message.header.frame_id = "camera"
                self.debug_publisher.publish(image_message)
            except Exception as error:
                self._log_throttled_error(f"发布调试图失败: {error}")

    # ------------------------------------------------------------------
    # 退出
    # ------------------------------------------------------------------
    def destroy_node(self) -> bool:
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception as error:  # pragma: no cover
                self.get_logger().warning(f"关闭后端时出错: {error}")
        if self.camera is not None:
            self.camera.release()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = YoloDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(f"节点异常退出: {error}")
        else:
            print(f"recognize_yolo 启动失败: {error}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
