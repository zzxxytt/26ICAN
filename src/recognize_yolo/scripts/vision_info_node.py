#!/usr/bin/env python3
"""识别信息的二次分发节点：自定义消息 -> JSON / vision_msgs。

detector_node 已经直接发布了自定义消息和 JSON。本节点的价值在于：

1. 把 JSON 发布的职责从推理节点里剥出来，让检测节点专注在一个输出上。
   不想用自定义消息的下游只订阅这里即可。
2. 额外提供 vision_msgs/Detection2DArray，兼容生态里现成的工具与算法。

订阅：
    ~/input_detections   recognize_yolo_msgs/msg/DetectionArray

发布：
    ~/detections_json         std_msgs/msg/String
    ~/vision_detections       vision_msgs/msg/Detection2DArray（装了 vision_msgs 时）
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from recognize_yolo.serialize import detection_array_to_json

try:
    from recognize_yolo_msgs.msg import DetectionArray as DetectionArrayMsg

    MESSAGES_AVAILABLE = True
except ImportError:  # pragma: no cover
    MESSAGES_AVAILABLE = False
    DetectionArrayMsg = None

try:
    from vision_msgs.msg import (
        BoundingBox2D,
        Detection2D,
        Detection2DArray,
        ObjectHypothesisWithPose,
    )

    VISION_MSGS_AVAILABLE = True
except ImportError:  # pragma: no cover
    VISION_MSGS_AVAILABLE = False


class VisionInfoNode(Node):
    """订阅检测结果，转成更好用的下游格式。"""

    def __init__(self) -> None:
        super().__init__("recognize_yolo_info")

        self.declare_parameter("input_topic", "/recognize_yolo/detections")
        self.declare_parameter("publish_json", True)
        self.declare_parameter("publish_vision_msgs", True)
        # 只转发高于该分数的目标，便于下游做二次过滤。
        self.declare_parameter("min_score", 0.0)

        if not MESSAGES_AVAILABLE:
            raise RuntimeError(
                "recognize_yolo 消息未编译，无法订阅检测结果。"
                "请先 colcon build --packages-select recognize_yolo"
            )

        input_topic = str(self.get_parameter("input_topic").value)
        self.min_score = float(self.get_parameter("min_score").value)

        self.subscription = self.create_subscription(
            DetectionArrayMsg, input_topic, self._on_detections, 10
        )

        self.json_publisher = (
            self.create_publisher(String, "~/detections_json", 10)
            if bool(self.get_parameter("publish_json").value)
            else None
        )

        self.want_vision_msgs = bool(self.get_parameter("publish_vision_msgs").value)
        self.vision_publisher = None
        if self.want_vision_msgs:
            if VISION_MSGS_AVAILABLE:
                self.vision_publisher = self.create_publisher(
                    Detection2DArray, "~/vision_detections", 10
                )
            else:
                self.get_logger().warning(
                    "未安装 vision_msgs，跳过 Detection2DArray 发布。"
                    "需要时执行：sudo apt install ros-$ROS_DISTRO-vision-msgs"
                )

        self.get_logger().info(
            f"识别信息转发已启动: 订阅 {input_topic}, json={self.json_publisher is not None}, "
            f"vision_msgs={self.vision_publisher is not None}"
        )

    def _on_detections(self, message) -> None:
        detections = [
            _DetectionsAdapter(item) for item in message.detections
        ]
        detections = [item for item in detections if item.score >= self.min_score]

        if self.json_publisher is not None:
            stamp = (
                float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
            )
            payload = detection_array_to_json(
                detections,
                image_width=int(message.image_width),
                image_height=int(message.image_height),
                stamp_sec=stamp,
                frame_id=message.header.frame_id,
                inference_ms=float(message.inference_ms),
                backend=message.backend,
                source=message.source,
                input_size=int(message.input_size),
            )
            json_message = String()
            json_message.data = payload
            self.json_publisher.publish(json_message)

        if self.vision_publisher is not None:
            self.vision_publisher.publish(self._to_vision_msgs(message, detections))

    def _to_vision_msgs(self, message, detections):
        """转成 vision_msgs/Detection2DArray。"""
        output = Detection2DArray()
        output.header = message.header

        for item in detections:
            detection = Detection2D()
            detection.header = message.header

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(item.class_id)
            hypothesis.hypothesis.score = float(item.score)
            detection.results.append(hypothesis)

            box = BoundingBox2D()
            box.center.position.x = float(item.center_x)
            box.center.position.y = float(item.center_y)
            box.center.theta = 0.0
            box.size_x = float(item.width)
            box.size_y = float(item.height)
            detection.bbox = box

            output.detections.append(detection)

        return output


class _DetectionsAdapter:
    """给消息里的检测项套一层，复用 serialize 里依赖 Detection 属性的代码。"""

    __slots__ = (
        "class_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax", "track_id",
    )

    def __init__(self, entry) -> None:
        self.class_id = int(entry.class_id)
        self.class_name = str(entry.class_name)
        self.score = float(entry.score)
        self.xmin = float(entry.xmin)
        self.ymin = float(entry.ymin)
        self.xmax = float(entry.xmax)
        self.ymax = float(entry.ymax)
        self.track_id = int(entry.track_id)

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = VisionInfoNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
