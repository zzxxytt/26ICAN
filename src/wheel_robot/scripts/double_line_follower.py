#!/usr/bin/env python3
"""Safe dual-white-line follower for the wheel-legged robot."""

from dataclasses import dataclass
import signal
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


@dataclass
class DetectionResult:
    valid: bool
    error: float
    confidence: float
    reason: str
    debug_frame: np.ndarray
    lane_width_pixels: float = 0.0


class DualWhiteLineDetector:
    """Find paired white boundaries and follow their midpoint."""

    def __init__(
        self,
        search_top_ratio: float,
        search_bottom_ratio: float,
        search_left_ratio: float,
        search_right_ratio: float,
        white_saturation_max: int,
        white_value_min: int,
        adaptive_bright_enabled: bool,
        adaptive_block_size: int,
        adaptive_c: float,
        min_boundary_lane_contrast: float,
        morphology_kernel: int,
        min_line_width_ratio: float,
        max_line_width_ratio: float,
        expected_line_width_ratio: float,
        width_tolerance_ratio: float,
        min_lane_width_ratio: float,
        max_lane_width_ratio: float,
        expected_lane_width_ratio: float,
        lane_width_tolerance_ratio: float,
        max_center_jump_ratio: float,
        min_valid_row_fraction: float,
        min_confidence: float,
        target_offset_pixels: float,
    ) -> None:
        self.search_top_ratio = search_top_ratio
        self.search_bottom_ratio = search_bottom_ratio
        self.search_left_ratio = search_left_ratio
        self.search_right_ratio = search_right_ratio
        self.white_saturation_max = white_saturation_max
        self.white_value_min = white_value_min
        self.adaptive_bright_enabled = adaptive_bright_enabled
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.min_boundary_lane_contrast = min_boundary_lane_contrast
        self.morphology_kernel = morphology_kernel
        self.min_line_width_ratio = min_line_width_ratio
        self.max_line_width_ratio = max_line_width_ratio
        self.expected_line_width_ratio = expected_line_width_ratio
        self.width_tolerance_ratio = width_tolerance_ratio
        self.min_lane_width_ratio = min_lane_width_ratio
        self.max_lane_width_ratio = max_lane_width_ratio
        self.expected_lane_width_ratio = expected_lane_width_ratio
        self.lane_width_tolerance_ratio = lane_width_tolerance_ratio
        self.max_center_jump_ratio = max_center_jump_ratio
        self.min_valid_row_fraction = min_valid_row_fraction
        self.min_confidence = min_confidence
        self.target_offset_pixels = target_offset_pixels

    @staticmethod
    def _runs(row: np.ndarray) -> List[Tuple[int, int]]:
        padded = np.pad((row > 0).astype(np.int8), (1, 1))
        changes = np.diff(padded)
        return list(
            zip(
                np.flatnonzero(changes == 1).tolist(),
                np.flatnonzero(changes == -1).tolist(),
            )
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        debug = frame.copy()
        height, width = frame.shape[:2]
        x0 = int(width * self.search_left_ratio)
        x1 = int(width * self.search_right_ratio)
        y0 = int(height * self.search_top_ratio)
        y1 = int(height * self.search_bottom_ratio)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return DetectionResult(False, 0.0, 0.0, "empty_roi", debug)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        fixed_mask = cv2.inRange(
            hsv,
            np.array((0, 0, self.white_value_min), dtype=np.uint8),
            np.array((180, self.white_saturation_max, 255), dtype=np.uint8),
        )
        if self.adaptive_bright_enabled:
            adaptive_mask = cv2.adaptiveThreshold(
                hsv[:, :, 2],
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.adaptive_block_size,
                self.adaptive_c,
            )
            low_saturation = cv2.inRange(
                hsv[:, :, 1], 0, self.white_saturation_max
            )
            adaptive_mask = cv2.bitwise_and(adaptive_mask, low_saturation)
            # The floor in the real scene is also bright and low-saturation.
            # Requiring both absolute whiteness and local contrast prevents a
            # large uniform floor area from becoming one white candidate.
            mask = cv2.bitwise_and(fixed_mask, adaptive_mask)
        else:
            mask = fixed_mask

        if self.morphology_kernel > 1:
            kernel = np.ones(
                (self.morphology_kernel, self.morphology_kernel), dtype=np.uint8
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        roi_height, roi_width = mask.shape
        min_line_width = max(2, int(roi_width * self.min_line_width_ratio))
        max_line_width = max(
            min_line_width + 1, int(roi_width * self.max_line_width_ratio)
        )
        expected_line_width = roi_width * self.expected_line_width_ratio
        line_tolerance = max(1.0, roi_width * self.width_tolerance_ratio)
        min_lane_width = roi_width * self.min_lane_width_ratio
        max_lane_width = roi_width * self.max_lane_width_ratio
        expected_lane_width = roi_width * self.expected_lane_width_ratio
        lane_tolerance = max(1.0, roi_width * self.lane_width_tolerance_ratio)
        max_jump_per_row = max(1.0, roi_width * self.max_center_jump_ratio)
        target_x = 0.5 * width + self.target_offset_pixels

        left_mask = np.zeros_like(mask)
        right_mask = np.zeros_like(mask)
        lane_centers: List[float] = []
        lane_center_points: List[Tuple[float, int]] = []
        lane_widths: List[float] = []
        boundary_widths: List[float] = []
        boundary_lane_contrasts: List[float] = []
        previous_center: Optional[float] = None
        previous_row: Optional[int] = None
        rows_with_one_candidate = 0

        for row_y in range(roi_height - 1, -1, -1):
            runs = []
            for start, end in self._runs(mask[row_y]):
                run_width = end - start
                if min_line_width <= run_width <= max_line_width:
                    runs.append(
                        (start, end, x0 + 0.5 * (start + end - 1), run_width)
                    )
            if len(runs) == 1:
                rows_with_one_candidate += 1
            if len(runs) < 2:
                continue

            pairs = []
            for left_index in range(len(runs) - 1):
                for right_index in range(left_index + 1, len(runs)):
                    left = runs[left_index]
                    right = runs[right_index]
                    lane_width = right[2] - left[2]
                    corridor = hsv[row_y, left[1] : right[0], 2]
                    if corridor.size == 0:
                        continue
                    boundary_value = 0.5 * (
                        float(np.mean(hsv[row_y, left[0] : left[1], 2]))
                        + float(np.mean(hsv[row_y, right[0] : right[1], 2]))
                    )
                    lane_value = float(np.median(corridor))
                    boundary_lane_contrast = boundary_value - lane_value
                    if (
                        min_lane_width <= lane_width <= max_lane_width
                        and boundary_lane_contrast
                        >= self.min_boundary_lane_contrast
                    ):
                        pairs.append(
                            (
                                left,
                                right,
                                0.5 * (left[2] + right[2]),
                                lane_width,
                                boundary_lane_contrast,
                            )
                        )
            if not pairs:
                continue

            reference = previous_center if previous_center is not None else target_x
            row_gap = 1 if previous_row is None else max(1, previous_row - row_y)
            allowed_jump = min(0.30 * roi_width, max_jump_per_row * row_gap)

            def pair_score(pair) -> float:
                left, right, center, lane_width, boundary_lane_contrast = pair
                lane_error = abs(lane_width - expected_lane_width) / lane_tolerance
                line_error = (
                    abs(left[3] - expected_line_width)
                    + abs(right[3] - expected_line_width)
                ) / (2.0 * line_tolerance)
                center_error = abs(center - reference) / allowed_jump
                contrast_reward = min(
                    1.0,
                    boundary_lane_contrast
                    / max(1.0, 3.0 * self.min_boundary_lane_contrast),
                )
                return (
                    lane_error
                    + 0.35 * line_error
                    + 0.40 * center_error
                    - 0.15 * contrast_reward
                )

            compatible = [
                pair
                for pair in pairs
                if previous_center is None or abs(pair[2] - previous_center) <= allowed_jump
            ]
            if not compatible:
                continue
            left, right, center, lane_width, contrast = min(
                compatible, key=pair_score
            )
            left_mask[row_y, left[0] : left[1]] = 255
            right_mask[row_y, right[0] : right[1]] = 255
            lane_centers.append(center)
            lane_center_points.append((center, row_y))
            lane_widths.append(lane_width)
            boundary_widths.extend((left[3], right[3]))
            boundary_lane_contrasts.append(contrast)
            previous_center = center
            previous_row = row_y

        red = np.zeros_like(roi)
        red[:, :, 2] = mask
        debug[y0:y1, x0:x1] = cv2.addWeighted(
            debug[y0:y1, x0:x1], 0.84, red, 0.16, 0.0
        )
        selected = np.zeros_like(roi)
        selected[:, :, 1] = left_mask
        selected[:, :, 0] = right_mask
        debug[y0:y1, x0:x1] = cv2.addWeighted(
            debug[y0:y1, x0:x1], 0.76, selected, 0.42, 0.0
        )
        cv2.rectangle(debug, (x0, y0), (x1, y1), (255, 180, 0), 2)
        cv2.line(
            debug, (int(target_x), y0), (int(target_x), y1), (255, 255, 0), 1
        )

        pair_rows = len(lane_centers)
        mean_lane_width = float(np.mean(lane_widths)) if lane_widths else 0.0
        mean_contrast = (
            float(np.mean(boundary_lane_contrasts))
            if boundary_lane_contrasts
            else 0.0
        )
        cv2.putText(
            debug,
            f"WHITE S<={self.white_saturation_max} V>={self.white_value_min} "
            f"pairs={pair_rows}/{roi_height} lane={mean_lane_width:.1f}px "
            f"dV={mean_contrast:.1f}",
            (max(8, x0), max(24, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if not lane_centers:
            reason = "only_one_white_line" if rows_with_one_candidate > 0 else "no_white_lines"
            return DetectionResult(False, 0.0, 0.0, reason, debug)

        coverage = pair_rows / roi_height
        line_quality = float(
            np.mean(
                [
                    max(0.0, 1.0 - abs(value - expected_line_width) / line_tolerance)
                    for value in boundary_widths
                ]
            )
        )
        lane_quality = float(
            np.mean(
                [
                    max(0.0, 1.0 - abs(value - expected_lane_width) / lane_tolerance)
                    for value in lane_widths
                ]
            )
        )
        if len(lane_centers) > 1:
            continuity = float(
                np.mean(
                    np.maximum(
                        0.0,
                        1.0 - np.abs(np.diff(lane_centers)) / max_jump_per_row,
                    )
                )
            )
        else:
            continuity = 0.0
        confidence = (
            0.45 * coverage
            + 0.20 * line_quality
            + 0.20 * lane_quality
            + 0.15 * continuity
        )
        lane_center = float(np.mean(lane_centers))
        error = float(
            np.clip((lane_center - target_x) / (0.5 * width), -1.0, 1.0)
        )

        for center, row_y in lane_center_points:
            cv2.circle(debug, (int(center), y0 + row_y), 2, (255, 0, 255), -1)
        cv2.circle(
            debug,
            (int(lane_center), y0 + roi_height // 2),
            9,
            (255, 0, 255),
            -1,
        )

        if coverage < self.min_valid_row_fraction:
            return DetectionResult(
                False,
                error,
                confidence,
                "insufficient_double_line",
                debug,
                mean_lane_width,
            )
        if confidence < self.min_confidence:
            return DetectionResult(
                False, error, confidence, "low_confidence", debug, mean_lane_width
            )
        return DetectionResult(
            True, error, confidence, "tracking", debug, mean_lane_width
        )


class DoubleLineFollower(Node):
    VALID_ROTATIONS = (0, 90, 180, 270)
    VALID_MIRRORS = ("none", "horizontal", "vertical", "both")

    def __init__(self) -> None:
        super().__init__("double_line_follower")
        self._declare_parameters()
        self._load_parameters()
        self._validate_parameters()

        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self._frame_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._stop_capture = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._frame_time = 0.0
        self._sequence = 0
        self._camera_ok = False
        self._capture: Optional[cv2.VideoCapture] = None
        self._last_sequence = -1
        self._last_command = (0.0, 0.0)
        self._last_command_valid = False
        self._last_motion_time = time.monotonic()
        self._previous_error = 0.0
        self._previous_control_time = 0.0
        self._have_error = False
        self._last_log = 0.0
        self._exit_requested = False
        self._closing = False
        self._window_ready = False

        self.detector = DualWhiteLineDetector(
            self.search_top_ratio,
            self.search_bottom_ratio,
            self.search_left_ratio,
            self.search_right_ratio,
            self.white_saturation_max,
            self.white_value_min,
            self.adaptive_bright_enabled,
            self.adaptive_block_size,
            self.adaptive_c,
            self.min_boundary_lane_contrast,
            self.morphology_kernel,
            self.min_line_width_ratio,
            self.max_line_width_ratio,
            self.expected_line_width_ratio,
            self.width_tolerance_ratio,
            self.min_lane_width_ratio,
            self.max_lane_width_ratio,
            self.expected_lane_width_ratio,
            self.lane_width_tolerance_ratio,
            self.max_center_jump_ratio,
            self.min_valid_row_fraction,
            self.min_confidence,
            self.target_offset_pixels,
        )

        if self.show_debug:
            try:
                cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
                self._window_ready = True
            except cv2.error as error:
                self.get_logger().error(f"调试窗口创建失败: {error}")

        self._publish_zero("startup")
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="double_line_camera", daemon=True
        )
        self._capture_thread.start()
        self._timer = self.create_timer(1.0 / self.control_rate, self._control_timer)
        mode = "运动已启用" if self.enable_motion else "DRY-RUN"
        self.get_logger().info(
            f"双白线巡线启动: {mode}, {self.control_rate:.1f} Hz；只发布 /cmd_vel"
        )
        self.get_logger().info(
            f"物理前进方向映射: linear.x = "
            f"{self.linear_direction_sign:+.0f} × 前进速度"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "enable_motion": False,
            "camera_backend": "v4l2",
            "camera_id": 0,
            "camera_device": "/dev/video0",
            "image_width": 640,
            "image_height": 480,
            "capture_frame_rate": 30.0,
            "pixel_format": "YUYV",
            "rotation_degrees": 0,
            "mirror_mode": "none",
            "camera_retry_seconds": 1.0,
            "image_timeout_seconds": 0.35,
            "control_rate": 15.0,
            "show_debug": True,
            "debug_window_name": "Dual White Line Follower (Q/Esc to stop)",
            "search_top_ratio": 0.65,
            "search_bottom_ratio": 0.86,
            "search_left_ratio": 0.02,
            "search_right_ratio": 0.98,
            "white_saturation_max": 75,
            "white_value_min": 120,
            "adaptive_bright_enabled": True,
            "adaptive_block_size": 51,
            "adaptive_c": -12.0,
            "min_boundary_lane_contrast": 8.0,
            "morphology_kernel": 3,
            "min_line_width_ratio": 0.008,
            "max_line_width_ratio": 0.10,
            "expected_line_width_ratio": 0.025,
            "width_tolerance_ratio": 0.035,
            "min_lane_width_ratio": 0.45,
            "max_lane_width_ratio": 0.85,
            "expected_lane_width_ratio": 0.64,
            "lane_width_tolerance_ratio": 0.22,
            "max_center_jump_ratio": 0.02,
            "min_valid_row_fraction": 0.35,
            "min_confidence": 0.52,
            "target_offset_pixels": 0.0,
            "forward_speed": 0.20,
            "minimum_forward_speed": 0.08,
            "linear_direction_sign": 1.0,
            "steering_kp": 1.00,
            "steering_kd": 0.025,
            "steering_sign": -1.0,
            "max_angular_speed": 0.40,
            "max_linear_acceleration": 0.30,
            "max_angular_acceleration": 1.20,
            "max_error_derivative": 3.0,
            "error_filter_alpha": 0.45,
            "curve_slowdown": 0.60,
            "status_log_period": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_parameters(self) -> None:
        for parameter in self._parameters.values():
            setattr(self, parameter.name, self.get_parameter(parameter.name).value)
        integer_names = (
            "camera_id",
            "image_width",
            "image_height",
            "rotation_degrees",
            "white_saturation_max",
            "white_value_min",
            "adaptive_block_size",
            "morphology_kernel",
        )
        bool_names = ("enable_motion", "show_debug", "adaptive_bright_enabled")
        string_names = (
            "camera_backend",
            "camera_device",
            "pixel_format",
            "mirror_mode",
            "debug_window_name",
        )
        for name in integer_names:
            setattr(self, name, int(getattr(self, name)))
        for name in bool_names:
            setattr(self, name, bool(getattr(self, name)))
        for name in string_names:
            setattr(self, name, str(getattr(self, name)))
        self.camera_backend = self.camera_backend.lower()
        self.pixel_format = self.pixel_format.upper()
        self.mirror_mode = self.mirror_mode.lower()
        for name in self._parameters:
            if name not in integer_names + bool_names + string_names:
                setattr(self, name, float(getattr(self, name)))

    def _validate_parameters(self) -> None:
        if self.camera_backend not in ("auto", "v4l2"):
            raise ValueError("camera_backend 只能是 auto 或 v4l2")
        if self.camera_id < 0 or self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("摄像头编号和图像尺寸无效")
        if self.capture_frame_rate <= 0.0:
            raise ValueError("capture_frame_rate 必须大于 0")
        if self.rotation_degrees not in self.VALID_ROTATIONS:
            raise ValueError("rotation_degrees 必须是 0/90/180/270")
        if self.mirror_mode not in self.VALID_MIRRORS:
            raise ValueError("mirror_mode 无效")
        if len(self.pixel_format) != 4:
            raise ValueError("pixel_format 必须是四字符格式")
        if not 10.0 <= self.control_rate <= 20.0:
            raise ValueError("control_rate 必须在 10~20 Hz")
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size 必须是大于等于 3 的奇数")
        if not 0 <= self.white_saturation_max <= 255:
            raise ValueError("white_saturation_max 无效")
        if not 0 <= self.white_value_min <= 255:
            raise ValueError("white_value_min 无效")
        if not 0.0 < self.min_boundary_lane_contrast <= 255.0:
            raise ValueError("min_boundary_lane_contrast 必须在 (0,255]")
        ratios = (
            "min_line_width_ratio",
            "max_line_width_ratio",
            "expected_line_width_ratio",
            "width_tolerance_ratio",
            "min_lane_width_ratio",
            "max_lane_width_ratio",
            "expected_lane_width_ratio",
            "lane_width_tolerance_ratio",
            "max_center_jump_ratio",
            "min_valid_row_fraction",
            "min_confidence",
            "error_filter_alpha",
            "curve_slowdown",
        )
        for name in ratios:
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} 必须在 (0,1]")
        if not 0.0 <= self.search_top_ratio < self.search_bottom_ratio <= 1.0:
            raise ValueError("搜索区域上下比例无效")
        if not 0.0 <= self.search_left_ratio < self.search_right_ratio <= 1.0:
            raise ValueError("搜索区域左右比例无效")
        if not self.min_line_width_ratio < self.max_line_width_ratio:
            raise ValueError("白线宽度范围无效")
        if not (
            self.min_line_width_ratio
            <= self.expected_line_width_ratio
            <= self.max_line_width_ratio
        ):
            raise ValueError("expected_line_width_ratio 必须在线宽范围内")
        if not self.min_lane_width_ratio < self.max_lane_width_ratio:
            raise ValueError("双线间距范围无效")
        if not (
            self.min_lane_width_ratio
            <= self.expected_lane_width_ratio
            <= self.max_lane_width_ratio
        ):
            raise ValueError("expected_lane_width_ratio 必须在双线间距范围内")
        if self.linear_direction_sign not in (-1.0, 1.0):
            raise ValueError("linear_direction_sign 只能是 -1 或 1")
        if self.steering_sign not in (-1.0, 1.0):
            raise ValueError("steering_sign 只能是 -1 或 1")
        if not 0.0 <= self.minimum_forward_speed <= self.forward_speed:
            raise ValueError("前进速度范围无效")
        if (
            self.max_angular_speed <= 0.0
            or self.max_linear_acceleration <= 0.0
            or self.max_angular_acceleration <= 0.0
            or self.max_error_derivative <= 0.0
            or self.status_log_period <= 0.0
        ):
            raise ValueError("速度、加速度、误差变化率和日志周期必须大于 0")
        if self.steering_kp < 0.0 or self.steering_kd < 0.0:
            raise ValueError("转向增益不能为负")
        if self.camera_retry_seconds <= 0.0 or self.image_timeout_seconds <= 0.0:
            raise ValueError("摄像头重试和图像超时必须大于 0")

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        if self.camera_backend == "v4l2":
            capture = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
        else:
            capture = cv2.VideoCapture(self.camera_id)
        if not capture.isOpened():
            capture.release()
            return None
        if self.camera_backend == "v4l2":
            capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.pixel_format)
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        capture.set(cv2.CAP_PROP_FPS, self.capture_frame_rate)
        self.get_logger().info(
            f"摄像头: {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
            f"{capture.get(cv2.CAP_PROP_FPS):.1f} fps"
        )
        return capture

    def _capture_loop(self) -> None:
        while not self._stop_capture.is_set():
            with self._capture_lock:
                capture = self._capture
            if capture is None or not capture.isOpened():
                capture = self._open_camera()
                with self._capture_lock:
                    self._capture = capture
                if capture is None:
                    with self._frame_lock:
                        self._camera_ok = False
                        self._frame = None
                    self._publish_zero("camera_open_failed")
                    self._stop_capture.wait(self.camera_retry_seconds)
                    continue
            ok, frame = capture.read()
            if not ok or frame is None:
                with self._frame_lock:
                    self._camera_ok = False
                    self._frame = None
                self._publish_zero("camera_read_failed")
                capture.release()
                with self._capture_lock:
                    self._capture = None
                self._stop_capture.wait(self.camera_retry_seconds)
                continue
            frame = self._transform(frame)
            with self._frame_lock:
                self._frame = frame
                self._frame_time = time.monotonic()
                self._sequence += 1
                self._camera_ok = True

    def _transform(self, frame: np.ndarray) -> np.ndarray:
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

    def _publish(self, forward_speed: float, angular_speed: float) -> None:
        message = Twist()
        message.linear.x = self.linear_direction_sign * float(forward_speed)
        message.angular.z = float(angular_speed)
        self.publisher.publish(message)

    def _publish_zero(self, _reason: str) -> None:
        try:
            self._publish(0.0, 0.0)
        except Exception:
            pass
        self._last_command = (0.0, 0.0)
        self._last_command_valid = False
        self._last_motion_time = time.monotonic()
        self._have_error = False

    @staticmethod
    def _approach(current: float, target: float, delta: float) -> float:
        return float(np.clip(target, current - delta, current + delta))

    def _rate_limit(self, linear: float, angular: float, now: float):
        dt = max(0.0, min(now - self._last_motion_time, 2.0 / self.control_rate))
        linear = self._approach(
            self._last_command[0], linear, self.max_linear_acceleration * dt
        )
        angular = self._approach(
            self._last_command[1], angular, self.max_angular_acceleration * dt
        )
        self._last_motion_time = now
        return linear, angular

    def _control_timer(self) -> None:
        try:
            self._control_cycle()
        except Exception as error:
            self.get_logger().error(f"控制异常，立即停车: {error}")
            self._publish_zero("control_exception")

    def _control_cycle(self) -> None:
        now = time.monotonic()
        with self._frame_lock:
            ok = self._camera_ok
            frame_time = self._frame_time
            sequence = self._sequence
            frame = self._frame.copy() if self._frame is not None else None
        if not ok or frame is None:
            self._publish_zero("camera_unavailable")
            self._log(now, "STOP camera unavailable")
            return
        if now - frame_time > self.image_timeout_seconds:
            self._publish_zero("image_timeout")
            self._log(now, "STOP image timeout")
            return
        if sequence == self._last_sequence:
            if self.enable_motion and self._last_command_valid:
                self._publish(*self._last_command)
            else:
                self._publish_zero("no_new_frame")
            return
        self._last_sequence = sequence

        result = self.detector.detect(frame)
        if not result.valid:
            self._publish_zero(result.reason)
            self._show(result, 0.0, 0.0, False)
            self._log(now, f"STOP {result.reason} conf={result.confidence:.2f}")
            return

        if self._have_error:
            filtered = self.error_filter_alpha * result.error + (
                1.0 - self.error_filter_alpha
            ) * self._previous_error
            dt = max(1e-3, now - self._previous_control_time)
            derivative = float(
                np.clip(
                    (filtered - self._previous_error) / dt,
                    -self.max_error_derivative,
                    self.max_error_derivative,
                )
            )
        else:
            filtered = result.error
            derivative = 0.0
            self._have_error = True
        self._previous_error = filtered
        self._previous_control_time = now
        angular = self.steering_sign * (
            self.steering_kp * filtered + self.steering_kd * derivative
        )
        angular = float(np.clip(angular, -self.max_angular_speed, self.max_angular_speed))
        turn_ratio = min(1.0, abs(angular) / self.max_angular_speed)
        linear = max(
            self.minimum_forward_speed,
            self.forward_speed * (1.0 - self.curve_slowdown * turn_ratio),
        )
        target = (linear, angular)
        if self.enable_motion:
            command = self._rate_limit(*target, now)
            self._publish(*command)
            self._last_command = command
            self._last_command_valid = True
        else:
            self._publish_zero("motion_disabled")
            command = target
        self._show(result, *command, self.enable_motion)
        mode = "RUN" if self.enable_motion else "DRY-RUN"
        self._log(
            now,
            f"{mode} conf={result.confidence:.2f} error={result.error:+.3f} "
            f"v={command[0]:.3f} w={command[1]:+.3f}",
        )

    def _show(
        self,
        result: DetectionResult,
        linear: float,
        angular: float,
        command_active: bool,
    ) -> None:
        if not self._window_ready:
            return
        if command_active:
            state, color, label = "RUN", (0, 255, 0), "cmd"
        elif self.enable_motion:
            state, color, label = "ARMED / STOP", (0, 165, 255), "cmd"
        else:
            state, color, label = "STOP / DRY-RUN", (0, 0, 255), "planned"
        debug = result.debug_frame
        cv2.putText(
            debug,
            f"{state} {result.reason} conf={result.confidence:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            f"error={result.error:+.3f} {label} v={linear:.3f} w={angular:+.3f}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.debug_window_name, debug)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self.request_exit("operator_exit")

    def _log(self, now: float, text: str) -> None:
        if now - self._last_log >= self.status_log_period:
            self.get_logger().info(text)
            self._last_log = now

    def request_exit(self, reason: str) -> None:
        self._publish_zero(reason)
        self._exit_requested = True

    @property
    def exit_requested(self) -> bool:
        return self._exit_requested

    def close(self) -> None:
        self._closing = True
        self._stop_capture.set()
        with self._capture_lock:
            capture = self._capture
        if capture is not None:
            capture.release()
        if hasattr(self, "_capture_thread"):
            self._capture_thread.join(timeout=1.0)
        if rclpy.ok():
            for _ in range(10):
                self._publish(0.0, 0.0)
                time.sleep(0.02)
        if self._window_ready:
            cv2.destroyWindow(self.debug_window_name)
        self.get_logger().info("双白线节点退出，已重复发布零速度")


def main(args=None) -> None:
    node = None
    exit_event = threading.Event()

    def signal_handler(signum, _frame) -> None:
        exit_event.set()
        if node is not None:
            node.request_exit(f"signal_{signum}")

    previous_sigint = signal.signal(signal.SIGINT, signal_handler)
    previous_sigterm = signal.signal(signal.SIGTERM, signal_handler)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = DoubleLineFollower()
        while rclpy.ok() and not node.exit_requested and not exit_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
