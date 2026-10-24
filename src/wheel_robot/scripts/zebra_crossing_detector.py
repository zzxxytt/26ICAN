#!/usr/bin/env python3

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

#重命名numpy图像
'''
OpenCV BGR图像
shape = (height, width, 3)
dtype = uint8
'''
BgrImage = NDArray[np.uint8]

'''
单通道二值图像：
0表示背景,255表示目标
shape = (height, width)
dtype = uint8
'''
BinaryImage = NDArray[np.uint8]

'''
OpenCV轮廓或四边形点集
常见shape为 (N, 1, 2) 或 (4, 2)
'''
IntPoints = NDArray[np.int32]

#######################################################################数据结构类型
class DetectionReason(str,Enum):
    """
    单帧检测结果或失败原因。
    """
    VALID = "valid"                             # 检测成功，当前帧满足斑马线几何条件。
    EMPTY_FRAME = "empty_frame"                 # 输入图像为空或尺寸无效。
    EMPTY_ROI = "empty_roi"                     # 配置的ROI超出图像范围，或裁剪后为空。
    TRACK_NOT_FOUND = "track_not_found"         # 没有找到可信的白色赛道区域。
    TRACK_TOO_SMALL = "track_too_small"         # 找到了白色区域，但面积或宽度不足，不能认为是赛道。
    NO_DARK_REGION = "no_dark_region"           # 赛道内部没有足够黑色像素。
    TOO_FEW_STRIPES = "too_few_stripes"         # 找到的横向黑条数量少于要求，例如只检测到一条接缝。
    TOO_MANY_STRIPES = "too_many_stripes"       # 黑色区域数量过多。
    STRIPE_TOO_NARROW = "stripe_too_narrow"     # 黑条没有横跨足够大的赛道宽度。
    STRIPE_TOO_WIDE = "stripe_too_wide"         # 黑色区域太宽或连成大块。
    IRREGULAR_THICKNESS = "irregular_thickness" # 黑条厚度差异过大。
    IRREGULAR_SPACING = "irregular_spacing"     # 相邻黑条间距差异过大。
    INVALID_ANGLE = "invalid_angle"             # 黑条方向不是近似横向。
    LOW_CONTRAST = "low_contrast"               # 黑条与周围白色赛道的亮度对比不足。
    LOW_CONFIDENCE = "low_confidence"           # 所有基本条件部分满足，但综合置信度不足。
    
class ZebraState(str, Enum):
    """斑马线多帧识别状态。"""

    SEARCH = "SEARCH"               # 当前没有检测到斑马线。
    CANDIDATE = "CANDIDATE"         # 单帧或少数帧发现斑马线，还没有达到确认条件。
    CONFIRMED = "CONFIRMED"         # 最近一段时间内多帧检测成功，已确认前方存在斑马线。
    APPROACHING = "APPROACHING"     # 斑马线在画面中持续向上移动，机器人正在靠近。
    ON_CROSSING = "ON_CROSSING"     # 最近斑马条已经到达画面底部附近。
    PASSED = "PASSED"               # 之前确认过斑马线，近处条纹随后从画面底部消失，判定已经通过。
    COOLDOWN = "COOLDOWN"           # 通过后进入冷却，防止同一斑马线被连续触发多次。

class ZebraEvent(str, Enum):
    """状态机产生的瞬时事件。"""

    NONE = "NONE"                   # 无事件触发。
    CONFIRMED = "CONFIRMED"         # 第一次完成多帧确认。
    ENTERED = "ENTERED"             # 进入斑马线近距离区域。
    PASSED = "PASSED"               # 已经通过斑马线。

@dataclass(frozen=True)
class RoiConfig:
    """原始图像中的斑马线搜索区域。"""

    top_ratio: float = 0.70      # ROI上边缘，不能设置得过低，否则远处斑马线发现太晚。
    bottom_ratio: float = 1.00    # ROI下边缘。保留少量底部区域可避免机器人自身结构进入画面。
    left_ratio: float = 0.15
    right_ratio: float = 0.85

NormalizedPoint = Tuple[float, float]


@dataclass(frozen=True)
class PerspectiveConfig:
    """将相机透视图转换成鸟瞰图。

    source_points中的坐标是相对于ROI宽高的归一化坐标。
    点的顺序统一为：
        左上、右上、右下、左下
    """

    enabled: bool = True

    source_points: Tuple[
        NormalizedPoint,
        NormalizedPoint,
        NormalizedPoint,
        NormalizedPoint,
    ] = (
        (0.30, 0.05),  # 左上
        (0.70, 0.05),  # 右上
        (0.95, 0.98),  # 右下
        (0.05, 0.98),  # 左下
    )

    # 输出鸟瞰图大小。
    output_width: int = 400
    output_height: int = 250

@dataclass
class StripeCountResult:
    raw_contour_count: int
    stripe_count: int
    zebra_exists: bool
    dominant_angle_deg: float
    debug_image: BgrImage
    layout_ok: bool = False
    layout_reason: str = ""
    axis_spread_deg: float = 0.0
    angle_std_deg: float = 0.0
    band_width_ratio: float = 0.0
    spacing_cv: float = 0.0
    curvature_gain: float = 0.0

@dataclass(frozen=True)
class StripeLayoutConfig:
    min_aspect_ratio: float = 1.4
    max_aspect_ratio: float = 8.0
    min_long_side_ratio: float = 0.18
    min_short_side_ratio: float = 0.012
    max_vertical_deviation_deg: float = 30.0
    max_angle_std_deg: float = 9.0
    # 斑马线条纹并排：长轴与“中心连线方向”接近垂直；圆环虚线串联：接近平行
    min_axis_spread_deg: float = 55.0
    min_band_width_ratio: float = 0.30
    max_spacing_cv: float = 0.60
    max_curvature_gain: float = 0.45
    max_line_rms_ratio: float = 0.08


@dataclass
class StripeLayoutResult:
    ok: bool
    reason: str
    axis_spread_deg: float = 0.0
    angle_std_deg: float = 0.0
    band_width_ratio: float = 0.0
    spacing_cv: float = 0.0
    curvature_gain: float = 0.0

###################################################################函数

def _axial_mean_deg(angles):
    z = np.mean(np.exp(1j * np.deg2rad(2.0 * np.asarray(angles, dtype=np.float64))))
    return float(0.5 * np.rad2deg(np.angle(z)))


def _axial_diff_deg(a, b):
    return abs(((a - b + 90.0) % 180.0) - 90.0)


def _axial_std_deg(angles):
    z = np.mean(np.exp(1j * np.deg2rad(2.0 * np.asarray(angles, dtype=np.float64))))
    r = min(1.0, max(1e-9, abs(z)))
    return float(0.5 * np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(r)))))


def analyze_stripe_layout(metas, width, height, cfg):
    """metas: [{'cx','cy','long','short','angle'}, ...]"""
    n = len(metas)
    if n < 3:
        return StripeLayoutResult(False, "too_few_stripes")
    centers = np.array([[m["cx"], m["cy"]] for m in metas], dtype=np.float64)
    angles = np.array([m["angle"] for m in metas], dtype=np.float64)
    mean = centers.mean(axis=0)
    shifted = centers - mean
    cov = np.cov(shifted.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    spread = float(np.rad2deg(np.arctan2(v[1], v[0])))
    spread = ((spread + 90.0) % 180.0) - 90.0
    axis = _axial_mean_deg(angles)
    axis_spread = _axial_diff_deg(axis, spread)
    angle_std = _axial_std_deg(angles)

    t = shifted @ v
    order = np.argsort(t)
    t_sorted = t[order]
    band = float(t_sorted[-1] - t_sorted[0])
    band_ratio = band / max(1.0, float(width))
    spacing = np.diff(t_sorted)
    spacing_cv = (
        float(np.std(spacing) / max(1e-6, np.mean(spacing)))
        if spacing.size > 1 else 0.0
    )

    normal = np.array([-v[1], v[0]])
    u = shifted @ v
    w = shifted @ normal
    c1 = np.polyfit(u, w, 1)
    rms1 = float(np.sqrt(np.mean((w - np.polyval(c1, u)) ** 2)))
    if n >= 3:
        c2 = np.polyfit(u, w, 2)
        rms2 = float(np.sqrt(np.mean((w - np.polyval(c2, u)) ** 2)))
    else:
        rms2 = rms1
    curvature_gain = (rms1 - rms2) / max(rms1, 1e-6)
    line_rms_ratio = rms1 / max(band, 1e-6)

    if angle_std > cfg.max_angle_std_deg:
        return StripeLayoutResult(False, "angle_not_parallel", axis_spread,
                                  angle_std, band_ratio, spacing_cv,
                                  curvature_gain)
    if axis_spread < cfg.min_axis_spread_deg:
        return StripeLayoutResult(False, "serial_not_side_by_side", axis_spread,
                                  angle_std, band_ratio, spacing_cv,
                                  curvature_gain)
    if band_ratio < cfg.min_band_width_ratio:
        return StripeLayoutResult(False, "band_too_narrow", axis_spread,
                                  angle_std, band_ratio, spacing_cv,
                                  curvature_gain)
    if spacing_cv > cfg.max_spacing_cv:
        return StripeLayoutResult(False, "spacing_irregular", axis_spread,
                                  angle_std, band_ratio, spacing_cv,
                                  curvature_gain)
    if curvature_gain > cfg.max_curvature_gain:
        return StripeLayoutResult(False, "curved_chain", axis_spread,
                                  angle_std, band_ratio, spacing_cv,
                                  curvature_gain)
    if line_rms_ratio > cfg.max_line_rms_ratio:
        return StripeLayoutResult(False, "centers_not_collinear", axis_spread,
                                  angle_std, band_ratio, spacing_cv,
                                  curvature_gain)
    return StripeLayoutResult(True, "zebra_layout", axis_spread, angle_std,
                              band_ratio, spacing_cv, curvature_gain)

def open_camera(
    device: str = "/dev/video0",
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> cv2.VideoCapture:
    """打开车载摄像头并设置MJPG、分辨率和帧率。"""

    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开摄像头 {device}")

    # V4L2下先设置格式，再设置分辨率和帧率。
    capture.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)

    print(
        f"摄像头已打开: {device}, "
        f"{actual_width}x{actual_height} @ {actual_fps:.1f} fps"
    )

    return capture

def extract_roi(
    frame: BgrImage,
    config: RoiConfig,
) -> Tuple[BgrImage, Tuple[int, int, int, int]]:
    """按照归一化比例裁剪ROI。

    返回：
        roi：裁剪后的图像
        rect：ROI在原图中的像素坐标(x0, y0, x1, y1)
    """

    height, width = frame.shape[:2]

    x0 = int(width * config.left_ratio)
    x1 = int(width * config.right_ratio)
    y0 = int(height * config.top_ratio)
    y1 = int(height * config.bottom_ratio)

    # 防止比例配置错误导致数组越界。
    x0 = max(0, min(x0, width - 1))
    x1 = max(x0 + 1, min(x1, width))
    y0 = max(0, min(y0, height - 1))
    y1 = max(y0 + 1, min(y1, height))

    roi = frame[y0:y1, x0:x1]

    if roi.size == 0:
        raise RuntimeError("ROI为空，请检查RoiConfig")

    return roi, (x0, y0, x1, y1)

def warp_perspective(
    roi: BgrImage,
    config: PerspectiveConfig,
) -> Tuple[BgrImage, IntPoints]:
    """将ROI中的梯形赛道变换为鸟瞰图。

    source_points使用相对于ROI宽高的归一化坐标，顺序必须是：
        左上、右上、右下、左下

    返回：
        bird_view：透视变换后的鸟瞰图
        source_pixels：ROI内的四个源点，供调试绘制
    """

    if not config.enabled:
        return roi.copy(), np.empty((0, 2), dtype=np.int32)

    roi_height, roi_width = roi.shape[:2]

    source_pixels = np.float32(
        [
            [x_ratio * roi_width, y_ratio * roi_height]
            for x_ratio, y_ratio in config.source_points
        ]
    )

    destination_pixels = np.float32(
        [
            [0, 0],
            [config.output_width - 1, 0],
            [config.output_width - 1, config.output_height - 1],
            [0, config.output_height - 1],
        ]
    )

    matrix = cv2.getPerspectiveTransform(
        source_pixels,
        destination_pixels,
    )

    bird_view = cv2.warpPerspective(
        roi,
        matrix,
        (config.output_width, config.output_height),
    )

    return bird_view, source_pixels.astype(np.int32)

def extract_white_track(
    bird_view: BgrImage,
    saturation_max: int = 70,
    value_min: int = 120,
) -> Tuple[BinaryImage, BgrImage]:
    """从鸟瞰图提取白色赛道。

    HSV中白色的基本特征：
        饱和度S较低
        亮度V较高

    返回：
        track_mask：二值赛道掩膜
        track_image：只保留白色赛道的彩色图像
    """

    hsv = cv2.cvtColor(bird_view, cv2.COLOR_BGR2HSV)

    lower_white = np.array(
        [0, 0, value_min],
        dtype=np.uint8,
    )
    upper_white = np.array(
        [180, saturation_max, 255],
        dtype=np.uint8,
    )

    white_mask = cv2.inRange(
        hsv,
        lower_white,
        upper_white,
    )

    # 开运算删除零散白点。
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    # 只用较小的闭运算连接白色区域中的小缺口。
    # 不能使用过大的核，否则会把黑色条纹填成白色。
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    track_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    track_image = cv2.bitwise_and(
        bird_view,
        bird_view,
        mask=track_mask,
    )

    return track_mask, track_image

def get_long_axis_angle(
    contour: IntPoints,
) -> Tuple[float, float, float]:
    """计算轮廓最小外接矩形长边的方向。

    返回：
        angle_deg：长边相对图像水平轴的角度
        long_side：矩形长边长度
        short_side：矩形短边长度
    """

    (_, _), (rect_width, rect_height), angle = cv2.minAreaRect(
        contour
    )

    if rect_width <= 0.0 or rect_height <= 0.0:
        return 0.0, 0.0, 0.0

    if rect_width >= rect_height:
        long_side = rect_width
        short_side = rect_height
        long_angle = angle
    else:
        long_side = rect_height
        short_side = rect_width
        long_angle = angle + 90.0

    # 规范到[-90°, 90°)。
    long_angle = (
        long_angle + 90.0
    ) % 180.0 - 90.0

    return (
        float(long_angle),
        float(long_side),
        float(short_side),
    )

def count_black_stripes(
    track_mask: BinaryImage,
    minimum_stripe_count: int = 3,
    maximum_stripe_count: int = 10,
    config: Optional[StripeLayoutConfig] = None,
) -> StripeCountResult:
    cfg = config if config is not None else StripeLayoutConfig()
    height, width = track_mask.shape
    image_area = float(height * width)

    black_mask = cv2.inRange(track_mask, 0, 127)
    black_mask = cv2.morphologyEx(
        black_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    black_mask = cv2.morphologyEx(
        black_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
    )
    contours, _ = cv2.findContours(
        black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    raw_contour_count = len(contours)

    valid_rectangles = []
    valid_angles = []
    metas = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.001 * image_area or area > 0.20 * image_area:
            continue
        angle, long_side, short_side = get_long_axis_angle(contour)
        if short_side <= 1.0:
            continue
        aspect = long_side / short_side
        if aspect < cfg.min_aspect_ratio or aspect > cfg.max_aspect_ratio:
            continue
        if long_side < cfg.min_long_side_ratio * height:
            continue
        if short_side < cfg.min_short_side_ratio * width:
            continue
        vertical_deviation = abs(abs(angle) - 90.0)
        if vertical_deviation > cfg.max_vertical_deviation_deg:
            continue
        rectangle = cv2.minAreaRect(contour)
        box = np.int32(np.round(cv2.boxPoints(rectangle)))
        valid_rectangles.append(box)
        valid_angles.append(angle)
        metas.append({
            "cx": float(np.mean(box[:, 0])),
            "cy": float(np.mean(box[:, 1])),
            "long": float(long_side),
            "short": float(short_side),
            "angle": float(angle),
        })

    stripe_count = len(valid_rectangles)
    dominant_angle = 0.0
    parallel = True
    if valid_angles:
        angles_array = np.asarray(valid_angles, dtype=np.float32)
        doubled = np.deg2rad(angles_array * 2.0)
        dominant_angle = 0.5 * np.rad2deg(
            np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled)))
        )
        differences = np.abs(
            (angles_array - dominant_angle + 90.0) % 180.0 - 90.0
        )
        parallel = bool(np.max(differences) <= 15.0)

    layout = analyze_stripe_layout(metas, width, height, cfg)
    zebra_exists = (
        minimum_stripe_count <= stripe_count <= maximum_stripe_count
        and parallel
        and layout.ok
    )

    debug_image = cv2.cvtColor(track_mask, cv2.COLOR_GRAY2BGR)
    for index, box in enumerate(valid_rectangles):
        cv2.polylines(debug_image, [box], True, (0, 0, 255), 2)
        center_x = int(np.mean(box[:, 0]))
        center_y = int(np.mean(box[:, 1]))
        cv2.putText(debug_image, str(index + 1), (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(debug_image, (0, 0), (width - 1, 84), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        debug_image, "RAW: %d  VALID: %d" % (raw_contour_count, stripe_count),
        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        debug_image,
        "ZEBRA: %s  ANGLE: %+.1f" % ("YES" if zebra_exists else "NO",
                                      dominant_angle),
        (8, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        debug_image,
        "layout=%s spread=%.0f std=%.0f band=%.2f cv=%.2f curve=%.2f" % (
            layout.reason, layout.axis_spread_deg, layout.angle_std_deg,
            layout.band_width_ratio, layout.spacing_cv, layout.curvature_gain,
        ),
        (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA,
    )
    return StripeCountResult(
        raw_contour_count=raw_contour_count,
        stripe_count=stripe_count,
        zebra_exists=bool(zebra_exists),
        dominant_angle_deg=float(dominant_angle),
        debug_image=debug_image,
        layout_ok=bool(layout.ok),
        layout_reason=layout.reason,
        axis_spread_deg=float(layout.axis_spread_deg),
        angle_std_deg=float(layout.angle_std_deg),
        band_width_ratio=float(layout.band_width_ratio),
        spacing_cv=float(layout.spacing_cv),
        curvature_gain=float(layout.curvature_gain),
    )

def _create_ros_publisher():
    """懒加载 rclpy 并创建结果发布器。

    没有 ROS 2 环境时返回全 None，脚本仍可当纯视觉程序跑。
    返回 (rclpy, node, publish_result)。
    """
    if os.environ.get("ZEBRA_ROS", "1") != "1":
        print("ZEBRA_ROS=0，不发布 ROS 话题")
        return None, None, None

    try:
        import rclpy
        from std_msgs.msg import Bool, UInt8
    except ImportError as error:
        print(f"没有 ROS 2 环境（{error}），只做本地显示，不发布话题")
        return None, None, None

    rclpy.init()
    node = rclpy.create_node("zebra_crossing_detector")
    detected_publisher = node.create_publisher(
        Bool, "/zebra_crossing/detected", 1
    )
    count_publisher = node.create_publisher(
        UInt8, "/zebra_crossing/stripe_count", 1
    )

    def publish_result(zebra_exists: bool, stripe_count: int) -> None:
        detected_publisher.publish(Bool(data=bool(zebra_exists)))
        count_publisher.publish(
            UInt8(data=min(int(stripe_count), 255))
        )

    node.get_logger().info(
        "发布 /zebra_crossing/detected (Bool) 和 "
        "/zebra_crossing/stripe_count (UInt8)"
    )
    return rclpy, node, publish_result


def main() -> None:
    roi_config = RoiConfig()
    perspective_config = PerspectiveConfig()

    capture = open_camera(
        device="/dev/video0",
        width=640,
        height=480,
        fps=30.0,
    )

    # 帧号从0开始，用于窗口底部显示和终端节流打印。
    frame_index = 0

    # ZEBRA_FULLSCREEN=1 时铺满整个屏幕，适合直接看车上的小屏。
    window_name = "Zebra Stripe Detection"
    fullscreen = os.environ.get("ZEBRA_FULLSCREEN", "0") == "1"

    if fullscreen:
        # 必须先建立 WINDOW_NORMAL 窗口，FULLSCREEN 属性才会生效；
        # 且只有 WINDOW_NORMAL 才会把图像缩放到窗口大小。
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    ros, node, publish_result = _create_ros_publisher()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("摄像头读取失败")
                continue
            frame_index += 1
            # 第一步：裁剪ROI。
            roi, (x0, y0, x1, y1) = extract_roi(
                frame,
                roi_config,
            )

            # 第二步：透视变换。
            bird_view, source_points = warp_perspective(
                roi,
                perspective_config,
            )

            # 第三步：提取白色赛道。
            track_mask, track_image = extract_white_track(
                bird_view,
                saturation_max=75,
                value_min=130,
            )

            stripe_result = count_black_stripes(
                track_mask,
                minimum_stripe_count=3,
                maximum_stripe_count=10,
            )
            
            debug_height = stripe_result.debug_image.shape[0]

            cv2.putText(
                stripe_result.debug_image,
                f"FRAME: {frame_index}",
                (8, debug_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

         
            # 在原始图像上画出ROI。
            camera_debug = frame.copy()
            cv2.rectangle(
                camera_debug,
                (x0, y0),
                (x1 - 1, y1 - 1),
                (255, 180, 0),
                2,
            )

            # 在ROI图上画出透视变换的四个源点。
            roi_debug = roi.copy()
            if source_points.size:
                cv2.polylines(
                    roi_debug,
                    [source_points],
                    isClosed=True,
                    color=(0, 255, 255),
                    thickness=2,
                )

                for index, point in enumerate(source_points):
                    x, y = point
                    cv2.circle(
                        roi_debug,
                        (int(x), int(y)),
                        5,
                        (0, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        roi_debug,
                        str(index),
                        (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # 这些窗口会通过MobaXterm X11显示。
            #cv2.imshow("01 Camera and ROI", camera_debug)
            #cv2.imshow("02 ROI and Perspective Points", roi_debug)
            #cv2.imshow("03 Bird View", bird_view)
            #cv2.imshow("05 Extracted White Track", track_image)

                       # 发布检测结果；没有 ROS 环境时跳过
            if publish_result is not None:
                publish_result(
                    stripe_result.zebra_exists,
                    stripe_result.stripe_count,
                )

            # 当前阶段只保留一个窗口，降低MobaXterm X11传输开销。
            cv2.imshow(window_name, stripe_result.debug_image)

            # 全屏只在第一帧设置一次：部分后端要求窗口已创建才生效。
            if fullscreen and frame_index == 1:
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )

            # 每15帧打印一次，避免SSH终端刷屏拖慢刷新。
            if frame_index % 15 == 0:
                print(
                    f"frame={frame_index}  "
                    f"原始轮廓={stripe_result.raw_contour_count}  "
                    f"有效条纹={stripe_result.stripe_count}  "
                    f"存在斑马线={stripe_result.zebra_exists}"
                )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()
        if node is not None:
            node.destroy_node()
            ros.shutdown()


if __name__ == "__main__":
    main()












































