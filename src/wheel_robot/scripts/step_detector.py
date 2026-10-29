#!/usr/bin/env python3
"""台阶检测：中心缺口框内连续无黑线的行数达到阈值即命中。

不要求近端/远端有线，不判断平台颜色、前缘或弯道。
StepTracker 仅提供可选的连续帧确认，默认一帧。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class StepConfig:
    # 巡线 ROI
    top_ratio: float = 0.56
    bottom_ratio: float = 0.98
    left_ratio: float = 0.02
    right_ratio: float = 0.98
    # 中心列带
    center_left_ratio: float = 0.30
    center_right_ratio: float = 0.70
    # 缺口搜索范围（ROI 内）
    gap_top_ratio: float = 0.00
    gap_bottom_ratio: float = 0.80
    # 黑线
    dark_value_max: int = 65
    dark_saturation_max: int = 255
    min_dark_pixels_per_row: int = 5
    # 以下旧颜色/近远线/边缘字段保留参数兼容性，不再参与判定。
    # 浅色
    bright_value_min: int = 110
    bright_saturation_max: int = 100
    # 平台表面：浅色 OR 木色，木色不要求纯白
    platform_value_min: int = 95
    platform_saturation_max: int = 150
    wood_hue_min: int = 5
    wood_hue_max: int = 35
    wood_saturation_min: int = 35
    # 近段
    near_bottom_ratio: float = 0.20
    near_min_line_rows: int = 5
    require_near_centered: bool = True
    near_center_tolerance_ratio: float = 0.18
    # 判定阈值
    min_gap_rows: int = 12
    gap_bright_min: float = 0.55
    gap_dark_max: float = 0.002
    gap_surface_min: float = 0.50
    require_far_line: bool = False
    # 弯道排除
    curve_min_rows: int = 8
    curve_max_dx_per_row: float = 0.10
    curve_min_dx: float = 0.18
    curve_min_support: float = 0.55
    curve_min_width_ratio: float = 0.006
    curve_max_width_ratio: float = 0.10
    # 平台前缘
    edge_gradient_min: int = 25
    edge_min_width_ratio: float = 0.45
    edge_row_tolerance: int = 7
    edge_max_offset_rows: int = 30
    require_edge: bool = True
    # 连续帧确认（默认一帧）；旧空间一致性参数不再参与判定。
    track_confirm_frames: int = 1
    track_max_frame_gap: float = 0.30
    track_edge_row_range: int = 8
    track_lateral_range: float = 0.10


@dataclass
class StepResult:
    detected: bool
    reason: str
    near_rows: int
    near_center_x: float
    gap_rows: int
    gap_bright_ratio: float
    gap_dark_ratio: float
    gap_start_row: int
    far_rows: int
    debug_image: BgrImage
    surface_ratio: float = 0.0
    wood_ratio: float = 0.0
    edge_row: int = -1
    edge_width_ratio: float = 0.0
    edge_strength: float = 0.0
    line_ends_at_edge: bool = False
    curve: bool = False
    curve_support: float = 0.0
    curve_dx: float = 0.0
    curve_side: int = 0
    gap_center_x: float = 0.5


def _find_longest_gap(row_has_line: BoolMask, start: int, end: int) -> Tuple[int, int]:
    best_start, best_len, run_start = start, 0, None
    for row in range(start, end):
        if not row_has_line[row]:
            if run_start is None:
                run_start = row
            run_len = row - run_start + 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_start = None
    return best_start, best_len


def detect_step(frame: BgrImage, config: StepConfig) -> StepResult:
    if frame is None or frame.size == 0:
        raise RuntimeError("空图像")
    if config.min_gap_rows < 1 or config.min_dark_pixels_per_row < 1:
        raise ValueError("min_gap_rows 和 min_dark_pixels_per_row 必须至少为 1")
    height, width = frame.shape[:2]
    x0 = max(0, min(int(width * config.left_ratio), width - 1))
    x1 = max(x0 + 1, min(int(width * config.right_ratio), width))
    y0 = max(0, min(int(height * config.top_ratio), height - 1))
    y1 = max(y0 + 1, min(int(height * config.bottom_ratio), height))
    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    dark = ((hsv[:, :, 2] <= config.dark_value_max)
            & (hsv[:, :, 1] <= config.dark_saturation_max))
    roi_h, roi_w = dark.shape
    left = max(0, min(int(round(roi_w * config.center_left_ratio)), roi_w - 1))
    right = max(left + 1, min(int(round(roi_w * config.center_right_ratio)), roi_w))
    top = max(0, min(int(round(roi_h * config.gap_top_ratio)), roi_h - 1))
    bottom = max(top + 1, min(int(round(roi_h * config.gap_bottom_ratio)), roi_h))
    dark_center = dark[:, left:right]
    # 少于该像素数视为无黑线，允许少量黑点噪声；设为 1 即严格无黑色像素。
    row_has_line = dark_center.sum(axis=1) >= config.min_dark_pixels_per_row
    best_start, gap_rows = _find_longest_gap(row_has_line, top, bottom)
    gap_dark_ratio = (
        float(dark_center[best_start:best_start + gap_rows].mean())
        if gap_rows else 1.0
    )
    detected = gap_rows >= config.min_gap_rows
    result = StepResult(
        detected=detected,
        reason="step" if detected else "gap_too_short",
        near_rows=0, near_center_x=0.5,
        gap_rows=gap_rows, gap_bright_ratio=0.0, gap_dark_ratio=gap_dark_ratio,
        gap_start_row=best_start, far_rows=0,
        debug_image=frame.copy(),
        gap_center_x=(left + right - 1) / (2.0 * max(1, roi_w - 1)),
    )
    result.debug_image = _draw_debug(
        frame, result, config, (x0, y0, x1, y1), dark,
        (left, top, right, bottom),
    )
    return result


class StepTracker:
    """只累计连续命中帧，不再要求边缘或横向位置一致。"""

    def __init__(self, config: StepConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._hits = 0
        self._last_time = None
        self.confirmed = False

    def update(self, result: StepResult, now: float) -> bool:
        if not result.detected:
            self.reset()
            return False
        if (self._last_time is not None
                and now - self._last_time > self.cfg.track_max_frame_gap):
            self._hits = 0
        self._last_time = now
        self._hits += 1
        self.confirmed = self._hits >= max(1, self.cfg.track_confirm_frames)
        return self.confirmed


def _draw_debug(frame, result, config, rect, dark, search):
    debug = frame.copy()
    x0, y0, x1, y1 = rect
    left, top, right, bottom = search
    overlay = debug[y0:y1, x0:x1].copy()
    overlay[dark] = (0, 0, 255)
    debug[y0:y1, x0:x1] = cv2.addWeighted(
        debug[y0:y1, x0:x1], 0.55, overlay, 0.45, 0.0,
    )
    cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), (255, 180, 0), 1)
    # 蓝框为实际搜索区，绿/橙框为最长连续缺口。
    cv2.rectangle(debug, (x0 + left, y0 + top),
                  (x0 + right - 1, y0 + bottom - 1), (255, 200, 0), 1)
    if result.gap_rows > 0:
        color = (0, 255, 0) if result.detected else (0, 165, 255)
        cv2.rectangle(
            debug, (x0 + left, y0 + result.gap_start_row),
            (x0 + right - 1, y0 + result.gap_start_row + result.gap_rows - 1),
            color, 2,
        )
    cv2.rectangle(debug, (0, 0), (debug.shape[1] - 1, 72), (0, 0, 0), -1)
    cv2.putText(debug, "STEP: %s  reason=%s" % (
        "YES" if result.detected else "no", result.reason),
        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
        (0, 255, 0) if result.detected else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(debug, "gap=%d/%d rows  black V<=%d  line>= %d pixels/row" % (
        result.gap_rows, config.min_gap_rows, config.dark_value_max,
        config.min_dark_pixels_per_row),
        (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return debug


def open_camera(
    device: str = "/dev/video0",
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    pixel_format: str = "YUYV",
) -> cv2.VideoCapture:
    """按和巡线节点一样的格式打开摄像头（v4l2 + 指定 FOURCC）。"""
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开摄像头 {device}")

    capture.set(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format)
    )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(
        f"摄像头已打开: {device}, {actual_width}x{actual_height} @ {actual_fps:.1f} fps"
    )
    return capture


def _print_config(config: StepConfig) -> None:
    print(
        f"center={config.center_left_ratio:.2f}~{config.center_right_ratio:.2f} "
        f"gap_y={config.gap_top_ratio:.2f}~{config.gap_bottom_ratio:.2f} "
        f"dark_value_max={config.dark_value_max} "
        f"min_dark_pixels_per_row={config.min_dark_pixels_per_row} "
        f"min_gap_rows={config.min_gap_rows}"
    )


def _adjust_config(config: StepConfig, key: int) -> Tuple[StepConfig, bool]:
    """a/z 调黑色阈值，s/x 调每行黑点数，f/v 调连续缺口行数。"""
    updated = config
    if key == ord("a"):
        updated = replace(config, dark_value_max=min(255, config.dark_value_max + 5))
    elif key == ord("z"):
        updated = replace(config, dark_value_max=max(0, config.dark_value_max - 5))
    elif key == ord("s"):
        updated = replace(config, min_dark_pixels_per_row=config.min_dark_pixels_per_row + 1)
    elif key == ord("x"):
        updated = replace(config, min_dark_pixels_per_row=max(1, config.min_dark_pixels_per_row - 1))
    elif key == ord("f"):
        updated = replace(config, min_gap_rows=min(200, config.min_gap_rows + 2))
    elif key == ord("v"):
        updated = replace(config, min_gap_rows=max(1, config.min_gap_rows - 2))
    elif key == ord(" "):
        _print_config(config)
        return config, False
    elif key in (ord("q"), ord("Q"), 27):
        return config, True
    if updated is not config:
        _print_config(updated)
    return updated, False


def main() -> None:
    parser = argparse.ArgumentParser(description="台阶检测（中心框内连续无黑线行数）")
    parser.add_argument(
        "--source",
        default="/dev/video0",
        help="摄像头设备、图片文件或视频文件；默认 /dev/video0",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pixel-format", default="YUYV")
    parser.add_argument(
        "--no-display", action="store_true", help="不弹窗口，只打印数值"
    )
    args = parser.parse_args()

    config = StepConfig()
    _print_config(config)

    source = args.source
    lower = source.lower()
    is_image = lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))

    if is_image:
        if not os.path.isfile(source):
            raise SystemExit(f"找不到图片 {source}")
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"图片读不出来 {source}")
        result = detect_step(frame, config)
        print(
            f"STEP={'YES' if result.detected else 'no '}  "
            f"reason={result.reason}  gap={result.gap_rows}  "
            f"dark={result.gap_dark_ratio:.3f}"
        )
        if not args.no_display:
            cv2.imshow("Step Detection", result.debug_image)
            print("按任意键关闭窗口")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    if source.startswith("/dev/video") or source.isdigit():
        capture = open_camera(
            device=source,
            width=args.width,
            height=args.height,
            fps=args.fps,
            pixel_format=args.pixel_format,
        )
    else:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise SystemExit(f"打不开视频源 {source}")
        print(f"视频源已打开: {source}")

    window_name = "Step Detection (a/z s/x f/v, space=print, q=quit)"
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("读取失败/视频结束")
                break
            frame_index += 1

            result = detect_step(frame, config)

            if frame_index % 15 == 0:
                print(
                    f"frame={frame_index}  "
                    f"STEP={'YES' if result.detected else 'no '}  "
                    f"reason={result.reason:<16s} "
                    f"gap={result.gap_rows:<3d} "
                    f"dark={result.gap_dark_ratio:.3f}"
                )

            if args.no_display:
                continue

            cv2.imshow(window_name, result.debug_image)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                config, should_exit = _adjust_config(config, key)
                if should_exit:
                    break
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()