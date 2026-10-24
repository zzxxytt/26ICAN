#!/usr/bin/env python3
"""台阶检测：浅色/木色平台把黑线截断。

判定链（缺一不可）：
  1. 近段有线且在中间；
  2. 中心列带出现连续缺口；
  3. 缺口是“平台表面”（浅色或木色，木色不要求纯白）；
  4. 缺口起点附近有横跨行驶路径的平台前缘，线在该边缘附近终止；
  5. 缺口里没有从近端黑线连出去的侧向弯曲轨迹（有则优先按弯道）；
  6. 连续多帧的前缘行、缺口行、横向位置相互对应（由 StepTracker 完成）。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from typing import List, Optional, Tuple

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
    # 时序一致性（StepTracker）
    track_confirm_frames: int = 4
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


def _row_runs(row: BoolMask) -> List[Tuple[int, int]]:
    padded = np.pad((row > 0).astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


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


def _longest_true_run(flags: BoolMask) -> int:
    best = run = 0
    for value in flags:
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _trace_curve(dark, center_left, center_right, near_x, gap_start, gap_rows, roi_w, cfg):
    """从画面底部、近端线位置向上追踪暗色轨迹。

    平台：线在缺口底部结束，往上没有暗块，dx 很小；
    弯道：线离开中心带继续向一侧延伸，dx 大且终点在带外。
    gap_rows 保留在签名里只为兼容调用，实际扫描 [gap_start, 图像底部]。
    """
    empty = (False, 0.0, 0.0, 0, [])
    h = dark.shape[0]
    if h - gap_start < cfg.curve_min_rows:
        return empty
    min_w = max(2, int(cfg.curve_min_width_ratio * roi_w))
    max_w = max(min_w + 1, int(cfg.curve_max_width_ratio * roi_w))
    max_dx = max(1.0, cfg.curve_max_dx_per_row * roi_w)
    x = float(near_x)
    points = []
    for row in range(h - 1, max(-1, gap_start - 1), -1):
        best = None
        for start, end in _row_runs(dark[row]):
            width = end - start
            if not (min_w <= width <= max_w):
                continue
            cx = 0.5 * (start + end - 1)
            distance = abs(cx - x)
            if distance > max_dx:
                continue
            if best is None or distance < best[0]:
                best = (distance, cx)
        if best is None:
            continue
        x = best[1]
        points.append((row, int(round(x))))
    if len(points) < cfg.curve_min_rows:
        return empty
    support = len(points) / max(1, h - gap_start)
    farthest = max(points, key=lambda p: abs(p[1] - near_x))
    dx = float(farthest[1] - near_x)
    center_mid = 0.5 * (center_left + center_right)
    band_half = 0.5 * (center_right - center_left)
    outside = abs(farthest[1] - center_mid) > band_half
    curve = (
        support >= cfg.curve_min_support
        and abs(dx) >= cfg.curve_min_dx * roi_w
        and outside
    )
    side = 1 if dx > 0 else (-1 if dx < 0 else 0)
    return (bool(curve), float(support), dx, side, points)


def _find_platform_edge(gray, line_end_row, cfg):
    """在线结束行附近找横跨行驶路径的平台前缘。

    前缘可以比线结束行高一段（平台立面有高度），但不能低太多。
    返回 (row, 宽度占比, 平均梯度)。
    """
    h, w = gray.shape
    grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    x0, x1 = int(0.08 * w), int(0.92 * w)
    if x1 <= x0:
        return -1, 0.0, 0.0
    band = grad[:, x0:x1]
    row0 = max(0, line_end_row - cfg.edge_max_offset_rows)
    row1 = min(h, line_end_row + cfg.edge_row_tolerance + 1)
    best = (-1, 0.0, 0.0)
    for row in range(row0, row1):
        ratio = _longest_true_run(
            band[row] >= cfg.edge_gradient_min
        ) / float(x1 - x0)
        if ratio >= cfg.edge_min_width_ratio and ratio > best[1]:
            best = (row, ratio, float(np.mean(band[row])))
    return best

def detect_step(frame: BgrImage, config: StepConfig) -> StepResult:
    if frame is None or frame.size == 0:
        raise RuntimeError("空图像")
    height, width = frame.shape[:2]
    x0 = max(0, min(int(width * config.left_ratio), width - 1))
    x1 = max(x0 + 1, min(int(width * config.right_ratio), width))
    y0 = max(0, min(int(height * config.top_ratio), height - 1))
    y1 = max(y0 + 1, min(int(height * config.bottom_ratio), height))
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        raise RuntimeError("ROI 为空")

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark = (value <= config.dark_value_max) & (sat <= config.dark_saturation_max)
    bright = (value >= config.bright_value_min) & (sat <= config.bright_saturation_max)
    platform_like = (value >= config.platform_value_min) & (
        sat <= config.platform_saturation_max
    )
    wood = (
        (hue >= config.wood_hue_min)
        & (hue <= config.wood_hue_max)
        & (sat >= config.wood_saturation_min)
        & (value >= 60)
    )
    surface = platform_like | wood
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi_h, roi_w = dark.shape
    center_left = int(round(roi_w * config.center_left_ratio))
    center_right = int(round(roi_w * config.center_right_ratio))
    center_left = max(0, min(center_left, roi_w - 2))
    center_right = max(center_left + 1, min(center_right, roi_w))
    dark_center = dark[:, center_left:center_right]
    surface_center = surface[:, center_left:center_right]
    row_has_line = dark_center.sum(axis=1) >= config.min_dark_pixels_per_row

    gap_start = int(round(roi_h * config.gap_top_ratio))
    gap_end = int(round(roi_h * config.gap_bottom_ratio))
    gap_start = max(0, min(gap_start, roi_h - 2))
    gap_end = max(gap_start + 1, min(gap_end, roi_h))

    near_start = int(round(roi_h * (1.0 - config.near_bottom_ratio)))
    near_start = max(0, min(near_start, roi_h - 1))
    near_dark = dark[near_start:roi_h, :]
    near_row_has_line = near_dark.sum(axis=1) >= config.min_dark_pixels_per_row
    near_rows = int(near_row_has_line.sum())
    if near_rows > 0:
        _ys, xs = np.nonzero(near_dark)
        near_center_x = float(xs.mean()) / max(1.0, float(roi_w - 1))
    else:
        near_center_x = 0.5
    near_centered = abs(near_center_x - 0.5) <= config.near_center_tolerance_ratio

    best_start, gap_rows = _find_longest_gap(row_has_line, gap_start, gap_end)
    if gap_rows > 0:
        band_dark = dark_center[best_start:best_start + gap_rows]
        band_bright = bright[best_start:best_start + gap_rows,
                             center_left:center_right]
        band_surface = surface_center[best_start:best_start + gap_rows]
        band_wood = wood[best_start:best_start + gap_rows,
                         center_left:center_right]
        gap_dark_ratio = float(band_dark.mean())
        gap_bright_ratio = float(band_bright.mean())
        surface_ratio = float(band_surface.mean())
        wood_ratio = float(band_wood.mean())
        ys, xs = np.nonzero(band_surface)
        if xs.size:
            gap_center_x = (center_left + float(xs.mean())) / max(1.0, roi_w - 1)
        else:
            gap_center_x = near_center_x
    else:
        gap_dark_ratio, gap_bright_ratio = 1.0, 0.0
        surface_ratio = wood_ratio = 0.0
        gap_center_x = near_center_x
        best_start = gap_start

    far_rows = int(row_has_line[gap_start:best_start].sum())

    near_center_px = near_center_x * (roi_w - 1)
    curve, curve_support, curve_dx, curve_side, curve_points = _trace_curve(
        dark, center_left, center_right, near_center_px,
        best_start, gap_rows, roi_w, config,
    )
    gap_bottom = best_start + gap_rows
    edge_row, edge_ratio, edge_strength = _find_platform_edge(
        gray, gap_bottom, config
    )
    line_ends_at_edge = (
        edge_row >= 0
        and gap_bottom - config.edge_max_offset_rows
        <= edge_row
        <= gap_bottom + config.edge_row_tolerance
    )
    edge_ok = (
        edge_row >= 0
        and edge_ratio >= config.edge_min_width_ratio
        and line_ends_at_edge
    )

    detected = (
        near_rows >= config.near_min_line_rows
        and (near_centered or not config.require_near_centered)
        and gap_rows >= config.min_gap_rows
        and surface_ratio >= config.gap_surface_min
        and gap_dark_ratio <= config.gap_dark_max
        and not curve
        and (edge_ok or not config.require_edge)
        and (far_rows > 0 or not config.require_far_line)
    )

    if curve:
        reason = "curve_not_step"
    elif near_rows < config.near_min_line_rows:
        reason = "near_line_lost"
    elif not near_centered and config.require_near_centered:
        reason = "near_off_center"
    elif gap_rows < config.min_gap_rows:
        reason = "gap_too_short"
    elif surface_ratio < config.gap_surface_min:
        reason = "gap_not_platform"
    elif gap_dark_ratio > config.gap_dark_max:
        reason = "gap_has_line"
    elif config.require_edge and edge_row < 0:
        reason = "no_platform_edge"
    elif config.require_edge and not line_ends_at_edge:
        reason = "line_not_at_edge"
    elif config.require_far_line and far_rows <= 0:
        reason = "no_far_line"
    else:
        reason = "step"

    result = StepResult(
        detected=bool(detected),
        reason=reason,
        near_rows=near_rows,
        near_center_x=near_center_x,
        gap_rows=gap_rows,
        gap_bright_ratio=gap_bright_ratio,
        gap_dark_ratio=gap_dark_ratio,
        gap_start_row=best_start,
        far_rows=far_rows,
        debug_image=frame.copy(),
        surface_ratio=surface_ratio,
        wood_ratio=wood_ratio,
        edge_row=edge_row,
        edge_width_ratio=edge_ratio,
        edge_strength=edge_strength,
        line_ends_at_edge=line_ends_at_edge,
        curve=curve,
        curve_support=curve_support,
        curve_dx=curve_dx,
        curve_side=curve_side,
        gap_center_x=gap_center_x,
    )
    result.debug_image = _draw_debug(
        frame, result, config, (x0, y0, x1, y1),
        dark, surface, wood, center_left, center_right,
        near_start, curve_points,
    )
    return result


class StepTracker:
    """连续帧空间一致性：前缘行、缺口起点、横向位置必须相互对应。"""

    def __init__(self, config: StepConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._history = []
        self.confirmed = False

    def update(self, result: StepResult, now: float) -> bool:
        if not result.detected:
            self.reset()
            return False
        if self._history and now - self._history[-1][0] > self.cfg.track_max_frame_gap:
            self._history.clear()

        # 纵向绝对行号会随接近平台而移动；这里只检查
        # “前缘相对线结束行的偏移”和“横向中心”是否稳定。
        gap_bottom = result.gap_start_row + result.gap_rows
        edge_offset = (
            result.edge_row - gap_bottom if result.edge_row >= 0 else -1
        )
        self._history.append((now, edge_offset, result.gap_center_x))

        keep = max(3, 3 * self.cfg.track_confirm_frames)
        if len(self._history) > keep:
            self._history = self._history[-keep:]
        recent = self._history[-self.cfg.track_confirm_frames:]
        if len(recent) < self.cfg.track_confirm_frames:
            self.confirmed = False
            return False

        offsets = [item[1] for item in recent if item[1] >= 0]
        centers = [item[2] for item in recent]
        center_ok = max(centers) - min(centers) <= self.cfg.track_lateral_range
        offset_ok = (
            len(offsets) < 2
            or max(offsets) - min(offsets) <= self.cfg.track_edge_row_range
        )
        self.confirmed = bool(center_ok and offset_ok)
        return self.confirmed


def _draw_debug(frame, result, config, rect, dark, surface, wood,
                center_left, center_right, near_start, curve_points):
    debug = frame.copy()
    x0, y0, x1, y1 = rect
    overlay = debug[y0:y1, x0:x1].copy()
    overlay[dark] = (0, 0, 255)
    overlay[surface] = (255, 160, 0)
    overlay[wood] = (0, 200, 0)
    debug[y0:y1, x0:x1] = cv2.addWeighted(
        debug[y0:y1, x0:x1], 0.55, overlay, 0.45, 0.0
    )
    cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), (255, 180, 0), 1)
    cv2.line(debug, (x0 + center_left, y0), (x0 + center_left, y1 - 1), (255, 200, 0), 1)
    cv2.line(debug, (x0 + center_right, y0), (x0 + center_right, y1 - 1), (255, 200, 0), 1)
    cv2.line(debug, (x0, y0 + near_start), (x1 - 1, y0 + near_start), (0, 255, 0), 1)
    if result.gap_rows > 0:
        color = (0, 255, 0) if result.detected else (0, 165, 255)
        cv2.rectangle(
            debug,
            (x0 + center_left, y0 + result.gap_start_row),
            (x0 + center_right - 1, y0 + result.gap_start_row + result.gap_rows - 1),
            color, 2,
        )
    if curve_points:
        pts = np.array([[x0 + x, y0 + y] for y, x in curve_points], dtype=np.int32)
        cv2.polylines(debug, [pts], False, (0, 255, 0), 2)
    if result.edge_row >= 0:
        cv2.line(debug, (x0, y0 + result.edge_row), (x1 - 1, y0 + result.edge_row),
                 (255, 255, 0), 2)
    cv2.rectangle(debug, (0, 0), (debug.shape[1] - 1, 96), (0, 0, 0), -1)
    title = "STEP: YES" if result.detected else "STEP: no"
    cv2.putText(debug, "%s  reason=%s" % (title, result.reason),
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (0, 255, 0) if result.detected else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(
        debug,
        "gap=%d surface=%.2f wood=%.2f dark=%.3f curve=%d/%.2f edge=%d/%.2f at_edge=%d"
        % (result.gap_rows, result.surface_ratio, result.wood_ratio,
           result.gap_dark_ratio, int(result.curve), result.curve_support,
           result.edge_row, result.edge_width_ratio, int(result.line_ends_at_edge)),
        (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        debug,
        "near=%d near_x=%.2f gap_min=%d surface_min=%.2f edge_w=%.2f"
        % (result.near_rows, result.near_center_x, config.min_gap_rows,
           config.gap_surface_min, config.edge_min_width_ratio),
        (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA,
    )
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
        "当前参数: "
        f"center={config.center_left_ratio:.2f}~{config.center_right_ratio:.2f}  "
        f"bright_value_min={config.bright_value_min}  "
        f"gap_bright_min={config.gap_bright_min:.2f}  "
        f"gap_dark_max={config.gap_dark_max:.3f}  "
        f"min_gap_rows={config.min_gap_rows}"
    )


def _adjust_config(config: StepConfig, key: int) -> Tuple[StepConfig, bool]:
    """按键实时调阈值，返回 (新参数, 是否退出)。"""
    updated = config
    if key == ord("a"):
        updated = replace(config, bright_value_min=min(255, config.bright_value_min + 10))
    elif key == ord("z"):
        updated = replace(config, bright_value_min=max(0, config.bright_value_min - 10))
    elif key == ord("s"):
        updated = replace(config, gap_bright_min=min(1.0, config.gap_bright_min + 0.05))
    elif key == ord("x"):
        updated = replace(config, gap_bright_min=max(0.0, config.gap_bright_min - 0.05))
    elif key == ord("d"):
        updated = replace(config, gap_dark_max=min(0.5, config.gap_dark_max + 0.002))
    elif key == ord("c"):
        updated = replace(config, gap_dark_max=max(0.0, config.gap_dark_max - 0.002))
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
    parser = argparse.ArgumentParser(description="台阶检测（中间列带被浅色台阶截断）")
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
            f"reason={result.reason}  near={result.near_rows}  "
            f"near_x={result.near_center_x:.2f}  gap={result.gap_rows}  "
            f"bright={result.gap_bright_ratio:.2f}  dark={result.gap_dark_ratio:.3f}"
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

    window_name = "Step Detection (a/z s/x d/c f/v, space=print, q=quit)"
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
                    f"near={result.near_rows:<3d} near_x={result.near_center_x:.2f} "
                    f"gap={result.gap_rows:<3d} "
                    f"bright={result.gap_bright_ratio:.2f} "
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