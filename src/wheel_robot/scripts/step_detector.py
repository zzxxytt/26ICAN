#!/usr/bin/env python3
"""台阶检测：浅色/白色的台阶把黑线截断一段距离。

判定思路和现场描述一一对应：
    1. 台阶在 ROI"左中右"的中间   → 只看水平方向中部那一条**中心列带**；
    2. 台阶表面偏浅偏白           → 中心列带里这段的"浅色像素占比"高；
    3. 台阶把黑线截断             → 中心列带里有一段连续的"没有黑线"的行；
    4. 车还没压上台阶             → 紧贴车轮的最底下一段，仍然看得到黑线。

四个条件同时成立才判为台阶：
    近段有线 + 近段线在中间 + 中心列带里连续 N 行没线 + 这段没线的区域整体偏浅

为什么这四个条件缺一不可：
    * 只有"中间没线"   → 可能是彻底丢线、车头偏出赛道，不是台阶；
    * 只有"中间偏浅"   → 可能是地板反光、白纸，线其实还在；
    * 只有"近段有线"   → 那是正常巡线，什么都没发生；
    * 线跑到画面边上去 → 中心列带本来就没线，必须靠"近段线重心在中间"排除。

只做视觉，不依赖 ROS；既可以被 icn.py import，也能单独跑起来调参：
    python3 step_detector.py                       # 用 /dev/video0
    python3 step_detector.py --source step.jpg     # 用一张照片调阈值
    python3 step_detector.py --source step.mp4     # 用一段录像调阈值
    python3 step_detector.py --no-display          # 无界面，只打印数值

运行中按键可以实时调阈值（屏幕上会显示当前值）：
    a / z   浅色亮度下限       bright_value_min   ∓ 10
    s / x   中间浅色占比下限   gap_bright_min     ∓ 0.05
    d / c   中间黑线占比上限   gap_dark_max       ∓ 0.002
    f / v   缺口最少行数       min_gap_rows       ∓ 2
    空格    打印当前参数
    q / Esc 退出
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class StepConfig:
    """台阶检测参数。

    ROI 用和巡线完全一样的比例（line_follower.yaml 的 search_*_ratio），
    这样"有没有黑线"的判断和巡线看到的是同一块地面，不会互相矛盾。
    """

    # ---------------- 巡线 ROI ----------------
    top_ratio: float = 0.56
    bottom_ratio: float = 0.98
    left_ratio: float = 0.02
    right_ratio: float = 0.98

    # ---------------- 左中右：只看 ROI 水平方向的中间这一段 ----------------
    center_left_ratio: float = 0.30
    center_right_ratio: float = 0.70

    # ---------------- 行搜索范围（从 ROI 顶部算起，避开画面最上/最下边缘）----------------
    gap_top_ratio: float = 0.05
    gap_bottom_ratio: float = 0.85

    # ---------------- 黑线（和 line_follower.yaml 的 hsv_upper_v 保持一致）----------------
    dark_value_max: int = 65
    dark_saturation_max: int = 255
    # 一行里至少多少个暗像素，才算"这一行有线"（中心列带宽 246px，线宽 15~25px，取 4 很稳）
    min_dark_pixels_per_row: int = 4

    # ---------------- 浅色 / 白（台阶面）----------------
    bright_value_min: int = 110
    bright_saturation_max: int = 100

    # ---------------- 近段（贴车轮那一段，用整宽判断）----------------
    near_bottom_ratio: float = 0.20
    near_min_line_rows: int = 5      # 近段至少这么多行有线，才认为"脚下还有线"
    # 近段黑线重心必须落在中间（偏离中心超过这个比例就不算），防止线偏到边上时误判
    require_near_centered: bool = True
    near_center_tolerance_ratio: float = 0.18

    # ---------------- 判定阈值 ----------------
    min_gap_rows: int = 12           # 中间连续没线的行数下限（越大越保守）
    gap_bright_min: float = 0.55     # 中间缺口里浅色像素占比下限
    gap_dark_max: float = 0.002      # 中间缺口里黑线像素占比上限
    # 可选收紧：要求缺口"上方"（更远处）重新出现黑线。默认关闭，
    # 因为台阶顶面如果整片都是白的，上面本来就不会有线。
    require_far_line: bool = False


@dataclass
class StepResult:
    """单帧台阶检测结果。"""

    detected: bool
    reason: str
    near_rows: int              # 近段（整宽）有黑线的行数
    near_center_x: float        # 近段黑线重心的横向位置，0~1（0.5 = 正中）
    gap_rows: int               # 中心列带里最长"连续无黑线"的行数
    gap_bright_ratio: float     # 这段缺口里浅色像素占比
    gap_dark_ratio: float       # 这段缺口里黑线像素占比
    gap_start_row: int          # 缺口起点（ROI 内行号，从 0 开始）
    far_rows: int               # 缺口上方那段有黑线的行数（require_far_line 用）
    debug_image: BgrImage


def _find_longest_gap(row_has_line: BoolMask, start: int, end: int) -> Tuple[int, int]:
    """在 [start, end) 里找最长的"连续没有黑线"的一段，返回 (起点, 长度)。"""
    best_start = start
    best_len = 0
    run_start: Optional[int] = None
    for row in range(start, end):
        if not row_has_line[row]:
            if run_start is None:
                run_start = row
            run_len = row - run_start + 1
            if run_len > best_len:
                best_len = run_len
                best_start = run_start
        else:
            run_start = None
    return best_start, best_len


def detect_step(frame: BgrImage, config: StepConfig) -> StepResult:
    """对一帧做台阶判定，返回结果和调试图。

    参数：
        frame  ：BGR 原图（640x480 或任意尺寸，比例裁剪按 config 走）
        config ：StepConfig

    返回：
        StepResult，其中 debug_image 已经画好 ROI、左中右分区、黑/浅掩膜和缺口框。
    """
    if frame is None or frame.size == 0:
        raise RuntimeError("空图像")

    height, width = frame.shape[:2]
    x0 = int(width * config.left_ratio)
    x1 = int(width * config.right_ratio)
    y0 = int(height * config.top_ratio)
    y1 = int(height * config.bottom_ratio)
    x0 = max(0, min(x0, width - 1))
    x1 = max(x0 + 1, min(x1, width))
    y0 = max(0, min(y0, height - 1))
    y1 = max(y0 + 1, min(y1, height))

    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        raise RuntimeError("ROI 为空，请检查 top/bottom/left/right 比例")

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    # 黑线像素 / 浅色（台阶面）像素
    dark: BoolMask = (value <= config.dark_value_max) & (
        saturation <= config.dark_saturation_max
    )
    bright: BoolMask = (value >= config.bright_value_min) & (
        saturation <= config.bright_saturation_max
    )

    roi_h, roi_w = dark.shape

    # 左中右：中心列带
    center_left = int(round(roi_w * config.center_left_ratio))
    center_right = int(round(roi_w * config.center_right_ratio))
    center_left = max(0, min(center_left, roi_w - 2))
    center_right = max(center_left + 1, min(center_right, roi_w))
    dark_center = dark[:, center_left:center_right]
    bright_center = bright[:, center_left:center_right]

    # 中心列带里，逐行判断"有没有黑线"
    row_has_line_center: BoolMask = (
        dark_center.sum(axis=1) >= config.min_dark_pixels_per_row
    )

    # 行搜索范围
    gap_start = int(round(roi_h * config.gap_top_ratio))
    gap_end = int(round(roi_h * config.gap_bottom_ratio))
    gap_start = max(0, min(gap_start, roi_h - 2))
    gap_end = max(gap_start + 1, min(gap_end, roi_h))

    # 近段（整宽，贴车轮那一段）：线还在不在、在不在中间
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
    near_centered = (
        abs(near_center_x - 0.5) <= config.near_center_tolerance_ratio
    )

    # 中心列带里找最长的"连续没有黑线"的一段
    best_start, gap_rows = _find_longest_gap(row_has_line_center, gap_start, gap_end)
    if gap_rows > 0:
        band_dark = dark_center[best_start : best_start + gap_rows]
        band_bright = bright_center[best_start : best_start + gap_rows]
        gap_dark_ratio = float(band_dark.mean())
        gap_bright_ratio = float(band_bright.mean())
    else:
        gap_dark_ratio = 1.0
        gap_bright_ratio = 0.0
        best_start = gap_start

    # 缺口上方（更远处）中心列带里还有没有线
    far_rows = int(row_has_line_center[gap_start:best_start].sum())

    detected = (
        near_rows >= config.near_min_line_rows
        and (near_centered or not config.require_near_centered)
        and gap_rows >= config.min_gap_rows
        and gap_bright_ratio >= config.gap_bright_min
        and gap_dark_ratio <= config.gap_dark_max
        and (far_rows > 0 or not config.require_far_line)
    )

    if detected:
        reason = "step"
    elif near_rows < config.near_min_line_rows:
        reason = "near_line_lost"
    elif not near_centered and config.require_near_centered:
        reason = "near_off_center"
    elif gap_rows < config.min_gap_rows:
        reason = "gap_too_short"
    elif gap_bright_ratio < config.gap_bright_min:
        reason = "gap_not_bright"
    elif gap_dark_ratio > config.gap_dark_max:
        reason = "gap_has_line"
    else:
        reason = "no_far_line"

    debug_image = _draw_debug(
        frame=frame,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        roi_w=roi_w,
        dark=dark,
        bright=bright,
        center_left=center_left,
        center_right=center_right,
        near_start=near_start,
        near_center_x=near_center_x,
        gap_begin=best_start,
        gap_rows=gap_rows,
        detected=detected,
        reason=reason,
        near_rows=near_rows,
        gap_bright_ratio=gap_bright_ratio,
        gap_dark_ratio=gap_dark_ratio,
        config=config,
    )

    return StepResult(
        detected=detected,
        reason=reason,
        near_rows=near_rows,
        near_center_x=near_center_x,
        gap_rows=gap_rows,
        gap_bright_ratio=gap_bright_ratio,
        gap_dark_ratio=gap_dark_ratio,
        gap_start_row=best_start,
        far_rows=far_rows,
        debug_image=debug_image,
    )


def _draw_debug(
    frame: BgrImage,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    roi_w: int,
    dark: BoolMask,
    bright: BoolMask,
    center_left: int,
    center_right: int,
    near_start: int,
    near_center_x: float,
    gap_begin: int,
    gap_rows: int,
    detected: bool,
    reason: str,
    near_rows: int,
    gap_bright_ratio: float,
    gap_dark_ratio: float,
    config: StepConfig,
) -> BgrImage:
    """把 ROI、左中右分区、黑/浅掩膜和缺口框画到原图上。"""
    debug = frame.copy()

    # 黑线 = 红，浅色 = 蓝
    overlay = debug[y0:y1, x0:x1].copy()
    overlay[:, :, 2][dark] = 255     # R
    overlay[:, :, 0][dark] = 0       # B
    overlay[:, :, 1][dark] = 0       # G
    overlay[:, :, 0][bright] = 255   # B
    overlay[:, :, 1][bright] = 160   # G
    overlay[:, :, 2][bright] = 0     # R
    debug[y0:y1, x0:x1] = cv2.addWeighted(
        debug[y0:y1, x0:x1], 0.55, overlay, 0.45, 0.0
    )

    # ROI 边框
    cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), (255, 180, 0), 1)

    # 左中右分区：两条竖线标出中心列带
    cv2.line(
        debug, (x0 + center_left, y0), (x0 + center_left, y1 - 1), (255, 200, 0), 1
    )
    cv2.line(
        debug, (x0 + center_right, y0), (x0 + center_right, y1 - 1), (255, 200, 0), 1
    )

    # 近段上界（绿）
    cv2.line(debug, (x0, y0 + near_start), (x1 - 1, y0 + near_start), (0, 255, 0), 1)

    # 近段黑线重心位置（绿色小圆点）
    near_px = int(x0 + near_center_x * (roi_w - 1))
    cv2.circle(debug, (near_px, y1 - 6), 5, (0, 255, 0), -1)

    # 中心列带里的最长缺口框
    if gap_rows > 0:
        color = (0, 255, 0) if detected else (0, 165, 255)
        cv2.rectangle(
            debug,
            (x0 + center_left, y0 + gap_begin),
            (x0 + center_right - 1, y0 + gap_begin + gap_rows - 1),
            color,
            2,
        )

    # 文字
    cv2.rectangle(debug, (0, 0), (debug.shape[1] - 1, 76), (0, 0, 0), -1)
    title = "STEP: YES" if detected else "STEP: no"
    cv2.putText(
        debug,
        f"{title}  reason={reason}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0) if detected else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug,
        f"near_rows={near_rows}  near_x={near_center_x:.2f}  gap={gap_rows}  "
        f"bright={gap_bright_ratio:.2f}  dark={gap_dark_ratio:.3f}",
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug,
        f"center={config.center_left_ratio:.2f}~{config.center_right_ratio:.2f}  "
        f"bright_v>={config.bright_value_min}  bright_min={config.gap_bright_min:.2f}  "
        f"dark_max={config.gap_dark_max:.3f}  gap_min={config.min_gap_rows}",
        (8, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
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