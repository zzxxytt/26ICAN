#!/usr/bin/env python3
"""Safe camera-based black-line follower for the wheel-legged robot."""

from dataclasses import dataclass
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, UInt8

# 斑马线检测是同一目录下的另一个独立程序（zebra_crossing_detector.py），
# 与巡线是"不同情况各用各的"关系，所以这里不做模块级 import：
# 统一在 _setup_zebra() 里按需加载，未启用时不 import、不建话题、不耗 CPU，
# 该文件不在时 line_follower.py 依然可以单独运行。


@dataclass
class DetectionResult:
    valid: bool
    error: float
    confidence: float
    reason: str
    debug_frame: np.ndarray
    corner: bool = False  # 近段处于"急弯横段"状态：应以原地转向为主
    # Optional vision-element overrides. Defaults preserve the original follower.
    speed_limit: Optional[float] = None
    inhibit_hold: bool = False


@dataclass
class BandResult:
    """近/中/远三段中某一段的检测结果。"""

    name: str
    weight: float
    top_row: int
    bottom_row: int
    band_rows: int
    rows: int
    widest: int
    picked_width: int
    center_x: float
    center_y: float
    error: float
    coverage: float
    width_quality: float
    continuity: float
    confidence: float
    valid: bool
    max_width: int = 0
    horizontal: bool = False
    corner: bool = False
    corner_rows: int = 0
    corner_bottom: bool = False
    selected: Optional[np.ndarray] = None


class HsvStripLineDetector:
    """HSV 黑线检测器：近/中/远三段独立筛选 + 加权融合。

    ROI 从画面底部往上分三段（near / mid / far），每段用各自的宽度窗口
    （远段胶带在透视上更窄）独立做"逐行宽度筛选 + 跨行连续性跟踪"，
    再按 band_weight_* 融合成一路误差：近段权重最大，相当于贴着车轮附近
    的线走；远段只提供弯道趋势，避免"只看脚下一条、没有预判"。
    """

    def __init__(
        self,
        search_top_ratio: float,
        search_bottom_ratio: float,
        search_left_ratio: float,
        search_right_ratio: float,
        hsv_lower: Tuple[int, int, int],
        hsv_upper: Tuple[int, int, int],
        adaptive_value_enabled: bool,
        adaptive_block_size: int,
        adaptive_c: float,
        morphology_kernel: int,
        erode_iterations: int,
        dilate_iterations: int,
        min_line_width_ratio: float,
        max_line_width_ratio: float,
        expected_line_width_ratio: float,
        width_tolerance_ratio: float,
        max_row_center_jump_ratio: float,
        min_valid_row_fraction: float,
        min_confidence: float,
        edge_recovery_min_row_fraction: float,
        edge_recovery_min_confidence: float,
        edge_recovery_error_threshold: float,
        target_offset_pixels: float,
        band_weight_near: float = 0.55,
        band_weight_mid: float = 0.30,
        band_weight_far: float = 0.15,
        band_split_near: float = 0.25,
        band_split_mid: float = 0.60,
        band_width_scale_mid: float = 0.72,
        band_width_scale_far: float = 0.52,
        band_gate_mid: float = 0.12,
        band_gate_far: float = 0.22,
        sharp_curve_width_ratio: float = 2.5,
        corner_gate_ratio: float = 0.18,
        debug_row_grid: bool = False,
        band_require_near: bool = True,
    ) -> None:
        self.search_top_ratio = search_top_ratio
        self.search_bottom_ratio = search_bottom_ratio
        self.search_left_ratio = search_left_ratio
        self.search_right_ratio = search_right_ratio
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.adaptive_value_enabled = adaptive_value_enabled
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.morphology_kernel = morphology_kernel
        self.erode_iterations = erode_iterations
        self.dilate_iterations = dilate_iterations
        self.min_line_width_ratio = min_line_width_ratio
        self.max_line_width_ratio = max_line_width_ratio
        self.expected_line_width_ratio = expected_line_width_ratio
        self.width_tolerance_ratio = width_tolerance_ratio
        self.max_row_center_jump_ratio = max_row_center_jump_ratio
        self.min_valid_row_fraction = min_valid_row_fraction
        self.min_confidence = min_confidence
        self.edge_recovery_min_row_fraction = edge_recovery_min_row_fraction
        self.edge_recovery_min_confidence = edge_recovery_min_confidence
        self.edge_recovery_error_threshold = edge_recovery_error_threshold
        self.target_offset_pixels = target_offset_pixels
        # 近/中/远三段融合（参考出厂 line_follower 的 0.55/0.30/0.15）
        self.band_weight_near = band_weight_near
        self.band_weight_mid = band_weight_mid
        self.band_weight_far = band_weight_far
        self.band_split_near = band_split_near
        self.band_split_mid = band_split_mid
        self.band_width_scale_mid = band_width_scale_mid
        self.band_width_scale_far = band_width_scale_far
        self.band_gate_mid = band_gate_mid
        self.band_gate_far = band_gate_far
        self.sharp_curve_width_ratio = sharp_curve_width_ratio
        self.corner_gate_ratio = corner_gate_ratio
        self.debug_row_grid = debug_row_grid
        # 上一帧的融合中心（画面坐标），作为"横段是不是同一条线"的闸门参考
        self._last_center_x: Optional[float] = None
        self.band_require_near = band_require_near

    @staticmethod
    def _row_runs(row: np.ndarray) -> List[Tuple[int, int]]:
        padded = np.pad((row > 0).astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        return list(zip(starts.tolist(), ends.tolist()))

    def _build_mask(self, roi: np.ndarray) -> np.ndarray:
        """HSV 阈值（可选 OR 自适应阈值）+ 形态学，得到暗色候选掩膜。"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        if self.adaptive_value_enabled:
            adaptive_mask = cv2.adaptiveThreshold(
                hsv[:, :, 2],
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.adaptive_block_size,
                self.adaptive_c,
            )
            mask = cv2.bitwise_or(mask, adaptive_mask)
        if self.morphology_kernel > 1:
            kernel = np.ones(
                (self.morphology_kernel, self.morphology_kernel), dtype=np.uint8
            )
            if self.erode_iterations > 0:
                mask = cv2.erode(mask, kernel, iterations=self.erode_iterations)
            if self.dilate_iterations > 0:
                mask = cv2.dilate(mask, kernel, iterations=self.dilate_iterations)
        return mask

    def _band_layout(
        self, roi_height: int
    ) -> List[Tuple[str, float, float, int, int]]:
        """按比例把 ROI 从下往上切成 near / mid / far，返回每段的布局信息。

        (名称, 权重, 宽度缩放, 顶部行, 底部行)
        band_split_near / band_split_mid 是"从底部量起"的占比：
        near 占最底 band_split_near（默认 25%），mid 到 band_split_mid（默认 60%），
        剩下的给 far。这样 near 段始终咬住画面最底部，不会被抬得太远。

        宽度缩放近似透视：远段胶带在像素上更窄，所以该段的
        min/max/expected/tolerance 同步乘以这个系数。
        """
        if roi_height < 9:
            return [("near", 1.0, 1.0, 0, roi_height)]
        near_top = roi_height - max(
            3, int(round(roi_height * self.band_split_near))
        )
        mid_top = roi_height - max(
            6, int(round(roi_height * self.band_split_mid))
        )
        near_top = min(max(near_top, 3), roi_height - 3)
        mid_top = min(max(mid_top, 0), near_top - 1)
        return [
            ("near", self.band_weight_near, 1.0, near_top, roi_height),
            (
                "mid",
                self.band_weight_mid,
                self.band_width_scale_mid,
                mid_top,
                near_top,
            ),
            ("far", self.band_weight_far, self.band_width_scale_far, 0, mid_top),
        ]

    def _scan_band(
        self,
        mask: np.ndarray,
        top_row: int,
        bottom_row: int,
        width_scale: float,
        x0: int,
        roi_width: int,
        target_x: float,
        max_jump: float,
        allow_corner: bool = False,
        reference_x: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[float], List[int], int, int]:
        """在一条搜索带内逐行挑胶带。

        返回 (带内选中掩膜, 中心, 宽度, 最宽段, 急弯行数)。

        allow_corner=True（只给近段用）时，若某一行没有"竖着"的胶带、却有一段
        横向暗带盖住上一帧的中心，就把这段横带的中点和宽度当作这一行的结果：
        横带的中点在几何上天然偏向拐弯一侧，正好是急弯需要的转向信号。
        """
        min_width = max(2, int(roi_width * self.min_line_width_ratio * width_scale))
        max_width = max(
            min_width + 1, int(roi_width * self.max_line_width_ratio * width_scale)
        )
        expected_width = roi_width * self.expected_line_width_ratio * width_scale
        width_tolerance = max(
            1.0, roi_width * self.width_tolerance_ratio * width_scale
        )
        corner_margin = roi_width * self.corner_gate_ratio

        band_height = bottom_row - top_row
        selected = np.zeros((band_height, roi_width), dtype=np.uint8)
        centers: List[float] = []
        widths: List[int] = []
        widest = 0
        corner_rows = 0
        corner_at_bottom = False
        previous_center: Optional[float] = None
        previous_row: Optional[int] = None

        # 从最靠近车轮的一行往上扫描：细缝宽度不足被跳过，
        # 横向大暗块在"不盖住上一帧中心"时也被跳过。
        for local_row in range(band_height - 1, -1, -1):
            candidates = []
            widest_span: Optional[Tuple[int, int, int]] = None
            for start, end in self._row_runs(mask[top_row + local_row]):
                width = end - start
                if width > widest:
                    widest = width
                if min_width <= width <= max_width:
                    run_center = x0 + 0.5 * (start + end - 1)
                    candidates.append((start, end, run_center, width))
                elif width > max_width:
                    span_start = x0 + start
                    span_end = x0 + end
                    if widest_span is None or width > widest_span[2]:
                        widest_span = (span_start, span_end, width)

            if not candidates:
                # 急弯分支：本行没有竖着的胶带，只剩横向段。
                # 条件：横段盖住上一帧中心（是同一条线），且这一行足够靠近画面底部。
                # "底部"这条是关键：远处 U 型弯的横带也会横跨画面中心，但它出现在
                # 画面中上部，不是车头脚下——只有最底部几行是横段，才说明真的到弯口了。
                near_bottom = local_row >= band_height - 3
                if (
                    allow_corner
                    and widest_span is not None
                    and reference_x is not None
                    and widest_span[0] - corner_margin
                    <= reference_x
                    <= widest_span[1] + corner_margin
                ):
                    span_start, span_end, span_width = widest_span
                    left = max(0, span_start - x0)
                    right = min(roi_width, span_end - x0)
                    if right > left:
                        selected[local_row, left:right] = 255
                    centers.append(0.5 * (span_start + span_end))
                    widths.append(span_width)
                    corner_rows += 1
                    if near_bottom:
                        corner_at_bottom = True
                continue

            reference = previous_center if previous_center is not None else target_x
            row_gap = 1 if previous_row is None else max(1, previous_row - local_row)
            allowed_jump = min(0.35 * roi_width, max_jump * row_gap)

            def score(candidate: Tuple[int, int, float, int]) -> float:
                _start, _end, center, width = candidate
                width_error = abs(width - expected_width) / width_tolerance
                center_error = abs(center - reference) / allowed_jump
                return width_error + 0.35 * center_error

            compatible = [
                candidate
                for candidate in candidates
                if previous_center is None
                or abs(candidate[2] - previous_center) <= allowed_jump
            ]
            if not compatible:
                continue
            start, end, center, width = min(compatible, key=score)
            selected[local_row, start:end] = 255
            centers.append(center)
            widths.append(width)
            previous_center = center
            previous_row = local_row
        return selected, centers, widths, widest, corner_rows, corner_at_bottom

    def _evaluate_band(
        self,
        name: str,
        weight: float,
        width_scale: float,
        mask: np.ndarray,
        top_row: int,
        bottom_row: int,
        x0: int,
        roi_width: int,
        frame_width: int,
        target_x: float,
        max_jump: float,
        allow_corner: bool = False,
        reference_x: Optional[float] = None,
    ) -> BandResult:
        """跑完一条带，算出它自己的中心、误差、覆盖率与置信度。"""
        selected, centers, widths, widest, corner_rows, corner_bottom = self._scan_band(
            mask,
            top_row,
            bottom_row,
            width_scale,
            x0,
            roi_width,
            target_x,
            max_jump,
            allow_corner,
            reference_x,
        )
        band_rows = max(1, bottom_row - top_row)
        coverage = len(centers) / band_rows
        picked_width = int(np.median(widths)) if widths else 0
        max_width = max(2, int(roi_width * self.max_line_width_ratio * width_scale))
        corner = corner_rows > 0
        # 有横向暗带、但我们并没有把它当作"同一条线在拐弯"来跟踪 → 这段不可信
        # （典型就是地图外的地面缝隙、垫子边缘）。
        horizontal = (
            widest >= max(max_width + 1, int(self.sharp_curve_width_ratio * max_width))
            and not corner
        )

        empty = BandResult(
            name=name,
            weight=weight,
            top_row=top_row,
            bottom_row=bottom_row,
            band_rows=band_rows,
            rows=len(centers),
            widest=widest,
            picked_width=picked_width,
            center_x=-1.0,
            center_y=-1.0,
            error=0.0,
            coverage=coverage,
            width_quality=0.0,
            continuity=0.0,
            confidence=0.0,
            valid=False,
            max_width=max_width,
            horizontal=horizontal,
            corner=corner,
            corner_rows=corner_rows,
            corner_bottom=corner_bottom,
            selected=selected,
        )
        if not centers:
            return empty

        moments = cv2.moments(selected)
        if moments["m00"] <= 0.0:
            return empty

        center_x = x0 + moments["m10"] / moments["m00"]
        center_y = top_row + moments["m01"] / moments["m00"]
        error = float(
            np.clip((center_x - target_x) / (0.5 * frame_width), -1.0, 1.0)
        )

        expected_width = roi_width * self.expected_line_width_ratio * width_scale
        width_tolerance = max(
            1.0, roi_width * self.width_tolerance_ratio * width_scale
        )
        width_quality = float(
            np.mean(
                [
                    max(0.0, 1.0 - abs(width - expected_width) / width_tolerance)
                    for width in widths
                ]
            )
        )
        if len(centers) > 1:
            continuity = float(
                np.mean(np.maximum(0.0, 1.0 - np.abs(np.diff(centers)) / max_jump))
            )
        else:
            continuity = 0.0
        confidence = 0.60 * coverage + 0.25 * width_quality + 0.15 * continuity

        # 单段至少要占三行、覆盖率 15% 以上，才算"这一段看到了线"
        valid = len(centers) >= max(3, int(0.15 * band_rows))

        return BandResult(
            name=name,
            weight=weight,
            top_row=top_row,
            bottom_row=bottom_row,
            band_rows=band_rows,
            rows=len(centers),
            widest=widest,
            picked_width=picked_width,
            center_x=center_x,
            center_y=center_y,
            error=error,
            coverage=coverage,
            width_quality=width_quality,
            continuity=continuity,
            confidence=confidence,
            valid=valid,
            max_width=max_width,
            horizontal=horizontal,
            corner=corner,
            corner_rows=corner_rows,
            corner_bottom=corner_bottom,
            selected=selected,
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self.last_diagnostics = "empty_roi"
        debug = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        x0 = int(frame_width * self.search_left_ratio)
        x1 = int(frame_width * self.search_right_ratio)
        y0 = int(frame_height * self.search_top_ratio)
        y1 = int(frame_height * self.search_bottom_ratio)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return DetectionResult(False, 0.0, 0.0, "empty_roi", debug)

        mask = self._build_mask(roi)
        roi_height, roi_width = mask.shape
        max_jump = max(1.0, roi_width * self.max_row_center_jump_ratio)
        target_x = 0.5 * frame_width + self.target_offset_pixels

        band_results = [
            self._evaluate_band(
                name,
                weight,
                width_scale,
                mask,
                top_row,
                bottom_row,
                x0,
                roi_width,
                frame_width,
                target_x,
                max_jump,
                # Keep the established near-band continuity fallback. Disabling
                # it also rejects straight tape that becomes wider in perspective.
                allow_corner=(name == "near"),
                reference_x=self._last_center_x,
            )
            for name, weight, width_scale, top_row, bottom_row in self._band_layout(
                roi_height
            )
        ]

        # ---- 两道闸：横向段不算数；上两段必须靠近段 ----
        # 1) 急弯把胶带压成近乎水平时，单行 run 会很宽，它的"中心"没有方向意义；
        #    地图外的地面缝隙/边缘也常以横向暗带形式出现，同样必须排除。
        for band in band_results:
            if band.horizontal:
                band.valid = False

        # 2) 中/远段的中心必须落在近段附近（透视上离得越远允许越松）。
        #    偏得太远说明它锁到了别的东西（缝隙、砖缝、地图边缘），直接作废，
        #    否则 0.30/0.15 的权重会把融合误差带偏，表现为"大红点乱跳、车不走"。
        near_band = band_results[0]
        if near_band.valid:
            for band in band_results[1:]:
                if not band.valid:
                    continue
                gate_ratio = (
                    self.band_gate_mid if band.name == "mid" else self.band_gate_far
                )
                if abs(band.center_x - near_band.center_x) > frame_width * gate_ratio:
                    band.valid = False

        # ---- 三段加权融合：近段主导，远段只给弯道趋势 ----
        valid_bands = [band for band in band_results if band.valid]
        weight_sum = sum(band.weight for band in valid_bands)
        tracking_possible = bool(valid_bands) and weight_sum > 0.0
        if self.band_require_near and not near_band.valid:
            tracking_possible = False

        if tracking_possible:
            fused_error = (
                sum(band.weight * band.error for band in valid_bands) / weight_sum
            )
            fused_coverage = (
                sum(band.weight * band.coverage for band in valid_bands) / weight_sum
            )
            fused_width_quality = (
                sum(band.weight * band.width_quality for band in valid_bands)
                / weight_sum
            )
            fused_continuity = (
                sum(band.weight * band.continuity for band in valid_bands)
                / weight_sum
            )
            fused_confidence = (
                0.60 * fused_coverage
                + 0.25 * fused_width_quality
                + 0.15 * fused_continuity
            )
            fused_center_x = target_x + fused_error * 0.5 * frame_width
            fused_center_y = y0 + (
                sum(band.weight * band.center_y for band in valid_bands) / weight_sum
            )
        else:
            fused_error = 0.0
            fused_coverage = 0.0
            fused_confidence = 0.0
            fused_center_x = target_x
            fused_center_y = float(y1)

        # 近段处于"急弯横段"状态：横段中点天然偏向拐弯一侧，可作为转向目标。
        # 要求横段占本段足够比例，避免个别行误触发。
        corner_active = bool(
            near_band.corner
            and near_band.valid
            and near_band.corner_bottom
            and near_band.corner_rows >= max(3, int(0.25 * near_band.band_rows))
        )
        if tracking_possible:
            # 供下一帧判定"横段是否还是同一条线"
            self._last_center_x = fused_center_x

        # ---- 调试图：红=候选掩膜，绿=各段选中的行，小点=分带中心，大点=融合中心 ----
        selected_full = np.zeros_like(mask)
        for band in band_results:
            if band.selected is not None:
                selected_full[band.top_row : band.bottom_row] = band.selected

        raw_pixels = int(np.count_nonzero(mask))
        picked_pixels = int(np.count_nonzero(selected_full))
        self.last_diagnostics = (
            f"mask={raw_pixels} picked={picked_pixels} "
            + " ".join(
                f"{b.name[0].upper()}={b.rows}/{b.band_rows}"
                f"(wide={b.widest},keep<={b.max_width},ok={int(b.valid)})" for b in band_results
            )
            + f" cov={fused_coverage:.2f}"
        )

        raw_overlay = np.zeros_like(roi)
        raw_overlay[:, :, 2] = mask
        debug[y0:y1, x0:x1] = cv2.addWeighted(
            debug[y0:y1, x0:x1], 0.82, raw_overlay, 0.18, 0.0
        )
        selected_overlay = np.zeros_like(roi)
        selected_overlay[:, :, 1] = selected_full
        debug[y0:y1, x0:x1] = cv2.addWeighted(
            debug[y0:y1, x0:x1], 0.78, selected_overlay, 0.35, 0.0
        )

        cv2.rectangle(debug, (x0, y0), (x1, y1), (255, 180, 0), 1)
        if self.debug_row_grid:
            # 每 40 行一条参考线，用来把"地面距离"和"图像行号"对上
            for grid_y in range(0, frame_height, 40):
                cv2.line(debug, (0, grid_y), (frame_width - 1, grid_y), (90, 90, 90), 1)
                cv2.putText(
                    debug,
                    str(grid_y),
                    (2, min(frame_height - 2, grid_y + 11)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (170, 170, 170),
                    1,
                    cv2.LINE_AA,
                )
        cv2.line(
            debug,
            (int(target_x), y0),
            (int(target_x), y1),
            (255, 255, 0),
            1,
        )

        band_colors = {
            "near": (0, 255, 0),
            "mid": (0, 255, 255),
            "far": (0, 140, 255),
        }
        for band in band_results:
            bottom = y0 + band.bottom_row
            color = band_colors.get(band.name, (200, 200, 200))
            cv2.line(debug, (x0, bottom - 1), (x1, bottom - 1), color, 1)
            cv2.putText(
                debug,
                f"{band.name} {band.rows}/{band.band_rows} "
                f"w={band.picked_width} max={band.widest}"
                f" keep<={band.max_width} ok={int(band.valid)}"
                + (" H" if band.horizontal else "")
                + (" C" if band.corner else ""),
                (x0 + 4, bottom - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
            if band.valid:
                cv2.circle(
                    debug,
                    (int(band.center_x), int(y0 + band.center_y)),
                    4,
                    color,
                    -1,
                )

        if tracking_possible:
            cv2.circle(
                debug,
                (int(fused_center_x), int(fused_center_y)),
                9,
                (255, 0, 255),
                -1,
            )

        rows_by_name = {band.name: band.rows for band in band_results}
        cv2.putText(
            debug,
            f"MASK pixels={raw_pixels} picked={picked_pixels} "
            f"near width={max(2, int(roi_width*self.min_line_width_ratio))}..{near_band.max_width}px",
            (max(8, x0), max(68, y0 - 54)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            f"HSV V<={int(self.hsv_upper[2])}+adaptive  "
            f"rows N={rows_by_name.get('near', 0)} "
            f"M={rows_by_name.get('mid', 0)} "
            f"F={rows_by_name.get('far', 0)}  "
            f"cov={fused_coverage:.2f} conf={fused_confidence:.2f}",
            (max(8, x0), max(24, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        width_line = "  ".join(
            f"{band.name[0].upper()}={band.picked_width}px" for band in band_results
        )
        cv2.putText(
            debug,
            f"width picked {width_line}  err={fused_error:+.3f}",
            (max(8, x0), max(46, y0 - 32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        if not tracking_possible:
            if any(band.horizontal for band in band_results):
                reason = "sharp_curve_ahead"
            elif valid_bands:
                reason = "near_band_lost"
            else:
                reason = "no_black_tape"
            return DetectionResult(False, 0.0, 0.0, reason, debug)

        regular_tracking = (
            fused_coverage >= self.min_valid_row_fraction
            and fused_confidence >= self.min_confidence
        )
        edge_recovery = (
            fused_coverage >= self.edge_recovery_min_row_fraction
            and fused_confidence >= self.edge_recovery_min_confidence
            and abs(fused_error) >= self.edge_recovery_error_threshold
        )
        if regular_tracking:
            return DetectionResult(
                True,
                fused_error,
                fused_confidence,
                "tracking",
                debug,
                corner=corner_active,
            )
        if edge_recovery:
            return DetectionResult(
                True,
                fused_error,
                fused_confidence,
                "edge_recovery",
                debug,
                corner=corner_active,
            )
        if fused_coverage < self.min_valid_row_fraction:
            return DetectionResult(
                False,
                fused_error,
                fused_confidence,
                "insufficient_vertical_tape",
                debug,
            )
        if fused_confidence < self.min_confidence:
            return DetectionResult(
                False, fused_error, fused_confidence, "low_confidence", debug
            )
        return DetectionResult(
            False, fused_error, fused_confidence, "invalid_line", debug
        )


class LineFollower(Node):
    VALID_ROTATIONS = (0, 90, 180, 270)
    VALID_MIRROR_MODES = ("none", "horizontal", "vertical", "both")

    def __init__(self) -> None:
        super().__init__("line_follower")
        self._declare_parameters()
        self._load_parameters()
        self._validate_parameters()

        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # 斑马线检测：独立的另一套程序，默认关闭。
        self._zebra_module = None
        self._zebra_detected_publisher = None
        self._zebra_count_publisher = None
        self._zebra_roi_config = None
        self._zebra_perspective_config = None
        self._zebra_hit = 0
        self._zebra_miss = 0
        self._zebra_confirmed = False
        self._zebra_cycle = 0
        self._setup_zebra()

        self._frame_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._stop_capture = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_time = 0.0
        self._frame_sequence = 0
        self._camera_healthy = False
        self._capture: Optional[cv2.VideoCapture] = None
        self._last_processed_sequence = -1
        self._last_command = (0.0, 0.0)
        self._last_command_valid = False
        self._last_motion_update_time = time.monotonic()
        self._previous_error = 0.0
        self._previous_control_time = 0.0
        self._have_previous_error = False
        self._lost_hold = 0
        self._last_status_log = 0.0
        self._last_failure_reason = "startup"
        self._exit_requested = False
        self._closing = False
        self._debug_window_ready = False
        self._debug_fullscreen_requested = False
        self._debug_canvas: Optional[np.ndarray] = None
        self._debug_canvas_target: Optional[Tuple[int, int]] = None
        self._debug_canvas_tick = 0
        self._debug_screen_size: Optional[Tuple[int, int]] = None

        self.detector = HsvStripLineDetector(
            self.search_top_ratio,
            self.search_bottom_ratio,
            self.search_left_ratio,
            self.search_right_ratio,
            (self.hsv_lower_h, self.hsv_lower_s, self.hsv_lower_v),
            (self.hsv_upper_h, self.hsv_upper_s, self.hsv_upper_v),
            self.adaptive_value_enabled,
            self.adaptive_block_size,
            self.adaptive_c,
            self.hsv_morphology_kernel,
            self.erode_iterations,
            self.dilate_iterations,
            self.min_line_width_ratio,
            self.max_line_width_ratio,
            self.expected_line_width_ratio,
            self.width_tolerance_ratio,
            self.max_row_center_jump_ratio,
            self.min_valid_row_fraction,
            self.min_confidence,
            self.edge_recovery_min_row_fraction,
            self.edge_recovery_min_confidence,
            self.edge_recovery_error_threshold,
            self.target_offset_pixels,
            self.band_weight_near,
            self.band_weight_mid,
            self.band_weight_far,
            self.band_split_near,
            self.band_split_mid,
            self.band_width_scale_mid,
            self.band_width_scale_far,
            self.band_gate_mid,
            self.band_gate_far,
            self.sharp_curve_width_ratio,
            self.corner_gate_ratio,
            self.debug_row_grid,
            self.band_require_near,
        )

        self._roundabout = None
        self._roundabout_owns_control = False
        self._roundabout_init_error = ""
        self._last_roundabout_mode = "off"

        if self.show_debug:
            try:
                cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
                self._debug_window_ready = True
            except cv2.error as error:
                self.get_logger().error(f"调试窗口创建失败，继续安全运行: {error}")

        self._publish_zero("startup")
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="line_follower_camera", daemon=True
        )
        self._capture_thread.start()
        self._control_timer_handle = self.create_timer(
            1.0 / self.control_rate, self._control_timer
        )

        mode_text = "运动已启用" if self.enable_motion else "架空调试（运动未启用）"
        self.get_logger().info(
            f"巡线节点启动: {mode_text}, 控制频率={self.control_rate:.1f} Hz"
        )
        self.get_logger().info(
            "本节点仅发布 /cmd_vel 的 linear.x 和 angular.z；所有失效状态发布零速度"
        )
        self.get_logger().info(
            f"物理前进方向映射: linear.x = {self.linear_direction_sign:+.0f} × 前进速度"
        )
        self.get_logger().info(
            f"巡线配置 source={__file__} near_continuity=on "
            f"ROI=({self.search_left_ratio:.2f},{self.search_top_ratio:.2f})"
            f"..({self.search_right_ratio:.2f},{self.search_bottom_ratio:.2f}) "
            f"V<={self.hsv_upper_v} adaptive={self.adaptive_value_enabled} "
            f"width={self.min_line_width_ratio:.3f}..{self.max_line_width_ratio:.3f} "
            f"scale_M/F={self.band_width_scale_mid:.2f}/{self.band_width_scale_far:.2f} "
            f"coverage>={self.min_valid_row_fraction:.2f} confidence>={self.min_confidence:.2f}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "enable_motion": False,
            "roundabout_mode": "off",  # off / observe / control; lazy optional module
            # 斑马线检测：同目录另一个独立程序的接口，默认关闭。
            # 关闭时不会 import zebra_crossing_detector，巡线零额外开销。
            "enable_zebra": False,
            "zebra_decimate": 2,
            "zebra_saturation_max": 75,
            "zebra_value_min": 130,
            "zebra_minimum_stripe_count": 3,
            "zebra_maximum_stripe_count": 10,
            "zebra_confirm_frames": 2,
            "zebra_release_frames": 2,
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
            # 车上的小屏默认全屏显示，整幅画面等比缩放到窗口内，不裁切
            "debug_fullscreen": True,
            # manual : 按 wmctrl 报的窗口物理几何做画布，imshow 就是 1:1（默认，最可控）
            # backend: 原图交给 OpenCV 缩放（后端不缩放时会贴在左上角）
            "debug_scale_mode": "manual",
            "debug_window_name": "Black Line Follower (Q/Esc to stop)",
            # 三段融合：ROI 从 0.56 起（约 268 行）到画面底部，内部再均分近/中/远
            "search_top_ratio": 0.56,
            "search_bottom_ratio": 0.98,
            "search_left_ratio": 0.02,
            "search_right_ratio": 0.98,
            "hsv_lower_h": 0,
            "hsv_lower_s": 0,
            "hsv_lower_v": 0,
            "hsv_upper_h": 180,
            "hsv_upper_s": 255,
            "hsv_upper_v": 70,
            "adaptive_value_enabled": True,
            "adaptive_block_size": 51,
            "adaptive_c": 12.0,
            "hsv_morphology_kernel": 3,
            "erode_iterations": 1,
            "dilate_iterations": 1,
            "min_line_width_ratio": 0.011,
            "max_line_width_ratio": 0.020,
            "expected_line_width_ratio": 0.015,
            "width_tolerance_ratio": 0.020,
            "max_row_center_jump_ratio": 0.20,
            "min_valid_row_fraction": 0.35,
            "min_confidence": 0.45,
            "edge_recovery_min_row_fraction": 0.25,
            "edge_recovery_min_confidence": 0.45,
            "edge_recovery_error_threshold": 0.70,
            "target_offset_pixels": 0.0,
            # 近/中/远三段融合权重（近段贴着车轮，权重最大）
            "band_weight_near": 0.55,
            "band_weight_mid": 0.30,
            "band_weight_far": 0.15,
            # 从 ROI 底部量起的占比：near 占最底 25%，mid 到 60%，其余给 far
            "band_split_near": 0.25,
            "band_split_mid": 0.60,
            # 远段期望宽度按透视缩小：该段的 min/max/expected/tolerance 同乘此系数
            "band_width_scale_mid": 0.72,
            "band_width_scale_far": 0.52,
            # 中/远段中心必须落在近段附近（占画面宽度的比例），否则判为锁错目标
            "band_gate_mid": 0.12,
            "band_gate_far": 0.22,
            # 段内最宽暗色 run 超过本段 max_width 的这个倍数，就当成横向段（急弯/横缝）
            "sharp_curve_width_ratio": 2.5,
            # 横向段盖住上一帧中心多少范围（画面宽比例）才算"同一条线在拐弯"
            "corner_gate_ratio": 0.18,
            # 近段丢失就不许动，避免只靠远处 15% 的趋势瞎打方向
            "band_require_near": True,
            # 单帧失效时保持上一帧指令的帧数，避免"抖一下就掉到 0"
            "lost_hold_frames": 10,
            # 调试图叠加 40 行一条的行号网格，用来量"画面底部对应地面多远"
            "debug_row_grid": False,
            "forward_speed": 0.20,
            "minimum_forward_speed": 0.20,
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
        names = [parameter.name for parameter in self._parameters.values()]
        for name in names:
            setattr(self, name, self.get_parameter(name).value)
        self.enable_motion = bool(self.enable_motion)
        self.enable_zebra = bool(self.enable_zebra)
        self.zebra_decimate = max(1, int(self.zebra_decimate))
        self.zebra_saturation_max = int(self.zebra_saturation_max)
        self.zebra_value_min = int(self.zebra_value_min)
        self.zebra_minimum_stripe_count = int(self.zebra_minimum_stripe_count)
        self.zebra_maximum_stripe_count = int(self.zebra_maximum_stripe_count)
        self.zebra_confirm_frames = max(1, int(self.zebra_confirm_frames))
        self.zebra_release_frames = max(1, int(self.zebra_release_frames))
        self.camera_backend = str(self.camera_backend).lower()
        self.camera_id = int(self.camera_id)
        self.camera_device = str(self.camera_device)
        self.image_width = int(self.image_width)
        self.image_height = int(self.image_height)
        self.capture_frame_rate = float(self.capture_frame_rate)
        self.pixel_format = str(self.pixel_format).upper()
        self.rotation_degrees = int(self.rotation_degrees)
        self.mirror_mode = str(self.mirror_mode).lower()
        self.camera_retry_seconds = float(self.camera_retry_seconds)
        self.image_timeout_seconds = float(self.image_timeout_seconds)
        self.control_rate = float(self.control_rate)
        self.show_debug = bool(self.show_debug)
        self.debug_fullscreen = bool(self.debug_fullscreen)
        self.debug_scale_mode = str(self.debug_scale_mode).lower()
        self.debug_window_name = str(self.debug_window_name)
        self.search_top_ratio = float(self.search_top_ratio)
        self.search_bottom_ratio = float(self.search_bottom_ratio)
        self.search_left_ratio = float(self.search_left_ratio)
        self.search_right_ratio = float(self.search_right_ratio)
        self.hsv_lower_h = int(self.hsv_lower_h)
        self.hsv_lower_s = int(self.hsv_lower_s)
        self.hsv_lower_v = int(self.hsv_lower_v)
        self.hsv_upper_h = int(self.hsv_upper_h)
        self.hsv_upper_s = int(self.hsv_upper_s)
        self.hsv_upper_v = int(self.hsv_upper_v)
        self.adaptive_value_enabled = bool(self.adaptive_value_enabled)
        self.adaptive_block_size = int(self.adaptive_block_size)
        self.adaptive_c = float(self.adaptive_c)
        self.hsv_morphology_kernel = int(self.hsv_morphology_kernel)
        self.erode_iterations = int(self.erode_iterations)
        self.dilate_iterations = int(self.dilate_iterations)
        self.min_line_width_ratio = float(self.min_line_width_ratio)
        self.max_line_width_ratio = float(self.max_line_width_ratio)
        self.expected_line_width_ratio = float(self.expected_line_width_ratio)
        self.width_tolerance_ratio = float(self.width_tolerance_ratio)
        self.max_row_center_jump_ratio = float(self.max_row_center_jump_ratio)
        self.min_valid_row_fraction = float(self.min_valid_row_fraction)
        self.min_confidence = float(self.min_confidence)
        self.edge_recovery_min_row_fraction = float(
            self.edge_recovery_min_row_fraction
        )
        self.edge_recovery_min_confidence = float(
            self.edge_recovery_min_confidence
        )
        self.edge_recovery_error_threshold = float(
            self.edge_recovery_error_threshold
        )
        self.target_offset_pixels = float(self.target_offset_pixels)
        self.band_weight_near = float(self.band_weight_near)
        self.band_weight_mid = float(self.band_weight_mid)
        self.band_weight_far = float(self.band_weight_far)
        self.band_split_near = float(self.band_split_near)
        self.band_split_mid = float(self.band_split_mid)
        self.band_width_scale_mid = float(self.band_width_scale_mid)
        self.band_width_scale_far = float(self.band_width_scale_far)
        self.band_gate_mid = float(self.band_gate_mid)
        self.band_gate_far = float(self.band_gate_far)
        self.sharp_curve_width_ratio = float(self.sharp_curve_width_ratio)
        self.band_require_near = bool(self.band_require_near)
        self.forward_speed = float(self.forward_speed)
        self.minimum_forward_speed = float(self.minimum_forward_speed)
        self.linear_direction_sign = float(self.linear_direction_sign)
        self.steering_kp = float(self.steering_kp)
        self.steering_kd = float(self.steering_kd)
        self.steering_sign = float(self.steering_sign)
        self.max_angular_speed = float(self.max_angular_speed)
        self.max_linear_acceleration = float(self.max_linear_acceleration)
        self.max_angular_acceleration = float(self.max_angular_acceleration)
        self.max_error_derivative = float(self.max_error_derivative)
        self.error_filter_alpha = float(self.error_filter_alpha)
        self.curve_slowdown = float(self.curve_slowdown)
        self.status_log_period = float(self.status_log_period)
        self.lost_hold_frames = max(0, int(self.lost_hold_frames))
        self.corner_gate_ratio = float(self.corner_gate_ratio)
        self.debug_row_grid = bool(self.debug_row_grid)

    def _validate_parameters(self) -> None:
        if self.camera_backend not in ("auto", "v4l2"):
            raise ValueError("camera_backend 只能是 auto 或 v4l2")
        if self.camera_id < 0 or self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("摄像头编号和图像尺寸无效")
        if self.capture_frame_rate <= 0.0 or not 10.0 <= self.control_rate <= 20.0:
            raise ValueError("采集帧率必须大于 0，control_rate 必须在 10~20 Hz")
        if len(self.pixel_format) != 4:
            raise ValueError("pixel_format 必须是四字符格式，例如 YUYV 或 MJPG")
        if self.rotation_degrees not in self.VALID_ROTATIONS:
            raise ValueError("rotation_degrees 只能是 0、90、180 或 270")
        if self.mirror_mode not in self.VALID_MIRROR_MODES:
            raise ValueError("mirror_mode 只能是 none、horizontal、vertical 或 both")
        if self.debug_scale_mode not in ("backend", "manual"):
            raise ValueError("debug_scale_mode 只能是 backend 或 manual")
        if not 0.0 <= self.search_left_ratio < self.search_right_ratio <= 1.0:
            raise ValueError("搜索区域左右比例无效")
        if not 0.0 <= self.search_top_ratio < self.search_bottom_ratio <= 1.0:
            raise ValueError("搜索区域上下比例无效")
        for name in (
            "hsv_lower_h",
            "hsv_lower_s",
            "hsv_lower_v",
            "hsv_upper_h",
            "hsv_upper_s",
            "hsv_upper_v",
        ):
            if not 0 <= getattr(self, name) <= 255:
                raise ValueError(f"{name} 必须在 0~255")
        if not (
            self.hsv_lower_h <= self.hsv_upper_h
            and self.hsv_lower_s <= self.hsv_upper_s
            and self.hsv_lower_v <= self.hsv_upper_v
        ):
            raise ValueError("HSV 下限不能高于上限")
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size 必须是大于等于 3 的奇数")
        if self.adaptive_c < 0.0:
            raise ValueError("adaptive_c 不能为负")
        if self.hsv_morphology_kernel < 1:
            raise ValueError("hsv_morphology_kernel 必须大于 0")
        if self.erode_iterations < 0 or self.dilate_iterations < 0:
            raise ValueError("腐蚀和膨胀次数不能为负")
        for name in (
            "min_line_width_ratio",
            "max_line_width_ratio",
            "expected_line_width_ratio",
            "width_tolerance_ratio",
            "max_row_center_jump_ratio",
            "min_valid_row_fraction",
            "min_confidence",
            "edge_recovery_min_row_fraction",
            "edge_recovery_min_confidence",
            "edge_recovery_error_threshold",
            "error_filter_alpha",
            "curve_slowdown",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} 必须在 (0, 1] 范围")
        if not self.min_line_width_ratio < self.max_line_width_ratio:
            raise ValueError("最小线宽必须小于最大线宽")
        if (
            self.band_weight_near < 0.0
            or self.band_weight_mid < 0.0
            or self.band_weight_far < 0.0
        ):
            raise ValueError("band_weight_* 不能为负")
        if self.band_weight_near + self.band_weight_mid + self.band_weight_far <= 0.0:
            raise ValueError("band_weight_* 之和必须大于 0")
        if not (
            0.0 < self.band_width_scale_mid <= 2.0
            and 0.0 < self.band_width_scale_far <= 2.0
        ):
            raise ValueError("band_width_scale_* 必须在 (0, 2] 范围")
        if not (
            0.0 < self.band_gate_mid <= 1.0 and 0.0 < self.band_gate_far <= 1.0
        ):
            raise ValueError("band_gate_* 必须在 (0, 1] 范围")
        if self.sharp_curve_width_ratio <= 1.0:
            raise ValueError("sharp_curve_width_ratio 必须大于 1")
        if not 0.0 < self.corner_gate_ratio <= 1.0:
            raise ValueError("corner_gate_ratio 必须在 (0, 1] 范围")
        if not 0.0 < self.band_split_near < self.band_split_mid <= 1.0:
            raise ValueError("band_split 必须满足 0 < near < mid <= 1")
        if not 0.0 <= self.minimum_forward_speed <= self.forward_speed:
            raise ValueError("minimum_forward_speed 必须在 0 和 forward_speed 之间")
        if self.linear_direction_sign not in (-1.0, 1.0):
            raise ValueError("linear_direction_sign 只能是 -1.0 或 1.0")
        if (
            self.max_angular_speed <= 0.0
            or self.max_linear_acceleration <= 0.0
            or self.max_angular_acceleration <= 0.0
            or self.status_log_period <= 0.0
        ):
            raise ValueError("速度限制和日志周期必须大于 0")
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
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f"巡线摄像头已连接: {actual_width}x{actual_height} @ {actual_fps:.1f} fps, "
            f"后端={capture.getBackendName()}"
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
                        self._camera_healthy = False
                        self._latest_frame = None
                    self._publish_zero("camera_open_failed")
                    self._stop_capture.wait(self.camera_retry_seconds)
                    continue

            ok, frame = capture.read()
            if not ok or frame is None:
                with self._frame_lock:
                    self._camera_healthy = False
                    self._latest_frame = None
                self._publish_zero("camera_read_failed")
                capture.release()
                with self._capture_lock:
                    self._capture = None
                self._stop_capture.wait(self.camera_retry_seconds)
                continue

            frame = self._transform_frame(frame)
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_time = time.monotonic()
                self._frame_sequence += 1
                self._camera_healthy = True

    def _transform_frame(self, frame: np.ndarray) -> np.ndarray:
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

    def _publish_command(self, linear_x: float, angular_z: float) -> None:
        message = Twist()
        message.linear.x = float(self.linear_direction_sign * linear_x)
        message.angular.z = float(angular_z)
        self.cmd_vel_publisher.publish(message)

    def _publish_zero(self, reason: str) -> None:
        if self._closing and not rclpy.ok():
            return
        try:
            self._publish_command(0.0, 0.0)
        except Exception as error:  # ROS may already be shutting down.
            if not self._closing:
                self.get_logger().error(f"零速度发布失败: {error}")
        self._last_command = (0.0, 0.0)
        self._last_command_valid = False
        self._last_motion_update_time = time.monotonic()
        self._last_failure_reason = reason
        self._have_previous_error = False

    @staticmethod
    def _approach(current: float, target: float, max_delta: float) -> float:
        return float(np.clip(target, current - max_delta, current + max_delta))

    def _limit_command_rate(
        self, linear: float, angular: float, now: float
    ) -> Tuple[float, float]:
        # Cap dt so a long camera startup cannot bypass the acceleration ramp.
        dt = max(
            0.0,
            min(now - self._last_motion_update_time, 2.0 / self.control_rate),
        )
        limited_linear = self._approach(
            self._last_command[0], linear, self.max_linear_acceleration * dt
        )
        limited_angular = self._approach(
            self._last_command[1], angular, self.max_angular_acceleration * dt
        )
        self._last_motion_update_time = now
        return limited_linear, limited_angular

    def _control_timer(self) -> None:
        try:
            self._run_control_cycle()
        except Exception as error:
            self.get_logger().error(f"巡线控制异常，立即停车: {error}")
            self._publish_zero("control_exception")

    def _apply_roundabout(self, frame, result, now, allowed=True):
        """Optional visual branch selector; off returns the original object."""
        mode = str(self.get_parameter("roundabout_mode").value).lower()
        if mode not in ("off", "observe", "control"):
            mode = "off"
        if mode != self._last_roundabout_mode:
            self._roundabout_init_error = ""
            self._last_roundabout_mode = mode
        if mode == "off":
            if self._roundabout is not None:
                self._roundabout.reset_mode(mode)
            self._roundabout_owns_control = False
            return result
        if self._roundabout is None:
            if self._roundabout_init_error:
                return result
            try:
                # No new import/dependency is required for baseline-only use.
                from roundabout import RoundaboutAdapter
                self._roundabout = RoundaboutAdapter(self)
            except Exception as error:
                self._roundabout_init_error = str(error)
                self.get_logger().error(f"圆环模块未加载，保留基础巡线: {error}")
                return result
        result = self._roundabout.process(frame, result, now, mode, allowed)
        self._roundabout_owns_control = self._roundabout.owns_control
        return result

    def _run_control_cycle(self) -> None:
        now = time.monotonic()
        with self._frame_lock:
            camera_healthy = self._camera_healthy
            frame_time = self._latest_frame_time
            sequence = self._frame_sequence
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if not camera_healthy or frame is None:
            self._publish_zero("camera_unavailable")
            self._log_status(now, "STOP camera unavailable")
            return
        frame_age = now - frame_time
        if frame_age > self.image_timeout_seconds:
            self._publish_zero("image_timeout")
            self._log_status(now, f"STOP image timeout {frame_age:.3f}s")
            return

        if sequence == self._last_processed_sequence:
            if self.enable_motion and self._last_command_valid:
                self._publish_command(*self._last_command)
            else:
                self._publish_zero("no_new_frame")
            return
        self._last_processed_sequence = sequence

        # 必须在巡线判定之前跑：走到斑马线前巡线会先丢线，
        # 放到后面就永远检测不到斑马线了
        self._update_zebra(frame)

        result = self.detector.detect(frame)
        result = self._apply_roundabout(frame, result, now)
        if time.monotonic()-frame_time > self.image_timeout_seconds:
            self._publish_zero("processing_image_timeout")
            self._show_debug(result, 0.0, 0.0, False)
            return
        if not result.valid:
            # 单帧失效不立刻归零：保持上一帧指令 lost_hold_frames 帧，
            # 连续丢线超过窗口才真正停车（定速下避免"抖一下就掉到 0"）。
            if (
                self.enable_motion
                and not result.inhibit_hold
                and self._last_command_valid
                and self._lost_hold > 0
            ):
                self._lost_hold -= 1
                self._publish_command(*self._last_command)
                self._show_debug(
                    result, self._last_command[0], self._last_command[1], True
                )
                self._log_status(
                    now,
                    f"HOLD {result.reason} ({self._lost_hold}) "
                    f"conf={result.confidence:.2f}",
                )
                return
            self._publish_zero(result.reason)
            self._show_debug(result, 0.0, 0.0, False)
            self._log_status(
                now, f"STOP {result.reason}, confidence={result.confidence:.2f} "
                + getattr(self.detector, "last_diagnostics", "")
            )
            return

        self._lost_hold = self.lost_hold_frames

        if self._have_previous_error:
            filtered_error = (
                self.error_filter_alpha * result.error
                + (1.0 - self.error_filter_alpha) * self._previous_error
            )
            dt = max(1e-3, now - self._previous_control_time)
            derivative = np.clip(
                (filtered_error - self._previous_error) / dt,
                -self.max_error_derivative,
                self.max_error_derivative,
            )
        else:
            filtered_error = result.error
            derivative = 0.0
            self._have_previous_error = True
        self._previous_error = float(filtered_error)
        self._previous_control_time = now

        # 单一速度控制：前进速度恒为 forward_speed，不做任何速度决策
        # （没有弯道降速、没有急弯停车）。只有丢线/相机异常/控制异常才归零。
        raw_angular = self.steering_sign * (
            self.steering_kp * filtered_error + self.steering_kd * derivative
        )
        angular = float(
            np.clip(raw_angular, -self.max_angular_speed, self.max_angular_speed)
        )
        linear = self.forward_speed
        if result.speed_limit is not None:
            linear = min(linear, result.speed_limit)
        planned_linear, planned_angular = linear, angular

        if self.enable_motion:
            command_linear, command_angular = self._limit_command_rate(
                planned_linear, planned_angular, now
            )
            if result.speed_limit is not None:
                command_linear = min(command_linear, result.speed_limit)
            self._publish_command(command_linear, command_angular)
            self._last_command = (command_linear, command_angular)
            self._last_command_valid = True
        else:
            self._publish_zero("motion_disabled")
            command_linear, command_angular = planned_linear, planned_angular

        self._show_debug(
            result, command_linear, command_angular, self.enable_motion
        )
        mode = "RUN" if self.enable_motion else "DRY-RUN"
        if self.enable_motion:
            command_text = (
                f"target v={planned_linear:.3f}, w={planned_angular:+.3f}, "
                f"cmd v={command_linear:.3f}, w={command_angular:+.3f}"
            )
        else:
            command_text = (
                f"planned v={planned_linear:.3f}, w={planned_angular:+.3f}, actual=0"
            )
        self._log_status(
            now,
            f"{mode} confidence={result.confidence:.2f}, error={result.error:+.3f}, "
            f"{command_text}",
        )

    def _show_debug(
        self,
        result: DetectionResult,
        planned_linear: float,
        planned_angular: float,
        motion_active: bool,
    ) -> None:
        if not self._debug_window_ready:
            return
        debug = result.debug_frame
        if motion_active:
            state = "RUN"
            command_label = "cmd"
            color = (0, 255, 0)
        elif self.enable_motion:
            state = "ARMED / STOP"
            command_label = "cmd"
            color = (0, 165, 255)
        else:
            state = "STOP / DRY-RUN"
            command_label = "planned"
            color = (0, 0, 255)
        cv2.putText(
            debug,
            f"{state}  {result.reason}  conf={result.confidence:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            f"error={result.error:+.3f} {command_label} "
            f"v={planned_linear:.3f} w={planned_angular:+.3f}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        display = debug
        ring = getattr(self, "_roundabout", None)
        if ring is not None and ring.mode != "off" and hasattr(ring, "panel"):
            display = np.concatenate((debug, ring.panel), axis=1)
        elif getattr(self, "_roundabout_init_error", ""):
            cv2.putText(display, "RING DISABLED: module/config error; see log",
                        (12, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1, cv2.LINE_AA)
        if self.debug_fullscreen:
            if not self._debug_fullscreen_requested:
                self._setup_debug_fullscreen(display)
            if self.debug_scale_mode == "manual":
                target_size = self._manual_canvas_size()
                if target_size is not None:
                    display = self._fit_to_content(display, target_size)
        cv2.imshow(self.debug_window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self.request_exit("operator_exit")
        elif key in (ord("f"), ord("F")):
            # 运行中装好 wmctrl 后按 F 重试，不用重启节点
            self.debug_fullscreen = True
            self._debug_fullscreen_requested = False
            self.get_logger().info("按 F：重新尝试把调试窗口切全屏")

    def _setup_debug_fullscreen(self, first_frame: np.ndarray) -> None:
        """把调试窗口铺满屏幕，三级兜底，并把结果写进日志。

        1. OpenCV 的 WND_PROP_FULLSCREEN（GTK/Win32 有效，Qt 后端常常静默忽略）；
        2. wmctrl / xdotool 走窗口管理器协议请求全屏；
        3. wmctrl 请求最大化（窗口管理器不认全屏时，几乎总能铺满）；
        4. 没有窗口管理器（裸 X，没有标题栏）时直接改成屏幕尺寸。
        结果日志能直接看出卡在哪一步；运行中按 F 可以重试。
        """
        self._debug_fullscreen_requested = True
        # 全屏属性要求窗口已经显示过，所以先画一帧
        cv2.imshow(self.debug_window_name, first_frame)
        cv2.waitKey(1)

        property_ok = True
        try:
            cv2.setWindowProperty(
                self.debug_window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        except cv2.error as error:
            property_ok = False
            self.get_logger().warning(f"OpenCV 全屏属性不可用: {error}")

        if self._wait_for_fullscreen():
            self._report_fullscreen(
                "OpenCV 全屏属性" if property_ok else "窗口管理器"
            )
            return

        external = self._request_fullscreen_external()
        if external and self._wait_for_fullscreen():
            self._report_fullscreen(external)
            return

        # 有些窗口管理器不认全屏但认最大化；桌面面板会留一条边，画面仍完整
        maximized = self._request_maximize_external()
        if maximized and self._wait_for_fullscreen():
            self._report_fullscreen(maximized)
            return

        if not self._has_window_manager():
            screen = self._query_screen_size()
            if screen is not None:
                cv2.resizeWindow(self.debug_window_name, screen[0], screen[1])
                cv2.moveWindow(self.debug_window_name, 0, 0)
                self._report_fullscreen("无窗口管理器，按屏幕尺寸铺满")
                return

        screen = self._query_screen_size()
        window = self._debug_content_size()
        screen_text = f"{screen[0]}x{screen[1]}" if screen else "未知"
        window_text = f"{window[0]}x{window[1]}" if window else "未知"
        missing = [
            name for name in ("wmctrl", "xdotool") if shutil.which(name) is None
        ]
        if missing:
            hint = (
                f"缺少 {'/'.join(missing)}，装好后在窗口里按 F 键即可重试："
                "sudo apt install wmctrl"
            )
        else:
            hint = (
                "wmctrl/xdotool 已安装但窗口尺寸没变，按 F 可再试一次；"
                "若 wmctrl -lG 看到窗口已经是屏幕尺寸，则是判定阈值问题，把这两行日志发出来"
            )
        self.get_logger().warning(
            f"调试窗口没能全屏（内容区 {window_text}，屏幕 {screen_text}，"
            f"WM 几何 {self._geometry_text()}），"
            f"按普通窗口显示。{hint}"
        )

    def _wait_for_fullscreen(self, timeout: float = 0.5) -> bool:
        """等窗口管理器把窗口调到屏幕大小；无法判定时立即返回 False。"""
        deadline = time.monotonic() + timeout
        while True:
            covered = self._window_covers_screen()
            if covered:
                return True
            if covered is None or time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _window_covers_screen(self) -> Optional[bool]:
        """窗口是否已经铺满屏幕；无法判定时返回 None。

        以 wmctrl -lG 的物理几何为准：某些 OpenCV 后端（GTK）的
        getWindowImageRect() 会返回不可靠的尺寸（实测 800x480 的窗口报 320x240），
        用它判断会出现"明明全屏了却说没全屏"。
        """
        screen = self._query_screen_size()
        if screen is None:
            return None
        geometry = self._window_geometry_from_wmctrl()
        if geometry is not None:
            width, height = geometry[2], geometry[3]
            return width >= screen[0] * 0.92 and height >= screen[1] * 0.92
        size = self._debug_content_size()
        if size is None:
            return None
        return size[0] >= screen[0] * 0.92 and size[1] >= screen[1] * 0.92

    def _query_screen_size(self) -> Optional[Tuple[int, int]]:
        """用 xdpyinfo 读屏幕分辨率，只读一次并缓存。"""
        if self._debug_screen_size is not None:
            return self._debug_screen_size
        if shutil.which("xdpyinfo") is None:
            return None
        try:
            completed = subprocess.run(
                ["xdpyinfo"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        match = re.search(r"dimensions:\s+(\d+)x(\d+)", completed.stdout or "")
        if match is None:
            return None
        self._debug_screen_size = (int(match.group(1)), int(match.group(2)))
        return self._debug_screen_size

    @staticmethod
    def _run_quiet(command: List[str]) -> bool:
        try:
            completed = subprocess.run(
                command, capture_output=True, timeout=3, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0

    def _request_fullscreen_external(self) -> str:
        """通过窗口管理器协议请求全屏，返回真正生效的工具名。"""
        if shutil.which("wmctrl") and self._run_quiet(
            ["wmctrl", "-r", self.debug_window_name, "-b", "add,fullscreen"]
        ):
            return "wmctrl 全屏"
        if shutil.which("xdotool") and self._run_quiet(
            [
                "xdotool",
                "search",
                "--name",
                "Black Line Follower",
                "windowstate",
                "--add",
                "FULLSCREEN",
            ]
        ):
            return "xdotool 全屏"
        return ""

    def _request_maximize_external(self) -> str:
        """退一步请求最大化：几乎总能铺满，只是可能留出桌面面板那条边。"""
        if shutil.which("wmctrl") and self._run_quiet(
            [
                "wmctrl",
                "-r",
                self.debug_window_name,
                "-b",
                "add,maximized_vert,maximized_horz",
            ]
        ):
            return "wmctrl 最大化"
        return ""

    def _has_window_manager(self) -> bool:
        """有没有 EWMH 窗口管理器；查不到就当作有，避免误改窗口尺寸。"""
        if shutil.which("xprop") is None:
            return True
        try:
            completed = subprocess.run(
                ["xprop", "-root", "_NET_SUPPORTING_WM_CHECK"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return True
        return "window id" in (completed.stdout or "")

    def _window_geometry_from_wmctrl(self) -> Optional[Tuple[int, int, int, int]]:
        """用 wmctrl -lG 读窗口的真实（物理像素）几何，返回 (x, y, w, h)。"""
        if shutil.which("wmctrl") is None:
            return None
        try:
            completed = subprocess.run(
                ["wmctrl", "-lG"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for line in (completed.stdout or "").splitlines():
            if "Black Line Follower" not in line:
                continue
            fields = line.split(None, 7)
            if len(fields) < 7:
                continue
            _wid, _desktop, x, y, width, height = fields[:6]
            try:
                return (int(x), int(y), int(width), int(height))
            except ValueError:
                return None
        return None

    def _geometry_text(self) -> str:
        geometry = self._window_geometry_from_wmctrl()
        if geometry is None:
            return "未知"
        x, y, width, height = geometry
        return f"{x},{y} {width}x{height}"

    def _manual_canvas_size(self) -> Optional[Tuple[int, int]]:
        """manual 模式的画布尺寸：优先用窗口管理器报的物理几何。

        画布做成和窗口一样大的物理像素，imshow 时就是 1:1，
        不论后端会不会缩放、系统有没有 DPR 缩放，都能正好铺满窗口。
        wmctrl 是外部进程，不能每帧都调用，所以缓存起来每 30 帧刷新一次。
        """
        self._debug_canvas_tick += 1
        if (
            self._debug_canvas_target is not None
            and self._debug_canvas_tick % 30 != 0
        ):
            return self._debug_canvas_target
        geometry = self._window_geometry_from_wmctrl()
        if geometry is not None:
            self._debug_canvas_target = (geometry[2], geometry[3])
        else:
            size = self._debug_content_size()
            if size is not None:
                self._debug_canvas_target = size
        return self._debug_canvas_target

    def _report_fullscreen(self, method: str) -> None:
        self._debug_canvas_target = None  # 窗口尺寸刚变过，缓存的画布尺寸作废
        canvas_text = ""
        if self.debug_scale_mode == "manual":
            canvas_size = self._manual_canvas_size()
            canvas_text = f"，画布={canvas_size[0]}x{canvas_size[1]}" if canvas_size else ""
        self.get_logger().info(
            f"调试窗口已全屏（{method}），WM 几何={self._geometry_text()}{canvas_text}"
        )

    def _debug_content_size(self) -> Optional[Tuple[int, int]]:
        """返回窗口当前可显示区域的 (宽, 高)，失败时返回 None。"""
        try:
            _, _, width, height = cv2.getWindowImageRect(self.debug_window_name)
        except cv2.error:
            return None
        if width <= 0 or height <= 0:
            return None
        return int(width), int(height)

    def _fit_to_content(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """等比缩放整幅画面到 size 之内并在两侧补黑边，保证没有任何裁切。

        只放大不缩小：某些系统上 getWindowImageRect() 报的是逻辑像素（DPR≈2），
        比真实画布小一半，按它缩图会出现"图像缩在左上角、周围一片白"。
        """
        target_width, target_height = size
        image_height, image_width = image.shape[:2]
        if image_width == target_width and image_height == target_height:
            return image
        if target_width < image_width or target_height < image_height:
            return image

        scale = min(target_width / image_width, target_height / image_height)
        scaled_width = max(1, int(round(image_width * scale)))
        scaled_height = max(1, int(round(image_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(
            image, (scaled_width, scaled_height), interpolation=interpolation
        )
        if scaled_width == target_width and scaled_height == target_height:
            return scaled

        canvas = self._debug_canvas
        if (
            canvas is None
            or canvas.shape[0] != target_height
            or canvas.shape[1] != target_width
        ):
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            self._debug_canvas = canvas
        offset_x = (target_width - scaled_width) // 2
        offset_y = (target_height - scaled_height) // 2
        canvas[
            offset_y : offset_y + scaled_height, offset_x : offset_x + scaled_width
        ] = scaled
        return canvas

    def _setup_zebra(self) -> None:
        """按需加载斑马线模块；未启用时保持零开销。

        巡线与斑马线是两个独立程序，默认不同时使用：
        enable_zebra=false（默认）时根本不 import zebra_crossing_detector，
        line_follower.py 可以单独部署、单独运行；需要时把 enable_zebra 设为 true。
        """
        if not self.enable_zebra:
            self.get_logger().info(
                "斑马线检测未启用（enable_zebra=false），只做巡线"
            )
            return

        try:
            import zebra_crossing_detector as zebra
        except ImportError as error:
            self.enable_zebra = False
            self.get_logger().error(
                f"斑马线模块不可用（{error}），已自动关闭斑马线检测，只做巡线"
            )
            return

        self._zebra_module = zebra
        self._zebra_roi_config = zebra.RoiConfig()
        self._zebra_perspective_config = zebra.PerspectiveConfig()
        self._zebra_detected_publisher = self.create_publisher(
            Bool, "/zebra_crossing/detected", 1
        )
        self._zebra_count_publisher = self.create_publisher(
            UInt8, "/zebra_crossing/stripe_count", 1
        )
        self.get_logger().info(
            "斑马线检测已启用（与巡线同帧运行，只发 /zebra_crossing/* 话题，不参与控制）"
        )

    def _update_zebra(self, frame: np.ndarray) -> None:
        """在同一帧上跑斑马线检测，结果直接发话题，不经过图像传输。"""
        zebra = self._zebra_module
        if zebra is None:
            return

        self._zebra_cycle += 1
        if self.zebra_decimate > 1 and self._zebra_cycle % self.zebra_decimate != 0:
            return

        roi, _ = zebra.extract_roi(frame, self._zebra_roi_config)
        bird_view, _ = zebra.warp_perspective(roi, self._zebra_perspective_config)
        track_mask, _ = zebra.extract_white_track(
            bird_view,
            saturation_max=self.zebra_saturation_max,
            value_min=self.zebra_value_min,
        )
        result = zebra.count_black_stripes(
            track_mask,
            minimum_stripe_count=self.zebra_minimum_stripe_count,
            maximum_stripe_count=self.zebra_maximum_stripe_count,
        )

        if result.zebra_exists:
            self._zebra_hit += 1
            self._zebra_miss = 0
        else:
            self._zebra_miss += 1
            self._zebra_hit = 0

        if not self._zebra_confirmed and self._zebra_hit >= self.zebra_confirm_frames:
            self._zebra_confirmed = True
            self.get_logger().info(
                f"斑马线确认: stripes={result.stripe_count}, "
                f"angle={result.dominant_angle_deg:+.1f}"
            )
        elif self._zebra_confirmed and self._zebra_miss >= self.zebra_release_frames:
            self._zebra_confirmed = False
            self.get_logger().info("斑马线释放")

        self._zebra_detected_publisher.publish(Bool(data=self._zebra_confirmed))
        self._zebra_count_publisher.publish(
            UInt8(data=min(int(result.stripe_count), 255))
        )

    def _log_status(self, now: float, message: str) -> None:
        if now - self._last_status_log >= self.status_log_period:
            self.get_logger().info(message)
            self._last_status_log = now

    @property
    def exit_requested(self) -> bool:
        return self._exit_requested

    def request_exit(self, reason: str) -> None:
        self._publish_zero(reason)
        self._exit_requested = True

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
                self._publish_command(0.0, 0.0)
                time.sleep(0.02)
        if self._debug_window_ready:
            cv2.destroyWindow(self.debug_window_name)
        self.get_logger().info("巡线节点退出，已重复发布零速度")


def main(args=None) -> None:
    node = None
    exit_signal = threading.Event()

    def handle_exit_signal(signum, _frame) -> None:
        exit_signal.set()
        if node is not None:
            node.request_exit(f"signal_{signum}")

    previous_sigint = signal.signal(signal.SIGINT, handle_exit_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_exit_signal)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = LineFollower()
        while rclpy.ok() and not node.exit_requested and not exit_signal.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (RuntimeError, ValueError) as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f"巡线节点启动失败: {error}")
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
