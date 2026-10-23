#!/usr/bin/env python3
"""Synthetic safety tests for dual-white-line detection."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from double_line_follower import DoubleLineFollower, DualWhiteLineDetector  # noqa: E402


def make_detector() -> DualWhiteLineDetector:
    return DualWhiteLineDetector(
        search_top_ratio=0.65,
        search_bottom_ratio=0.86,
        search_left_ratio=0.02,
        search_right_ratio=0.98,
        white_saturation_max=75,
        white_value_min=120,
        adaptive_bright_enabled=True,
        adaptive_block_size=51,
        adaptive_c=-12.0,
        min_boundary_lane_contrast=8.0,
        morphology_kernel=3,
        min_line_width_ratio=0.008,
        max_line_width_ratio=0.10,
        expected_line_width_ratio=0.025,
        width_tolerance_ratio=0.035,
        min_lane_width_ratio=0.45,
        max_lane_width_ratio=0.85,
        expected_lane_width_ratio=0.64,
        lane_width_tolerance_ratio=0.22,
        max_center_jump_ratio=0.02,
        min_valid_row_fraction=0.35,
        min_confidence=0.52,
        target_offset_pixels=0.0,
    )


def dark_floor(value: int = 55) -> np.ndarray:
    frame = np.full((480, 640, 3), value, dtype=np.uint8)
    noise = np.random.default_rng(11).normal(0, 2, frame.shape[:2])
    for channel in range(3):
        frame[:, :, channel] = np.clip(
            frame[:, :, channel].astype(np.float32) + noise, 0, 255
        ).astype(np.uint8)
    return frame


def draw_boundary(
    frame: np.ndarray, points, thickness: int = 16, value: int = 235
) -> None:
    cv2.polylines(
        frame,
        [np.asarray(points, dtype=np.int32)],
        False,
        (value, value, value),
        thickness,
        cv2.LINE_AA,
    )


class DoubleLineDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = make_detector()

    def test_straight_pair_follows_midpoint(self) -> None:
        frame = dark_floor()
        draw_boundary(frame, [(180, 479), (180, 250)])
        draw_boundary(frame, [(460, 479), (460, 250)])
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertGreaterEqual(result.confidence, 0.55)
        self.assertAlmostEqual(result.error, 0.0, delta=0.04)
        self.assertAlmostEqual(result.lane_width_pixels, 280.0, delta=8.0)

    def test_pair_shifted_left_has_negative_error(self) -> None:
        frame = dark_floor()
        draw_boundary(frame, [(80, 479), (120, 250)])
        draw_boundary(frame, [(360, 479), (400, 250)])
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertLess(result.error, -0.15)

    def test_pair_shifted_right_has_positive_error(self) -> None:
        frame = dark_floor()
        draw_boundary(frame, [(280, 479), (240, 250)])
        draw_boundary(frame, [(560, 479), (520, 250)])
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(result.error, 0.15)

    def test_matching_curves_are_continuous(self) -> None:
        frame = dark_floor()
        draw_boundary(frame, [(170, 479), (175, 420), (205, 350), (260, 250)])
        draw_boundary(frame, [(450, 479), (455, 420), (485, 350), (540, 250)])
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(result.error, 0.02)

    def test_horizontal_bright_crossing_is_skipped(self) -> None:
        frame = dark_floor()
        draw_boundary(frame, [(180, 479), (180, 250)])
        draw_boundary(frame, [(460, 479), (460, 250)])
        cv2.line(frame, (0, 375), (639, 375), (245, 245, 245), 12)
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.error, 0.0, delta=0.05)

    def test_only_one_white_line_is_invalid(self) -> None:
        frame = dark_floor()
        draw_boundary(frame, [(210, 479), (230, 250)])
        result = self.detector.detect(frame)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "only_one_white_line")

    def test_no_line_is_invalid(self) -> None:
        result = self.detector.detect(dark_floor())
        self.assertFalse(result.valid)
        self.assertEqual(result.confidence, 0.0)

    def test_wrong_lane_spacing_is_invalid(self) -> None:
        frame = dark_floor()
        draw_boundary(frame, [(280, 479), (280, 250)])
        draw_boundary(frame, [(350, 479), (350, 250)])
        result = self.detector.detect(frame)
        self.assertFalse(result.valid)

    def test_large_bright_region_is_not_a_lane_pair(self) -> None:
        frame = dark_floor()
        frame[300:479, 170:470] = 235
        result = self.detector.detect(frame)
        self.assertFalse(result.valid)

    def test_adaptive_threshold_finds_dim_white_pair(self) -> None:
        frame = dark_floor()
        draw_boundary(frame, [(180, 479), (180, 250)], value=135)
        draw_boundary(frame, [(460, 479), (460, 250)], value=135)
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)

    def test_normal_command_ramp_limits_each_step(self) -> None:
        self.assertAlmostEqual(DoubleLineFollower._approach(0.0, 0.20, 0.02), 0.02)
        self.assertAlmostEqual(DoubleLineFollower._approach(0.10, 0.20, 0.02), 0.12)
        self.assertAlmostEqual(DoubleLineFollower._approach(0.20, 0.0, 0.02), 0.18)
        self.assertAlmostEqual(DoubleLineFollower._approach(0.0, -0.40, 0.08), -0.08)


if __name__ == "__main__":
    unittest.main()
