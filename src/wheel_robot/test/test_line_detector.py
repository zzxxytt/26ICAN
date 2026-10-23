#!/usr/bin/env python3
"""Synthetic safety tests for black-tape detection."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from line_follower import HsvStripLineDetector, LineFollower  # noqa: E402


def make_detector() -> HsvStripLineDetector:
    return HsvStripLineDetector(
        search_top_ratio=0.78,
        search_bottom_ratio=0.98,
        search_left_ratio=0.02,
        search_right_ratio=0.98,
        hsv_lower=(0, 0, 0),
        hsv_upper=(180, 255, 70),
        adaptive_value_enabled=True,
        adaptive_block_size=51,
        adaptive_c=12.0,
        morphology_kernel=3,
        erode_iterations=1,
        dilate_iterations=1,
        min_line_width_ratio=0.011,
        max_line_width_ratio=0.08,
        expected_line_width_ratio=0.015,
        width_tolerance_ratio=0.020,
        max_row_center_jump_ratio=0.06,
        min_valid_row_fraction=0.45,
        min_confidence=0.55,
        edge_recovery_min_row_fraction=0.25,
        edge_recovery_min_confidence=0.45,
        edge_recovery_error_threshold=0.70,
        target_offset_pixels=0.0,
    )


def light_floor() -> np.ndarray:
    frame = np.full((480, 640, 3), 190, dtype=np.uint8)
    noise = np.random.default_rng(7).normal(0, 3, frame.shape[:2])
    for channel in range(3):
        frame[:, :, channel] = np.clip(
            frame[:, :, channel].astype(np.float32) + noise, 0, 255
        ).astype(np.uint8)
    return frame


def draw_tape(
    frame: np.ndarray, points, thickness: int = 38, value: int = 15
) -> None:
    cv2.polylines(
        frame,
        [np.asarray(points, dtype=np.int32)],
        False,
        (value, value, value),
        thickness,
        cv2.LINE_AA,
    )


class LineDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = make_detector()

    def test_straight_tape_with_crossing_grout(self) -> None:
        frame = light_floor()
        cv2.line(frame, (0, 330), (639, 330), (45, 45, 45), 5)
        cv2.line(frame, (585, 0), (585, 479), (45, 45, 45), 5)
        draw_tape(frame, [(320, 479), (320, 190)])
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertGreaterEqual(result.confidence, 0.55)
        self.assertAlmostEqual(result.error, 0.0, delta=0.06)

    def test_left_curve_has_negative_error(self) -> None:
        frame = light_floor()
        draw_tape(frame, [(320, 479), (305, 400), (260, 310), (185, 190)])
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertLess(result.error, -0.02)

    def test_right_curve_has_positive_error(self) -> None:
        frame = light_floor()
        draw_tape(frame, [(320, 479), (340, 400), (385, 310), (455, 190)])
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(result.error, 0.02)

    def test_thin_vertical_grout_is_not_a_line(self) -> None:
        frame = light_floor()
        cv2.line(frame, (500, 0), (500, 479), (30, 30, 30), 3)
        result = self.detector.detect(frame)
        self.assertFalse(result.valid)

    def test_horizontal_grout_is_not_a_line(self) -> None:
        frame = light_floor()
        cv2.line(frame, (0, 350), (639, 350), (30, 30, 30), 6)
        result = self.detector.detect(frame)
        self.assertFalse(result.valid)

    def test_lost_line_is_invalid(self) -> None:
        result = self.detector.detect(light_floor())
        self.assertFalse(result.valid)
        self.assertEqual(result.confidence, 0.0)

    def test_adaptive_threshold_finds_bright_reflective_tape(self) -> None:
        frame = light_floor()
        draw_tape(frame, [(320, 479), (320, 375)], thickness=10, value=105)
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.error, 0.0, delta=0.05)

    def test_reflective_curve_with_only_half_valid_rows(self) -> None:
        frame = light_floor()
        draw_tape(frame, [(320, 479), (325, 450), (385, 422)], thickness=10)
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(result.error, 0.0)

    def test_curve_reconnects_after_reflective_gap(self) -> None:
        frame = light_floor()
        draw_tape(frame, [(300, 479), (310, 450)], thickness=10)
        draw_tape(frame, [(390, 430), (470, 390)], thickness=10)
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(result.error, 0.0)

    def test_partial_line_at_edge_enters_recovery(self) -> None:
        frame = light_floor()
        draw_tape(frame, [(20, 479), (20, 438)], thickness=10)
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.reason, "edge_recovery")
        self.assertLess(result.error, -0.70)

    def test_partial_line_near_center_does_not_enter_recovery(self) -> None:
        frame = light_floor()
        draw_tape(frame, [(320, 479), (320, 438)], thickness=10)
        result = self.detector.detect(frame)
        self.assertFalse(result.valid)

    def test_large_dark_shadow_is_rejected(self) -> None:
        frame = light_floor()
        frame[210:479, 350:600] = 65
        result = self.detector.detect(frame)
        self.assertFalse(result.valid)

    def test_normal_command_ramp_limits_each_step(self) -> None:
        self.assertAlmostEqual(LineFollower._approach(0.0, 0.20, 0.02), 0.02)
        self.assertAlmostEqual(LineFollower._approach(0.10, 0.20, 0.02), 0.12)
        self.assertAlmostEqual(LineFollower._approach(0.20, 0.0, 0.02), 0.18)
        self.assertAlmostEqual(LineFollower._approach(0.0, -0.40, 0.08), -0.08)


if __name__ == "__main__":
    unittest.main()
