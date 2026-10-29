"""Synthetic images for the gap-only step detector (no ROS required)."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from step_detector import StepConfig, StepTracker, detect_step


class GapOnlyStepTest(unittest.TestCase):
    def setUp(self):
        self.cfg = StepConfig(top_ratio=0.0, bottom_ratio=1.0,
                              left_ratio=0.0, right_ratio=1.0,
                              center_left_ratio=0.3, center_right_ratio=0.7,
                              gap_top_ratio=0.0, gap_bottom_ratio=0.8,
                              min_gap_rows=10)

    def line_frame(self):
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        frame[:, 47:53] = 0
        return frame

    def test_missing_near_line_and_non_platform_color_still_detect(self):
        # Saturated red fails the old shallow/wood surface tests; no near/far line.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 2] = 200
        result = detect_step(frame, self.cfg)
        self.assertTrue(result.detected)
        self.assertEqual(result.gap_rows, 80)
        self.assertTrue(StepTracker(self.cfg).update(result, 1.0))

    def test_exact_threshold_and_one_row_short(self):
        for rows in (9, 10, 11):
            with self.subTest(rows=rows):
                frame = self.line_frame()
                frame[20:20 + rows, 47:53] = 200
                result = detect_step(frame, self.cfg)
                self.assertEqual(result.gap_rows, rows)
                self.assertEqual(result.detected, rows >= 10)

    def test_separated_short_gaps_do_not_add_up(self):
        frame = self.line_frame()
        frame[20:26, 47:53] = 200
        frame[30:36, 47:53] = 200
        result = detect_step(frame, self.cfg)
        self.assertFalse(result.detected)
        self.assertEqual(result.gap_rows, 6)

    def test_only_search_box_counts(self):
        frame = self.line_frame()
        frame[80:, :] = 200  # No near line, but outside vertical search box.
        self.assertFalse(detect_step(frame, self.cfg).detected)
        frame[:, 47:53] = 200
        frame[:, 10:20] = 0  # Black line outside horizontal box does not block.
        self.assertTrue(detect_step(frame, self.cfg).detected)

    def test_black_pixel_count_threshold(self):
        for count in (4, 5):
            with self.subTest(count=count):
                frame = np.full((100, 100, 3), 200, dtype=np.uint8)
                frame[:, 40:40 + count] = 0
                self.assertEqual(detect_step(frame, self.cfg).detected, count < 5)
        self.assertFalse(detect_step(frame, replace(
            self.cfg, min_dark_pixels_per_row=1)).detected)

    def test_optional_confirmation_resets_on_miss_and_timeout(self):
        tracker = StepTracker(replace(self.cfg, track_confirm_frames=2))
        hit = detect_step(np.full((100, 100, 3), 200, dtype=np.uint8), self.cfg)
        miss = detect_step(self.line_frame(), self.cfg)
        self.assertFalse(tracker.update(hit, 1.0))
        self.assertTrue(tracker.update(hit, 1.1))
        self.assertFalse(tracker.update(miss, 1.2))
        self.assertFalse(tracker.update(hit, 1.3))
        self.assertFalse(tracker.update(hit, 2.0))
        # Geometry changes cannot veto a valid sequence anymore.
        moved = replace(hit, edge_row=99, gap_center_x=0.9)
        self.assertTrue(tracker.update(moved, 2.1))


if __name__ == "__main__":
    unittest.main()
