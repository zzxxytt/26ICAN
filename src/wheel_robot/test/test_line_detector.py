#!/usr/bin/env python3
"""Synthetic safety tests for black-tape detection."""

from pathlib import Path
import sys
import unittest
import math
import time
import threading
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from line_follower import HsvStripLineDetector, LineFollower, DetectionResult  # noqa: E402
from roundabout import (Config, Path as RingPath, Observation, Heading, HeadingTracker,
                        Machine, State, Vision, RoundaboutAdapter, quaternion_angles)  # noqa: E402


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


class FieldLineRegressionTest(unittest.TestCase):
    """Straight tape can exceed the row-width window, not just a tight turn."""
    def setUp(self):
        self.detector = make_detector()
        self.detector.search_top_ratio = 0.56
        self.detector.min_line_width_ratio = 0.020
        self.detector.max_line_width_ratio = 0.060
        self.detector.expected_line_width_ratio = 0.032
        self.detector.max_row_center_jump_ratio = 0.05
        self.detector.min_valid_row_fraction = 0.35
        self.detector.min_confidence = 0.45
        self.detector.band_split_near = 0.20
        self.detector.band_weight_near = 0.30
        self.detector.band_weight_mid = 0.45
        self.detector.band_weight_far = 0.25
        self.detector.band_gate_far = 0.27
        self.seed = np.full((480, 640, 3), 220, np.uint8)
        cv2.rectangle(self.seed, (310, 0), (329, 479), (10, 10, 10), -1)

    def wide_frame(self):
        frame = np.full_like(self.seed, 220)
        cv2.rectangle(frame, (299, 0), (340, 479), (10, 10, 10), -1)
        return frame

    def test_straight_tape_width_increase_keeps_tracking(self):
        self.assertTrue(self.detector.detect(self.seed).valid)
        result = self.detector.detect(self.wide_frame())
        self.assertTrue(result.valid, result.reason)
        self.assertLess(abs(result.error), 0.02)
        self.assertIn("mask=", self.detector.last_diagnostics)
        self.assertIn("wide=42", self.detector.last_diagnostics)

    def test_partial_width_transition_does_not_drop_row_coverage(self):
        self.assertTrue(self.detector.detect(self.seed).valid)
        frame = self.wide_frame()
        frame[460:] = self.seed[460:]
        result = self.detector.detect(frame)
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(result.confidence, 0.75)

    def test_blank_frame_remains_invalid_after_tracked_tape(self):
        self.detector.detect(self.seed)
        result = self.detector.detect(np.full_like(self.seed, 220))
        self.assertFalse(result.valid)
        self.assertIn("mask=0 picked=0", self.detector.last_diagnostics)

    def test_unassociated_wide_region_does_not_bootstrap_tracking(self):
        result = self.detector.detect(self.wide_frame())
        self.assertFalse(result.valid)
        self.assertIn("picked=0", self.detector.last_diagnostics)


class FieldWidthCalibrationTest(unittest.TestCase):
    """Exercise the shipped ICN line parameters without a warm-up frame."""

    def setUp(self):
        import inspect
        import yaml
        config_dir = SCRIPTS_DIR.parent / "config"
        params = {}
        # Match icn.launch.py's parameter-file ordering.
        for name in ("line_follower.yaml", "icn.yaml"):
            params.update(yaml.safe_load((config_dir / name).read_text(
                encoding="utf-8"))["line_follower"]["ros__parameters"])
        kwargs = {}
        for name in inspect.signature(HsvStripLineDetector).parameters:
            if name in ("hsv_lower", "hsv_upper"):
                kwargs[name] = tuple(params[name + "_" + c] for c in "hsv")
            elif name == "morphology_kernel":
                kwargs[name] = params["hsv_morphology_kernel"]
            elif name in params:
                kwargs[name] = params[name]
        self.detector = HsvStripLineDetector(**kwargs)

    @staticmethod
    def perspective_tape(shift=0):
        frame = np.full((480, 640, 3), 220, np.uint8)
        # Approximate the screenshot's central tape, not its painted HUD.
        for y in range(480):
            width = int(np.interp(y, [268, 349, 430, 469], [29, 35, 42, 46]))
            center = int(np.interp(y, [268, 469], [370, 384])) + shift
            frame[y, center - width // 2:center - width // 2 + width] = 15
        return frame

    def test_field_widths_acquire_on_first_frame_without_corner(self):
        result = self.detector.detect(self.perspective_tape())
        self.assertTrue(result.valid, self.detector.last_diagnostics)
        self.assertFalse(result.corner)
        self.assertGreater(result.confidence, 0.80)
        self.assertGreater(result.error, 0.13)
        self.assertLess(result.error, 0.23)

    def test_old_width_window_reproduces_cold_start_loss(self):
        self.detector.max_line_width_ratio = 0.060
        self.detector.expected_line_width_ratio = 0.032
        self.detector.width_tolerance_ratio = 0.020
        self.detector.band_width_scale_mid = 0.72
        self.detector.band_width_scale_far = 0.52
        self.assertFalse(self.detector.detect(self.perspective_tape()).valid)

    def test_left_right_tape_has_correct_error_sign(self):
        for shift, sign in ((-120, -1), (0, 1)):
            with self.subTest(shift=shift):
                self.detector._last_center_x = None
                result = self.detector.detect(self.perspective_tape(shift))
                self.assertTrue(result.valid)
                self.assertGreater(sign * result.error, 0.10)

    def test_previous_narrow_straight_still_acquires(self):
        frame = np.full((480, 640, 3), 220, np.uint8)
        frame[:, 310:330] = 15
        result = self.detector.detect(frame)
        self.assertTrue(result.valid)
        self.assertLess(abs(result.error), 0.02)

    def test_blank_after_tracking_stays_invalid(self):
        self.assertTrue(self.detector.detect(self.perspective_tape()).valid)
        self.assertFalse(self.detector.detect(np.full((480, 640, 3), 220, np.uint8)).valid)

    def test_grout_horizontal_bar_and_floor_edge_do_not_acquire(self):
        for kind in ("grout", "horizontal", "edge"):
            with self.subTest(kind=kind):
                self.detector._last_center_x = None
                frame = np.full((480, 640, 3), 220, np.uint8)
                if kind == "grout":
                    frame[:, 319:322] = 15
                elif kind == "horizontal":
                    frame[440:460, :] = 15
                else:
                    frame[:, :180] = 15
                self.assertFalse(self.detector.detect(frame).valid)


def ring_path(curved=True):
    return RingPath(np.array([-0.34, 0.0, 0.5] if curved else [0.0, 0.0, 0.5]), 0.9, 5, 0.9)


def ring_view(curved=True, straight=True, dashed=True):
    """Forward camera synthetic ground ROI, not a top-down full-circle image."""
    cfg = Config(adaptive=False)
    frame = np.full((480, 640, 3), 210, np.uint8)
    for is_curve in ([True] if curved else [])+([False] if straight else []):
        curve = ring_path(is_curve)
        for k in range(6):
            lo, hi = k/6+0.005, (k+1)/6-(0.055 if dashed else 0.0)
            t = np.linspace(lo, hi, 18)
            xs = (cfg.roi_left+(cfg.roi_right-cfg.roi_left)*curve.x(t))*640
            ys = (cfg.roi_bottom-t*(cfg.roi_bottom-cfg.roi_top))*480
            cv2.polylines(frame, [np.column_stack((xs, ys)).astype(np.int32)], False, (15, 15, 15), 9)
    return frame


class RoundaboutVisionTest(unittest.TestCase):
    def inspect_paths(self, paths):
        vision = Vision(Config(adaptive=False))
        with patch.object(vision, "_trace", return_value=paths):
            return vision.detect(np.full((480, 640, 3), 220, np.uint8))

    def test_reverse_bending_third_hypothesis_is_not_the_entry_arc(self):
        arc, tangent = ring_path(), ring_path(False)
        mixed = RingPath(np.array([0.32, -0.32, 0.5]), 0.99, 5, 0.9)
        obs = self.inspect_paths([mixed, tangent, arc])
        self.assertTrue(obs.junction)
        self.assertIs(obs.arc, arc)
        self.assertIs(obs.tangent, tangent)
        self.assertIn("bend", obs.entry["paths"][0]["arc_fail"])
        self.assertEqual(obs.entry["reject"], "OK")

    def test_straight_left_branch_reports_missing_curvature(self):
        left = RingPath(np.array([0.0, -0.35, 0.5]), 0.95, 4, 0.9)
        obs = self.inspect_paths([left, ring_path(False)])
        self.assertFalse(obs.junction)
        self.assertEqual(obs.entry["reject"], "NO_ARC")
        self.assertEqual(obs.entry["paths"][0]["arc_fail"], ["bend"])
        self.assertIn("heading", obs.entry["paths"][0]["tangent_fail"])

    def test_arc_dash_and_reach_failures_are_visible(self):
        arc = ring_path()
        arc.dashes = 2
        arc.points = np.array([[0.1, 0.5], [0.6, 0.38]])
        obs = self.inspect_paths([arc, ring_path(False)])
        self.assertFalse(obs.junction)
        self.assertEqual(obs.entry["paths"][0]["arc_fail"], ["dash", "reach"])

    def test_pair_near_gap_and_far_separation_have_distinct_reasons(self):
        arc = ring_path()
        arc.coefficients[2] = 0.25
        obs = self.inspect_paths([arc, ring_path(False)])
        self.assertEqual(obs.entry["reject"], "NEAR_GAP")
        arc = RingPath(np.array([-0.12, 0.0, 0.5]), 0.95, 4, 0.9)
        tangent = RingPath(np.array([0.0, 0.0, 0.47]), 0.95, 4, 0.9)
        obs = self.inspect_paths([arc, tangent])
        self.assertEqual(obs.entry["reject"], "FAR_SEPARATION")

    def test_dashed_junction_keeps_arc_and_tangent_separate(self):
        obs = Vision(Config(adaptive=False)).detect(ring_view())
        self.assertTrue(obs.junction, [(p.coefficients.tolist(), p.dashes) for p in obs.paths])
        self.assertLess(obs.arc.x(0.8), obs.tangent.x(0.8)-0.08)
        self.assertGreaterEqual(obs.arc.dashes, 3)

    def test_single_dashed_straight_is_not_an_entry(self):
        obs = Vision(Config(adaptive=False)).detect(ring_view(curved=False))
        self.assertFalse(obs.junction)
        self.assertIsNotNone(obs.tangent)

    def test_solid_curve_does_not_trigger_entry(self):
        obs = Vision(Config(adaptive=False)).detect(ring_view(straight=False, dashed=False))
        self.assertFalse(obs.junction)

    def test_blank_shadow_and_zebra_do_not_trigger(self):
        vision = Vision(Config(adaptive=False))
        for value in (0, 210):
            self.assertFalse(vision.detect(np.full((480, 640, 3), value, np.uint8)).junction)
        frame = np.full((480, 640, 3), 210, np.uint8)
        for y in range(240, 470, 30):
            cv2.rectangle(frame, (50, y), (590, y+12), (0, 0, 0), -1)
        self.assertFalse(vision.detect(frame).junction)

    def test_invalid_frame_and_config_rejected(self):
        with self.assertRaises(ValueError):
            Vision(Config(roi_top=0.99, roi_bottom=0.2))
        with self.assertRaises(ValueError):
            Vision(Config()).detect(np.zeros((20, 20), np.uint8))


class RoundaboutStateTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(confirm_frames=3, exit_confirm_frames=3, recovery_frames=3, rearm_clear_frames=3)
        self.machine = Machine(self.cfg)
        self.arc, self.straight = ring_path(), ring_path(False)
        self.junction = Observation([self.arc, self.straight], self.arc, self.straight, True)
        self.arc_only = Observation([self.arc], self.arc)
        self.straight_only = Observation([self.straight], tangent=self.straight)
        self.now = 1.0

    def tick(self, obs, degrees=0, base=True, heading=True):
        self.now += 0.1
        pose = Heading(math.radians(degrees), self.now, 1) if heading else None
        return self.machine.update(obs, pose, self.now, base, 0.0)

    def enter(self):
        for _ in range(self.cfg.confirm_frames):
            decision = self.tick(self.junction)
        self.assertEqual(self.machine.state, State.ENTER)
        self.assertIs(decision.path, self.arc)
        return decision

    def test_no_takeover_on_one_frame_or_without_heading(self):
        self.assertFalse(self.tick(self.junction).owned)
        self.tick(self.straight_only)
        for _ in range(8):
            self.assertFalse(self.tick(self.junction, heading=False).owned)

    def test_complete_lap_requires_measured_yaw_and_exit_then_cooldown(self):
        self.enter()
        for degrees in (40, 100, 200, 300):
            self.assertIs(self.tick(self.arc_only, degrees).path, self.arc)
        # Same entrance image before a lap must never pick the tangent.
        self.assertIs(self.tick(self.junction, 310).path, self.arc)
        for _ in range(3):
            decision = self.tick(self.junction, 358)
        self.assertEqual(self.machine.state, State.EXIT)
        self.assertIs(decision.path, self.straight)
        for _ in range(3):
            decision = self.tick(self.straight_only, 360)
        self.assertEqual(self.machine.completed, 1)
        self.assertEqual(self.machine.state, State.COOLDOWN)
        self.assertFalse(decision.owned)
        for _ in range(6):
            self.assertFalse(self.tick(self.junction, 360).owned)

    def test_short_gap_prediction_then_stop_without_fake_success(self):
        self.enter()
        decision = self.tick(Observation(), 5, base=False)
        self.assertTrue(decision.predicted)
        self.assertLessEqual(decision.speed, self.cfg.predict_speed)
        for _ in range(4):
            decision = self.tick(Observation(), 10, base=False)
        self.assertTrue(decision.owned)
        self.assertFalse(decision.valid)
        self.assertEqual(self.machine.state, State.RECOVER)
        self.assertEqual(self.machine.completed, 0)
        # A single stable line restores baseline only after consecutive frames.
        for i in range(3):
            decision = self.tick(self.straight_only, 10)
            if i < 2:
                self.assertTrue(decision.owned)
        self.assertFalse(decision.owned)
        self.assertEqual(self.machine.completed, 0)

    def test_reverse_yaw_stale_sensor_and_reset_abort(self):
        self.enter()
        self.assertFalse(self.tick(self.arc_only, -25).valid)
        self.assertEqual(self.machine.state, State.RECOVER)
        self.setUp()
        self.enter()
        self.assertFalse(self.tick(self.arc_only, heading=False).valid)
        self.setUp()
        self.enter()
        bad = Heading(1.0, self.now+0.1, 2)
        self.assertFalse(self.machine.update(self.arc_only, bad, self.now+0.1).valid)

    def test_static_command_or_elapsed_time_cannot_complete_lap(self):
        self.enter()
        for _ in range(90):
            self.tick(self.arc_only, 0)
        self.assertEqual(self.machine.completed, 0)
        self.assertEqual(self.machine.aborted, 1)
        self.assertIn(self.machine.state, (State.RECOVER, State.COOLDOWN))

    def test_incomplete_second_lap_is_not_reused_progress(self):
        self.enter()
        self.tick(self.arc_only, 100)
        self.machine.change(State.SEARCH, self.now, "new_lap")
        for _ in range(3):
            self.tick(self.junction, 100)
        self.assertAlmostEqual(self.machine.degrees, 0.0)

    def test_heading_unwrap_and_jump_rejection(self):
        tracker = HeadingTracker(self.cfg)
        self.assertTrue(tracker.update(math.radians(179), 1.0))
        self.assertTrue(tracker.update(math.radians(-179), 1.1))
        self.assertAlmostEqual(math.degrees(tracker.value.total), 2.0)
        epoch = tracker.value.epoch
        self.assertFalse(tracker.update(0.0, 1.2))
        self.assertGreater(tracker.value.epoch, epoch)
        self.assertFalse(tracker.update(float("nan"), 1.3))

    def test_bundled_nonstandard_imu_layout(self):
        q = SimpleNamespace(x=math.cos(math.pi/4), y=0.0, z=0.0, w=math.sin(math.pi/4))
        yaw, tilt = quaternion_angles(q, "wxyz_in_xyzw")
        self.assertAlmostEqual(yaw, math.pi/2)
        self.assertAlmostEqual(tilt, 0.0)


def fake_line_node(result):
    node = LineFollower.__new__(LineFollower)
    node._frame_lock = threading.Lock()
    node._camera_healthy = True
    node._latest_frame_time = time.monotonic()
    node._frame_sequence = 1
    node._last_processed_sequence = 0
    node._latest_frame = result.debug_frame.copy()
    node.image_timeout_seconds = 1.0
    node.detector = SimpleNamespace(detect=lambda _: result)
    node._update_zebra = lambda _: None
    node._apply_roundabout = lambda f, r, n: r
    node.enable_motion = True
    node._last_command_valid = True
    node._last_command = (0.3, 0.2)
    node._lost_hold = node.lost_hold_frames = 3
    node._have_previous_error = False
    node._previous_error = 0.0
    node.error_filter_alpha = 0.3
    node.max_error_derivative = 3.0
    node.steering_sign = -1.0
    node.steering_kp = 2.0
    node.steering_kd = 0.025
    node.max_angular_speed = 0.9
    node.forward_speed = 0.35
    node._limit_command_rate = lambda v, w, now: (max(v, 0.2), w)
    node._log_status = lambda *args: None
    node._show_debug = lambda *args: None
    node.commands = []
    node._publish_command = lambda v, w: node.commands.append((v, w))
    node._publish_zero = lambda reason: node.commands.append((0.0, 0.0))
    return node


class RoundaboutIntegrationTest(unittest.TestCase):
    def make_adapter(self):
        adapter = RoundaboutAdapter.__new__(RoundaboutAdapter)
        adapter.cfg = Config(adaptive=False, max_processing_seconds=1.0)
        adapter.vision, adapter.machine = Vision(adapter.cfg), Machine(adapter.cfg)
        adapter.tracker = HeadingTracker(adapter.cfg)
        adapter.mode, adapter.failed, adapter.owns_control = "observe", "", False
        adapter.last_status, adapter.elapsed_ms = 0.0, 0.0
        adapter.observation = Observation()
        adapter.node = SimpleNamespace(show_debug=False, target_offset_pixels=0.0,
                                       _have_previous_error=True, _lost_hold=3, _last_command_valid=True)
        adapter.publish = lambda now: None
        return adapter

    def test_off_returns_original_result_and_does_not_import_module(self):
        result = DetectionResult(True, 0.1, 0.9, "line", light_floor())
        node = SimpleNamespace(get_parameter=lambda _: SimpleNamespace(value="off"),
                               _last_roundabout_mode="off", _roundabout=None)
        self.assertIs(LineFollower._apply_roundabout(node, None, result, 1.0), result)
        self.assertFalse(node._roundabout_owns_control)

    def test_basic_line_control_unchanged(self):
        node = fake_line_node(DetectionResult(True, 0.2, 0.9, "line", light_floor()))
        node._run_control_cycle()
        self.assertEqual(node.commands[-1], (0.35, -0.4))

    def test_ring_speed_cap_applies_even_during_deceleration(self):
        node = fake_line_node(DetectionResult(True, -0.2, 0.9, "ring", light_floor(), speed_limit=0.08, inhibit_hold=True))
        node._run_control_cycle()
        self.assertEqual(node.commands[-1], (0.08, 0.4))

    def test_ring_loss_does_not_replay_last_command(self):
        node = fake_line_node(DetectionResult(False, 0.0, 0.0, "ring_lost", light_floor(), speed_limit=0.0, inhibit_hold=True))
        node._run_control_cycle()
        self.assertEqual(node.commands, [(0.0, 0.0)])

    def test_baseline_loss_still_uses_original_hold(self):
        node = fake_line_node(DetectionResult(False, 0.0, 0.0, "lost", light_floor()))
        node._run_control_cycle()
        self.assertEqual(node.commands, [(0.3, 0.2)])

    def test_observe_mode_does_not_reset_baseline_controller(self):
        adapter = RoundaboutAdapter.__new__(RoundaboutAdapter)
        adapter.cfg, adapter.mode, adapter.owns_control = Config(), "off", False
        adapter.node = SimpleNamespace(_have_previous_error=True, _lost_hold=3, _last_command_valid=True)
        adapter.reset_mode("observe")
        self.assertTrue(adapter.node._have_previous_error)
        self.assertEqual(adapter.node._lost_hold, 3)
        self.assertTrue(adapter.node._last_command_valid)

    def test_observe_does_not_change_result_even_if_virtual_machine_enters(self):
        adapter = self.make_adapter()
        baseline = DetectionResult(True, 0.07, 0.82, "tracking", ring_view())
        for i in range(6):
            now = 1.0+i*0.1
            adapter.tracker.update(0.0, now)
            result = adapter.process(ring_view(), baseline, now, "observe")
            self.assertIs(result, baseline)
            self.assertFalse(adapter.owns_control)
        self.assertEqual(adapter.machine.state, State.ENTER)
        self.assertTrue(adapter.node._have_previous_error)
        self.assertEqual(adapter.node._lost_hold, 3)

    def test_idle_vision_exception_falls_back_but_active_exception_stops(self):
        adapter = self.make_adapter()
        baseline = DetectionResult(True, 0.0, 0.9, "tracking", ring_view())
        adapter.vision.detect = lambda _: (_ for _ in ()).throw(ValueError("bad camera shape"))
        self.assertIs(adapter.process(ring_view(), baseline, 1.0, "control"), baseline)
        adapter.failed = ""
        adapter.machine.change(State.FOLLOW, 1.0, "test")
        result = adapter.process(ring_view(), baseline, 1.1, "control")
        self.assertFalse(result.valid)
        self.assertTrue(result.inhibit_hold)
        self.assertEqual(result.speed_limit, 0.0)
        self.assertEqual(adapter.machine.completed, 0)

    def test_higher_priority_element_prevents_takeover(self):
        adapter = self.make_adapter()
        baseline = DetectionResult(True, 0.0, 0.9, "tracking", ring_view())
        for i in range(6):
            now = 1.0+i*0.1
            adapter.tracker.update(0.0, now)
            self.assertIs(adapter.process(ring_view(), baseline, now, "control", allowed=False), baseline)
        self.assertFalse(adapter.owns_control)

    def test_stale_camera_and_duplicate_frame_do_not_update_ring(self):
        node = fake_line_node(DetectionResult(True, 0.0, 0.9, "tracking", light_floor()))
        node._apply_roundabout = lambda *args: self.fail("must not process stale or repeated frame")
        node._latest_frame_time = time.monotonic()-2.0
        node._run_control_cycle()
        self.assertEqual(node.commands[-1], (0.0, 0.0))
        node._latest_frame_time = time.monotonic()
        node._last_processed_sequence = node._frame_sequence
        node._run_control_cycle()

    def test_sensor_stamp_qos_adapter_rejects_replay_and_bad_quaternion(self):
        adapter = self.make_adapter()
        adapter.heading_source, adapter.imu_layout, adapter.yaw_sign = "imu", "xyzw", 1.0
        adapter.last_sensor_stamp = -math.inf
        adapter.node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
        msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=0)),
                              orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                              orientation_covariance=[0.0]*9)
        adapter._heading(msg)
        stamp = adapter.tracker.value.stamp
        msg.orientation.z = 0.2
        adapter._heading(msg)  # duplicate source timestamp
        self.assertEqual(adapter.tracker.value.stamp, stamp)
        msg.header.stamp.sec = 8
        adapter._heading(msg)  # delayed delivery
        self.assertEqual(adapter.tracker.value.stamp, stamp)
        msg.header.stamp.sec = 10
        msg.header.stamp.nanosec = 1
        msg.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)
        adapter._heading(msg)
        self.assertEqual(adapter.tracker.value.stamp, stamp)

    def test_image_sequence_and_measured_heading_complete_exactly_once(self):
        adapter = self.make_adapter()
        frames = {"junction": ring_view(), "arc": ring_view(straight=False),
                  "straight": ring_view(curved=False)}
        # Synthetic sensor poses are fed through the same unwrap/jump gate.
        # This tests the full software chain; it is not a real-robot trial.
        sequence = [("junction", 0)]*5
        sequence += [("arc", deg) for deg in range(4, 353, 4)]
        sequence += [("junction", 356)]*5+[("straight", 360)]*7
        for i, (kind, degree) in enumerate(sequence):
            now = 1.0+i*0.1
            adapter.tracker.update(math.radians(degree), now)
            frame = frames[kind]
            baseline = DetectionResult(True, 0.0, 0.9, "tracking", frame.copy())
            adapter.process(frame, baseline, now, "control")
        self.assertEqual(adapter.machine.completed, 1)
        self.assertEqual(adapter.machine.aborted, 0)
        self.assertFalse(adapter.owns_control)
        self.assertEqual(adapter.machine.state, State.COOLDOWN)

    def test_icn_ring_ownership_suppresses_false_step_and_zebra_transitions(self):
        from icn import IcnNode, MissionState
        node = IcnNode.__new__(IcnNode)
        node.state = MissionState.LINE
        node.mission_timeout_seconds = 0.0
        node._roundabout_owns_control = True
        node.enable_zebra = node._zebra_armed = node._zebra_confirmed = True
        node.step_enabled = node._step_armed = node._step_confirmed = True
        node._zebra_cooldown_until = node._step_cooldown_until = 0.0
        node._set_state = lambda *args: self.fail("ring must own element decision")
        node._enter_stair = lambda *args: self.fail("dashes are not a step")
        node._step_state(1.0)
        self.assertEqual(node.state, MissionState.LINE)


if __name__ == "__main__":
    unittest.main()
