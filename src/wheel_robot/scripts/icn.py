#!/usr/bin/env python3
"""ICN 任务节点：巡线 + 赛道阶段 + 元素动作。

阶段顺序（每圈固定）：
    FIRST_CORNER -> SEEK_PLATFORM -> CROSS_PLATFORM -> SEEK_ROUNDABOUT
    -> ROUNDABOUT -> SEEK_ZEBRA -> LAP_COMPLETE -> 下一圈

基础巡线始终运行；阶段只决定元素能否触发。
未被阶段允许的元素：不累计确认、不触发动作（可只画调试）。
台阶动作结束不等于越过平台，必须再做“巡线恢复 + 缺口消失”确认。
"""

from __future__ import annotations

from enum import Enum
import os
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String, UInt8

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import line_follower as line_lib
    import zebra_crossing_detector as zebra_lib
    import step_detector as step_lib
    import track_phase as phase_lib
except ImportError as error:
    raise SystemExit(
        "icn.py 需要和 line_follower.py、zebra_crossing_detector.py、"
        f"step_detector.py、track_phase.py 放在同一目录：{error}"
    )


class MissionState(str, Enum):
    LINE = "LINE"
    ZEBRA = "ZEBRA"
    STAIR = "STAIR"
    FINISH = "FINISH"


ELEMENT_DEFAULTS = {
    # ---------------- 斑马线 ----------------
    "zebra_roi_top_ratio": 0.70,
    "zebra_roi_bottom_ratio": 1.00,
    "zebra_roi_left_ratio": 0.15,
    "zebra_roi_right_ratio": 0.85,
    "zebra_perspective_enabled": True,
    "zebra_approach_slowdown": 0.70,
    "zebra_stop_seconds": 0.0,
    "zebra_cross_speed": 0.12,
    "zebra_cross_min_seconds": 0.5,
    "zebra_cross_max_seconds": 4.0,
    "zebra_cooldown_seconds": 1.0,
    "zebra_exit_grace_frames": 8,
    "zebra_target_count": 1,
    "zebra_trigger_max_error": 0.40,
    # 斑马线：整体排列约束
    "zebra_min_aspect_ratio": 1.4,
    "zebra_max_aspect_ratio": 8.0,
    "zebra_min_long_side_ratio": 0.18,
    "zebra_min_short_side_ratio": 0.012,
    "zebra_max_vertical_deviation_deg": 30.0,
    "zebra_max_angle_std_deg": 9.0,
    "zebra_min_axis_spread_deg": 55.0,
    "zebra_min_band_width_ratio": 0.30,
    "zebra_max_spacing_cv": 0.60,
    "zebra_max_curvature_gain": 0.45,
    "zebra_max_line_rms_ratio": 0.08,

    # ---------------- 台阶 ----------------
    "step_enabled": True,
    "step_decimate": 2,
    "step_confirm_frames": 3,
    "step_action": "stop",
    "step_jump_enabled": False,  # 显式启用后，每圈进入台阶动作时请求一次跳跃
    "step_hold_seconds": 0.0,
    "step_stop_seconds": 1.0,
    "step_climb_speed": 0.15,
    "step_climb_seconds": 3.0,
    "step_cooldown_seconds": 2.0,
    "step_exit_grace_frames": 10,
    "step_trigger_max_error": 0.35,
    "step_center_left_ratio": 0.30,
    "step_center_right_ratio": 0.70,
    "step_bright_value_min": 110,
    "step_gap_bright_min": 0.55,
    "step_min_gap_rows": 12,
    # 旧视觉参数保留兼容；简化检测不再使用颜色、弯道、前缘条件。
    "step_platform_value_min": 95,
    "step_platform_saturation_max": 150,
    "step_wood_hue_min": 5,
    "step_wood_hue_max": 35,
    "step_wood_saturation_min": 35,
    "step_gap_surface_min": 0.50,
    "step_curve_min_rows": 8,
    "step_curve_max_dx_per_row": 0.10,
    "step_curve_min_dx": 0.18,
    "step_curve_min_support": 0.55,
    "step_curve_min_width_ratio": 0.006,
    "step_curve_max_width_ratio": 0.10,
    "step_edge_gradient_min": 25,
    "step_edge_min_width_ratio": 0.45,
    "step_edge_row_tolerance": 7,
    "step_require_edge": True,
    "step_track_confirm_frames": 1,
    "step_track_max_frame_gap": 0.30,
    "step_track_edge_row_range": 8,
    "step_track_lateral_range": 0.10,
    "mission_timeout_seconds": 0.0,
}


class IcnNode(line_lib.LineFollower):
    def __init__(self) -> None:
        super().__init__()

        for name, default in ELEMENT_DEFAULTS.items():
            self.declare_parameter(name, default)
        for name, default in ELEMENT_DEFAULTS.items():
            setattr(self, name, self._read_param(name, default))

        # 阶段参数：以 PhaseConfig 的字段为唯一来源，避免两处手写不一致
        defaults = phase_lib.PhaseConfig()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(
                "track_phase." + name, getattr(defaults, name)
            )
        self.declare_parameter("track_phase.start_phase", "FIRST_CORNER")
        self.declare_parameter("track_phase.debug_blocked_detectors", True)
        self.track_phase_debug_blocked = bool(
            self.get_parameter("track_phase.debug_blocked_detectors").value
        )

        self.state = MissionState.LINE
        self._mission_start = time.monotonic()
        self._crossings = 0

        # ---------------- 斑马线证据 ----------------
        self._zebra_armed = True
        self._zebra_count = 0
        self._zebra_debug: Optional[np.ndarray] = None
        self._zebra_enter_time = 0.0
        self._zebra_cooldown_until = 0.0
        self._last_warn = 0.0
        self._zebra_roi = zebra_lib.RoiConfig(
            top_ratio=self.zebra_roi_top_ratio,
            bottom_ratio=self.zebra_roi_bottom_ratio,
            left_ratio=self.zebra_roi_left_ratio,
            right_ratio=self.zebra_roi_right_ratio,
        )
        self._zebra_perspective = zebra_lib.PerspectiveConfig(
            enabled=self.zebra_perspective_enabled
        )

        # ---------------- 台阶证据 ----------------
        # 参数以 StepConfig 字段为唯一来源；ROI/暗阈值沿用巡线
        step_defaults = step_lib.StepConfig()
        skip_step_fields = {
            "top_ratio", "bottom_ratio", "left_ratio",
            "right_ratio", "dark_value_max",
        }
        for name in step_defaults.__dataclass_fields__:
            if name in skip_step_fields:
                continue
            param = "step_" + name
            if not self.has_parameter(param):
                self.declare_parameter(param, getattr(step_defaults, name))
        step_values = {
            name: self._read_param(
                "step_" + name, getattr(step_defaults, name)
            )
            for name in step_defaults.__dataclass_fields__
            if name not in skip_step_fields
        }
        step_values.update(
            top_ratio=0.40,
            bottom_ratio=0.80,
            left_ratio=self.search_left_ratio,
            right_ratio=self.search_right_ratio,
            dark_value_max=self.hsv_upper_v,
        )
        self._step_cfg = step_lib.StepConfig(**step_values)
        self._step_tracker = step_lib.StepTracker(self._step_cfg)

        self._step_cycle = 0
        self._step_hit = 0
        self._step_confirmed = False
        self._step_armed = True
        self._step_cooldown_until = 0.0
        self._stair_enter_time = 0.0
        self._step_gap = 0
        self._step_bright = 0.0
        self._step_last: Optional[step_lib.StepResult] = None
        self._step_raw = False

        # ---------------- 斑马线布局参数 ----------------
        zebra_defaults = zebra_lib.StripeLayoutConfig()
        for name in zebra_defaults.__dataclass_fields__:
            param = "zebra_" + name
            if not self.has_parameter(param):
                self.declare_parameter(param, getattr(zebra_defaults, name))
        self._stripe_cfg = zebra_lib.StripeLayoutConfig(
            **{
                name: self._read_param(
                    "zebra_" + name, getattr(zebra_defaults, name)
                )
                for name in zebra_defaults.__dataclass_fields__
            }
        )
        self._zebra_raw = False
        self._zebra_layout_reason = ""
        self._ring_state_text = "N/A"

        # ---------------- 赛道阶段 ----------------
        self._phase = phase_lib.TrackPhaseManager(
            config=self._build_phase_config(),
            start_phase=self._read_start_phase(),
            on_change=self._on_phase_change,
            roundabout_enabled=str(self.get_parameter("roundabout_mode").value).lower()
            in ("observe", "control"),
        )

        # ---------------- 输出话题 ----------------
        self._zebra_pub = self._zebra_detected_publisher or self.create_publisher(
            Bool, "/zebra_crossing/detected", 1
        )
        self._zebra_count_pub = (
            self._zebra_count_publisher
            or self.create_publisher(UInt8, "/zebra_crossing/stripe_count", 1)
        )
        self._step_pub = self.create_publisher(Bool, "/step/detected", 1)
        # 默认可靠、volatile：不向后启动的底盘重放旧跳跃请求。
        self._vision_step_pub = self.create_publisher(Bool, "/vision/step_detected", 10)
        self._state_pub = self.create_publisher(String, "/icn/state", 10)
        self._phase_pub = self.create_publisher(String, "/icn/phase", 10)
        self._lap_pub = self.create_publisher(UInt8, "/icn/lap", 10)
        self._publish_state()
        self._publish_phase()

        self.get_logger().info(
            "ICN 任务层已加载：阶段=%s（第 1 圈），斑马线=%s，台阶=%s（动作=%s），圆环=%s"
            % (
                self._phase.phase.value,
                "启用" if self.enable_zebra else "关闭",
                "启用" if self.step_enabled else "关闭",
                self.step_action,
                self.get_parameter("roundabout_mode").value,
            )
        )

    # ============================================================ 参数
    def _read_param(self, name: str, default):
        value = self.get_parameter(name).value
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, int):
            return int(value)
        return str(value)

    def _build_phase_config(self) -> phase_lib.PhaseConfig:
        defaults = phase_lib.PhaseConfig()
        values = {
            name: self._read_param(
                "track_phase." + name, getattr(defaults, name)
            )
            for name in defaults.__dataclass_fields__
        }
        return phase_lib.PhaseConfig(**values)

    def _read_start_phase(self) -> phase_lib.TrackPhase:
        raw = str(
            self._read_param("track_phase.start_phase", "FIRST_CORNER")
        ).strip().upper()
        try:
            return phase_lib.TrackPhase(raw)
        except ValueError:
            self.get_logger().warning(
                "未知 track_phase.start_phase=%r，按 FIRST_CORNER 处理" % raw
            )
            return phase_lib.TrackPhase.FIRST_CORNER

    # ============================================================ 阶段回调
    def _on_phase_change(self, transition: phase_lib.PhaseTransition) -> None:
        self._reset_element_evidence(transition.reason)
        text = transition.text()
        if transition.reason.endswith("timeout"):
            self.get_logger().warning(text)
        else:
            self.get_logger().info(text)
        self._publish_phase()

    def _publish_phase(self) -> None:
        self._phase_pub.publish(String(data=self._phase.phase.value))
        self._lap_pub.publish(UInt8(data=max(0, min(self._phase.lap, 255))))

    def _reset_element_evidence(self, reason: str) -> None:
        self._zebra_hit = 0
        self._zebra_miss = 0
        self._zebra_confirmed = False
        self._zebra_armed = True
        self._zebra_cooldown_until = 0.0

        self._step_hit = 0
        self._step_confirmed = False
        self._step_armed = True
        self._step_cooldown_until = 0.0
        self._step_gap = 0
        self._step_bright = 0.0
        self._step_last = None

        self._zebra_pub.publish(Bool(data=False))
        self._step_pub.publish(Bool(data=False))
        self.get_logger().info("清空元素确认计数（%s）" % reason)
        self._zebra_raw = False
        self._zebra_layout_reason = ""
        self._step_raw = False
        if hasattr(self, "_step_tracker"):
            self._step_tracker.reset()

    def _observe_blocked_detectors(self) -> bool:
        return bool(self.track_phase_debug_blocked) and bool(self.show_debug)

    # ============================================================ 阶段证据
    def _observe_phase_line(self, now: float, result) -> None:
        self._phase.observe_line(
            now=now,
            valid=bool(result.valid),
            error=float(result.error),
            confidence=float(result.confidence),
            corner=bool(result.corner),
        )

    def _observe_phase_roundabout(self, now: float) -> None:
        ring = getattr(self, "_roundabout", None)
        mode = str(self.get_parameter("roundabout_mode").value).lower()
        if ring is None:
            self._ring_state_text = "N/A"
            self._phase.observe_roundabout(
                now, mode=mode, state="", completed=0, junction=False
            )
            return
        machine = ring.machine
        self._ring_state_text = machine.state.value
        self._phase.observe_roundabout(
            now,
            mode=mode,
            state=machine.state.value,
            completed=int(machine.completed),
            junction=bool(ring.observation.junction) and mode != "off",
        )

    # ============================================================ 斑马线
    def _update_zebra_element(self, frame: np.ndarray, phase_open: bool) -> None:
        if not self.enable_zebra:
            # 斑马线检测关闭时，把上次残留的“看到/确认”状态清掉，
            # 避免 HUD 和 /zebra_crossing/detected 停在旧值上。
            self._zebra_raw = False
            self._zebra_confirmed = False
            self._zebra_hit = 0
            self._zebra_miss = 0
            self._zebra_count = 0
            self._zebra_layout_reason = ""
            self._zebra_pub.publish(Bool(data=False))
            return
        if not phase_open and not self._observe_blocked_detectors():
            self._zebra_hit = 0
            self._zebra_miss = 0
            self._zebra_confirmed = False
            self._zebra_raw = False
            self._zebra_pub.publish(Bool(data=False))
            return

        self._zebra_cycle += 1
        if self.zebra_decimate > 1 and self._zebra_cycle % self.zebra_decimate != 0:
            return

        try:
            roi, _ = zebra_lib.extract_roi(frame, self._zebra_roi)
            bird_view, _ = zebra_lib.warp_perspective(roi, self._zebra_perspective)
            track_mask, _ = zebra_lib.extract_white_track(
                bird_view,
                saturation_max=self.zebra_saturation_max,
                value_min=self.zebra_value_min,
            )
            result = zebra_lib.count_black_stripes(
                track_mask,
                minimum_stripe_count=self.zebra_minimum_stripe_count,
                maximum_stripe_count=self.zebra_maximum_stripe_count,
                config=self._stripe_cfg,
            )
        except Exception as error:
            self._warn("斑马线检测异常: %s" % error)
            return

        self._zebra_debug = result.debug_image
        self._zebra_count = int(result.stripe_count)
        self._zebra_raw = bool(result.zebra_exists)
        self._zebra_layout_reason = result.layout_reason

        if not phase_open:
            self._zebra_hit = 0
            self._zebra_miss = 0
            self._zebra_confirmed = False
            self._zebra_pub.publish(Bool(data=False))
            self._zebra_count_pub.publish(
                UInt8(data=min(self._zebra_count, 255))
            )
            return

        if result.zebra_exists:
            self._zebra_hit += 1
            self._zebra_miss = 0
        else:
            self._zebra_miss += 1
            self._zebra_hit = 0

        if not self._zebra_confirmed and self._zebra_hit >= self.zebra_confirm_frames:
            self._zebra_confirmed = True
            self.get_logger().info(
                "斑马线确认: stripes=%d angle=%+.1f spread=%.0f band=%.2f phase=%s"
                % (self._zebra_count, result.dominant_angle_deg,
                   result.axis_spread_deg, result.band_width_ratio,
                   self._phase.phase.value)
            )
        elif self._zebra_confirmed and self._zebra_miss >= self.zebra_release_frames:
            self._zebra_confirmed = False
            self.get_logger().info("斑马线已离开视野")

        if not self._zebra_confirmed and self._zebra_miss >= self.zebra_release_frames:
            self._zebra_armed = True

        self._zebra_pub.publish(Bool(data=self._zebra_confirmed))
        self._zebra_count_pub.publish(UInt8(data=min(self._zebra_count, 255)))

    # ============================================================ 台阶
    def _clear_step_evidence(self) -> None:
        if self._step_confirmed:
            self._step_pub.publish(Bool(data=False))
        self._step_hit = 0
        self._step_confirmed = False
        self._step_gap = 0
        self._step_bright = 0.0
        self._step_last = None
        self._step_raw = False

    def _update_step_element(
        self, frame: np.ndarray, now: float,
        phase_open: bool, need_clear_evidence: bool,
    ) -> bool:
        if not self.step_enabled:
            self._step_raw = False
            return False
        run = (
            phase_open
            or need_clear_evidence
            or self._observe_blocked_detectors()
        )
        if not run:
            self._clear_step_evidence()
            return False

        self._step_cycle += 1
        if self.step_decimate > 1 and self._step_cycle % self.step_decimate != 0:
            return False

        try:
            step = step_lib.detect_step(frame, self._step_cfg)
        except Exception as error:
            self._step_last = None
            self._step_raw = False
            self._warn("台阶检测异常: %s" % error)
            return False

        self._step_last = step
        self._step_raw = bool(step.detected)
        self._step_gap = step.gap_rows
        self._step_bright = step.gap_bright_ratio

        if not phase_open:
            self._step_hit = 0
            self._step_confirmed = False
            self._step_tracker.reset()
            self._step_pub.publish(Bool(data=False))
            return True

        was_confirmed = self._step_confirmed
        self._step_confirmed = self._step_tracker.update(step, now)
        if self._step_confirmed and not was_confirmed:
            self.get_logger().info(
                "台阶确认(连续无黑线): gap=%dpx min=%d phase=%s 动作=%s"
                % (step.gap_rows, self._step_cfg.min_gap_rows,
                   self._phase.phase.value, self.step_action)
            )
        self._step_pub.publish(Bool(data=self._step_confirmed))
        return True

    # ============================================================ 状态机
    def _set_state(self, state: MissionState, reason: str) -> None:
        if self.state is state:
            return
        self.state = state
        self.get_logger().info("状态切换 -> %s（%s）" % (state.value, reason))
        self._publish_state()

    def _publish_state(self) -> None:
        self._state_pub.publish(String(data=self.state.value))

    def _leave_zebra(self, now: float) -> None:
        self._zebra_cooldown_until = now + self.zebra_cooldown_seconds
        self._zebra_confirmed = False
        self._zebra_armed = False
        self._zebra_hit = 0
        self._zebra_miss = 0
        self._have_previous_error = False
        self._lost_hold = int(self.zebra_exit_grace_frames)
        self._set_state(MissionState.LINE, "zebra_passed")

    def _enter_stair(self, now: float) -> None:
        if not self._phase.begin_platform(now):
            self._warn("台阶确认被阶段拒绝（不在找平台阶段或本圈已消费）")
            return
        self._stair_enter_time = now
        self._set_state(
            MissionState.STAIR, "step_confirmed action=%s" % self.step_action
        )
        # 新增：只有允许运动时，才把台阶确认结果发送给底盘。
        if self.enable_motion:
            # 先复位，再发布确认结果，第二圈可以再次产生上升沿。
            self._vision_step_pub.publish(Bool(data=False))
            self._vision_step_pub.publish(Bool(data=True))

            self.get_logger().info(
                "第 %d 圈：发布台阶确认结果 /vision/step_detected"
                % self._phase.lap
            )
    def _leave_stair(self, now: float) -> None:
        self._step_cooldown_until = now + self.step_cooldown_seconds
        self._step_confirmed = False
        self._step_armed = False
        self._step_hit = 0
        self._have_previous_error = False
        self._lost_hold = int(self.step_exit_grace_frames)
        self._phase.platform_action_done(
            now, step_available=self.step_enabled
        )
        self._set_state(
            MissionState.LINE, "step_action_done_wait_platform_clear"
        )

    def _step_action_command(self, now: float):
        elapsed = now - self._stair_enter_time
        action = self.step_action

        if action == "stop":
            if self.step_hold_seconds > 0.0 and elapsed >= self.step_hold_seconds:
                return 0.0, 0.0, True
            return 0.0, 0.0, False

        if action == "wait_climb":
            if elapsed < self.step_stop_seconds:
                return 0.0, 0.0, False
            if elapsed < self.step_stop_seconds + self.step_climb_seconds:
                return self.step_climb_speed, 0.0, False
            return 0.0, 0.0, True

        if action == "climb":
            if elapsed < self.step_climb_seconds:
                return self.step_climb_speed, 0.0, False
            return 0.0, 0.0, True

        self._warn("未知 step_action=%s，按停车处理" % action)
        return 0.0, 0.0, False

    def _enter_finish(self, reason: str) -> None:
        self._set_state(MissionState.FINISH, reason)
        self._publish_zero(reason)

    @staticmethod
    def _error_within(result, limit: float) -> bool:
        if result is None or not result.valid:
            return True
        return abs(float(result.error)) <= float(limit)

    def _zebra_ready(self, now: float, result) -> bool:
        if not self.enable_zebra:
            return False
        if not self._phase.may_trigger(phase_lib.TrackElement.ZEBRA):
            return False
        if not self._zebra_armed or not self._zebra_confirmed:
            return False
        if now < self._zebra_cooldown_until:
            return False
        return self._error_within(result, self.zebra_trigger_max_error)

    def _step_ready(self, now: float, result) -> bool:
        if not self.step_enabled:
            return False
        if not self._phase.may_trigger(phase_lib.TrackElement.STEP):
            return False
        if not self._step_armed or not self._step_confirmed:
            return False
        if now < self._step_cooldown_until:
            return False
        return True

    def _step_state(self, now: float, result=None) -> None:
        if (
            self.mission_timeout_seconds > 0.0
            and now - self._mission_start > self.mission_timeout_seconds
            and self.state is not MissionState.FINISH
        ):
            self._enter_finish("mission_timeout")
        if self.state is MissionState.FINISH:
            return

        if self.state is MissionState.LINE:
            if self._roundabout_owns_control:
                return
            if self._zebra_ready(now, result):
                self._zebra_enter_time = now
                self._phase.zebra_entered(now)
                self._set_state(MissionState.ZEBRA, "zebra_confirmed")
                return
            if self._step_ready(now, result):
                self._enter_stair(now)
            return

        if self.state is MissionState.ZEBRA:
            elapsed = now - self._zebra_enter_time
            if elapsed > self.zebra_cross_max_seconds:
                self.get_logger().warning(
                    "斑马线通过超时（%.1fs），强制退回巡线" % elapsed
                )
                self._leave_zebra(now)
                return
            released = (
                not self._zebra_confirmed
                and self._zebra_miss >= self.zebra_release_frames
                and elapsed >= self.zebra_cross_min_seconds
            )
            if released:
                self._leave_zebra(now)
                self._crossings += 1
                self.get_logger().info(
                    "完成第 %d 次斑马线通过" % self._crossings
                )
                self._phase.zebra_passed(now)
                if (
                    self.zebra_target_count > 0
                    and self._crossings >= self.zebra_target_count
                ):
                    self._enter_finish("zebra_target_reached")

    # ============================================================ 控制
    def _steering(self, now: float, result) -> float:
        if self._have_previous_error:
            filtered = (
                self.error_filter_alpha * result.error
                + (1.0 - self.error_filter_alpha) * self._previous_error
            )
            dt = max(1e-3, now - self._previous_control_time)
            derivative = float(
                np.clip(
                    (filtered - self._previous_error) / dt,
                    -self.max_error_derivative,
                    self.max_error_derivative,
                )
            )
        else:
            filtered = float(result.error)
            derivative = 0.0
            self._have_previous_error = True
        self._previous_error = filtered
        self._previous_control_time = now
        raw = self.steering_sign * (
            self.steering_kp * filtered + self.steering_kd * derivative
        )
        return float(
            np.clip(raw, -self.max_angular_speed, self.max_angular_speed)
        )

    def _emit(self, now, linear, angular, speed_limit=None):
        if not self.enable_motion:
            self._publish_zero("motion_disabled")
            return linear, angular
        cmd_linear, cmd_angular = self._limit_command_rate(linear, angular, now)
        if speed_limit is not None:
            cmd_linear = min(cmd_linear, speed_limit)
        self._publish_command(cmd_linear, cmd_angular)
        self._last_command = (cmd_linear, cmd_angular)
        self._last_command_valid = True
        return cmd_linear, cmd_angular

    def _ring_should_run(self) -> bool:
        return self._phase.permits(phase_lib.TrackElement.ROUNDABOUT)

    def _run_control_cycle(self) -> None:
        now = time.monotonic()
        with self._frame_lock:
            camera_healthy = self._camera_healthy
            frame_time = self._latest_frame_time
            sequence = self._frame_sequence
            frame = (
                self._latest_frame.copy()
                if self._latest_frame is not None
                else None
            )

        if not camera_healthy or frame is None:
            self._publish_zero("camera_unavailable")
            self._log_status(now, "STOP 摄像头不可用")
            return
        if now - frame_time > self.image_timeout_seconds:
            self._publish_zero("image_timeout")
            self._log_status(now, "STOP 图像超时 %.3fs" % (now - frame_time))
            return
        if sequence == self._last_processed_sequence:
            if self.enable_motion and self._last_command_valid:
                self._publish_command(*self._last_command)
            else:
                self._publish_zero("no_new_frame")
            return
        self._last_processed_sequence = sequence

        # 1) 基础巡线：始终运行
        result = self.detector.detect(frame)

        # 2) 阶段证据：首弯进度 / 平台通过
        self._phase.set_roundabout_enabled(
            str(self.get_parameter("roundabout_mode").value).lower()
            in ("observe", "control"), now,
        )
        self._observe_phase_line(now, result)

        # 3) 圆环：只在圆环阶段允许接管
        ring_relevant = self._ring_should_run()
        if ring_relevant or self._roundabout_owns_control:
            result = self._apply_roundabout(
                frame,
                result,
                now,
                allowed=(
                    ring_relevant and self.state is MissionState.LINE
                ),
            )
        else:
            self._roundabout_owns_control = False
        self._observe_phase_roundabout(now)

        if time.monotonic() - frame_time > self.image_timeout_seconds:
            self._publish_zero("processing_image_timeout")
            self._show_debug(result, 0.0, 0.0, False)
            return

        # 4) 元素检测：阶段门控
        current_phase = self._phase.phase
        zebra_open = self._phase.permits(phase_lib.TrackElement.ZEBRA)
        step_open = self._phase.permits(phase_lib.TrackElement.STEP)
        platform_clear_needed = (
            current_phase is phase_lib.TrackPhase.CROSS_PLATFORM
            and self._phase.platform_done
        )

        self._update_zebra_element(frame, phase_open=zebra_open)
        step_fresh = self._update_step_element(
            frame,
            now,
            phase_open=step_open,
            need_clear_evidence=platform_clear_needed,
        )
        if (
            platform_clear_needed
            and step_fresh
            and self._step_last is not None
        ):
            self._phase.observe_step(now, self._step_last.detected)

        # 5) 控制权状态机
        self._step_state(now, result)

        # 6) 输出
        if self.state is MissionState.FINISH:
            self._publish_zero("mission_finish")
            self._show_debug(result, 0.0, 0.0, False)
            self._log_status(now, "FINISH 任务完成，原地停车")
            return

        if self.state is MissionState.ZEBRA:
            elapsed = now - self._zebra_enter_time
            if elapsed < self.zebra_stop_seconds:
                self._publish_zero("zebra_stop")
                self._show_debug(result, 0.0, 0.0, False)
                self._log_status(
                    now,
                    "ZEBRA 停车等待 %.1f/%.1fs %s"
                    % (
                        elapsed,
                        self.zebra_stop_seconds,
                        self._phase.describe(),
                    ),
                )
                return
            cmd_linear, cmd_angular = self._emit(
                now, self.zebra_cross_speed, 0.0
            )
            self._show_debug(
                result, cmd_linear, cmd_angular, self.enable_motion
            )
            self._log_status(
                now,
                "ZEBRA 直行通过 v=%.3f w=%+.3f stripes=%d"
                % (cmd_linear, cmd_angular, self._zebra_count),
            )
            return

        if self.state is MissionState.STAIR:
            linear, angular, finished = self._step_action_command(now)
            if finished:
                self._leave_stair(now)
            else:
                cmd_linear, cmd_angular = self._emit(now, linear, angular)
                self._show_debug(
                    result, cmd_linear, cmd_angular, self.enable_motion
                )
                self._log_status(
                    now,
                    "STAIR 动作=%s v=%.3f gap=%dpx"
                    % (
                        self.step_action,
                        cmd_linear,
                        self._step_gap,
                    ),
                )
                return

        if not result.valid:
            if (
                self.enable_motion
                and not result.inhibit_hold
                and self._last_command_valid
                and self._lost_hold > 0
            ):
                self._lost_hold -= 1
                self._publish_command(*self._last_command)
                self._show_debug(
                    result,
                    self._last_command[0],
                    self._last_command[1],
                    True,
                )
                self._log_status(
                    now,
                    "LINE 丢线保持 %s（剩 %d 帧） phase=%s"
                    % (
                        result.reason,
                        self._lost_hold,
                        self._phase.describe(),
                    ),
                )
                return
            self._publish_zero(result.reason)
            self._show_debug(result, 0.0, 0.0, False)
            self._log_status(
                now,
                "STOP %s conf=%.2f phase=%s %s"
                % (
                    result.reason,
                    result.confidence,
                    self._phase.describe(),
                    self._phase.evidence_text(),
                ),
            )
            return

        self._lost_hold = self.lost_hold_frames
        approaching = (
            zebra_open and self._zebra_armed and self._zebra_hit > 0
        )
        scale = self.zebra_approach_slowdown if approaching else 1.0
        linear = self.forward_speed * scale
        if result.speed_limit is not None:
            linear = min(linear, result.speed_limit)
        cmd_linear, cmd_angular = self._emit(
            now, linear, self._steering(now, result), result.speed_limit
        )
        self._show_debug(result, cmd_linear, cmd_angular, self.enable_motion)
        self._log_status(
            now,
            "LINE v=%.3f w=%+.3f err=%+.3f conf=%.2f phase=%s %s%s"
            % (
                cmd_linear,
                cmd_angular,
                result.error,
                result.confidence,
                self._phase.describe(),
                self._phase.evidence_text(),
                " 减速接近斑马线" if approaching else "",
            ),
        )

    # ============================================================ 调试
    def _show_debug(
        self, result, planned_linear: float,
        planned_angular: float, motion_active: bool,
    ) -> None:
        debug = result.debug_frame

        # 斑马线小窗移到右下角，避免遮挡状态栏
        if self._zebra_debug is not None:
            try:
                panel = cv2.resize(self._zebra_debug, (240, 150))
                panel_h, panel_w = panel.shape[:2]
                x = debug.shape[1] - panel_w - 8
                y = debug.shape[0] - panel_h - 8
                if x > 0 and y > 0:
                    debug[y:y + panel_h, x:x + panel_w] = panel
                    cv2.rectangle(
                        debug, (x - 1, y - 1),
                        (x + panel_w, y + panel_h), (255, 180, 0), 1,
                    )
            except cv2.error:
                pass

        # 状态栏背景
        cv2.rectangle(
            debug, (0, 60),
            (min(debug.shape[1] - 1, 580), 190), (0, 0, 0), -1,
        )

        cv2.putText(
            debug,
            "COURSE: %s   LAP: %d" % (self._phase.course_name(), self._phase.lap),
            (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (0, 255, 255), 2, cv2.LINE_AA,
        )

        step = self._step_last
        step_extra = ""
        if step is not None:
            step_extra = "  gap=%d/%d rows" % (
                step.gap_rows, self._step_cfg.min_gap_rows,
            )
        cv2.putText(
            debug,
            "STEP:  raw=%d  allowed=%d  triggered=%d%s" % (
                int(self._step_raw),
                int(self._phase.may_trigger(phase_lib.TrackElement.STEP)),
                int(self.state is MissionState.STAIR),
                step_extra,
            ),
            (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (0, 200, 255), 1, cv2.LINE_AA,
        )

        cv2.putText(
            debug,
            "ZEBRA: raw=%d  allowed=%d  triggered=%d  n=%d  layout=%s" % (
                int(self._zebra_raw),
                int(self._phase.may_trigger(phase_lib.TrackElement.ZEBRA)),
                int(self.state is MissionState.ZEBRA),
                self._zebra_count, self._zebra_layout_reason or "-",
            ),
            (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (255, 255, 0), 1, cv2.LINE_AA,
        )

        cv2.putText(
            debug,
            "RING:  allowed=%d  state=%s  ctrl=%d" % (
                int(self._phase.permits(phase_lib.TrackElement.ROUNDABOUT)),
                self._ring_state_text,
                int(self._roundabout_owns_control),
            ),
            (12, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (0, 255, 0), 1, cv2.LINE_AA,
        )

        cv2.putText(
            debug,
            "MISSION %s  %s" % (
                self.state.value, self._phase.evidence_text(),
            ),
            (12, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (200, 200, 200), 1, cv2.LINE_AA,
        )

        # 蓝框为实际搜索区，红/橙框为本帧最长缺口。
        if step is not None:
            frame_h, frame_w = debug.shape[:2]
            rx0 = int(frame_w * self._step_cfg.left_ratio)
            rx1 = int(frame_w * self._step_cfg.right_ratio)
            ry0 = int(frame_h * self._step_cfg.top_ratio)
            ry1 = int(frame_h * self._step_cfg.bottom_ratio)
            rx_center0 = rx0 + int(round((rx1 - rx0) * self._step_cfg.center_left_ratio))
            rx_center1 = rx0 + int(round((rx1 - rx0) * self._step_cfg.center_right_ratio))
            search_top = ry0 + int(round((ry1 - ry0) * self._step_cfg.gap_top_ratio))
            search_bottom = ry0 + int(round((ry1 - ry0) * self._step_cfg.gap_bottom_ratio))
            cv2.rectangle(
                debug, (rx_center0, search_top), (rx_center1 - 1, search_bottom - 1),
                (255, 200, 0), 1,
            )
            if step.gap_rows > 0:
                cv2.rectangle(
                    debug, (rx_center0, ry0 + step.gap_start_row),
                    (rx_center1 - 1, ry0 + step.gap_start_row + step.gap_rows - 1),
                    (0, 0, 255) if self._step_confirmed else (0, 165, 255), 2,
                )

        super()._show_debug(
            result, planned_linear, planned_angular, motion_active
        )

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warn >= 2.0:
            self._last_warn = now
            self.get_logger().warning(message)


def main(args=None) -> None:
    node = None
    exit_signal = threading.Event()

    def handle_exit_signal(signum, _frame) -> None:
        exit_signal.set()
        if node is not None:
            node.request_exit("signal_%s" % signum)

    previous_sigint = signal.signal(signal.SIGINT, handle_exit_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_exit_signal)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = IcnNode()
        while (
            rclpy.ok()
            and not node.exit_requested
            and not exit_signal.is_set()
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
    except (RuntimeError, ValueError) as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print("ICN 任务节点启动失败: %s" % error)
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
