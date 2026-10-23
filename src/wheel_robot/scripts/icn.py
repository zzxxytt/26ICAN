#!/usr/bin/env python3
"""ICN 比赛任务节点：以巡线为主体，按状态机处理赛道上出现的其他元素。

当前实现：
    LINE   —— 黑线巡线（完全复用 line_follower.py 的检测器与控制律）
    ZEBRA  —— 斑马线（复用 zebra_crossing_detector.py 的纯视觉函数）
    STAIR  —— 台阶（复用 step_detector.py，默认动作=停车，动作可替换）
    FINISH —— 任务结束，原地停车

为什么合并成一个节点：
    1) /dev/video0 同一时刻只能被一个进程稳定打开；
    2) /cmd_vel 只能有一个"决策者"；
    3) 三个检测必须看同一帧，否则"看到元素"和"正在压线"对不上。
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
except ImportError as error:  # 部署问题，直接说清楚
    raise SystemExit(
        "icn.py 需要和 line_follower.py、zebra_crossing_detector.py、"
        f"step_detector.py 放在同一目录：{error}"
    )


class MissionState(str, Enum):
    """任务状态。同一时刻只有一个状态，也就只有一个模块拥有控制权。"""

    LINE = "LINE"      # 巡线（默认状态，比赛主体时间都在这里）
    ZEBRA = "ZEBRA"    # 斑马线：可先停车等待，再低速直行通过
    STAIR = "STAIR"    # 台阶：默认停车，动作内容可替换
    FINISH = "FINISH"  # 任务结束，停车


# 只声明父类没有的参数（父类已有的斑马线参数直接复用，重复声明会报错）
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

    # ---------------- 台阶 ----------------
    "step_enabled": True,
    "step_decimate": 2,
    "step_confirm_frames": 3,
    # 动作插槽：stop / wait_climb / climb
    "step_action": "stop",
    "step_hold_seconds": 0.0,
    "step_stop_seconds": 1.0,
    "step_climb_speed": 0.15,
    "step_climb_seconds": 3.0,
    "step_cooldown_seconds": 2.0,
    "step_exit_grace_frames": 10,
    # 检测阈值（改这里即可，不用动 step_detector.py）
    "step_center_left_ratio": 0.30,
    "step_center_right_ratio": 0.70,
    "step_bright_value_min": 110,
    "step_gap_bright_min": 0.55,
    "step_min_gap_rows": 12,

    "mission_timeout_seconds": 0.0,
}


class IcnNode(line_lib.LineFollower):
    """巡线节点 + 元素状态机。父类负责"怎么看"，这里负责"听谁的"。"""

    def __init__(self) -> None:
        super().__init__()  # 摄像头、检测器、控制、调试窗全在这里建好

        for name, default in ELEMENT_DEFAULTS.items():
            self.declare_parameter(name, default)
        for name, default in ELEMENT_DEFAULTS.items():
            setattr(self, name, self._read_param(name, default))

        # ---- 任务状态 ----
        self.state = MissionState.LINE
        self._mission_start = time.monotonic()
        self._crossings = 0

        # ---- 斑马线多帧确认 ----
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

        # ---- 台阶 ----
        self._step_cfg = step_lib.StepConfig(
            top_ratio=self.search_top_ratio,
            bottom_ratio=self.search_bottom_ratio,
            left_ratio=self.search_left_ratio,
            right_ratio=self.search_right_ratio,
            center_left_ratio=self.step_center_left_ratio,
            center_right_ratio=self.step_center_right_ratio,
            dark_value_max=self.hsv_upper_v,
            bright_value_min=self.step_bright_value_min,
            gap_bright_min=self.step_gap_bright_min,
            min_gap_rows=self.step_min_gap_rows,
        )
        self._step_cycle = 0
        self._step_hit = 0
        self._step_confirmed = False
        self._step_armed = True
        self._step_cooldown_until = 0.0
        self._stair_enter_time = 0.0
        self._step_gap = 0
        self._step_bright = 0.0
        self._step_last: Optional[step_lib.StepResult] = None

        # ---- 输出话题 ----
        self._zebra_pub = self._zebra_detected_publisher or self.create_publisher(
            Bool, "/zebra_crossing/detected", 1
        )
        self._zebra_count_pub = (
            self._zebra_count_publisher
            or self.create_publisher(UInt8, "/zebra_crossing/stripe_count", 1)
        )
        self._step_pub = self.create_publisher(Bool, "/step/detected", 1)
        self._state_pub = self.create_publisher(String, "/icn/state", 10)
        self._publish_state()

        self.get_logger().info(
            f"ICN 任务层已加载：斑马线={'启用' if self.enable_zebra else '关闭'}，"
            f"台阶={'启用' if self.step_enabled else '关闭'}"
            f"（动作={self.step_action}），斑马线目标次数={self.zebra_target_count}"
        )

    # ============================================================ 参数工具
    def _read_param(self, name: str, default):
        value = self.get_parameter(name).value
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, int):
            return int(value)
        return str(value)

    # ============================================================ 斑马线检测
    def _update_zebra_element(self, frame: np.ndarray) -> None:
        """每 zebra_decimate 帧跑一次斑马线检测，并做多帧确认/释放。"""
        if not self.enable_zebra:
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
            )
        except Exception as error:
            self._warn(f"斑马线检测异常: {error}")
            return

        self._zebra_debug = result.debug_image
        self._zebra_count = int(result.stripe_count)

        if result.zebra_exists:
            self._zebra_hit += 1
            self._zebra_miss = 0
        else:
            self._zebra_miss += 1
            self._zebra_hit = 0

        if not self._zebra_confirmed and self._zebra_hit >= self.zebra_confirm_frames:
            self._zebra_confirmed = True
            self.get_logger().info(
                f"斑马线确认: stripes={self._zebra_count}, "
                f"angle={result.dominant_angle_deg:+.1f}"
            )
        elif self._zebra_confirmed and self._zebra_miss >= self.zebra_release_frames:
            self._zebra_confirmed = False
            self.get_logger().info("斑马线已离开视野")

        if not self._zebra_confirmed and self._zebra_miss >= self.zebra_release_frames:
            self._zebra_armed = True

        self._zebra_pub.publish(Bool(data=self._zebra_confirmed))
        self._zebra_count_pub.publish(UInt8(data=min(self._zebra_count, 255)))

    # ============================================================ 台阶检测
    def _update_step_element(self, frame: np.ndarray) -> None:
        """每 step_decimate 帧跑一次台阶检测。"""
        if not self.step_enabled:
            return
        self._step_cycle += 1
        if self.step_decimate > 1 and self._step_cycle % self.step_decimate != 0:
            return

        try:
            step = step_lib.detect_step(frame, self._step_cfg)
        except Exception as error:
            self._warn(f"台阶检测异常: {error}")
            return

        self._step_last = step
        self._step_gap = step.gap_rows
        self._step_bright = step.gap_bright_ratio

        if step.detected:
            self._step_hit += 1
        else:
            self._step_hit = 0

        if not self._step_confirmed and self._step_hit >= self.step_confirm_frames:
            self._step_confirmed = True
            self.get_logger().info(
                f"台阶确认: gap={step.gap_rows}px bright={step.gap_bright_ratio:.2f} "
                f"near={step.near_rows} near_x={step.near_center_x:.2f} "
                f"动作={self.step_action}"
            )
        elif self._step_confirmed and self._step_hit == 0:
            self._step_confirmed = False

        self._step_pub.publish(Bool(data=self._step_confirmed))

    # ============================================================ 状态机
    def _set_state(self, state: MissionState, reason: str) -> None:
        if self.state is state:
            return
        self.state = state
        self.get_logger().info(f"状态切换 -> {state.value}（{reason}）")
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
        self._stair_enter_time = now
        self._set_state(
            MissionState.STAIR, f"step_confirmed action={self.step_action}"
        )

    def _leave_stair(self, now: float) -> None:
        self._step_cooldown_until = now + self.step_cooldown_seconds
        self._step_confirmed = False
        self._step_armed = False
        self._step_hit = 0
        self._have_previous_error = False
        self._lost_hold = int(self.step_exit_grace_frames)
        self._set_state(MissionState.LINE, "step_action_done")

    def _step_action_command(self, now: float):
        """台阶动作入口 —— 换动作只改这个函数。

        返回 (linear, angular, finished)：
            finished=False 继续执行本动作
            finished=True  本动作结束，立刻交回巡线
        """
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

        self._warn(f"未知 step_action={action}，按停车处理")
        return 0.0, 0.0, False

    def _enter_finish(self, reason: str) -> None:
        self._set_state(MissionState.FINISH, reason)
        self._publish_zero(reason)

    def _step_state(self, now: float) -> None:
        """只做状态迁移，不发速度。"""
        if (
            self.mission_timeout_seconds > 0.0
            and now - self._mission_start > self.mission_timeout_seconds
        ):
            if self.state is not MissionState.FINISH:
                self._enter_finish("mission_timeout")
            return

        if self.state is MissionState.LINE:
            if (
                self.enable_zebra
                and self._zebra_armed
                and self._zebra_confirmed
                and now >= self._zebra_cooldown_until
            ):
                self._zebra_enter_time = now
                self._set_state(MissionState.ZEBRA, "zebra_confirmed")
                return
            if (
                self.step_enabled
                and self._step_armed
                and self._step_confirmed
                and now >= self._step_cooldown_until
            ):
                self._enter_stair(now)
            return

        if self.state is MissionState.ZEBRA:
            elapsed = now - self._zebra_enter_time
            if elapsed > self.zebra_cross_max_seconds:
                self.get_logger().warning(
                    f"斑马线通过超时（{elapsed:.1f}s），强制退回巡线"
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
                self.get_logger().info(f"完成第 {self._crossings} 次斑马线通过")
                if (
                    self.zebra_target_count > 0
                    and self._crossings >= self.zebra_target_count
                ):
                    self._enter_finish("zebra_target_reached")

    # ============================================================ 控制
    def _steering(self, now: float, result) -> float:
        """巡线误差 -> 角速度（低通 + PD + 限幅）。"""
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
        return float(np.clip(raw, -self.max_angular_speed, self.max_angular_speed))

    def _emit(self, now: float, linear: float, angular: float) -> tuple:
        """唯一速度出口：运动开启时限加速度后发布，架空时发零只做显示。"""
        if not self.enable_motion:
            self._publish_zero("motion_disabled")
            return linear, angular
        cmd_linear, cmd_angular = self._limit_command_rate(linear, angular, now)
        self._publish_command(cmd_linear, cmd_angular)
        self._last_command = (cmd_linear, cmd_angular)
        self._last_command_valid = True
        return cmd_linear, cmd_angular

    def _run_control_cycle(self) -> None:
        """每一拍：取帧 -> 三个检测 -> 状态迁移 -> 统一出口。"""
        now = time.monotonic()
        with self._frame_lock:
            camera_healthy = self._camera_healthy
            frame_time = self._latest_frame_time
            sequence = self._frame_sequence
            frame = (
                self._latest_frame.copy() if self._latest_frame is not None else None
            )

        if not camera_healthy or frame is None:
            self._publish_zero("camera_unavailable")
            self._log_status(now, "STOP 摄像头不可用")
            return
        if now - frame_time > self.image_timeout_seconds:
            self._publish_zero("image_timeout")
            self._log_status(now, f"STOP 图像超时 {now - frame_time:.3f}s")
            return
        if sequence == self._last_processed_sequence:
            if self.enable_motion and self._last_command_valid:
                self._publish_command(*self._last_command)
            else:
                self._publish_zero("no_new_frame")
            return
        self._last_processed_sequence = sequence

        result = self.detector.detect(frame)        # 巡线：每帧都跑
        self._update_zebra_element(frame)           # 斑马线：抽帧跑
        self._update_step_element(frame)            # 台阶：抽帧跑
        self._step_state(now)                       # 状态迁移

        # ---------------- FINISH ----------------
        if self.state is MissionState.FINISH:
            self._publish_zero("mission_finish")
            self._show_debug(result, 0.0, 0.0, False)
            self._log_status(now, "FINISH 任务完成，原地停车")
            return

        # ---------------- ZEBRA ----------------
        if self.state is MissionState.ZEBRA:
            elapsed = now - self._zebra_enter_time
            if elapsed < self.zebra_stop_seconds:
                self._publish_zero("zebra_stop")
                self._show_debug(result, 0.0, 0.0, False)
                self._log_status(
                    now,
                    f"ZEBRA 停车等待 {elapsed:.1f}/{self.zebra_stop_seconds:.1f}s",
                )
                return
            cmd_linear, cmd_angular = self._emit(now, self.zebra_cross_speed, 0.0)
            self._show_debug(result, cmd_linear, cmd_angular, self.enable_motion)
            self._log_status(
                now,
                f"ZEBRA 直行通过 v={cmd_linear:.3f} w={cmd_angular:+.3f} "
                f"stripes={self._zebra_count}",
            )
            return

        # ---------------- STAIR ----------------
        if self.state is MissionState.STAIR:
            linear, angular, finished = self._step_action_command(now)
            if finished:
                self._leave_stair(now)      # 交回巡线，本拍继续往下走 LINE
            else:
                cmd_linear, cmd_angular = self._emit(now, linear, angular)
                self._show_debug(result, cmd_linear, cmd_angular, self.enable_motion)
                self._log_status(
                    now,
                    f"STAIR 动作={self.step_action} v={cmd_linear:.3f} "
                    f"gap={self._step_gap}px bright={self._step_bright:.2f}",
                )
                return

        # ---------------- LINE ----------------
        if not result.valid:
            if (
                self.enable_motion
                and self._last_command_valid
                and self._lost_hold > 0
            ):
                self._lost_hold -= 1
                self._publish_command(*self._last_command)
                self._show_debug(
                    result, self._last_command[0], self._last_command[1], True
                )
                self._log_status(
                    now, f"LINE 丢线保持 {result.reason}（剩 {self._lost_hold} 帧）"
                )
                return
            self._publish_zero(result.reason)
            self._show_debug(result, 0.0, 0.0, False)
            self._log_status(now, f"STOP {result.reason} conf={result.confidence:.2f}")
            return

        self._lost_hold = self.lost_hold_frames
        approaching = self._zebra_armed and self._zebra_hit > 0
        scale = self.zebra_approach_slowdown if approaching else 1.0
        cmd_linear, cmd_angular = self._emit(
            now, self.forward_speed * scale, self._steering(now, result)
        )
        self._show_debug(result, cmd_linear, cmd_angular, self.enable_motion)
        self._log_status(
            now,
            f"LINE v={cmd_linear:.3f} w={cmd_angular:+.3f} "
            f"err={result.error:+.3f} conf={result.confidence:.2f}"
            + (" 减速接近斑马线" if approaching else ""),
        )

    # ============================================================ 调试显示
    def _show_debug(
        self, result, planned_linear: float, planned_angular: float, motion_active: bool
    ) -> None:
        """斑马线小窗 + 任务状态行 + 台阶状态条 + 台阶缺口框，再交给父类显示。"""
        debug = result.debug_frame

        # 右上角：斑马线鸟瞰图
        if self._zebra_debug is not None:
            try:
                panel = cv2.resize(self._zebra_debug, (240, 150))
                panel_h, panel_w = panel.shape[:2]
                x = debug.shape[1] - panel_w - 8
                y = 66
                if x > 0 and y + panel_h <= debug.shape[0]:
                    debug[y : y + panel_h, x : x + panel_w] = panel
                    cv2.rectangle(
                        debug, (x - 1, y - 1), (x + panel_w, y + panel_h),
                        (255, 180, 0), 1,
                    )
            except cv2.error:
                pass

        # 任务状态行
        target = "inf" if self.zebra_target_count <= 0 else str(self.zebra_target_count)
        cv2.putText(
            debug,
            f"MISSION {self.state.value}  zebra={int(self._zebra_confirmed)}"
            f"/{self._zebra_count}  cross={self._crossings}/{target}",
            (12, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 台阶状态条 + 缺口框
        step = self._step_last
        if step is not None:
            if self._step_confirmed:
                text = (
                    f"STEP DETECTED  gap={step.gap_rows}px "
                    f"bright={step.gap_bright_ratio:.2f}  action={self.step_action}"
                )
                color = (
                    (0, 255, 255) if self.state is MissionState.STAIR else (0, 0, 255)
                )
                thickness = 2
            else:
                text = (
                    f"STEP no  gap={step.gap_rows}px "
                    f"bright={step.gap_bright_ratio:.2f}  {step.reason}"
                )
                color = (180, 180, 180)
                thickness = 1
            cv2.putText(
                debug, text, (12, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA,
            )
            if step.gap_rows > 0:
                frame_h, frame_w = debug.shape[:2]
                rx0 = int(frame_w * self.search_left_ratio)
                rx1 = int(frame_w * self.search_right_ratio)
                ry0 = int(frame_h * self.search_top_ratio)
                rx_center0 = rx0 + int(
                    (rx1 - rx0) * self.step_center_left_ratio
                )
                rx_center1 = rx0 + int(
                    (rx1 - rx0) * self.step_center_right_ratio
                )
                cv2.rectangle(
                    debug,
                    (rx_center0, ry0 + step.gap_start_row),
                    (rx_center1 - 1, ry0 + step.gap_start_row + step.gap_rows - 1),
                    (0, 0, 255) if self._step_confirmed else (0, 165, 255),
                    2,
                )

        super()._show_debug(result, planned_linear, planned_angular, motion_active)

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
            node.request_exit(f"signal_{signum}")

    previous_sigint = signal.signal(signal.SIGINT, handle_exit_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_exit_signal)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = IcnNode()
        while rclpy.ok() and not node.exit_requested and not exit_signal.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (RuntimeError, ValueError) as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f"ICN 任务节点启动失败: {error}")
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