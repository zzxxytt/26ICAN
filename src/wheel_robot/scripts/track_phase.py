#!/usr/bin/env python3
"""赛道阶段与元素顺序管理（纯 Python，无 ROS/OpenCV）。

一圈固定顺序：
    FIRST_CORNER -> SEEK_PLATFORM -> CROSS_PLATFORM -> SEEK_ROUNDABOUT
    -> ROUNDABOUT -> SEEK_ZEBRA -> LAP_COMPLETE -> 下一圈 FIRST_CORNER
    roundabout_mode=off 时跳过 SEEK_ROUNDABOUT / ROUNDABOUT，继续巡线找斑马线。

规则：
    * 每个阶段只允许自己的元素触发；may_trigger 还要求本圈未消费过该元素；
    * 阶段只能单向前进，下一圈由 start_next_lap() 显式重置；
    * 切换阶段通过 on_change 通知外层清空确认计数和残留；
    * 首弯结束 = 累计 |误差|×dt + 峰值误差 + 连续直线帧，不靠开机延时；
    * 台阶动作结束 != 越过平台，还要“巡线恢复 + 缺口消失”连续成立。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Dict, FrozenSet, Optional, Tuple


class TrackPhase(str, Enum):
    FIRST_CORNER = "FIRST_CORNER"          # 起步至首弯结束：不开放元素
    SEEK_PLATFORM = "SEEK_PLATFORM"        # 只开放台阶
    CROSS_PLATFORM = "CROSS_PLATFORM"      # 台阶动作/通过：禁止新元素抢占
    SEEK_ROUNDABOUT = "SEEK_ROUNDABOUT"    # 只开放圆环
    ROUNDABOUT = "ROUNDABOUT"              # 只有圆环流程
    SEEK_ZEBRA = "SEEK_ZEBRA"              # 只开放斑马线
    LAP_COMPLETE = "LAP_COMPLETE"          # 本圈完成，等待下一圈


class TrackElement(str, Enum):
    STEP = "STEP"
    ROUNDABOUT = "ROUNDABOUT"
    ZEBRA = "ZEBRA"


PHASE_ORDER: Tuple[TrackPhase, ...] = (
    TrackPhase.FIRST_CORNER,
    TrackPhase.SEEK_PLATFORM,
    TrackPhase.CROSS_PLATFORM,
    TrackPhase.SEEK_ROUNDABOUT,
    TrackPhase.ROUNDABOUT,
    TrackPhase.SEEK_ZEBRA,
    TrackPhase.LAP_COMPLETE,
)

PHASE_ELEMENTS: Dict[TrackPhase, FrozenSet[TrackElement]] = {
    TrackPhase.FIRST_CORNER: frozenset(),
    TrackPhase.SEEK_PLATFORM: frozenset({TrackElement.STEP}),
    TrackPhase.CROSS_PLATFORM: frozenset(),
    TrackPhase.SEEK_ROUNDABOUT: frozenset({TrackElement.ROUNDABOUT}),
    TrackPhase.ROUNDABOUT: frozenset({TrackElement.ROUNDABOUT}),
    TrackPhase.SEEK_ZEBRA: frozenset({TrackElement.ZEBRA}),
    TrackPhase.LAP_COMPLETE: frozenset(),
}

PHASE_LABELS: Dict[TrackPhase, str] = {
    TrackPhase.FIRST_CORNER: "首弯",
    TrackPhase.SEEK_PLATFORM: "找平台",
    TrackPhase.CROSS_PLATFORM: "过平台",
    TrackPhase.SEEK_ROUNDABOUT: "找圆环",
    TrackPhase.ROUNDABOUT: "绕圆环",
    TrackPhase.SEEK_ZEBRA: "找斑马线",
    TrackPhase.LAP_COMPLETE: "本圈完成",
}

PHASE_COURSE: Dict[TrackPhase, str] = {
    TrackPhase.FIRST_CORNER: "START",
    TrackPhase.SEEK_PLATFORM: "SEEK_PLATFORM",
    TrackPhase.CROSS_PLATFORM: "CROSS_PLATFORM",
    TrackPhase.SEEK_ROUNDABOUT: "SEEK_RING",
    TrackPhase.ROUNDABOUT: "RING",
    TrackPhase.SEEK_ZEBRA: "SEEK_ZEBRA",
    TrackPhase.LAP_COMPLETE: "LAP_DONE",
}

# 圆环子状态机里代表“入口已锁定”的状态；CONFIRM 只是候选。
RING_ENTRY_STATES: FrozenSet[str] = frozenset(
    {"ENTER", "FOLLOW", "EXIT_READY", "EXIT"}
)


@dataclass(frozen=True)
class PhaseConfig:
    # ---- 首弯结束 ----
    corner_error_threshold: float = 0.30
    corner_min_peak_error: float = 0.55
    corner_min_turn: float = 0.30
    straight_error_max: float = 0.12
    straight_min_confidence: float = 0.55
    straight_confirm_frames: int = 12
    first_corner_timeout_seconds: float = 0.0   # 0=禁用，仅调试

    # ---- 平台通过 ----
    platform_settle_seconds: float = 0.30
    platform_clear_frames: int = 10
    platform_clear_min_confidence: float = 0.55
    platform_clear_error_max: float = 0.20

    # ---- 圆环 ----
    roundabout_min_seconds: float = 1.0
    roundabout_clear_frames: int = 8

    # ---- 下一圈 ----
    lap_restart_clear_frames: int = 8

    def validate(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool):
                continue
            if not math.isfinite(value):
                raise ValueError("track_phase.%s 必须是有限数值" % name)
        if not 0.0 < self.corner_error_threshold <= 1.0:
            raise ValueError("corner_error_threshold 必须在 (0,1]")
        if not 0.0 < self.corner_min_peak_error <= 1.0:
            raise ValueError("corner_min_peak_error 必须在 (0,1]")
        if self.corner_min_turn < 0.0:
            raise ValueError("corner_min_turn 不能为负")
        if not 0.0 < self.straight_error_max <= 1.0:
            raise ValueError("straight_error_max 必须在 (0,1]")
        if not 0.0 < self.straight_min_confidence <= 1.0:
            raise ValueError("straight_min_confidence 必须在 (0,1]")
        if self.straight_confirm_frames < 2:
            raise ValueError("straight_confirm_frames 至少为 2")
        if self.first_corner_timeout_seconds < 0.0:
            raise ValueError("first_corner_timeout_seconds 不能为负")
        if self.platform_settle_seconds < 0.0:
            raise ValueError("platform_settle_seconds 不能为负")
        if self.platform_clear_frames < 2:
            raise ValueError("platform_clear_frames 至少为 2")
        if not 0.0 < self.platform_clear_min_confidence <= 1.0:
            raise ValueError("platform_clear_min_confidence 必须在 (0,1]")
        if not 0.0 < self.platform_clear_error_max <= 1.0:
            raise ValueError("platform_clear_error_max 必须在 (0,1]")
        if self.roundabout_min_seconds < 0.0:
            raise ValueError("roundabout_min_seconds 不能为负")
        if self.roundabout_clear_frames < 2:
            raise ValueError("roundabout_clear_frames 至少为 2")
        if self.lap_restart_clear_frames < 2:
            raise ValueError("lap_restart_clear_frames 至少为 2")


@dataclass(frozen=True)
class PhaseTransition:
    old: TrackPhase
    new: TrackPhase
    lap: int
    reason: str
    detail: str = ""

    def text(self) -> str:
        suffix = "，%s" % self.detail if self.detail else ""
        return "赛道阶段 %s→%s（第 %d 圈，%s%s）" % (
            PHASE_LABELS[self.old], PHASE_LABELS[self.new],
            self.lap, self.reason, suffix,
        )


class TrackPhaseManager:
    def __init__(
        self,
        config: Optional[PhaseConfig] = None,
        start_phase: TrackPhase = TrackPhase.FIRST_CORNER,
        on_change: Optional[Callable[[PhaseTransition], None]] = None,
        roundabout_enabled: bool = True,
    ) -> None:
        self.cfg = config if config is not None else PhaseConfig()
        self.cfg.validate()
        if not isinstance(start_phase, TrackPhase):
            start_phase = TrackPhase(str(start_phase))
        self.roundabout_enabled = bool(roundabout_enabled)
        self.phase = (
            TrackPhase.SEEK_ZEBRA
            if not self.roundabout_enabled and start_phase in (
                TrackPhase.SEEK_ROUNDABOUT, TrackPhase.ROUNDABOUT
            ) else start_phase
        )
        self.lap = 1
        self.phase_since: Optional[float] = None
        self.last_transition: Optional[PhaseTransition] = None
        self._on_change = on_change
        self._consumed: Dict[TrackElement, bool] = {
            element: False for element in TrackElement
        }
        self._reset_phase_evidence()

    # ---------------- 查询 ----------------
    def permits(self, element: TrackElement) -> bool:
        if element is TrackElement.ROUNDABOUT and not self.roundabout_enabled:
            return False
        return element in PHASE_ELEMENTS[self.phase]

    def set_roundabout_enabled(self, enabled: bool, now: float) -> None:
        # 独立于阶段证据保存，下一圈重置证据时不能重新打开圆环。
        self.roundabout_enabled = bool(enabled)
        if not self.roundabout_enabled and self.phase in (
            TrackPhase.SEEK_ROUNDABOUT, TrackPhase.ROUNDABOUT
        ):
            self._advance(TrackPhase.SEEK_ZEBRA, now, "roundabout_disabled",
                          "圆环关闭，继续巡线寻找斑马线")

    def may_trigger(self, element: TrackElement) -> bool:
        return self.permits(element) and not self._consumed[element]

    def is_consumed(self, element: TrackElement) -> bool:
        return self._consumed[element]

    def allowed_elements(self) -> Tuple[TrackElement, ...]:
        return tuple(e for e in TrackElement if self.permits(e))

    def phase_age(self, now: float) -> float:
        return 0.0 if self.phase_since is None else max(0.0, now - self.phase_since)

    @property
    def platform_done(self) -> bool:
        return self._platform_done
    
    def course_name(self) -> str:
        return PHASE_COURSE[self.phase]

    def describe(self) -> str:
        allowed = ",".join(e.value for e in self.allowed_elements()) or "-"
        return "%s/L%d[allow=%s]" % (self.phase.value, self.lap, allowed)

    def evidence_text(self) -> str:
        cfg = self.cfg
        if self.phase is TrackPhase.FIRST_CORNER:
            return "turn=%.2f/%.2f peak=%.2f/%.2f straight=%d/%d" % (
                self._turn_progress, cfg.corner_min_turn,
                self._turn_peak, cfg.corner_min_peak_error,
                self._straight_frames, cfg.straight_confirm_frames,
            )
        if self.phase is TrackPhase.CROSS_PLATFORM:
            if not self._platform_done:
                return "platform action running"
            return "line_clear=%d/%d step_clear=%d/%d" % (
                self._line_clear_frames, cfg.platform_clear_frames,
                self._step_clear_frames, cfg.platform_clear_frames,
            )
        if self.phase is TrackPhase.ROUNDABOUT:
            return "completed=%d junction_clear=%d/%d" % (
                int(self._ring_completed), self._ring_clear_frames,
                cfg.roundabout_clear_frames,
            )
        if self.phase is TrackPhase.LAP_COMPLETE:
            return "lap_clear=%d/%d" % (
                self._lap_clear_frames, cfg.lap_restart_clear_frames
            )
        return "waiting for element"

    # ---------------- 内部工具 ----------------
    def _ensure_phase_start(self, now: float) -> None:
        if self.phase_since is None:
            self.phase_since = now

    def _report(self, old, new, now, reason, detail="") -> None:
        transition = PhaseTransition(old, new, self.lap, reason, detail)
        self.last_transition = transition
        if self._on_change is not None:
            self._on_change(transition)

    def _advance(self, new_phase, now, reason, detail="") -> bool:
        old = self.phase
        try:
            new_index = PHASE_ORDER.index(new_phase)
            old_index = PHASE_ORDER.index(old)
        except ValueError:
            return False
        if new_index <= old_index:
            return False
        self.phase = new_phase
        self.phase_since = now
        self._reset_phase_evidence()
        self._report(old, new_phase, now, reason, detail)
        return True

    def _reset_phase_evidence(self) -> None:
        self._last_line_time: Optional[float] = None
        # 首弯
        self._turn_progress = 0.0
        self._turn_peak = 0.0
        self._turn_seen = False
        self._straight_frames = 0
        # 平台
        self._platform_started = False
        self._platform_done = False
        self._platform_done_time = 0.0
        self._step_clear_required = True
        self._line_clear_frames = 0
        self._step_clear_frames = 0
        # 圆环
        self._ring_entry_time = 0.0
        self._ring_entry_completed = 0
        self._ring_completed = False
        self._ring_clear_frames = 0
        # 下一圈
        self._lap_clear_frames = 0

    # ---------------- 巡线证据 ----------------
    def observe_line(self, now, valid, error, confidence, corner=False) -> None:
        self._ensure_phase_start(now)
        dt = 0.0 if self._last_line_time is None else min(
            0.25, max(0.0, now - self._last_line_time)
        )
        self._last_line_time = now
        if self.phase is TrackPhase.FIRST_CORNER:
            self._observe_first_corner(now, valid, error, confidence, corner, dt)
        elif self.phase is TrackPhase.CROSS_PLATFORM:
            self._observe_platform_line(now, valid, error, confidence)
        elif self.phase is TrackPhase.LAP_COMPLETE:
            self._observe_lap_restart(now, valid, confidence)

    def _observe_first_corner(self, now, valid, error, confidence, corner, dt):
        cfg = self.cfg
        if valid:
            magnitude = abs(float(error))
            self._turn_peak = max(self._turn_peak, magnitude)
            if corner or magnitude >= cfg.corner_error_threshold:
                effective = (
                    max(magnitude, cfg.corner_min_peak_error) if corner else magnitude
                )
                self._turn_progress += effective * dt
            if (self._turn_peak >= cfg.corner_min_peak_error
                    and self._turn_progress >= cfg.corner_min_turn):
                self._turn_seen = True
            straight_now = (
                self._turn_seen and not corner
                and magnitude <= cfg.straight_error_max
                and confidence >= cfg.straight_min_confidence
            )
            self._straight_frames = self._straight_frames + 1 if straight_now else 0
        else:
            self._straight_frames = 0

        if self._straight_frames >= cfg.straight_confirm_frames:
            self._advance(
                TrackPhase.SEEK_PLATFORM, now, "first_corner_completed",
                "turn=%.2f peak=%.2f" % (self._turn_progress, self._turn_peak),
            )
            return
        if (cfg.first_corner_timeout_seconds > 0.0
                and self.phase_age(now) >= cfg.first_corner_timeout_seconds):
            self._advance(
                TrackPhase.SEEK_PLATFORM, now, "first_corner_timeout",
                "仅调试兜底，正常比赛请关闭",
            )

    def _observe_platform_line(self, now, valid, error, confidence):
        cfg = self.cfg
        if not self._platform_done:
            return
        if now - self._platform_done_time < cfg.platform_settle_seconds:
            return
        if (valid and confidence >= cfg.platform_clear_min_confidence
                and abs(float(error)) <= cfg.platform_clear_error_max):
            self._line_clear_frames += 1
        else:
            self._line_clear_frames = 0
        self._check_platform_cleared(now)

    def _observe_lap_restart(self, now, valid, confidence):
        if valid and confidence >= self.cfg.straight_min_confidence:
            self._lap_clear_frames += 1
        else:
            self._lap_clear_frames = 0
        if self._lap_clear_frames >= self.cfg.lap_restart_clear_frames:
            self.start_next_lap(now)

    # ---------------- 平台证据 ----------------
    def begin_platform(self, now) -> bool:
        if self.phase is not TrackPhase.SEEK_PLATFORM:
            return False
        if self._consumed[TrackElement.STEP]:
            return False
        if not self._advance(TrackPhase.CROSS_PLATFORM, now,
                             "step_confirmed", "平台动作开始"):
            return False
        self._consumed[TrackElement.STEP] = True
        self._platform_started = True
        self._platform_done = False
        self._line_clear_frames = 0
        self._step_clear_frames = 0
        return True

    def platform_action_done(self, now, step_available=True) -> None:
        if self.phase is not TrackPhase.CROSS_PLATFORM:
            return
        self._platform_done = True
        self._platform_done_time = now
        self._line_clear_frames = 0
        self._step_clear_frames = 0
        self._step_clear_required = bool(step_available)

    def observe_step(self, now, detected) -> None:
        if self.phase is not TrackPhase.CROSS_PLATFORM:
            return
        if not self._platform_done or not self._step_clear_required:
            return
        self._step_clear_frames = 0 if detected else self._step_clear_frames + 1
        self._check_platform_cleared(now)

    def _check_platform_cleared(self, now) -> None:
        cfg = self.cfg
        if self.phase is not TrackPhase.CROSS_PLATFORM or not self._platform_done:
            return
        if now - self._platform_done_time < cfg.platform_settle_seconds:
            return
        if self._line_clear_frames < cfg.platform_clear_frames:
            return
        if (self._step_clear_required
                and self._step_clear_frames < cfg.platform_clear_frames):
            return
        self._advance(
            TrackPhase.SEEK_ROUNDABOUT if self.roundabout_enabled else TrackPhase.SEEK_ZEBRA,
            now, "platform_cleared" if self.roundabout_enabled else "roundabout_disabled",
            "line_clear=%d step_clear=%d step_required=%d" % (
                self._line_clear_frames, self._step_clear_frames,
                int(self._step_clear_required),
            ),
        )

    # ---------------- 圆环证据 ----------------
    def observe_roundabout(self, now, mode="off", state="", completed=0,
                           junction=False) -> None:
        self._ensure_phase_start(now)
        self.set_roundabout_enabled(str(mode).lower() in ("observe", "control"), now)
        if not self.roundabout_enabled:
            return
        if self.phase is TrackPhase.SEEK_ROUNDABOUT:
            if state in RING_ENTRY_STATES:
                if self._advance(TrackPhase.ROUNDABOUT, now,
                                 "roundabout_entry",
                                 "state=%s mode=%s" % (state, mode)):
                    self._consumed[TrackElement.ROUNDABOUT] = True
                    self._ring_entry_completed = int(completed)
                    self._ring_entry_time = now
            return

        if self.phase is not TrackPhase.ROUNDABOUT:
            return
        if int(completed) > self._ring_entry_completed:
            self._ring_completed = True
        if junction:
            self._ring_clear_frames = 0
        else:
            self._ring_clear_frames += 1

        if (self._ring_completed
                and self._ring_clear_frames >= self.cfg.roundabout_clear_frames):
            self._advance(
                TrackPhase.SEEK_ZEBRA, now, "roundabout_completed",
                "completed=%d junction_clear=%d" % (
                    int(completed), self._ring_clear_frames
                ),
            )
            return

        if (str(mode).lower() != "control"
                and self.phase_age(now) >= self.cfg.roundabout_min_seconds
                and self._ring_clear_frames >= self.cfg.roundabout_clear_frames):
            self._advance(
                TrackPhase.SEEK_ZEBRA, now,
                "roundabout_cleared_without_control",
                "mode=%s 未接管，按视觉离开判定" % mode,
            )

    # ---------------- 斑马线与下一圈 ----------------
    def zebra_entered(self, now) -> None:
        self._consumed[TrackElement.ZEBRA] = True

    def zebra_passed(self, now) -> bool:
        if self.phase is not TrackPhase.SEEK_ZEBRA:
            return False
        return self._advance(TrackPhase.LAP_COMPLETE, now,
                             "zebra_passed", "lap=%d" % self.lap)

    def start_next_lap(self, now) -> bool:
        if self.phase is not TrackPhase.LAP_COMPLETE:
            return False
        old = self.phase
        self.lap += 1
        self.phase = TrackPhase.FIRST_CORNER
        self.phase_since = now
        for element in TrackElement:
            self._consumed[element] = False
        self._reset_phase_evidence()
        self._report(old, TrackPhase.FIRST_CORNER, now,
                     "next_lap", "第 %d 圈" % self.lap)
        return True


__all__ = [
    "PHASE_ELEMENTS", "PHASE_LABELS", "PHASE_ORDER", "RING_ENTRY_STATES",
    "PhaseConfig", "PhaseTransition", "TrackElement", "TrackPhase",
    "TrackPhaseManager",
]
