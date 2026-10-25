#!/usr/bin/env python3
"""Dashed tangent roundabout: pure OpenCV perception and a ROS-free state machine.

Image coordinates: x is right, t is forward (bottom=0, top=1). CCW entry
from the bottom tangent is a LEFT bend in a forward-facing, unmirrored camera.
No whole-mask closing, largest-blob steering, timer-based lap completion, or
integration of commanded angular velocity is used here.
"""
from dataclasses import dataclass, field, fields, replace
from enum import Enum
import math
import json
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Config:
    roi_top: float = 0.42
    roi_bottom: float = 0.98
    roi_left: float = 0.02
    roi_right: float = 0.98
    work_width: int = 320
    work_height: int = 200
    black_max: int = 75
    adaptive: bool = True
    # Optional calibrated trapezoid, relative to the cropped ROI.
    perspective: bool = False
    top_left: float = 0.20
    top_right: float = 0.80
    bottom_left: float = 0.0
    bottom_right: float = 1.0
    min_width: float = 0.009
    max_width: float = 0.09
    max_gap: float = 0.24
    min_span: float = 0.48
    min_confidence: float = 0.50
    branch_separation: float = 0.08
    image_turn_sign: int = -1
    lookahead: float = 0.55
    max_track_jump: float = 0.16
    confirm_frames: int = 4
    exit_confirm_frames: int = 4
    recovery_frames: int = 8
    enter_speed: float = 0.08
    follow_speed: float = 0.12
    exit_speed: float = 0.08
    predict_speed: float = 0.05
    predict_seconds: float = 0.20
    predict_distance: float = 0.015
    entry_timeout: float = 8.0
    lap_timeout: float = 60.0
    exit_timeout: float = 6.0
    enter_degrees: float = 35.0
    exit_min_degrees: float = 325.0
    exit_max_degrees: float = 395.0
    exit_heading_tolerance: float = 25.0
    cooldown_seconds: float = 8.0
    rearm_clear_frames: int = 20
    sensor_timeout: float = 0.40
    max_yaw_rate: float = 3.0
    max_tilt_degrees: float = 20.0
    max_processing_seconds: float = 0.15

    def validate(self):
        for name, value in vars(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"roundabout.{name} must be finite")
        if not 0 <= self.roi_left < self.roi_right <= 1 or not 0 <= self.roi_top < self.roi_bottom <= 1:
            raise ValueError("invalid roundabout ROI")
        if not 0 <= self.top_left < self.top_right <= 1 or not 0 <= self.bottom_left < self.bottom_right <= 1:
            raise ValueError("invalid perspective trapezoid")
        if not 160 <= self.work_width <= 480 or not 100 <= self.work_height <= 320:
            raise ValueError("roundabout working image is outside bounded size")
        if not 0 < self.min_width < self.max_width < 0.3 or not 0 < self.max_gap < 0.5:
            raise ValueError("invalid width/gap limits")
        if not 0.3 <= self.min_span <= 1 or not 0 < self.min_confidence <= 1:
            raise ValueError("invalid support limits")
        if not 0 < self.branch_separation < 0.5 or not 0 < self.max_track_jump < 0.5:
            raise ValueError("invalid association gates")
        if self.image_turn_sign not in (-1, 1) or not 0.2 <= self.lookahead <= 0.8:
            raise ValueError("invalid image direction/lookahead")
        for name in ("confirm_frames", "exit_confirm_frames", "recovery_frames", "rearm_clear_frames"):
            if not 2 <= getattr(self, name) <= 100:
                raise ValueError(f"invalid {name}")
        for name in ("enter_speed", "follow_speed", "exit_speed", "predict_speed"):
            if not 0 < getattr(self, name) <= 0.4:
                raise ValueError(f"invalid {name}")
        if self.predict_speed > min(self.enter_speed, self.follow_speed, self.exit_speed):
            raise ValueError("prediction must not increase speed")
        for name in ("predict_seconds", "predict_distance", "entry_timeout", "lap_timeout", "exit_timeout", "cooldown_seconds", "sensor_timeout", "max_yaw_rate", "max_processing_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        if self.predict_seconds > 0.5 or self.predict_distance > 0.05:
            raise ValueError("prediction horizon too large")
        if not 10 <= self.enter_degrees < 90 or not 300 <= self.exit_min_degrees < 360 < self.exit_max_degrees <= 420:
            raise ValueError("invalid measured lap angle gates")
        if not 5 <= self.exit_heading_tolerance <= 40 or not 1 <= self.max_tilt_degrees <= 40:
            raise ValueError("invalid attitude gates")


@dataclass
class Path:
    coefficients: np.ndarray
    confidence: float
    dashes: int
    span: float
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    def x(self, t):
        return np.polyval(self.coefficients, t)

    def supports(self, t):
        return len(self.points) == 0 or self.points[0, 0]-0.03 <= t <= self.points[-1, 0]+0.03

    @property
    def bend(self):
        return float(2 * self.coefficients[0])

    @property
    def heading(self):
        return float(self.coefficients[0] + self.coefficients[1])

    @property
    def straight(self):
        return abs(self.bend) < 0.16 and abs(self.heading) < 0.30


@dataclass
class Observation:
    paths: List[Path] = field(default_factory=list)
    arc: Optional[Path] = None
    tangent: Optional[Path] = None
    junction: bool = False
    reason: str = "no_paths"
    debug: Optional[np.ndarray] = None
    inverse: Optional[np.ndarray] = None
    entry: dict = field(default_factory=dict)  # Diagnostics only; never a control gate.


class Vision:
    """Bounded beam search keeps competing branches separate across dash gaps."""
    def __init__(self, config: Config):
        config.validate()
        self.cfg = config

    def detect(self, frame: np.ndarray) -> Observation:
        c = self.cfg
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("expected uint8 BGR frame")
        h, w = frame.shape[:2]
        x0, x1 = int(w*c.roi_left), int(w*c.roi_right)
        y0, y1 = int(h*c.roi_top), int(h*c.roi_bottom)
        if x1-x0 < 16 or y1-y0 < 16:
            raise ValueError("ROI too small")
        left, right = (c.top_left, c.top_right) if c.perspective else (0.0, 1.0)
        bl, br = (c.bottom_left, c.bottom_right) if c.perspective else (0.0, 1.0)
        source = np.float32([[x0+left*(x1-x0-1), y0], [x0+right*(x1-x0-1), y0],
                             [x0+br*(x1-x0-1), y1-1], [x0+bl*(x1-x0-1), y1-1]])
        dest = np.float32([[0, 0], [c.work_width-1, 0], [c.work_width-1, c.work_height-1], [0, c.work_height-1]])
        transform = cv2.getPerspectiveTransform(source, dest)
        work = cv2.warpPerspective(frame, transform, (c.work_width, c.work_height), borderValue=(255, 255, 255))
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, c.black_max)
        if c.adaptive:
            local = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
            mask = cv2.bitwise_or(mask, local)
        # Do not close/dilate the gaps: that joins the tangent to the circle.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        obs = Observation(debug=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), inverse=np.linalg.inv(transform))
        if np.mean(mask > 0) > 0.45 or n > 1200:
            obs.reason = "shadow_or_noise"
            return obs
        paths = self._trace(mask, labels, stats)
        obs.paths = paths
        arcs = [p for p in paths if p.dashes >= 3 and p.supports(0.8) and p.bend*c.image_turn_sign > 0.10
                and (p.x(0.85)-p.x(0.15))*c.image_turn_sign > 0.06]
        tangents = [p for p in paths if p.straight]
        pairs = [(a, s) for a in arcs for s in tangents if a is not s and s.dashes >= 2 and s.supports(0.8)
                 and abs(a.x(0.10)-s.x(0.10)) < 0.16
                 and (a.x(0.85)-s.x(0.85))*c.image_turn_sign > c.branch_separation]
        if pairs:
            obs.arc, obs.tangent = max(pairs, key=lambda pair: pair[0].confidence+pair[1].confidence)
            obs.junction = True
        else:
            obs.arc = max(arcs, key=lambda p: p.confidence, default=None)
            obs.tangent = max(tangents, key=lambda p: p.confidence, default=None)
        # Expose the exact existing gates without relaxing them or changing paths.
        supported_tangents = [p for p in tangents if p.dashes >= 2 and p.supports(0.8)]
        near_pairs = [(a, s) for a in arcs for s in supported_tangents
                      if a is not s and abs(a.x(0.10)-s.x(0.10)) < 0.16]
        reject = ("OK" if pairs else "NO_PATHS" if not paths else
                  "NO_ARC" if not arcs else "NO_TANGENT" if not tangents else
                  "TANGENT_SUPPORT" if not supported_tangents else
                  "NEAR_GAP" if not near_pairs else "FAR_SEPARATION")
        path_details = []
        for p in paths:
            arc_fail = [name for name, passed in (
                ("dash", p.dashes >= 3), ("reach", p.supports(0.8)),
                ("bend", p.bend*c.image_turn_sign > 0.10),
                ("dir", (p.x(0.85)-p.x(0.15))*c.image_turn_sign > 0.06)) if not passed]
            tangent_fail = [name for name, passed in (
                ("bend", abs(p.bend) < 0.16), ("heading", abs(p.heading) < 0.30)) if not passed]
            path_details.append({"dashes": int(p.dashes), "confidence": float(p.confidence),
                                 "bend": p.bend, "heading": p.heading,
                                 "reach": bool(p.supports(0.8)),
                                 "support": ([float(p.points[0, 0]), float(p.points[-1, 0])]
                                             if len(p.points) else None),
                                 "arc_fail": arc_fail, "tangent_fail": tangent_fail})
        obs.entry = {"reject": reject, "arcs": len(arcs), "tangents": len(tangents),
                     "supported_tangents": len(supported_tangents),
                     "near_pairs": len(near_pairs), "pairs": len(pairs), "paths": path_details}
        obs.reason = "junction" if obs.junction else ("paths" if paths else "no_paths")
        for index, path in enumerate(paths):
            # Draw observed support, not a full-height extrapolation that can
            # make a shared-dash hypothesis look like an extra physical branch.
            lo, hi = (path.points[0, 0], path.points[-1, 0]) if len(path.points) else (0.03, 0.95)
            t = np.linspace(lo, hi, 50)
            xy = np.column_stack((path.x(t)*c.work_width, (1-t)*(c.work_height-1))).astype(np.int32)
            role = "A" if path is obs.arc else "T" if path is obs.tangent else "?"
            color = (0, 220, 0) if role == "A" else (255, 220, 0) if role == "T" else (140, 140, 140)
            cv2.polylines(obs.debug, [xy], False, color, 2 if role != "?" else 1)
            for pt, px in path.points[::3]:
                cv2.circle(obs.debug, (int(px*c.work_width), int((1-pt)*(c.work_height-1))), 1, color, -1)
            detail = path_details[index]
            arc_text = "+".join(detail["arc_fail"]) or "OK"
            tangent_text = "+".join(detail["tangent_fail"]) or "OK"
            cv2.putText(obs.debug, f"P{index}/{role} d={path.dashes} A:{arc_text} T:{tangent_text}",
                        (6, 14+index*16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)
        return obs

    def _trace(self, mask, labels, stats):
        c = self.cfg
        h, w = mask.shape
        stride = 4
        # Beam: (points [(t,x,label)], cost, slope). No pixels shared across
        # branch hypotheses are averaged together.
        beams = []
        finished = []
        for y in range(h-2, 0, -stride):
            t = 1-y/(h-1)
            diff = np.diff(np.pad((mask[y] > 0).astype(np.int8), (1, 1)))
            runs = []
            for start, end in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)):
                width = end-start
                if c.min_width*w <= width <= c.max_width*w:
                    x = (start+end-1)/2
                    label = int(labels[y, int(x)])
                    if stats[label, cv2.CC_STAT_AREA] >= 8:
                        runs.append((x/w, label))
            # Excess texture/stripes are not allowed to make processing unbounded.
            if len(runs) > 12:
                runs = []
            expanded = []
            for pts, cost, slope in beams:
                gap = t-pts[-1][0]
                if gap > c.max_gap:
                    finished.append((pts, cost, slope))
                    continue
                expanded.append((pts, cost+0.02, slope))
                for x, label in runs:
                    dx = x-pts[-1][1]
                    if abs(dx) > 0.018+1.1*gap:
                        continue
                    new_slope = dx/max(gap, 1e-6)
                    if len(pts) > 3 and abs(new_slope-slope) > 0.90:
                        continue
                    bend_cost = 0.05*abs(new_slope-slope)
                    expanded.append((pts+[(t, x, label)], cost+bend_cost-0.20, 0.5*slope+0.5*new_slope))
            if t < 0.18:
                for x, label in runs:
                    if abs(x-0.5) < 0.32:
                        expanded.append(([(t, x, label)], abs(x-0.5)*0.3, 0.0))
            # Preserve spatial/heading diversity; otherwise the straight branch
            # crowds the curved branch out of a simple top-N search.
            groups = {}
            for beam in sorted(expanded, key=lambda b: b[1]):
                pts, _, slope = beam
                key = (round(pts[-1][1]*w/7), round(slope/0.2), round(pts[-1][0]*h/stride))
                if key not in groups:
                    groups[key] = beam
                if len(groups) >= 28:
                    break
            beams = list(groups.values())
        finished.extend(beams)
        candidates = []
        for pts, cost, _ in sorted(finished, key=lambda b: b[1])[:100]:
            if len(pts) < 9:
                continue
            arr = np.asarray(pts)
            span = float(arr[-1, 0]-arr[0, 0])
            if span < c.min_span or arr[0, 0] > 0.18:
                continue
            coef = np.polyfit(arr[:, 0], arr[:, 1], 2)
            residual = np.abs(np.polyval(coef, arr[:, 0])-arr[:, 1])
            keep = residual < 0.035
            if np.mean(keep) < 0.8:
                continue
            coef = np.polyfit(arr[keep, 0], arr[keep, 1], 2)
            rms = float(np.sqrt(np.mean((np.polyval(coef, arr[keep, 0])-arr[keep, 1])**2)))
            coverage = min(1.0, len(pts)*stride/(h*span))
            confidence = 0.40*min(1, span/0.8)+0.35*min(1, coverage/0.5)+0.25*max(0, 1-rms/0.035)
            if confidence < c.min_confidence or rms > 0.03:
                continue
            if np.any(np.abs(np.polyval(coef, [0.1, 0.5, 0.85])-0.5) > 0.65):
                continue
            path = Path(coef, confidence, len(np.unique(arr[keep, 2])), span, arr[:, :2])
            if any(np.max(np.abs(path.x(np.array([0.1, 0.5, 0.85]))-p.x(np.array([0.1, 0.5, 0.85])))) < 0.045 for p in candidates):
                continue
            candidates.append(path)
            if len(candidates) >= 6:
                break
        return candidates


@dataclass
class Heading:
    total: float = 0.0
    stamp: float = -math.inf
    epoch: int = 0
    tilt: float = 0.0


class HeadingTracker:
    """Unwrap measured yaw, reject jumps, expose resets as a new epoch."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.value = Heading()
        self.previous = None

    def update(self, yaw, now, tilt=0.0):
        if not all(math.isfinite(x) for x in (yaw, now, tilt)):
            return False
        dt = now-self.value.stamp
        if dt <= 0:
            return False
        delta = 0.0 if self.previous is None else math.atan2(math.sin(yaw-self.previous), math.cos(yaw-self.previous))
        reset = self.previous is None or dt > self.cfg.sensor_timeout
        if not reset and abs(delta) > self.cfg.max_yaw_rate*dt+0.05:
            self.value.epoch += 1
            self.previous = None
            self.value.stamp = -math.inf
            return False
        if reset:
            self.value.epoch += 1
            self.value.total = 0.0
        else:
            self.value.total += delta
        self.previous = yaw
        self.value.stamp = now
        self.value.tilt = tilt
        return True


class State(str, Enum):
    SEARCH = "SEARCH"
    CONFIRM = "CONFIRM"
    ENTER = "ENTER"
    FOLLOW = "FOLLOW"
    EXIT_READY = "EXIT_READY"
    EXIT = "EXIT"
    COOLDOWN = "COOLDOWN"
    RECOVER = "RECOVER"


@dataclass
class Decision:
    owned: bool = False
    valid: bool = False
    path: Optional[Path] = None
    speed: float = 0.0
    predicted: bool = False
    reason: str = "baseline"


class Machine:
    ACTIVE = (State.ENTER, State.FOLLOW, State.EXIT_READY, State.EXIT, State.RECOVER)

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = State.SEARCH
        self.hits = self.exit_hits = self.clear = 0
        self.started = self.state_since = 0.0
        self.origin = 0.0
        self.epoch = -1
        self.degrees = 0.0
        self.last_path = None
        self.entry_path = None
        self.last_seen = -math.inf
        self.last_tick = None
        self.completed = self.aborted = 0
        self.reason = "search"
        self._recovery_error = None

    @property
    def active(self):
        return self.state in self.ACTIVE

    def change(self, state, now, reason):
        self.state, self.state_since, self.reason = state, now, reason
        self.hits = self.exit_hits = self.clear = 0

    def fault(self, now, reason):
        if self.state != State.RECOVER:
            self.aborted += 1
            self.change(State.RECOVER, now, reason)
            self.last_path = None
            self._recovery_error = None
        return Decision(True, reason=self.reason)

    def _choose(self, obs, now, exit_path=False):
        c = self.cfg
        candidates = obs.paths
        if obs.junction:
            # Explicit task choice overrides the visually easiest straight line.
            candidates = [obs.tangent if exit_path else obs.arc]
        elif exit_path:
            candidates = [p for p in candidates if p.straight]
        candidates = [p for p in candidates if p is not None and p.confidence >= c.min_confidence
                      and p.supports(c.lookahead)
                      and (exit_path or p.bend*c.image_turn_sign >= -0.20)]
        if self.last_path is not None and not exit_path:
            sample = np.array([0.12, 0.4, 0.7])
            candidates = [p for p in candidates if np.mean(np.abs(p.x(sample)-self.last_path.x(sample))) <= c.max_track_jump]
        def score(p):
            continuity = 0 if self.last_path is None else float(np.mean(np.abs(p.x(np.array([0.15, 0.5, 0.8]))-self.last_path.x(np.array([0.15, 0.5, 0.8])))))
            return p.confidence-2*continuity
        candidates.sort(key=score, reverse=True)
        if len(candidates) > 1 and not obs.junction and score(candidates[0])-score(candidates[1]) < 0.04:
            return None
        return candidates[0] if candidates else None

    def update(self, obs, heading, now, base_valid=False, base_error=0.0):
        c = self.cfg
        if self.last_tick is not None and (now <= self.last_tick or now-self.last_tick > c.sensor_timeout):
            if self.active:
                self.fault(now, "frame_gap")
            elif self.state == State.CONFIRM:
                self.change(State.SEARCH, now, "frame_gap")
            else:
                self.hits = self.clear = 0  # A camera gap cannot bypass cooldown.
        self.last_tick = now
        fresh = heading is not None and 0 <= now-heading.stamp <= c.sensor_timeout
        level = fresh and heading.tilt <= math.radians(c.max_tilt_degrees)
        if self.state == State.RECOVER:
            unique = len(obs.paths) == 1 and not obs.junction
            stable = self._recovery_error is None or abs(base_error-self._recovery_error) < 0.12
            self.clear = self.clear+1 if base_valid and unique and stable and abs(base_error) < 0.65 else 0
            self._recovery_error = base_error
            if self.clear >= c.recovery_frames:
                self.change(State.COOLDOWN, now, "abort_to_stable_line")
                return Decision(reason=self.reason)
            return Decision(True, reason=self.reason)
        if self.state == State.COOLDOWN:
            clear = base_valid and not obs.junction and len(obs.paths) <= 1
            self.clear = self.clear+1 if clear else 0
            if now-self.state_since >= c.cooldown_seconds and self.clear >= c.rearm_clear_frames:
                self.change(State.SEARCH, now, "rearmed_after_clear")
            return Decision(reason=self.reason)
        if self.state in (State.SEARCH, State.CONFIRM):
            evidence = obs.junction and obs.arc is not None and obs.tangent is not None
            if not evidence:
                self.change(State.SEARCH, now, "search")
                return Decision()
            if self.state == State.SEARCH:
                self.change(State.CONFIRM, now, "confirm_junction")
                self.entry_path = None
            if self.entry_path is not None:
                sample = np.array([0.1, 0.5, 0.8])
                if np.mean(np.abs(obs.arc.x(sample)-self.entry_path.x(sample))) > 0.10:
                    self.hits = 0
            self.entry_path = obs.arc
            self.hits += 1
            if not level:
                self.hits = 0
                self.reason = "wait_heading_or_level"
                return Decision(reason=self.reason)
            if self.hits < c.confirm_frames:
                return Decision(reason="confirm_junction")
            self.origin, self.epoch = heading.total, heading.epoch
            self.degrees = 0.0
            self.started = now
            self.last_path = obs.arc
            self.last_seen = now
            self.change(State.ENTER, now, "arc_locked")
        if not level or heading.epoch != self.epoch:
            return self.fault(now, "heading_stale_reset_or_tilt")
        self.degrees = math.degrees(heading.total-self.origin)  # CCW is positive ROS yaw.
        if self.degrees < -20:
            return self.fault(now, "wrong_turn_direction")
        if now-self.started > c.lap_timeout or self.degrees > c.exit_max_degrees:
            return self.fault(now, "lap_limit_no_exit")
        if self.state == State.ENTER:
            if self.degrees >= c.enter_degrees:
                self.change(State.FOLLOW, now, "entered_circle")
            elif now-self.state_since > c.entry_timeout:
                return self.fault(now, "entry_timeout")
        if self.state == State.FOLLOW and self.degrees >= c.exit_min_degrees:
            self.change(State.EXIT_READY, now, "lap_progress_ready")
        if self.state == State.EXIT_READY:
            near_heading = abs(self.degrees-360) <= c.exit_heading_tolerance
            exit_evidence = obs.junction and obs.tangent is not None and near_heading
            self.exit_hits = self.exit_hits+1 if exit_evidence else 0
            if self.exit_hits >= c.exit_confirm_frames:
                self.change(State.EXIT, now, "tangent_locked")
                self.last_path = None
        if self.state == State.EXIT:
            if now-self.state_since > c.exit_timeout:
                return self.fault(now, "exit_timeout")
            straight_only = len(obs.paths) == 1 and obs.paths[0].straight and not obs.junction
            self.clear = self.clear+1 if straight_only and base_valid else 0
            if self.clear >= c.exit_confirm_frames:
                self.completed += 1
                self.change(State.COOLDOWN, now, "lap_complete")
                return Decision(reason=self.reason)
        chosen = self._choose(obs, now, self.state == State.EXIT)
        speed = c.exit_speed if self.state == State.EXIT else (c.enter_speed if self.state == State.ENTER else c.follow_speed)
        if chosen is not None:
            self.last_path, self.last_seen = chosen, now
            return Decision(True, True, chosen, speed, reason=self.state.value.lower())
        age = now-self.last_seen
        if self.last_path is not None and age <= c.predict_seconds and age*c.predict_speed <= c.predict_distance:
            return Decision(True, True, self.last_path, c.predict_speed, True, "short_gap_prediction")
        return self.fault(now, "track_lost_or_ambiguous")


def quaternion_angles(q, layout="xyzw"):
    """The bundled chassis puts (w,x,y,z) into ROS fields (x,y,z,w)."""
    if layout == "wxyz_in_xyzw":
        w, x, y, z = q.x, q.y, q.z, q.w
    elif layout == "xyzw":
        x, y, z, w = q.x, q.y, q.z, q.w
    else:
        raise ValueError("unknown quaternion layout")
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(norm) or not 0.5 <= norm <= 1.5:
        raise ValueError("invalid quaternion")
    x, y, z, w = (v/norm for v in (x, y, z, w))
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    return yaw, max(abs(roll), abs(pitch))


class RoundaboutAdapter:
    def __init__(self, node):
        # ROS stays optional: offline detection/tests need only NumPy + OpenCV.
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String
        self._status_type = String
        self.node = node
        defaults = Config()
        values = {}
        for item in fields(defaults):
            name = "roundabout."+item.name
            default = getattr(defaults, item.name)
            if not node.has_parameter(name):
                node.declare_parameter(name, default)
            values[item.name] = node.get_parameter(name).value
        self.cfg = Config(**values)
        self.cfg.validate()
        extra = {"heading_source": "imu", "imu_topic": "/imu/data", "odom_topic": "/odom",
                 "imu_layout": "wxyz_in_xyzw", "yaw_sign": 1.0}
        for key, default in extra.items():
            name = "roundabout."+key
            if not node.has_parameter(name):
                node.declare_parameter(name, default)
            setattr(self, key, node.get_parameter(name).value)
        if self.heading_source not in ("imu", "odom") or self.imu_layout not in ("xyzw", "wxyz_in_xyzw") or self.yaw_sign not in (-1.0, 1.0):
            raise ValueError("invalid heading source/layout/sign")
        self.vision = Vision(self.cfg)
        self.tracker = HeadingTracker(self.cfg)
        self.machine = Machine(self.cfg)
        self.mode = "off"
        self.observation = Observation()
        self.decision = Decision()
        self.owns_control = False
        self.failed = ""
        self.last_status = 0.0
        self.last_sensor_stamp = -math.inf
        self.elapsed_ms = 0.0
        self.status_pub = node.create_publisher(String, "/roundabout/status", 1)
        msg_type = Imu if self.heading_source == "imu" else Odometry
        topic = self.imu_topic if self.heading_source == "imu" else self.odom_topic
        self.subscription = node.create_subscription(msg_type, topic, self._heading, qos_profile_sensor_data)

    def _heading(self, message):
        now = time.monotonic()
        try:
            stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
            ros_now = self.node.get_clock().now().nanoseconds*1e-9
            # Replayed/duplicate/future/zero stamps cannot manufacture progress.
            if stamp <= 0 or stamp <= self.last_sensor_stamp or not -0.10 <= ros_now-stamp <= self.cfg.sensor_timeout:
                return
            self.last_sensor_stamp = stamp
            if self.heading_source == "imu":
                if message.orientation_covariance[0] < 0:
                    return
                yaw, tilt = quaternion_angles(message.orientation, self.imu_layout)
            else:
                yaw, tilt = quaternion_angles(message.pose.pose.orientation)
            self.tracker.update(yaw*self.yaw_sign, now, tilt)
        except (AttributeError, ValueError, OverflowError):
            return  # Old valid reading ages out; invalid data never refreshes it.

    def reset_mode(self, mode):
        if mode != self.mode:
            was_owned = self.owns_control
            self.machine = Machine(self.cfg)
            self.mode = mode
            self.failed = ""
            self.owns_control = False
            if was_owned:
                self.node._have_previous_error = False
                self.node._lost_hold = 0
                self.node._last_command_valid = False

    def process(self, frame, baseline, now, mode, allowed=True):
        self.reset_mode(mode)
        if mode == "off":
            self.owns_control = False
            return baseline
        was_owned = self.owns_control
        decision = Decision()
        start = time.monotonic()
        try:
            if self.failed:
                raise RuntimeError(self.failed)
            self.observation = self.vision.detect(frame)
            self.elapsed_ms = (time.monotonic()-start)*1000
            if self.elapsed_ms > self.cfg.max_processing_seconds*1000:
                raise RuntimeError("processing_budget_exceeded")
            if not allowed:
                if self.machine.active:
                    self.machine.fault(now, "other_element_priority")
                elif self.machine.state == State.CONFIRM:
                    self.machine.change(State.SEARCH, now, "other_element_priority")
            else:
                decision = self.machine.update(self.observation, self.tracker.value, now, baseline.valid, baseline.error)
        except Exception as error:
            # Inactive failure is isolated from basic line following. An active
            # failure must not send the robot straight through the junction.
            self.failed = f"{type(error).__name__}: {error}"[:140]
            if self.machine.active or was_owned:
                decision = self.machine.fault(now, "module_fault_toggle_off_to_reset")
            else:
                decision = Decision(reason="module_disabled: "+self.failed)
        self.decision = decision
        self.owns_control = mode == "control" and allowed and decision.owned
        if was_owned != self.owns_control:
            self.node._have_previous_error = False
            self.node._lost_hold = 0
            self.node._last_command_valid = False
        result = baseline
        if self.owns_control:
            error = 0.0
            if decision.valid and decision.path is not None:
                # Project the target back to the original camera coordinates so
                # existing steering sign/normalisation and target offset apply.
                t = self.cfg.lookahead
                point = np.float32([[[decision.path.x(t)*self.cfg.work_width, (1-t)*(self.cfg.work_height-1)]]])
                if self.observation.inverse is None:
                    decision.valid = False
                else:
                    raw = cv2.perspectiveTransform(point, self.observation.inverse)[0, 0]
                    target = frame.shape[1]*0.5+self.node.target_offset_pixels
                    error = float(np.clip((raw[0]-target)/(frame.shape[1]*0.5), -1, 1))
                    if not math.isfinite(error):
                        decision.valid = False
            result = replace(baseline, valid=decision.valid, error=error,
                             confidence=decision.path.confidence if decision.path is not None else 0.0,
                             reason="ring_"+decision.reason, corner=False,
                             speed_limit=decision.speed if decision.valid else 0.0, inhibit_hold=True)
        # Debug/telemetry exceptions never influence actuation or baseline data.
        try:
            if self.node.show_debug:
                self.draw(result.debug_frame)
            self.publish(now)
        except Exception as error:
            if now-self.last_status > 2:
                self.node.get_logger().warning(f"圆环调试输出异常: {error}")
                self.last_status = now
        return result

    def draw(self, debug, now=None):
        c, obs, m = self.cfg, self.observation, self.machine
        if self.decision.path is not None and obs.inverse is not None:
            t = np.linspace(0.04, 0.94, 45)
            xy = np.float32(np.column_stack((self.decision.path.x(t)*c.work_width, (1-t)*(c.work_height-1))))
            raw = cv2.perspectiveTransform(xy[None, :, :], obs.inverse)[0].astype(np.int32)
            cv2.polylines(debug, [raw], False, (0, 255, 0), 3)
        # A right-hand panel avoids covering the baseline/step/zebra overlays.
        h, w = debug.shape[:2]
        panel = np.full((h, 340, 3), 24, np.uint8)
        age = (time.monotonic() if now is None else now)-self.tracker.value.stamp
        lines = [f"RING {self.mode.upper()}  {m.state.value}",
                 f"control={'RING' if self.owns_control else 'BASE LINE'}",
                 f"paths={len(obs.paths)} junction={int(obs.junction)} hits={m.hits}/{c.confirm_frames}",
                 f"yaw={m.degrees:.1f}  exit>={c.exit_min_degrees:.0f}",
                 f"sensor={self.heading_source} age={age:.2f}s",
                 f"complete={m.completed} abort={m.aborted}",
                 f"exit_hits={m.exit_hits}/{c.exit_confirm_frames} clear={m.clear}",
                 f"{self.elapsed_ms:.1f} ms  predict={int(self.decision.predicted)}",
                 self.failed or self.decision.reason or m.reason,
                 "entry="+obs.entry.get("reject", obs.reason),
                 f"A={obs.entry.get('arcs', 0)} T={obs.entry.get('tangents', 0)} "
                 f"near_pairs={obs.entry.get('near_pairs', 0)}"]
        for i, line in enumerate(lines):
            cv2.putText(panel, line[:48], (8, 20+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (0, 210, 255) if i == 0 else (220, 220, 220), 1, cv2.LINE_AA)
        if obs.debug is not None and h > 270:
            ph = min(h-265, 200)
            panel[260:260+ph, 10:330] = cv2.resize(obs.debug, (320, ph))
        # Assigned by process below; no resize of source pixels before control.
        self.panel = panel

    def publish(self, now):
        if now-self.last_status < 0.2:
            return
        self.last_status = now
        m = self.machine
        msg = self._status_type()
        msg.data = json.dumps({"mode": self.mode, "state": m.state.value,
                               "owns_control": self.owns_control, "reason": self.failed or self.decision.reason,
                               "yaw_degrees": round(m.degrees, 1), "paths": len(self.observation.paths),
                               "junction": self.observation.junction, "completed": m.completed,
                               "entry": self.observation.entry,
                               "aborted": m.aborted, "predicted": self.decision.predicted,
                               "processing_ms": round(self.elapsed_ms, 1)})
        self.status_pub.publish(msg)


def main():
    """Offline/live-camera visual inspection only. Never imports ROS or sends motion."""
    import argparse
    import csv
    from pathlib import Path as FilePath

    parser = argparse.ArgumentParser(description="Roundabout visual replay (no robot commands)")
    parser.add_argument("--source", default="0", help="camera index, image or video")
    parser.add_argument("--config", help="icn.yaml; requires PyYAML")
    parser.add_argument("--yaw-csv", help="measured relative time,yaw_degrees CSV with header")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--snapshot", help="save last annotated frame")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    values = {}
    if args.config:
        import yaml
        data = yaml.safe_load(FilePath(args.config).read_text(encoding="utf-8"))
        params = data["line_follower"]["ros__parameters"]
        values = {f.name: params["roundabout."+f.name] for f in fields(Config) if "roundabout."+f.name in params}
    cfg = Config(**values)
    vision, tracker, machine = Vision(cfg), HeadingTracker(cfg), Machine(cfg)
    headings = []
    if args.yaw_csv:
        with open(args.yaw_csv, encoding="utf-8-sig", newline="") as stream:
            headings = [(float(r["time"]), math.radians(float(r["yaw_degrees"]))) for r in csv.DictReader(stream)]
        if any(not math.isfinite(t+y) for t, y in headings) or any(b[0] <= a[0] for a, b in zip(headings, headings[1:])):
            raise ValueError("yaw CSV must contain finite, strictly increasing relative times")
    still = None
    if FilePath(args.source).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        still = cv2.imread(args.source)
        if still is None:
            raise ValueError("cannot read source image")
    capture = None if still is not None else cv2.VideoCapture(int(args.source) if args.source.isdecimal() else args.source)
    if capture is not None and not capture.isOpened():
        raise ValueError("cannot open camera/video")
    fps = capture.get(cv2.CAP_PROP_FPS) if capture is not None else 30.0
    fps = fps if math.isfinite(fps) and fps > 0 else 30.0
    viewer = RoundaboutAdapter.__new__(RoundaboutAdapter)
    viewer.cfg, viewer.machine, viewer.tracker = cfg, machine, tracker
    viewer.mode, viewer.owns_control, viewer.failed = "observe", False, ""
    viewer.heading_source = "csv" if headings else "none"
    index = heading_index = 0
    last = None
    try:
        while True:
            ok, frame = (True, still.copy()) if still is not None else capture.read()
            if not ok:
                break
            now = index/fps
            while heading_index < len(headings) and headings[heading_index][0] <= now:
                t, yaw = headings[heading_index]
                tracker.update(yaw, t)
                heading_index += 1
            start = time.perf_counter()
            obs = vision.detect(frame)
            viewer.elapsed_ms = (time.perf_counter()-start)*1000
            # Replay uses geometric single-line evidence; robot integration uses
            # the real baseline detector for recovery/exit confirmation.
            decision = machine.update(obs, tracker.value, now, len(obs.paths) == 1)
            viewer.observation, viewer.decision = obs, decision
            viewer.draw(frame, now)
            last = np.concatenate((frame, viewer.panel), axis=1)
            if index % max(1, round(fps/2)) == 0:
                print(json.dumps({"frame": index, "paths": len(obs.paths), "junction": obs.junction,
                                  "state": machine.state.value, "reason": decision.reason,
                                  "yaw": round(machine.degrees, 1), "ms": round(viewer.elapsed_ms, 1)}))
            if not args.no_display:
                cv2.imshow("Roundabout OBSERVE ONLY - Q/Esc exits", last)
                if cv2.waitKey(0 if still is not None else max(1, round(1000/fps))) & 0xFF in (27, ord("q")):
                    break
            index += 1
            if still is not None or (args.max_frames > 0 and index >= args.max_frames):
                break
        if args.snapshot and last is not None and not cv2.imwrite(args.snapshot, last):
            raise OSError("cannot write snapshot")
    finally:
        if capture is not None:
            capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
