"""可视化：把检测框画到图上，并输出一帧的可读摘要。

注意 OpenCV 的 putText 不支持中文，画面上的文字一律用 ASCII，
中文信息走 ROS 日志与 JSON 话题。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .decode import Detection

# 固定调色板，保证同一类别在不同帧颜色一致，方便肉眼追踪。
_PALETTE: Sequence[Tuple[int, int, int]] = (
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (147, 69, 52),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
)


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    """按类别取一个稳定的 BGR 颜色。"""
    return _PALETTE[int(class_id) % len(_PALETTE)]


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    line_width: int = 2,
    show_score: bool = True,
) -> np.ndarray:
    """在图像副本上画框，返回新图（不修改入参）。"""
    canvas = image.copy()
    height, width = canvas.shape[:2]

    for detection in detections:
        color = color_for_class(detection.class_id)
        x1 = int(round(detection.xmin))
        y1 = int(round(detection.ymin))
        x2 = int(round(detection.xmax))
        y2 = int(round(detection.ymax))

        # 裁剪到画布内，避免 cv2.rectangle 在越界时报错或画出画面。
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)

        label = detection.class_name or f"id{detection.class_id}"
        if show_score:
            label = f"{label} {detection.score:.2f}"

        font_scale = max(0.4, min(0.7, width / 1280.0))
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        # 标签放在框上方，顶到画面边缘时改放框内。
        label_top = y1 - text_h - baseline - 2
        if label_top < 0:
            label_top = y1 + 2
        cv2.rectangle(
            canvas,
            (x1, label_top),
            (min(x1 + text_w + 4, width - 1), label_top + text_h + baseline + 2),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, label_top + text_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 中心点：跟随控制节点最关心的量，直接画出来便于现场对中。
        center = (int(round(detection.center_x)), int(round(detection.center_y)))
        if 0 <= center[0] < width and 0 <= center[1] < height:
            cv2.circle(canvas, center, 3, color, -1)

    return canvas


def draw_status_bar(
    image: np.ndarray,
    backend: str,
    inference_ms: float,
    detection_count: int,
    fps: float = 0.0,
) -> np.ndarray:
    """在左上角叠加一行状态信息。"""
    canvas = image.copy()
    text = (
        f"{backend} | {inference_ms:.1f}ms | "
        f"{detection_count} obj | {fps:.1f} fps"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (32, 32, 32), -1)
    cv2.putText(
        canvas, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 255, 255), 1, cv2.LINE_AA,
    )
    return canvas


def summarize_detections(detections: Sequence[Detection], limit: int = 5) -> str:
    """生成一行文本摘要，用于日志。"""
    if not detections:
        return "无目标"
    parts: List[str] = []
    for detection in detections[:limit]:
        parts.append(
            f"{detection.class_name}({detection.score:.2f})"
            f"@({detection.center_x:.0f},{detection.center_y:.0f})"
        )
    if len(detections) > limit:
        parts.append(f"...共 {len(detections)} 个")
    return " ".join(parts)
