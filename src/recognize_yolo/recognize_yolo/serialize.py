"""把检测结果序列化成 JSON。

给两类下游用：

* 还没准备好写 ROS 节点的脚本 / 上位机，直接订阅
  ``~/detections_json`` 解析字符串即可。
* 记录数据集与事后复盘，省得再从 rosbag 里反解自定义消息。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

from .decode import Detection

#: JSON 里报的 schema 版本，字段有增删时递增，下游好做兼容。
SCHEMA_VERSION = 1


def detection_to_dict(detection: Detection) -> Dict[str, Any]:
    """单个检测结果 -> dict。"""
    return {
        "class_id": int(detection.class_id),
        "class_name": detection.class_name or f"class_{detection.class_id}",
        "score": round(float(detection.score), 4),
        "bbox": [
            round(float(detection.xmin), 2),
            round(float(detection.ymin), 2),
            round(float(detection.xmax), 2),
            round(float(detection.ymax), 2),
        ],
        "center": [round(float(detection.center_x), 2), round(float(detection.center_y), 2)],
        "track_id": int(detection.track_id),
    }


def detection_array_to_dict(
    detections: Sequence[Detection],
    image_width: int,
    image_height: int,
    stamp_sec: float,
    frame_id: str = "",
    inference_ms: float = 0.0,
    backend: str = "",
    source: str = "",
    input_size: int = 0,
) -> Dict[str, Any]:
    """一帧的完整结果 -> dict，可直接 json.dumps。"""
    payload: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "stamp": round(float(stamp_sec), 4),
        "frame_id": frame_id,
        "source": source,
        "backend": backend,
        "input_size": int(input_size),
        "image": {"width": int(image_width), "height": int(image_height)},
        "inference_ms": round(float(inference_ms), 2),
        "count": len(detections),
        "detections": [detection_to_dict(item) for item in detections],
    }
    return payload


def detection_array_to_json(
    detections: Sequence[Detection],
    image_width: int,
    image_height: int,
    stamp_sec: float,
    frame_id: str = "",
    inference_ms: float = 0.0,
    backend: str = "",
    source: str = "",
    input_size: int = 0,
    ensure_ascii: bool = False,
) -> str:
    """一帧的完整结果 -> JSON 字符串。"""
    payload = detection_array_to_dict(
        detections,
        image_width=image_width,
        image_height=image_height,
        stamp_sec=stamp_sec,
        frame_id=frame_id,
        inference_ms=inference_ms,
        backend=backend,
        source=source,
        input_size=input_size,
    )
    return json.dumps(payload, ensure_ascii=ensure_ascii, separators=(",", ":"))


def find_best_detection(
    detections: Sequence[Detection],
    class_name: Optional[str] = None,
    class_id: Optional[int] = None,
) -> Optional[Detection]:
    """挑出指定类别里分数最高的一个。

    跟随类控制节点常用：只关心画面里最可信的那个球/人/瓶子。
    class_name 与 class_id 都不给时返回全局最高分的目标。
    """
    candidates = list(detections)
    if class_name is not None:
        candidates = [item for item in candidates if item.class_name == class_name]
    if class_id is not None:
        candidates = [item for item in candidates if item.class_id == class_id]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.score)
