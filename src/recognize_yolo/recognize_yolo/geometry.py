"""通用几何与后处理工具，只依赖 numpy。"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """把 (x_center, y_center, w, h) 转成 (xmin, ymin, xmax, ymax)。

    输入输出最后一维都是 4，支持任意前置维度。
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.shape[-1] != 4:
        raise ValueError(f"最后一维必须是 4，收到 {boxes.shape}")

    out = np.empty_like(boxes)
    half_w = boxes[..., 2] / 2.0
    half_h = boxes[..., 3] / 2.0
    out[..., 0] = boxes[..., 0] - half_w
    out[..., 1] = boxes[..., 1] - half_h
    out[..., 2] = boxes[..., 0] + half_w
    out[..., 3] = boxes[..., 1] + half_h
    return out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """把 (xmin, ymin, xmax, ymax) 转成 (x_center, y_center, w, h)。"""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.shape[-1] != 4:
        raise ValueError(f"最后一维必须是 4，收到 {boxes.shape}")

    out = np.empty_like(boxes)
    out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2.0
    out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2.0
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """单个框与一组框的 IoU，返回形状 (N,) 的数组。"""
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    inter_xmin = np.maximum(box[0], boxes[:, 0])
    inter_ymin = np.maximum(box[1], boxes[:, 1])
    inter_xmax = np.minimum(box[2], boxes[:, 2])
    inter_ymax = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(inter_xmax - inter_xmin, 0.0, None)
    inter_h = np.clip(inter_ymax - inter_ymin, 0.0, None)
    inter = inter_w * inter_h

    area_box = max((box[2] - box[0]), 0.0) * max((box[3] - box[1]), 0.0)
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0.0, None
    )
    union = area_box + areas - inter
    # 除以 0 保护：完全退化（面积为 0）的框直接返回 0 而不是 NaN。
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> List[int]:
    """标准非极大值抑制，返回保留下来的下标（按分数从高到低）。"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError("boxes 与 scores 数量不一致")
    if boxes.shape[0] == 0:
        return []

    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0 and len(keep) < max_detections:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        ious = box_iou(boxes[best], boxes[order[1:]])
        remaining = np.where(ious <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> List[int]:
    """按类别分别做 NMS 再合并。

    不同类别的框即使重叠也不互相抑制，避免「人」压掉「背包」这类错误。
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    if not (boxes.shape[0] == scores.shape[0] == class_ids.shape[0]):
        raise ValueError("boxes / scores / class_ids 数量必须一致")

    keep: List[int] = []
    for class_id in np.unique(class_ids):
        mask = np.where(class_ids == class_id)[0]
        local_keep = nms(boxes[mask], scores[mask], iou_threshold, max_detections)
        keep.extend(mask[local_keep].tolist())

    keep.sort(key=lambda index: float(scores[index]), reverse=True)
    return keep[:max_detections]


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """把框裁到图像范围内，避免发布越界坐标。"""
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    boxes[..., 0] = np.clip(boxes[..., 0], 0.0, float(width))
    boxes[..., 1] = np.clip(boxes[..., 1], 0.0, float(height))
    boxes[..., 2] = np.clip(boxes[..., 2], 0.0, float(width))
    boxes[..., 3] = np.clip(boxes[..., 3], 0.0, float(height))
    return boxes


def scale_boxes(
    boxes: np.ndarray,
    ratio: float,
    pad_x: float,
    pad_y: float,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    """把 letterbox 输入坐标系下的框还原回原图坐标系。

    ratio / pad_x / pad_y 由 preprocess.letterbox 返回。
    """
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    boxes[..., [0, 2]] = (boxes[..., [0, 2]] - pad_x) / ratio
    boxes[..., [1, 3]] = (boxes[..., [1, 3]] - pad_y) / ratio
    return clip_boxes(boxes, original_width, original_height)


def box_area(boxes: np.ndarray) -> np.ndarray:
    """框面积，形状 (N,)。"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0.0, None
    )


def filter_by_min_area(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, min_area: float
) -> Sequence[int]:
    """丢弃面积过小的框，用于抑制远处的噪声误检。"""
    if min_area <= 0.0:
        return list(range(int(np.asarray(scores).reshape(-1).shape[0])))
    areas = box_area(boxes)
    return np.where(areas >= float(min_area))[0].tolist()
