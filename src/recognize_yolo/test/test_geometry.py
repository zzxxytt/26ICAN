"""几何与 NMS 单元测试。"""

import numpy as np

from recognize_yolo.geometry import (
    batched_nms,
    box_iou,
    clip_boxes,
    nms,
    scale_boxes,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_xywh_xyxy_roundtrip():
    boxes = np.array([[10, 20, 4, 8], [0, 0, 2, 2]], dtype=np.float32)
    restored = xyxy_to_xywh(xywh_to_xyxy(boxes))
    assert np.allclose(boxes, restored)


def test_box_iou_basic():
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float32)

    ious = box_iou(box, boxes)

    assert abs(ious[0] - 1.0) < 1e-6
    # 交集 25，并集 175
    assert abs(ious[1] - 25.0 / 175.0) < 1e-6
    assert abs(ious[2]) < 1e-6


def test_box_iou_survives_degenerate_box():
    """零面积框不能让 IoU 变成 NaN。"""
    box = np.array([5, 5, 5, 5], dtype=np.float32)
    boxes = np.array([[5, 5, 5, 5]], dtype=np.float32)

    ious = box_iou(box, boxes)

    assert not np.isnan(ious).any()


def test_nms_keeps_highest_score():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    keep = nms(boxes, scores, iou_threshold=0.5)

    assert keep == [0, 2]


def test_nms_respects_max_detections():
    boxes = np.array([[index * 100, 0, index * 100 + 10, 10] for index in range(10)], dtype=np.float32)
    scores = np.linspace(0.9, 0.1, 10).astype(np.float32)

    keep = nms(boxes, scores, iou_threshold=0.5, max_detections=3)

    assert len(keep) == 3


def test_batched_nms_does_not_suppress_across_classes():
    """重叠但类别不同，两个框都要留下。"""
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int32)

    keep = batched_nms(boxes, scores, class_ids, iou_threshold=0.5)

    assert len(keep) == 2


def test_batched_nms_suppresses_same_class():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 0], dtype=np.int32)

    keep = batched_nms(boxes, scores, class_ids, iou_threshold=0.5)

    assert keep == [0]


def test_clip_boxes_clamps_to_image():
    boxes = np.array([[-5, -5, 200, 200]], dtype=np.float32)
    clipped = clip_boxes(boxes, 100, 50)
    assert clipped[0].tolist() == [0.0, 0.0, 100.0, 50.0]


def test_scale_boxes_undoes_letterbox():
    """ratio 2、左上补 10 像素：输入坐标 (20,30) 应还原成原图 (5,10)。"""
    boxes = np.array([[20, 30, 40, 50]], dtype=np.float32)
    restored = scale_boxes(boxes, ratio=2.0, pad_x=10.0, pad_y=10.0,
                           original_width=1000, original_height=1000)
    assert restored[0].tolist() == [5.0, 10.0, 15.0, 20.0]
