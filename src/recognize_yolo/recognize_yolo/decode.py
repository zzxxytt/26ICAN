"""YOLO 原始输出张量 -> 检测框。

这里刻意做成「版本无关 + 布局自适应」，原因是香橙派 4 Pro 的 NPU 只能拿到
去掉后处理头的原始输出，而且不同导出脚本出来的张量布局并不一致：

* YOLOv5 官方 export.py 带 --grid 时，输出是 (1, 3, 85, 80, 80) 这类
  通道数 = 5 + 类别数 的格式，而 AW NPU Model Zoo 打印出来是
  ``ouput 0 dim 80 80 85 3``，说明送进 NPU 的张量被转置过。
* YOLOv8 输出是 (1, 4 + 类别数, 8400)，只有一组大张量。

因此本模块先根据每个张量的形状判断它是「通道优先」还是「锚点优先」，
再统一整理成 (N, 4 + 1 + num_classes) 的标准行向量，最后做解码 + NMS。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import batched_nms, scale_boxes, xywh_to_xyxy


@dataclass
class Detection:
    """输入图坐标系下的单个检测结果。"""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    score: float
    class_id: int
    class_name: str = ""
    track_id: int = -1

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass
class DecodeConfig:
    """解码参数，全部来自 ROS 参数，方便现场调。"""

    num_classes: int = 80
    input_size: int = 640
    # 置信度阈值
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    min_box_area: float = 0.0
    # True 表示网络输出里已经乘过 sigmoid / 已经是概率，不需要再激活。
    apply_sigmoid: bool = False
    #: {网格边长: stride}。留空时按 input_size / 网格边长 自动推算。
    strides: Dict[int, int] = field(default_factory=dict)
    class_names: Sequence[str] = field(default_factory=list)

    def name_of(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        return f"class_{class_id}"


def _valid_channel_counts(num_classes: int) -> Tuple[int, ...]:
    """模型输出里「通道维」可能的长度。

    YOLOv5 带 objectness 是 4 + 1 + 类别数，YOLOv8 是 4 + 类别数。
    这两个值就是判断某个轴是不是通道维的唯一可靠依据 ——
    不要用「哪个轴最小」之类的启发式，那在 H 或 W 恰好较小时会猜错。
    """
    return (5 + num_classes, 4 + num_classes)


def _find_channel_axis(array: np.ndarray, num_classes: int) -> Optional[int]:
    """找出哪个轴是通道维，找不到返回 None。"""
    expected = _valid_channel_counts(num_classes)
    matches = [axis for axis, size in enumerate(array.shape) if size in expected]
    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    # 多个轴都命中时（极小的测试张量会出现），优先 5+类别数，其次取更大的轴。
    preferred = 5 + num_classes
    for axis in matches:
        if array.shape[axis] == preferred:
            return axis
    return max(matches, key=lambda axis: array.shape[axis])


def _as_2d_arrays(
    tensors: Sequence[np.ndarray], num_classes: int
) -> List[Tuple[np.ndarray, Optional[int]]]:
    """把每个输出头整理成 (N, C)，并附带该头的网格边长（没有网格时为 None）。

    C 必然是 4+类别数 或 5+类别数，据此定位通道维，其余维乘起来就是 N。
    YOLOv5 的 (1, 3, 85, 80, 80)、NPU 打印出来的 (80, 80, 85, 3)、
    YOLOv8 的 (1, 84, 8400) 都能正确摊平。

    网格边长只在「剩下的空间维恰好构成一个正方形网格」时给出，
    用来判断是否需要做「网格偏移 -> 绝对坐标」的换算。
    """
    results: List[Tuple[np.ndarray, Optional[int]]] = []
    expected = _valid_channel_counts(num_classes)

    for tensor in tensors:
        array = np.asarray(tensor, dtype=np.float32)

        # 去掉 batch 维：NPU 读回来常见 (3, 80, 80, 85)，ONNX 是 (1, 3, 85, 80, 80)。
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]

        if array.ndim == 2:
            if array.shape[1] not in expected and array.shape[0] in expected:
                array = array.transpose(1, 0)
            # (N, C) 这种形状已经没有网格信息了（YOLOv8 就是这个形式）
            results.append((array, None))
            continue

        if array.ndim < 2:
            raise ValueError(f"输出张量维度过低: {array.shape}")

        channel_axis = _find_channel_axis(array, num_classes)
        if channel_axis is None:
            raise ValueError(
                f"无法在输出张量 {array.shape} 中找到通道维，"
                f"期望某一维等于 {expected[0]}（YOLOv5 带 objectness）"
                f" 或 {expected[1]}（YOLOv8 无 objectness）。"
                f"请核对 num_classes 参数是否与模型一致"
            )

        channels = array.shape[channel_axis]
        moved = np.moveaxis(array, channel_axis, -1)
        spatial = moved.shape[:-1]

        # 空间维之外可能还有一个 anchor 维（YOLOv5 是 3 个 anchor）。
        # 它要被折进「点」的数量里，不能当成网格的一边，否则网格判定会失败、
        # 网格偏移就不会被还原。
        #
        # 注意顺序：reshape 后行索引按「最后一个空间维最快变化」展开，
        # 也就是 (anchor, row, col, C) -> 行号 = anchor*H*W + row*W + col。
        # _add_grid_offsets 就是按这个顺序还原 grid_x / grid_y 的。
        grid_size: Optional[int] = None

        if len(spatial) == 3 and spatial[1] == spatial[2]:
            grid_size = int(spatial[1])
        elif len(spatial) == 2 and spatial[0] == spatial[1]:
            grid_size = int(spatial[0])

        results.append((moved.reshape(-1, channels), grid_size))

    if not results:
        raise ValueError("没有可用的模型输出张量")

    channel_counts = {row.shape[1] for row, _ in results}
    if len(channel_counts) != 1:
        raise ValueError(
            "各输出头的通道数不一致 "
            f"{sorted(channel_counts)}，请确认 num_classes 与模型是否匹配"
        )

    return results


def _add_grid_offsets(
    boxes_xywh: np.ndarray, grid_size: int, stride: float
) -> np.ndarray:
    """把「相对网格单元的偏移」换算成输入图上的绝对像素坐标。

    YOLOv5 训练与导出时，网络回归的是 (tx, ty, tw, th)，推理端要还原成：

        x_center = (grid_x + tx) * stride
        y_center = (grid_y + ty) * stride
        w        = tw * stride
        h        = th * stride

    漏掉这一步、把 tx 直接当像素坐标用，框会整体缩到画面左上角的一小块里
    —— 这是「模型明明能跑但框全错」最典型的原因之一。
    """
    total = boxes_xywh.shape[0]
    if total == 0:
        return boxes_xywh
    if total % (grid_size * grid_size) != 0:
        raise ValueError(
            f"框数量 {total} 不是网格 {grid_size}x{grid_size} 的整数倍，"
            f"无法还原网格坐标"
        )

    anchors = total // (grid_size * grid_size)
    # 行顺序是 anchor 外层、网格内层，且网格内 **列（x）变化最快**：
    #   行号 = anchor * H * W + row * W + col
    # 所以列索引要 tile，行索引要 repeat。写反了框会沿对角线镜像。
    column_pattern = np.tile(np.arange(grid_size, dtype=np.float32), grid_size)
    row_pattern = np.repeat(np.arange(grid_size, dtype=np.float32), grid_size)
    grid_x = np.tile(column_pattern, anchors)
    grid_y = np.tile(row_pattern, anchors)

    out = boxes_xywh.copy()
    out[:, 0] = (grid_x + out[:, 0]) * stride
    out[:, 1] = (grid_y + out[:, 1]) * stride
    out[:, 2] = out[:, 2] * stride
    out[:, 3] = out[:, 3] * stride
    return out


def _resolve_stride(
    config: "DecodeConfig", grid_size: int
) -> Optional[float]:
    """确定某个尺度头的 stride。

    优先用参数里显式配置的值，其次按 input_size / grid_size 推算
    （640/80=8、640/40=16、640/20=32，正好是 YOLOv5 的三个尺度）。
    """
    explicit = config.strides.get(grid_size) if config.strides else None
    if explicit:
        return float(explicit)

    if config.input_size and config.input_size % grid_size == 0:
        return float(config.input_size // grid_size)

    return None


def _maybe_sigmoid(values: np.ndarray) -> np.ndarray:
    """数值明显超出 [0,1] 时说明是 logits，需要补一次 sigmoid。"""
    if values.size == 0:
        return values
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        return 1.0 / (1.0 + np.exp(-values))
    return values


def decode_yolo_outputs(
    tensors: Sequence[np.ndarray],
    config: DecodeConfig,
    letterbox_info=None,
) -> List[Detection]:
    """把模型原始输出解码为检测结果列表。

    参数
    ----
    tensors:
        每个输出头的 numpy 数组，batch 维可选。
    config:
        解码参数。
    letterbox_info:
        preprocess.letterbox 返回的几何信息。给了就顺便把框映射回原图。

    返回
    ----
    Detection 列表，按 score 从高到低。
    """
    if config.num_classes <= 0:
        raise ValueError("num_classes 必须大于 0")

    heads = _as_2d_arrays(tensors, config.num_classes)

    # 先在「每个头自己的坐标系」里解码，最后统一做网格还原。
    # 这样做是因为不同尺度的头 stride 不同，必须分开处理。
    decoded_boxes: List[np.ndarray] = []
    decoded_scores: List[np.ndarray] = []
    decoded_classes: List[np.ndarray] = []

    for rows, grid_size in heads:
        expected_with_objectness = 4 + 1 + config.num_classes
        expected_without_objectness = 4 + config.num_classes

        if rows.shape[1] == expected_with_objectness:
            has_objectness = True
        elif rows.shape[1] == expected_without_objectness:
            has_objectness = False
        else:
            raise ValueError(
                f"输出通道数 {rows.shape[1]} 与 num_classes={config.num_classes} 不匹配，"
                f"期望 {expected_with_objectness}（YOLOv5 带 objectness）"
                f" 或 {expected_without_objectness}（YOLOv8 无 objectness）"
            )

        boxes_xywh = rows[:, :4]
        if has_objectness:
            objectness = rows[:, 4]
            class_scores = rows[:, 5:]
        else:
            objectness = None
            class_scores = rows[:, 4:]

        if config.apply_sigmoid:
            class_scores = _maybe_sigmoid(class_scores)
            if objectness is not None:
                objectness = _maybe_sigmoid(objectness)

        # 还原成输入图上的绝对坐标。网格信息缺失（YOLOv8 的单张大表）时
        # 认为网络输出的已经是绝对坐标，不做换算。
        if grid_size is not None:
            stride = _resolve_stride(config, grid_size)
            if stride:
                boxes_xywh = _add_grid_offsets(boxes_xywh, grid_size, stride)

        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        if objectness is not None:
            confidences = confidences * objectness

        keep_mask = confidences >= float(config.conf_threshold)
        if not np.any(keep_mask):
            continue

        decoded_boxes.append(xywh_to_xyxy(boxes_xywh[keep_mask]))
        decoded_scores.append(confidences[keep_mask].astype(np.float32))
        decoded_classes.append(class_ids[keep_mask])

    if not decoded_boxes:
        return []

    boxes_xyxy = np.concatenate(decoded_boxes, axis=0)
    confidences = np.concatenate(decoded_scores, axis=0)
    class_ids = np.concatenate(decoded_classes, axis=0)

    keep_indices = batched_nms(
        boxes_xyxy,
        confidences,
        class_ids,
        iou_threshold=config.iou_threshold,
        max_detections=config.max_detections,
    )
    if not keep_indices:
        return []

    boxes_xyxy = boxes_xyxy[keep_indices]
    confidences = confidences[keep_indices]
    class_ids = class_ids[keep_indices]

    if letterbox_info is not None:
        boxes_xyxy = scale_boxes(
            boxes_xyxy,
            letterbox_info.ratio,
            letterbox_info.pad_x,
            letterbox_info.pad_y,
            letterbox_info.original_width,
            letterbox_info.original_height,
        )

    detections: List[Detection] = []
    for box, score, class_id in zip(boxes_xyxy, confidences, class_ids):
        if config.min_box_area > 0.0:
            area = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
            if area < config.min_box_area:
                continue
        detections.append(
            Detection(
                xmin=float(box[0]),
                ymin=float(box[1]),
                xmax=float(box[2]),
                ymax=float(box[3]),
                score=float(score),
                class_id=int(class_id),
                class_name=config.name_of(int(class_id)),
            )
        )

    detections.sort(key=lambda item: item.score, reverse=True)
    return detections


def load_class_names(path: Optional[str]) -> List[str]:
    """读取类别名文件，每行一个名字，# 开头为注释。"""
    if not path:
        return []
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            names.append(stripped)
    return names


def describe_tensor_shape(tensor: np.ndarray) -> Tuple[int, ...]:
    """给日志用的张量形状描述。"""
    return tuple(int(dim) for dim in np.asarray(tensor).shape)
