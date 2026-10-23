"""解码链路单元测试：不依赖 ROS、不依赖 NPU，电脑上直接跑就能验证。

关于测试数据的写法（很重要，看懂这段能省很多调试时间）：

YOLOv5 训练和导出时，网络回归的是**相对网格单元的偏移**，不是像素坐标。
推理端必须还原：

    x_center = (grid_x + tx) * stride
    y_center = (grid_y + ty) * stride
    w        = tw * stride
    h        = th * stride

所以测试里给的 tx/ty/tw/th 都是「网格单位」：0.5 表示半个格子。
本文件所有期望的像素坐标都按这个公式手算得出，写在断言旁边。

注意这里从 normalize 而不是 preprocess 导入归一化与 LetterboxInfo：
preprocess 依赖 OpenCV，而解码与归一化本身只需要 numpy。
"""

import numpy as np

from recognize_yolo.decode import DecodeConfig, decode_yolo_outputs
from recognize_yolo.normalize import LetterboxInfo, to_nchw_float


def _make_config(num_classes=3, **kwargs):
    base = dict(
        num_classes=num_classes,
        input_size=64,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_names=["alpha", "beta", "gamma"][:num_classes],
    )
    base.update(kwargs)
    return DecodeConfig(**base)


def _row(cx, cy, w, h, obj, class_scores):
    return [cx, cy, w, h, obj] + list(class_scores)


def test_decodes_channel_last_head():
    """(H, W, C) 布局，YOLOv5 常规导出。

    输入 64、网格 2x2 -> stride 32。
    网格单元 (row=0, col=0)，偏移 (0.5, 0.5)，边长 (0.5, 0.5)：
        x_center = (0 + 0.5) * 32 = 16
        y_center = (0 + 0.5) * 32 = 16
        w = h     = 0.5 * 32      = 16
    -> 框 (8, 8) - (24, 24)
    """
    config = _make_config()
    rows = np.zeros((1, 2, 2, 5 + 3), dtype=np.float32)
    rows[0, 0, 0] = _row(0.5, 0.5, 0.5, 0.5, 0.9, [0.9, 0.0, 0.0])

    result = decode_yolo_outputs([rows], config)

    assert len(result) == 1
    assert result[0].class_name == "alpha"
    assert abs(result[0].xmin - 8.0) < 1e-4
    assert abs(result[0].ymin - 8.0) < 1e-4
    assert abs(result[0].xmax - 24.0) < 1e-4
    assert abs(result[0].ymax - 24.0) < 1e-4


def test_decodes_channel_first_head():
    """(C, H, W) 布局，NPU 常见。同样的数据必须解出同样的框。"""
    config = _make_config()
    channel_last = np.zeros((1, 2, 2, 5 + 3), dtype=np.float32)
    channel_last[0, 0, 0] = _row(0.5, 0.5, 0.5, 0.5, 0.9, [0.9, 0.0, 0.0])
    channel_last[0, 1, 1] = _row(0.25, 0.25, 0.5, 0.5, 0.8, [0.0, 0.8, 0.0])
    channel_first = channel_last.transpose(0, 3, 1, 2)

    from_last = decode_yolo_outputs([channel_last], config)
    from_first = decode_yolo_outputs([channel_first], config)

    assert len(from_last) == len(from_first) == 2
    for left, right in zip(from_last, from_first):
        assert left.class_id == right.class_id
        assert abs(left.xmin - right.xmin) < 1e-4
        assert abs(left.ymax - right.ymax) < 1e-4


def test_decodes_anchor_first_head():
    """(A, C, H, W) 布局，YOLOv5 带 --grid 导出的 (1, 3, 85, 80, 80)。

    输入 64、网格 4x4 -> stride 16。
    网格 (0, 0)，偏移 (0.5, 0.5)，边长 (0.5, 0.5)：
        中心 = 8，边长 = 8 -> 框 (4, 4) - (12, 12)
    """
    config = _make_config(num_classes=2, input_size=64)
    anchors, channels, height, width = 3, 5 + 2, 4, 4
    tensor = np.zeros((1, anchors, channels, height, width), dtype=np.float32)
    tensor[0, 0, :, 0, 0] = _row(0.5, 0.5, 0.5, 0.5, 0.95, [0.95, 0.0])

    result = decode_yolo_outputs([tensor], config)

    assert len(result) == 1
    assert result[0].class_name == "alpha"
    assert abs(result[0].xmin - 4.0) < 1e-4
    assert abs(result[0].xmax - 12.0) < 1e-4


def test_grid_offsets_are_applied():
    """回归测试：必须把「网格偏移」换算成绝对坐标。

    这是最容易出错的一步：直接用 tx 当像素坐标，框会挤在画面左上角。
    三个不同网格上的同一个框，解码后应落在画面里三个不同位置。
    """
    config = _make_config(num_classes=1, input_size=64)
    channels = 5 + 1

    # 8x8 网格（stride 8），放在 (row=7, col=0)
    tensor = np.zeros((1, channels, 8, 8), dtype=np.float32)
    tensor[0, :, 7, 0] = _row(0.5, 0.5, 1.0, 1.0, 0.9, [0.9])

    result = decode_yolo_outputs([tensor], config)

    assert len(result) == 1
    # x_center = (0 + 0.5) * 8 = 4, y_center = (7 + 0.5) * 8 = 60, w = h = 8
    assert abs(result[0].center_x - 4.0) < 1e-4, (
        f"x 中心应为 4，实际 {result[0].center_x}；"
        f"若得到 0.5 说明网格偏移没有乘以 stride"
    )
    assert abs(result[0].center_y - 60.0) < 1e-4, (
        f"y 中心应为 60，实际 {result[0].center_y}"
    )
    assert abs(result[0].width - 8.0) < 1e-4


def test_decodes_multiple_heads():
    """跨尺度重复框要被 NMS 合并，而真正分开的目标要保留。

    构造要把「网格单元中心」算准，否则两个头指的就不是同一个位置了：
        8x8  头 -> stride 8， 单元宽 8， 想让中心落在 28 就取单元 (3, 3)
        4x4  头 -> stride 16，单元宽 16，中心 28 需要 ty = 28/16 - 1 = 0.75
        2x2  头 -> stride 32，单元宽 32，中心只有 16 和 48，放不下 28
    所以这里只用 8x8 与 4x4 两个头制造重复框（两者中心都是 28，IoU=1），
    再用 2x2 头放一个远处目标 (4, 4)，验证 NMS 不会把它一起合掉。
    """
    config = _make_config(num_classes=1, input_size=64)
    channels = 5 + 1

    large = np.zeros((1, channels, 8, 8), dtype=np.float32)
    medium = np.zeros((1, channels, 4, 4), dtype=np.float32)
    small = np.zeros((1, channels, 2, 2), dtype=np.float32)

    # 同一个目标：中心 (28, 28)、边长 12
    # 8x8 头： (3 + 0.5) * 8 = 28， 12/8 = 1.5
    large[0, :, 3, 3] = _row(0.5, 0.5, 12 / 8.0, 12 / 8.0, 0.70, [0.7])
    # 4x4 头： (1 + 0.75) * 16 = 28， 12/16 = 0.75
    medium[0, :, 1, 1] = _row(0.75, 0.75, 12 / 16.0, 12 / 16.0, 0.90, [0.9])
    # 另一个远处的目标：2x2 头单元 (0,0) 中心 -> (0.5 * 32) = 16
    small[0, :, 0, 0] = _row(0.5, 0.5, 12 / 32.0, 12 / 32.0, 0.60, [0.6])

    result = decode_yolo_outputs([large, medium, small], config)

    assert len(result) == 2, (
        f"应得到 2 个框（重复的合并、远处的保留），实际 {len(result)} 个: "
        f"{[(round(d.center_x), round(d.center_y), round(d.score, 2)) for d in result]}"
    )
    # 结果按分数降序。
    # 注意 YOLOv5 的最终置信度 = objectness × 类别分，所以 medium 头是 0.9 × 0.9 = 0.81，
    # 而不是 0.9。同一位置上的 large 头只有 0.7 × 0.7 = 0.49，被 NMS 抑制掉。
    assert abs(result[0].score - 0.81) < 1e-5
    assert abs(result[0].center_x - 28.0) < 0.5
    assert abs(result[0].center_y - 28.0) < 0.5
    # 远处那个目标保留，中心 (16, 16)，分数 = 0.6 × 0.6
    assert abs(result[1].score - 0.36) < 1e-5
    assert abs(result[1].center_x - 16.0) < 0.5
    assert abs(result[1].center_y - 16.0) < 0.5


def test_explicit_stride_overrides_auto():
    """参数里显式给了 strides 时应当优先于自动推算。"""
    config = _make_config(
        num_classes=1, input_size=64, strides={2: 40}
    )
    tensor = np.zeros((1, 5 + 1, 2, 2), dtype=np.float32)
    tensor[0, :, 0, 0] = _row(0.5, 0.5, 0.5, 0.5, 0.9, [0.9])

    result = decode_yolo_outputs([tensor], config)

    assert len(result) == 1
    # 用 stride 40 而不是自动推算的 32
    assert abs(result[0].center_x - 20.0) < 1e-4
    assert abs(result[0].width - 20.0) < 1e-4


def test_objectness_multiplies_class_score():
    config = _make_config(num_classes=2)
    rows = np.zeros((1, 1, 1, 5 + 2), dtype=np.float32)
    rows[0, 0, 0] = _row(0.5, 0.5, 0.5, 0.5, 0.5, [0.8, 0.1])

    result = decode_yolo_outputs([rows], config)

    assert len(result) == 1
    assert abs(result[0].score - 0.4) < 1e-6


def test_conf_threshold_filters_everything():
    config = _make_config(conf_threshold=0.9)
    rows = np.zeros((1, 1, 1, 5 + 3), dtype=np.float32)
    rows[0, 0, 0] = _row(0.5, 0.5, 0.5, 0.5, 0.3, [0.3, 0.0, 0.0])

    assert decode_yolo_outputs([rows], config) == []


def test_yolov8_layout_without_objectness():
    """YOLOv8 是 4 + 类别数，没有 objectness，且输出是 (1, C, N) 大表。"""
    config = _make_config(num_classes=3)
    rows = np.zeros((1, 4 + 3, 10), dtype=np.float32)
    # YOLOv8 已经输出绝对像素坐标，不需要网格还原
    rows[0, :4, 0] = [20, 20, 6, 6]
    rows[0, 4:, 0] = [0.1, 0.85, 0.05]

    result = decode_yolo_outputs([rows], config)

    assert len(result) == 1
    assert result[0].class_name == "beta"
    assert abs(result[0].score - 0.85) < 1e-6
    # 坐标应原样保留（中心 20，宽 6）
    assert abs(result[0].center_x - 20.0) < 1e-4
    assert abs(result[0].width - 6.0) < 1e-4


def test_letterbox_maps_box_back_to_original_image():
    """letterbox + 解码后，框应该回到原图坐标系。

    手工按 letterbox 的几何构造 LetterboxInfo，专门验证「解码 + 坐标还原」
    这一段数学。80x40 的图放到 64x64：ratio=0.8，缩放到 64x32，上下各补 16。

    框放在 8x8 网格的单元 (3, 3) 中心、边长 1 格：
        输入坐标中心 = (3.5 * 8, 3.5 * 8) = (28, 28)，边长 8
    -> 原图：x = (28 - 0) / 0.8 = 35，y = (28 - 16) / 0.8 = 15，边长 = 10
    """
    config = _make_config(num_classes=1, input_size=64)
    info = LetterboxInfo(
        ratio=0.8,
        pad_x=0.0,
        pad_y=16.0,
        input_size=64,
        original_width=80,
        original_height=40,
    )

    rows = np.zeros((1, 5 + 1, 8, 8), dtype=np.float32)
    rows[0, :, 3, 3] = _row(0.5, 0.5, 1.0, 1.0, 0.9, [0.9])

    result = decode_yolo_outputs([rows], config, letterbox_info=info)

    assert len(result) == 1
    assert abs(result[0].center_x - 35.0) < 0.5
    assert abs(result[0].center_y - 15.0) < 0.5
    # 框必须落在原图范围内
    assert 0.0 <= result[0].xmin and result[0].xmax <= 80.0
    assert 0.0 <= result[0].ymin and result[0].ymax <= 40.0


def test_letterbox_geometry_is_uniform():
    """letterbox 的宽高必须同比例，否则水平方向会整体偏移。

    用一个宽高比很极端的图来暴露问题：320x64 放到 640。
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        return  # 没有 cv2 时跳过（板子上一定有）

    from recognize_yolo.preprocess import letterbox

    frame = np.zeros((64, 320, 3), dtype=np.uint8)
    padded, info = letterbox(frame, 640, stride=32)

    assert padded.shape == (640, 640, 3)
    scaled_width = 640 - info.pad_x * 2
    scaled_height = 640 - info.pad_y * 2
    original_ratio = 320.0 / 64.0
    scaled_ratio = scaled_width / float(scaled_height)
    assert abs(scaled_ratio - original_ratio) / original_ratio < 0.06, (
        f"缩放后宽高比 {scaled_ratio:.3f} 与原始 {original_ratio:.3f} 偏差过大"
    )


def test_channel_count_mismatch_raises():
    config = _make_config(num_classes=80)
    rows = np.zeros((1, 5 + 3, 4, 4), dtype=np.float32)

    try:
        decode_yolo_outputs([rows], config)
    except ValueError as error:
        assert "num_classes" in str(error)
    else:  # pragma: no cover
        raise AssertionError("通道数不匹配时应当抛 ValueError")


def test_to_nchw_float_normalizes_and_reorders():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # B 通道
    blob = to_nchw_float(image, scale=1.0 / 255.0, rgb=True)

    assert blob.shape == (1, 3, 2, 2)
    assert blob.dtype == np.float32
    # BGR -> RGB 后，B 通道变成最后一个
    assert abs(blob[0, 0, 0, 0] - 0.0) < 1e-6
    assert abs(blob[0, 2, 0, 0] - 1.0) < 1e-6
