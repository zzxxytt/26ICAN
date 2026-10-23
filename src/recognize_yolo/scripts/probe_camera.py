#!/usr/bin/env python3
"""摄像头参数协商探针：查清 CAP_PROP_* 为什么设不上。

在板子上运行：
    python3 probe_camera.py
    python3 probe_camera.py --device /dev/video1
    python3 probe_camera.py --force-v4l2      # 强制 V4L2 后端对比

它会依次尝试多种打开方式与参数组合，每次打印：
  * 使用的后端
  * 实际协商到的分辨率 / 帧率 / 像素格式
  * 设置前后是否发生变化

目的是回答一个问题：**怎么才能真的拿到 640x480 @ 30fps MJPG**。
"""

from __future__ import annotations

import argparse
import sys

import cv2


def fourcc_to_str(value: int) -> str:
    """把 CAP_PROP_FOURCC 的整数值转成 'MJPG' 这样的四字符码。"""
    try:
        code = int(value)
    except (TypeError, ValueError):
        return f"(非法值 {value!r})"
    if code == 0:
        return "(0，未报告)"
    chars = "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))
    return f"{chars} ({code})"


def describe(capture: cv2.VideoCapture) -> str:
    """读出当前的分辨率 / 帧率 / 格式。"""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = capture.get(cv2.CAP_PROP_FOURCC)
    return f"{width}x{height} @ {fps:.1f}fps 格式={fourcc_to_str(fourcc)}"


def try_open(source, backend: int | None, label: str, log: list) -> None:
    """按指定后端打开，并报告初始状态。"""
    print()
    print("=" * 70)
    print(f"【{label}】")
    print("=" * 70)

    if backend is None:
        capture = cv2.VideoCapture(source)
        backend_name = "自动选择"
    else:
        capture = cv2.VideoCapture(source, backend)
        backend_name = "指定"

    if not capture.isOpened():
        print("  打开失败")
        log.append((label, "打开失败"))
        return

    try:
        actual_backend = capture.getBackendName()
    except Exception as error:
        actual_backend = f"(取不到: {error})"

    print(f"  后端: {actual_backend}  ({backend_name})")
    print(f"  打开后初始状态: {describe(capture)}")

    # ---- 尝试设置 MJPG + 640x480 + 30fps，顺序：格式 -> 分辨率 -> 帧率 ----
    print()
    print("  --- 按要求设置 MJPG / 640x480 / 30fps（格式优先）---")
    fourcc_value = cv2.VideoWriter_fourcc(*"MJPG")
    ok_fourcc = capture.set(cv2.CAP_PROP_FOURCC, fourcc_value)
    print(f"  set(FOURCC, MJPG) 返回 {ok_fourcc}")

    ok_w = capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    ok_h = capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"  set(WIDTH=640) 返回 {ok_w}, set(HEIGHT=480) 返回 {ok_h}")

    ok_fps = capture.set(cv2.CAP_PROP_FPS, 30.0)
    print(f"  set(FPS=30) 返回 {ok_fps}")

    after = describe(capture)
    print(f"  设置后状态: {after}")

    # ---- 实际读一帧，确认分辨率 ----
    ok, frame = capture.read()
    if ok and frame is not None:
        height, width = frame.shape[:2]
        print(f"  实际读到的一帧: {width}x{height}")
        frame_info = f"{width}x{height}"
    else:
        print("  读帧失败")
        frame_info = "读帧失败"

    verdict = "成功" if frame_info == "640x480" else f"不符（{frame_info}）"
    log.append((label, f"{actual_backend} -> {after} / 帧 {frame_info} -> {verdict}"))

    # ---- 再试：只设格式，看 FOURCC 是否能读回 ----
    print()
    print("  --- 单独验证 FOURCC 是否可写 ---")
    capture.set(cv2.CAP_PROP_FOURCC, fourcc_value)
    readback = capture.get(cv2.CAP_PROP_FOURCC)
    print(f"  写入 MJPG 后读回: {fourcc_to_str(readback)}")
    if int(readback) == 0:
        print("  -> FOURCC 读回为 0：这个后端不支持通过 CAP_PROP_FOURCC 设置格式")

    capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="摄像头参数协商探针")
    parser.add_argument("--device", default="0",
                        help="摄像头编号(0) 或设备路径(/dev/video0)")
    args = parser.parse_args()

    source = args.device
    if source.isdigit():
        source = int(source)

    print(f"OpenCV {cv2.__version__}")
    print(f"设备: {args.device}")
    try:
        print(f"可用后端: {[b for b in cv2.videoio_registry.getBackends()]}")
    except Exception:
        pass

    log: list = []

    # 后端常量在不同版本里名字略有差异，逐个探测
    backends = [("自动选择", None)]
    for name in ("CAP_V4L2", "CAP_GSTREAMER", "CAP_ANY"):
        value = getattr(cv2, name, None)
        if value is not None and name != "CAP_ANY":
            backends.append((name, value))

    for label, backend in backends:
        try:
            try_open(source, backend, label, log)
        except Exception as error:
            print(f"  [异常] {type(error).__name__}: {error}")
            log.append((label, f"异常 {error}"))

    print()
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    for label, result in log:
        print(f"  {label:14} {result}")

    print()
    print("=" * 70)
    print("怎么读这个结果")
    print("=" * 70)
    print("  * 如果某个后端下『实际读到的一帧』是 640x480，就用它对应的方式")
    print("  * 如果所有后端都拿到 1280x720，说明 CAP_PROP 设不上，")
    print("    需要在 CameraSource 里改用 GStreamer pipeline 字符串指定格式")
    print("  * FOURCC 读回 0 也是同一个信号")
    return 0


if __name__ == "__main__":
    sys.exit(main())
