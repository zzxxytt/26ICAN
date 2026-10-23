#!/usr/bin/env python3
"""数据集采集工具：在板子上运行，把摄像头画面采集成训练用图片。

支持的三种采法，覆盖数据采集里最常见的需求：

    interval  按固定间隔自动抽帧（最常用）
              适合让机器人绕目标走一圈、或手拿着目标在镜头前缓慢变换角度
    manual    按键才存一张
              适合精确控制"每个角度来几张"，不浪费存储
    video     录一段视频，回到电脑上再抽帧
              适合快速过一遍场景，事后慢慢挑

设计上刻意复用 recognize_yolo.camera_source.CameraSource，
这样采集用的画面方向、镜像、后端选择与实际推理时**完全一致**
—— 否则会出现"采集的图和推理看到的图方向不同"这种极难排查的问题。

依赖：python3-opencv（相机）、python3-numpy。不依赖 ROS，可以直接跑。

用法示例：

    # 每 0.5 秒存一张，最多 300 张，存到 ~/dataset_raw/session_01
    python3 capture_dataset.py --mode interval --interval 0.5 --max-frames 300

    # 手动模式：窗口里按空格存图，按 q 退出
    python3 capture_dataset.py --mode manual

    # 录像模式：录到按 q 为止
    python3 capture_dataset.py --mode video

    # 摄像头装反了？按实际安装方向来
    python3 capture_dataset.py --mode interval --rotation 180
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 终端按键读取：POSIX 用 termios，Windows 用 msvcrt。
# 这样纯 SSH（没有图形界面）也能按键采集。
if os.name == "nt":  # pragma: no cover - 板上不会走这条
    import msvcrt
else:
    import termios
    import tty

# 允许从包内导入（板上直接跑脚本时用）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from recognize_yolo.camera_source import CameraSource
except ImportError:  # pragma: no cover - 包不可用时给出明确提示
    print("无法导入 recognize_yolo.camera_source。")
    print("请在包目录下运行，或先执行：")
    print("    export PYTHONPATH=$HOME/wheel_robot/src/recognize_yolo:$PYTHONPATH")
    raise SystemExit(2)


WINDOW_NAME = "capture_dataset (space=save  q/ESC=quit)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集 YOLO 训练数据集的原始图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("key", "interval", "manual", "video"),
                        default="interval",
                        help="采集模式：key=终端按键存图（SSH 友好，推荐）；"
                             "interval=定时抽帧；manual=窗口里按键存图；video=录像")
    parser.add_argument("--out", type=Path, default=None,
                        help="输出目录；默认 ~/dataset_raw/<日期时间>")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="interval 模式的抽帧间隔（秒），默认 0.5")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多保存多少张，0 表示不限制")
    parser.add_argument("--label", default="",
                        help="本次采集的类别名或场景备注，写进 session.json 便于回溯")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG 质量 1~100，默认 95")

    # 摄像头参数：默认值与 recognize_yolo 的参数文件保持一致
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-device", default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--pixel-format", default="MJPG",
        help="V4L2 像素格式。默认 MJPG：未压缩的 YUYV 在高分辨率下帧率很低"
             "（例如 1280x720 只有 10fps），而 MJPG 同样分辨率能到 30fps。"
             "设为 AUTO 则交给驱动协商。",
    )
    parser.add_argument(
        "--capture-backend", default="auto", choices=("auto", "v4l2", "gstreamer"),
        help="OpenCV 采集后端。auto 时：指定了 pixel-format 就自动用 V4L2"
             "（GStreamer 后端不支持 CAP_PROP_*，参数会被全部忽略）；"
             "pixel-format=AUTO 时让 OpenCV 自选。"
             "板上实测只有 V4L2 能真正设置 MJPG 和分辨率。",
    )
    parser.add_argument(
        "--frame-rate", type=float, default=30.0,
        help="请求的帧率。0 表示不指定，由驱动决定。"
             "按键采集时建议保持 30：取帧延迟直接影响按下瞬间画面的新鲜度。",
    )
    parser.add_argument("--rotation", type=int, default=0, choices=(0, 90, 180, 270))
    parser.add_argument("--mirror", default="none",
                        choices=("none", "horizontal", "vertical", "both"))
    parser.add_argument("--no-preview", action="store_true",
                        help="不开预览窗口（无显示器 / 纯 SSH 环境用；key 模式本来就不开窗口）")
    return parser.parse_args()


def make_output_dir(raw: Path | None, label: str) -> Path:
    """建输出目录并写一个说明文件，方便几天后回想这批数据是怎么采的。"""
    if raw is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{label}" if label else ""
        directory = Path.home() / "dataset_raw" / f"{stamp}{suffix}"
    else:
        directory = raw

    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "videos").mkdir(parents=True, exist_ok=True)
    return directory


def save_frame(frame: np.ndarray, directory: Path, index: int, quality: int) -> Path:
    """存一张 jpg。文件名带序号，保证排序与采集顺序一致。"""
    path = directory / "images" / f"frame_{index:05d}.jpg"
    ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"写入失败: {path}")
    return path


def interval_mode(camera: CameraSource, directory: Path, args: argparse.Namespace) -> int:
    """按固定间隔抽帧。"""
    saved = 0
    last_save = 0.0
    started = time.monotonic()

    print(f"按间隔 {args.interval}s 抽帧。"
          f"{'按 q 或 ESC 结束' if not args.no_preview else '按 Ctrl+C 结束'}")

    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue

        now = time.monotonic()
        if now - last_save >= args.interval:
            saved += 1
            path = save_frame(frame, directory, saved, args.jpeg_quality)
            last_save = now
            elapsed = now - started
            print(f"\r已保存 {saved:4d} 张  ({elapsed:.0f}s, "
                  f"{saved / max(elapsed, 1e-6):.1f} 张/秒)  {path.name}",
                  end="", flush=True)

            if args.max_frames and saved >= args.max_frames:
                print(f"\n已达到上限 {args.max_frames} 张，结束。")
                break

        if not args.no_preview:
            preview = frame.copy()
            _draw_hud(preview, saved, args)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    print()
    return saved


class TerminalKeys:
    """从终端读单个按键，不需要图形界面。

    为什么需要它：manual 模式靠 cv2.imshow 的窗口收按键，而 SSH 远程
    （没有 X11 转发）根本开不出窗口。这个类直接读 stdin，
    MobaXterm / ssh 终端里就能按空格存图。

    实现要点：
      * POSIX 下用 termios 把终端切到 cbreak 模式（按键立即到达，不等回车）
      * 同时关掉 ECHO，否则按下的字符会回显、把提示行冲乱
      * timeout=0.05 的轮询间隔，让采集循环保持在约 20Hz，
        既不空转烧 CPU，按键也不会觉得迟钝
      * 必须保证 restore() 一定被调用，否则退出后终端会一直不回显
    """

    def __init__(self, timeout: float = 0.05) -> None:
        self.timeout = timeout
        self._fd = None
        self._saved = None
        self._is_tty = sys.stdin.isatty()

    def __enter__(self) -> "TerminalKeys":
        if self._is_tty and os.name != "nt":
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            # 关掉行缓冲与回显，但保留 ISIG，这样 Ctrl+C 仍能中断
            new = termios.tcgetattr(self._fd)
            new[3] &= ~(termios.ICANON | termios.ECHO)
            new[6][termios.VMIN] = 0
            new[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSADRAIN, new)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.restore()
        return False

    def restore(self) -> None:
        if self._fd is not None and self._saved is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:
                pass
            self._fd = None
            self._saved = None

    def get(self) -> str:
        """返回这一个轮询周期内按下的字符（没有则返回空串）。"""
        if not self._is_tty:
            # 不是终端（例如被重定向/放进管道）时退化为按行读，
            # 这样至少还能用，只是需要按回车。
            ready, _, _ = select.select([sys.stdin], [], [], self.timeout)
            if not ready:
                return ""
            line = sys.stdin.readline()
            return line.strip()[:1] if line else "q"

        if os.name == "nt":  # pragma: no cover
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    return msvcrt.getwch()
                time.sleep(0.01)
            return ""

        ready, _, _ = select.select([sys.stdin], [], [], self.timeout)
        if not ready:
            return ""
        try:
            data = os.read(sys.stdin.fileno(), 16)
        except OSError:
            return ""
        return data.decode("utf-8", errors="replace")


def key_mode(camera: CameraSource, directory: Path, args: argparse.Namespace) -> int:
    """终端按键存图：按一下空格存一张。SSH 远程可用，不需要图形界面。

    与 manual 模式的区别：不收 cv2 窗口的按键，而是直接读终端 stdin。
    """
    print("=" * 60)
    print("终端按键采集模式")
    print("  空格  -> 存一张")
    print("  a     -> 查看已保存数量与耗时")
    print("  q/ESC -> 结束采集")
    print("  只存图不显示窗口，SSH 远程也能用")
    print("=" * 60)
    # 隐藏光标，避免它跟着状态行乱跳
    print("\033[?25l", end="", flush=True)

    saved = 0
    started = time.monotonic()
    last_status = started
    last_report = started

    try:
        with TerminalKeys() as keys:
            while True:
                frame = camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                key = keys.get()

                if key == " ":
                    saved += 1
                    path = save_frame(frame, directory, saved, args.jpeg_quality)
                    elapsed = time.monotonic() - started
                    shape = frame.shape[1], frame.shape[0]
                    print(f"\r\033[K已保存 {saved:4d} 张  "
                          f"({shape[0]}x{shape[1]}, {elapsed:.0f}s)  {path.name}",
                          end="", flush=True)
                    if args.max_frames and saved >= args.max_frames:
                        print(f"\n已达到上限 {args.max_frames} 张，结束。")
                        break
                elif key == "a":
                    elapsed = time.monotonic() - started
                    print(f"\r\033[K[统计] 已保存 {saved} 张，"
                          f"已运行 {elapsed:.0f}s，输出目录 {directory}",
                          end="", flush=True)
                elif key in ("q", "\x1b", "\x03"):  # q / ESC / Ctrl+C
                    print()
                    break

                # 每 10 秒提示一次「还活着」，否则终端看着像卡住了
                now = time.monotonic()
                if now - last_report > 10.0:
                    last_report = now
                    print(f"\r\033[K等待按键…… 已保存 {saved} 张 "
                          f"(空格=存图, q=结束)", end="", flush=True)
                last_status = now
    finally:
        # 无论如何都要恢复光标
        print("\033[?25h", end="", flush=True)

    print()
    return saved


def manual_mode(camera: CameraSource, directory: Path, args: argparse.Namespace) -> int:
    """按键存图。"""
    if args.no_preview:
        print("manual 模式需要预览窗口来按键，不能与 --no-preview 同时使用。")
        return 0

    saved = 0
    print("按 空格 存一张，按 q 或 ESC 结束。")

    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue

        preview = frame.copy()
        _draw_hud(preview, saved, args)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            saved += 1
            path = save_frame(frame, directory, saved, args.jpeg_quality)
            print(f"  已保存 {saved} 张 -> {path.name}")
            if args.max_frames and saved >= args.max_frames:
                print(f"已达到上限 {args.max_frames} 张，结束。")
                break
        elif key in (ord("q"), 27):
            break

    return saved


def video_mode(camera: CameraSource, directory: Path, args: argparse.Namespace) -> int:
    """录一段视频，回到电脑上再抽帧。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / "videos" / f"clip_{stamp}.mp4"

    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    frames = 0

    print("开始录像。按 q 或 ESC 结束。")
    print("提示：录完可以在电脑上用 extract_frames.py 抽帧。")

    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue

        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"无法创建视频文件: {path}")
                print("如果缺 mp4v 编码器，可以先用 interval 模式采图。")
                return 0

        writer.write(frame)
        frames += 1

        if not args.no_preview:
            preview = frame.copy()
            cv2.putText(preview, f"REC {frames} frames", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        else:
            if args.max_frames and frames >= args.max_frames * 30:
                break

    if writer is not None:
        writer.release()
    print(f"\n录像结束，共 {frames} 帧 -> {path}")
    return frames


def _draw_hud(frame: np.ndarray, saved: int, args: argparse.Namespace) -> None:
    """在预览画面上叠一行状态，避免盯着窗口不知道存了多少。"""
    text = f"saved={saved}"
    if args.mode == "interval":
        text += f"  interval={args.interval}s"
    if args.max_frames:
        text += f"  /{args.max_frames}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (32, 32, 32), -1)
    cv2.putText(frame, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)


def write_session_info(directory: Path, args: argparse.Namespace, saved: int) -> None:
    """把本次采集的参数记下来。数据集的"来龙去脉"比图片本身更容易丢。"""
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "saved_frames": saved,
        "label": args.label,
        "camera": {
            "camera_id": args.camera_id,
            "camera_device": args.camera_device,
            "width": args.width,
            "height": args.height,
            "pixel_format": args.pixel_format,
            "frame_rate": args.frame_rate,
            "capture_backend": args.capture_backend,
            "rotation": args.rotation,
            "mirror": args.mirror,
        },
        "interval_seconds": args.interval if args.mode == "interval" else None,
        "jpeg_quality": args.jpeg_quality,
        "note": (
            "rotation 与 mirror 必须与推理时 recognize_yolo 的参数一致，"
            "否则标注出来的框在部署后会左右/上下颠倒"
        ),
    }
    path = directory / "session.json"
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"采集信息已写入 {path}")


def main() -> int:
    args = parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    directory = make_output_dir(args.out, args.label)
    print(f"输出目录: {directory}")

    camera = CameraSource(
        camera_id=args.camera_id,
        camera_device=args.camera_device,
        width=args.width,
        height=args.height,
        frame_rate=args.frame_rate,
        pixel_format=args.pixel_format,
        rotation_degrees=args.rotation,
        mirror_mode=args.mirror,
        capture_backend=args.capture_backend,
    )
    print(f"摄像头已打开，旋转={args.rotation}°，镜像={args.mirror}")
    print(f"  实际参数: {camera.actual_width}x{camera.actual_height} "
          f"@ {camera.actual_fps:.1f} fps, 格式={camera.actual_format or '未报告'}")

    # 参数没完全生效时给出可操作的排查方向
    mismatch = (camera.actual_width, camera.actual_height) != (args.width, args.height)
    format_missing = args.pixel_format != "AUTO" and not camera.actual_format
    if mismatch or format_missing:
        print()
        print("  [注意] 摄像头没有完全接受请求的参数，按顺序排查：")
        print("     1) 后端：本脚本在指定 --pixel-format 时会自动用 V4L2，")
        print("        因为 GStreamer 后端不支持 CAP_PROP_*，参数会被静默忽略。")
        print("        可以用 probe_camera.py 实测各后端的真实行为：")
        print("          python3 src/recognize_yolo/scripts/probe_camera.py")
        print("     2) 摄像头编号：ls /dev/video* 确认 --camera-id 对不对")
        print("     3) 能力表：v4l2-ctl --device /dev/video0 --list-formats-ext")

    print()
    print("  提示：采集分辨率决定了标注和训练的分辨率，")
    print("        建议与部署时模型的输入尺寸保持一致（看 --probe 打印的输入边长）")

    saved = 0
    try:
        if args.mode == "key":
            saved = key_mode(camera, directory, args)
        elif args.mode == "interval":
            saved = interval_mode(camera, directory, args)
        elif args.mode == "manual":
            saved = manual_mode(camera, directory, args)
        else:
            saved = video_mode(camera, directory, args)
    except KeyboardInterrupt:
        print("\n收到中断，正在收尾……")
    finally:
        camera.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    write_session_info(directory, args, saved)

    print()
    print(f"采集完成：{saved} 张")
    if saved == 0:
        print("注意：一张都没存。如果不是你主动退出的，检查：")
        print("  - 是不是在 ssh 终端里跑（key 模式需要终端；用 --mode interval 可以完全免交互）")
        print("  - 空格键有没有被 MobaXterm 的快捷键截走")
    print("下一步（在电脑上）：")
    print("  1) 把 images 目录同步到电脑")
    print("  2) 用 labelImg 标注，格式选 YOLO")
    print("  3) 用 prepare_dataset.py 划分 train/val 并生成 dataset.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
