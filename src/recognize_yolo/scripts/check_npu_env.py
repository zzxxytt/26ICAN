#!/usr/bin/env python3
"""上板第一步：一条命令确认 A733 的 NPU 环境到底通没通。

不用 ROS 也能跑：

    python3 scripts/check_npu_env.py
    python3 scripts/check_npu_env.py --model ~/models/yolov5s_rt_uint8_a733.nb

输出分成「必须项」和「可选项」，全绿再往下走，否则先去解决红项。
"""

from __future__ import annotations

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"

results: List[Tuple[str, str, str]] = []


def record(level: str, title: str, detail: str = "") -> None:
    results.append((level, title, detail))
    print(f"{level} {title}" + (f"  -> {detail}" if detail else ""))


def check_architecture() -> None:
    """确认是 aarch64 的 ARM 板子，而不是在电脑上误跑。"""
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        record(OK, f"CPU 架构: {machine}")
    else:
        record(FAIL, f"CPU 架构: {machine}", "NPU 脚本必须在香橙派板子上运行")


def check_soc() -> None:
    """读 /proc/device-tree 确认 SoC 是 A733 系列。"""
    candidates = glob.glob("/proc/device-tree/compatible") + glob.glob(
        "/sys/firmware/devicetree/base/compatible"
    )
    if not candidates:
        record(WARN, "SoC 识别", "找不到 device-tree compatible 节点")
        return

    try:
        with open(candidates[0], "rb") as handle:
            raw = handle.read()
    except OSError as error:
        record(WARN, "SoC 识别", f"读取失败: {error}")
        return

    compatible = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    if "a733" in compatible.lower() or "sun60i" in compatible.lower():
        record(OK, "SoC 识别: Allwinner A733 系列", compatible[:80])
    else:
        record(
            WARN,
            "SoC 识别",
            f"未匹配到 A733（compatible={compatible[:80]}）。"
            f"若确实不是 A733，请改用 CPU/ONNX 路线",
        )


def check_kernel() -> None:
    release = platform.release()
    record(OK, f"内核版本: {release}")


def check_npu_device_nodes() -> None:
    """VIPLite 会在 /dev 下暴露设备节点，名字因 SDK 版本而异。"""
    patterns = [
        "/dev/vipcore",
        "/dev/vip*",
        "/dev/galcore*",
        "/dev/npu*",
    ]
    found: List[str] = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    found = sorted(set(found))

    if found:
        record(OK, "NPU 设备节点", ", ".join(found))
    else:
        record(
            FAIL,
            "NPU 设备节点",
            "未找到 /dev/vipcore 或类似节点。"
            "说明内核里没加载 VIPLite 驱动，需要换用官方带 NPU 的固件",
        )


def check_vip_lite_library() -> None:
    """找 libvip-lite.so，worker 运行时要能加载到。"""
    found: List[str] = []
    search_dirs = [
        "/usr/lib",
        "/usr/local/lib",
        "/usr/lib/aarch64-linux-gnu",
        "/lib/aarch64-linux-gnu",
        os.path.expanduser("~/linux_aw_npu"),
        os.path.expanduser("~/npu_runtime"),
    ]
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for pattern in ("libvip-lite.so*", "libvip_lite.so*"):
            found.extend(glob.glob(os.path.join(directory, "**", pattern), recursive=True))

    if found:
        record(OK, "VIPLite 运行时库", found[0])
    else:
        record(
            FAIL,
            "VIPLite 运行时库",
            "未找到 libvip-lite.so。需要从 AW NPU Model Zoo 的 "
            "common/npuruntime/lib_linux_aarch64/A733/ 复制过来",
        )


def check_vpm_run() -> None:
    """vpm_run 是官方自带的 NBG 测试工具，拿它验证模型最省事。"""
    path = shutil.which("vpm_run")
    if path:
        record(OK, "vpm_run 测试工具", path)
    else:
        record(
            WARN,
            "vpm_run 测试工具",
            "未找到 vpm_run，无法用官方工具直测 .nb；"
            "可以只依赖本包的 yolo_npu_worker",
        )


def check_npu_runtime_processes() -> None:
    """看一眼有没有其它进程占着 NPU。"""
    try:
        output = subprocess.run(
            ["bash", "-lc", "lsmod | grep -i -E 'vip|npu|galcore' || true"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception as error:
        record(WARN, "NPU 内核模块", f"查询失败: {error}")
        return

    if output:
        record(OK, "NPU 内核模块", output.replace("\n", " | ")[:120])
    else:
        record(WARN, "NPU 内核模块", "lsmod 里没看到相关模块（可能已内建进内核）")


def check_model_file(model_path: Optional[str]) -> None:
    if not model_path:
        record(WARN, "NBG 模型", "未通过 --model 指定，跳过检查")
        return

    expanded = os.path.expanduser(model_path)
    if not os.path.isfile(expanded):
        record(FAIL, "NBG 模型", f"文件不存在: {expanded}")
        return

    size_mb = os.path.getsize(expanded) / (1024 * 1024)
    record(OK, "NBG 模型", f"{expanded} ({size_mb:.2f} MB)")


def check_python_packages() -> None:
    """列出关键 Python 依赖。"""
    packages = {
        "cv2": "python3-opencv",
        "numpy": "python3-numpy",
    }
    missing: List[str] = []
    for module, apt_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{module} ({apt_name})")

    if missing:
        record(FAIL, "Python 依赖", "缺失: " + ", ".join(missing))
    else:
        record(OK, "Python 依赖", "cv2 / numpy 均可导入")

    try:
        import onnxruntime  # noqa: F401

        record(OK, "ONNX Runtime", "可用，CPU 兜底路线可用")
    except ImportError:
        record(
            WARN,
            "ONNX Runtime",
            "未安装，CPU 兜底路线不可用（sudo apt install python3-onnxruntime）",
        )


def check_ros_environment() -> None:
    distro = os.environ.get("ROS_DISTRO", "")
    if distro:
        record(OK, "ROS 2 发行版", f"ROS_DISTRO={distro}")
    else:
        record(WARN, "ROS 2 发行版", "环境里没有 ROS_DISTRO，请先 source /opt/ros/<distro>/setup.bash")

    try:
        import rclpy  # noqa: F401

        record(OK, "rclpy", "可导入")
    except ImportError:
        record(WARN, "rclpy", "当前 Python 环境导入不到 rclpy（source 后再试）")


def check_worker_binary(explicit_path: Optional[str]) -> Optional[str]:
    """找编译好的 yolo_npu_worker。"""
    if explicit_path:
        expanded = os.path.expanduser(explicit_path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            record(OK, "yolo_npu_worker", expanded)
            return expanded
        record(FAIL, "yolo_npu_worker", f"不存在或不可执行: {expanded}")
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "yolo_npu_worker"),
        os.path.join(here, "..", "lib", "recognize_yolo", "yolo_npu_worker"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            record(OK, "yolo_npu_worker", os.path.abspath(candidate))
            return os.path.abspath(candidate)

    record(
        WARN,
        "yolo_npu_worker",
        "未找到。需要在工作空间执行 colcon build，并保证编译时能找到 VIPLite",
    )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="检查香橙派 4 Pro 的 NPU 推理环境")
    parser.add_argument("--model", default="", help="NBG 模型路径，例如 ~/models/xxx.nb")
    parser.add_argument("--worker", default="", help="yolo_npu_worker 可执行文件路径")
    args = parser.parse_args()

    print("=" * 72)
    print("香橙派 4 Pro (Allwinner A733 / Vivante VIP9000) NPU 环境自检")
    print("=" * 72)

    check_architecture()
    check_soc()
    check_kernel()
    check_npu_device_nodes()
    check_npu_runtime_processes()
    check_vip_lite_library()
    check_vpm_run()
    check_model_file(args.model or None)
    check_python_packages()
    check_ros_environment()
    worker = check_worker_binary(args.worker or None)

    failures = [item for item in results if item[0] == FAIL]
    warnings = [item for item in results if item[0] == WARN]

    print("-" * 72)
    print(f"结论: {len(failures)} 项失败, {len(warnings)} 项警告, "
          f"{len(results) - len(failures) - len(warnings)} 项正常")

    if failures:
        print("\n需要先解决的问题：")
        for _, title, detail in failures:
            print(f"  - {title}: {detail}")
        return 1

    print("\n环境看起来可用。")
    if worker:
        print(f"下一步：{worker} --model <你的.nb> --probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
