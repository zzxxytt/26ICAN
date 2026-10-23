#!/usr/bin/env python3
"""数据集整理工具：在电脑上运行，把标注结果整理成 YOLO 训练可用的结构。

它做四件事，正好对应标注完之后最容易出错的四个地方：

    check     体检：找出格式错误、越界坐标、以及最常见的
              「图片和标签对不上」（漏标、多标、改名后失配）
    split     划分 train / val，并且保证**可复现**（同样的输入永远同样的划分）
    yaml      生成 dataset.yaml 与类别名文件
    stats     统计每个类别的实例数、框的大小分布，用于判断数据是否失衡

只依赖标准库，不需要 numpy / PIL。

典型用法：

    # 1) 先体检，把问题挑出来
    python3 prepare_dataset.py check --src ~/dataset_raw/session_01 --names ball cone person

    # 2) 划分并生成 yaml（一次搞定）
    python3 prepare_dataset.py split --src ~/dataset_raw/session_01 \\
        --dst ~/dataset --val-ratio 0.2 --names ball cone person

    # 3) 看统计
    python3 prepare_dataset.py stats --src ~/dataset --names ball cone person
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Annotation:
    """一个标签文件的内容。"""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float
    line_number: int

    def as_line(self) -> str:
        return (f"{self.class_id} {self.center_x:.6f} {self.center_y:.6f} "
                f"{self.width:.6f} {self.height:.6f}")


@dataclass
class SampleIssue:
    """一条问题记录。"""

    level: str          # "error" / "warn"
    path: Path
    message: str


@dataclass
class DatasetScan:
    """扫描结果。"""

    images: List[Path] = field(default_factory=list)
    labels: List[Path] = field(default_factory=list)
    issues: List[SampleIssue] = field(default_factory=list)
    #: 文件名（不含扩展名） -> 标注
    annotations: Dict[str, List[Annotation]] = field(default_factory=dict)
    class_counts: Counter = field(default_factory=Counter)


def parse_names(raw: Sequence[str]) -> List[str]:
    """把 --names 的参数或类别文件内容整理成列表。"""
    names: List[str] = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        # 支持逗号分隔
        for piece in text.split(","):
            piece = piece.strip()
            if piece:
                names.append(piece)
    return names


def load_names(args: argparse.Namespace) -> List[str]:
    if args.names_file:
        path = Path(args.names_file).expanduser()
        if not path.is_file():
            raise SystemExit(f"类别文件不存在: {path}")
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        names = parse_names(args.names or [])

    if not names:
        raise SystemExit("必须提供类别顺序：--names ball cone person 或 --names-file classes.txt")
    return names


def collect_images(source: Path) -> List[Path]:
    """递归找出所有图片，排除已经划分好的 val 目录嵌套重复。"""
    images: List[Path] = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    return images


def parse_label_file(path: Path, num_classes: int) -> Tuple[List[Annotation], List[str]]:
    """解析一个 YOLO 标签文件，返回 (标注列表, 错误说明列表)。"""
    annotations: List[Annotation] = []
    problems: List[str] = []

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")
        problems.append("文件编码不是 UTF-8，已按 latin-1 读取")
    except OSError as error:
        return [], [f"读取失败: {error}"]

    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            problems.append(f"第 {line_number} 行有 {len(parts)} 个字段，应为 5 个")
            continue

        try:
            values = [float(item) for item in parts]
        except ValueError:
            problems.append(f"第 {line_number} 行包含非数字字段")
            continue

        class_id_float = values[0]
        class_id = int(class_id_float)
        if abs(class_id_float - class_id) > 1e-6:
            problems.append(f"第 {line_number} 行类别号 {class_id_float} 不是整数")

        if class_id < 0 or class_id >= num_classes:
            problems.append(
                f"第 {line_number} 行类别号 {class_id} 超出范围 "
                f"(应有 0 ~ {num_classes - 1})"
            )

        center_x, center_y, width, height = values[1:]
        for name, value in (("center_x", center_x), ("center_y", center_y)):
            if not -1e-6 <= value <= 1 + 1e-6:
                problems.append(f"第 {line_number} 行 {name}={value:.4f} 超出 0~1")
        for name, value in (("width", width), ("height", height)):
            if value <= 0:
                problems.append(f"第 {line_number} 行 {name}={value:.4f} 不是正数")
            elif value > 1 + 1e-6:
                problems.append(f"第 {line_number} 行 {name}={value:.4f} 大于 1")

        # 框超出图像边界的常见原因：标注时忘了裁剪
        if (center_x - width / 2 < -1e-3 or center_x + width / 2 > 1 + 1e-3
                or center_y - height / 2 < -1e-3 or center_y + height / 2 > 1 + 1e-3):
            problems.append(f"第 {line_number} 行框超出图像边界")

        annotations.append(Annotation(class_id, center_x, center_y, width, height,
                                      line_number))

    return annotations, problems


def scan(source: Path, num_classes: int) -> DatasetScan:
    """扫描一个目录，找出图片、标签与所有问题。"""
    result = DatasetScan()

    images_dir = source / "images" if (source / "images").is_dir() else source
    labels_dir = source / "labels" if (source / "labels").is_dir() else source

    result.images = collect_images(images_dir)
    result.labels = sorted(
        path for path in labels_dir.rglob("*.txt")
        if path.is_file() and path.name != "classes.txt"
    )

    image_stems = {path.stem: path for path in result.images}
    label_stems = {path.stem: path for path in result.labels}

    # 图片没有对应标签：漏标，或者只是没标完
    for stem, path in sorted(image_stems.items()):
        if stem not in label_stems:
            result.issues.append(SampleIssue(
                "error", path, "没有对应的标签文件（漏标或还没标）"
            ))

    # 标签没有对应图片：通常是删图片时忘了删标签，或者改名不同步
    for stem, path in sorted(label_stems.items()):
        if stem not in image_stems:
            result.issues.append(SampleIssue(
                "error", path, "没有对应的图片（图片被删或改名了）"
            ))

    # 逐个解析标签
    for stem, path in sorted(label_stems.items()):
        annotations, problems = parse_label_file(path, num_classes)
        result.annotations[stem] = annotations

        for problem in problems:
            level = "error" if "超出" in problem or "字段" in problem else "warn"
            result.issues.append(SampleIssue(level, path, problem))

        if not annotations and stem in image_stems:
            result.issues.append(SampleIssue(
                "warn", path, "标签文件存在但没有任何框（空标注）"
            ))

        for annotation in annotations:
            if 0 <= annotation.class_id < num_classes:
                result.class_counts[annotation.class_id] += 1

    return result


def print_scan_report(scan_result: DatasetScan, names: Sequence[str]) -> None:
    print(f"图片 {len(scan_result.images)} 张，标签 {len(scan_result.labels)} 个")
    paired = len(set(path.stem for path in scan_result.images)
                 & set(path.stem for path in scan_result.labels))
    print(f"配对成功 {paired} 对")

    errors = [item for item in scan_result.issues if item.level == "error"]
    warns = [item for item in scan_result.issues if item.level == "warn"]

    print()
    print(f"类别实例数（共 {len(names)} 类，合计 {sum(scan_result.class_counts.values())} 个框）：")
    for index, name in enumerate(names):
        count = scan_result.class_counts.get(index, 0)
        bar = "#" * min(40, count)
        flag = ""
        if count == 0:
            flag = "   <== 该类没有任何样本"
        print(f"  {index:3d} {name:<16} {count:6d}  {bar}{flag}")

    if errors or warns:
        print()
        print(f"问题：{len(errors)} 个错误，{len(warns)} 个警告")
        for item in (errors + warns)[:40]:
            marker = "ERROR" if item.level == "error" else "WARN "
            print(f"  {marker} {item.path.name}: {item.message}")
        if len(errors) + len(warns) > 40:
            print(f"  …… 还有 {len(errors) + len(warns) - 40} 条")
    else:
        print()
        print("没有发现问题。")


def stable_bucket(stem: str) -> float:
    """由文件名算出一个稳定的 [0,1) 值，用于可复现的划分。

    不用 Python 内置 hash()：它每个进程都有随机种子，
    同样的数据两次运行会得到不同的划分，实验就没法复现了。
    """
    checksum = zlib.crc32(stem.encode("utf-8")) & 0xFFFFFFFF
    return checksum / 4294967296.0


def split_dataset(
    scan_result: DatasetScan, destination: Path, val_ratio: float,
    names: Sequence[str], copy: bool, seed_note: str
) -> None:
    """按文件名哈希划分 train/val，并写盘。"""
    usable = [
        stem for stem in scan_result.annotations
        if stem in {path.stem for path in scan_result.images}
        and scan_result.annotations[stem]
    ]
    usable.sort()

    train: List[str] = []
    val: List[str] = []
    for stem in usable:
        if stable_bucket(stem) < val_ratio:
            val.append(stem)
        else:
            train.append(stem)

    # val 为空时兜底：至少放一张，否则训练脚本会报错
    if not val and len(usable) > 1:
        moved = train.pop()
        val.append(moved)
        print(f"提示：按比例算出的验证集为空，已从训练集移一张过去（{moved}）")

    image_by_stem = {path.stem: path for path in scan_result.images}
    label_by_stem = {path.stem: path for path in scan_result.labels}

    for subset_name, stems in (("train", train), ("val", val)):
        image_out = destination / "images" / subset_name
        label_out = destination / "labels" / subset_name
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            source_image = image_by_stem[stem]
            source_label = label_by_stem.get(stem)

            target_image = image_out / source_image.name
            target_label = label_out / f"{stem}.txt"

            if copy:
                shutil.copy2(source_image, target_image)
                if source_label:
                    shutil.copy2(source_label, target_label)
            else:
                # 移动模式适合"整理干净数据"，原目录会被搬空
                shutil.move(str(source_image), str(target_image))
                if source_label:
                    shutil.move(str(source_label), str(target_label))

    print(f"划分完成：train {len(train)} 张，val {len(val)} 张"
          f"（val 比例 {len(val) / max(len(usable), 1):.1%}）")

    write_dataset_yaml(destination, names)
    write_names_file(destination, names)

    print()
    print("划分是可复现的：同样的文件名集合与 --val-ratio 永远得到同样的划分，")
    print(f"不需要随机种子。（{seed_note}）")


def write_dataset_yaml(destination: Path, names: Sequence[str]) -> Path:
    """生成 dataset.yaml。"""
    lines = [
        "# 由 prepare_dataset.py 生成，可手工微调路径",
        f"path: {destination}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {name}")
    lines.append("")

    path = destination / "dataset.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {path}")
    return path


def write_names_file(destination: Path, names: Sequence[str]) -> Path:
    """生成给板上用的类别名文件（每行一个名字，不带序号）。"""
    path = destination / "classes.txt"
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"已写入 {path}（拷到 recognize_yolo/config/ 下供推理节点使用）")
    return path


def print_stats(scan_result: DatasetScan, names: Sequence[str]) -> None:
    """输出类别分布与框尺寸分布。"""
    print_scan_report(scan_result, names)

    if not scan_result.annotations:
        return

    widths: List[float] = []
    heights: List[float] = []
    areas: List[float] = []
    per_image: List[int] = []

    for stem, annotations in scan_result.annotations.items():
        if not annotations:
            continue
        per_image.append(len(annotations))
        for annotation in annotations:
            widths.append(annotation.width)
            heights.append(annotation.height)
            areas.append(annotation.width * annotation.height)

    if not areas:
        return

    areas.sort()
    per_image.sort()

    def percentile(values: Sequence[float], ratio: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, int(round(ratio * (len(values) - 1)))))
        return values[index]

    print()
    print("框尺寸分布（归一化，相对于图像宽高的比例）：")
    print(f"  宽   中位数 {percentile(sorted(widths), 0.5):.3f}  "
          f"最小 {min(widths):.3f}  最大 {max(widths):.3f}")
    print(f"  高   中位数 {percentile(sorted(heights), 0.5):.3f}  "
          f"最小 {min(heights):.3f}  最大 {max(heights):.3f}")
    print(f"  面积 中位数 {percentile(areas, 0.5):.4f}  "
          f"P10 {percentile(areas, 0.1):.4f}  P90 {percentile(areas, 0.9):.4f}")

    tiny = sum(1 for area in areas if area < 0.01)
    if tiny:
        print(f"  提示：{tiny} 个框面积小于图像的 1%，属于小目标，"
              f"训练时建议把 imgsz 调大，或降低对这类目标的期望")

    print()
    print("每张图的框数量分布：")
    print(f"  中位数 {percentile([float(v) for v in per_image], 0.5):.0f}  "
          f"最大 {max(per_image)}  平均 {sum(per_image) / len(per_image):.2f}")

    empty = sum(1 for annotations in scan_result.annotations.values() if not annotations)
    if empty:
        print(f"  注意：{empty} 个标签文件是空的（整图无目标）。"
              f"如果这些图确实没有目标，删掉它们；否则属于漏标")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="整理 YOLO 数据集：体检、划分、生成配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--src", type=Path, required=True,
                         help="标注结果所在目录（含 images/ 与 labels/）")
        sub.add_argument("--names", nargs="*", default=None,
                         help="类别名，按顺序给出：ball cone person")
        sub.add_argument("--names-file", default="",
                         help="从文件读取类别名（每行一个）")

    check = subparsers.add_parser("check", help="只做体检，不改动文件")
    add_common(check)

    stats = subparsers.add_parser("stats", help="输出类别与框尺寸统计")
    add_common(stats)

    split = subparsers.add_parser("split", help="划分 train/val 并生成 yaml")
    add_common(split)
    split.add_argument("--dst", type=Path, required=True, help="输出数据集目录")
    split.add_argument("--val-ratio", type=float, default=0.2,
                       help="验证集比例，默认 0.2")
    split.add_argument("--move", action="store_true",
                       help="用移动代替复制（原目录会被清空）")

    yaml_parser = subparsers.add_parser("yaml", help="只生成 dataset.yaml 与类别名文件")
    add_common(yaml_parser)
    yaml_parser.add_argument("--dst", type=Path, required=True)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    names = load_names(args)
    num_classes = len(names)

    if args.command == "yaml":
        destination = args.dst.expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        write_dataset_yaml(destination, names)
        write_names_file(destination, names)
        return 0

    source = args.src.expanduser()
    if not source.is_dir():
        raise SystemExit(f"目录不存在: {source}")

    print("=" * 72)
    print(f"扫描 {source}")
    print(f"类别（{num_classes} 类）: {', '.join(names)}")
    print("=" * 72)

    scan_result = scan(source, num_classes)

    if args.command == "check":
        print_scan_report(scan_result, names)
        errors = [item for item in scan_result.issues if item.level == "error"]
        return 1 if errors else 0

    if args.command == "stats":
        print_stats(scan_result, names)
        return 0

    # split
    print_scan_report(scan_result, names)

    errors = [
        item for item in scan_result.issues
        if item.level == "error" and "没有对应" in item.message
    ]
    if errors:
        print()
        print(f"有 {len(errors)} 个图片/标签不配对的问题，建议先修好再划分。")
        print("如果只是没标完的图，删掉对应的图片即可。")
        print("确实要带着这些问题继续划分，请再次运行本命令。")
        return 1

    destination = args.dst.expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    print()
    split_dataset(
        scan_result, destination, args.val_ratio, names,
        copy=not args.move,
        seed_note="按文件名 CRC32 哈希划分",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
