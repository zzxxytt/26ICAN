"""不依赖第三方库的校验脚本。

为什么不用 pytest / pyyaml：

板上和电脑上都可能没有这些包，而「验证」这件事本身不应该需要先联网装东西。
这里用标准库实现一个够用的 YAML 子集解析器和一个极简测试运行器，
覆盖本项目实际用到的写法（嵌套映射、标量、行内列表、注释、多行文本）。

用法：
    python .verify_recognize_yolo.py
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_DIRS = [
    PACKAGE_ROOT / "recognize_yolo",
    PACKAGE_ROOT / "scripts",
    PACKAGE_ROOT / "launch",
    PACKAGE_ROOT / "test",
]


# ---------------------------------------------------------------------------
# 极简 YAML 解析（只支持本项目用到的子集）
# ---------------------------------------------------------------------------
class YamlError(ValueError):
    pass


def _parse_scalar(text: str) -> Any:
    """把 YAML 标量转成 Python 值。"""
    value = text.strip()
    if value == "":
        return None

    # 去掉行尾注释：只在 # 前有空白、且不在引号内时才当注释。
    if not value.startswith(("'", '"')):
        hash_index = value.find(" #")
        if hash_index >= 0:
            value = value[:hash_index].strip()
        if value == "":
            return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]

    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "~"):
        return None

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(",")]

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """解析缩进式的 YAML 映射。不支持锚点、多文档、块标量。"""
    root: Dict[str, Any] = {}
    # 栈里存 (缩进, 容器)
    stack: List[Tuple[int, Any]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        if stripped.startswith("- "):
            raise YamlError(f"第 {line_number} 行：本解析器不支持块序列")

        if ":" not in stripped:
            raise YamlError(f"第 {line_number} 行：`{stripped}` 不是 key: value 形式")

        key, _, remainder = stripped.partition(":")
        key = key.strip()
        if not key:
            raise YamlError(f"第 {line_number} 行：键名为空")

        # 弹栈到当前缩进对应的父容器
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]

        if remainder.strip() == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(remainder)

    return root


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------
def check_undefined_names() -> bool:
    """静态检查：文件里用到的名字有没有导入/定义。

    为什么需要这一步：Python 的 compile() 只检查语法，不解析名字。
    「@dataclass 装饰器但忘了 from dataclasses import dataclass」这种错误
    语法完全合法，compile() 通过，要到真正 import 时才炸
    （NameError: name 'dataclass' is not defined）。
    本项目的模块大多依赖 cv2 / rclpy，无法在当前环境真正 import，
    所以用 AST 收集顶层可见名字，再逐个核对被引用到的名字。
    """
    import ast

    print()
    print("=" * 72)
    print("0) 名字定义检查（抓「用了未导入的名字」）")
    print("=" * 72)

    # 内置名与常见注入名
    builtins_set = set(dir(__builtins__)) if isinstance(__builtins__, dict) is False else set(__builtins__)
    try:
        import builtins as _builtins
        builtins_set = set(dir(_builtins))
    except Exception:
        pass
    builtins_set |= {"__name__", "__file__", "__doc__", "__package__", "__spec__",
                     "__loader__", "__builtins__", "self", "cls"}

    problems: List[str] = []
    checked = 0

    for directory in SOURCE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            checked += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            except SyntaxError:
                continue  # 语法检查那一步会报

            defined: set = set(builtins_set)

            # 1) 收集模块级定义的名字
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        defined.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    defined.add(node.id)
                elif isinstance(node, ast.arg):
                    defined.add(node.arg)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    defined.add(node.name)
                elif isinstance(node, (ast.comprehension,)):
                    for sub in ast.walk(node.target):
                        if isinstance(sub, ast.Name):
                            defined.add(sub.id)
                elif isinstance(node, ast.Global):
                    defined.update(node.names)
                elif isinstance(node, ast.alias):
                    defined.add((node.asname or node.name).split(".")[0])

            # 2) 收集被引用但没定义的名字（只看加载，不看属性名）
            used: dict = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used.setdefault(node.id, node.lineno)

            undefined = {
                name: line for name, line in used.items() if name not in defined
            }
            if undefined:
                for name, line in sorted(undefined.items(), key=lambda item: item[1]):
                    problems.append(
                        f"{path.relative_to(PACKAGE_ROOT)}:{line}: 使用了未定义/未导入的名字 '{name}'"
                    )
                    print(f"  FAIL {path.relative_to(PACKAGE_ROOT)}:{line} "
                          f"-> '{name}' 未定义或未导入")

    print(f"  共检查 {checked} 个文件")
    if problems:
        return False
    print("  所有引用到的名字都有定义或导入")
    return True


def check_syntax() -> bool:
    print("=" * 72)
    print("1) Python 语法检查")
    print("=" * 72)

    failures: List[str] = []
    checked = 0

    for directory in SOURCE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            checked += 1
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except SyntaxError as error:
                failures.append(f"{path.relative_to(PACKAGE_ROOT)}:{error.lineno}: {error.msg}")
                print(f"  FAIL {path.relative_to(PACKAGE_ROOT)} -> 第 {error.lineno} 行 {error.msg}")

    print(f"  共检查 {checked} 个文件")
    if failures:
        return False
    print("  全部通过")
    return True


def check_yaml() -> Tuple[bool, Dict[str, set]]:
    print()
    print("=" * 72)
    print("2) 参数文件检查（内置 YAML 子集解析）")
    print("=" * 72)

    ok = True
    all_keys: set = set()

    param_files = sorted((PACKAGE_ROOT / "config").glob("*.yaml"))
    if not param_files:
        print("  FAIL 没有找到任何参数文件")
        return False, all_keys

    for path in param_files:
        try:
            data = parse_simple_yaml(path.read_text(encoding="utf-8"))
        except YamlError as error:
            print(f"  FAIL {path.name}: {error}")
            ok = False
            continue

        if "recognize_yolo" not in data:
            print(f"  FAIL {path.name}: 缺少 recognize_yolo 顶层键")
            ok = False
            continue

        node = data["recognize_yolo"]
        if not isinstance(node, dict) or "ros__parameters" not in node:
            print(f"  FAIL {path.name}: 缺少 ros__parameters")
            ok = False
            continue

        params = node["ros__parameters"]
        if not isinstance(params, dict) or not params:
            print(f"  FAIL {path.name}: ros__parameters 为空")
            ok = False
            continue

        all_keys |= set(params.keys())

        # 关键字段必须存在且类型正确
        problems = []
        if params.get("backend") not in ("npu", "onnx", "mock"):
            problems.append(f"backend={params.get('backend')!r} 不是 npu/onnx/mock")
        input_size = params.get("input_size")
        if not isinstance(input_size, int) or input_size <= 0 or input_size % 32 != 0:
            problems.append(f"input_size={input_size!r} 不是 32 的正整数倍")
        num_classes = params.get("num_classes")
        if not isinstance(num_classes, int) or num_classes <= 0:
            problems.append(f"num_classes={num_classes!r} 非法")
        conf = params.get("conf_threshold")
        if not isinstance(conf, float) or not 0.0 < conf <= 1.0:
            problems.append(f"conf_threshold={conf!r} 应在 (0,1] 区间")
        iou = params.get("iou_threshold")
        if not isinstance(iou, float) or not 0.0 < iou <= 1.0:
            problems.append(f"iou_threshold={iou!r} 应在 (0,1] 区间")

        if problems:
            print(f"  FAIL {path.name}: " + "; ".join(problems))
            ok = False
        else:
            print(f"  OK   {path.name}  ({len(params)} 个参数, "
                  f"backend={params.get('backend')}, input_size={input_size})")

    return ok, all_keys


def check_launch(yaml_keys: set) -> bool:
    print()
    print("=" * 72)
    print("3) launch 文件检查")
    print("=" * 72)

    launch_files = sorted((PACKAGE_ROOT / "launch").glob("*.launch.py"))
    if not launch_files:
        print("  FAIL 没有 launch 文件")
        return False

    ok = True
    for path in launch_files:
        text = path.read_text(encoding="utf-8")

        # 1) 语法
        try:
            compile(text, str(path), "exec")
        except SyntaxError as error:
            print(f"  FAIL {path.name}: 第 {error.lineno} 行 {error.msg}")
            ok = False
            continue

        # 2) generate_launch_description 必须存在
        if "def generate_launch_description" not in text:
            print(f"  FAIL {path.name}: 缺少 generate_launch_description")
            ok = False
            continue

        # 3) launch 里引用的参数名都应当在 yaml 里定义
        referenced = {
            name for name in re.findall(r'"([a-z][a-z0-9_]{2,})"', text) if name in yaml_keys
        }
        unknown = {
            name
            for name in re.findall(r'LaunchConfiguration\("([a-z0-9_]+)"\)', text)
            if name not in yaml_keys and name not in {"params_file", "start_info_node", "backend"}
        }

        if unknown:
            print(f"  FAIL {path.name}: 引用了 yaml 中不存在的参数 {sorted(unknown)}")
            ok = False
        else:
            print(f"  OK   {path.name}  (引用到 yaml 参数 {len(referenced)} 个)")

    print(f"  yaml 中定义的参数共 {len(yaml_keys)} 个")
    return ok


def check_class_names() -> bool:
    print()
    print("=" * 72)
    print("4) 类别名文件检查")
    print("=" * 72)

    path = PACKAGE_ROOT / "config" / "classes_coco.txt"
    if not path.is_file():
        print("  FAIL 缺少 config/classes_coco.txt")
        return False

    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if len(names) != 80:
        print(f"  FAIL COCO 类别数应为 80，实际 {len(names)}")
        return False
    if names[0] != "person" or names[-1] != "toothbrush":
        print(f"  FAIL 类别顺序异常: 首={names[0]!r} 末={names[-1]!r}")
        return False

    print(f"  OK   {path.name}: {len(names)} 个类别，顺序与 COCO 一致")
    return True


def check_messages() -> bool:
    print()
    print("=" * 72)
    print("5) 自定义消息定义检查")
    print("=" * 72)

    # 消息定义在独立的接口包里。
    # ROS 2 不允许同一个 CMake 工程既生成接口又调用
    # ament_python_install_package()，所以消息必须单独成包。
    messages_root = PACKAGE_ROOT.parent / "recognize_yolo_msgs"
    print(f"  消息包: {messages_root}")

    if not messages_root.is_dir():
        print("  FAIL 找不到同级目录 recognize_yolo_msgs")
        print("       消息定义必须放在独立的接口包里（ROS 2 的硬性限制）")
        return False

    detection = messages_root / "msg" / "Detection.msg"
    array = messages_root / "msg" / "DetectionArray.msg"

    ok = True
    for path in (detection, array):
        if not path.is_file():
            print(f"  FAIL 缺少 {path.name}")
            ok = False
            continue
        # 逐行检查字段语法：类型 名字，或注释/空行
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_/]*(<[^>]+>)?(\[[^\]]*\])?\s+[A-Za-z_][A-Za-z0-9_]*$", stripped):
                print(f"  FAIL {path.name} 第 {line_number} 行字段语法可疑: {stripped}")
                ok = False

    if ok:
        detection_fields = [
            line.split()[-1]
            for line in detection.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for required in ("center_x", "center_y", "norm_center_x", "norm_center_y",
                         "class_id", "class_name", "score"):
            if required not in detection_fields:
                print(f"  FAIL Detection.msg 缺少字段 {required}")
                ok = False
        if ok:
            print(f"  OK   Detection.msg 有 {len(detection_fields)} 个字段，"
                  f"控制节点需要的字段齐全")

    return ok


# ---------------------------------------------------------------------------
# 极简测试运行器
# ---------------------------------------------------------------------------
def run_test_files() -> bool:
    print()
    print("=" * 72)
    print("6) 单元测试（内置运行器，断言式测试函数）")
    print("=" * 72)

    test_files = sorted((PACKAGE_ROOT / "test").glob("test_*.py"))
    if not test_files:
        print("  FAIL 没有找到测试文件")
        return False

    # 让测试文件能 import recognize_yolo
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

    passed = 0
    failures: List[str] = []

    for path in test_files:
        module_name = f"_verify_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            print(f"  FAIL 无法加载 {path.name}")
            failures.append(f"{path.name}: 无法加载")
            continue

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            print(f"  FAIL {path.name} 导入失败: {error}")
            failures.append(f"{path.name}: 导入失败 {error}")
            continue

        test_functions: List[Callable[[], None]] = []
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            attribute = getattr(module, name)
            if callable(attribute):
                test_functions.append(attribute)

        if not test_functions:
            print(f"  WARN {path.name} 里没有 test_* 函数")
            continue

        for function in test_functions:
            try:
                function()
            except Exception as error:
                print(f"  FAIL {path.name}::{function.__name__}")
                print(f"       {type(error).__name__}: {error}")
                failures.append(f"{path.name}::{function.__name__}")
                traceback.print_exc(limit=2)
            else:
                passed += 1

        print(f"  OK   {path.name}  ({len(test_functions)} 个测试)")

    print(f"  通过 {passed} 个测试，失败 {len(failures)} 个")
    return not failures


def main() -> int:
    # Windows 控制台默认用 GBK，强制 UTF-8 以免中文输出变成乱码。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    names_ok = check_undefined_names()
    syntax_ok = check_syntax()
    yaml_ok, yaml_keys = check_yaml()
    launch_ok = check_launch(yaml_keys)
    classes_ok = check_class_names()
    messages_ok = check_messages()
    tests_ok = run_test_files()

    results = {
        "名字定义": names_ok,
        "语法检查": syntax_ok,
        "参数文件": yaml_ok,
        "launch 文件": launch_ok,
        "类别名文件": classes_ok,
        "消息定义": messages_ok,
        "单元测试": tests_ok,
    }

    print()
    print("=" * 72)
    print("汇总")
    print("=" * 72)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
