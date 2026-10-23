"""启动 YOLO 识别节点，并按需自动带上识别信息转发节点。

用法：

    # NPU 路线，显式给出模型与 worker 路径
    ros2 launch recognize_yolo yolo_detect.launch.py \\
        npu_model_path:=$HOME/models/yolov5s_rt_uint8_a733.nb \\
        npu_worker_path:=$HOME/wheel_robot/install/recognize_yolo/lib/recognize_yolo/yolo_npu_worker

    # 还没准备好模型，先验证话题链路
    ros2 launch recognize_yolo yolo_detect.launch.py backend:=mock

    # 用 ONNX 兜底
    ros2 launch recognize_yolo yolo_detect.launch.py \\
        backend:=onnx onnx_model_path:=$HOME/models/yolov8n.onnx input_size:=320
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _build_nodes(context, *args, **kwargs):
    """根据 backend 决定参数文件和转发节点，写成一个 setup 函数更直观。"""
    backend = LaunchConfiguration("backend").perform(context).strip().lower()
    params_file = LaunchConfiguration("params_file").perform(context)

    if not params_file:
        # 没显式指定就按后端挑默认参数文件。
        default_name = "yolo_params_cpu.yaml" if backend == "onnx" else "yolo_params_npu.yaml"
        params_file = PathJoinSubstitution(
            [FindPackageShare("recognize_yolo"), "config", default_name]
        )

    # 以 YAML 为基线，再用命令行参数覆盖，方便临时调参不用改文件。
    overrides = {
        "backend": backend,
        "npu_model_path": LaunchConfiguration("npu_model_path").perform(context),
        "npu_worker_path": LaunchConfiguration("npu_worker_path").perform(context),
        "onnx_model_path": LaunchConfiguration("onnx_model_path").perform(context),
        "class_names_file": LaunchConfiguration("class_names_file").perform(context),
        "image_topic": LaunchConfiguration("image_topic").perform(context),
        "source": LaunchConfiguration("source").perform(context),
        "input_size": LaunchConfiguration("input_size").perform(context),
        "conf_threshold": LaunchConfiguration("conf_threshold").perform(context),
        "num_classes": LaunchConfiguration("num_classes").perform(context),
        "process_rate": LaunchConfiguration("process_rate").perform(context),
        "publish_debug_image": LaunchConfiguration("publish_debug_image").perform(context),
        # 下面这几条在 SSH 远程调试时特别有用：
        # 看画面确认安装方向后直接改，不用去编辑 yaml
        "rotation_degrees": LaunchConfiguration("rotation_degrees").perform(context),
        "mirror_mode": LaunchConfiguration("mirror_mode").perform(context),
        "image_width": LaunchConfiguration("image_width").perform(context),
        "image_height": LaunchConfiguration("image_height").perform(context),
        "camera_id": LaunchConfiguration("camera_id").perform(context),
    }

    # 只有用户真的传了值才覆盖，空字符串会覆盖掉 YAML 里的有效值。
    parameters = [params_file]
    typed_overrides = {}

    string_keys = {
        "backend", "npu_model_path", "npu_worker_path", "onnx_model_path",
        "class_names_file", "image_topic", "source", "mirror_mode",
    }
    for key, raw in overrides.items():
        if raw is None or str(raw).strip() == "":
            continue
        if key in string_keys:
            typed_overrides[key] = str(raw)
        elif key == "publish_debug_image":
            typed_overrides[key] = str(raw).strip().lower() in ("1", "true", "yes", "on")
        elif key in ("input_size", "num_classes", "rotation_degrees",
                     "image_width", "image_height", "camera_id"):
            typed_overrides[key] = int(float(raw))
        else:
            typed_overrides[key] = float(raw)

    if typed_overrides:
        parameters.append(typed_overrides)

    nodes = [
        Node(
            package="recognize_yolo",
            executable="detector_node.py",
            name="recognize_yolo",
            output="screen",
            emulate_tty=True,
            parameters=parameters,
        )
    ]

    if LaunchConfiguration("start_info_node").perform(context).strip().lower() in (
        "1", "true", "yes", "on",
    ):
        nodes.append(
            Node(
                package="recognize_yolo",
                executable="vision_info_node.py",
                name="recognize_yolo_info",
                output="screen",
                emulate_tty=True,
                parameters=[{
                    "input_topic": "/recognize_yolo/detections",
                    "publish_json": True,
                    "publish_vision_msgs": True,
                }],
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "backend", default_value="npu",
                description="推理后端：npu / onnx / mock",
            ),
            DeclareLaunchArgument(
                "params_file", default_value="",
                description="参数文件；留空则按 backend 自动选择",
            ),
            DeclareLaunchArgument(
                "npu_model_path", default_value="",
                description="NBG 模型路径，backend=npu 时必填",
            ),
            DeclareLaunchArgument(
                "npu_worker_path", default_value="",
                description="yolo_npu_worker 可执行文件路径，backend=npu 时必填",
            ),
            DeclareLaunchArgument(
                "onnx_model_path", default_value="",
                description="ONNX 模型路径，backend=onnx 时必填",
            ),
            DeclareLaunchArgument(
                "class_names_file", default_value="",
                description="类别名文件路径；自己训练的模型必须给",
            ),
            DeclareLaunchArgument(
                "source", default_value="",
                description="图像来源：camera / image_topic",
            ),
            DeclareLaunchArgument(
                "image_topic", default_value="",
                description="source=image_topic 时订阅的话题",
            ),
            DeclareLaunchArgument(
                "input_size", default_value="",
                description="模型输入边长，必须与模型一致",
            ),
            DeclareLaunchArgument(
                "num_classes", default_value="",
                description="类别数，必须与模型一致",
            ),
            DeclareLaunchArgument(
                "conf_threshold", default_value="",
                description="置信度阈值",
            ),
            DeclareLaunchArgument(
                "process_rate", default_value="",
                description="每秒推理次数，0 表示每来一帧就推理",
            ),
            DeclareLaunchArgument(
                "publish_debug_image", default_value="",
                description="是否发布带框调试图",
            ),
            DeclareLaunchArgument(
                "rotation_degrees", default_value="",
                description="摄像头旋转角度：0 / 90 / 180 / 270",
            ),
            DeclareLaunchArgument(
                "mirror_mode", default_value="",
                description="镜像方式：none / horizontal / vertical / both",
            ),
            DeclareLaunchArgument(
                "image_width", default_value="",
                description="摄像头采集宽度",
            ),
            DeclareLaunchArgument(
                "image_height", default_value="",
                description="摄像头采集高度",
            ),
            DeclareLaunchArgument(
                "camera_id", default_value="",
                description="摄像头编号",
            ),
            DeclareLaunchArgument(
                "start_info_node", default_value="false",
                description="是否同时启动识别信息转发节点（JSON / vision_msgs）",
            ),
            OpaqueFunction(function=_build_nodes),
        ]
    )
