# recognize_yolo

香橙派 4 Pro（Allwinner A733 / Vivante VIP9000 NPU）上的 YOLO 识别包，
把识别结果以 ROS 2 话题的形式输出，供跟随、避障、抓取等上层节点使用。

这是 `wheel_robot` 工作空间的一个 ROS 2 包，用 `colcon build` 编译。

> 完整的环境搭建、数据标注、模型训练、NBG 转换与部署流程，见
> **`docs/recognize_yolo_方案.docx`**。本文件只讲「怎么用这个包」。
>
> 通过 **MobaXterm / SSH 远程操作**板子（无显示器看画面、断线不中断节点、
> 远程调参）见 **`docs/recognize_yolo_MobaXterm远程连接.docx`**。

---

## 0. 这个包由两个 ROS 2 包组成

```
src/
├── recognize_yolo_msgs/     纯接口包：只定义消息，无任何代码
│   └── msg/{Detection,DetectionArray}.msg
└── recognize_yolo/          实现包：Python 节点 + 可选 C++ NPU worker
```

**为什么要拆成两个包？** 这不是为了整洁，而是 ROS 2 的硬性限制。官方文档写明：

> Calling `rosidl_generate_interfaces` and `ament_python_install_package` in the
> same CMake project does not work. It is best practice to instead separate out
> the message generation into a separate package.

原因是 `rosidl` 在生成 Python 消息代码时会自己调用一次
`ament_python_install_package()`，与包内自己的调用撞名，构建时报：

```
add_custom_target cannot create target "ament_cmake_python_copy_XXX"
because another target with the same name already exists
```

参考：[ament_cmake_python 官方文档](https://docs.ros.org/en/humble/How-To-Guides/Ament-CMake-Python-Documentation.html)、
[ros2/rosidl_python#141](https://github.com/ros2/rosidl_python/issues/141)。

所以**编译时必须带上接口包**，否则 `recognize_yolo` 会因为找不到
`recognize_yolo_msgs` 而失败：

```bash
colcon build --packages-select recognize_yolo_msgs recognize_yolo
```

---

## 1. 这个包给什么

| 输出 | 话题 | 类型 | 说明 |
| --- | --- | --- | --- |
| 识别结果 | `~/detections` | `recognize_yolo_msgs/msg/DetectionArray` | 推荐，一次拿到本帧全部目标 |
| JSON | `~/detections_json` | `std_msgs/msg/String` | 免自定义消息，脚本/上位机直接解析 |
| 调试图 | `~/debug_image` | `sensor_msgs/msg/Image` | 带框画面，可用参数关闭 |
| vision_msgs | `~/vision_detections` | `vision_msgs/msg/Detection2DArray` | 由 `vision_info_node.py` 提供 |

节点全名是 `recognize_yolo`，所以 `~/detections` 展开后是
`/recognize_yolo/detections`。

`DetectionArray` 每个目标带这些字段：`class_id`、`class_name`、`score`、
`xmin/ymin/xmax/ymax`、`center_x/center_y`、`norm_center_x/norm_center_y`、
`track_id`。控制节点一般只需要 `norm_center_x`（与分辨率无关的横向偏差）。

**本包只做感知，不发布 `/cmd_vel` 或任何控制指令。** 控制逻辑放在你自己的
跟随节点里，这样感知和控制可以分别调、分别测。

---

## 2. 编译

```bash
cd ~/wheel_robot
source /opt/ros/$ROS_DISTRO/setup.bash

# 只编译本包
colcon build --packages-select recognize_yolo_msgs recognize_yolo
source install/setup.bash
```

NPU worker 是可选的。如果编译时找不到 VIPLite 运行时，CMake 会打印一条
警告并跳过它，Python 部分照常可用（可以先用 `backend:=onnx` 或 `mock`）。
**只有 Warning 没有 Error 就是成功**，不要因为看到这条警告就以为失败了。

### 编译报错怎么查

**`rosidl_generate_interfaces() the passed dependency 'std_msgs' has not been found before using find_package()`**

CMakeLists 里缺少 `find_package(std_msgs REQUIRED)`。`DEPENDENCIES` 只是声明
依赖关系，不会替你调用 `find_package`。本包已包含这一行，若出现说明文件被改过。

**`add_custom_target cannot create target "ament_cmake_python_copy_XXX" because another target with the same name already exists`**

几乎总是**构建目录的残留状态**，不是文件问题。确认文件本身没被改坏：

```bash
cd ~/wheel_robot
grep -c '^ament_python_install_package' src/recognize_yolo/CMakeLists.txt   # 应为 1
wc -c src/recognize_yolo/CMakeLists.txt                                     # 应为 5158
```

数字对得上就清掉构建目录重建：

```bash
rm -rf build/recognize_yolo install/recognize_yolo
colcon build --packages-select recognize_yolo_msgs recognize_yolo --cmake-clean-cache
```

这个报错的**真正根因**是 ROS 2 的已知限制：同一个包里既调用
`rosidl_generate_interfaces()` 又调用 `ament_python_install_package()`。
本包已按官方建议把消息拆到 `recognize_yolo_msgs`，所以正常情况下不会再遇到。
如果你看到这个错，检查是不是有人把消息生成又加回了 `recognize_yolo/CMakeLists.txt`。

在电脑上可以提前发现这类问题：

```bash
python3 tools/check_cmake_refs.py    # 检查依赖、顺序、关键调用是否重复
```

> 不要加 `--symlink-install`。它主要给 `ament_python` 包用；本包是
> `ament_cmake`（要编译消息和 C++ worker），加了没好处，还可能让
> `AMENT_CMAKE_SYMLINK_INSTALL=TRUE` 走进 `ament_cmake_python` 的 symlink 分支。

## 3. 编译 NPU worker（可选）

要编译 NPU worker，需要先从全志 AW NPU Model Zoo 取到运行时：

```bash
# linux_aw_npu/model_zoo/v0.8.0/common/npuruntime 目录
export NPU_RUNTIME=$HOME/linux_aw_npu/model_zoo/v0.8.0/common/npuruntime

colcon build --packages-select recognize_yolo_msgs recognize_yolo \
  --cmake-args -DRECOGNIZE_YOLO_NPU_ROOT=$NPU_RUNTIME
```

`-DRECOGNIZE_YOLO_NPU=OFF` 可以显式关掉 NPU 部分。

---

## 4. 上板第一步：环境自检

```bash
python3 ~/wheel_robot/src/recognize_yolo/scripts/check_npu_env.py \
    --model ~/models/yolov5s_rt_uint8_a733.nb \
    --worker ~/wheel_robot/install/recognize_yolo/lib/recognize_yolo/yolo_npu_worker
```

它按「必须项 / 可选项」逐条检查架构、SoC、`/dev/vipcore`、`libvip-lite.so`、
`vpm_run`、Python 依赖、ROS 环境、worker 与模型文件。全绿再往下走。

---

## 5. 跑起来

### 4.1 先验证话题链路（不需要模型和 NPU）

```bash
ros2 launch recognize_yolo yolo_detect.launch.py backend:=mock
```

另开终端：

```bash
ros2 topic echo /recognize_yolo/detections --once
ros2 topic hz   /recognize_yolo/detections
```

这一步能跑通，说明消息、话题、参数都正常，后面出问题就一定在模型或 NPU 侧。

### 4.2 NPU 路线

```bash
ros2 launch recognize_yolo yolo_detect.launch.py \
    npu_model_path:=$HOME/models/yolov5s_rt_uint8_a733.nb \
    npu_worker_path:=$HOME/wheel_robot/install/recognize_yolo/lib/recognize_yolo/yolo_npu_worker \
    class_names_file:=$HOME/wheel_robot/src/recognize_yolo/config/classes_coco.txt
```

先单独确认 worker 与模型匹配，比在 ROS 里排查容易得多：

```bash
~/wheel_robot/install/recognize_yolo/lib/recognize_yolo/yolo_npu_worker \
    --model ~/models/yolov5s_rt_uint8_a733.nb --probe
```

`--probe` 会打印每个输入输出张量的形状、量化格式与 scale。
**把打印出来的输入张量边长和 `input_size` 参数对齐**，这是最常见的踩坑点。

### 4.3 ONNX 兜底

```bash
ros2 launch recognize_yolo yolo_detect.launch.py \
    backend:=onnx onnx_model_path:=$HOME/models/yolov8n.onnx \
    input_size:=320 class_names_file:=.../classes_coco.txt process_rate:=5.0
```

### 4.4 加上信息转发节点

```bash
ros2 launch recognize_yolo yolo_detect.launch.py \
    npu_model_path:=... npu_worker_path:=... start_info_node:=true
```

---

## 6. 常用调试命令

```bash
# 看带框画面（板上直接看窗口）
ros2 run rqt_image_view rqt_image_view /recognize_yolo/debug_image

# 看 JSON
ros2 topic echo /recognize_yolo/detections_json

# 核对模型输出的张量形状（解码不对时先开这个）
ros2 run recognize_yolo detector_node.py --ros-args -p log_tensor_shapes:=true

# 只保留分数高的目标，减少日志噪声
ros2 run recognize_yolo detector_node.py --ros-args -p conf_threshold:=0.5
```

---

## 7. 参数速查

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `backend` | `npu` | `npu` / `onnx` / `mock` |
| `input_size` | `640` | 模型输入边长，必须是 32 的倍数 |
| `input_color` | `rgb` | NPU 输入通道顺序；框对但类别错先查这里 |
| `num_classes` | `80` | 必须与模型一致，否则解码直接报错 |
| `class_names_file` | `""` | 类别名文件；自己训练的模型必须给 |
| `source` | `camera` | `camera` 直接开摄像头 / `image_topic` 订阅话题 |
| `conf_threshold` | `0.25` | 误检多往上调，漏检多往下调 |
| `iou_threshold` | `0.45` | 同一目标多个框调小，邻近目标被合并调大 |
| `min_box_area` | `0.0` | 过滤过小框的像素面积 |
| `process_rate` | `10.0` | 每秒推理次数，0 表示每帧都推理 |
| `publish_debug_image` | `true` | 关掉可省 CPU |
| `publish_json` | `true` | 是否发布 JSON 话题 |
| `apply_sigmoid` | `false` | 确认模型输出是 logits 时才打开 |
| `rotation_degrees` | `0` | 按实际安装方向设 0/90/180/270 |
| `mirror_mode` | `none` | `none`/`horizontal`/`vertical`/`both` |

参数文件在 `config/` 下：`yolo_params_npu.yaml`、`yolo_params_cpu.yaml`。

用 launch 启动时，参数写成 `名字:=值`（不要用 `--ros-args -p`，那是 `ros2 run` 的语法）：

```bash
# 远程调试最常用：确认画面方向、降低分辨率省带宽
ros2 launch recognize_yolo yolo_detect.launch.py \
    npu_model_path:=... npu_worker_path:=... \
    rotation_degrees:=90 image_width:=320 image_height:=240
```

---

## 8. 采集自己的数据集

本包带了两个脚本，覆盖从拍到整理的全过程，都不依赖 ROS：

```bash
# 板上：四种采集模式
python3 capture_dataset.py --mode key                                       # 终端按键，按空格存一张
python3 capture_dataset.py --mode interval --interval 0.5 --max-frames 300  # 定时抽帧
python3 capture_dataset.py --mode manual                                    # 窗口里按键存图（需图形界面）
python3 capture_dataset.py --mode video                                     # 录视频
```

**`key` 模式推荐用于 SSH 远程**：它直接读终端 stdin，不需要 X11 转发或图形界面，
在 MobaXterm 窗口里按空格就能存一张。按键说明：

| 按键 | 作用 |
| --- | --- |
| 空格 | 存一张 |
| a | 显示已保存数量与耗时 |
| q / ESC | 结束采集 |

它只存图、不显示画面。**要在存图的同时看画面**，另开一个终端跑
`ros2 run web_video_server web_video_server`，浏览器打开 `http://<板子IP>:8080/`。

采集时会写一份 `session.json`，记录旋转、镜像、间隔等参数，方便回溯。
**`--rotation` / `--mirror` 必须与推理时一致**，否则标注框在部署后会方向错误。

```bash
# 电脑上：体检 -> 划分 -> 统计
python3 prepare_dataset.py check --src ~/dataset_raw/session_01 --names ball cone
python3 prepare_dataset.py split --src ~/dataset_raw/session_01 \
    --dst ~/dataset --val-ratio 0.2 --names ball cone
python3 prepare_dataset.py stats --src ~/dataset --names ball cone
```

`check` 会查出图片/标签不配对、类别号越界、坐标越界、空标签等标注环节最常见的错误。
`split` 按文件名 CRC32 哈希划分，**同样输入永远得到同样的划分**，便于复现实验。
它会同时生成 `dataset.yaml` 和 `classes.txt`，后者拷到板上给 `class_names_file` 用。

完整操作流程（含 labelImg 快捷键与标注规范）见 `docs/recognize_yolo_方案.docx` 第三章。

---

## 9. 目录结构

```
recognize_yolo/
├── CMakeLists.txt              消息 + Python 包 + 可选 C++ worker
├── package.xml
├── msg/
│   ├── Detection.msg           单个目标
│   └── DetectionArray.msg      一帧结果
├── recognize_yolo/             Python 包（可脱离 ROS 导入）
│   ├── backend.py              后端抽象与工厂
│   ├── backend_npu.py          VIPLite worker 子进程 + 二进制协议
│   ├── backend_onnx.py         onnxruntime 兜底 / 精度基准
│   ├── backend_mock.py         无模型时验证链路
│   ├── decode.py               YOLO 输出解码（含网格偏移还原 + NMS）
│   ├── normalize.py            归一化与 letterbox 几何（不依赖 OpenCV）
│   ├── preprocess.py           letterbox 与图像预处理
│   ├── geometry.py             IoU / NMS / 坐标换算
│   ├── camera_source.py        摄像头采集与重连
│   ├── drawing.py              画框与状态条
│   └── serialize.py            识别信息 -> JSON
├── scripts/
│   ├── detector_node.py        主节点
│   ├── vision_info_node.py     JSON / vision_msgs 转发
│   ├── check_npu_env.py        NPU 环境自检
│   ├── capture_dataset.py      板端数据采集（interval / manual / video）
│   └── prepare_dataset.py      电脑端数据集体检、划分、生成 yaml
├── src/                        C++ NPU worker
│   ├── vip_runner.{h,cpp}      VIPLite 生命周期封装
│   ├── postprocess.{h,cpp}     量化还原
│   └── yolo_npu_worker.cpp     stdin/stdout 协议服务
├── launch/yolo_detect.launch.py
├── config/                     参数文件与类别名
├── test/                       不依赖 ROS 的单元测试
└── docs/                       方案文档（Word）
```

---

## 10. 设计上的几个取舍

**为什么解码在 Python、不在 C++。**
C++ 只做 Python 做不到的事：驱动 NPU、把张量搬出来。解码、NMS、阈值筛选
全部留在 numpy 里，改参数不用重新编译，出错也更容易复现。

**为什么 worker 是独立进程而不是 ctypes。**
VIPLite 的 `vip_init` / `vip_destroy` 必须严格配对，Python 侧异常退出会留下
没释放的网络资源，往往要重启系统。放进子进程，进程死了内核自动回收。顺带
好处是前向不占 GIL，不会拖慢 ROS 回调。

**为什么后端可换。**
NPU 环境没通不该阻塞上层开发。`mock` 用来验证话题链路，`onnx` 既是兜底也是
精度对照 —— 同一张图分别过 NPU 和 ONNX，就能判断是量化掉精度还是后处理写错了。

**为什么 `vip_flush_buffer` 在推理前后各调一次。**
A733 的 NPU 与 CPU 不共享缓存。少了前一次会读到上一帧的旧数据（表现为框抖动
或恒定不变），少了后一次会读到陈旧的输出。这是「模型跑通但结果不对」的常见原因。

---

## 11. 本地（不上板）验证

```bash
python3 src/recognize_yolo/.verify_recognize_yolo.py
```

做语法检查、YAML 校验和单元测试。解码与 NMS 的逻辑不依赖 ROS 和 NPU，
在电脑上就能测，推到板子之前先跑一遍能省很多时间。

---

## 12. 已知限制

* `src/vip_runner.cpp` 按全志 SDK 文档的 VIPLite API 编写，但**必须用你自己
  SDK 里的 `vip_lite.h` 编译核对**：不同 SDK 版本里 `vip_init` 的签名、
  网络类型枚举名、部分 `VIP_BUFFER_PROP_*` 宏存在差异。首次编译若报错，
  对着 `vip_lite.h` 改这几处即可（代码里都有注释标出）。
* 未实现目标跟踪，`track_id` 恒为 `-1`。
* 未做多模型并行、多摄像头输入。
