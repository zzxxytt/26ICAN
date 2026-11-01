# 26ICAN 轮足机器人智能驾驶挑战赛

本项目是比赛用轮足机器人的 ROS 2 工作空间。车载计算机从相机画面提取赛道信息，执行巡线、斑马线、台阶及环岛任务，通过串口桥接节点与 ESP32 底盘通信。

## 代码与资料

| 路径 | 内容 |
| --- | --- |
| [wheel_robot](src/wheel_robot/) | 比赛主程序。入口是 `scripts/icn.py`，巡线参数在 `config/line_follower.yaml`，赛道元素参数在 `config/icn.yaml`。 |
| [wheeled_legged_pkg](src/wheeled_legged_pkg/) | 底盘串口通信、运动指令和传感器数据处理。 |
| [wheel_legged_urdf_pkg](src/wheel_legged_urdf_pkg/) | 机器人 URDF、零件网格及 RViz 显示配置。 |
| [recognize_yolo](src/recognize_yolo/README.md) | YOLO 目标检测、数据采集及香橙派推理部署说明。 |
| [recognize_yolo_msgs](src/recognize_yolo_msgs/) | 检测结果使用的 ROS 2 消息定义。 |
| [simple_follower_ros2](src/simple_follower_ros2/) | 激光、视觉和 ArUco 跟随示例。 |
| [比赛文档](docs/) | 赛前培训课件和技术报告模板。 |
| [轮足机器人学习资料](轮足机器人（2026）学习资料/) | 使用说明、实验教程、硬件模型及出厂源码。超过 100 MiB 的视频和镜像保留在本地。 |

## 构建并启动比赛节点

在已安装 ROS 2、colcon 和 rosdep 的 Linux 车载环境中，进入仓库根目录并加载对应 ROS 2 发行版的环境。以下命令构建比赛节点及其工作空间依赖。

```bash
rosdep install --from-paths src/wheel_robot src/wheeled_legged_pkg --ignore-src -r -y
colcon build --symlink-install --packages-up-to wheel_robot
source install/setup.bash
```

连接相机后，先以关闭运动输出的模式检查识别画面。环岛观测参数会记录检测结果，便于调试。

```bash
ros2 launch wheel_robot icn.launch.py enable_motion:=false roundabout_mode:=observe
```

实车运行时，先启动底盘节点，将串口参数改为 ESP32 的实际设备路径。

```bash
ros2 launch wheel_robot base.launch.py serial_port:=/dev/ttyUSB0
```

随后关闭观测节点，在另一个已加载工作空间环境的终端运行 `ros2 launch wheel_robot icn.launch.py enable_motion:=true`。环岛策略默认读取 `config/icn.yaml`，调试步骤见[环岛说明](src/wheel_robot/docs/roundabout.md)。
