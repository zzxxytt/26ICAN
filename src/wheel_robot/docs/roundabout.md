# 下方切线进入、逆时针一圈、沿切线退出

圆环代码只有一个新增程序 `scripts/roundabout.py`，包含视觉检测、轨迹候选、状态机、只读姿态接口、屏幕显示和离线查看入口。基础巡线的 `HsvStripLineDetector` 未改动，PD 公式、正常巡线速度与原参数保持原样。圆环接管时仅提供新的目标误差、速度上限以及禁止沿用旧指令的标志。

## 替换与构建

交付 ZIP 内是完整 `src/wheel_robot/`，包含原有的其他视觉程序，按相同目录覆盖到 ROS 2 工作空间即可。不要只复制 `icn.py`：它依赖同目录的巡线、斑马线、台阶和圆环文件。

在机器人 ROS 2 工作空间根目录运行：

```bash
colcon build --packages-select wheel_robot --symlink-install
source install/setup.bash
```

此包仍不启动底盘，底盘按原有方式启动。摄像头只由一个节点打开，`/cmd_vel` 继续只有原巡线/任务节点一个发布出口。

本次没有修改底盘、电机、姿态控制、通信协议、跳跃动作。`CMakeLists.txt` 新增圆环安装项，并补装原来被 `icn.py` 导入却漏装的 `step_detector.py`；`package.xml` 加入可选里程计接口需要的 `nav_msgs` 依赖。

## 三种模式

| 模式 | 行为 |
| --- | --- |
| `off` | 不加载圆环模块，原巡线及任务流程工作。 |
| `observe` | 运行圆环观察状态机并显示候选，不改变原检测结果和控制输出。默认 ICN 启动模式。 |
| `control` | 入口连续确认、姿态新鲜且平稳后，圆环子状态机才接管。 |

```bash
# 原巡线照旧使用，默认 roundabout_mode=off
ros2 launch wheel_robot line_follow.launch.py enable_motion:=true

# 架空观察，所有实际速度为零
ros2 launch wheel_robot icn.launch.py enable_motion:=false roundabout_mode:=observe

# 圆环低速实测；底盘应已启动，已核对图像与转角方向
ros2 launch wheel_robot icn.launch.py enable_motion:=true roundabout_mode:=control

# ICN 完全关闭圆环，保留原台阶/斑马线/巡线流程
ros2 launch wheel_robot icn.launch.py enable_motion:=true roundabout_mode:=off

# 运行中交回基础巡线（节点仍有正常相机帧时于下一帧生效）
ros2 param set /line_follower roundabout_mode off

# 查看结构化调试状态
ros2 topic echo /roundabout/status
```

仅 `roundabout_mode` 支持直接在运行中切换；其他 `roundabout.*` 参数改 YAML 后重启节点。观察模式中的圆环状态可以变化，但 `control=BASE LINE` 表示它没有控制机器人。架空调试不应把命令转角当作实际绕行进度，需真实转动机身或使用记录的姿态。

## 状态与判定

| 状态 | 工作与转移条件 |
| --- | --- |
| `SEARCH` | 基础巡线。寻找近端相接、远端分离的虚线圆弧和切线。 |
| `CONFIRM` | 连续多帧检测同一入口；缺姿态、倾斜或候选跳变时不累计有效确认。 |
| `ENTER` | 锁定向图像左侧弯的圆弧，禁止选取直达 6 的切线。记录实际入环朝向。 |
| `FOLLOW` | 实测正向累计转角超过 35°，沿圆弧候选跟踪；未达到出环门槛不选切线。 |
| `EXIT_READY` | 累计转角至少 325°；还需接近原进入朝向、再次看到接合区和切线，并连续确认。 |
| `EXIT` | 锁定出口切线；连续看到单一路线且原巡线有效，才记一次完成。 |
| `COOLDOWN` | 已完成或已中止，交还基础巡线。需时间与连续离开接合区证据共同满足才能重新找环。 |
| `RECOVER` | 圆环内丢线、反向绕行、姿态失效或超时后停车。稳定单一路线和原巡线连续有效才交还；不记成功。 |

`completed` 只在 `EXIT` 确认后增加。时间上限仅用于中止，不用于宣布绕完。到达第二圈入口时重新记录朝向，上一圈累计角度不会带入下一圈。圆环接管期间抑制台阶与斑马线的误触发；已有台阶、斑马线、结束状态仍保有控制优先权。

整个圆环在俯视地图中先向右上走，但机器人沿底部切线面向右侧时，这是车体的左转，因此非镜像前视图的 `image_turn_sign=-1`。不要把地图向右上误设置成相机向右转。

## 检测与防御

- 独立提取圆环 ROI，用灰度暗线与局部自适应阈值提取候选。不对整幅虚线做大范围闭运算。
- 多路径搜索跨过有限虚线间隙，保留圆弧和切线各自的假设，不平均两条路线的黑点。
- 按线宽、近端支持、可见跨度、不同虚线段数量、拟合残差、候选分离和前帧轨迹一致性筛选。入口需要远端真实可见的支持，不凭短线大范围外推判断岔路。
- 环内正常跟踪局部二次曲线，投影回原相机坐标后复用原 PD 转向控制。速度仅在接管时限制，原巡线正常速度不变。
- 短空白最多预测 0.20 秒，并受 `predict_distance / predict_speed` 的更短时间限制；默认预测速度 0.05 m/s。这里是短时几何延续，没有声称进行完整视觉里程计或精确运动补偿。
- 环内预测超过限额立即零速，不进入原巡线的 `lost_hold_frames` 指令保持分支。减速上限也作用于加速度限幅后的实际指令，防止上一帧高速残留。
- 相机不可用、重复帧、过期帧不推进状态；处理耗时导致图像过期也停车。传感器重复时间戳、过期、异常四元数、异常角跳变不计进度。
- 圆环尚未接管时，模块加载/检测失败保留原巡线；接管后模块异常锁定停车，屏幕给出原因。若检测仍正常，只是暂时丢线，可自动稳定恢复；若模块本身异常锁存，切 `off` 清除，再决定是否重新 `control`。
- `off` 模式无需圆环文件或新增传感器消息依赖才能单独运行原 `line_follower.py`。如安装整个 ROS 包，依赖仍需按 package.xml 安装。

## 姿态排列必须核对

本仓库 `wl_base_node.cpp` 把原始四元数 `q0(w),q1(x),q2(y),q3(z)` 写入 ROS 的 `orientation.x,y,z,w`。因此默认配置为：

```yaml
roundabout.heading_source: "imu"
roundabout.imu_layout: "wxyz_in_xyzw"
roundabout.yaw_sign: 1.0
```

如果实际底盘已改为标准 ROS 四元数，配置 `roundabout.imu_layout: "xyzw"`。也可改 `roundabout.heading_source: "odom"`，从 `/odom` 标准姿态累计转角；该源通常没有真实俯仰信息，不能依靠它做落地倾斜检测。两种来源运行中不自动切换，防止零点不同造成假圈数。

观察模式中，手动逆时针转动车体，确认进入环岛后 `yaw` 持续增加、跨 ±180° 不跳回零。姿态不可用时显示 `wait_heading_or_level`，圆环不接管，基础巡线仍照常工作，因此可能直接经过切线，不会伪报完成圆环。

## 屏幕读法与调参顺序

右侧独立调试区不遮挡原巡线、台阶与斑马线标注，显示模式、子状态、控制归属、候选数量、入口确认次数、累计转角、传感器年龄、完成/中止次数、出口确认次数、丢线预测与处理耗时。下方小图显示阈值图和独立候选曲线；绿色大图曲线是当前圆环选中的轨迹。旁边数值 `编号:置信度/虚线段数` 用于区分候选。

1. 先 `observe`、`enable_motion:=false` 看原巡线是否仍正常显示。
2. 用实拍入口画面核对同时出现圆弧与切线，`paths>=2` 且 `junction=1`。普通直线、单个弯道、台阶和斑马线不应稳定触发 `junction`。
3. 入口看不全先调整相机俯角或 `roi_top`，再调 `black_max`、`min_width/max_width`。`min_width/max_width` 是处理图宽度比例，`max_gap/min_span` 是处理图高度比例。
4. 让 `CONFIRM` 稳定，不要用降低所有门槛解决问题。需多段虚线；若前视相机只能看到一个黑块，此方法没有足够选路信息，应先改善视野。
5. 确认姿态排列与逆时针正方向，再打开 `control`。默认入/出环 0.08 m/s、环内 0.12 m/s，只影响环岛。
6. 调 `lookahead` 和 `max_track_jump` 处理跟踪：前者影响预瞄位置，后者限制帧间换线。先确保选支正确再考虑转向增益。
7. 出环只微调 `exit_min_degrees`、`exit_heading_tolerance` 和确认帧数，超时不是完成条件。若到 395°仍无出口证据，会中止，查看 `reason`。

`perspective=false` 时只归一化 ROI，不假定已完成地面标定。开启时必须按实际相机标定 `top_left/top_right/bottom_left/bottom_right` 梯形；这些比例相对于圆环 ROI。落地后的俯仰会影响固定变换，IMU 模式通过倾斜门槛抑制接管，并没有偷偷改动底盘姿态。

## 离线图片与录像

同一个程序可以脱离 ROS 使用，不创建任何速度发布者：

```bash
python3 src/wheel_robot/scripts/roundabout.py --source entrance.jpg --config src/wheel_robot/config/icn.yaml
python3 src/wheel_robot/scripts/roundabout.py --source lap.mp4 --config src/wheel_robot/config/icn.yaml
python3 src/wheel_robot/scripts/roundabout.py --source entrance.jpg --no-display --snapshot debug.png
```

可选 `--yaw-csv measured_yaw.csv`，表头为 `time,yaw_degrees`，time 是相对于录像起点的秒数，严格递增，角度必须是记录的真实朝向。无姿态数据时只确认几何候选，不伪造绕行角。录像中的完成/恢复仅用几何单线证据演示；机器人上额外要求原巡线检测有效。标准机器人环境安装桌面 OpenCV 才能开窗；headless OpenCV 只能 `--no-display`。

## 验证范围

本次使用 OpenCV 4.13、NumPy 2.5 离线检查。ROS 导入在 Windows 测试中使用仅导入的替身，未模拟或宣称通过真实 ROS 通信及实车测试。

- 新增 26 项视觉、状态机、接口与控制保护测试全部通过，放在原有 `test/test_line_detector.py` 的 `Roundabout*Test` 中，没有再新增测试程序文件。包含从合成图像、模拟传感器姿态到入环/完整绕行/出环的整个软件流程。
- 对修改前后 `HsvStripLineDetector` 做 AST 比较，完全一致。150 帧合成场景的检测有效性、误差、置信度、原因、调试图及基础控制输出逐项一致。
- 40 帧加入噪声的合成交叠画面全部检测到独立圆弧与切线。开发机耗时约 19 ms 中位数，不能代表机器人板卡速度。
- 原有 13 项单黑线测试中有 5 项失败；修改前原代码复跑出现完全相同失败及数值。本次未调整这些旧用例或原巡线算法来掩盖失败：涉及左右弯误差阈值、反光缺口跨段选择、边缘恢复名称及短线有效性。

ROS 环境仅运行本次新增测试：

```bash
cd src/wheel_robot/test
python3 -m unittest test_line_detector.RoundaboutVisionTest test_line_detector.RoundaboutStateTest test_line_detector.RoundaboutIntegrationTest
```

上述检查证明模块旁路和所测保护分支的行为，不能保证未实拍标定的赛道一次过环。先记录实际相机画面与姿态，用观察模式检查，再在低速下验证。

## 版本管理与回退

修改前基线提交：`5cf32cb`；开发分支：`feature/vision-roundabout`。PPT、学习资料、构建产物、临时图片、测试依赖及交付 ZIP 均不进入 Git。

最快恢复原巡线行为是将 `roundabout_mode` 设为 `off`。需要恢复整个视觉包时，先保存后来自己的修改，再在仓库根目录将 `src/wheel_robot` 从基线提交恢复；这会覆盖该目录后续修改，因此不在运行流程里自动执行。
