# 下方切线进入、逆时针一圈、沿切线退出

## 圆环右侧面板：多条候选与入口未确认

`OBSERVE SEARCH / control=BASE LINE / paths=3 junction=0 hits=0/4` 表示发现三个拟合候选，但还没确认圆环入口；观察模式始终沿用基础巡线。三个候选可能共享虚线块，不等于三条真实分支。混合连接、拟合误差和超出采样范围的外推都可能产生看似反向弯曲的线。单张屏幕照片无法区分这些原因，不能以 `paths>=2` 直接触发圆环。

本次仅增加诊断、修正绘图范围，不改变候选生成、入口判据、模式默认值或控制输出。拟合线只画实际采样的前后范围，并叠加采样点；绿色 A 是当前圆弧候选，青色 T 是切线候选，灰色是其余候选（可能不合格，也可能合格但未选中），颜色本身不代表已经接管。

面板新增 `entry=...`、`A/T` 数量和 `near_pairs`：

| 提示 | 含义 |
| --- | --- |
| NO_PATHS | 没有合格路径 |
| NO_ARC | 没有通过圆弧条件的路径 |
| NO_TANGENT | 没有通过直切线条件的路径 |
| TANGENT_SUPPORT | 切线虚线块不足或远端支持不足 |
| NEAR_GAP | 圆弧与切线在近端不够靠拢 |
| FAR_SEPARATION | 圆弧与切线在远端分离不够或方向不符 |
| OK | 本帧入口几何通过，仍需连续确认、姿态和阶段许可 |

每条路径的小字 `P编号/角色 d=块数 A:... T:...` 列出失败项：`dash` 是圆弧不足 3 块；`reach` 是未覆盖归一化前向位置 0.8；`bend` 是曲率未满足对应条件；`dir` 是圆弧偏移方向或幅度不足；`heading` 是直切线斜率过大。`OK` 表示通过该类条件。详细数值（曲率、斜率、置信度、采样范围）随 `/roundabout/status` 的 `entry` 字段输出，可用 `ros2 topic echo /roundabout/status` 查看。

`yaw=0` 在 SEARCH 中不代表 IMU 没工作：该值是入环锁定后的相对累计转角。`age=0.07s` 低于默认 0.40s 超时，只能说明最近接收了姿态；它不能验证四元数布局和转向符号。`41.9ms` 是圆环视觉本帧处理耗时，不是整套程序帧率。`complete/abort=0` 与 SEARCH 一致。

覆盖后仍用 `bash src/wheel_robot/scripts/run_icn_on_screen.sh` 启动，默认保持 observe。正式接管需显式传 `roundabout_mode:=control`，但切模式不会让 `junction=0` 自动通过。当前还需要现场新的 entry/路径诊断或原始视频确认具体失败项。新增四项入口诊断检查，连同现场巡线与圆环既有检查共 40 项通过；尚未确认现场混合候选的确切成因。

## 2026-10-25 现场截图：冷启动线宽窗口校准

现场截图中黑线进入 ROI，但 N/M/F 仅选中 2/5/12 行，段内最宽暗段约 46/42/35px。截图中的 `max` 是观测最宽值，不是允许上限；`err=0`、`conf=0` 在此表示没有有效融合结果，不表示已经对准线。截图没有新版 `keep<=` 诊断字样，部署时还需核对实际加载的源码和参数。

旧共享配置 `max_line_width_ratio=0.060`、中远缩放 `0.72/0.52` 在 640×480 下允许约 36/26/19px，过窄的窗口会把明显黑线排除。前一修复恢复的近段连续跟踪依赖上一帧中心，无法解决第一帧超宽，也无法恢复被拒绝的中远段。因此这次校准共享 `line_follower.yaml`：最大宽度比 `0.090`，中远缩放 `0.90/0.78`，允许上限约 55/49/43px；期望宽度比 `0.065`、软评分容差 `0.035`。保留黑色阈值、覆盖率/置信度门槛、PD、速度、丢线停车和阶段控制。

直接使用现有屏幕入口，不需要先启动单独巡线。覆盖完整包后在机器人工作空间执行：

```bash
cd ~/wheel_robot
colcon build --packages-select wheel_robot --symlink-install
bash src/wheel_robot/scripts/run_icn_on_screen.sh
```

脚本自行 source 当前工作空间环境，并打印 ICN 程序、共用巡线程序及参数文件的真实路径。节点启动日志应有 `width=0.020..0.090 scale_M/F=0.90/0.78`；640px 图像 HUD 各段应显示 `keep<=55/49/43`。如果仍显示旧值，先排查覆盖目录或安装副本。脚本默认的圆环模式仍为 observe，这次只处理基础巡线；下文已列出的阶段风险仍未改动。

验证采用按截图线宽构造的图像，不是带 HUD 截图回灌，更不是实车视频：旧窗口冷启动失败，新窗口首帧有效且不置 corner；左右误差符号、20px 窄直线、丢线、细缝、横条和宽边缘检查通过。现场回归及圆环测试共 36 项通过。现场实际曝光、反光、俯仰和弯道仍需复测，截图不足以证明整圈可靠。

## Linux 启动报 `python3\r` 的处理

`/usr/bin/env: ‘python3\r’: No such file or directory` 表示脚本第一行用了 CRLF 换行，Linux 将回车当成解释器名称的一部分。程序尚未执行，不属于摄像头或巡线检测失败；colcon 构建成功也不会自动修正换行。

仓库已通过根目录 `.gitattributes` 将 wheel_robot 的 Python、Shell 和配置等文本固定为 LF。替换包也按 LF 导出并逐文件检查。如果板子上已经有 CRLF 文件，可在工作空间根目录执行以下修复；只替换换行字节，保留现有参数和程序内容，同时处理可能残留的安装副本：

```bash
cd ~/wheel_robot
python3 - <<'PY'
from pathlib import Path
for root in (Path('src/wheel_robot'), Path('install/wheel_robot/lib/wheel_robot')):
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.sh'):
            data = path.read_bytes()
            if b'\r\n' in data:
                path.write_bytes(data.replace(b'\r\n', b'\n'))
                print('LF:', path)
PY
colcon build --packages-select wheel_robot --symlink-install
source install/setup.bash
ros2 launch wheel_robot line_follow.launch.py enable_motion:=true
```

单独巡线验证结束后，先退出该节点，再启动 ICN。仓库内的屏幕入口为 `bash src/wheel_robot/scripts/run_icn_on_screen.sh`，实际启动 `icn.launch.py`；仓库没有名为 `icn.sh` 的文件，板子上的同名包装脚本需另外核对。两个入口都依赖 `line_follower.py`。启动日志应包含 `near_continuity=on` 和实际 `source` 路径，新版 STOP 日志应带 `mask/picked`；仅修复换行不会把板子上的旧算法升级成新版本。

## 2026-10-25 巡线回归修复与阶段复核

此节以现场调参快照 `b717ae5` 为修复前版本，阶段实现为 `52e6ad3`。当前修复先恢复基础巡线；下述阶段风险仍需处理，不代表整圈比赛已经验证。

### 两个丢线提示的已复现原因

现场版本把 `HsvStripLineDetector.detect()` 中近段的 `allow_corner` 从 `(name == "near")` 改成 `False`，关闭了原来依赖上一帧中心的超宽线段连续跟踪。它影响的并非只有弯道：直道黑线因透视、距离或俯仰而超过宽度窗口时，也会被原窗口拒绝。

640×480、当前 ROI 宽约 614 像素时，近段线宽窗口约 12～36 像素。用一帧 20 像素宽的黑线建立跟踪，再输入同一位置的 42 像素宽直线：旧版有效，现场版本报 `no_black_tape`。让近段只有部分行保持 20 像素宽，其余变为 42 像素，则现场版本报 `insufficient_vertical_tape`。恢复后，两者均与旧版一样有效。

这说明报错不一定是图像里没有黑色，而可能是所有黑线候选被宽度规则过滤。`sharp_curve_width_ratio=999` 只抑制横向干扰标记，无法放宽 `max_line_width_ratio`，也不会让已经被拒绝的像素重新参与跟踪。

本次恢复近段连续跟踪和原横向干扰阈值 `2.5`；保留现场的 `forward_speed=0.25`、`steering_kp=1.40` 以及新增阶段代码。没有通过全面放宽黑色、置信度或覆盖率阈值掩盖丢线。

### 新增诊断输出

- 启动日志打印实际 `source` 路径、`near_continuity=on`、ROI、亮度和线宽窗口，便于发现运行了旧的 install 副本或加载了不同参数。
- 图像显示 `MASK pixels`、`picked` 和近段线宽范围。各带显示实际最宽段 `max`、普通宽度上限 `keep<=` 和是否有效 `ok`。
- 丢线日志追加 `mask/picked`、各带行数、最宽段和覆盖率。`picked` 是宽度/连续性筛选后像素数量，后续段间门控仍可能将该带判无效。

排查时：`mask=0` 才优先看曝光/亮度/ROI；`mask>0` 且 `picked=0` 优先看线宽和关联；有 picked 但 `ok=0` 看近段支持、横线或段间门控；覆盖率不足则看虚线空隙、反光和各带有效行数。此补偿要求之前有可靠轨迹，冷启动时整条线都超宽仍可能拒绝，不能据此盲目扩大到任何黑块都通过。

### 阶段代码中尚未修复的独立风险

1. `track_phase.py::_observe_first_corner` 积分的是图像误差绝对值，不是实测转角。默认要求误差峰值 0.55，640 像素宽图像约相当于融合中心偏离 176 像素。跟线较好时可能一直达不到。即使 `corner=True` 增加了累计量，峰值仍只更新原始误差；复现输入可得到 `turn=1.05/0.30 peak=0.05/0.55`，阶段保持 FIRST_CORNER，所有元素继续被禁用。应使用真实朝向变化并结合直线恢复判断，不能简单把“跟线越差”当作“转弯越充分”。
2. `_check_platform_cleared` 只有动作结束、恢复线和检测器不再报告平台的条件，没有实际前进/跨越证据。默认台阶动作依然只是停车 5 秒。合成输入不含任何运动证据，也能进入 SEEK_ROUNDABOUT；检测器因光照漏检不应被直接解释为越过了平台。应与实际越障动作完成和运动证据配合。
3. `observe_roundabout` 在非 control 模式下允许“超过 1 秒、连续 8 次没有 junction”就进入 SEEK_ZEBRA，即使圆环完成计数仍为 0，甚至子状态是 RECOVER。当前默认 observe 仅适合观察，不能据此认定完成绕环。正式任务应以真实绕行完成事件解锁后续元素，观察状态与实际任务进度需要分开。

以上三项为复核发现，未在此次基础巡线修复中改动，避免同时改变整圈任务策略而无法判断恢复效果。恢复连续跟踪中的 `corner` 标志也不是实际转角测量，不能把它单独当作首弯完成证据。

### 验证与车上复测

新增 4 项现场巡线回归测试与既有 26 项圆环测试共 30 项通过。150 帧合成场景中，修复后的检测决策和基础控制输出与原始基线一致；调试图因新增诊断文字有意变化。完整单黑线测试为 38 通过、5 失败，5 项为先前已存在的相同失败。尚无本次实车录像或 ROS 实机验证，不能据此保证所有真实丢线原因都已消除。

在机器人工作空间根目录：

```bash
colcon build --packages-select wheel_robot --symlink-install
source install/setup.bash
# 先独立验证基础巡线，隔离阶段和其他元素
ros2 launch wheel_robot line_follow.launch.py enable_motion:=true
# 基础巡线稳定后再查看阶段流程
ros2 launch wheel_robot icn.launch.py enable_motion:=true roundabout_mode:=observe
```

启动时确认日志包含 `near_continuity=on`，且 `source` 指向这次安装的代码。若仍报错，保留完整 STOP 行和对应原始画面，才能区分参数/成像问题与过滤问题。保留失效停车，不应靠无限沿用上次速度来消除提示。

以下为圆环模块的使用说明，阶段限制及最新修复以上面的复核记录为准。圆环功能集中在 `scripts/roundabout.py`，包含视觉检测、轨迹候选、状态机、只读姿态接口、屏幕显示和离线查看入口。基础巡线沿用原有检测策略和 PD 公式，最新速度与增益以当前 YAML 为准。圆环接管时提供新的目标误差、速度上限以及禁止沿用旧指令的标志。

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
