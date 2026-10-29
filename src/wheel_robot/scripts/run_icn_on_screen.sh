#!/usr/bin/env bash
# 在香橙派自己的屏幕上运行 ICN 比赛任务节点（不经过 MobaXterm 的 X11 转发）。
#
# 和 run_line_follow_on_screen.sh 的差别只有四处：
#   1. 启动 icn.launch.py（巡线 + 斑马线状态机），不是 line_follow.launch.py；
#   2. 启动前检查 install 里有没有 icn.py / icn.launch.py / icn.yaml，
#      新文件忘了重编译时当场拦住，不会跑到一半才发现；
#   3. 旧进程检查覆盖所有会抢 /dev/video0 的程序（icn / line_follower / zebra / camera_viewer）；
#   4. 提醒底盘是否在跑，避免"节点在跑、车不动"。
#
# 用法：
#   bash run_icn_on_screen.sh                        # 默认能动
#   bash run_icn_on_screen.sh enable_motion:=false   # 架空，只看图不动车
#   bash run_icn_on_screen.sh roundabout_mode:=off   # 无圆环阶段，过平台后巡线找斑马线
#   bash run_icn_on_screen.sh roundabout_mode:=observe # 有圆环识别阶段，不接管转向
#   不传 roundabout_mode 时读取 icn.yaml（当前为 off）。
#   bash run_icn_on_screen.sh roundabout_mode:=control # 有圆环识别阶段，并接管绕环
#   LOCAL_DISPLAY=:1 bash run_icn_on_screen.sh       # 桌面在别的 display
#   KEEPAWAKE=0 bash run_icn_on_screen.sh            # 不碰熄屏设置
#
# 其余参数原样透传给 ros2 launch。
# enable_motion 优先级：命令行 > launch 默认值 > YAML，所以本脚本默认显式传 true。

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../.." && pwd)"
TARGET_DISPLAY="${LOCAL_DISPLAY:-:0}"
USER_ID="$(id -u)"

echo "工作空间   = $WS"
if [ ! -f "$WS/install/setup.bash" ]; then
    echo "找不到 $WS/install/setup.bash" >&2
    echo "先编译：cd $WS && colcon build --packages-select wheel_robot" >&2
    exit 1
fi

# ---------- 1. ROS 环境 ----------
set +u
if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    # shellcheck disable=SC1090
    . "/opt/ros/$ROS_DISTRO/setup.bash"
else
    for cand in /opt/ros/humble /opt/ros/jazzy /opt/ros/iron; do
        if [ -f "$cand/setup.bash" ]; then
            # shellcheck disable=SC1090
            . "$cand/setup.bash"
            break
        fi
    done
fi
# shellcheck disable=SC1091
. "$WS/install/setup.bash"
set -u
echo "ROS_DISTRO = ${ROS_DISTRO:-<未知>}"

# ---------- 2. 把 DISPLAY 指到板子本地 ----------
# 覆盖掉 SSH 转发过来的 localhost:10.0
export DISPLAY="$TARGET_DISPLAY"
# 桌面是 Wayland 时，不让 Qt 走 wayland 插件，统一走 X11 / XWayland
unset WAYLAND_DISPLAY

# ---------- 3. 逐个试 XAUTHORITY，直到真的连上 X ----------
candidates=()
if [ -n "${XAUTHORITY:-}" ] && [ -f "${XAUTHORITY:-/nonexistent}" ]; then
    candidates+=("$XAUTHORITY")
fi
candidates+=(
    "$HOME/.Xauthority"
    "/run/user/$USER_ID/gdm/Xauthority"
)
for f in /run/user/"$USER_ID"/.mutter-Xwaylandauth.*; do
    [ -e "$f" ] && candidates+=("$f")
done

# $1 = 待测试的 XAUTHORITY
try_display() {
    if command -v xdpyinfo >/dev/null 2>&1; then
        DISPLAY="$TARGET_DISPLAY" XAUTHORITY="$1" xdpyinfo >/dev/null 2>&1
    else
        # 没装 x11-utils 时，退化成用 OpenCV 在子进程里试开一个窗口
        DISPLAY="$TARGET_DISPLAY" XAUTHORITY="$1" python3 - >/dev/null 2>&1 <<'PY'
import cv2
import sys
try:
    cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
    cv2.destroyWindow("__probe__")
except cv2.error:
    sys.exit(1)
PY
    fi
}

XAUTH_OK=""
for auth in "${candidates[@]}"; do
    if try_display "$auth"; then
        XAUTH_OK="$auth"
        break
    fi
done
# 最后一种可能：X 服务端允许本地连接（xhost +local:）
if [ -z "$XAUTH_OK" ] && try_display /dev/null; then
    XAUTH_OK=/dev/null
fi

if [ -z "$XAUTH_OK" ]; then
    cat >&2 <<'EOF'

连不上板子本地显示，窗口不会出现在车屏上。按顺序排查：
  1. 板子的桌面是否已经登录？没人登录就没有 X 会话，先在板子屏幕上登录一次。
  2. 桌面是 Wayland 且没开 XWayland？换成 X11 会话再试。
  3. 报 "Authorization required" / "no authorization protocol specified"
     说明 DISPLAY 通了但 cookie 不对：不要用 sudo 跑本脚本。
  4. 桌面不在 :0？试试 LOCAL_DISPLAY=:1 bash 本脚本。
EOF
    echo "先诊断：bash $HERE/opi_screen_setup.sh diag" >&2
    exit 1
fi

export XAUTHORITY="$XAUTH_OK"
echo "DISPLAY    = $DISPLAY"
echo "XAUTHORITY = $XAUTHORITY"
if command -v xdpyinfo >/dev/null 2>&1; then
    echo "分辨率     = $(DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" xdpyinfo | awk '/dimensions:/ {print $2; exit}')"
fi

# ---------- 4. 防熄屏 / 防 DPMS ----------
KEEPAWAKE="${KEEPAWAKE:-1}"
KEEPAWAKE_REFRESH="${KEEPAWAKE_REFRESH:-30}"

blank_keeper_pid=""
blank_keeper_stop() {
    if [ -n "$blank_keeper_pid" ]; then
        kill "$blank_keeper_pid" 2>/dev/null || true
        wait "$blank_keeper_pid" 2>/dev/null || true
        blank_keeper_pid=""
    fi
}
# 退出（含 Ctrl+C）时收掉后台复核循环，避免留下孤儿进程
trap blank_keeper_stop EXIT INT TERM

disable_blanking() {
    # 顺序有讲究：先把 DPMS 三个超时归零，最后再 -dpms 关掉，
    # 反过来写的话后面的 "dpms 0 0 0" 会把 DPMS 重新 Enable。
    xset s off 2>/dev/null
    xset s noblank 2>/dev/null
    xset dpms 0 0 0 2>/dev/null
    xset -dpms 2>/dev/null
    xset s 0 0 2>/dev/null
}

if [ "$KEEPAWAKE" = "0" ]; then
    echo "防熄屏     = 已跳过（KEEPAWAKE=0）"
elif ! command -v xset >/dev/null 2>&1; then
    echo "防熄屏     = 跳过：没装 xset，先 sudo apt install x11-xserver-utils" >&2
else
    disable_blanking
    if xset q 2>/dev/null | grep -Eq "DPMS is Disabled|Server is not enabled"; then
        echo "防熄屏     = 已关闭屏保与 DPMS"
    else
        echo "防熄屏     = 已下发 xset，但 DPMS 仍为 Enabled（桌面电源管理可能接管）"
        echo "             长期生效跑一次：bash $HERE/opi_screen_setup.sh persist"
    fi
    if [ "$KEEPAWAKE_REFRESH" != "0" ]; then
        (
            while sleep "$KEEPAWAKE_REFRESH"; do
                disable_blanking
            done
        ) &
        blank_keeper_pid=$!
        echo "            每 ${KEEPAWAKE_REFRESH}s 复核一次，退出节点时自动停止"
    fi
fi

# ---------- 5. 检查 ICN 和共用巡线的安装文件 ----------
PKG_PREFIX="$(ros2 pkg prefix wheel_robot 2>/dev/null || true)"
if [ -z "$PKG_PREFIX" ]; then
    echo "找不到 wheel_robot 包，先编译：" >&2
    echo "  cd $WS && colcon build --packages-select wheel_robot --symlink-install" >&2
    exit 1
fi

missing=0
for f in \
    "$PKG_PREFIX/lib/wheel_robot/icn.py" \
    "$PKG_PREFIX/lib/wheel_robot/line_follower.py" \
    "$PKG_PREFIX/share/wheel_robot/launch/icn.launch.py" \
    "$PKG_PREFIX/share/wheel_robot/config/line_follower.yaml" \
    "$PKG_PREFIX/share/wheel_robot/config/icn.yaml"
do
    if [ ! -f "$f" ]; then
        echo "缺少安装产物：$f" >&2
        missing=1
    fi
done
if [ "$missing" = "1" ]; then
    cat >&2 <<EOF

新加的文件没进 install/，ros2 launch 会报 executable 'icn.py' not found。
两步：
  1) 确认 $WS/src/wheel_robot/CMakeLists.txt 的 install(PROGRAMS ...) 里有 scripts/icn.py
  2) 重新编译（加 --symlink-install，之后改 yaml/py 都不用再编译）：
       cd $WS && colcon build --packages-select wheel_robot --symlink-install
EOF
    exit 1
fi

echo "ICN 程序   = $(readlink -f "$PKG_PREFIX/lib/wheel_robot/icn.py")"
echo "巡线程序   = $(readlink -f "$PKG_PREFIX/lib/wheel_robot/line_follower.py")"
echo "巡线参数   = $(readlink -f "$PKG_PREFIX/share/wheel_robot/config/line_follower.yaml")"

# ---------- 6. 摄像头 / 旧进程 / 底盘检查 ----------
stale=0
for pat in 'icn\.py' 'line_follower\.py' 'zebra_crossing_detector\.py' 'camera_viewer\.py'; do
    if pgrep -af "$pat" >/dev/null 2>&1; then
        echo "警告：已经有进程在跑（$pat），/dev/video0 不能被两个进程同时打开：" >&2
        pgrep -af "$pat" >&2
        stale=1
    fi
done
if [ "$stale" = "1" ]; then
    echo "      先在对应终端 Ctrl+C（或 pkill -f），再启动本脚本。" >&2
fi

if [ -e /dev/video0 ]; then
    if command -v fuser >/dev/null 2>&1; then
        busy="$(fuser /dev/video0 2>/dev/null || true)"
        if [ -n "$busy" ]; then
            echo "警告：/dev/video0 已被占用（pid:$busy），节点会一直 STOP camera_unavailable" >&2
        fi
    fi
else
    echo "警告：/dev/video0 不存在，节点会一直 STOP camera_unavailable" >&2
fi

if ! timeout 3 ros2 node list 2>/dev/null | grep -qiE 'wheeltec|base|chassis'; then
    echo "提示：没看到底盘节点，/cmd_vel 可能没人订阅，车不会动。"
    echo "      底盘一般单独启动：ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py"
fi

if ! command -v wmctrl >/dev/null 2>&1 && ! command -v xdotool >/dev/null 2>&1; then
    echo "提示：wmctrl / xdotool 都没装，全屏只能依赖 OpenCV 自带属性；"
    echo "      若日志提示「调试窗口没能全屏」，装一下即可：sudo apt install wmctrl"
fi

# ---------- 7. 启动 ----------
launch_args=("$@")
case " $* " in
    *enable_motion:=*) ;;
    *) launch_args+=("enable_motion:=true") ;;
esac

echo "启动：ros2 launch wheel_robot icn.launch.py ${launch_args[*]}"
echo "窗口标题：Black Line Follower (Q/Esc to stop)（icn 继承父类的窗口名，按 Q 或 Esc 退出）"
echo "右上角小窗 = 斑马线鸟瞰图；黄字 MISSION 行 = 当前状态 / 斑马线确认 / 已过次数"
echo "斑马线动作读 icn.yaml：zebra_stop_seconds（停车秒数）+ zebra_cross_speed（通过速度）"
# 不用 exec：要保留 trap，节点退出后好把防熄屏的复核循环收掉
ros2 launch wheel_robot icn.launch.py "${launch_args[@]}"
exit_code=$?
blank_keeper_stop
exit "$exit_code"
