#!/usr/bin/env bash
# 在香橙派自己的屏幕上运行巡线节点（不经过 MobaXterm 的 X11 转发）。
#
# 问题：SSH 会话里 DISPLAY=localhost:10.0，OpenCV 的 imshow 窗口会画到 Windows 上，
#       ros2 launch 只是把这个变量继承给 line_follower.py，所以必须改环境再启动。
#
# 本脚本做四件事：
#   1. 把 DISPLAY 强制指向板子本地桌面（默认 :0）；
#   2. 逐个尝试 XAUTHORITY，直到 xdpyinfo 真的能连上，连不上就直接报错退出，
#      不会带着错误的 DISPLAY 去启动节点（Qt 后端连不上时可能直接 abort）；
#   3. 关掉熄屏 / DPMS / 屏保，并在启动后定时复核（桌面电源管理会改回自己的设置）；
#   4. 检查 /dev/video0 与残留的 line_follower 进程，然后启动节点。
#
# 防熄屏相关环境变量：
#   KEEPAWAKE=0          bash ...   不碰屏幕省电设置
#   KEEPAWAKE_REFRESH=0  bash ...   只下发一次，不做定时复核
#   KEEPAWAKE_REFRESH=30 bash ...   复核间隔秒数（默认 30）
#
# 用法：
#   bash run_line_follow_on_screen.sh                       # 默认能动（自动追加 enable_motion:=true）
#   bash run_line_follow_on_screen.sh enable_motion:=false  # 架空调试，只识别不动车
#   LOCAL_DISPLAY=:1 bash run_line_follow_on_screen.sh      # 桌面在别的 display
#
# 其余参数原样透传给 ros2 launch。
# 注意 enable_motion 的优先级：命令行 > launch 默认值 > YAML。
# 因为 launch 里那个默认值会覆盖 YAML，所以本脚本默认显式传 enable_motion:=true，
# 这样即使 install/ 里的 launch 还是旧的（默认 false），也能直接进入运动模式。

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
    # "DPMS is Disabled" = 已关闭；"Server is not enabled" = 服务器级就没开 DPMS
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

# ---------- 5. 摄像头 / 旧进程检查 ----------
if pgrep -af 'line_follower\.py' >/dev/null 2>&1; then
    echo "警告：已经有 line_follower 在运行，/dev/video0 不能被两个进程同时打开：" >&2
    pgrep -af 'line_follower\.py' >&2
    echo "      先在对应终端 Ctrl+C，或 pkill -f 'line_follower\\.py'，再启动本脚本。" >&2
fi

if [ -e /dev/video0 ]; then
    if command -v fuser >/dev/null 2>&1; then
        busy="$(fuser /dev/video0 2>/dev/null || true)"
        if [ -n "$busy" ]; then
            echo "警告：/dev/video0 已被占用（pid:$busy），巡线会一直 STOP camera_open_failed" >&2
        fi
    fi
else
    echo "警告：/dev/video0 不存在，节点会一直 STOP camera_open_failed" >&2
fi

# 全屏兜底工具：节点里会自动依次尝试 OpenCV 属性、wmctrl、xdotool
if ! command -v wmctrl >/dev/null 2>&1 && ! command -v xdotool >/dev/null 2>&1; then
    echo "提示：wmctrl / xdotool 都没装，全屏只能依赖 OpenCV 自带属性；"
    echo "      若节点日志提示「调试窗口没能全屏」，装一下再重启节点（不用重编）："
    echo "      sudo apt install wmctrl"
fi

# ---------- 6. 启动 ----------
# 默认开启运动：不依赖 install/ 里 line_follow.launch.py 的默认值
# （launch 的 enable_motion 参数优先级高于 YAML，老 launch 会把 YAML 的 true 覆盖成 false）。
# 要架空调试就显式传 enable_motion:=false。
launch_args=("$@")
case " $* " in
    *enable_motion:=*) ;;
    *) launch_args+=("enable_motion:=true") ;;
esac

echo "启动：ros2 launch wheel_robot line_follow.launch.py ${launch_args[*]}"
echo "窗口标题应为 Black Line Follower (Q/Esc to stop)，按 Q 或 Esc 退出"
# 这里不用 exec：要保留 trap，节点退出后好把防熄屏的复核循环收掉
ros2 launch wheel_robot line_follow.launch.py "${launch_args[@]}"
exit_code=$?
blank_keeper_stop
exit "$exit_code"
