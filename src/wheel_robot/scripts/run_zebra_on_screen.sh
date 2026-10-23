#!/usr/bin/env bash
# 在香橙派自己的屏幕上运行斑马线检测（不经过 MobaXterm 的 X11 转发）。
#
# 用法：
#   bash run_zebra_on_screen.sh                    # 显示到板子本地屏幕 :0
#   LOCAL_DISPLAY=:1 bash run_zebra_on_screen.sh   # 指定其它 display
#   FULLSCREEN=1 bash run_zebra_on_screen.sh       # 运行后把窗口切成全屏
#
# 关键点：SSH 里如果开了 X11 转发，DISPLAY 会是 localhost:10.0，
# 画面就跑到 Windows 的 MobaXterm 上了。本脚本显式覆盖成 :0。
#
# 本脚本不改动 detector 源码；全屏是通过窗口管理器 wmctrl 从外部完成的，
# 需要 sudo apt install wmctrl。

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DETECTOR="$HERE/zebra_crossing_detector.py"
WINDOW_NAME="Zebra Stripe Detection"

if [ ! -f "$DETECTOR" ]; then
    echo "找不到 $DETECTOR" >&2
    exit 1
fi

# 覆盖掉 SSH 转发过来的 DISPLAY
export DISPLAY="${LOCAL_DISPLAY:-:0}"

# 指向真实桌面的 X authority。SSH 会话里通常不继承，需要显式指定。
if [ -z "${XAUTHORITY:-}" ] || [ ! -f "${XAUTHORITY:-/nonexistent}" ]; then
    for cand in "$HOME/.Xauthority" "/run/user/$(id -u)/gdm/Xauthority"; do
        if [ -f "$cand" ]; then
            export XAUTHORITY="$cand"
            break
        fi
    done
fi

echo "DISPLAY    = $DISPLAY"
echo "XAUTHORITY = ${XAUTHORITY:-<未设置>}"

if command -v xdpyinfo >/dev/null 2>&1; then
    if ! xdpyinfo >/dev/null 2>&1; then
        echo
        echo "连不上 $DISPLAY，画面不会出现在车屏上。常见原因：" >&2
        echo "  1. 板子的桌面没登录（没人登录就没有 X 会话）" >&2
        echo "  2. 没有 XAUTHORITY 权限（换用户或用 sudo 会丢权限）" >&2
        echo "  3. 桌面跑的是 Wayland 而不是 X11" >&2
        echo "先跑：bash $HERE/opi_screen_setup.sh diag" >&2
        exit 1
    fi
    echo "X 连接正常：$(xdpyinfo | awk '/dimensions:/ {print $2; exit}')"
else
    echo "xdpyinfo 不可用，跳过连通性检查（sudo apt install x11-utils）"
fi

cd "$HERE" || exit 1

if [ "${FULLSCREEN:-0}" != "1" ]; then
    exec python3 "$DETECTOR" "$@"
fi

if ! command -v wmctrl >/dev/null 2>&1; then
    echo "FULLSCREEN=1 需要 wmctrl：sudo apt install wmctrl" >&2
    echo "改为普通窗口运行。"
    exec python3 "$DETECTOR" "$@"
fi

python3 "$DETECTOR" "$@" &
detector_pid=$!

# 等窗口出现再切全屏，最多等 10 秒
for _ in $(seq 1 40); do
    if wmctrl -l 2>/dev/null | grep -q "$WINDOW_NAME"; then
        if wmctrl -r "$WINDOW_NAME" -b add,fullscreen; then
            echo "已切换为全屏"
        else
            echo "wmctrl 切全屏失败，保持普通窗口" >&2
        fi
        break
    fi
    sleep 0.25
done

wait "$detector_pid"
