#!/usr/bin/env bash
# ===========================================================================
#  数据集采集的便捷入口
#
#  解决两件容易出错的事：
#    1. 文件在 ~/wheel_robot/src/recognize_yolo/scripts/ 下，
#       从别的目录直接跑 `python3 src/...` 会报 No such file or directory
#    2. 采集前要 source 工作空间，否则 import recognize_yolo 会失败
#
#  用法（类别名和输出目录换成你自己的）：
#      bash ~/wheel_robot/capture.sh ball session_ball_01
#
#  想手动指定全部参数时，直接用 python3 跑原脚本：
#      cd ~/wheel_robot
#      source install/setup.bash
#      python3 src/recognize_yolo/scripts/capture_dataset.py \
#          --mode key --label ball --out ~/dataset_raw/session_ball_01
# ===========================================================================

set -e

WORKSPACE="$HOME/wheel_robot"
SCRIPT="$WORKSPACE/src/recognize_yolo/scripts/capture_dataset.py"

LABEL="${1:-}"
SESSION="${2:-}"

if [ ! -f "$SCRIPT" ]; then
    echo "[错误] 找不到采集脚本: $SCRIPT"
    echo "       确认工作空间在 $WORKSPACE，或者改这个脚本里的 WORKSPACE 变量"
    exit 2
fi

if [ -z "$LABEL" ] || [ -z "$SESSION" ]; then
    echo "用法: bash ~/wheel_robot/capture.sh <类别名> <本次会话目录名>"
    echo
    echo "  例如: bash ~/wheel_robot/capture.sh ball session_ball_01"
    echo
    echo "会自动创建输出目录 ~/dataset_raw/<会话目录名>"
    exit 2
fi

# 进工作空间并加载环境（脚本要 import recognize_yolo.camera_source）
cd "$WORKSPACE"
if [ -f install/setup.bash ]; then
    # shellcheck disable=SC1091
    source install/setup.bash
else
    echo "[提示] 还没编译过，install/setup.bash 不存在。"
    echo "       先用 PYTHONPATH 直接指向源码目录："
    export PYTHONPATH="$WORKSPACE/src/recognize_yolo:$PYTHONPATH"
fi

OUT="$HOME/dataset_raw/$SESSION"
mkdir -p "$OUT"

echo "类别: $LABEL"
echo "输出: $OUT"
echo

# 前两个位置参数已经用掉了，shift 掉再透传剩余参数给 Python 脚本，
# 否则 "ball session_ball_01" 会被当成未知选项传下去。
shift 2

exec python3 "$SCRIPT" \
    --mode key \
    --label "$LABEL" \
    --out "$OUT" \
    --pixel-format MJPG \
    --width 640 --height 480 \
    --frame-rate 30 \
    "$@"
