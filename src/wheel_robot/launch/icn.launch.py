"""ICN 比赛任务：巡线为主体 + 斑马线等元素。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("wheel_robot")
    line_params = PathJoinSubstitution([share, "config", "line_follower.yaml"])
    element_params = PathJoinSubstitution([share, "config", "icn.yaml"])
    enable_motion = LaunchConfiguration("enable_motion")
    roundabout_mode = LaunchConfiguration("roundabout_mode")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "roundabout_mode", default_value="observe",
                choices=["off", "observe", "control"],
                description="off: original line following; observe: HUD only; control: circle task",
            ),
            DeclareLaunchArgument(
                "enable_motion",
                default_value="true",
                description="true 才真的发速度；先架空看调试窗",
            ),
            Node(
                package="wheel_robot",
                executable="icn.py",
                name="line_follower",
                output="screen",
                emulate_tty=True,
                parameters=[
                    line_params,      # 巡线参数：和单独跑 line_follower 时完全同一份
                    element_params,   # 元素参数：斑马线/任务动作
                    {
                        "enable_motion": ParameterValue(
                            enable_motion, value_type=bool
                        ),
                        "roundabout_mode": ParameterValue(roundabout_mode, value_type=str),
                    },
                ],
            ),
        ]
    )
