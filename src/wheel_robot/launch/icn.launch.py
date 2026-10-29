"""ICN 比赛任务：巡线为主体 + 斑马线等元素。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _launch_nodes(context):
    share = FindPackageShare("wheel_robot")
    line_params = PathJoinSubstitution([share, "config", "line_follower.yaml"])
    element_params = PathJoinSubstitution([share, "config", "icn.yaml"])
    enable_motion = LaunchConfiguration("enable_motion")
    roundabout_mode = LaunchConfiguration("roundabout_mode").perform(context)
    overrides = {"enable_motion": ParameterValue(enable_motion, value_type=bool)}
    if roundabout_mode:
        overrides["roundabout_mode"] = roundabout_mode
    return [Node(
        package="wheel_robot", executable="icn.py", name="line_follower",
        output="screen", emulate_tty=True,
        parameters=[line_params, element_params, overrides],
    )]


def generate_launch_description():

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "roundabout_mode", default_value="",
                choices=["", "off", "observe", "control"],
                description="empty: use icn.yaml; off: skip ring; observe: detect only; control: circle task",
            ),
            DeclareLaunchArgument(
                "enable_motion",
                default_value="true",
                description="true 才真的发速度；先架空看调试窗",
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
