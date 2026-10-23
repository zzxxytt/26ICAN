"""Start only the safe vision line follower (the chassis is launched separately)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution(
        [FindPackageShare("wheel_robot"), "config", "line_follower.yaml"]
    )
    params_file = LaunchConfiguration("params_file")
    enable_motion = LaunchConfiguration("enable_motion")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Line follower ROS parameter file",
            ),
            DeclareLaunchArgument(
                "enable_motion",
                default_value="true",
                description="Explicitly enable /cmd_vel motion after dry-run testing",
            ),
            Node(
                package="wheel_robot",
                executable="line_follower.py",
                name="line_follower",
                output="screen",
                emulate_tty=True,
                parameters=[
                    params_file,
                    {
                        "enable_motion": ParameterValue(
                            enable_motion, value_type=bool
                        )
                    },
                ],
            ),
        ]
    )
