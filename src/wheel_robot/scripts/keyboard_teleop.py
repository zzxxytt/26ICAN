#!/usr/bin/env python3
"""Safe keyboard teleoperation for the wheeled-legged robot.

Upper-level applications live in the wheel_robot package.  The separate
wheeled_legged_pkg package remains responsible for the ESP32 serial bridge.
"""

import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


HELP = """
轮腿机器人键盘遥控（SWA 必须处于 2）

  W / S : 前进 / 后退（按住才持续，松开自动停止）
  A / D : 左转 / 右转（按住才持续，松开自动停止）
  空格  : 立即停止
  R / F : 腿高增加 / 减少 0.005 m
  T / G : 线速度增加 / 减少 0.02 m/s
  Y / H : 角速度增加 / 减少 0.10 rad/s
  J+回车: 预备并确认一次跳跃（先自动停车）
  Q     : 停止并退出

提示：终端的按键自动重复用于“按住”控制；松开后 0.35 秒内自动归零。
"""


class KeyboardTeleop(Node):
    def __init__(self) -> None:
        super().__init__("wheel_robot_keyboard_teleop")

        self.declare_parameter("linear_speed", 0.10)
        self.declare_parameter("angular_speed", 0.40)
        self.declare_parameter("linear_direction_sign", 1.0)
        self.declare_parameter("height", 0.20)
        self.declare_parameter("deadman_timeout", 0.35)

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_speed = float(self.get_parameter("angular_speed").value)
        self.linear_direction_sign = float(
            self.get_parameter("linear_direction_sign").value
        )
        self.height = float(self.get_parameter("height").value)
        self.deadman_timeout = float(self.get_parameter("deadman_timeout").value)
        if self.linear_direction_sign not in (-1.0, 1.0):
            raise ValueError("linear_direction_sign 只能是 -1.0 或 1.0")

        self.linear = 0.0
        self.angular = 0.0
        self.last_motion_key = 0.0
        self.jump_armed_until = 0.0
        self.jump_start_at = 0.0
        self.jump_end_at = 0.0
        self.running = True

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.posture_pub = self.create_publisher(JointState, "/cmd_posture", 10)
        self.jump_pub = self.create_publisher(Bool, "/cmd_jump", 10)
        self.timer = self.create_timer(0.05, self.publish_commands)

        print(HELP, flush=True)
        self.print_status("就绪，当前速度为零")

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def stop_motion(self) -> None:
        self.linear = 0.0
        self.angular = 0.0

    def print_status(self, action: str) -> None:
        print(
            f"\r{action:<18} | 指令 v={self.linear:+.2f} m/s  "
            f"w={self.angular:+.2f} rad/s  腿高={self.height:.3f} m",
            end="",
            flush=True,
        )

    def handle_key(self, key: str) -> None:
        key = key.lower()
        now = time.monotonic()

        if key == "w":
            self.jump_armed_until = 0.0
            self.linear = self.linear_direction_sign * self.linear_speed
            self.angular = 0.0
            self.last_motion_key = now
            self.print_status("前进")
        elif key == "s":
            self.jump_armed_until = 0.0
            self.linear = -self.linear_direction_sign * self.linear_speed
            self.angular = 0.0
            self.last_motion_key = now
            self.print_status("后退")
        elif key == "a":
            self.jump_armed_until = 0.0
            self.linear, self.angular = 0.0, self.angular_speed
            self.last_motion_key = now
            self.print_status("左转")
        elif key == "d":
            self.jump_armed_until = 0.0
            self.linear, self.angular = 0.0, -self.angular_speed
            self.last_motion_key = now
            self.print_status("右转")
        elif key == " ":
            self.jump_armed_until = 0.0
            self.stop_motion()
            self.print_status("急停")
        elif key == "j":
            self.stop_motion()
            self.jump_armed_until = now + 3.0
            self.print_status("跳跃已预备，按回车确认")
        elif key in ("\r", "\n") and now < self.jump_armed_until:
            self.jump_armed_until = 0.0
            # Send stationary commands first, then emit a short rising-edge
            # pulse. The ESP32 independently checks zero speed and ground load.
            self.jump_start_at = now + 0.40
            self.jump_end_at = self.jump_start_at + 0.20
            self.print_status("跳跃将在 0.4 秒后触发")
        elif key == "r":
            self.height = self.clamp(self.height + 0.005, 0.14, 0.36)
            self.print_status("升高")
        elif key == "f":
            self.height = self.clamp(self.height - 0.005, 0.14, 0.36)
            self.print_status("降低")
        elif key == "t":
            self.linear_speed = self.clamp(self.linear_speed + 0.02, 0.02, 0.50)
            self.print_status(f"线速度档 {self.linear_speed:.2f}")
        elif key == "g":
            self.linear_speed = self.clamp(self.linear_speed - 0.02, 0.02, 0.50)
            self.print_status(f"线速度档 {self.linear_speed:.2f}")
        elif key == "y":
            self.angular_speed = self.clamp(self.angular_speed + 0.10, 0.10, 2.00)
            self.print_status(f"角速度档 {self.angular_speed:.2f}")
        elif key == "h":
            self.angular_speed = self.clamp(self.angular_speed - 0.10, 0.10, 2.00)
            self.print_status(f"角速度档 {self.angular_speed:.2f}")
        elif key == "q":
            self.stop_motion()
            self.running = False

    def check_deadman(self) -> None:
        moving = self.linear != 0.0 or self.angular != 0.0
        if moving and time.monotonic() - self.last_motion_key > self.deadman_timeout:
            self.stop_motion()
            self.print_status("松键自动停止")

    def publish_commands(self) -> None:
        self.check_deadman()

        velocity = Twist()
        velocity.linear.x = self.linear
        velocity.angular.z = self.angular
        self.cmd_vel_pub.publish(velocity)

        posture = JointState()
        posture.header.stamp = self.get_clock().now().to_msg()
        posture.name = ["joint_height", "joint_roll", "joint_pitching"]
        posture.position = [self.height, 0.0, 0.0]
        self.posture_pub.publish(posture)

        now = time.monotonic()
        jump = Bool()
        jump.data = self.jump_start_at <= now < self.jump_end_at and self.jump_start_at > 0.0
        self.jump_pub.publish(jump)
        if self.jump_end_at > 0.0 and now >= self.jump_end_at:
            self.jump_start_at = 0.0
            self.jump_end_at = 0.0

    def publish_final_stop(self) -> None:
        self.stop_motion()
        self.jump_armed_until = 0.0
        self.jump_start_at = 0.0
        self.jump_end_at = 0.0
        # Repeat the stop so wl_base_node receives it even during a transient
        # scheduling delay, then leave its own one-second watchdog as backup.
        for _ in range(10):
            self.publish_commands()
            time.sleep(0.02)


def main(args=None) -> None:
    if not sys.stdin.isatty():
        print("错误：键盘遥控必须在交互式终端中运行。", file=sys.stderr)
        return

    rclpy.init(args=args)
    node = KeyboardTeleop()
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.02)
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if readable:
                node.handle_key(sys.stdin.read(1))
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_final_stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\n已发送零速度，键盘遥控退出。", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
