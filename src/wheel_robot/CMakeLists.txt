cmake_minimum_required(VERSION 3.8)
project(wheel_robot)

find_package(ament_cmake REQUIRED)

# Python application nodes are kept in scripts/ to match the workspace layout.
install(PROGRAMS
  scripts/keyboard_teleop.py
  scripts/diagnostics_monitor.py
  scripts/camera_viewer.py
  scripts/line_follower.py
  scripts/double_line_follower.py
  scripts/zebra_crossing_detector.py
  scripts/step_detector.py
  scripts/roundabout.py
  scripts/icn.py   
  scripts/track_phase.py
  DESTINATION lib/${PROJECT_NAME}
)

# Future launch files and YAML parameters are installed automatically.
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY docs DESTINATION share/${PROJECT_NAME})

ament_package()
