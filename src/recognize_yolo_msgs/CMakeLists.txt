cmake_minimum_required(VERSION 3.8)
project(recognize_yolo_msgs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 消息里用到了 std_msgs/Header，必须先 find_package。
# rosidl_generate_interfaces 的 DEPENDENCIES 只是声明依赖关系，
# 不会替你调用 find_package。
find_package(std_msgs REQUIRED)

# ---------------------------------------------------------------------------
# 接口定义
#
# 这个包**只有**消息，没有 Python / C++ 代码，因此不会调用
# ament_python_install_package()，也就避开了与 rosidl 的已知冲突。
# 详见 package.xml 里的说明。
# ---------------------------------------------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  DEPENDENCIES std_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
