cmake_minimum_required(VERSION 3.8)
project(recognize_yolo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# 安装 Python 包要用它。注意这里**不能**出现 rosidl_generate_interfaces：
# ROS 2 已知限制 —— 同一个 CMake 工程里同时调用
# rosidl_generate_interfaces() 和 ament_python_install_package() 会因
# add_custom_target 重名而构建失败（rosidl 生成 Python 消息代码时自己会调用
# 一次 ament_python_install_package）。消息已拆到 recognize_yolo_msgs 包里。
# 参考：https://docs.ros.org/en/humble/How-To-Guides/Ament-CMake-Python-Documentation.html
find_package(ament_cmake_python REQUIRED)

# 必须在 ament_python_install_package() 之前
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------------------------------------------------------------------------
# C++ NPU 推理进程（可选）
#
# 这个可执行文件只有在找得到 Vivante VIPLite 运行时的板子上才需要编译。
# 传入 -DRECOGNIZE_YOLO_NPU=OFF 可以跳过它，此时 Python 后端仍可正常构建。
# ---------------------------------------------------------------------------
option(RECOGNIZE_YOLO_NPU "Build the VIPLite NPU worker" ON)
option(RECOGNIZE_YOLO_NPU_ROOT "Root of the VIPLite runtime copied from the AW model zoo" "")

set(NPU_WORKER_TARGET "")

if(RECOGNIZE_YOLO_NPU)
  # OpenCV 与线程库只在这个可执行文件里用到，放在分支内按需查找。
  find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
  find_package(Threads REQUIRED)

  find_path(VIP_LITE_INCLUDE_DIR
    NAMES vip_lite.h
    HINTS
      ${RECOGNIZE_YOLO_NPU_ROOT}/include
      /usr/include
      /usr/local/include
      /opt/vip/include
  )

  find_library(VIP_LITE_LIBRARY
    NAMES vip-lite vip_lite
    HINTS
      ${RECOGNIZE_YOLO_NPU_ROOT}/lib
      ${RECOGNIZE_YOLO_NPU_ROOT}/lib_linux_aarch64/A733
      /usr/lib
      /usr/local/lib
      /usr/lib/aarch64-linux-gnu
  )

  if(VIP_LITE_INCLUDE_DIR AND VIP_LITE_LIBRARY)
    message(STATUS "recognize_yolo: VIPLite 已找到, 将编译 NPU worker")
    message(STATUS "  include: ${VIP_LITE_INCLUDE_DIR}")
    message(STATUS "  library: ${VIP_LITE_LIBRARY}")

    add_executable(yolo_npu_worker
      src/yolo_npu_worker.cpp
      src/vip_runner.cpp
      src/postprocess.cpp
    )
    target_include_directories(yolo_npu_worker PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${VIP_LITE_INCLUDE_DIR}
    )
    target_link_libraries(yolo_npu_worker PRIVATE
      ${VIP_LITE_LIBRARY}
      ${OpenCV_LIBS}
      Threads::Threads
    )
    set(NPU_WORKER_TARGET yolo_npu_worker)
  else()
    message(WARNING
      "recognize_yolo: 未找到 VIPLite 运行时 (vip_lite.h / libvip-lite.so)。\n"
      "  跳过 yolo_npu_worker 的编译。\n"
      "  这是正常的降级：Python 部分（mock / onnx 后端）不受影响。\n"
      "  要启用 NPU，先从全志 AW NPU Model Zoo 复制 npuruntime 目录，然后重新编译:\n"
      "    colcon build --packages-select recognize_yolo \\\n"
      "      --cmake-args -DRECOGNIZE_YOLO_NPU_ROOT=$HOME/linux_aw_npu/model_zoo/v0.8.0/common/npuruntime")
  endif()
endif()

# ---------------------------------------------------------------------------
# Python 包与节点
# ---------------------------------------------------------------------------
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/detector_node.py
  scripts/vision_info_node.py
  scripts/check_npu_env.py
  scripts/capture_dataset.py
  scripts/prepare_dataset.py
  DESTINATION lib/${PROJECT_NAME}
)

if(NPU_WORKER_TARGET)
  install(TARGETS ${NPU_WORKER_TARGET}
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# 模型与标签不进 git，运行时通过参数指向实际路径。
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
  OPTIONAL
)

# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_postprocess test/test_postprocess.py)
  ament_add_pytest_test(test_geometry test/test_geometry.py)
endif()

ament_package()
