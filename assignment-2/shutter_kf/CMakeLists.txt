cmake_minimum_required(VERSION 3.8)
project(shutter_kf)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(shutter_lookat REQUIRED)

# Install Python scripts. Globbed rather than listed because the relevant python
# files change as students work on the assignment.
# CONFIGURE_DEPENDS re-runs the glob at build time, so a newly created script is installed
# by the next `colcon build` without a manual reconfigure.
file(GLOB PYTHON_SCRIPT_FILES
  CONFIGURE_DEPENDS
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py"
)
install(PROGRAMS
  ${PYTHON_SCRIPT_FILES}
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Test dependencies
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
