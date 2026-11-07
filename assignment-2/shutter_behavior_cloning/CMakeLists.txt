cmake_minimum_required(VERSION 3.8)
project(shutter_behavior_cloning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Install Python scripts. Globbed rather than listed because the relevant python
# files change as students work on the assignment (e.g., learn_policy.py).
# CONFIGURE_DEPENDS re-runs the glob at build time, so a newly created script is installed
# by the next `colcon build` without a manual reconfigure.
# Helper modules (look_at_solver.py, servo_joint_client.py, transformations.py) are installed
# next to the nodes, so that the nodes can import them.
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

# Install data directory (where the expert saves state-action pairs by default)
install(DIRECTORY data
  DESTINATION share/${PROJECT_NAME}/
)

# --- Testing ---
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Only needed to declare the tests below, so they are found here rather than at
  # the top of the file: a plain `colcon build` should not require them.
  find_package(ament_cmake_ros REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  # Integration tests

  function(add_ros_isolated_launch_test path)
    set(RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
    add_launch_test("${path}" RUNNER "${RUNNER}" ${ARGN})

    # CTest names the test after its path, with '/' replaced by '_' (e.g., 'test_test_policy.py').
    string(REPLACE "/" "_" ctest_name ${path})

    # Set the PYTHONPATH for this test, so that it finds this package's scripts (look_at_solver.py and
    # generate_random_target.py) and the PyTorch installed in your Python virtual environment, which
    # run_policy.py needs. VIRTUAL_ENV is read when the package is built, so build with the environment active.
    set_tests_properties(${ctest_name} PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts:$ENV{PYTHONPATH}:$ENV{VIRTUAL_ENV}/lib/python3.12/site-packages"
    )
  endfunction()

  # The test brings up the MuJoCo simulation and MoveIt Servo (~30 seconds) and then runs 25 trials of
  # a few seconds each. The number below is a ceiling for a hung test, not an expected runtime.
  add_ros_isolated_launch_test(test/test_policy.py TIMEOUT 400)

  # Checks that the model files set in train_utils.py can be downloaded from Google Drive. It needs
  # gdown and PyTorch from your Python virtual environment, so the PYTHONPATH is set as above.
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_download_model test/test_download_model.py
    ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts:$ENV{PYTHONPATH}:$ENV{VIRTUAL_ENV}/lib/python3.12/site-packages"
    TIMEOUT 120
  )
endif()

ament_package()
