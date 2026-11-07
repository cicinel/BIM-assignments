cmake_minimum_required(VERSION 3.8)
project(shutter_kf_public_tests)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(shutter_lookat REQUIRED)
find_package(shutter_kf REQUIRED)

# --- Testing ---
if(BUILD_TESTING)
  # Only needed to declare the tests below, so they are found here rather than at
  # the top of the file: a plain `colcon build` should not require them.
  find_package(ament_cmake_ros REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  # Integration tests

  function(add_ros_isolated_launch_test path)
    set(RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
    add_launch_test("${path}" RUNNER "${RUNNER}" ${ARGN})

    # CTest names the test after its path, with '/' replaced by '_' (e.g., 'test_test_kalman_filter.py').
    string(REPLACE "/" "_" ctest_name ${path})

    # Set the PYTHONPATH for this test, so that it finds the modules in the 'test' directory.
    set_tests_properties(${ctest_name} PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/test:$ENV{PYTHONPATH}"
    )
  endfunction()

  # test_kalman_filter.py checks the filter's functions without any other nodes.
  add_ros_isolated_launch_test(test/test_kalman_filter.py)

  # test_filter_quality.py brings up the MuJoCo simulation and follow_target.launch.py, waits for
  # the filter's output and then compares it with the true target for 30 seconds, so the 60-second
  # default of add_launch_test is not enough. The number below is a ceiling for a hung test, not
  # an expected runtime.
  add_ros_isolated_launch_test(test/test_filter_quality.py TIMEOUT 240)

  # test_future_filter_quality.py (Part IV, CPSC-5590) brings up the same simulation with delayed
  # observations plus future_kalman_filter.py, and compares the forecast on /future_target with the
  # true target and with simple baselines for 30 seconds.
  add_ros_isolated_launch_test(test/test_future_filter_quality.py TIMEOUT 240)
endif()

ament_package()
