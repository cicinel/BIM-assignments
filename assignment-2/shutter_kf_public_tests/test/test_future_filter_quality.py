#!/usr/bin/env python3
# Public tests for CPSC-5590 Assignment 2 - Part IV

PKG = "shutter_kf_public_tests"
NAME = 'test_future_filter_quality'

import os
import time
import unittest
import numpy as np

import rclpy
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped
from shutter_lookat.msg import Target

from ament_index_python.packages import get_package_prefix, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_testing.actions import ReadyToTest


OBSERVATION_DELAY_FRAMES = 12   # frames by which the observations on /target are delayed (0.4 s at 30 Hz)
MA_WINDOW = 10                  # number of observations averaged by the moving-average baseline
WARMUP_SAMPLES = 60             # true target samples (2 s at 30 Hz) ignored after the first forecast, while the filter settles
MIN_SAMPLES = 100               # minimum number of evaluated true target samples (after the warm-up) needed to judge the forecast
RAW_RATIO = 0.7                 # the forecast's error must be at most this fraction of the raw observations' error


def find_future_filter():
    """
    Find the installed future_kalman_filter.py script
    :return: path to the script if it is installed and executable, otherwise None
    """
    try:
        path = os.path.join(get_package_prefix('shutter_kf'), 'lib', 'shutter_kf', 'future_kalman_filter.py')
    except PackageNotFoundError:
        return None
    return path if os.access(path, os.X_OK) else None


FUTURE_FILTER_PATH = find_future_filter()


def generate_test_description():
    """
    Bring up follow_target.launch.py with delayed observations and the robot following /future_target, plus
    your future_kalman_filter.py. The path type comes from the PATH_TYPE environment variable (horizontal,
    vertical or circular) because `colcon test` has no way to pass launch arguments to a test.
    """

    # Declare launch arguments
    path_type_arg = DeclareLaunchArgument(
        'path_type',
        default_value=os.environ.get('PATH_TYPE', 'circular'),
        description='Path type for target movement (horizontal, vertical or circular)'
    )

    # Include shutter_kf follow_target launch file, without the MuJoCo window, the face or RViz,
    # so that the test can run over ssh
    follow_target_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('shutter_kf'),
                'launch',
                'follow_target.launch.py'
            ])
        ]),
        launch_arguments={
            'add_noise': 'true',
            'path_type': LaunchConfiguration('path_type'),
            'observation_delay_frames': str(OBSERVATION_DELAY_FRAMES),
            'follow_target_topic': '/future_target',
            'headless': 'true',
            'face': 'false',
            'rviz': 'false',
        }.items()
    )

    actions = [path_type_arg, follow_target_launch]

    # Your future Kalman filter. If it is not installed, the test below fails right away.
    if FUTURE_FILTER_PATH is not None:
        actions.append(Node(
            package='shutter_kf',
            executable='future_kalman_filter.py',
            name='future_kalman_filter',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ))

    return (
        LaunchDescription(actions + [
            # launch_testing gives the launch description 15 seconds to signal that it is ready,
            # so this cannot be used to wait out the simulation's startup -- the test below
            # waits for the filter's output instead.
            TimerAction(
                period=2.0,
                actions=[ReadyToTest()]
            ),
        ]), {}
    )


def position(pose_msg):
    """
    :param pose_msg: PoseStamped message
    :return: position in the message as a numpy array
    """
    p = pose_msg.pose.position
    return np.array([p.x, p.y, p.z])


class TestFutureFilterQuality(unittest.TestCase):
    """
    Public tests for future_kalman_filter.py

    The observations on /target are delayed, so the best estimate of where the target is now has to be
    forecast from them. Every time the true target is published, the test compares it with the latest
    position received from:
      - /future_target: your forecast
      - /target: the raw (delayed and noisy) observations
      - a moving average of the last MA_WINDOW observations on /target
      - /filtered_target: your Kalman filter from Part III
    and checks that your forecast is much closer to the true target than the three baselines.
    """

    @classmethod
    def setUpClass(cls):
        """Initialize ROS for the test class."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shutdown ROS after the test class."""
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node(NAME, parameter_overrides=[Parameter('use_sim_time', value=True)])

        self.latest_future = None           # latest position on /future_target
        self.latest_filtered = None         # latest position on /filtered_target
        self.latest_raw = None              # latest position on /target
        self.recent_raw = []                # last MA_WINDOW positions on /target
        self.errors = None                  # errors for each estimator per true target sample, once we are collecting
        self.num_true_msgs = 0              # messages received on /true_target while collecting errors
        self.num_skipped = 0                # true target samples skipped during the warm-up

    def tearDown(self):
        """Destroy the node after each test."""
        self.node.destroy_node()

    def _future_callback(self, msg):
        self.latest_future = position(msg)

    def _filtered_callback(self, msg):
        self.latest_filtered = position(msg)

    def _target_callback(self, msg):
        self.latest_raw = position(msg.pose)
        self.recent_raw.append(self.latest_raw)
        if len(self.recent_raw) > MA_WINDOW:
            self.recent_raw.pop(0)

    def _true_callback(self, msg):
        if self.errors is None:
            return
        self.num_true_msgs += 1
        if self.latest_future is None or self.latest_filtered is None or self.latest_raw is None:
            return
        if self.num_skipped < WARMUP_SAMPLES:
            self.num_skipped += 1
            return

        true_pos = position(msg)
        self.errors['future'].append(np.linalg.norm(self.latest_future - true_pos))
        self.errors['raw'].append(np.linalg.norm(self.latest_raw - true_pos))
        self.errors['moving_average'].append(np.linalg.norm(np.mean(self.recent_raw, axis=0) - true_pos))
        self.errors['filtered'].append(np.linalg.norm(self.latest_filtered - true_pos))

    def test_future_quality(self):
        """
        Check that the forecast of the target's position is much better than the baselines
        """
        self.assertIsNotNone(FUTURE_FILTER_PATH,
                             "Could not find an executable future_kalman_filter.py in the shutter_kf package. "
                             "Did you create it in shutter_kf/scripts, make it executable (chmod +x) and rebuild "
                             "your workspace? See IV-1.")

        self.node.create_subscription(PoseStamped, "/future_target", self._future_callback, 5)
        self.node.create_subscription(PoseStamped, "/filtered_target", self._filtered_callback, 5)
        self.node.create_subscription(Target, "/target", self._target_callback, 5)
        self.node.create_subscription(PoseStamped, "/true_target", self._true_callback, 5)

        # Wait for the first output of both filters. The simulation takes ~20 seconds to come up and start
        # publishing /clock, and the target is only generated once the clock is running.
        print(f"[{NAME}] Waiting up to 90 s for /future_target and /filtered_target "
              f"(the simulation takes ~20 s to start)...", flush=True)
        timeout_t = time.time() + 90.0
        while rclpy.ok() and (self.latest_future is None or self.latest_filtered is None) and time.time() < timeout_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertIsNotNone(self.latest_future,
                             "Did not get any message on /future_target within 90 seconds. "
                             "Check the future_kalman_filter node's output for errors.")
        self.assertIsNotNone(self.latest_filtered,
                             "Did not get any message on /filtered_target within 90 seconds. "
                             "Check the kalman_filter node's output for errors (Part III).")

        # compare the estimates with the true target for 30 seconds
        print(f"[{NAME}] Got the first outputs of both filters. Skipping the first {WARMUP_SAMPLES} true target "
              f"samples (~{WARMUP_SAMPLES / 30:.0f} s) while the filters settle, then comparing for 30 s...", flush=True)
        self.errors = {'future': [], 'raw': [], 'moving_average': [], 'filtered': []}
        timeout_t = time.time() + 30.0
        while rclpy.ok() and time.time() < timeout_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # every evaluated sample adds one error per estimator, so all the lists in self.errors have the same length
        num_samples = len(self.errors['future'])
        if num_samples < MIN_SAMPLES:
            if self.latest_raw is None:
                cause = ("No observation was received on /target, so there was nothing to compare with. "
                         "Check that the generate_continuous_target node is running (ros2 topic hz /target).")
            else:
                cause = (f"Only {self.num_true_msgs} messages were received on /true_target in 30 secs "
                         f"({self.num_true_msgs / 30.0:.1f} Hz instead of ~30 Hz), and the first {WARMUP_SAMPLES} "
                         f"are skipped as warm-up. The target is generated on the simulation clock, so the "
                         f"simulation is probably running much slower than real time (e.g., because the computer "
                         f"is overloaded), or the generate_continuous_target node stopped. Check the nodes' output "
                         f"for errors. While the launch file runs, `ros2 topic hz /true_target` should report ~30 Hz.")
            self.fail(f"Evaluated only {num_samples} true target samples, but at least {MIN_SAMPLES} are needed. {cause}")

        err = {k: float(np.mean(v)) for k, v in self.errors.items()}
        summary = (f"Avg. error over {num_samples} samples: /future_target={err['future']:.4f}, "
                   f"raw /target={err['raw']:.4f}, moving average={err['moving_average']:.4f}, "
                   f"/filtered_target={err['filtered']:.4f} (forecast / raw ratio={err['future'] / err['raw']:.3f}).")
        print(f"[{NAME}] {summary}", flush=True)

        self.assertLessEqual(err['future'], RAW_RATIO * err['raw'],
            f"The forecast's error should be at most {RAW_RATIO} times the error of the raw observations. {summary}")
        self.assertLess(err['future'], err['moving_average'],
            f"The forecast's error should be lower than the error of a moving average of the observations. {summary}")
        self.assertLess(err['future'], err['filtered'],
            f"The forecast's error should be lower than the error of /filtered_target. {summary}")
