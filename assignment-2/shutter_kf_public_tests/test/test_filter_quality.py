#!/usr/bin/env python3
# Public tests for CPSC-4590/5590 Assignment 2 - Part III

PKG = "shutter_kf_public_tests"
NAME = 'test_filter_quality'

import os
import unittest
import numpy as np
import time

import rclpy
from rclpy.parameter import Parameter
import message_filters
from geometry_msgs.msg import PoseStamped

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_testing.actions import ReadyToTest


def generate_test_description():
    """
    Bring up follow_target.launch.py: the simulation, the moving target, your Kalman filter and the expert
    that makes the robot follow the filtered target. The path type comes from the PATH_TYPE environment
    variable (horizontal, vertical or circular) because `colcon test` has no way to pass launch arguments
    to a test.
    """

    # Declare launch arguments
    add_noise_arg = DeclareLaunchArgument(
        'add_noise',
        default_value='true',
        description='Add noise to target observations'
    )

    path_type_arg = DeclareLaunchArgument(
        'path_type',
        default_value=os.environ.get('PATH_TYPE', 'horizontal'),
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
            'add_noise': LaunchConfiguration('add_noise'),
            'path_type': LaunchConfiguration('path_type'),
            'headless': 'true',
            'face': 'false',
            'rviz': 'false',
        }.items()
    )

    return (
        LaunchDescription([
            add_noise_arg,
            path_type_arg,
            follow_target_launch,
            # launch_testing gives the launch description 15 seconds to signal that it is ready,
            # so this cannot be used to wait out the simulation's startup -- the test below
            # waits for the filter's output instead.
            TimerAction(
                period=2.0,
                actions=[ReadyToTest()]
            ),
        ]), {}
    )


class TestFilterQuality(unittest.TestCase):
    """
    Public tests for kalman_filter.py
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
        self.filtered_topic = "/filtered_target"    # filtered topic
        self.gt_topic = "/true_target"              # true topic
        self.msg_list = []
        self.got_filtered_msg = False

        self.node = rclpy.create_node(NAME, parameter_overrides=[Parameter('use_sim_time', value=True)])

    def tearDown(self):
        """Destroy the node after each test."""
        self.node.destroy_node()

    def _callback(self, filtered_msg, gt_msg):
        self.msg_list.append((filtered_msg, gt_msg))

    def _filtered_callback(self, msg):
        self.got_filtered_msg = True

    def test_quality(self):
        """
        Check that the filter quality meets the requirements
        """
        # Wait for the filter's first output. The simulation takes ~20 seconds to come up and start
        # publishing /clock, and the target is only generated once the clock is running.
        self.node.create_subscription(PoseStamped, self.filtered_topic, self._filtered_callback, 5)
        print(f"[{NAME}] Waiting up to 90 s for {self.filtered_topic} (the simulation takes ~20 s to start)...", flush=True)
        timeout_t = time.time() + 90.0
        while rclpy.ok() and not self.got_filtered_msg and time.time() < timeout_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertTrue(self.got_filtered_msg,
                        f"Did not get any message on {self.filtered_topic} within 90 seconds. "
                        f"Check the kalman_filter node's output for errors.")

        self.msg_list = []

        sub1 = message_filters.Subscriber(self.node, PoseStamped, self.filtered_topic)
        sub2 = message_filters.Subscriber(self.node, PoseStamped, self.gt_topic)
        subs = [sub1, sub2]
        ts = message_filters.ApproximateTimeSynchronizer(subs, 10, 0.25)
        ts.registerCallback(self._callback)

        print(f"[{NAME}] Got the first output of the filter. Comparing it with the true target for 30 s...", flush=True)
        timeout_t = time.time() + 30  # 30 seconds in the future

        # collect synchronized messages
        while rclpy.ok() and time.time() < timeout_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # did we succeed in getting messages?
        self.assertGreaterEqual(len(self.msg_list), 10,
            f"Got less than 10 synchronized messages in 30 secs (num. messages={len(self.msg_list)}).")
        print(f"[{NAME}] Got {len(self.msg_list)} synchronized messages.", flush=True)

        # compute error
        err = []
        for i in range(len(self.msg_list)):
            m1 = self.msg_list[i][0]
            m2 = self.msg_list[i][1]
            diffx = m1.pose.position.x - m2.pose.position.x
            diffy = m1.pose.position.y - m2.pose.position.y
            diffz = m1.pose.position.z - m2.pose.position.z
            l2_err = np.sqrt(diffx*diffx + diffy*diffy + diffz*diffz)
            err.append(l2_err)

        avg_err = np.mean(err)
        std_err = np.std(err)
        print(f"[{NAME}] Avg err={avg_err}", flush=True)
        print(f"[{NAME}] Std err={std_err}", flush=True)

        self.assertLessEqual(avg_err, 0.04,
            f"The average error {avg_err} (+- {std_err}) was greater than 0.04.")
