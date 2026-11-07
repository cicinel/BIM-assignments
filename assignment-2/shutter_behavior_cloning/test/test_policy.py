#!/usr/bin/env python3
# Public tests for CPSC-4590/5590 Assignment 2 - Part II
#
# Brings up the simulation, MoveIt Servo and your run_policy.py node, publishes 25 targets
# (the same 25 on every run), and compares the joint command that your policy sends for each
# target with the expert's solution for it. A trial is acceptable if both joint_1 and joint_3
# are within 1.5 degrees of the expert's solution.
#
# The results are written to the file given by the OUTPUT_FILE environment variable or, if it is
# not set, to ~/ros2_ws/test_policy_output.txt (or to the current directory if ~/ros2_ws does not
# exist, which is <workspace>/build/shutter_behavior_cloning under `colcon test`). The joint positions that the robot reached are also reported, but
# only for information: the trials are graded on the commanded joints.
#
# Usage:
#   MODEL=<path_to_model> NORMP=<path_to_norm_params> colcon test --packages-select shutter_behavior_cloning \
#       --event-handlers console_direct+ --ctest-args -R test_policy
#
# Set RUN_EXPERT=true instead of MODEL and NORMP to run the test with the expert as the policy.

PKG = "shutter_behavior_cloning"
NAME = 'test_policy'

import os
import time
import unittest
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
import tf2_ros
import tf2_geometry_msgs
from urdf_parser_py.urdf import URDF

# The CMakeLists.txt puts this package's scripts directory on the PYTHONPATH for the test
from generate_random_target import dflt_x_min, dflt_x_max, dflt_y_min, dflt_y_max, dflt_z_min, dflt_z_max
from look_at_solver import transform_msg_to_T, get_joint, solve_look_at

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, RegisterEventHandler, \
    TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_testing.actions import ReadyToTest


def generate_test_description():
    # The policy's files, and whether to test the expert instead, come from environment variables
    # because `colcon test` has no way to pass launch arguments to a test.
    run_expert_arg = DeclareLaunchArgument(
        'run_expert',
        default_value=os.environ.get('RUN_EXPERT', 'false'),
        description='Run the expert instead of the learned policy'
    )

    model_arg = DeclareLaunchArgument(
        'model',
        default_value=os.environ.get('MODEL', ''),
        description='Path to model file'
    )

    normp_arg = DeclareLaunchArgument(
        'normp',
        default_value=os.environ.get('NORMP', ''),
        description='Path to normalization parameters file'
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Start RViz to watch the test'
    )

    # Bring up the MuJoCo simulation without the MuJoCo window or the face, so that the test can
    # run over ssh. MuJoCo owns the clock, which is why every node started here sets use_sim_time.
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('shutter_lookat'),
                'launch',
                'start_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'headless': 'true',
            'face': 'false',
            'rviz': LaunchConfiguration('rviz'),
            'rviz_config': PathJoinSubstitution([
                FindPackageShare('shutter_lookat'),
                'config',
                'lookat-target.rviz'
            ]),
        }.items()
    )

    # MoveIt Servo, started once the simulation's controllers are ready (as in collect_data.launch.py)
    wait_for_controllers_node = Node(
        package='shutter_behavior_cloning',
        executable='wait_for_controllers.py',
        name='wait_for_controllers',
        output='screen'
    )

    servo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('shutter_servo'),
                'launch',
                'shutter_servo.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': 'true',
        }.items()
    )

    start_servo = RegisterEventHandler(OnProcessExit(
        target_action=wait_for_controllers_node,
        on_exit=lambda event, context: [servo_launch] if event.returncode == 0 else
        [LogInfo(msg="MoveIt Servo was not started because the simulation's controllers were not ready")]
    ))

    # Expert, which stands in for the policy when run_expert is true. Its copy of the joint
    # commands is remapped to the topic that the test listens to.
    expert_opt_node = Node(
        package='shutter_behavior_cloning',
        executable='expert_opt.py',
        name='expert_opt',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'save_state_actions': False
        }],
        remappings=[
            ('/expert_joint_command', '/policy_joint_command')
        ],
        condition=IfCondition(LaunchConfiguration('run_expert'))
    )

    # Student's code node
    run_policy_node = Node(
        package='shutter_behavior_cloning',
        executable='run_policy.py',
        name='run_policy',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'model': LaunchConfiguration('model'),
            'norm_params': LaunchConfiguration('normp')
        }],
        condition=UnlessCondition(LaunchConfiguration('run_expert'))
    )

    return (
        LaunchDescription([
            run_expert_arg,
            model_arg,
            normp_arg,
            rviz_arg,
            sim_launch,
            wait_for_controllers_node,
            start_servo,
            expert_opt_node,
            run_policy_node,
            # launch_testing gives the launch description 15 seconds to signal that it is ready,
            # so this cannot be used to wait out the simulation's startup -- the test below
            # polls for what it needs instead.
            TimerAction(
                period=2.0,
                actions=[ReadyToTest()]
            ),
        ]), {}
    )


def make_pose(x, y, z, node, frame_id):
    """
    Make a target pose
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param node: ROS 2 node for getting current time
    :param frame_id: frame id
    :return PoseStamped with (x,y,z) position
    """
    pose_msg = PoseStamped()
    pose_msg.header.stamp = node.get_clock().now().to_msg()
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.position.x = x
    pose_msg.pose.position.y = y
    pose_msg.pose.position.z = z
    pose_msg.pose.orientation.w = 1.0
    return pose_msg


def min_angle_diff(a1, a2):
    """
    Compute minimum angle difference (a1 - a2)
    :param a1: angle (radians)
    :param a2: angle (radians)
    :return: a1 - a2, in [-pi, pi]
    """
    d = a1 - a2
    return np.arctan2(np.sin(d), np.cos(d))


class TestPolicy(unittest.TestCase):
    """
    Public tests for run_policy.py
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
        """Create a node for each test."""
        self.node = rclpy.create_node(NAME, parameter_overrides=[Parameter('use_sim_time', value=True)])

        arm_id = "shutter"
        self.target_frame = arm_id + "_base_footprint"     # frame of the targets
        self.base_link = arm_id + "_base_link"
        self.biceps_link = arm_id + "_biceps_link"
        self.camera_link = arm_id + "_camera_color_optical_frame"

        self.command_timeout = 10.0     # seconds (wall clock) to wait for the policy's command for a target
        self.settle_timeout = 5.0       # seconds (wall clock) to wait for the robot to stop moving

        # OUTPUT_FILE overrides where the results are written. Otherwise, they go to ~/ros2_ws or, if that
        # does not exist, to the current directory (<workspace>/build/<package> under `colcon test`).
        ws_dir = os.path.expanduser("~/ros2_ws")
        default_output = os.path.join(ws_dir if os.path.isdir(ws_dir) else os.getcwd(), "test_policy_output.txt")
        self.output_file = os.path.expanduser(os.environ.get('OUTPUT_FILE', '')) or default_output

        # make positions repeatable
        np.random.seed(0)

        # generate samples
        num_samples = 25
        self.positions = np.column_stack((np.random.uniform(low=dflt_x_min, high=dflt_x_max, size=(num_samples, 1)),
                                          np.random.uniform(low=dflt_y_min, high=dflt_y_max, size=(num_samples, 1)),
                                          np.random.uniform(low=dflt_z_min, high=dflt_z_max, size=(num_samples, 1))))

        # latest joint positions and policy commands
        self.joints = None              # dict from joint name to position
        self.commands = []              # joint commands received on /policy_joint_command

        self.node.create_subscription(JointState, '/joint_states', self._joints_callback, 5)
        self.node.create_subscription(Float64MultiArray, '/policy_joint_command', self._command_callback, 5)
        self.target_pub = self.node.create_publisher(PoseStamped, '/target', 5)
        self.marker_pub = self.node.create_publisher(Marker, '/target_marker', 5)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

    def tearDown(self):
        """Destroy the node after each test."""
        self.node.destroy_node()

    def _joints_callback(self, msg):
        """
        Joints callback
        :param msg: joint state
        """
        self.joints = dict(zip(msg.name, msg.position))

    def _command_callback(self, msg):
        """
        Policy command callback
        :param msg: joint command [joint_1, joint_2, joint_3, joint_4]
        """
        self.commands.append(list(msg.data))

    def spin_for(self, duration):
        """
        Process callbacks for some time
        :param duration: seconds (wall clock)
        """
        end_t = time.time() + duration
        while rclpy.ok() and time.time() < end_t:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def get_robot_model(self, timeout):
        """
        Get joint_1 and joint_3 from the URDF that robot_state_publisher is publishing
        :param timeout: seconds (wall clock)
        :return: URDF joints for joint_1 and joint_3, or None if the robot model was not available
        """
        client = AsyncParameterClient(self.node, "/robot_state_publisher")
        end_t = time.time() + timeout
        while rclpy.ok() and time.time() < end_t:
            if not client.wait_for_services(timeout_sec=1.0):
                continue
            future = client.get_parameters(['robot_description'])
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
            if future.result() is not None and future.result().values[0].string_value:
                robot = URDF.from_xml_string(future.result().values[0].string_value)
                return get_joint(robot, 'joint_1'), get_joint(robot, 'joint_3')
        return None

    def lookup(self, target_frame, source_frame):
        """
        Get the latest transform between two frames
        :return: TransformStamped, or None if it is not available
        """
        try:
            return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(),
                                                   timeout=Duration(seconds=0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    def wait_for_system(self, timeout):
        """
        Wait until the simulation, MoveIt Servo and the policy are up
        :param timeout: seconds (wall clock)
        :return: None if everything is up, or a string that describes what is missing
        """
        missing = "Timed out."
        end_t = time.time() + timeout
        while rclpy.ok() and time.time() < end_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.joints is None:
                missing = "No joint states were received on /joint_states."
            elif self.lookup(self.base_link, self.target_frame) is None or \
                    self.lookup(self.biceps_link, self.base_link) is None or \
                    self.lookup(self.camera_link, self.biceps_link) is None:
                missing = "The robot's transforms are not in the tf tree."
            elif self.node.count_subscribers('/servo/joint_command') == 0:
                missing = "MoveIt Servo (shutter_servo) did not start."
            elif self.node.count_subscribers('/target') == 0:
                missing = "The run_policy node did not subscribe to /target. Check that it started without errors."
            else:
                return None
        return missing

    def wait_until_still(self, threshold=1e-3, interval=0.5):
        """
        Wait until joint_1 and joint_3 stop moving, or for at most self.settle_timeout seconds
        :param threshold: maximum change in radians over the interval
        :param interval: seconds (wall clock)
        :return: joint_1 and joint_3 positions
        """
        end_t = time.time() + self.settle_timeout
        prev = (self.joints['joint_1'], self.joints['joint_3'])
        while rclpy.ok() and time.time() < end_t:
            self.spin_for(interval)
            current = (self.joints['joint_1'], self.joints['joint_3'])
            if all(np.fabs(min_angle_diff(c, p)) < threshold for c, p in zip(current, prev)):
                break
            prev = current
        return self.joints['joint_1'], self.joints['joint_3']

    def expert_solution(self, pose_msg, joint1, joint3):
        """
        Compute the expert's joint_1 and joint_3 positions for a target
        :param pose_msg: target in the target_frame
        :param joint1: current joint_1 position
        :param joint3: current joint_3 position
        :return: joint_1 and joint_3 positions, or None if the robot's transforms were not available
        """
        transforms = [self.lookup(self.base_link, pose_msg.header.frame_id),
                      self.lookup(self.biceps_link, self.base_link),
                      self.lookup(self.camera_link, self.biceps_link)]
        if any(t is None for t in transforms):
            return None
        pose_in_base = tf2_geometry_msgs.do_transform_pose_stamped(pose_msg, transforms[0])
        p = [pose_in_base.pose.position.x, pose_in_base.pose.position.y, pose_in_base.pose.position.z]
        T1 = transform_msg_to_T(transforms[1])
        T2 = transform_msg_to_T(transforms[2])
        return solve_look_at(p, T1, T2, self.joint1_model, self.joint3_model, joint1, joint3)

    def publish_marker(self, pose_msg):
        """
        Publish a marker to visualize the target in RViz
        :param pose_msg: target
        """
        marker_msg = Marker()
        marker_msg.header = pose_msg.header
        marker_msg.action = Marker.ADD
        marker_msg.color.a = 0.5
        marker_msg.color.b = 1.0
        marker_msg.lifetime.sec = 10
        marker_msg.id = 0
        marker_msg.ns = "target"
        marker_msg.type = Marker.SPHERE
        marker_msg.pose = pose_msg.pose
        marker_msg.scale.x = 0.1
        marker_msg.scale.y = 0.1
        marker_msg.scale.z = 0.1
        self.marker_pub.publish(marker_msg)

    def run_trial(self, x, y, z):
        """
        Publish a target and get the policy's command for it
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :return: expert's (j1, j3), policy's (j1, j3) and reached (j1, j3); the last two are None if the
                 policy did not send a command
        """
        # start from rest, so that the joint positions that the policy reads are the ones used here
        joint1, joint3 = self.wait_until_still()

        pose_msg = make_pose(x, y, z, self.node, self.target_frame)
        expert = self.expert_solution(pose_msg, joint1, joint3)
        self.assertIsNotNone(expert, "Failed to compute the expert's solution for target {}: the robot's transforms "
                                     "are not available.".format((x, y, z)))

        num_commands = len(self.commands)
        self.target_pub.publish(pose_msg)
        self.publish_marker(pose_msg)

        end_t = time.time() + self.command_timeout
        while rclpy.ok() and len(self.commands) == num_commands and time.time() < end_t:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if len(self.commands) == num_commands:
            return expert, None, None

        command = self.commands[num_commands]
        policy = (command[0], command[2])
        reached = self.wait_until_still()
        return expert, policy, reached

    def test_reaching_targets(self, max_ang_difference=np.deg2rad(1.5)):
        """
        Test the policy's commands for the targets
        :param max_ang_difference: maximum angular difference to consider a trial acceptable (in radians)
        """
        models = self.get_robot_model(timeout=120.0)
        self.assertIsNotNone(models, "Failed to get the robot model from /robot_state_publisher.")
        self.joint1_model, self.joint3_model = models

        # the simulation takes ~20 seconds to come up, and loading a model can take a while too
        missing = self.wait_for_system(timeout=120.0)
        self.assertIsNone(missing, "The system did not come up: {}".format(missing))

        headers = ["TRIAL", "X", "Y", "Z", "EXPERTJ1", "EXPERTJ3", "POLICYJ1", "POLICYJ3",
                   "DIFFJ1", "DIFFJ3", "ACCEPTABLE", "REACHEDJ1", "REACHEDJ3"]
        num_acceptable = 0
        no_command = []
        with open(self.output_file, 'w') as fid:
            fid.write("".join([h.ljust(12) for h in headers]) + "\n")

            for i in range(self.positions.shape[0]):
                p = self.positions[i]
                expert, policy, reached = self.run_trial(p[0], p[1], p[2])
                trial = "Trial {}/{}: target at ({:.3f}, {:.3f}, {:.3f})".format(
                    i + 1, self.positions.shape[0], p[0], p[1], p[2])

                if policy is None:
                    self.node.get_logger().warn(trial + " -> no command received")
                    no_command.append(i)
                    row = ["%4d" % i, "%.3f" % p[0], "%.3f" % p[1], "%.3f" % p[2],
                           "%.5f" % expert[0], "%.5f" % expert[1]] + ["-"] * 4 + ["0"] + ["-"] * 2
                else:
                    d1 = np.fabs(min_angle_diff(expert[0], policy[0]))
                    d3 = np.fabs(min_angle_diff(expert[1], policy[1]))
                    acceptable = int(d1 < max_ang_difference and d3 < max_ang_difference)
                    num_acceptable += acceptable
                    if acceptable:
                        self.node.get_logger().info(trial + " -> acceptable")
                    else:
                        self.node.get_logger().warn(trial + " -> not acceptable (DIFFJ1 = {:.5f}, DIFFJ3 = {:.5f} "
                                                            "radians)".format(d1, d3))
                    row = ["%4d" % i, "%.3f" % p[0], "%.3f" % p[1], "%.3f" % p[2],
                           "%.5f" % expert[0], "%.5f" % expert[1], "%.5f" % policy[0], "%.5f" % policy[1],
                           "%.5f" % d1, "%.5f" % d3, "%1d" % acceptable, "%.5f" % reached[0], "%.5f" % reached[1]]

                fid.write("".join([x.ljust(12) for x in row]) + "\n")
                fid.flush()

                # don't wait for the remaining trials if the policy is not sending any commands at all
                if len(no_command) == 3 and len(self.commands) == 0:
                    break

            summary = "{} of {} trials were acceptable".format(num_acceptable, self.positions.shape[0])
            fid.write("# " + summary + "\n")

        self.node.get_logger().info(summary + ". Wrote the results to " + self.output_file)
        self.assertEqual(len(no_command), 0,
                         "The policy did not send a command on /policy_joint_command within {} seconds for trials {}. "
                         "Check the run_policy node's output for errors.".format(self.command_timeout, no_command))
