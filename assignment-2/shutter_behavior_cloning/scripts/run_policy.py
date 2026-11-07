#!/usr/bin/env python3

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

import json
import torch
import numpy as np
import sys
import os

from servo_joint_client import ServoJointClient


class RunPolicyNode(Node):
    """
    Node that controls the robot to make it point towards a target.
    """

    def __init__(self):
        super().__init__('run_policy')

        # params
        self.declare_parameter("arm_id", "shutter")
        self.declare_parameter("model", "")
        self.declare_parameter("norm_params", "")

        # the targets are expected in the robot's base_footprint frame (e.g., shutter_base_footprint)
        self.target_frame = self.get_parameter("arm_id").value + "_base_footprint"
        self.model_file = self.get_parameter("model").value              # required path to model file
        self.normp_file = self.get_parameter("norm_params").value        # optional path to normalization parameters (empty str means no norm params)

        # TODO - complete the line below to load up your model and create any necessary class instance variables.
        # Load the model's weights with torch.load(self.model_file, weights_only=True), and the normalization
        # parameters (if any) with json.load().
        self.model = ...

        # joint values
        self.current_pose = None  # [joint_1, joint_2, joint_3, joint_4]

        # joint commands go to MoveIt Servo
        self.joint_client = ServoJointClient(self)
        # copy of the joint commands, for testing and debugging
        self.command_pub = self.create_publisher(Float64MultiArray, "/policy_joint_command", 5)

        # joint subscriber
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joints_callback, 5)
        self.target_sub = self.create_subscription(PoseStamped, '/target', self.target_callback, 5)

    def joints_callback(self, msg):
        """
        Joints callback
        :param msg: joint state
        """
        # /joint_states can report the joints in any order, so index them by name
        positions = dict(zip(msg.name, msg.position))
        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4']
        if not all(name in positions for name in joint_names):
            self.get_logger().error("Missing joints from joint state! Got {}".format(msg.name),
                                    throttle_duration_sec=5.0)
            return
        self.current_pose = [positions[name] for name in joint_names]

    def compute_joints_position(self, msg, joint1, joint3):
        """
        Helper function to compute the required motion to make the robot's camera look towards the target
        :param msg: target message that was received by the target callback
        :param joint1: current joint 1 position
        :param joint3: current joint 3 position
        :return: tuple with new joint positions for joint1 and joint3, or None if the computation failed
        """

        # TODO - remove None return statement and complete function with the logic that runs your model to compute new
        # joint positions 1 & 3 for the robot...
        return None

    def target_callback(self, msg):
        """
        Target callback
        :param msg: target message
        """
        if self.current_pose is None:
            self.get_logger().warn("Joint positions are unknown. Waiting to receive joint states.")
            return

        # check that the data is consistent with our model and that we have current joint information...
        if msg.header.frame_id != self.target_frame:
            self.get_logger().error("Expected the input target to be in the frame '{}' but got the {} frame instead. "
                                    "Failed to command the robot".format(self.target_frame, msg.header.frame_id))
            return

        # compute the required motion to make the robot look towards the target
        joint3 = self.current_pose[2]
        joint1 = self.current_pose[0]
        joint_positions = self.compute_joints_position(msg, joint1, joint3)
        if joint_positions is None:
            # we are done. we did not get a solution
            self.get_logger().warn("The compute_joints_position() function returned None. Failed to command the robot.")
            return
        else:
            # upack result
            new_j1, new_j3 = joint_positions

        # command the robot
        command = [float(new_j1), 0.0, float(new_j3), 0.0]
        self.joint_client.send_joint_goal(command)
        self.command_pub.publish(Float64MultiArray(data=command))


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = RunPolicyNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
