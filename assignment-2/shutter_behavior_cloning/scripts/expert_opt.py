#!/usr/bin/env python3

import os
from datetime import datetime
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
import tf2_ros
import tf2_geometry_msgs
from urdf_parser_py.urdf import URDF

from look_at_solver import transform_msg_to_T, get_joint, solve_look_at
from servo_joint_client import ServoJointClient


class ExpertNode(Node):
    """
    Node that simulates an expert controller using optimization. It controls two joints of the robot to make it
    point towards a target.
    """

    def __init__(self, name='expert_opt'):
        super().__init__(name)

        # params
        # Robot id. All of Shutter's links and frames are named ${arm_id}_<name>.
        self.declare_parameter("arm_id", "shutter")
        self.declare_parameter("save_state_actions", True)
        self.declare_parameter("output_file", "")

        arm_id = self.get_parameter("arm_id").value
        self.base_link = arm_id + "_base_link"
        self.biceps_link = arm_id + "_biceps_link"
        self.camera_link = arm_id + "_camera_color_optical_frame"
        self.save_state_actions = self.get_parameter("save_state_actions").value

        if self.save_state_actions:
            # Get package path using ament
            import ament_index_python
            try:
                bc_dir = ament_index_python.get_package_share_directory("shutter_behavior_cloning")
                default_output_file = os.path.join(bc_dir, "data", "state_actions.txt")
            except:
                # Fallback to current directory if package not found
                default_output_file = os.path.join(os.getcwd(), "data", "state_actions.txt")

            output_file_param = self.get_parameter("output_file").value
            self.output_file = output_file_param if output_file_param else default_output_file

            base_dir = os.path.dirname(self.output_file)
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)

            self.fid = open(self.output_file, 'a')  # open output buffer to record state-action pairs...
            date = datetime.now()
            self.fid.write("# data from {}\n".format(date.strftime("%d/%m/%Y %H:%M:%S")))
            self.get_logger().info("Saving state-action pairs to {}".format(self.output_file))

        else:
            self.fid = None

        # joint values
        self.move = True
        self.current_pose = None  # [joint_1, joint_2, joint_3, joint_4]

        # get robot model
        self.parameter_client = AsyncParameterClient(self, "/robot_state_publisher")
        while not self.parameter_client.wait_for_services(timeout_sec=5.0):
            self.get_logger().info("Waiting for /robot_state_publisher to get the robot model...")
        future = self.parameter_client.get_parameters(['robot_description'])
        rclpy.spin_until_future_complete(self, future)

        urdf_xml = future.result().values[0].string_value
        self.robot = URDF.from_xml_string(urdf_xml)
        self.joint1_model = get_joint(self.robot, 'joint_1')
        self.joint3_model = get_joint(self.robot, 'joint_3')

        # joint commands go to MoveIt Servo
        self.joint_client = ServoJointClient(self)
        # copy of the joint commands, for testing and debugging
        self.command_pub = self.create_publisher(Float64MultiArray, "/expert_joint_command", 5)

        # tf subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # joint subscriber
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joints_callback, 5)
        self.target_sub = self.create_subscription(PoseStamped, '/target', self.target_callback, 5)
        self.move_sub = self.create_subscription(Bool, '/move_towards_target', self.move_callback, 5)

    def cleanup(self):
        """
        Be good with the environment.
        """
        if self.fid is not None:
            self.fid.close()

    def move_callback(self, msg):
        """
        Move on/off callback
        """
        self.move = msg.data
        self.get_logger().warn("Motion is: {}".format(self.move))

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

    def get_p_T1_T2(self, msg):
        """
        Helper function for compute_joints_position()
        :param msg: target message
        :return: target in baselink, transform from base_link to biceps, transform from biceps to camera
        """

        # transform the target to baselink if it's not in that frame already
        if msg.header.frame_id != self.base_link:
            try:
                transform = self.tf_buffer.lookup_transform(self.base_link,
                                                            msg.header.frame_id,  # source frame
                                                            rclpy.time.Time(),
                                                            timeout=rclpy.duration.Duration(seconds=1.0))
                pose_transformed = tf2_geometry_msgs.do_transform_pose_stamped(msg, transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn("Failed to compute new position for the robot because {}".format(e))
                return None, None, None
        else:
            pose_transformed = msg

        p = [pose_transformed.pose.position.x,
             pose_transformed.pose.position.y,
             pose_transformed.pose.position.z]

        # get transform from base link to camera link (base_link -> biceps_link and biceps_link -> camera_link)
        try:
            transform = self.tf_buffer.lookup_transform(self.biceps_link,
                                                        self.base_link,  # source frame
                                                        rclpy.time.Time(),
                                                        timeout=rclpy.duration.Duration(seconds=1.0))
            T1 = transform_msg_to_T(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn("Failed to compute new position for the robot because {}".format(e))
            T1 = None

        try:
            transform = self.tf_buffer.lookup_transform(self.camera_link,
                                                        self.biceps_link,  # source frame
                                                        rclpy.time.Time(),
                                                        timeout=rclpy.duration.Duration(seconds=1.0))
            T2 = transform_msg_to_T(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn("Failed to compute new position for the robot because {}".format(e))
            T2 = None

        return p, T1, T2

    def compute_joints_position(self, msg, joint1, joint3):
        """
        Helper function to compute the required motion to make the robot's camera look towards the target
        :param msg: target message
        :param joint1: current joint 1 position
        :param joint3: current joint 3 position
        :return: new joint positions for joint1 and joint3; or None if something went wrong
        """
        p, T1, T2 = self.get_p_T1_T2(msg)
        if p is None or T1 is None or T2 is None:
            return None

        return solve_look_at(p, T1, T2, self.joint1_model, self.joint3_model, joint1, joint3)

    def target_callback(self, msg):
        """
        Target callback
        :param msg: target message
        """
        if self.current_pose is None:
            self.get_logger().warn("Joint positions are unknown. Waiting to receive joint states.")
            return

        if not self.move:
            return

        # compute the required motion to make the robot look towards the target
        joint3 = self.current_pose[2]
        joint1 = self.current_pose[0]
        joint_angles = self.compute_joints_position(msg, joint1, joint3)
        if joint_angles is None:
            # we are done. the node might not be ready yet...
            return
        else:
            # upack results
            new_j1, new_j3 = joint_angles

        # write state and action (offset motion) to disk
        if self.fid is not None:
            self.fid.write("%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" %
                           (msg.header.frame_id, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                            joint1, joint3, new_j1, new_j3))
            self.fid.flush()

        # command the robot
        command = [float(new_j1), 0.0, float(new_j3), 0.0]
        self.joint_client.send_joint_goal(command)
        self.command_pub.publish(Float64MultiArray(data=command))


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = ExpertNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
