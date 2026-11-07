#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


# default volume for targets
dflt_x_min = 0.5
dflt_x_max = 3.5
dflt_y_min = -3.0
dflt_y_max = 3.0
dflt_z_min = 0.0
dflt_z_max = 3.0


def random_pose(x_min, x_max, y_min, y_max, z_min, z_max, node, frame_id):
    """
    Generate random target within a 3D space
    :param x_min: minimum x
    :param x_max: maximum x
    :param y_min: minimum y
    :param y_max: maximum y
    :param z_min: minimum z
    :param z_max: maximum z
    :param node: ROS 2 node for getting current time
    :param frame_id: frame in which the target is generated (e.g., shutter_base_footprint)
    :return PoseStamped with random pose
    """
    pose_msg = PoseStamped()
    pose_msg.header.stamp = node.get_clock().now().to_msg()
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.position.x = np.random.uniform(low=x_min, high=x_max, size=None)
    pose_msg.pose.position.y = np.random.uniform(low=y_min, high=y_max, size=None)
    pose_msg.pose.position.z = np.random.uniform(low=z_min, high=z_max, size=None)
    pose_msg.pose.orientation.w = 1.0
    return pose_msg


class GenerateRandomTargetNode(Node):
    """
    Node that publishes random targets at a constant frame rate.
    """

    def __init__(self):
        super().__init__('generate_random_target')

        # Get ROS params
        # Robot id. All of Shutter's links and frames are named ${arm_id}_<name>,
        # so the targets are published in ${arm_id}_base_footprint.
        self.declare_parameter("arm_id", "shutter")
        self.declare_parameter("x_min", dflt_x_min)
        self.declare_parameter("x_max", dflt_x_max)
        self.declare_parameter("y_min", dflt_y_min)
        self.declare_parameter("y_max", dflt_y_max)
        self.declare_parameter("z_min", dflt_z_min)
        self.declare_parameter("z_max", dflt_z_max)
        self.declare_parameter("publish_rate", 1.0)

        x_min = self.get_parameter("x_min").value
        x_max = self.get_parameter("x_max").value
        y_min = self.get_parameter("y_min").value
        y_max = self.get_parameter("y_max").value
        z_min = self.get_parameter("z_min").value
        z_max = self.get_parameter("z_max").value
        publish_rate = self.get_parameter("publish_rate").value
        self.frame_id = self.get_parameter("arm_id").value + "_base_footprint"

        # Define publishers
        self.target_pub = self.create_publisher(PoseStamped, '/target', 5)
        self.marker_pub = self.create_publisher(Marker, '/target_marker', 5)

        # local vars
        self.timestamp_buffer = None

        # Create timer for publishing
        self.timer = self.create_timer(1.0/publish_rate, self.publish_target)

        # Store parameters for use in callback
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def publish_target(self):
        """Publish target and marker"""
        # publish the location of the target as a PoseStamped
        pose_msg = random_pose(self.x_min, self.x_max, self.y_min, self.y_max, 
                              self.z_min, self.z_max, self, self.frame_id)

        # Check if current Time exceeds clock speed
        if self.timestamp_buffer is not None and self.timestamp_buffer >= pose_msg.header.stamp.sec:
            self.get_logger().warn('Publish rate exceeds clock speed; check your clock publish rate')
            return

        self.target_pub.publish(pose_msg)
        self.timestamp_buffer = pose_msg.header.stamp.sec

        # publish a marker to visualize the target in RViz
        marker_msg = Marker()
        marker_msg.header = pose_msg.header
        marker_msg.action = Marker.ADD
        marker_msg.color.a = 0.5
        marker_msg.color.b = 1.0
        marker_msg.lifetime.sec = 10
        marker_msg.lifetime.nanosec = 0
        marker_msg.id = 0
        marker_msg.ns = "target"
        marker_msg.type = Marker.SPHERE
        marker_msg.pose = pose_msg.pose
        marker_msg.scale.x = 0.1
        marker_msg.scale.y = 0.1
        marker_msg.scale.z = 0.1
        self.marker_pub.publish(marker_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = GenerateRandomTargetNode()
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
