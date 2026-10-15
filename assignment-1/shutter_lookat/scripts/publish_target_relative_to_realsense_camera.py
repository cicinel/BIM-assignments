#!/usr/bin/env python3

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import TransformStamped
from shutter_lookat.msg import Target
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


class PublishTargetRelativeToRealsenseCamera(Node):
    def __init__(self):
        super().__init__('publish_target_relative_to_realsense_camera')

        self.camera_frame = 'shutter_camera_color_optical_frame'
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.subscription = self.create_subscription(Target, '/target', self.target_callback, 10)

    def target_callback(self, msg):
        p = msg.pose
     
        try:
            t = self.tf_buffer.lookup_transform(
                self.camera_frame,
                p.header.frame_id,
                Time.from_msg(p.header.stamp),
                timeout=Duration(seconds=0.2),
            )
        except TransformException:
            self.get_logger().warning('Waiting for the camera transform')
            return

        p2 = do_transform_pose(p.pose, t)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = p.header.stamp
        tf_msg.header.frame_id = self.camera_frame
        tf_msg.child_frame_id = 'target'
        tf_msg.transform.translation.x = p2.position.x
        tf_msg.transform.translation.y = p2.position.y
        tf_msg.transform.translation.z = p2.position.z
        tf_msg.transform.rotation = p2.orientation
        self.tf_broadcaster.sendTransform(tf_msg)


def main():
    rclpy.init()
    n = PublishTargetRelativeToRealsenseCamera()
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(n)
    ex.spin()

    ex.shutdown()
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
