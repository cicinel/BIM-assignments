#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
import rclpy.logging
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from shutter_lookat.msg import Target

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

def project_3D_point(x, y, z, K):
    """
    Project 3D point [x, y, z]^T in camera frame onto the camera image using the intrinsic parameters K
    :param x: x coordinate of the 3D point (in the camera frame)
    :param y: y coordinate of the 3D point (in the camera frame)
    :param z: z coordinate of the 3D point (in the camera frame)
    :param K: 3x3 numpy array with intrinsic camera parameters
    :return: tuple with 2D coordinates of the projected 3D point on the camera plane
    """
    p = K @ np.array([x, y, z])
    u = p[0] / p[2]
    v = p[1] / p[2]
    return u, v

def draw_image(x, y, z, K, width, height, **kwargs):
    """
    Project the target and create a virtual camera image
    :param x: x coordinate of the 3D point (in the camera frame)
    :param y: y coordinate of the 3D point (in the camera frame)
    :param z: z coordinate of the 3D point (in the camera frame)
    :param K: 3x3 numpy array with intrinsic camera parameters
    :param width: desired image width
    :param height: desired image height
    :param kwargs: extra parameters for the function (used in Part IV of the assignment)
    :return: color image as a numpy array. The image should have dimensions height x width x 3 and be of type uint8.
    """
    # NOTE: You should project the position of the moving object on the camera image using
    # the project_3D_point() function. You need to complete it as part of this assignment.

    img = np.zeros((height, width, 3), np.uint8)
    img[:, :] = (255, 255, 255)

    if z <= 0:
        rclpy.logging.get_logger('virtual_camera').warning('Target is behind the camera')
        return img

    u, v = project_3D_point(x, y, z, K)
    cv2.circle(img, (int(round(u)), int(round(v))), 12, (0, 0, 255), 3)
    return img

class VirtualCameraNode(Node):
    """
    ROS 2 node for Part III for the assignment 1. Renders an image from a virtual camera.
    """

    def __init__(self):
        super().__init__('virtual_camera')

        self.f = 'shutter_camera_color_optical_frame'
        self.K = np.array([[349.0, 0.0, 320.0],
                           [0.0, 349.0, 240.0],
                           [0.0, 0.0, 1.0]])
        self.buf = tf2_ros.Buffer(node=self)
        self.listener = tf2_ros.TransformListener(self.buf, self)
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/virtual_camera/image_raw', 10)
        self.pub2 = self.create_publisher(CameraInfo, '/virtual_camera/camera_info', 10)

        self.info = CameraInfo()
        self.info.width = 640
        self.info.height = 480
        self.info.k = [349.0, 0.0, 320.0, 0.0, 349.0, 240.0, 0.0, 0.0, 1.0]
        self.info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [349.0, 0.0, 320.0, 0.0,
                       0.0, 349.0, 240.0, 0.0,
                       0.0, 0.0, 1.0, 0.0]
        self.info.distortion_model = 'plumb_bob'


        # finally, start receiving target messages in your node...
        self.create_subscription(Target, 'target', self.target_callback, 5)

    def target_callback(self, target_msg):
        """
        Target callback
        :param target_msg: target message
        """

        p = target_msg.pose
        try:
            t = self.buf.lookup_transform(
                self.f, p.header.frame_id, Time.from_msg(p.header.stamp),
                timeout=Duration(seconds=0.2))
        except tf2_ros.TransformException:
            self.get_logger().warning('Waiting for the camera transform')
            return

        p2 = tf2_geometry_msgs.do_transform_pose(p.pose, t).position
        img = draw_image(p2.x, p2.y, p2.z, self.K, 640, 480)

        m = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.f
        self.pub.publish(m)

        self.info.header = m.header
        self.pub2.publish(self.info)


def main(args=None):
    rclpy.init(args=args)
    n = VirtualCameraNode()
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(n)
    ex.spin()
    ex.shutdown()
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
