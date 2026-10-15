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

    p = K @ np.array([x, y, z])
    u = p[0] / p[2]
    v = p[1] / p[2]
    return u, v

def compute_q(pt, radius):
    q = np.cross(pt, [1.0, 0.0, 0.0])
    d = np.linalg.norm(q)
    if d == 0:
        q = np.array([0.0, 1.0, 0.0])
        d = 1.0
    return q * radius / d


def compute_rotation_axis(pt):
    return pt / np.linalg.norm(pt)


def rotate_q(q, rotation_axis, angle):
    a = rotation_axis
    c = np.cos(angle)
    s = np.sin(angle)
    p = q * c
    p2 = np.cross(a, q) * s
    p3 = a * np.dot(a, q) * (1 - c)
    return p + p2 + p3


def draw_image(x, y, z, K, width, height, **kwargs):
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :] = (255, 255, 255)

    if z <= 0:
        rclpy.logging.get_logger('fancy_virtual_camera').warning('Target is behind the camera')
        return img

    p = np.array([x, y, z])
    q = compute_q(p, kwargs['radius'])
    a = compute_rotation_axis(p)
    pts = np.zeros((20, 2), np.int32)

    for i in range(20):
        t = 2 * np.pi * i / 20
        p2 = p + rotate_q(q, a, t)
        if p2[2] <= 0:
            rclpy.logging.get_logger('fancy_virtual_camera').warning('Outline is behind the camera')
            return img
        u, v = project_3D_point(p2[0], p2[1], p2[2], K)
        pts[i] = [int(round(u)), int(round(v))]

    cv2.drawContours(img, [pts], 0, (255, 0, 0), 3)
    return img


class FancyVirtualCameraNode(Node):
    """
    ROS 2 node for Part IV for assignment 1. Renders an image from a virtual camera.
    """

    def __init__(self):
        super().__init__('fancy_virtual_camera')

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
        img = draw_image(p2.x, p2.y, p2.z, self.K, 640, 480, radius=target_msg.radius)

        m = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.f
        self.pub.publish(m)

        self.info.header = m.header
        self.pub2.publish(self.info)


def main(args=None):
    rclpy.init(args=args)
    n = FancyVirtualCameraNode()
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(n)
    ex.spin()
    ex.shutdown()
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
