#!/usr/bin/env python3
"""Wait until the simulation has finished bringing up its controllers, then exit.

The simulation spawns its controllers one after the other: it activates the
follow_trajectory_controller and loads the joint_group_controller inactive. MoveIt Servo
(shutter_servo) needs to switch to the joint_group_controller when it starts, and that
switch fails if the controllers are still being spawned. The launch files in this
assignment therefore start shutter_servo only after this node exits successfully.
"""

import sys
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from controller_manager_msgs.srv import ListControllers


class WaitForControllersNode(Node):
    """
    Node that waits until the follow_trajectory_controller is active and the joint_group_controller is loaded.
    """

    def __init__(self):
        super().__init__('wait_for_controllers')
        self.declare_parameter("timeout", 120.0)  # seconds (wall clock)
        self.timeout = self.get_parameter("timeout").value
        self.client = self.create_client(ListControllers, '/controller_manager/list_controllers')

    def controllers_ready(self):
        """
        :return: True if the follow_trajectory_controller is active and the joint_group_controller is inactive,
                 or if the joint_group_controller is already active (e.g., MoveIt Servo ran before)
        """
        future = self.client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is None:
            return False
        states = {c.name: c.state for c in future.result().controller}
        if states.get('joint_group_controller') == 'active':
            return True
        return states.get('follow_trajectory_controller') == 'active' and \
            states.get('joint_group_controller') == 'inactive'

    def wait(self):
        """
        Wait for the controllers
        :return: True if they are ready, False if the timeout expired
        """
        start = self.get_clock().now()
        while rclpy.ok() and (self.get_clock().now() - start).nanoseconds < self.timeout * 1e9:
            if self.client.wait_for_service(timeout_sec=1.0) and self.controllers_ready():
                self.get_logger().info("The simulation's controllers are ready")
                return True
            self.get_logger().info("Waiting for the simulation's controllers...", throttle_duration_sec=5.0)
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().error("The simulation's controllers were not ready after {} seconds".format(self.timeout))
        return False


def main(args=None):
    rclpy.init(args=args)
    node = None
    ready = False

    try:
        node = WaitForControllersNode()
        ready = node.wait()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(0 if ready else 1)


if __name__ == '__main__':
    main()
