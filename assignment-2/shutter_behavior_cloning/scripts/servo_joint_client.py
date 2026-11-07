"""
Helper to command Shutter's arm joints through MoveIt Servo, so that nodes only need to call send_joint_goal().

Each goal is published on the /servo/joint_command topic. The shutter_servo node turns it into commands for
MoveIt Servo, which moves the arm smoothly towards the latest goal using the joint_group_controller. Servo does
not plan a path around obstacles, but it stays within the joint limits and slows down and stops before the
robot collides with itself. shutter_servo activates the joint_group_controller when it starts, so make sure 
that shutter_servo/launch/shutter_servo.launch.py is running.

Goals can be sent at any rate: once in a while (e.g., one per target in Part II of the assignment) or
continuously (e.g., at 30 Hz to follow a moving target in Part III). The arm keeps moving towards the
latest goal until it gets there.
"""

from std_msgs.msg import Float64MultiArray


class ServoJointClient(object):
    """
    Commands joint_1 to joint_4 of Shutter's arm through MoveIt Servo.
    """

    JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4']

    def __init__(self, node):
        """
        Constructor
        :param node: ROS node that will use this client
        """
        self.node = node
        self._servo_pub = node.create_publisher(Float64MultiArray, '/servo/joint_command', 1)

    def send_joint_goal(self, positions):
        """
        Command the arm to move to the given joint positions. This function does not block.
        :param positions: list with the goal positions for joint_1, joint_2, joint_3 and joint_4
        """
        assert len(positions) == len(self.JOINT_NAMES), \
            "Expected {} joint positions but got {}".format(len(self.JOINT_NAMES), len(positions))
        self._servo_pub.publish(Float64MultiArray(data=[float(p) for p in positions]))
