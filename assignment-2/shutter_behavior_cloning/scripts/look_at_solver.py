"""
Math for making Shutter's camera look at a target by moving joint_1 and joint_3.

This module has no ROS node in it, so that the expert (expert_opt.py), your policy (run_policy.py)
and the tests can all share the same frame utilities and the same expert solution.
"""

import numpy as np
from scipy.optimize import least_squares
import transformations as tft


def transform_msg_to_T(trans):
    """
    Convert TransformStamped message to 4x4 transformation matrix
    :param trans: TransformStamped message
    :return: 4x4 transformation matrix
    """
    # extract relevant information from transform
    q = [trans.transform.rotation.x,
         trans.transform.rotation.y,
         trans.transform.rotation.z,
         trans.transform.rotation.w]
    t = [trans.transform.translation.x,
         trans.transform.translation.y,
         trans.transform.translation.z]
    # convert to matrices
    Rq = tft.quaternion_matrix(q)
    Tt = tft.translation_matrix(t)
    return np.dot(Tt, Rq)


def make_joint_rotation(angle, rotation_axis='x'):
    """
    Make rotation matrix for joint (assumes that joint angle is zero)
    :param angle: joint angle
    :param rotation_axis: rotation axis as string or vector
    :return: rotation matrix
    """
    # set axis vector if input is string
    if not isinstance(rotation_axis, list):
        assert rotation_axis in ['x', 'y', 'z'], "Invalid rotation axis '{}'".format(rotation_axis)
        if rotation_axis == 'x':
            axis = (1.0, 0.0, 0.0)
        elif rotation_axis == 'y':
            axis = (0.0, 1.0, 0.0)
        else:
            axis = (0.0, 0.0, 1.0)
    else:
        axis = rotation_axis
    # make rotation matrix
    R = tft.rotation_matrix(angle, axis)
    return R


def target_in_camera_frame(angles, target_pose, rotation_axis1, rotation_axis2, T1, T2):
    """
    Transform target to camera frame
    :param angles: joint angles
    :param target_pose: target pose
    :param rotation_axis1: rotation axis of joint_1 (str or vector)
    :param rotation_axis2: rotation axis of joint_3 (str or vector)
    :param T1: transform - base_link to biceps
    :param T2: transform - biceps to camera_link
    :return: x and y coordinates of the target in the camera frame
    """

    # make transform for joint 1
    R1 = make_joint_rotation(angles[0], rotation_axis=rotation_axis1)

    # make transform for joint 3
    R2 = make_joint_rotation(angles[1], rotation_axis=rotation_axis2)

    # transform target to camera_link...
    p = np.array([[target_pose[0], target_pose[1], target_pose[2], 1.0]]).transpose()

    # target in base_link
    p1 = np.dot(np.dot(T1, R1), p)

    # target in camera_link
    result = np.dot(np.dot(T2, R2), p1)

    return result[0:2].flatten()


def get_joint(robot, name):
    """
    Get a joint from the robot model by name
    :param robot: URDF robot model (from urdf_parser_py)
    :param name: joint name (e.g., 'joint_1')
    :return: URDF joint, which has the joint's axis and limits
    """
    for joint in robot.joints:
        if joint.name == name:
            return joint
    raise ValueError("Joint '{}' is not in the robot model".format(name))


def solve_look_at(p, T1, T2, joint1_model, joint3_model, joint1, joint3):
    """
    Compute the joint_1 and joint_3 positions that make the robot's camera look towards the target
    :param p: target position [x, y, z] in the base_link frame
    :param T1: transform - base_link to biceps
    :param T2: transform - biceps to camera_link
    :param joint1_model: URDF joint for joint_1 (see get_joint())
    :param joint3_model: URDF joint for joint_3 (see get_joint())
    :param joint1: current joint_1 position
    :param joint3: current joint_3 position
    :return: new joint_1 and joint_3 positions, within the joints' limits
    """
    # compute the required motion for the robot using black-box optimization.
    # The +1e-3 perturbation pushes x0 off the saddle point that exists at (0, 0) when the target's y is
    # exactly 0. Without it, the solver reports "converged" at the saddle with res.x ~ 0 even though the
    # cost is > 0, and the robot does not move.
    x0 = [-np.arctan2(p[1], p[0]) + 1e-3, 1e-3]
    res = least_squares(target_in_camera_frame, x0,
                        bounds=([-np.pi, -np.pi * 0.5], [np.pi, np.pi * 0.5]),
                        args=(p, joint1_model.axis, joint3_model.axis, T1, T2))

    # cap the new joint positions based on the joint limits
    new_j1 = float(np.clip(joint1 - res.x[0], joint1_model.limit.lower, joint1_model.limit.upper))
    new_j3 = float(np.clip(joint3 - res.x[1], joint3_model.limit.lower, joint3_model.limit.upper))

    return new_j1, new_j3
