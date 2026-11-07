#!/usr/bin/env python3

import copy
import queue
import numpy as np
import rclpy
from rclpy.node import Node
from shutter_lookat.msg import Target
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from threading import Lock


def KF_predict_step(mu, Sigma, A, R):
    """
    Prediction step for the kalman filter
    :param mu: prior state mean
    :param Sigma: prior state covariance
    :param A: linear transition function for the state
    :param R: covariance for the noise of the transition model
    :return: predicted mean and covariance for the state based on the transition model A and the process noise R.
    """
    # TODO. Complete. Set the predicted_mu and predicted_Sigma variables according to the Kalman Filter's prediction step.
    return predicted_mu, predicted_Sigma


def KF_measurement_update_step(pred_mu, pred_Sigma, z, C, Q):
    """
    Correction step for the kalman filter
    :param pred_mu: predicted mean for the state (from KF_predict_step). Should be a 9x1 vector.
    :param pred_Sigma: predicted covariance for the state (from KF_predict_step). Should be a 9x9 matrix.
    :param z: measurement.
    :param C: matrix that transforms the state into measurement by Cx.
    :param Q: covariance for the noise of the observation or measurement model.
    :return: corrected mean and covariance for the state based on the observation z, the linear model C, and the measurement cov. Q.
    """
    # TODO. Complete. Set the corrected_mu and corrected_Sigma variables according to the Kalman Filter's measurement update step.
    return corrected_mu, corrected_Sigma


class KalmanFilterNode(Node):
    """Kalman Filter node"""

    def __init__(self):
        super().__init__('kalman_filter_node')

        # Note parameters
        self.declare_parameter("frame_rate", 20)        # fps
        frame_rate = self.get_parameter("frame_rate").value

        # Filter variables
        self.mu = None                                     # state mean
        self.Sigma = None                                  # state covariance
        self.R = None                                      # covariance for the process model
        self.Q = None                                      # covariance for the measurement model
        self.A = None                                      # matrix that predicts the new state based on the prior state
        self.C = None                                      # matrix that transforms states into observations

        # Initialize constant filter values
        self.initialize_process_covariance()
        self.initialize_measurement_covariance()
        self.assemble_C_matrix()

        # other node variables
        self.frame_id = None                               # frame id for the observations and filtering
        self.latest_observation_msg = None                 # latest PointStamped observation that was received by the node
        self.mutex = Lock()                                # mutex for the observation
        self.observed_positions = queue.Queue(50)          # FIFO queue of observed positions (for visualization)
        self.tracked_positions = queue.Queue(50)           # FIFO queue of tracked positions (for visualization)
        self.last_time = None                              # Last time-stamp of when a filter update was done

        # Publishers
        self.pub_filtered = self.create_publisher(PoseStamped, "/filtered_target", 5)
        self.pub_markers = self.create_publisher(MarkerArray, "/filtered_markers", 5)

        # Subscribers
        self.obs_sub = self.create_subscription(Target, "/target", self.obs_callback, 5)

        # Create timer for main loop
        self.timer = self.create_timer(1.0/frame_rate, self.main_loop)

    def main_loop(self):
        """Main processing loop"""
        
        # publish visualization info on every iteration
        self.publish_position_markers()

        # copy the last received message for the filter update
        self.mutex.acquire()
        obs_msg = copy.deepcopy(self.latest_observation_msg) 
        self.latest_observation_msg = None
        self.mutex.release()

        # initialize filter if state is None and save the current stamp as last_time
        if self.mu is None and obs_msg is not None:
            self.initialize_mu_and_sigma(obs_msg)
            self.last_time = self.get_clock().now()
            return

        # do nothing until we have initialized the filter
        if self.last_time is None:
            return

        # compute elapsed time from last prediction
        current_time = self.get_clock().now()
        delta_t = (current_time - self.last_time).nanoseconds / 1e9
        assert delta_t >= 0, "Negative delta_t = {}?".format(delta_t) # sanity check!

        # assemble A matrix: helps generate new state from prior state and elapsed time
        self.assemble_A_matrix(delta_t)

        # prediction step: predict new mean and covariance
        self.mu, self.Sigma = KF_predict_step(self.mu, self.Sigma, self.A, self.R)

        # save the time of when we made the prediction
        self.last_time = current_time

        # don't correct the state if we don't have a new observation
        if obs_msg is None:
            # store tracked state for visualization purposes
            self.store_tracked_state(self.mu, current_time)
            return

        # assemble observation vector
        z = self.assemble_observation_vector(obs_msg)

        # measurement update step: correct mean and covariance
        self.mu, self.Sigma = KF_measurement_update_step(self.mu, self.Sigma, z, self.C, self.Q)

        # store tracked state for visualization purposes
        self.store_tracked_state(self.mu, current_time)

        # store observation for visualization purposes
        self.store_tracked_obs(z)

    def obs_callback(self, msg):
        """
        Observation callback. Stores the observation in self.latest_observation_msg.
        :param msg: Target message with (x,y,z) location observation
        """
        self.mutex.acquire()

        # save observation
        self.latest_observation_msg = msg

        # save obs frame
        if self.frame_id is None:
            self.frame_id = self.latest_observation_msg.pose.header.frame_id
        elif self.frame_id != self.latest_observation_msg.pose.header.frame_id:
            self.get_logger().warn("Did the frame of the observations changed? Check the data!")

        self.mutex.release()

    def store_tracked_state(self, mu, current_stamp):
        """
        Store tracked state and publish result
        :param mu: state (position and vel)
        :param current_stamp: stamp for when the prediction was made
        """

        # build list of tracked positions so that we can draw a motion track for the target
        if self.tracked_positions.full():
            self.tracked_positions.get_nowait()

        self.tracked_positions.put_nowait(mu[:3, 0])

        # published filtered state
        msg = PoseStamped()
        msg.header.stamp = current_stamp.to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = mu[0,0]
        msg.pose.position.y = mu[1,0]
        msg.pose.position.z = mu[2,0]
        self.pub_filtered.publish(msg)

    def publish_position_markers(self):
        """
        Publish position markers for the observations and tracked states
        """
        if self.frame_id is None or (self.observed_positions.empty() and self.tracked_positions.empty()):
            return # can't publish markers without knowing their frame or without data

        def make_marker(queue_obj, stamp, ns, r=1.0, g=1.0, b=1.0, line_thickness=0.2):
            """                                                                                                                                
            Helper function to create marker for a single track of positions                                                                   
            :param queue_obj: queue object with position data                                                                                  
            :param stamp: stamp for marker                                                                                                     
            :param identifier: marker id                                                                                                       
            :param r: red intensity [0,1]                                                                                                      
            :param g: green intensity [0,1]                                                                                                    
            :param b: blue intensity [0,1]                                                                                                     
            """
            list_array = list(queue_obj.queue)
            list_array = [Point(x=x[0], y=x[1], z=x[2]) for x in list_array]
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.frame_id
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.scale.x = line_thickness
            marker.scale.y = line_thickness
            marker.scale.z = line_thickness
            marker.color.a = 1.0
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.pose.orientation.w = 1.0
            marker.points = list_array
            marker.ns = ns
            marker.id = 0

            end_marker = Marker()
            end_marker.header.stamp = stamp
            end_marker.header.frame_id = self.frame_id
            end_marker.type = end_marker.SPHERE
            end_marker.action = end_marker.ADD
            end_marker.scale.x = line_thickness*2
            end_marker.scale.y = line_thickness*2
            end_marker.scale.z = line_thickness*2
            end_marker.color.a = 1.0
            end_marker.color.r = r
            end_marker.color.g = g
            end_marker.color.b = b
            end_marker.pose.orientation.w = 1.0
            end_marker.pose.position = list_array[-1]
            end_marker.ns = ns
            end_marker.id = 1

            return marker, end_marker

        # create marker array object to publish all the tracks at once
        ma = MarkerArray()
        s = self.get_clock().now().to_msg()

        # create visualization marker for observations -- they will appear white
        if not self.observed_positions.empty():
            marker_obs, _ = make_marker(self.observed_positions, s, "observations", 1.0, 1.0, 1.0, 0.03)
            ma.markers.append(marker_obs)

        # create visualization marker for filtered (current) positions -- they will appear as thick green lines
        if not self.tracked_positions.empty():
            marker_fil, end_marker_fil = make_marker(self.tracked_positions, s, "filtered", 0.0, 1.0, 0.0, 0.01)
            ma.markers.append(marker_fil)
            ma.markers.append(end_marker_fil)

        # publish marker array
        self.pub_markers.publish(ma)

    def store_tracked_obs(self, z):
        """
        Store observations in queue
        :param z: latest observation
        """
        # build list of observed positions so that we can compared observed vs. tracked positions
        if self.observed_positions.full():
            self.observed_positions.get_nowait()

        self.observed_positions.put_nowait(z[:,0].transpose())

    def assemble_A_matrix(self, delta_t):
        """
        Method that assembles the A matrix for the KF_predict_step
        :param delta_t: elapsed time (in seconds) since last prediction
        """
        # TODO. Remove the "pass" line below and set self.A based on the elapsed time delta_t
        pass

    def assemble_C_matrix(self):
        """
        Method that assembles the C matrix for the KF_measurement_step
        """
        # TODO. Remove the "pass" line below and set self.C such that self.C x self.mu returns the expected measurement
        pass

    def initialize_process_covariance(self):
        """
        Method that sets the process covariance R for the filter node
        """
        # TODO. Remove the "pass" line below and set self.R
        pass

    def initialize_measurement_covariance(self):
        """
        Method that sets the process covariance Q for the filter node
        """
        # TODO. Remove the "pass" line below and set self.Q
        pass

    def initialize_mu_and_sigma(self, obs_msg):
        """
        Method that initializes the state (sets self.mu and self.Sigma).
        :param obs_msg Observation message with the latest measured position for the target
        """
        # TODO. Remove the "pass" line below and set self.mu and self.Sigma to their initial values here.
        pass 

    def assemble_observation_vector(self, obs_msg):
        """
        Build the observation vector as a numpy array with the data from the Observation message
        :param obs_msg: latest Target message that has been received by the node
        :return: numpy array representing the observation vector (with the 3D position of the target)
        """
        # TODO. Complete. Build the numpy array z such that it corresponds to the observed target location.
        return z


def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = KalmanFilterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("ROS Interrupt.")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()