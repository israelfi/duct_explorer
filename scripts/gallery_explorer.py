#!/usr/bin/env python
import numpy as np
from typing import Tuple
from datetime import datetime
from math import sqrt, cos, sin, atan, pi, atan2

import rospy
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String


class DuctExplorer:
    def __init__(self):
        self.message_log('Starting gallery_explorer node...')

        self.__set_parameters()

        # Data used by this node
        self.robot_pos = np.zeros(3)
        self.robot_angles = np.zeros(3)

        self.laser_half = 120
        self.laser = {'ranges': np.array([]),
                      'angles': np.array([]),
                      'angle_min': 0.0,
                      'angle_max': 0.0,
                      'angle_increment': 0.0}
        self.virtual_laser = {'ranges': np.array([]),
                              'angles': np.array([]),
                              'angle_min': 0.0,
                              'angle_max': 0.0,
                              'angle_increment': 0.0}

        self.vel = Twist()
        self.vel.linear.x = 0
        self.vel.linear.y = 0
        self.vel.linear.z = 0
        self.vel.angular.x = 0
        self.vel.angular.y = 0
        self.vel.angular.z = 0

        self.state = 'none'
        self.previous_state = self.state

        self.reverse_mode = False

        self.started_pose = False
        self.started_laser = False
        self.started_state = False

        self.__init_node()

        self.rate.sleep()

    def print_parameters(self) -> None:
        """
        Print the parameters of the node
        Returns: None
        """
        print()
        self.message_log('----------------- PARAMETERS -----------------')
        self.message_log(f'Feedback linearization distance (m): {self.d}')
        self.message_log(f'Convergence gain: {self.kf}')
        self.message_log(f'Linear speed reference: {self.vr}')
        self.message_log(f'Distance from wall (m): {self.epsilon}')
        self.message_log('-----------------------------------------------\n')

    def __set_parameters(self) -> None:
        """
        Set the parameters of the node
        Returns: None
        """
        # Distance used in feedback linearization
        self.d = 0.15
        # Convergence gain
        self.kf = 1
        # Linear speed reference
        self.vr = 0.25
        # epsilon: distance from wall
        # simulation
        # self.epsilon = 0.6
        self.epsilon = 0.75
        # CORO corridor
        # epsilon = 1.25
        self.print_parameters()

    def __init_node(self) -> None:
        """
        Initialize the node subscribing to the needed topics
        Returns:

        """
        rospy.init_node('gallery', anonymous=True)
        self.freq = 10
        self.rate = rospy.Rate(self.freq)

        # Topics that this node interacts with
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/tf", TFMessage, self.callback_pose)
        rospy.Subscriber("/scan", LaserScan, self.callback_laser)
        rospy.Subscriber("/state", String, self.callback_state)

    @staticmethod
    def message_log(msg: str, end='\n') -> None:
        """
        Prints a message to the console with timestamp.
        Args:
            msg: message to be printed
            end: print function argument

        Returns: None
        """
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"[{dt_string}] {msg}", end=end)

    def callback_state(self, state):
        self.state = state.data
        self.started_state = True

    def callback_laser(self, data: LaserScan) -> None:
        """
        Callback routine to get the data from the laser sensor
        Args:
            data: laser data
        """
        laser = {'ranges': np.array(data.ranges),
                 'angles': np.array([]),
                 'angle_min': data.angle_min,
                 'angle_max': data.angle_max,
                 'angle_increment': data.angle_increment}

        angle = laser['angle_min']
        laser['angles'] = np.zeros(laser['ranges'].shape[0])
        for i in range(laser['ranges'].shape[0]):
            laser['angles'][i] = angle
            angle += laser['angle_increment']
        self.laser = laser
        self.create_virtual_laser()
        self.started_laser = True

    def callback_pose(self, data: TFMessage) -> None:
        """
        Callback routine to get the pose information of the robot
        Args:
            data: data from the TF topic
        """
        for T in data.transforms:
            # Choose the transform of the EspeleoRobo
            # Real Espeleo robot uses base_link
            # base_init, base_link, espeleo_robo/base_link
            if T.child_frame_id == "espeleo_robo/base_link":
                # Get the orientation
                x_q = T.transform.rotation.x
                y_q = T.transform.rotation.y
                z_q = T.transform.rotation.z
                w_q = T.transform.rotation.w
                self.robot_angles = euler_from_quaternion([x_q, y_q, z_q, w_q])

                # Get the position
                self.robot_pos[0] = T.transform.translation.x
                self.robot_pos[1] = T.transform.translation.y
                self.robot_pos[2] = T.transform.translation.z
        self.started_pose = True

    def dead_end_detected(self):
        dead_end = (self.previous_state != self.state
                    and 'StandardControl' in self.previous_state
                    and 'StandardControl' in self.state)
        return dead_end

    def create_virtual_laser(self, dist: float = 0.2) -> None:
        """
        Simulates a laser in front of real laser sensor. The purpuse of this virtual laser is to smoothen the angular
        displacement.
        Args:
            dist: how far the virtual laser will be in front of the current position
        Returns: None
        """
        virtual_laser = {'ranges': np.zeros(self.laser['ranges'].shape[0]),
                         'angles': np.zeros(self.laser['ranges'].shape[0]),
                         'angle_min': self.laser['angle_min'],
                         'angle_max': self.laser['angle_max'],
                         'angle_increment': self.laser['angle_increment']}

        for i in range(self.laser['ranges'].shape[0]):
            x = self.laser['ranges'][i] * np.cos(self.laser['angles'][i])
            y = self.laser['ranges'][i] * np.sin(self.laser['angles'][i])

            x -= dist

            d = np.sqrt(x ** 2 + y ** 2)
            th = atan2(y, x)
            virtual_laser['ranges'][i] = d
            virtual_laser['angles'][i] = th

        new_data = []
        for i in range(virtual_laser['ranges'].shape[0]):
            new_data.append((virtual_laser['ranges'][i], virtual_laser['angles'][i]))

        # Ordering using angle
        new_data.sort(key=lambda k: k[1])

        for i in range(len(virtual_laser['ranges'])):
            virtual_laser['ranges'][i] = new_data[i][0]
            virtual_laser['angles'][i] = new_data[i][1]

        virtual_laser['ranges'] = np.array(virtual_laser['ranges'])
        virtual_laser['angles'] = np.array(virtual_laser['angles'])

        self.virtual_laser = virtual_laser

    def closest_obstacle(self, right_side: bool = True, half: int = None,
                         virtual_laser: bool = False) -> Tuple[float, float, int]:
        """
        Returns the closest obstacle to the robot using a laser sensor.
        Args:
            right_side: a boolean indicating it is desired to know the closest obstacle on the right side (True) or on
            the left side (False)
            half: an integer informing where to section the right and left side
            virtual_laser: bool to indicate if virtual laser must be used

        Returns: a tuple with:
            - minimum distance of the closest obstacle
            - angle where this distance was measured
            - index of this measurement in the laser vector
        """
        if virtual_laser:
            laser_data = self.virtual_laser
        else:
            laser_data = self.laser

        if half is None:
            half = int(laser_data['ranges'].shape[0] / 2)
        if right_side:
            min_dist = np.min(laser_data['ranges'][:half])
            index = np.where(laser_data['ranges'][:half] == min_dist)[0][0]
            angle_of_closest_obstacle = laser_data['angles'][:half][index]
        else:
            min_dist = np.min(laser_data['ranges'][half:])
            index = np.where(laser_data['ranges'][half:] == min_dist)[0][0]
            angle_of_closest_obstacle = laser_data['angles'][half:][index]
        return min_dist, angle_of_closest_obstacle, index

    def follow_wall(self) -> tuple:
        """
        Implementation of a vector field based control that makes the robot follow a tangent path alongside a wall

        Returns: atuple with linear and angular velocities in the robot frame
        """

        delta_m, phi_m, index_r = self.closest_obstacle(right_side=self.reverse_mode, virtual_laser=False)

        self.message_log(f'Closest obstacle distance (m): {round(delta_m, 2)} | '
                         f'Angle (deg): {round(np.rad2deg(phi_m), 2)}')

        signal = -1 * self.reverse_mode + 1 * (not self.reverse_mode)

        G = (2 / pi) * atan(self.kf * (delta_m - self.epsilon)) * signal
        H = - sqrt(1 - G * G) * signal  # + right side to the wall; - left side to the wall

        v = self.vr * (cos(phi_m) * G - sin(phi_m) * H) * signal
        omega = self.vr * (sin(phi_m) * G / self.d + cos(phi_m) * H / self.d)

        return v, omega

    def wait_for_subscribed_topics(self) -> None:
        """
        Waits for the topics that are subscribed to by this topic.
        Returns: None
        """
        self.message_log('Waiting for messages in subscribed topics...')
        while not self.started_laser:
            continue
        self.message_log('Topics ready!')

    def publish_speeds(self, linear: float, angular: float) -> None:
        """
        Publishes the linear and angular speeds to the robot.
        Args:
            linear: linear speed in the robot frame
            angular: angular speed in the robot frame

        Returns: None
        """
        self.message_log(f'{self.reverse_mode} Linear speed: {round(linear, 3)} | Angular Speed: {round(angular, 3)}')
        self.vel.linear.x = linear
        self.vel.angular.z = angular
        self.pub_cmd_vel.publish(self.vel)

    def check_robot_direction(self):
        dead_end = self.dead_end_detected()
        self.reverse_mode = (not self.reverse_mode) * dead_end + self.reverse_mode * (not dead_end)
        self.previous_state = self.state

    def main_service(self) -> None:
        """
        Main service method.

        Returns: None
        """
        self.wait_for_subscribed_topics()
        while not rospy.is_shutdown():
            self.check_robot_direction()
            v, w = self.follow_wall()
            if self.state == 'Exit':
                v, w = 0, 0
            self.publish_speeds(linear=v, angular=w)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        service = DuctExplorer()
        service.main_service()
    except rospy.ROSInterruptException:
        pass
