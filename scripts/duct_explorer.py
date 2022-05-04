#!/usr/bin/env python
import numpy as np
import rospy
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

from math import sqrt, cos, sin, atan, pi, atan2


class DuctExplorer:
    def __init__(self):
        rospy.init_node('duct_explorer', anonymous=True)
        self.freq = 10
        self.rate = rospy.Rate(self.freq)

        # Data used by this node
        self.robot_pos = np.zeros(3)
        self.robot_angles = np.zeros(3)  # Euler angles
        self.laser = {'ranges': np.array([]),
                      'angles': np.array([]),
                      'angle_min': 0.0,
                      'angle_max': 0.0,
                      'angle_increment': 0.0}
        self.virtual_laser = {'ranges': [], 'angles': [], 'angle_min': 0.0, 'angle_max': 0.0, 'angle_increment': 0.0}

        # Topics that this node interacts with
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/tf", TFMessage, self.callback_pose)
        rospy.Subscriber("/scan", LaserScan, self.callback_laser)

        self.vel = Twist()
        self.vel.linear.x = 0
        self.vel.linear.y = 0
        self.vel.linear.z = 0
        self.vel.angular.x = 0
        self.vel.angular.y = 0
        self.vel.angular.z = 0

        self.half = 120
        self.started_pose = False
        self.started_laser = False

        self.rate.sleep()

    def callback_laser(self, data):
        """
        Callback routine to get the data from the laser sensor
        Args:
            data: laser data
        """
        self.laser['ranges'] = np.array(data.ranges)
        self.laser['angle_min'] = data.angle_min
        self.laser['angle_max'] = data.angle_max
        self.laser['angle_increment'] = data.angle_increment

        number_of_beams = int((self.laser['angle_max'] - self.laser['angle_min']) / self.laser['angle_increment'])

        angle = self.laser['angle_min']
        self.laser['angles'] = np.zeros(self.laser['ranges'].shape[0])
        for i in range(self.laser['ranges'].shape[0]):
            self.laser['angles'][i] = angle
            angle += self.laser['angle_increment']
        self.create_virtual_laser()
        self.started_laser = True

    def callback_pose(self, data):
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

    def create_virtual_laser(self, dist=0.2):
        """
        Simulates a laser in front of real laser sensor
        Args:
            dist: how far the virtual laser will be in front of the current position
        Returns:
        """
        self.virtual_laser['angle_min'] = self.laser['angle_min']
        self.virtual_laser['angle_max'] = self.laser['angle_max']
        self.virtual_laser['angle_increment'] = self.laser['angle_increment']

        self.virtual_laser['ranges'] = []
        self.virtual_laser['angles'] = []
        for i in range(self.laser['ranges'].shape[0]):
            x = self.laser['ranges'][i] * np.cos(self.laser['angles'][i])
            y = self.laser['ranges'][i] * np.sin(self.laser['angles'][i])

            x -= dist

            d = np.sqrt(x ** 2 + y ** 2)
            th = atan2(y, x)
            self.virtual_laser['ranges'].append(d)
            self.virtual_laser['angles'].append(th)
        new_data = []
        for i in range(len(self.virtual_laser['ranges'])):
            new_data.append((self.virtual_laser['ranges'][i], self.virtual_laser['angles'][i]))

        # Ordering using angle
        new_data.sort(key=lambda k: k[1])

        for i in range(len(self.virtual_laser['ranges'])):
            self.virtual_laser['ranges'][i] = new_data[i][0]
            self.virtual_laser['angles'][i] = new_data[i][1]
        self.virtual_laser['ranges'] = np.array(self.virtual_laser['ranges'])
        self.virtual_laser['angles'] = np.array(self.virtual_laser['angles'])

    def closest_obstacle(self, right_side=True, half=None, virtual_laser=False):
        """
        Returns the closest obstacle to the robot using a laser sensor.
        Args:
            right_side: a boolean indicating it is desired to know the closest obstacle on the right side (True) or on
            the left side (False)
            half: an integer informing where to section the right and left side
            virtual_laser: bool to indicate if must use a virtual laser

        Returns: a tuple with the minimum distance of the closest obstacle, the angle where this distance was measured
        and the index of this measurement in the laser vector
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

    def follow_corridor(self):
        """
        Control method to follow a corridor in the center of it
        Returns: linear and angular velocities in the robot frame
        """
        d = 0.15  # distance used in feedback linearization
        kf = 1  # convergence gain
        vr = 0.25  # linear velocity reference

        dist_r, phi_r, index_r = self.closest_obstacle(right_side=True, half=self.half)
        dist_l, phi_l, index_l = self.closest_obstacle(right_side=False, half=self.half)

        alpha = (phi_l - phi_r - pi) / 2.0

        phi_D = phi_r + alpha
        phi_T = phi_r + alpha + pi / 2.0

        D = (dist_l - dist_r) / (2 * cos(alpha))

        G = -(2 / pi) * atan(kf * D)
        H = sqrt(1 - G * G)

        vx = G * cos(phi_D) + H * cos(phi_T)  # (body)
        vy = G * sin(phi_D) + H * sin(phi_T)  # (body)

        v = vr * vx
        omega = vr * (vy / (d * 0.5))  # Angular rotation

        return v, omega

    def follow_wall(self):
        """
        Control method to follow a wall
        Returns: linear and angular velocities in the robot frame
        """
        d = 0.15  # distance used in feedback linearization
        kf = 1  # convergence gain
        vr = 0.25  # linear velocity reference
        # simulation
        epsilon = 0.6
        # CORO corridor
        # epsilon = 1.25

        delta_m, phi_m, index_r = self.closest_obstacle(right_side=False, virtual_laser=True)

        G = (2 / pi) * atan(kf * (delta_m - epsilon))
        H = - sqrt(1 - G * G)  # + right side to the wall; - left side to the wall

        v = vr * (cos(phi_m) * G - sin(phi_m) * H)
        omega = vr * (sin(phi_m) * G / d + cos(phi_m) * H / d)

        return v, omega

    def main_service(self):
        while not self.started_pose or not self.started_laser:
            continue
        while not rospy.is_shutdown():
            # v, w = self.follow_corridor()
            v, w = self.follow_wall()
            self.vel.linear.x = v
            self.vel.angular.z = w

            self.pub_cmd_vel.publish(self.vel)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        service = DuctExplorer()
        service.main_service()
    except rospy.ROSInterruptException:
        pass
