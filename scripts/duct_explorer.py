#!/usr/bin/env python
import numpy as np
import rospy
import roslib
from sensor_msgs.msg import Joy
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

from math import sqrt, cos, sin, atan, pi


class DuctExplorer:
    def __init__(self):
        rospy.init_node('duct_explorer', anonymous=True)
        self.freq = 10
        self.rate = rospy.Rate(self.freq)

        # Data used by this node
        self.robot_pos = np.zeros(3)
        self.robot_angles = np.zeros(3)  # Euler angles
        self.laser = {'ranges': [], 'angles': [], 'angle_min': 0.0, 'angle_max': 0.0, 'angle_increment': 0.0}

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

        self.rate.sleep()

    def callback_laser(self, data: LaserScan):
        self.laser['ranges'] = np.array(data.ranges)
        self.laser['angle_min'] = data.angle_min
        self.laser['angle_max'] = data.angle_max
        self.laser['angle_increment'] = data.angle_increment

        number_of_beams = int((self.laser['angle_max'] - self.laser['angle_min']) / self.laser['angle_increment'])

        angle = self.laser['angle_min']
        self.laser['angles'] = []
        for i in range(number_of_beams):
            self.laser['angles'].append(angle)
            angle += self.laser['angle_increment']
        self.laser['angles'] = np.array(self.laser['angles'])

    def callback_pose(self, data):
        for T in data.transforms:
            # Choose the transform of the EspeleoRobo
            if T.child_frame_id == "base_link":
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

    def closest_obstacle(self, right_side=True, half=None):
        print(self.laser['ranges'].shape[0])
        if half is None:
            half = int(self.laser['ranges'].shape[0] / 2)
        if right_side:
            min_dist = np.min(self.laser['ranges'][:half])
            index = np.where(self.laser['ranges'][:half] == min_dist)[0][0]
            angle_of_closest_obstacle = self.laser['angles'][:half][index]
        else:
            min_dist = np.min(self.laser['ranges'][half:])
            index = np.where(self.laser['ranges'][half:] == min_dist)[0][0]
            angle_of_closest_obstacle = self.laser['angles'][half:][index]
        return min_dist, angle_of_closest_obstacle, index

    def follow_corridor(self):
        d = 0.15  # distance used in feedback linearization
        kf = 1  # convergence gain
        vr = 0.5  # linear velocity reference

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

        v = vr * (vx)
        omega = vr * (vy / (d * 0.5))  # Angular rotation

        return v, omega

    def feedback_linearization(self):
        pass

    def main_service(self):
        while not rospy.is_shutdown():
            v, w = self.follow_corridor()
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
