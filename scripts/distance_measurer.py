#!/usr/bin/env python

import rospy
import numpy as np

from datetime import datetime
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan, JointState, Imu
from utils.bib_espeleo_differential import EspeleoDifferential


class DistanceMeasurer:
    _REAL_ODOMETRY = True

    def __init__(self):

        # Times used to integrate velocity to pose
        self.current_time = 0.0
        self.last_time = 0.0

        self.distance = 0.0
        self.robot_angles_imu = np.zeros(3)

        self.espeleo = EspeleoDifferential()
        rospy.init_node('distance_measurer', anonymous=True)
        self.msg = String()

        # Motor velocities
        self.motor_velocity = np.zeros(6)

        self.dist_pub = rospy.Publisher("distance_measurer", String, queue_size=50)

        rospy.Subscriber("/device1/get_joint_state", JointState, self.motor1_callback)
        rospy.Subscriber("/device2/get_joint_state", JointState, self.motor2_callback)
        rospy.Subscriber("/device3/get_joint_state", JointState, self.motor3_callback)
        rospy.Subscriber("/device4/get_joint_state", JointState, self.motor4_callback)
        rospy.Subscriber("/device5/get_joint_state", JointState, self.motor5_callback)
        rospy.Subscriber("/device6/get_joint_state", JointState, self.motor6_callback)

        if self._REAL_ODOMETRY:
            self.msg_prefix = 'real odometry'
        else:
            self.msg_prefix = 'wheel distance'

        rospy.spin()

    @staticmethod
    def message_log(msg: str):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"[{dt_string}] {msg}")

    # Motor velocity callbacks
    def motor1_callback(self, message):
        self.motor_velocity[0] = message.velocity[0]

    def motor2_callback(self, message):
        self.motor_velocity[1] = message.velocity[0]

    def motor3_callback(self, message):
        self.motor_velocity[2] = message.velocity[0]

    def motor4_callback(self, message):
        self.motor_velocity[3] = message.velocity[0]

    def motor5_callback(self, message):
        self.motor_velocity[4] = message.velocity[0]

    def motor6_callback(self, message):
        self.motor_velocity[5] = message.velocity[0]
        self.current_time = message.header.stamp.secs + message.header.stamp.nsecs * 0.000000001

        if self.last_time > 0.0:
            self.odometry_calculations()
        else:
            self.last_time = self.current_time

    def callback_imu(self, data):
        quat = np.zeros(4)
        quat[0] = data.orientation.x
        quat[1] = data.orientation.y
        quat[2] = data.orientation.z
        quat[3] = data.orientation.w

        self.robot_angles_imu = euler_from_quaternion(quat)

    def odometry_calculations(self):
        v_r, v_l = self.espeleo.left_right_velocity(self.motor_velocity)

        dt = self.current_time - self.last_time
        self.last_time = self.current_time
        if self._REAL_ODOMETRY:
            v_espeleo = self.espeleo.wheel_radius * (v_r + v_l) / 2
            vx = v_espeleo * np.cos(self.robot_angles_imu[-1])
            vy = v_espeleo * np.sin(self.robot_angles_imu[-1])

            dist_dt = np.sqrt((vx * dt) ** 2 + (vy * dt) ** 2)
        else:
            v_espeleo = self.espeleo.wheel_radius * (abs(v_r) + abs(v_l)) / 2
            # Integrations
            dist_dt = v_espeleo * dt  # if v_espeleo * dt > 0.005 else 0

        self.distance += dist_dt

        self.msg.data = f'{self.distance}'
        self.dist_pub.publish(self.msg)
        self.message_log(f'Distance ({self.msg_prefix}): {round(self.distance, 3)}')


if __name__ == '__main__':
    try:
        service = DistanceMeasurer()
    except rospy.ROSInterruptException:
        pass
