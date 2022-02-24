#!/usr/bin/env python
import numpy as np
import rospy
import roslib
from sensor_msgs.msg import Joy
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


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

        self.rate.sleep()

    def main_service(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def callback_laser(self, data: LaserScan):
        self.laser['ranges'] = list(data.ranges)
        self.laser['angle_min'] = data.angle_min
        self.laser['angle_max'] = data.angle_max
        self.laser['angle_increment'] = data.angle_increment

        number_of_beams = int((self.laser['angle_max'] - self.laser['angle_min'])/self.laser['angle_increment'])

        angle = self.laser['angle_min']
        self.laser['angles'] = []
        # print(number_of_beams)
        for i in range(number_of_beams):
            self.laser['angles'].append(angle)
            angle += self.laser['angle_increment']

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

    def follow_corridor(self):
        pass

    def feedback_linearization(self):
        pass



if __name__ == '__main__':
    try:
        service = DuctExplorer()
        service.main_service()
    except rospy.ROSInterruptException:
        pass