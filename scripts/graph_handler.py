#!/usr/bin/env python

import rospy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


class GraphHandler:
    def __init__(self):
        rospy.init_node('duct_explorer', anonymous=True)
        self.freq = 10
        self.rate = rospy.Rate(self.freq)

        self.G = nx.Graph()
        self.node_count = 0

        self.min_angle = 25
        self.min_dist = 3

        self.stage = 0

        self.robot_pos = np.zeros(3)
        self.robot_angles = np.zeros(3)  # Euler angles
        self.laser = {'ranges': np.array([]), 'angles': np.array([]), 'angle_min': 0.0, 'angle_max': 0.0,
                      'angle_increment': 0.0}
        rospy.Subscriber("/tf", TFMessage, self.callback_pose)
        rospy.Subscriber("/scan", LaserScan, self.callback_laser)

        self.setup_ready = False
        self.bifurcation_count = 0
        self.dead_end_count = 0

        self.rate.sleep()

    @staticmethod
    def message_log(msg: str):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"[{dt_string}] {msg}")

    def callback_laser(self, data: LaserScan):
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
        self.laser['angles'] = np.zeros(number_of_beams)
        for i in range(number_of_beams):
            self.laser['angles'][i] = angle
            angle += self.laser['angle_increment']
        self.laser['angles'] = np.array(self.laser['angles'])

    def callback_pose(self, data):
        """
        Callback routine to get the pose information of the robot
        Args:
            data: data from the TF topic
        """
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

        self.setup_ready = True

    def detect_bifurcation(self):
        front = 60
        angles = []
        for i in range(self.laser['ranges'][front:-front].shape[0]):
            if self.laser['ranges'][front:-front][i] > self.min_dist:
                angles.append(self.laser['angles'][front:-front][i])
        if not angles:
            return

        beam_groups = [[angles[0]]]
        for i in range(len(angles) - 1):
            if abs(np.rad2deg(angles[i + 1]) - np.rad2deg(angles[i])) < 10:
                beam_groups[-1].append(angles[i + 1])
            else:
                beam_groups.append([angles[i + 1]])

        bifurcation_directions = []
        for i in range(len(beam_groups)):
            bifurcation_directions.append(np.mean(beam_groups[i]))
        # self.message_log(f"{bifurcation_directions}")
        return len(bifurcation_directions) > 1

    def detect_dead_end(self):
        min_dist_to_detect_dead_end = 1
        if np.mean(self.laser['ranges']) < min_dist_to_detect_dead_end:
            self.message_log('Dead end')
            return True
        return False

    def add_first_node(self):
        self.G.add_node(self.node_count, pos=(self.robot_pos[0], self.robot_pos[1]), type='start')

    def draw_graph(self):
        color_map = []
        node_types = nx.get_node_attributes(self.G, 'type')
        for node in self.G:
            type = node_types[node]
            if type == 'start':
                color_map.append('black')
            elif type == 'bifurcation':
                color_map.append('blue')
            elif type == 'dead end':
                color_map.append('red')

        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, pos, node_color=color_map)
        plt.pause(0.01)

    def state_machine(self):
        if self.detect_bifurcation():
            self.message_log('Bifurcation')
            self.bifurcation_count += 1
            if self.bifurcation_count > 10:
                self.bifurcation_count = 0
                self.G.add_node(self.node_count + 1, pos=(self.robot_pos[0], self.robot_pos[1]), type='bifurcation')
                self.G.add_edge(self.node_count, self.node_count + 1)
                self.node_count += 1
                while self.detect_bifurcation():
                    continue

        elif self.detect_dead_end():
            self.message_log('Dead end')
            self.dead_end_count += 1
            if self.dead_end_count > 10:
                self.dead_end_count = 0
                self.G.add_node(self.node_count + 1, pos=(self.robot_pos[0], self.robot_pos[1]), type='dead end')
                self.G.add_edge(self.node_count, self.node_count + 1)
                self.node_count += 1
                while self.detect_dead_end():
                    continue

    def main_service(self):
        while not self.setup_ready:
            continue
        self.add_first_node()
        while not rospy.is_shutdown():
            self.state_machine()
            self.draw_graph()
            self.rate.sleep()


if __name__ == '__main__':
    try:
        service = GraphHandler()
        service.main_service()
    except rospy.ROSInterruptException:
        pass
