#!/usr/bin/env python

import time
import rospy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from math import hypot
from datetime import datetime

from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class TopologicalMapping:
    __MIN_DIST = 2.5

    def __init__(self, name):
        self.node_name = name
        self.G = nx.Graph()
        self.node_count = 0
        self.previous_node = None

        self.robot_pos = np.zeros(3)
        self.robot_angles = np.zeros(3)  # Euler angles

        self.started_pose = False
        self.started_state = False

        self.state = 'none'
        self.previous_state = self.state

        self.fig = plt.figure(figsize=(8, 5))
        self.fig.canvas.mpl_connect('key_release_event', self.close_graph)

        self.__init_node()

    def __init_node(self):
        rospy.init_node(self.node_name, anonymous=True)

        self.freq = 10
        self.rate = rospy.Rate(self.freq)

        rospy.Subscriber("/tf", TFMessage, self.callback_pose)
        rospy.Subscriber("/state", String, self.callback_state)

    def close_graph(self, event):
        if event.key == 'escape':
            self.message_log('Closing topological mapping node')
            exit(0)
        else:
            return

    @staticmethod
    def euclidian_distance(p1, p2):
        return hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def message_log(msg: str):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"[{dt_string}] {msg}")

    def callback_state(self, state):
        self.state = state.data
        self.started_state = True

    def callback_pose(self, data):
        """
        Callback routine to get the pose information of the robot
        Args:
            data: data from the TF topic
        """
        for T in data.transforms:
            # Choose the transform of the EspeleoRobo
            if T.child_frame_id == "truth_link":
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

    def add_first_node(self):
        self.G.add_node(self.node_count, pos=(self.robot_pos[0], self.robot_pos[1]), type='start')

    def add_node_to_graph(self, node_type):
        original_state = self.state
        self.G.add_node(self.node_count + 1, pos=(self.robot_pos[0], self.robot_pos[1]), type=node_type)
        if self.previous_node is not None:
            self.G.add_edge(self.previous_node, self.node_count + 1)
            self.previous_node = None
        else:
            self.G.add_edge(self.node_count, self.node_count + 1)
        self.node_count += 1
        self.message_log(f'New node postion: {self.robot_pos}')
        self.message_log(f"Total Nodes: {self.G.number_of_nodes()}")
        while self.state == original_state:
            self.draw_graph()

    @staticmethod
    def _can_not_remove_nodes(node_one_type, node_two_type):
        return (node_one_type != node_two_type
                or node_one_type == 'dead end'
                or node_two_type == 'dead end')

    def check_repeated_nodes(self):
        node_positions = nx.get_node_attributes(self.G, 'pos')
        node_types = nx.get_node_attributes(self.G, 'type')
        for i in range(self.G.number_of_nodes()):
            for j in range(i + 1, self.G.number_of_nodes()):
                if self._can_not_remove_nodes(node_types[i], node_types[j]):
                    continue

                d = self.euclidian_distance(node_positions[i], node_positions[j])

                # If two nodes are close to each other, it is considered that they are only one node
                if d < self.__MIN_DIST:
                    self.message_log(f'Nodes {i} and {j} are {d} meters close. Removing {j}.')
                    # Nodes connected to the repeated node j
                    nodes_connected = [x[1] for x in self.G.edges(j)]
                    self.previous_node = i
                    self.G.remove_node(j)
                    self.node_count -= 1
                    self.message_log(f"Total Nodes: {self.G.number_of_nodes()}")
                    for k in nodes_connected:
                        if self.G.nodes[i]['pos'] == self.G.nodes[k]['pos']:
                            continue
                        self.G.add_edge(i, k)
                    break

    def dead_end_detected(self):
        dead_end = (self.previous_state != self.state
                    and 'StandardControl' in self.previous_state
                    and 'StandardControl' in self.state)
        return dead_end

    def draw_graph(self):
        plt.cla()
        color_map = []
        node_types = nx.get_node_attributes(self.G, 'type')
        for node in self.G:
            node_type = node_types[node]
            if node_type == 'start':
                color_map.append('black')
            elif node_type == 'bifurcation':
                color_map.append('blue')
            elif node_type == 'dead end':
                color_map.append('red')

        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, pos, node_color=color_map)
        a = [node for node in self.G]
        plt.pause(0.1)

    def read_state_data(self):
        if 'Bifurcation' in self.state:
            self.message_log('Bifurcation')
            self.add_node_to_graph(node_type='bifurcation')
        elif self.dead_end_detected():
            self.message_log('Dead end')
            self.add_node_to_graph(node_type='dead end')
        self.previous_state = self.state

    def main_service(self):
        self.message_log(f'Waiting for subscribed topics...')
        while not self.started_pose or not self.started_state:
            continue
        self.message_log(f'Subscribed topics ready')

        self.add_first_node()
        while not rospy.is_shutdown():
            self.read_state_data()
            self.draw_graph()
            self.check_repeated_nodes()
            self.rate.sleep()


if __name__ == '__main__':
    node_name = "toplogical_mapping"
    print(node_name)
    service = TopologicalMapping(name=node_name)
    service.main_service()

    print(f"{node_name} node stopped")
