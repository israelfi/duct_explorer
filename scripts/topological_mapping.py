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
    __DIST_TOL = 0.5

    def __init__(self, name, right_angles=False):
        self.node_name = name
        self.right_angles = right_angles

        self.G = nx.Graph()
        self.node_count = 0
        self.previous_node = None
        self.last_node_is_dead_end = False
        self.last_visited_node = None

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

        if self.last_node_is_dead_end:
            self.last_visited_node = list(self.G.edges(self.node_count))[0][1]
            self.wait_for_new_state(original_state)
            return

        is_dead_end = node_type == 'dead end'
        x_pos, y_pos = self.wait_for_new_state(original_state, dead_end=is_dead_end)
        self.G.add_node(self.node_count + 1, pos=(x_pos, y_pos), type=node_type)
        neighbour_node = self.find_neighbour_of_new_node()
        self.add_edge_to_new_node(neighbour_node)

        self.__update_graph_information()

        self.message_log(f'New node postion: {(x_pos, y_pos)}')
        self.message_log(f"Total Nodes: {self.G.number_of_nodes()}")

        self.draw_graph()

    def update_node_positions(self):
        """
        If nodes are close in the x or y axises, makes them have the same position
        """
        nodes = list(self.G.nodes)
        for i in range(self.G.number_of_nodes()):
            current_node = nodes[i]
            x_cur, y_cur = self.G.nodes[current_node]['pos']
            x_cur = [x_cur]
            y_cur = [y_cur]
            for j in range(self.G.number_of_nodes()):
                next_node = nodes[j]
                x_next_node, y_next_node = self.G.nodes[next_node]['pos']
                if abs(x_next_node - x_cur[0]) < self.__DIST_TOL:
                    x_cur.append(x_next_node)
                if abs(y_next_node - y_cur[0]) < self.__DIST_TOL:
                    y_cur.append(y_next_node)
            self.G.nodes[current_node]['pos'] = (np.mean(x_cur), np.mean(y_cur))

    def __update_graph_information(self):
        self.last_visited_node = self.node_count + 1
        self.node_count += 1

    def add_edge_to_new_node(self, neighbour_node):
        if self.previous_node is not None:
            self.G.add_edge(self.previous_node, self.node_count + 1)
            self.previous_node = None
        else:
            self.G.add_edge(neighbour_node, self.node_count + 1)

    def wait_for_new_state(self, original_state, dead_end=False):
        x_pos = [self.robot_pos[0]]
        y_pos = [self.robot_pos[1]]
        if dead_end:
            return np.round((x_pos[0], y_pos[0]), 2)

        while self.state == original_state:
            x_pos.append(self.robot_pos[0])
            y_pos.append(self.robot_pos[1])
            self.draw_graph()

        node_position = np.round((np.mean(x_pos), np.mean(y_pos)), 2)
        return node_position

    def find_neighbour_of_new_node(self):
        if self.last_visited_node is None:
            neighbour_node = self.node_count
        else:
            neighbour_node = self.last_visited_node
        return neighbour_node

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
        plt.pause(0.1)

    def read_state_data_and_add_node_to_graph(self):
        if 'Bifurcation' in self.state:
            self.message_log('Detected Bifurcation')
            self.add_node_to_graph(node_type='bifurcation')
            self.last_node_is_dead_end = False
        elif self.dead_end_detected():
            self.message_log('Detected Dead end')
            self.add_node_to_graph(node_type='dead end')
            self.last_node_is_dead_end = True
        self.previous_state = self.state

    def __wait_for_subscribed_topics(self):
        self.message_log(f'Waiting for subscribed topics...')
        while not self.started_pose or not self.started_state:
            continue
        self.message_log(f'Subscribed topics ready')

    def main_service(self):
        self.__wait_for_subscribed_topics()

        self.add_first_node()
        while not rospy.is_shutdown():
            self.read_state_data_and_add_node_to_graph()
            self.draw_graph()
            self.check_repeated_nodes()
            if self.right_angles:
                self.update_node_positions()
            self.rate.sleep()


if __name__ == '__main__':
    node_name = "toplogical_mapping"
    print(node_name)
    service = TopologicalMapping(name=node_name, right_angles=False)
    service.main_service()

    print(f"{node_name} node stopped")
