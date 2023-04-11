#!/usr/bin/env python

import time
import rospy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import cv2
from math import hypot, atan2, sqrt
from scipy import signal
from datetime import datetime

from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped


class TopologicalMapping:
    __MIN_DIST = 2.5
    __DIST_TOL = 0.5
    __MIN_DIST_FOR_LOCAL_MINIMA = 0.75

    def __init__(self, name, right_angles=False):
        self.node_name = name
        self.right_angles = right_angles

        self.G = nx.Graph()
        self.node_count = 0
        self.previous_node = None
        self.last_node_is_dead_end = False
        self.last_visited_node = None

        self.started_pose = False
        self.started_state = False
        self.started_laser = False

        self.robot_pos = np.zeros(3)
        self.robot_angles = np.zeros(3)  # Euler angles

        self.state = 'none'
        self.previous_state = self.state

        self.laser = {'ranges': np.array([]),
                      'angles': np.array([]),
                      'angle_min': 0.0,
                      'angle_max': 0.0,
                      'angle_increment': 0.0}

        self.fig = plt.figure(figsize=(8, 5))
        self.fig.canvas.mpl_connect('key_release_event', self.close_graph)

        self.robot_angles_imu = np.zeros(3)

        self.__init_node()

    def __init_node(self):
        rospy.init_node(self.node_name, anonymous=True)

        self.freq = 10
        self.rate = rospy.Rate(self.freq)

        use_odom = False
        if use_odom:
            # rospy.Subscriber("/odom", Odometry, self.callback_odometry)
            rospy.Subscriber("/robot_pose_ekf/odom_combined", PoseWithCovarianceStamped, self.callback_ekf)
        else:
            rospy.Subscriber("/tf", TFMessage, self.callback_pose)

        rospy.Subscriber("/state", String, self.callback_state)
        rospy.Subscriber("/scan", LaserScan, self.callback_laser)
        rospy.Subscriber("/imu_data", Imu, self.callback_imu)

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

    def callback_imu(self, data):
        quat = np.zeros(4)
        quat[0] = data.orientation.x
        quat[1] = data.orientation.y
        quat[2] = data.orientation.z
        quat[3] = data.orientation.w

        self.robot_angles_imu = euler_from_quaternion(quat)
        self.started_imu = True

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

    def callback_ekf(self, data):
        robot_pos = np.zeros(3)
        robot_pos[0] = data.pose.pose.position.x
        robot_pos[1] = data.pose.pose.position.y
        robot_pos[2] = data.pose.pose.position.z

        quat = np.zeros(4)
        quat[0] = data.pose.pose.orientation.x
        quat[1] = data.pose.pose.orientation.y
        quat[2] = data.pose.pose.orientation.z
        quat[3] = data.pose.pose.orientation.w

        self.robot_pos[0] = robot_pos[0]
        self.robot_pos[1] = robot_pos[1]
        self.robot_pos[2] = robot_pos[2]

        self.started_pose = True

    def callback_odometry(self, data):
        robot_pos = np.zeros(3)
        robot_pos[0] = data.pose.pose.position.x
        robot_pos[1] = data.pose.pose.position.y
        robot_pos[2] = data.pose.pose.position.z

        quat = np.zeros(4)
        quat[0] = data.pose.pose.orientation.x
        quat[1] = data.pose.pose.orientation.y
        quat[2] = data.pose.pose.orientation.z
        quat[3] = data.pose.pose.orientation.w

        self.robot_pos[0] = robot_pos[0]
        self.robot_pos[1] = robot_pos[1]
        self.robot_pos[2] = robot_pos[2]

        self.started_pose = True

    def callback_laser(self, data):
        """
        Callback routine to get the data from the laser sensor
        Args:
            data: laser data
        """
        ranges = np.array(data.ranges)

        self.laser['angle_min'] = data.angle_min
        self.laser['angle_max'] = data.angle_max
        self.laser['angle_increment'] = data.angle_increment

        angles = np.linspace(start=self.laser['angle_min'],
                             stop=self.laser['angle_max'],
                             num=ranges.shape[0])

        # Removing beams with 'inf' measurements
        is_not_inf = ranges != np.inf
        self.laser['ranges'] = ranges[is_not_inf]
        self.laser['angles'] = angles[is_not_inf]

        self.started_laser = True

    def add_first_node(self):
        self.G.add_node(self.node_count, pos=(self.robot_pos[0], self.robot_pos[1]), type='start', angles=[])

    def remove_repeated_points(self, points, check_distance):
        new_points = points[:]
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist_between_points = self.euclidian_distance(points[i], points[j])
                if dist_between_points < check_distance and points[j] in new_points:
                    new_points.remove(points[j])
        return new_points

    def estimate_bifurcation_position(self, local_minima_x, local_minima_y):
        x_robot = self.robot_pos[0]
        y_robot = self.robot_pos[1]
        if self.state is None or 'Bifurcation' not in self.state:
            return [None, None]

        points_in_range = []
        range_to_check = 2
        for i in range(len(local_minima_x)):
            distance_from_robot = np.sqrt(local_minima_x[i] ** 2 + local_minima_y[i] ** 2)
            if distance_from_robot < range_to_check:
                points_in_range.append([local_minima_x[i] + x_robot, local_minima_y[i] + y_robot])

        points_in_range = self.remove_repeated_points(points=points_in_range,
                                                      check_distance=self.__MIN_DIST_FOR_LOCAL_MINIMA)

        if len(points_in_range) < 2:
            return [None, None]

        # bifurcation_in_robot_frame = np.mean(points_in_range, axis=0)
        # bifurcation_in_world_frame = bifurcation_in_robot_frame + self.robot_pos[:1]
        return np.mean(points_in_range, axis=0)

    def get_bifurcation_position(self):
        ranges = self.laser['ranges']
        angles = self.laser['angles']

        # The variable 'adjacent_beam_to_check' indicates how many beams in the neighourhood should be checked
        # to check if a beam is or is not a local minima
        adjacent_beam_to_check = 8
        m = int(adjacent_beam_to_check / 2)

        # Calculating local minimums using for scipy
        local_minima_indexes = signal.argrelextrema(ranges, np.less, order=m)
        diff_ranges = ranges[local_minima_indexes]
        diff_angles = angles[local_minima_indexes]

        x_diff = diff_ranges * np.cos(diff_angles)
        y_diff = diff_ranges * np.sin(diff_angles)

        estimated_bifurcation = self.estimate_bifurcation_position(local_minima_x=x_diff,
                                                                   local_minima_y=y_diff)
        return estimated_bifurcation

    def add_node_to_graph(self, node_type):
        original_state = self.state

        if self.last_node_is_dead_end:
            self.last_visited_node = list(self.G.edges(self.node_count))[0][1]
            self.wait_for_new_state(original_state)
            return

        is_dead_end = node_type == 'dead end'
        # angles = []
        # if not is_dead_end:
        #     angles = self.get_corridors_angles()
        x_pos, y_pos, angles = self.wait_for_new_state(original_state, dead_end=is_dead_end)
        self.G.add_node(self.node_count + 1, pos=(x_pos, y_pos), type=node_type, angles=np.round(np.rad2deg(angles)).tolist())
        neighbour_node = self.find_neighbour_of_new_node()
        self.add_edge_to_new_node(neighbour_node)

        self.__update_graph_information()

        self.message_log(f'New node postion: {(x_pos, y_pos)}')
        self.message_log(f"Total Nodes: {self.G.number_of_nodes()}")

        self.draw_graph()

    def get_corridors_angles(self):
        x_size = 15
        y_size = 15
        ranges = self.laser['ranges']
        angles = self.laser['angles']

        robot_angle = self.robot_angles_imu[-1]
        robot_angle = robot_angle if robot_angle > 0 else np.pi - abs(robot_angle)

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        point_cloud = np.column_stack((x, y))
        # Convert point cloud to image
        resolution = 0.1  # meters/pixel
        image_size = (int(x_size / resolution), int(y_size / resolution))  # 15x15 meters
        origin = (image_size[0] // 2, image_size[1] // 2)
        pts = np.round((point_cloud / resolution + np.array(origin)).astype(int))  # pixel coords for the point cloud points
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < image_size[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < image_size[1]), :]
        image = np.zeros(image_size, dtype=np.uint8)
        image[pts[:, 0], pts[:, 1]] = 255

        minLineLength = 8
        lines = cv2.HoughLinesP(image=image, rho=1, theta=np.pi / 180, threshold=10, lines=np.array([]),
                                minLineLength=minLineLength, maxLineGap=5)

        line_angles = []
        if lines is not None:
            lines_reshape = np.reshape(lines, newshape=(lines.shape[0], 4))
            line_angles = []
            for line in lines_reshape:
                # calculate angle
                angle = atan2(line[3] - line[1], line[2] - line[0])
                # TODO: map angles from 0 to 180
                angle = angle if angle > 0 else np.pi - abs(angle)
                angle += robot_angle

                angle = angle if angle <= np.pi else angle - np.pi
                line_angles.append(angle)

            # new_angles = self.remove_repeated_angles(line_angles)

        return line_angles

    def remove_repeated_angles(self, line_angles):
        diff = 15
        print(f'{line_angles = }')
        new_angles = line_angles[:]
        for a1 in range(len(line_angles)):
            for a2 in range(a1 + 1, len(line_angles)):
                if line_angles[a2] in new_angles and (abs(line_angles[a1] - line_angles[a2]) <= np.deg2rad(diff) or abs(line_angles[a1] - line_angles[a2]) >= np.deg2rad(180 - diff)):
                    new_angles.remove(line_angles[a2])
        print(*np.round(np.rad2deg(new_angles)))
        return new_angles

    @staticmethod
    def distance_between_segments(A, B, C, D):
        # Calculate the vectors for each line segment
        AB = (B[0] - A[0], B[1] - A[1])
        CD = (D[0] - C[0], D[1] - C[1])
        AC = (C[0] - A[0], C[1] - A[1])

        # Calculate the length of each line segment
        len_AB = sqrt(AB[0] ** 2 + AB[1] ** 2)
        len_CD = sqrt(CD[0] ** 2 + CD[1] ** 2)

        # If either segment has length 0, return the distance between the two endpoints
        if len_AB == 0:
            return sqrt((C[0] - A[0]) ** 2 + (C[1] - A[1]) ** 2)
        if len_CD == 0:
            return sqrt((D[0] - B[0]) ** 2 + (D[1] - B[1]) ** 2)

        # Calculate the unit vectors for each segment
        u_AB = (AB[0] / len_AB, AB[1] / len_AB)
        u_CD = (CD[0] / len_CD, CD[1] / len_CD)

        # Calculate the dot products of the vectors
        dot_AB_CD = AB[0] * CD[0] + AB[1] * CD[1]
        dot_AB_AC = AB[0] * AC[0] + AB[1] * AC[1]
        dot_CD_AC = CD[0] * -AC[0] + CD[1] * -AC[1]

        # Calculate the distance between the segments
        if dot_AB_CD == 0 and dot_AB_AC == 0:
            # The segments are collinear and overlapping, so the distance is 0
            return 0
        if dot_AB_CD == 0:
            # The segments are perpendicular, so the distance is the distance between one endpoint of one segment and the other segment
            return sqrt(
                min((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2, (B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2, (A[0] - D[0]) ** 2 + (A[1] - D[1]) ** 2,
                    (B[0] - D[0]) ** 2 + (B[1] - D[1]) ** 2))
        s_CD = dot_CD_AC / dot_AB_CD
        s_AB = dot_AB_AC / dot_AB_CD

        # Check if the intersection point is within the bounds of both segments
        if 0 <= s_CD <= 1 and 0 <= s_AB <= 1:
            # The segments intersect, so the distance is 0
            return 0

        # Calculate the distance between the closest endpoints of the two segments
        return min(sqrt((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2), sqrt((B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2),
                   sqrt((A[0] - D[0]) ** 2 + (A[1] - D[1]) ** 2), sqrt((B[0] - D[0]) ** 2 + (B[1] - D[1]) ** 2))

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
        x_pos = []
        y_pos = []
        if dead_end:
            x_pos = [self.robot_pos[0]]
            y_pos = [self.robot_pos[1]]
            return np.round(x_pos[0], 2), np.round(y_pos[0], 2), []

        angles = []
        angles_dict = {'count': 1}
        while self.state == original_state:
            x_bifurcation, y_bifurcation = self.get_bifurcation_position()
            if x_bifurcation is None:
                continue
            # print('getting lines')
            angles.extend(self.get_corridors_angles())
            angles = self.remove_repeated_angles(angles)
            # print('done getting lines')
            # =========
            # TODO: para checar se os ângulos de corredores são válidos: ver quantas vezes o ângulo foi medido. se for menor
            # do que um limite (i.e. 70%) não considerar.
            if not angles_dict:
                angles_dict = {angle: 1 for angle in angles}
            else:
                print(angles_dict)
                angles_dict['count'] += 1
                for angle in angles:
                    if angle in angles_dict:
                        angles_dict[angle] += 1
                    else:
                        angles_dict[angle] = 1

            x_pos.append(x_bifurcation)
            y_pos.append(y_bifurcation)
            # print('drawing graph')
            self.draw_graph()
            # print('done drawing graph')

        print(f'Number of measurements in bifurcation: {angles_dict["count"]}')

        filtered_angles = []
        for angle_key in angles_dict.keys():
            if angle_key == 'count':
                continue
            if angles_dict[angle_key] / angles_dict['count'] > 0.7:
                filtered_angles.append(angle_key)

        print('removing repeated angles')
        # new_angles = self.remove_repeated_angles(angles)
        print('done removing repeated angles')

        node_position = np.round((np.mean(x_pos), np.mean(y_pos)), 2)
        return node_position[0], node_position[1], filtered_angles

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
                    # Considering the new node position as a mean of the positions of the repeated nodes
                    self.G.nodes[i]['pos'] = (np.mean([node_positions[i][0], node_positions[j][0]]),
                                              np.mean([node_positions[i][1], node_positions[j][1]]))
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
        labels = nx.get_node_attributes(self.G, 'angles')

        nx.draw(self.G, pos, node_color=color_map)
        nx.draw_networkx_labels(self.G, pos, labels)

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
    # right_angles indicates if the intersections are in a 90-degree angle
    service = TopologicalMapping(name=node_name, right_angles=False)
    service.main_service()

    print(f"{node_name} node stopped")
