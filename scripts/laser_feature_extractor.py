#!/usr/bin/env python
"""
This is a script to plot the data from a planar laser in euclidian coordinates. It also has two main pruporses:
    1) Local minimum identification
    2) Corner detection (it is still a very naive approach)

Author: Israel Amaral
"""

import rospy
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats, signal
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from sensor_msgs.msg import LaserScan, JointState, Imu


class DataPlotter:
    LASER_MAX_RANGE = 50

    def __init__(self):
        self.min_dist_to_detect_bifurcation = 2.5
        self.min_dist_to_dead_end = 1.2
        self.laser = {'ranges': np.array([]),
                      'angles': np.array([]),
                      'angle_min': 0.0,
                      'angle_max': 0.0,
                      'angle_increment': 0.0}

        self.distance = 0.0
        rospy.init_node('laser_line_plot', anonymous=True)

        # Subscribed topics
        rospy.Subscriber("/scan", LaserScan, self.callback_laser)

        self.started_laser = False

        self.figure = plt.figure(figsize=(12, 12))

        self.layout_is_tightened = False

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

    def get_cartesian_data_position(self, range_data=None, angle_data=None):
        """
        Calculates the (x,y) coordinates of the laser data given in polar coordinates
        Args:
            range_data: range vector
            angle_data: angle vector

        Returns:

        """
        ranges, angles = self.laser['ranges'], self.laser['angles']
        if range_data is not None and angle_data is not None:
            ranges, angles = range_data, angle_data
        x_data = ranges * np.cos(angles)
        y_data = ranges * np.sin(angles)
        return x_data, y_data

    def update(self, num):
        plt.clf()

        ranges = self.laser['ranges']
        angles = self.laser['angles']

        # The variable 'adjacent_beam_to_check' indicates how many beams in the neighourhood should be checked
        # to verify if a beam is or is not a local minima
        adjacent_beam_to_check = 8
        m = int(adjacent_beam_to_check / 2)

        # Calculating local minimums using scipy argrelextrema
        local_minima_indexes = signal.argrelextrema(ranges, np.less, order=m)

        local_minima_ranges = ranges[local_minima_indexes]
        local_minima_angles = angles[local_minima_indexes]

        x_local_minima = local_minima_ranges * np.cos(local_minima_angles)
        y_local_minima = local_minima_ranges * np.sin(local_minima_angles)

        x_laser, y_laser = self.get_cartesian_data_position(range_data=ranges, angle_data=angles)
        plt.subplot(211)
        plt.title('Laser Measurements')
        plt.ylabel('$y$')
        plt.xlabel('$x$')

        # Map limits
        map_lim = 8
        plt.ylim([-map_lim, map_lim])
        plt.xlim([-map_lim, map_lim])

        plt.scatter([0], [0], color='black', marker='s', label='Robot')
        plt.scatter(x_laser, y_laser, label='Laser Data', s=3)
        plt.scatter(x_local_minima, y_local_minima, color='red', label='Local Minima', s=30)

        # Calculating the absolute diffence of the beams ranges between
        # to their adjacent neighbours
        delta_x = np.abs(np.diff(x_laser))
        delta_y = np.abs(np.diff(y_laser))

        # Numpy diff is given by out[i] = a[i+1] - a[i]. In the second half of the vector, I desire to consider
        # out[i] = a[i] - a[i-1] instead so the corners are correctly identified
        half = int(x_laser.shape[0] / 2)
        shifted_delta_x = np.hstack([[0], delta_x[half:]])[:-1]
        shifted_delta_y = np.hstack([[0], delta_y[half:]])[:-1]
        delta_x = np.hstack([delta_x[:half], shifted_delta_x])
        delta_y = np.hstack([delta_y[:half], shifted_delta_y])
        vector_of_differences = np.vstack((x_laser, y_laser)).T

        # Defining the corners: every measurement that has a z-score of
        # the difference to its adjacent neighbhour grater than 'z_score_threshold' is a corner
        z_score_threshold = 5
        corner_indexes = stats.zscore(delta_x) > z_score_threshold, stats.zscore(delta_y) > z_score_threshold
        corner_points = vector_of_differences[0:-1][corner_indexes]

        plt.scatter(corner_points[:, 0], corner_points[:, 1], color='orange', marker='X', label='Corners')

        plt.legend(loc='upper right')
        plt.gca().set_aspect('equal', adjustable='box')

        plt.subplot(212)
        plt.title('Adjacent Beams Difference')
        plt.plot(delta_x, label='$x$ diff')
        plt.plot(delta_y, label='$y$ diff')

        a_x = delta_x[corner_indexes]
        a_y = delta_y[corner_indexes]

        plt.scatter(np.linspace(start=-1, stop=delta_x.shape[0], num=delta_x.shape[0])[corner_indexes], a_x,
                    marker='x', label='Corners $x$ coord.')
        plt.scatter(np.linspace(start=-1, stop=delta_y.shape[0], num=delta_y.shape[0])[corner_indexes], a_y,
                    marker='x', label='Corners $y$ coord.')

        plt.legend()

        plt.ylabel('Difference')
        plt.xlabel('Laser Beam Index')

        if not self.layout_is_tightened:
            plt.tight_layout()
            self.layout_is_tightened = True
        return


if __name__ == '__main__':
    save_animation = False
    try:
        service = DataPlotter()
        print(f"Laser data ready: {service.started_laser}")

        while not service.started_laser:
            continue

        print('Everything good to go!')

        if save_animation:
            service.animation = FuncAnimation(service.figure, service.update, interval=50, save_count=4000)
            f = "laser_feature_extractor.mp4"
            write_mp4 = animation.FFMpegWriter(fps=25)
            service.animation.save(f, writer=write_mp4)
        else:
            service.animation = FuncAnimation(service.figure, service.update, interval=5)
        plt.show()
    except rospy.ROSInterruptException:
        pass
