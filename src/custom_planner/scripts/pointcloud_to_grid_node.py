#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2


class PointCloudToGrid:
    def __init__(self):
        self.cloud_topic = rospy.get_param('~cloud_topic', '/b/cloud_registered')
        self.map_topic = rospy.get_param('~map_topic', '/fast_lio_2d_map')
        self.output_frame = rospy.get_param('~output_frame', 'map')

        self.resolution = rospy.get_param('~resolution', 0.05)
        self.width_m = rospy.get_param('~width_m', 60.0)
        self.height_m = rospy.get_param('~height_m', 40.0)
        self.origin_x = rospy.get_param('~origin_x', -30.0)
        self.origin_y = rospy.get_param('~origin_y', -20.0)

        self.z_min = rospy.get_param('~z_min', 0.15)
        self.z_max = rospy.get_param('~z_max', 2.0)
        self.publish_rate = rospy.get_param('~publish_rate', 5.0)
        self.base_value = rospy.get_param('~base_value', 0)  # 0 表示默认空闲，-1 表示默认未知
        self.min_points_per_cell = rospy.get_param('~min_points_per_cell', 1)

        self.width = int(math.ceil(self.width_m / self.resolution))
        self.height = int(math.ceil(self.height_m / self.resolution))
        self.latest_grid = self.create_empty_grid()

        self.map_pub = rospy.Publisher(self.map_topic, OccupancyGrid, queue_size=1, latch=True)
        rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 0.1)), self.publish_map)
        rospy.loginfo(f"点云转二维栅格节点启动: {self.cloud_topic} -> {self.map_topic}")

    def create_empty_grid(self):
        grid = np.full((self.height, self.width), int(self.base_value), dtype=np.int16)
        if self.base_value not in (-1, 0):
            grid.fill(0)
        return grid

    def cloud_cb(self, msg):
        counts = np.zeros((self.height, self.width), dtype=np.uint16)

        # 只取一定高度范围内的点，避免地面和过高结构影响二维障碍物
        for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            if z < self.z_min or z > self.z_max:
                continue

            gx = int((x - self.origin_x) / self.resolution)
            gy = int((y - self.origin_y) / self.resolution)
            if 0 <= gx < self.width and 0 <= gy < self.height:
                counts[gy, gx] += 1

        grid = self.create_empty_grid()
        grid[counts >= self.min_points_per_cell] = 100
        self.latest_grid = grid

    def publish_map(self, event=None):
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.output_frame
        msg.info.resolution = float(self.resolution)
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = self.latest_grid.astype(np.int8).flatten().tolist()
        self.map_pub.publish(msg)


if __name__ == '__main__':
    try:
        rospy.init_node('pointcloud_to_grid_node')
        PointCloudToGrid()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
