#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger, TriggerResponse


class FastLio2DMapper:
    def __init__(self):
        self.cloud_topic = rospy.get_param('~cloud_topic', '/b/cloud_registered')
        self.map_topic = rospy.get_param('~map_topic', '/fast_lio_2d_map')
        self.output_dir = rospy.get_param('~output_dir', '')
        self.map_name = rospy.get_param('~map_name', 'map_2d')
        self.frame_id = rospy.get_param('~frame_id', 'map')

        self.resolution = rospy.get_param('~resolution', 0.05)
        self.width_m = rospy.get_param('~width_m', 60.0)
        self.height_m = rospy.get_param('~height_m', 40.0)
        self.origin_x = rospy.get_param('~origin_x', -30.0)
        self.origin_y = rospy.get_param('~origin_y', -20.0)
        self.z_min = rospy.get_param('~z_min', 0.15)
        self.z_max = rospy.get_param('~z_max', 2.0)
        self.min_points_per_cell = rospy.get_param('~min_points_per_cell', 1)
        self.publish_rate = rospy.get_param('~publish_rate', 2.0)
        self.unknown_value = rospy.get_param('~unknown_value', 0)

        self.width = int(math.ceil(self.width_m / self.resolution))
        self.height = int(math.ceil(self.height_m / self.resolution))
        self.hit_counts = np.zeros((self.height, self.width), dtype=np.uint16)
        self.grid = self.create_grid()

        self.map_pub = rospy.Publisher(self.map_topic, OccupancyGrid, queue_size=1, latch=True)
        self.cloud_sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_cb, queue_size=1)
        self.save_srv = rospy.Service('~save_map', Trigger, self.save_map_cb)
        self.clear_srv = rospy.Service('~clear_map', Trigger, self.clear_map_cb)
        rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 0.1)), self.publish_map)

        rospy.loginfo(f"二维建图启动: {self.cloud_topic} -> {self.map_topic}")
        rospy.loginfo("保存地图: rosservice call /fast_lio_2d_mapper/save_map")

    def create_grid(self):
        value = int(self.unknown_value)
        if value not in (-1, 0):
            value = 0
        return np.full((self.height, self.width), value, dtype=np.int16)

    def cloud_cb(self, msg):
        changed = False
        for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            # 建图只记录高于地面的障碍物点
            if z < self.z_min or z > self.z_max:
                continue

            gx = int((x - self.origin_x) / self.resolution)
            gy = int((y - self.origin_y) / self.resolution)
            if 0 <= gx < self.width and 0 <= gy < self.height:
                if self.hit_counts[gy, gx] < 65535:
                    self.hit_counts[gy, gx] += 1
                if self.hit_counts[gy, gx] >= self.min_points_per_cell:
                    self.grid[gy, gx] = 100
                    changed = True

        if changed:
            self.map_pub.publish(self.create_msg())

    def publish_map(self, event=None):
        self.map_pub.publish(self.create_msg())

    def create_msg(self):
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.info.resolution = float(self.resolution)
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = self.grid.astype(np.int8).flatten().tolist()
        return msg

    def clear_map_cb(self, request):
        self.hit_counts.fill(0)
        self.grid = self.create_grid()
        self.publish_map()
        return TriggerResponse(success=True, message="二维地图已清空")

    def save_map_cb(self, request):
        if not self.output_dir:
            return TriggerResponse(success=False, message="未设置 output_dir")

        os.makedirs(self.output_dir, exist_ok=True)
        pgm_path = os.path.join(self.output_dir, self.map_name + '.pgm')
        yaml_path = os.path.join(self.output_dir, self.map_name + '.yaml')

        self.write_pgm(pgm_path)
        self.write_yaml(yaml_path, os.path.basename(pgm_path))
        return TriggerResponse(success=True, message=f"地图已保存: {yaml_path}")

    def write_pgm(self, path):
        image = np.full((self.height, self.width), 254, dtype=np.uint8)
        image[self.grid < 0] = 205
        image[self.grid > 50] = 0
        image = np.flipud(image)

        with open(path, 'wb') as f:
            f.write(f"P5\n{self.width} {self.height}\n255\n".encode('ascii'))
            f.write(image.tobytes())

    def write_yaml(self, path, image_name):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"image: {image_name}\n")
            f.write("mode: trinary\n")
            f.write(f"resolution: {self.resolution}\n")
            f.write(f"origin: [{self.origin_x}, {self.origin_y}, 0]\n")
            f.write("negate: 0\n")
            f.write("occupied_thresh: 0.65\n")
            f.write("free_thresh: 0.25\n")


if __name__ == '__main__':
    try:
        rospy.init_node('fast_lio_2d_mapper')
        FastLio2DMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
