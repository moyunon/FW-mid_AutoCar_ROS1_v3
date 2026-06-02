#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import math
import os

import cv2
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan, GetPlanResponse


class AStarPlanner:
    def __init__(self):
        self.map_source = rospy.get_param('~map_source', 'static')  # static 或 realtime
        self.map_yaml_path = rospy.get_param('~map_yaml_path', '')
        self.realtime_map_topic = rospy.get_param('~realtime_map_topic', '/fast_lio_2d_map')
        self.robot_radius = rospy.get_param('~robot_radius_m', 0.35)
        self.safety_margin = rospy.get_param('~safety_margin_m', 0.10)
        self.unknown_as_obstacle = rospy.get_param('~unknown_as_obstacle', False)
        self.auto_forbidden_enabled = rospy.get_param('~auto_forbidden_enabled', True)
        self.auto_forbidden_window_m = rospy.get_param('~auto_forbidden_window_m', 1.0)
        self.auto_forbidden_min_points = rospy.get_param('~auto_forbidden_min_points', 8)
        self.auto_forbidden_close_m = rospy.get_param('~auto_forbidden_close_m', 0.8)
        self.auto_forbidden_min_area_m2 = rospy.get_param('~auto_forbidden_min_area_m2', 0.5)
        self.auto_forbidden_padding_m = rospy.get_param('~auto_forbidden_padding_m', 0.2)

        self.resolution = None
        self.origin = [0.0, 0.0, 0.0]
        self.width = 0
        self.height = 0
        self.grid_map = None
        self.costmap = None
        self.map_frame = 'map'

        self.map_pub = rospy.Publisher('~map', OccupancyGrid, queue_size=1, latch=True)
        self.costmap_pub = rospy.Publisher('~costmap', OccupancyGrid, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher('~visual_plan', Path, queue_size=1)
        self.plan_srv = rospy.Service('~get_plan', GetPlan, self.plan_cb)

        if self.map_source == 'static':
            self.load_static_map(self.map_yaml_path)
        elif self.map_source == 'realtime':
            rospy.Subscriber(self.realtime_map_topic, OccupancyGrid, self.realtime_map_cb, queue_size=1)
            rospy.loginfo(f"A* 使用实时地图: {self.realtime_map_topic}")
        else:
            rospy.logerr("~map_source 只能是 static 或 realtime")

        rospy.Timer(rospy.Duration(2.0), self.publish_maps)
        rospy.loginfo("A* 规划器已启动，等待规划请求")

    def load_static_map(self, yaml_path):
        if not os.path.exists(yaml_path):
            rospy.logerr(f"找不到静态地图文件: {yaml_path}")
            return

        with open(yaml_path, 'r', encoding='utf-8') as f:
            map_data = yaml.safe_load(f)

        image_path = os.path.join(os.path.dirname(yaml_path), map_data['image'])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            rospy.logerr(f"无法读取地图图片: {image_path}")
            return

        img = cv2.flip(img, 0)
        self.resolution = float(map_data['resolution'])
        self.origin = list(map_data['origin'])
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.map_frame = rospy.get_param('~map_frame', 'map')

        # PGM 中黑色为障碍，白色为可通行
        grid = np.zeros((self.height, self.width), dtype=np.int16)
        grid[img < 128] = 100
        grid[img >= 128] = 0
        self.set_grid_map(grid)
        rospy.loginfo(f"静态地图加载完成: {yaml_path}")

    def realtime_map_cb(self, msg):
        if msg.info.width == 0 or msg.info.height == 0:
            return

        data = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        grid = np.zeros_like(data, dtype=np.int16)
        grid[data > 50] = 100
        if self.unknown_as_obstacle:
            grid[data < 0] = 100

        self.resolution = float(msg.info.resolution)
        self.origin = [
            msg.info.origin.position.x,
            msg.info.origin.position.y,
            0.0,
        ]
        self.width = msg.info.width
        self.height = msg.info.height
        self.map_frame = msg.header.frame_id or 'map'
        self.set_grid_map(grid)

    def set_grid_map(self, grid):
        self.grid_map = grid.astype(np.int16)
        self.apply_dense_forbidden_zones(self.grid_map)
        self.costmap = self.build_costmap(self.grid_map)

    def apply_dense_forbidden_zones(self, grid):
        if not self.auto_forbidden_enabled:
            return

        occupied = np.zeros_like(grid, dtype=np.uint8)
        occupied[grid > 50] = 1
        if np.count_nonzero(occupied) == 0:
            return

        # 用局部窗口统计障碍物点密度，密集区域视为停车区/杂物区候选
        window_cells = max(1, int(math.ceil(self.auto_forbidden_window_m / self.resolution)))
        density = cv2.boxFilter(
            occupied,
            ddepth=cv2.CV_32S,
            ksize=(window_cells, window_cells),
            normalize=False,
        )
        dense_mask = np.zeros_like(occupied, dtype=np.uint8)
        dense_mask[density >= self.auto_forbidden_min_points] = 255

        # 闭运算把车与车之间的小缝连接起来，再对连通块取矩形包络
        close_cells = max(1, int(math.ceil(self.auto_forbidden_close_m / self.resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_cells, close_cells))
        dense_mask = cv2.morphologyEx(dense_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(dense_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area_cells = max(1, int(self.auto_forbidden_min_area_m2 / (self.resolution ** 2)))
        padding_cells = max(0, int(math.ceil(self.auto_forbidden_padding_m / self.resolution)))

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < min_area_cells:
                continue

            x0 = max(0, x - padding_cells)
            y0 = max(0, y - padding_cells)
            x1 = min(self.width, x + w + padding_cells)
            y1 = min(self.height, y + h + padding_cells)
            grid[y0:y1, x0:x1] = 100

    def build_costmap(self, grid):
        total_inflation_m = self.robot_radius + self.safety_margin
        inflation_cells = int(math.ceil(total_inflation_m / self.resolution))
        if inflation_cells <= 0:
            return grid.copy()

        kernel_size = inflation_cells * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        occupied = np.zeros_like(grid, dtype=np.uint8)
        occupied[grid > 50] = 100
        return cv2.dilate(occupied, kernel).astype(np.int16)

    def plan_cb(self, request):
        response = GetPlanResponse()
        if self.grid_map is None or self.costmap is None:
            rospy.logwarn("规划失败：地图尚未就绪")
            return response

        start_grid = self.world_to_grid(request.start.pose.position.x, request.start.pose.position.y)
        goal_grid = self.world_to_grid(request.goal.pose.position.x, request.goal.pose.position.y)

        if not self.is_free(start_grid):
            rospy.logwarn("规划失败：起点不可通行或超出地图")
            return response
        if not self.is_free(goal_grid):
            rospy.logwarn("规划失败：终点不可通行或超出地图")
            return response

        path_grid = self.a_star(start_grid, goal_grid)
        if not path_grid:
            rospy.logerr("规划失败：无法找到安全路径")
            return response

        path_msg = self.create_path_msg(path_grid)
        response.plan = path_msg
        self.path_pub.publish(path_msg)
        rospy.loginfo(f"规划成功，路点数: {len(path_grid)}")
        return response

    def a_star(self, start, goal):
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_free(neighbor):
                    continue

                step_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                new_cost = g_score[current] + step_cost
                if neighbor not in g_score or new_cost < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = new_cost
                    heuristic = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                    heapq.heappush(open_set, (new_cost + heuristic, neighbor))
        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def is_valid_grid(self, grid):
        return 0 <= grid[0] < self.width and 0 <= grid[1] < self.height

    def is_free(self, grid):
        return self.is_valid_grid(grid) and self.costmap[grid[1], grid[0]] <= 50

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.origin[0]) / self.resolution)
        gy = int((wy - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = gx * self.resolution + self.origin[0]
        wy = gy * self.resolution + self.origin[1]
        return wx, wy

    def create_path_msg(self, path_grid):
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.Time.now()
        for gx, gy in path_grid:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = self.grid_to_world(gx, gy)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def publish_maps(self, event=None):
        if self.grid_map is None or self.costmap is None:
            return
        self.map_pub.publish(self.create_grid_msg(self.grid_map))
        self.costmap_pub.publish(self.create_grid_msg(self.costmap))

    def create_grid_msg(self, data_array):
        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.Time.now()
        msg.info.resolution = float(self.resolution)
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin[0]
        msg.info.origin.position.y = self.origin[1]
        msg.info.origin.orientation.w = 1.0
        msg.data = data_array.astype(np.int8).flatten().tolist()
        return msg


if __name__ == '__main__':
    try:
        rospy.init_node('astar_planner_node')
        AStarPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
