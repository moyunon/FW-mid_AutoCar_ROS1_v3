#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan, GetPlanResponse
from geometry_msgs.msg import PoseStamped
import yaml
import cv2
import numpy as np
import heapq
import os
import math

class AStarPlanner:
    def __init__(self):
        # ================= 声明参数：考虑小车体积 =================
        map_yaml_path = rospy.get_param('~map_yaml_path', '')
        # 小车的外接圆半径 (例如：长0.6m宽0.4m的车，外接圆半径约0.36m)
        self.robot_radius = rospy.get_param('~robot_radius_m', 0.35) 
        # 额外的安全贴边距离
        self.safety_margin = rospy.get_param('~safety_margin_m', 0.10)
        
        if not os.path.exists(map_yaml_path):
            rospy.logerr(f"找不到地图文件: {map_yaml_path}")
            return
            
        # ================= 代价地图构建 =================
        self.load_and_process_map(map_yaml_path)
        
        # ================= ROS 1 接口定义 =================
        self.map_pub = rospy.Publisher('~map', OccupancyGrid, queue_size=1)
        self.costmap_pub = rospy.Publisher('~costmap', OccupancyGrid, queue_size=1)
        self.path_pub = rospy.Publisher('~visual_plan', Path, queue_size=1)
        
        self.plan_srv = rospy.Service('~get_plan', GetPlan, self.plan_cb)
        
        # 2秒一次的定时器，用于发布地图供 RViz 显示
        rospy.Timer(rospy.Duration(2.0), self.publish_maps)
        rospy.loginfo("考虑小车体积的 A* 规划器服务已启动！等待请求...")

    def load_and_process_map(self, yaml_path):
        with open(yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)
            
        self.resolution = map_data['resolution']
        self.origin = map_data['origin']
        image_name = map_data['image']
        
        map_dir = os.path.dirname(yaml_path)
        image_path = os.path.join(map_dir, image_name)
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.flip(img, 0) 
        
        self.width = img.shape[1]
        self.height = img.shape[0]
        
        self.grid_map = np.zeros((self.height, self.width), dtype=np.uint8)
        self.grid_map[img < 128] = 100
        self.grid_map[img >= 128] = 0   
        
        # ================= 核心修改：基于小车体积计算膨胀 =================
        # 总膨胀距离 = 小车半径 + 安全距离
        total_inflation_m = self.robot_radius + self.safety_margin
        # 将物理距离转换为地图上的栅格数
        inflation_cells = int(math.ceil(total_inflation_m / self.resolution))
        
        # 使用椭圆形态学核 (在宽高相等时表现为平滑的圆形核)
        kernel_size = inflation_cells * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 执行膨胀操作
        self.costmap = cv2.dilate(self.grid_map, kernel)
        
        rospy.loginfo(f"地图加载完毕。小车半径: {self.robot_radius}m, 安全余量: {self.safety_margin}m")
        rospy.loginfo(f"实际总膨胀半径: {total_inflation_m:.2f}m (折合 {inflation_cells} 个栅格)")

    def plan_cb(self, request):
        rospy.loginfo("收到规划请求...")
        response = GetPlanResponse()
        
        start_grid = self.world_to_grid(request.start.pose.position.x, request.start.pose.position.y)
        goal_grid = self.world_to_grid(request.goal.pose.position.x, request.goal.pose.position.y)
        
        if not self.is_valid_grid(start_grid) or self.costmap[start_grid[1], start_grid[0]] > 50:
            rospy.logwarn("规划失败：起点在障碍物内或距障碍物太近（侵入小车安全体积）！")
            return response
            
        if not self.is_valid_grid(goal_grid) or self.costmap[goal_grid[1], goal_grid[0]] > 50:
            rospy.logwarn("规划失败：终点在障碍物内或距障碍物太近（侵入小车安全体积）！")
            return response
            
        path_grid = self.a_star(start_grid, goal_grid)
        
        if path_grid:
            rospy.loginfo(f"规划成功！生成路点数: {len(path_grid)}")
            path_msg = self.create_path_msg(path_grid)
            response.plan = path_msg
            self.path_pub.publish(path_msg)
        else:
            rospy.logerr("规划失败：无法找到安全的连通路径。")
            
        return response

    def a_star(self, start, goal):
        neighbors = [(0,1),(1,0),(0,-1),(-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        open_set = []
        heapq.heappush(open_set, (0, start[0], start[1]))
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, current_x, current_y = heapq.heappop(open_set)
            current = (current_x, current_y)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
                
            for dx, dy in neighbors:
                neighbor = (current_x + dx, current_y + dy)
                if not self.is_valid_grid(neighbor): continue
                # 如果碰到代价地图上的膨胀区，视为碰撞
                if self.costmap[neighbor[1], neighbor[0]] > 50: continue
                    
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h_score = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                    f_score = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score, neighbor[0], neighbor[1]))
        return None

    def is_valid_grid(self, grid):
        return 0 <= grid[0] < self.width and 0 <= grid[1] < self.height

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.origin[0]) / self.resolution)
        gy = int((wy - self.origin[1]) / self.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        wx = gx * self.resolution + self.origin[0]
        wy = gy * self.resolution + self.origin[1]
        return (wx, wy)

    def create_path_msg(self, path_grid):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = rospy.Time.now()
        for p in path_grid:
            pose = PoseStamped()
            pose.header = msg.header
            wx, wy = self.grid_to_world(p[0], p[1])
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def publish_maps(self, event=None):
        def create_grid_msg(data_array):
            msg = OccupancyGrid()
            msg.header.frame_id = 'map'
            msg.header.stamp = rospy.Time.now()
            msg.info.resolution = float(self.resolution)
            msg.info.width = self.width
            msg.info.height = self.height
            msg.info.origin.position.x = self.origin[0]
            msg.info.origin.position.y = self.origin[1]
            msg.info.origin.orientation.w = 1.0
            msg.data = data_array.flatten().tolist()
            return msg
        self.map_pub.publish(create_grid_msg(self.grid_map))
        self.costmap_pub.publish(create_grid_msg(self.costmap))

if __name__ == '__main__':
    try:
        rospy.init_node('astar_planner_node')
        node = AStarPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan, GetPlanResponse
from geometry_msgs.msg import PoseStamped
import yaml
import cv2
import numpy as np
import heapq
import os
import math

class AStarPlanner:
    def __init__(self):
        # ================= 声明参数：考虑小车体积 =================
        map_yaml_path = rospy.get_param('~map_yaml_path', '')
        # 小车的外接圆半径 (例如：长0.6m宽0.4m的车，外接圆半径约0.36m)
        self.robot_radius = rospy.get_param('~robot_radius_m', 0.35) 
        # 额外的安全贴边距离
        self.safety_margin = rospy.get_param('~safety_margin_m', 0.10)
        
        if not os.path.exists(map_yaml_path):
            rospy.logerr(f"找不到地图文件: {map_yaml_path}")
            return
            
        # ================= 代价地图构建 =================
        self.load_and_process_map(map_yaml_path)
        
        # ================= ROS 1 接口定义 =================
        self.map_pub = rospy.Publisher('~map', OccupancyGrid, queue_size=1)
        self.costmap_pub = rospy.Publisher('~costmap', OccupancyGrid, queue_size=1)
        self.path_pub = rospy.Publisher('~visual_plan', Path, queue_size=1)
        
        self.plan_srv = rospy.Service('~get_plan', GetPlan, self.plan_cb)
        
        # 2秒一次的定时器，用于发布地图供 RViz 显示
        rospy.Timer(rospy.Duration(2.0), self.publish_maps)
        rospy.loginfo("考虑小车体积的 A* 规划器服务已启动！等待请求...")

    def load_and_process_map(self, yaml_path):
        with open(yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)
            
        self.resolution = map_data['resolution']
        self.origin = map_data['origin']
        image_name = map_data['image']
        
        map_dir = os.path.dirname(yaml_path)
        image_path = os.path.join(map_dir, image_name)
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.flip(img, 0) 
        
        self.width = img.shape[1]
        self.height = img.shape[0]
        
        self.grid_map = np.zeros((self.height, self.width), dtype=np.uint8)
        self.grid_map[img < 128] = 100
        self.grid_map[img >= 128] = 0   
        
        # ================= 核心修改：基于小车体积计算膨胀 =================
        # 总膨胀距离 = 小车半径 + 安全距离
        total_inflation_m = self.robot_radius + self.safety_margin
        # 将物理距离转换为地图上的栅格数
        inflation_cells = int(math.ceil(total_inflation_m / self.resolution))
        
        # 使用椭圆形态学核 (在宽高相等时表现为平滑的圆形核)
        kernel_size = inflation_cells * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 执行膨胀操作
        self.costmap = cv2.dilate(self.grid_map, kernel)
        
        rospy.loginfo(f"地图加载完毕。小车半径: {self.robot_radius}m, 安全余量: {self.safety_margin}m")
        rospy.loginfo(f"实际总膨胀半径: {total_inflation_m:.2f}m (折合 {inflation_cells} 个栅格)")

    def plan_cb(self, request):
        rospy.loginfo("收到规划请求...")
        response = GetPlanResponse()
        
        start_grid = self.world_to_grid(request.start.pose.position.x, request.start.pose.position.y)
        goal_grid = self.world_to_grid(request.goal.pose.position.x, request.goal.pose.position.y)
        
        if not self.is_valid_grid(start_grid) or self.costmap[start_grid[1], start_grid[0]] > 50:
            rospy.logwarn("规划失败：起点在障碍物内或距障碍物太近（侵入小车安全体积）！")
            return response
            
        if not self.is_valid_grid(goal_grid) or self.costmap[goal_grid[1], goal_grid[0]] > 50:
            rospy.logwarn("规划失败：终点在障碍物内或距障碍物太近（侵入小车安全体积）！")
            return response
            
        path_grid = self.a_star(start_grid, goal_grid)
        
        if path_grid:
            rospy.loginfo(f"规划成功！生成路点数: {len(path_grid)}")
            path_msg = self.create_path_msg(path_grid)
            response.plan = path_msg
            self.path_pub.publish(path_msg)
        else:
            rospy.logerr("规划失败：无法找到安全的连通路径。")
            
        return response

    def a_star(self, start, goal):
        neighbors = [(0,1),(1,0),(0,-1),(-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        open_set = []
        heapq.heappush(open_set, (0, start[0], start[1]))
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, current_x, current_y = heapq.heappop(open_set)
            current = (current_x, current_y)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
                
            for dx, dy in neighbors:
                neighbor = (current_x + dx, current_y + dy)
                if not self.is_valid_grid(neighbor): continue
                # 如果碰到代价地图上的膨胀区，视为碰撞
                if self.costmap[neighbor[1], neighbor[0]] > 50: continue
                    
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h_score = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                    f_score = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score, neighbor[0], neighbor[1]))
        return None

    def is_valid_grid(self, grid):
        return 0 <= grid[0] < self.width and 0 <= grid[1] < self.height

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.origin[0]) / self.resolution)
        gy = int((wy - self.origin[1]) / self.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        wx = gx * self.resolution + self.origin[0]
        wy = gy * self.resolution + self.origin[1]
        return (wx, wy)

    def create_path_msg(self, path_grid):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = rospy.Time.now()
        for p in path_grid:
            pose = PoseStamped()
            pose.header = msg.header
            wx, wy = self.grid_to_world(p[0], p[1])
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def publish_maps(self, event=None):
        def create_grid_msg(data_array):
            msg = OccupancyGrid()
            msg.header.frame_id = 'map'
            msg.header.stamp = rospy.Time.now()
            msg.info.resolution = float(self.resolution)
            msg.info.width = self.width
            msg.info.height = self.height
            msg.info.origin.position.x = self.origin[0]
            msg.info.origin.position.y = self.origin[1]
            msg.info.origin.orientation.w = 1.0
            msg.data = data_array.flatten().tolist()
            return msg
        self.map_pub.publish(create_grid_msg(self.grid_map))
        self.costmap_pub.publish(create_grid_msg(self.costmap))

if __name__ == '__main__':
    try:
        rospy.init_node('astar_planner_node')
        node = AStarPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
