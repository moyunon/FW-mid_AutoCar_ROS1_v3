#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import threading

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


class LocalNavigator:
    def __init__(self):
        self.max_v = rospy.get_param('~max_speed_x', 1.0)
        self.max_vy = rospy.get_param('~max_speed_y', 0.5)
        self.max_w = rospy.get_param('~max_yaw_rate', 1.0)
        self.robot_radius = rospy.get_param('~robot_radius', 0.4)
        self.ground_z = rospy.get_param('~ground_filter_z', 0.15)
        self.obstacle_z_max = rospy.get_param('~obstacle_z_max', 2.0)
        self.pointcloud_topic = rospy.get_param('~pointcloud_topic', '/b/cloud_registered_body')
        self.odom_topic = rospy.get_param('~odom_topic', '/b/Odometry')
        self.use_fake_pose = rospy.get_param('~use_fake_pose', False)

        self.cmd_pub = rospy.Publisher('/fw_mid/command_dict', String, queue_size=10)
        self.goal_sub = rospy.Subscriber('/goal_pose', PoseStamped, self.goal_cb)
        self.pc_sub = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.pointcloud_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=10)

        self.plan_srv_name = '/astar_planner_node/get_plan'
        self.state = 'WAITING'
        self.global_path = []
        self.current_pose = (0.0, 0.0, 0.0)
        self.obstacles = []

        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.loginfo(f"局部导航启动，点云: {self.pointcloud_topic}, 里程计: {self.odom_topic}")

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_pose = (x, y, yaw)

    def pointcloud_cb(self, msg):
        obs_list = []
        for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            # 只把高于地面、低于最高障碍物阈值的点作为局部障碍
            if self.ground_z < z < self.obstacle_z_max:
                obs_list.append((x, y))
        self.obstacles = obs_list

    def goal_cb(self, msg):
        rospy.loginfo("收到目标点，开始请求静态地图上的全局路径")
        self.state = 'PLANNING'
        self.request_global_plan(msg)

    def request_global_plan(self, goal_pose):
        def call_service_thread():
            try:
                rospy.wait_for_service(self.plan_srv_name, timeout=5.0)
                plan_client = rospy.ServiceProxy(self.plan_srv_name, GetPlan)

                req = GetPlanRequest()
                req.start.header.frame_id = 'map'
                req.start.pose.position.x = float(self.current_pose[0])
                req.start.pose.position.y = float(self.current_pose[1])
                req.start.pose.orientation.w = 1.0
                req.goal = goal_pose

                self.plan_response_cb(plan_client(req))
            except Exception as exc:
                rospy.logerr(f"全局规划服务调用失败: {exc}")
                self.state = 'WAITING'

        threading.Thread(target=call_service_thread, daemon=True).start()

    def plan_response_cb(self, response):
        if response.plan.poses:
            self.global_path = [(p.pose.position.x, p.pose.position.y) for p in response.plan.poses]
            self.state = 'NAVIGATING'
            rospy.loginfo(f"全局路径获取成功，路点数: {len(self.global_path)}")
        else:
            self.state = 'WAITING'
            rospy.logerr("全局规划失败")

    def control_loop(self, event=None):
        if self.state == 'WAITING':
            self.publish_cmd(0.0, 0.0, 0.0)
            return

        if self.state != 'NAVIGATING':
            return

        if len(self.global_path) < 2:
            rospy.loginfo("到达目标点，控制归零")
            self.publish_cmd(0.0, 0.0, 0.0)
            self.state = 'WAITING'
            return

        target = self.global_path[min(5, len(self.global_path) - 1)]
        vx, vy, wz = self.dwa_compute_velocity(target)
        self.publish_cmd(vx, vy, wz)

        if self.use_fake_pose:
            self.fake_pose_update(vx, vy, wz, 0.1)

        dist_to_target = math.hypot(target[0] - self.current_pose[0], target[1] - self.current_pose[1])
        if dist_to_target < 0.2:
            self.global_path = self.global_path[1:]

    def dwa_compute_velocity(self, target):
        best_score = -float('inf')
        best_cmd = (0.0, 0.0, 0.0)

        v_samples = np.linspace(0.0, self.max_v, 5)
        vy_samples = np.linspace(-self.max_vy, self.max_vy, 3)
        w_samples = np.linspace(-self.max_w, self.max_w, 5)

        for vx in v_samples:
            for vy in vy_samples:
                for wz in w_samples:
                    trajectory = self.simulate_trajectory(vx, vy, wz, time=2.0)
                    if self.check_collision(trajectory):
                        continue

                    end_x, end_y = trajectory[-1]
                    heading_score = -math.hypot(target[0] - end_x, target[1] - end_y)
                    speed_score = math.hypot(vx, vy)
                    score = heading_score + 0.5 * speed_score
                    if score > best_score:
                        best_score = score
                        best_cmd = (float(vx), float(vy), float(wz))

        return best_cmd

    def simulate_trajectory(self, vx, vy, wz, time, dt=0.1):
        traj = []
        x, y, theta = self.current_pose
        for _ in np.arange(0.0, time, dt):
            x += (vx * math.cos(theta) - vy * math.sin(theta)) * dt
            y += (vx * math.sin(theta) + vy * math.cos(theta)) * dt
            theta += wz * dt
            traj.append((x, y))
        return traj

    def check_collision(self, trajectory):
        if not self.obstacles:
            return False
        for px, py in trajectory:
            for ox, oy in self.obstacles:
                if math.hypot(px - ox, py - oy) < self.robot_radius:
                    return True
        return False

    def fake_pose_update(self, vx, vy, wz, dt):
        x, y, theta = self.current_pose
        x += (vx * math.cos(theta) - vy * math.sin(theta)) * dt
        y += (vx * math.sin(theta) + vy * math.cos(theta)) * dt
        theta += wz * dt
        self.current_pose = (x, y, theta)

    def publish_cmd(self, vx, vy, wz_rad):
        if vx == 0.0 and vy == 0.0 and wz_rad == 0.0:
            gear = 1
        elif abs(vy) > 0.01:
            gear = 8
        else:
            gear = 6

        cmd_dict = {
            'gear': gear,
            'vx': round(vx, 3),
            'vy': round(vy, 3),
            'wz': round(math.degrees(wz_rad), 2),
        }
        msg = String()
        msg.data = json.dumps(cmd_dict)
        self.cmd_pub.publish(msg)


if __name__ == '__main__':
    try:
        rospy.init_node('local_navigator_node')
        LocalNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
