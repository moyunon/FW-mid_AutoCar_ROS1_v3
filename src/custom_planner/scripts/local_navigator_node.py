#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry  # 引入了标准里程计消息
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import math
import numpy as np
from std_msgs.msg import String
import json
import threading

class LocalNavigator:
    def __init__(self):
        # ROS 1 参数读取
        self.max_v = rospy.get_param('~max_speed_x', 1.0)
        self.max_vy = rospy.get_param('~max_speed_y', 0.5)
        self.max_w = rospy.get_param('~max_yaw_rate', 1.0)
        self.r_radius = rospy.get_param('~robot_radius', 0.4)
        self.ground_z = rospy.get_param('~ground_filter_z', 0.15)
        
        # ROS 1 接口发布者与订阅者
        self.cmd_pub = rospy.Publisher('/fw_mid/command_dict', String, queue_size=10)
        self.goal_sub = rospy.Subscriber('/goal_pose', PoseStamped, self.goal_cb)
        self.pc_sub = rospy.Subscriber('/lidar_points', PointCloud2, self.pointcloud_cb)
        
        # 【核心新增】：订阅真实的定位里程计话题（由 fastlio2 或定位节点发布）
        # 如果你以后运行的定位话题叫其他名字，请在这里修改话题名
        self.odom_sub = rospy.Subscriber('/Odometry', Odometry, self.odom_cb)
        
        # 服务名称定义
        self.plan_srv_name = '/astar_planner_node/get_plan'
        
        # 状态机变量
        self.state = "WAITING" # WAITING, PLANNING, NAVIGATING
        self.global_path = []
        self.current_pose = (0.0, 0.0, 0.0) # (x, y, yaw)
        self.obstacles = [] 
        
        # 10Hz 控制循环
        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.loginfo("全向局部导航器已启动，等待终点目标...")

    def odom_cb(self, msg):
        """ 【核心新增】：接收真实的定位数据，动态更新小车的实时位姿 """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # 提取四元数
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        # 四元数转航向角 Yaw (纯数学解算，避免外部依赖库报错)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 实时更新位置，打破原点死锁
        self.current_pose = (x, y, yaw)

    def pointcloud_cb(self, msg):
        """ 过滤点云，提取局部障碍物 """
        obs_list = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            if p[2] > self.ground_z: 
                obs_list.append((p[0], p[1]))
        self.obstacles = obs_list

    def goal_cb(self, msg):
        rospy.loginfo("接收到新目标，准备请求全局路径...")
        self.state = "PLANNING"
        self.request_global_plan(msg)

    def request_global_plan(self, goal_pose):
        """ 独立线程调用规划服务，防止阻塞控制循环 """
        def call_service_thread():
            try:
                rospy.loginfo("等待全局规划服务...")
                rospy.wait_for_service(self.plan_srv_name, timeout=5.0)
                plan_client = rospy.ServiceProxy(self.plan_srv_name, GetPlan)
                
                req = GetPlanRequest()
                req.start.header.frame_id = 'map'
                req.start.pose.position.x = float(self.current_pose[0])
                req.start.pose.position.y = float(self.current_pose[1])
                req.start.pose.orientation.w = 1.0
                req.goal = goal_pose
                
                response = plan_client(req)
                self.plan_response_cb(response)
                
            except Exception as e:
                rospy.logerr(f"服务调用异常/超时: {e}")
                self.state = "WAITING"
                
        threading.Thread(target=call_service_thread, daemon=True).start()

    def plan_response_cb(self, response):
        if len(response.plan.poses) > 0:
            self.global_path = [(p.pose.position.x, p.pose.position.y) for p in response.plan.poses]
            self.state = "NAVIGATING"
            rospy.loginfo(f"全局路径获取成功，开始局部导航！路点数:{len(self.global_path)}")
        else:
            self.state = "WAITING"
            rospy.logerr("全局规划失败！")

    def control_loop(self, event=None):
        """ 主控制循环，10Hz 运行 """
        if self.state == "WAITING":
            self.publish_cmd(0.0, 0.0, 0.0)
            return
            
        if self.state == "NAVIGATING":
            if len(self.global_path) < 2:
                rospy.loginfo("🏁 已经顺利到达目的地！控制归零。")
                self.publish_cmd(0.0, 0.0, 0.0)
                self.state = "WAITING"
                return
                
            # 动态选取前方的路点作为局部追踪目标
            target = self.global_path[min(5, len(self.global_path)-1)] 
            vx, vy, w = self.dwa_compute_velocity(target)
            
            if vx == 0.0 and vy == 0.0 and w == 0.0:
                rospy.logwarn("前方受阻或已极度接近路点！尝试平移微调...")
                self.publish_cmd(0.0, 0.1, 0.0) 
                # 【仿真测试用】：如果你现在没有开雷达定位节点，请取消注释下面这行：
                self.fake_pose_update(0.0, 0.1, 0.0, 0.1)
            else:
                self.publish_cmd(vx, vy, w)
                # 👇【重要测试提示】：如果你在进行悬空/架空联调，没有开真实的雷达建图定位，
                # 请取消注释下面这行代码！它会启动“数学积分仿真”，你会看到 Vx, Vy 随着运动动态改变，最后减速停下！
                self.fake_pose_update(vx, vy, w, 0.1)
                
            # 根据实时位姿与当前局部目标的物理距离，滚动剔除已经走过的路点
            dist_to_target = math.hypot(target[0] - self.current_pose[0], target[1] - self.current_pose[1])
            if dist_to_target < 0.2:
                self.global_path = self.global_path[1:]

    def fake_pose_update(self, vx, vy, w, dt):
        """ 全向底盘运动学积分：在无传感器参与时，根据下发速度脑补当前坐标 """
        x, y, theta = self.current_pose
        x += (vx * math.cos(theta) - vy * math.sin(theta)) * dt
        y += (vx * math.sin(theta) + vy * math.cos(theta)) * dt
        theta += w * dt
        self.current_pose = (x, y, theta)

    def dwa_compute_velocity(self, target):
        best_score = -float('inf')
        best_cmd = (0.0, 0.0, 0.0)
        
        v_samples = np.linspace(0, self.max_v, 5)
        vy_samples = np.linspace(-self.max_vy, self.max_vy, 3) 
        w_samples = np.linspace(-self.max_w, self.max_w, 5)
        
        for vx in v_samples:
            for vy in vy_samples:
                for w in w_samples:
                    trajectory = self.simulate_trajectory(vx, vy, w, time=2.0)
                    if self.check_collision(trajectory):
                        continue
                        
                    end_pose = trajectory[-1]
                    heading_score = -math.hypot(target[0] - end_pose[0], target[1] - end_pose[1])
                    speed_score = math.hypot(vx, vy)
                    
                    total_score = heading_score * 1.0 + speed_score * 0.5
                    if total_score > best_score:
                        best_score = total_score
                        best_cmd = (vx, vy, w)
                        
        return best_cmd

    def simulate_trajectory(self, vx, vy, w, time, dt=0.1):
        traj = []
        x, y, theta = self.current_pose
        for _ in np.arange(0, time, dt):
            x += (vx * math.cos(theta) - vy * math.sin(theta)) * dt
            y += (vx * math.sin(theta) + vy * math.cos(theta)) * dt
            theta += w * dt
            traj.append((x, y))
        return traj

    def check_collision(self, trajectory):
        if not self.obstacles: return False
        for p in trajectory:
            for obs in self.obstacles:
                if math.hypot(p[0] - obs[0], p[1] - obs[1]) < self.r_radius:
                    return True
        return False

    def publish_cmd(self, vx, vy, w_rad):
        if vx == 0.0 and vy == 0.0 and w_rad == 0.0:
            gear = 1  
        elif abs(vy) > 0.01:
            gear = 8
        else:
            gear = 6
            
        w_deg = math.degrees(w_rad) 

        cmd_dict = {
            "gear": gear,
            "vx": round(vx, 3),
            "vy": round(vy, 3),
            "wz": round(w_deg, 2)
        }
        
        msg = String()
        msg.data = json.dumps(cmd_dict)
        self.cmd_pub.publish(msg)

if __name__ == '__main__':
    try:
        rospy.init_node('local_navigator_node')
        node = LocalNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
