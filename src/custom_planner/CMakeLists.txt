cmake_minimum_required(VERSION 3.0.2)
project(custom_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package()

# 告诉 ROS 1 把这两个 Python 文件作为可执行节点
catkin_install_python(PROGRAMS
  scripts/astar_node.py
  scripts/local_navigator_node.py
  scripts/pointcloud_to_grid_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
