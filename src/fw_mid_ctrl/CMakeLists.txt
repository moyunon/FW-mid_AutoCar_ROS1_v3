cmake_minimum_required(VERSION 3.0.2)
project(fw_mid_ctrl)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES fw_mid_ctrl
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)

# 告诉 ROS 1 把这个 Python 文件作为可执行节点安装
catkin_install_python(PROGRAMS
  scripts/can_driver_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
