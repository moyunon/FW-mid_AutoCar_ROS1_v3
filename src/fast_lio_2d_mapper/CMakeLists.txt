cmake_minimum_required(VERSION 3.0.2)
project(fast_lio_2d_mapper)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  rospy
  sensor_msgs
  std_srvs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/cloud_to_2d_map_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
