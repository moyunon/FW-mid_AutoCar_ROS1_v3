import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('custom_planner')
    map_yaml_path = os.path.join(pkg_dir, 'maps', 'underground', 'map_2d.yaml')

    # 1. 全局静态地图规划服务节点 (Stage 1 & 2)
    planner_node = Node(
        package='custom_planner',
        executable='astar_node',
        name='astar_planner_node',
        output='screen',
        parameters=[{
            'map_yaml_path': map_yaml_path,
            'robot_radius_m': 0.4,
            'safety_margin_m': 0.15
        }]
    )

    # 2. 动态局部导航节点 (Stage 3, 4, 5)
    navigator_node = Node(
        package='custom_planner',
        executable='local_navigator_node', # 需要在 setup.py 中注册
        name='local_navigator_node',
        output='screen',
        parameters=[{
            'max_speed_x': 1.0,
            'max_speed_y': 0.5,
            'max_yaw_rate': 0.8,
            'ground_filter_z': 0.15 # 高于 15cm 的点才视为障碍
        }]
    )

    # 3. 雷达倾斜 TF 广播
    # 假设雷达安装在车头(x=0.34m, FW-mid长680mm的一半), 高度(z=0.3m), 向前倾斜约10度(pitch=0.1745 rad)
    # 参数顺序: x y z yaw pitch roll frame_id child_frame_id
    tf_lidar_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.34', '0.0', '0.3', '0.0', '0.1745', '0.0', 'base_link', 'lidar_link'],
        output='screen'
    )

    # 4. RViz 可视化
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        planner_node,
        navigator_node,
        tf_lidar_node,
        rviz_node
    ])