import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 声明可以通过命令行传入的参数，默认值为 'can0'
    can_interface_arg = DeclareLaunchArgument(
        'can_interface',
        default_value='can0',
        description='The socketcan interface to use (e.g., can0, can1)'
    )

    # 定义底层的 CAN 驱动节点
    can_driver_node = Node(
        package='fw_mid_ctrl',
        executable='can_driver_node',
        name='fw_mid_can_driver',
        output='screen', # 将节点的日志输出到屏幕，方便查看反馈信息
        parameters=[{
            'can_interface': LaunchConfiguration('can_interface')
        }]
    )

    return LaunchDescription([
        can_interface_arg,
        can_driver_node
    ])