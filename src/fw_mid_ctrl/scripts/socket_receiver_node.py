#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import socket

def ros1_socket_receiver():
    # 1. 初始化 ROS 1 接收节点
    rospy.init_node('ros1_socket_receiver', anonymous=True)
    
    # 2. 创建发布者，将接到的 JSON 字符串高频喂给底盘 CAN 驱动
    pub = rospy.Publisher('/fw_mid/command_dict', String, queue_size=10)

    # 3. 绑定端口，准备捕获局域网内甩过来的数据
    bind_ip = "0.0.0.0"  # 监听所有网卡接口
    bind_port = 9999
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_ip, bind_port))
    
    rospy.loginfo(f"ROS 1 车机接收端启动！正在 9999 端口死守电脑的指令...")

    while not rospy.is_shutdown():
        try:
            # 设置超时防止死锁
            sock.settimeout(1.0)
            data, addr = sock.recvfrom(1024)
            
            # 4. 解析收到的原始文本，瞬间发布为 ROS 1 话题
            
            json_str = data.decode('utf-8')
            pub.publish(json_str)
            rospy.loginfo(json_str)
        except socket.timeout:
            continue
        except Exception as e:
            rospy.logerr(f"接收链路异常: {e}")
            break

if __name__ == '__main__':
    try:
        ros1_socket_receiver()
    except rospy.ROSInterruptException:
        pass
