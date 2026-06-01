#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Float32MultiArray
import json
import can
import struct
import math
import threading

class FwMidCanDriver:
    def __init__(self):
        # 声明参数：CAN接口名称，默认 can0
        can_iface = rospy.get_param('~can_interface', 'can0')
        
        # 订阅由 custom_planner 发出的 JSON 字典控制指令
        self.cmd_sub = rospy.Subscriber('/fw_mid/command_dict', String, self.cmd_cb)
        
        # 发布解析后的反馈信息
        self.fb_vel_pub = rospy.Publisher('/fw_mid/feedback/velocity', Float32MultiArray, queue_size=10)
        self.fb_bms_pub = rospy.Publisher('/fw_mid/feedback/bms', Float32MultiArray, queue_size=10)
        
        # Alive Rolling Counter (0-15循环)
        self.alive_counter = 0

        # 初始化 CAN 总线
        try:
            self.bus = can.interface.Bus(channel=can_iface, bustype='socketcan')
            rospy.loginfo(f"成功连接到CAN接口: {can_iface} (请确保波特率已设为手册要求的500K)")
            
            # 开启一个独立线程，专门用于持续接收和解析 CAN 总线数据
            self.receive_thread = threading.Thread(target=self.receive_can_messages)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
        except Exception as e:
            rospy.logerr(f"无法打开CAN接口 {can_iface}, 请确认系统是否已绑定硬件！错误: {e}")
            self.bus = None

    # ================= CAN 发送逻辑 =================
    def cmd_cb(self, msg):
        """ 解析 JSON 字典并打包成 CAN 帧发送 """
        if self.bus is None:
            return
            
        try:
            cmd = json.loads(msg.data)
            gear = int(cmd.get('gear', 6))       # 默认 6: 4T4D模式
            vx = float(cmd.get('vx', 0.0))       # m/s
            vy = float(cmd.get('vy', 0.0))       # m/s
            wz = float(cmd.get('wz', 0.0))       # °/s
            
            self.send_can_ctrl_msg(gear, vx, vy, wz)
            
        except json.JSONDecodeError:
            rospy.logerr("收到的指令不是有效的JSON字典格式！")

    def send_can_ctrl_msg(self, gear, vx, vy, wz):
        """ 将物理值转为 FW-mid 硬件报文 (Intel 格式) """
        vx_raw = int(vx / 0.001)  # 0.001m/s/bit
        vy_raw = int(vy / 0.001)  # 0.001m/s/bit
        wz_raw = int(wz / 0.01)   # 0.01°/s/bit
        
        # 限制在 16-bit 有符号整数范围内
        vx_raw = max(-32768, min(32767, vx_raw))
        vy_raw = max(-32768, min(32767, vy_raw))
        wz_raw = max(-32768, min(32767, wz_raw))
        
        # 转换为无符号 16-bit 表示法
        vx_u16 = vx_raw & 0xFFFF
        vy_u16 = vy_raw & 0xFFFF
        wz_u16 = wz_raw & 0xFFFF
        
        payload = 0
        payload |= (gear & 0x0F) << 0         # Start bit: 0, Length: 4
        payload |= (vx_u16 & 0xFFFF) << 4     # Start bit: 4, Length: 16
        payload |= (wz_u16 & 0xFFFF) << 20    # Start bit: 20, Length: 16
        payload |= (vy_u16 & 0xFFFF) << 36    # Start bit: 36, Length: 16
        payload |= (self.alive_counter & 0x0F) << 52 # Start bit: 52, Length: 4
        
        data = bytearray(struct.pack('<Q', payload))
        
        bcc = 0
        for i in range(7):
            bcc ^= data[i]
        data[7] = bcc # Check BCC
        
        msg = can.Message(arbitration_id=0x18C4D1D0, data=data, is_extended_id=True)
        
        try:
            self.bus.send(msg)
        except can.CanError as e:
            rospy.logerr(f"CAN发送失败: {e}")
            
        self.alive_counter = (self.alive_counter + 1) % 16

    # ================= CAN 接收与解析逻辑 =================
    def receive_can_messages(self):
        """ 在独立线程中持续监听 CAN 总线并解析反馈 """
        while not rospy.is_shutdown() and self.bus is not None:
            try:
                # 阻塞式接收，超时 0.1 秒
                msg = self.bus.recv(timeout=0.1)
                if msg is None:
                    continue
                
                # 验证 BCC 校验和
                if len(msg.data) == 8:
                    calc_bcc = 0
                    for i in range(7):
                        calc_bcc ^= msg.data[i]
                    if calc_bcc != msg.data[7]:
                        rospy.logwarn(f"接收到ID为 {hex(msg.arbitration_id)} 的帧，但BCC校验失败！")
                        continue

                # 1. 解析运动控制状态反馈
                if msg.arbitration_id == 0x18C4D1EF:
                    self.parse_ctrl_fb(msg.data)
                    
                # 2. 解析电池状态反馈
                elif msg.arbitration_id == 0x18C4E1EF:
                    self.parse_bms_fb(msg.data)
                    
                # 3. 解析IO状态反馈
                elif msg.arbitration_id == 0x18C4DAEF:
                    self.parse_io_fb(msg.data)
                    
            except Exception as e:
                # 忽略关闭时的异常
                pass

    def parse_ctrl_fb(self, data):
        """ 解析 0x18C4D1EF 运动控制反馈 """
        payload = struct.unpack('<Q', data)[0]
        
        gear_raw = (payload >> 0) & 0x0F
        vx_raw   = (payload >> 4) & 0xFFFF
        wz_raw   = (payload >> 20) & 0xFFFF
        vy_raw   = (payload >> 36) & 0xFFFF
        
        if vx_raw & 0x8000: vx_raw -= 0x10000
        if vy_raw & 0x8000: vy_raw -= 0x10000
        if wz_raw & 0x8000: wz_raw -= 0x10000
            
        vx = vx_raw * 0.001
        vy = vy_raw * 0.001
        wz = wz_raw * 0.01 
        
        gear_dict = {0:"Disable", 1:"驻车", 2:"空档", 6:"4T4D", 8:"横移"}
        gear_str = gear_dict.get(gear_raw, "未知")
        rospy.loginfo(f"[运动反馈] 档位: {gear_str}, Vx: {vx:.3f} m/s, Vy: {vy:.3f} m/s, Wz: {wz:.2f} °/s")
        
        msg = Float32MultiArray()
        msg.data = [float(gear_raw), vx, vy, wz]
        self.fb_vel_pub.publish(msg)

    def parse_bms_fb(self, data):
        """ 解析 0x18C4E1EF 电池状态反馈 """
        payload = struct.unpack('<Q', data)[0]
        
        vol_raw = (payload >> 0) & 0xFFFF
        cur_raw = (payload >> 16) & 0xFFFF
        cap_raw = (payload >> 32) & 0xFFFF
        
        if cur_raw & 0x8000: cur_raw -= 0x10000
            
        voltage = vol_raw * 0.01
        current = cur_raw * 0.01
        capacity = cap_raw * 0.01
        
        rospy.loginfo(f"[电池反馈] 电压: {voltage:.2f} V, 电流: {current:.2f} A, 剩余容量: {capacity:.2f} Ah")
        
        msg = Float32MultiArray()
        msg.data = [voltage, current, capacity]
        self.fb_bms_pub.publish(msg)

    def parse_io_fb(self, data):
        """ 解析 0x18C4DAEF I/O 控制状态反馈 """
        payload = struct.unpack('<Q', data)[0]
        
        estop_raw = (payload >> 40) & 0x01
        rc_status_raw = (payload >> 41) & 0x01
        
        estop_str = "被按下 (停车)" if estop_raw == 1 else "已释放 (正常)"
        rc_str = "遥控器控制" if rc_status_raw == 1 else "指令(CAN)控制"
        
        rospy.loginfo(f"[IO反馈] 急停状态: {estop_str} | 控制权: {rc_str}")

if __name__ == '__main__':
    try:
        rospy.init_node('fw_mid_can_driver')
        node = FwMidCanDriver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
