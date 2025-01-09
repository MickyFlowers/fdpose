#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from naviai_manip_srvs.srv import PoseEst, PoseEstRequest
from naviai_manip_msgs.msg import DetItem, ObjPose
import numpy as np
def load_color_image_as_ros_msg(image_path):
    # 使用OpenCV加载彩色图像
    image = cv2.imread(image_path)
    if image is None:
        rospy.logerr(f"Failed to load color image at path: {image_path}")
        return None
    # 转换为ROS图像消息
    bridge = CvBridge()
    return bridge.cv2_to_imgmsg(image, encoding="bgr8")

def load_depth_image_as_ros_msg(depth_image_path):
    # Load the depth image as a grayscale (assumed as 16-bit depth image)
    depth_image_gray = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the image was loaded correctly
    if depth_image_gray is None:
        rospy.logerr(f"Failed to load depth image at path: {depth_image_path}")
        return None

    # Ensure the image is in 16-bit format for mono16 encoding
    depth_image_16u = depth_image_gray.astype(np.uint16)
    
    # Convert to ROS Image message
    bridge = CvBridge()
    return bridge.cv2_to_imgmsg(depth_image_16u, encoding="mono16")

def load_mask_image_as_ros_msg(mask_image_path):
    # 使用OpenCV加载mask图像（假设为灰度图）
    mask_image_gray = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if mask_image_gray is None:
        rospy.logerr(f"Failed to load mask image at path: {mask_image_path}")
        return None
    # 转换为16位灰度图像
    mask_image_16bit = (mask_image_gray.astype(np.uint16)) * 256  # 将范围从 0-255 扩展到 0-65535
    
    # 转换为ROS图像消息
    bridge = CvBridge()
    return bridge.cv2_to_imgmsg(mask_image_16bit, encoding="mono16")


def object_pose_estimation_client(color_image_path, depth_image_path):
    rospy.wait_for_service('pose_estimation_service')
    try:
        # 创建服务代理
        object_pose_estimation = rospy.ServiceProxy('pose_estimation_service', PoseEst)
        
        # 创建请求
        req = PoseEstRequest()
        
        # 读取并转换彩色图像和深度图像
        req.color_image = load_color_image_as_ros_msg(color_image_path)
        req.depth_image = load_depth_image_as_ros_msg(depth_image_path)  
        if req.color_image is None or req.depth_image is None:
            rospy.logerr("Failed to load images. Exiting client.")
            return
        mask_image_path = "/home/catkin_ws/src/fdpose/scripts/demo_data/spatula/masks/016.png"
        mask_image = load_mask_image_as_ros_msg(mask_image_path)
        # 示例分割信息
        # 添加分割信息的items列表
        item = DetItem()
        item.label = "spatula"
        item.confidence = 0.95
        item.mask = mask_image
        req.items = [item]  # 将items设为包含一个DetItem对象的列表
        

        # 发送请求并获取响应
        # rospy.loginfo(req.items)
        resp = object_pose_estimation(req)
        rospy.loginfo("Received object poses:")


    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

if __name__ == "__main__":
    rospy.init_node('object_pose_estimation_client')
    
    # 设置图像路径
    color_image_path = "/home/catkin_ws/src/fdpose/scripts/demo_data/spatula/rgb/016.png"
    depth_image_path = "/home/catkin_ws/src/fdpose/scripts/demo_data/spatula/depth/016.png"
    
    object_pose_estimation_client(color_image_path, depth_image_path)