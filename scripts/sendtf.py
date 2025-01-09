import rospy
import tf
import numpy as np
from tf.transformations import euler_from_matrix, quaternion_matrix, quaternion_from_matrix
from geometry_msgs.msg import TransformStamped

# 初始化 ROS 节点
rospy.init_node('pose_transformer')

# 创建 tf 广播器
br = tf.TransformBroadcaster()

def main():
    # 从 .npy 文件加载相机到机器人基座的 T_robot_camera 变换数据
    # 从文件中加载相机到机器人基座的 T_robot_camera 变换
    T_robot_camera_path = '/home/catkin_ws/src/naviai_manipulation/naviai_manip_main/camera_info/cam2torso.npy'  # 替换为实际的文件路径
    T_robot_camera = np.load(T_robot_camera_path)

    # 检查 T_robot_camera 是否为 4x4 矩阵
    if T_robot_camera.shape != (4, 4):
        rospy.logerr("Loaded T_robot_camera is not a 4x4 matrix.")
        exit()

    # 从文件中加载基座到机器人坐标系的 T_base_robot 变换
    T_base_robot_path = '/home/catkin_ws/src/naviai_manipulation/naviai_manip_main/camera_info/torso2sacrum.npy'  # 替换为实际的文件路径
    T_base_robot = np.load(T_base_robot_path)

    # 检查 T_base_robot 是否为 4x4 矩阵
    if T_base_robot.shape != (4, 4):
        rospy.logerr("Loaded T_base_robot is not a 4x4 matrix.")
        exit()

    # 计算相机到机器人坐标系的最终变换 T_robot
    T_camera_robot = np.dot(T_base_robot, T_robot_camera)

    # 提取平移和旋转（四元数）信息
    translation = T_camera_robot[:3, 3]
    rotation_quaternion = quaternion_from_matrix(T_camera_robot)

    # 发布相机到机器人坐标系的 tf 变换
    br.sendTransform(
        translation,
        rotation_quaternion,
        rospy.Time.now(),
        "camera_link",       # 相机 frame
        "SACRUM"         # 机器人坐标系 frame
    )

    # 从 .npy 文件加载物品姿态矩阵 pose
    pose_file_path = '/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/Chips_can.npy'  # 替换为实际的文件路径
    pose = np.load(pose_file_path)
    
    # 检查 pose 是否为 4x4 矩阵
    if pose.shape != (4, 4):
        rospy.logerr("Loaded pose is not a 4x4 matrix.")
        return

    # 使用 T_robot_camera 转换物品姿态到机器人坐标系
    pose_in_robot = np.dot(T_robot_camera, pose)

    # 提取物品在机器人坐标系下的位置（平移）和方向（欧拉角）
    translation_robot = pose_in_robot[:3, 3]
    roll, pitch, yaw = euler_from_matrix(pose_in_robot, 'sxyz')

    # 将角度转换为度数
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    # 输出物品在机器人坐标系下的位置和姿态
    print(f"Position in robot frame: {translation_robot}")
    print(f"Orientation in robot frame: Roll={roll_deg:.2f}, Pitch={pitch_deg:.2f}, Yaw={yaw_deg:.2f}")

    # 发布物品姿态到 tf 树，以便在 rviz 中查看
    br.sendTransform(
        translation_robot,
        quaternion_from_matrix(pose_in_robot),
        rospy.Time.now(),
        "object_in_robot_frame",      # child frame
        "SACRUM"             # parent frame
    )


    # 从 .npy 文件加载手相对于物体的姿态
    hand_to_object_path = '/home/catkin_ws/src/fdpose/scripts/debug/right_tcp_in_Chips_can.npy'  # 替换为手相对于物体的姿态的实际文件路径
    hand_to_object = np.load(hand_to_object_path)
    
    # 检查手相对于物体的姿态矩阵是否为 4x4
    if hand_to_object.shape != (4, 4):
        rospy.logerr("Loaded hand_to_object is not a 4x4 matrix.")
        return

    # 提取手相对于物体的平移和旋转
    translation_hand_to_object = hand_to_object[:3, 3]
    rotation_hand_to_object = quaternion_from_matrix(hand_to_object)

    # 发布手相对于物体的姿态
    br.sendTransform(
        translation_hand_to_object,
        rotation_hand_to_object,
        rospy.Time.now(),
        "hand_to_object_frame",        # child frame (手的 frame)
        "object_in_robot_frame"       # parent frame (物体的 frame)
    )


    # # 读取当前cook, pan, hand的位姿矩阵
    # # T_cook = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/induction_cooker.npy')
    # # T_pan = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/pan.npy')

    # # 加载变换矩阵
    # T_robot_camera = np.load('/home/catkin_ws/src/naviai_manipulation/naviai_manip_main/camera_info/cam2torso.npy')
    # T_base_robot = np.load('/home/catkin_ws/src/naviai_manipulation/naviai_manip_main/camera_info/torso2sacrum.npy')
    # pose = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/pan.npy')
    # hand_to_object = np.load('/home/catkin_ws/src/fdpose/scripts/debug/left_tcp_in_pan.npy')
    # T_cook = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/induction_cooker.npy')
    # T_pan_in_cook = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/pan_in_cook.npy')
    
    # # 验证变换矩阵是否为 4x4
    # for T in [T_robot_camera, T_base_robot, pose, hand_to_object, T_cook, T_pan_in_cook]:
    #     if T.shape != (4, 4):
    #         rospy.logerr("One of the transformation matrices is not a 4x4 matrix.")
    #         return

    # # 将灶台和锅的位姿从相机坐标系转换到基座坐标系（SACRUM）
    # T_camera_base = np.dot(T_base_robot, T_robot_camera)  # 相机到基座的变换
    # T_cook_base = np.dot(T_camera_base, T_cook)           # 灶台在基座坐标系下的位姿
    # T_pan_base = np.dot(T_camera_base, pose)              # 锅在基座坐标系下的位姿

    # # 计算锅在灶台上的最终位姿
    # T_pan_on_cooktop_base = np.dot(T_cook_base, T_pan_in_cook)

    # # 提取锅在灶台上的平移和旋转（四元数）信息
    # translation_pan_on_cooktop = T_pan_on_cooktop_base[:3, 3]
    # rotation_pan_on_cooktop = quaternion_from_matrix(T_pan_on_cooktop_base)

    # # 发布锅在灶台上的位姿到 TF 树
    # br.sendTransform(
    #     translation_pan_on_cooktop,
    #     rotation_pan_on_cooktop,
    #     rospy.Time.now(),
    #     "pan_on_cooktop",   # 锅在灶台上的 frame
    #     "SACRUM"            # 基座坐标系 frame
    # )

    # # 计算手的目标位姿以便将锅放置在灶台上
    # hand_target_pose = np.dot(T_pan_on_cooktop_base, hand_to_object)
    
    # # 提取手的平移和旋转信息
    # translation_hand = hand_target_pose[:3, 3]
    # rotation_hand = quaternion_from_matrix(hand_target_pose)
    
    # # 发布手的目标位姿
    # br.sendTransform(
    #     translation_hand,
    #     rotation_hand,
    #     rospy.Time.now(),
    #     "hand_target_pose",  # 手的目标位置 frame
    #     "SACRUM"             # 基座坐标系 frame
    # )

    # # 额外的调试输出
    # print(f"锅在灶台上的位置：{translation_pan_on_cooktop}")
    # roll, pitch, yaw = euler_from_matrix(T_pan_on_cooktop_base)
    # print(f"锅的姿态：Roll={np.degrees(roll):.2f}, Pitch={np.degrees(pitch):.2f}, Yaw={np.degrees(yaw):.2f}")

# 循环发布 tf 变换
if __name__ == "__main__":
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        main()
        rate.sleep()