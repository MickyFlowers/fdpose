import rospy
import tf
import numpy as np
from tf.transformations import euler_from_matrix, quaternion_matrix, quaternion_from_matrix
from geometry_msgs.msg import TransformStamped

# 初始化 ROS 节点
rospy.init_node('pose_transformer')

# 创建 tf 广播器
br = tf.TransformBroadcaster()
# 加载和发布物体姿态的函数
def load_and_publish_pose(pose_file_path, child_frame, parent_frame="SACRUM"):
    pose = np.load(pose_file_path)
    if pose.shape != (4, 4):
        rospy.logerr(f"Loaded pose from {pose_file_path} is not a 4x4 matrix.")
        return
    pose_in_robot = np.dot(T_robot_camera, pose)
    translation_robot = pose_in_robot[:3, 3]
    rotation_robot = quaternion_from_matrix(pose_in_robot)
    br.sendTransform(
        translation_robot,
        rotation_robot,
        rospy.Time.now(),
        child_frame,
        parent_frame
    )

# 加载和发布 hand_to_object 的函数
def load_and_publish_hand_to_object_pose(hand_to_object_path, child_frame, parent_frame):
    hand_to_object = np.load(hand_to_object_path)
    if hand_to_object.shape != (4, 4):
        rospy.logerr(f"Loaded hand_to_object pose from {hand_to_object_path} is not a 4x4 matrix.")
        return
    translation_hand_to_object = hand_to_object[:3, 3]
    rotation_hand_to_object = quaternion_from_matrix(hand_to_object)
    br.sendTransform(
        translation_hand_to_object,
        rotation_hand_to_object,
        rospy.Time.now(),
        child_frame,
        parent_frame
    )
T_robot_camera_path = '/home/catkin_ws/src/naviai_manipulation/naviai_manip_main/camera_info/cam2torso.npy'  # 替换为实际的文件路径
T_robot_camera = np.load(T_robot_camera_path)
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

    # 发布 pan 的姿态
    load_and_publish_pose('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/pan.npy', "pan_in_robot_frame")

    # 发布 bowl 的姿态
    load_and_publish_pose('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/bowl.npy', "bowl_in_robot_frame")

    # 发布 spatula 的姿态
    load_and_publish_pose('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/spatula.npy', "spatula_in_robot_frame")

    # 发布 right_tcp_in_pan 的姿态
    load_and_publish_hand_to_object_pose('/home/catkin_ws/src/fdpose/scripts/debug/right_tcp_in_pan.npy', "right_tcp_in_pan", "pan_in_robot_frame")

    # 发布 left_tcp_in_pan 的姿态
    load_and_publish_hand_to_object_pose('/home/catkin_ws/src/fdpose/scripts/debug/left_tcp_in_pan.npy', "left_tcp_in_pan", "pan_in_robot_frame")

    # 发布 right_tcp_in_bowl 的姿态
    load_and_publish_hand_to_object_pose('/home/catkin_ws/src/fdpose/scripts/debug/right_tcp_in_bowl.npy', "right_tcp_in_bowl", "bowl_in_robot_frame")

    # 发布 left_tcp_in_bowl 的姿态
    load_and_publish_hand_to_object_pose('/home/catkin_ws/src/fdpose/scripts/debug/left_tcp_in_bowl.npy', "left_tcp_in_bowl", "bowl_in_robot_frame")

    # 发布 right_tcp_in_spatula 的姿态
    load_and_publish_hand_to_object_pose('/home/catkin_ws/src/fdpose/scripts/debug/right_tcp_in_spatula.npy', "right_tcp_in_spatula", "spatula_in_robot_frame")

    # 发布 left_tcp_in_spatula 的姿态
    load_and_publish_hand_to_object_pose('/home/catkin_ws/src/fdpose/scripts/debug/left_tcp_in_spatula.npy', "left_tcp_in_spatula", "spatula_in_robot_frame")


    # 加载变换矩阵
    pose = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/pan.npy')
    hand_to_object = np.load('/home/catkin_ws/src/fdpose/scripts/debug/right_tcp_in_pan.npy')
    T_cook = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/induction_cooker.npy')
    T_pan_in_cook = np.load('/home/catkin_ws/src/fdpose/scripts/debug/right_pan_in_cook.npy')

    # 验证变换矩阵是否为 4x4
    for T in [T_robot_camera, T_base_robot, pose, hand_to_object, T_cook, T_pan_in_cook]:
        if T.shape != (4, 4):
            rospy.logerr("One of the transformation matrices is not a 4x4 matrix.")
            return

    # 将灶台和锅的位姿从相机坐标系转换到基座坐标系（SACRUM）
    T_camera_base = np.dot(T_base_robot, T_robot_camera)  # 相机到基座的变换
    T_cook_base = np.dot(T_camera_base, T_cook)           # 灶台在基座坐标系下的位姿
    T_pan_base = np.dot(T_camera_base, pose)              # 锅在基座坐标系下的位姿

    # 计算锅在灶台上的最终位姿
    T_pan_on_cooktop_base = np.dot(T_cook_base, T_pan_in_cook)

    # 提取锅在灶台上的平移和旋转（四元数）信息
    translation_pan_on_cooktop = T_pan_on_cooktop_base[:3, 3]
    rotation_pan_on_cooktop = quaternion_from_matrix(T_pan_on_cooktop_base)

    # 发布锅在灶台上的位姿到 TF 树
    br.sendTransform(
        translation_pan_on_cooktop,
        rotation_pan_on_cooktop,
        rospy.Time.now(),
        "place_pan_on_cooktop",   # 锅在灶台上的 frame
        "SACRUM"                  # 基座坐标系 frame
    )

    # 计算手的目标位姿以便将锅放置在灶台上
    hand_target_pose = np.dot(T_pan_on_cooktop_base, hand_to_object)

    # 提取手的平移和旋转信息
    translation_hand = hand_target_pose[:3, 3]
    rotation_hand = quaternion_from_matrix(hand_target_pose)

    # 发布手的目标位姿
    br.sendTransform(
        translation_hand,
        rotation_hand,
        rospy.Time.now(),
        "reach_hand_to_place_pan_on_cooktop",  # 手的目标位置 frame 
        "SACRUM"                               # 基座坐标系 frame
    )
    
    # 加载 bowl 和 spatula 相关的姿态矩阵
    bowl_pose = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/bowl.npy')
    spatula_pose = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/spatula.npy')

    hand_to_bowl = np.load('/home/catkin_ws/src/fdpose/scripts/debug/right_tcp_in_bowl.npy')
    hand_to_spatula = np.load('/home/catkin_ws/src/fdpose/scripts/debug/right_tcp_in_spatula.npy')

    T_bowl_in_pan = np.load('/home/catkin_ws/src/fdpose/scripts/debug/right_bowl_in_pan.npy')
    T_spatula_in_pan = np.load('/home/catkin_ws/src/fdpose/scripts/debug/right_spatula_in_pan.npy')

    # 验证所有变换矩阵是否为 4x4
    for T in [bowl_pose, spatula_pose, hand_to_bowl, hand_to_spatula, T_bowl_in_pan, T_spatula_in_pan]:
        if T.shape != (4, 4):
            rospy.logerr("One of the transformation matrices is not a 4x4 matrix.")
            return

    # 计算 bowl 在 pan 上的位姿
    T_bowl_on_pan_base = np.dot(T_pan_base, T_bowl_in_pan)

    # 提取 bowl 在 pan 上的平移和旋转信息
    translation_bowl_on_pan = T_bowl_on_pan_base[:3, 3]
    rotation_bowl_on_pan = quaternion_from_matrix(T_bowl_on_pan_base)

    # 发布 bowl 在 pan 上的位姿
    br.sendTransform(
        translation_bowl_on_pan,
        rotation_bowl_on_pan,
        rospy.Time.now(),
        "place_bowl_on_pan",   # bowl 在 pan 上的 frame
        "SACRUM"               # 基座坐标系 frame
    )

    # 计算 reach_hand_to_place_bowl_on_pan 的手的目标位姿
    hand_target_pose_bowl = np.dot(T_bowl_on_pan_base, hand_to_bowl)
    translation_hand_bowl = hand_target_pose_bowl[:3, 3]
    rotation_hand_bowl = quaternion_from_matrix(hand_target_pose_bowl)

    # 发布手的目标位姿用于放置 bowl 在 pan 上
    br.sendTransform(
        translation_hand_bowl,
        rotation_hand_bowl,
        rospy.Time.now(),
        "reach_hand_to_place_bowl_on_pan",  # 手的目标位置 frame
        "SACRUM"                            # 基座坐标系 frame
    )

    # 计算 spatula 在 pan 上的位姿
    T_spatula_on_pan_base = np.dot(T_pan_base, T_spatula_in_pan)

    # 提取 spatula 在 pan 上的平移和旋转信息
    translation_spatula_on_pan = T_spatula_on_pan_base[:3, 3]
    rotation_spatula_on_pan = quaternion_from_matrix(T_spatula_on_pan_base)

    # 发布 spatula 在 pan 上的位姿
    br.sendTransform(
        translation_spatula_on_pan,
        rotation_spatula_on_pan,
        rospy.Time.now(),
        "place_spatula_on_pan",   # spatula 在 pan 上的 frame
        "SACRUM"                  # 基座坐标系 frame
    )

    # 计算 reach_hand_to_place_spatula_on_pan 的手的目标位姿
    hand_target_pose_spatula = np.dot(T_spatula_on_pan_base, hand_to_spatula)
    translation_hand_spatula = hand_target_pose_spatula[:3, 3]
    rotation_hand_spatula = quaternion_from_matrix(hand_target_pose_spatula)

    # 发布手的目标位姿用于放置 spatula 在 pan 上
    br.sendTransform(
        translation_hand_spatula,
        rotation_hand_spatula,
        rospy.Time.now(),
        "reach_hand_to_place_spatula_on_pan",  # 手的目标位置 frame
        "SACRUM"                               # 基座坐标系 frame
    )

# 循环发布 tf 变换
if __name__ == "__main__":
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        main()
        rate.sleep()