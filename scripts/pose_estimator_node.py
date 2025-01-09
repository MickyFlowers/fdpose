# #!/usr/bin/env python3
# import time
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# from estimater import *
# from datareader import *
# import rospy
# import cv2
# import numpy as np
# import trimesh
# import os
# import logging
# from sensor_msgs.msg import Image
# from naviai_manip_srvs.srv import PoseEst, PoseEstResponse,PoseEstResponse
# from naviai_manip_msgs.msg import DetItem, ObjPose
# from cv_bridge import CvBridge

# class PoseEstimatorNode:
#     def __init__(self):
#         rospy.init_node('pose_estimator_node', anonymous=True)
        
#         # 获取工作目录路径
#         self.code_dir = os.path.dirname(os.path.realpath(__file__))
#         self.mesh_file = f'{self.code_dir}/model/bowl.obj'
#         self.debug_dir = f'{self.code_dir}/debug'
#         self.test_scene_dir = f'{code_dir}/demo_data/spatula'

#         self.est_refine_iter = 5
#         self.track_refine_iter = 2
#         # 加载 mesh 文件
#         self.mesh = trimesh.load(self.mesh_file)
#         self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
#         self.bbox = np.stack([-self.extents/2, self.extents/2], axis=0).reshape(2,3)
#         # 赋值 self.bbox
#         self.bbox = np.array([
#             [-0.15351711, -0.08931366, -0.03679115],
#             [ 0.15351711,  0.08931366,  0.03679115]
#         ])
#         print("self.bbox ",self.bbox )
#         print("self.to_origin ",self.to_origin )
#         # 初始化评分和姿态精炼模型
#         self.scorer = ScorePredictor()
#         self.refiner = PoseRefinePredictor()
#         self.glctx = dr.RasterizeCudaContext()
#         self.est = FoundationPose(
#             model_pts=self.mesh.vertices,
#             model_normals=self.mesh.vertex_normals,
#             mesh=self.mesh,
#             scorer=self.scorer,
#             refiner=self.refiner,
#             debug_dir=self.debug_dir,
#             debug=3,
#             glctx=self.glctx
#         )
        
#         # 初始化 cv_bridge
#         self.bridge = CvBridge()

#         # 创建调试目录
#         os.makedirs(f'{self.debug_dir}/track_vis', exist_ok=True)
#         os.makedirs(f'{self.debug_dir}/ob_in_cam', exist_ok=True)

#         # 定义服务
#         self.service = rospy.Service('pose_estimation_service', PoseEst, self.handle_pose_estimation)
#         rospy.loginfo("Pose Estimation Service is ready")

#         self.rgb_image = None
#         self.depth_image = None
#         self.K = np.array([[615, 0, 320], [0, 615, 240], [0, 0, 1]])  # 假设的内参矩阵
#         rospy.loginfo("Pose Estimator Node initialized")

#     def handle_pose_estimation(self, req):
#         # 将ROS Image消息转换为OpenCV图像
#         depth_path = f'{self.code_dir}/demo_data/spatula/depth'
#         rgb_path = f'{self.code_dir}/demo_data/spatula/rgb'
        
#         rgb_image = self.bridge.imgmsg_to_cv2(req.color_image, "bgr8")
#         depth_image = self.bridge.imgmsg_to_cv2(req.depth_image, "mono16")
#         # 确保目录存在，如果不存在则创建
#         os.makedirs(depth_path, exist_ok=True)
#         os.makedirs(rgb_path, exist_ok=True)

#         # 假设rgb_image和depth_image已经通过前面的代码正确获取
#         # 生成文件名，可以根据你的需求修改命名规则，这里简单使用序号0.jpg作为示例
#         rgb_filename = os.path.join(rgb_path, "0.png")
#         depth_filename = os.path.join(depth_path, "0.png")

#         # 保存RGB图像
#         cv2.imwrite(rgb_filename, rgb_image)
#         # 保存深度图像，注意深度图像的数据类型可能需要特殊处理，这里假设当前形式可以直接保存
#         cv2.imwrite(depth_filename, depth_image)
#         debug = 3
#         debug_dir = self.debug_dir


        
#         obj_poses = []
#         for i, item in enumerate(req.items):
            
#             mask_image = self.bridge.imgmsg_to_cv2(item.mask, "mono16")  # 将mask转换为单通道图像
#             mask_path = f'{self.code_dir}/demo_data/spatula/masks'
#             mask_filename = os.path.join(mask_path, f"0.png")
            
#             if mask_image.max() > 0:  # Avoid division by zero
#                 mask_image_normalized = (mask_image / mask_image.max()) * 255
#                 mask_image_normalized = mask_image_normalized.astype(np.uint8)
#             else:
#                 mask_image_normalized = mask_image.astype(np.uint8)

#             # Save the normalized mask image
#             cv2.imwrite(mask_filename, mask_image_normalized)

#             reader = YcbineoatReader(video_dir=self.test_scene_dir , shorter_side=None, zfar=np.inf)
#             # 开始姿态估计
#             color = reader.get_color(i)
#             depth = reader.get_depth(i)
#             start_register_time = time.time()
#             mask = reader.get_mask(i).astype(bool)  # 获取物体mask
#             pose = self.est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)
#             register_time = time.time() - start_register_time
#             logging.info(f"第 {i} 帧初始注册时间: {register_time:.2f} 秒") 
#             print("Pose value:", pose)
#             print("Pose type:", type(pose))


#             pose_result = ObjPose()
#             pose_result.label = item.label
#             pose_result.pose = pose.flatten().astype(np.float32)
#             obj_poses.append(pose_result) 

#             # 将物体姿态保存到文本文件
#             start_save_pose_time = time.time()
#             os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
#             np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
#             np.save(f'{debug_dir}/ob_in_cam/{1}.npy', pose.reshape(4,4)) 
#             save_pose_time = time.time() - start_save_pose_time
#             logging.info(f"保存姿态时间: {save_pose_time:.2f} 秒")

#             # 可视化代码
#             start_save_vis_time = time.time()
#             if debug >= 1:
#                 vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)


#             # 如果调试级别设置为 2 或更高，保存跟踪可视化
#             if debug >= 2:
#                 os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
#                 imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
#                 save_vis_time = time.time() - start_save_vis_time
#                 logging.info(f"保存跟踪可视化时间: {save_vis_time:.2f} 秒")


#         # 返回响应
#         return PoseEstResponse(obj_poses=obj_poses)

#     def spin(self):
#         rospy.spin()


# if __name__ == '__main__':
#     try:
#         node = PoseEstimatorNode()
#         node.spin()
#     except rospy.ROSInterruptException:
#         pass


#!/usr/bin/env python3
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from estimater import *
from datareader import *
import rospy
import cv2
import numpy as np
import trimesh
import os
import logging
from sensor_msgs.msg import Image
from naviai_manip_srvs.srv import PoseEst, PoseEstResponse,PoseEstResponse
from naviai_manip_msgs.msg import DetItem, ObjPose
from cv_bridge import CvBridge

class PoseEstimatorNode:
    def __init__(self):
        rospy.init_node('pose_estimator_track_node', anonymous=True)
        
        # 获取工作目录路径
        self.code_dir = os.path.dirname(os.path.realpath(__file__))
        
        self.debug_dir = f'{self.code_dir}/debug'
        self.test_scene_dir = f'{code_dir}/demo_data/current'

        self.est_refine_iter = 3
        self.track_refine_iter = 2
        

        self.last_obj_poses = None

        self.mesh_file_dir = f'{self.code_dir}/model/huojia'
        # Load each mesh file in the directory and store in a dictionary
        self.objects = {}
        for mesh_file in os.listdir(self.mesh_file_dir):
            if mesh_file.endswith('.obj') or mesh_file.endswith('.ply'):  # Adjust as necessary for your mesh file types
                mesh_path = os.path.join(self.mesh_file_dir, mesh_file)
                mesh = trimesh.load(mesh_path)

                # Calculate oriented bounds and bounding box for each mesh
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

                # Initialize score and refine models, and set up FoundationPose for each mesh
                scorer = ScorePredictor()
                refiner = PoseRefinePredictor()
                glctx = dr.RasterizeCudaContext()

                # Store mesh information in the dictionary with the mesh filename (without extension) as the key
                mesh_key = os.path.splitext(mesh_file)[0]
                self.objects[mesh_key] = {
                    'mesh': mesh,
                    'to_origin': to_origin,
                    'extents': extents,
                    'bbox': bbox,
                    'pose_estimator': FoundationPose(
                        model_pts=mesh.vertices,
                        model_normals=mesh.vertex_normals,
                        mesh=mesh,
                        scorer=scorer,
                        refiner=refiner,
                        debug_dir=self.debug_dir,
                        debug=0,
                        glctx=glctx
                    )
                }
        
        # 初始化 cv_bridge
        self.bridge = CvBridge()

        # 创建调试目录
        os.makedirs(f'{self.debug_dir}/track_vis', exist_ok=True)
        os.makedirs(f'{self.debug_dir}/ob_in_cam', exist_ok=True)
        self.last_obj_poses = None
        # 定义服务
        self.service = rospy.Service('pose_estimation_service', PoseEst, self.handle_pose_estimation)
        rospy.loginfo("Pose Estimation Service is ready")

        self.rgb_image = None
        self.depth_image = None
        rospy.loginfo("Pose Estimator Node initialized")
        self.last_detection_time = 0
        self.first_time_for_label = {}

    def handle_pose_estimation(self, req):
        
        # 将ROS Image消息转换为OpenCV图像
        depth_path = f'{self.code_dir}/demo_data/current/depth'
        rgb_path = f'{self.code_dir}/demo_data/current/rgb'
        
        rgb_image = self.bridge.imgmsg_to_cv2(req.color_image, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(req.depth_image, "mono16")
        # 确保目录存在，如果不存在则创建
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(rgb_path, exist_ok=True)

        # 假设rgb_image和depth_image已经通过前面的代码正确获取
        # 生成文件名，可以根据你的需求修改命名规则，这里简单使用序号0.jpg作为示例
        rgb_filename = os.path.join(rgb_path, "0.png")
        depth_filename = os.path.join(depth_path, "0.png")

        # 保存RGB图像
        cv2.imwrite(rgb_filename, rgb_image)
        # 保存深度图像，注意深度图像的数据类型可能需要特殊处理，这里假设当前形式可以直接保存
        cv2.imwrite(depth_filename, depth_image)
        
        obj_poses = []
        detection_interval = 180  # Interval to reset detection (in seconds)
        position_threshold = 0.05  # 5 cm threshold
        angle_threshold = 5        # 5 degrees threshold

        for i, item in enumerate(req.items):
            if item.label not in self.first_time_for_label:
                self.first_time_for_label[item.label] = True
            mask_image = self.bridge.imgmsg_to_cv2(item.mask, "mono16")  # 将mask转换为单通道图像
            mask_path = f'{self.code_dir}/demo_data/current/masks'
            mask_filename = os.path.join(mask_path, f"0.png")
            
            if mask_image.max() > 0:  # Avoid division by zero
                mask_image_normalized = (mask_image / mask_image.max()) * 255
                mask_image_normalized = mask_image_normalized.astype(np.uint8)
            else:
                mask_image_normalized = mask_image.astype(np.uint8)

            # Save the normalized mask image
            cv2.imwrite(mask_filename, mask_image_normalized)
            reader = YcbineoatReader(video_dir=self.test_scene_dir, shorter_side=None, zfar=np.inf)

            # Check if item label is in self.objects
            if item.label in self.objects:
                color = reader.get_color(0)
                depth = reader.get_depth(0)
                self.first_time = False
                mask = reader.get_mask(0).astype(bool)
                # Run initial pose estimation
                self.first_time_for_label[item.label] = False  # Set first-time flag for this label to False after initial processing
                start_register_time = time.time()
                pose = self.objects[item.label]["pose_estimator"].register(
                    K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter
                )
                register_time = time.time() - start_register_time
                print(f"Frame {i} initial registration time: {register_time:.2f} seconds")                
                
                # # Check if it's time to re-run detection based on interval
                # if time.time() - self.last_detection_time > detection_interval or self.first_time_for_label[item.label]:
                #     self.first_time = False
                #     mask = reader.get_mask(0).astype(bool)
                #     # Run initial pose estimation
                #     self.first_time_for_label[item.label] = False  # Set first-time flag for this label to False after initial processing
                #     start_register_time = time.time()
                #     pose = self.objects[item.label]["pose_estimator"].register(
                #         K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter
                #     )
                #     register_time = time.time() - start_register_time
                #     print(f"Frame {i} initial registration time: {register_time:.2f} seconds")
                #     # Update last detection time
                    
                # else:
                #     # For subsequent frames, use tracker
                #     est = self.objects[item.label]["pose_estimator"]
                #     start_track_time = time.time()
                #     pose = est.track_one(
                #         rgb=color, depth=depth, K=reader.K, iteration=self.track_refine_iter
                #     )
                #     track_time = time.time() - start_track_time
                #     print(f"Frame {item.label} tracking time: {track_time:.2f} seconds")

                #     position_diff = 0
                #     rotation_diff = 0
                #     # for last_pose_result in self.last_obj_poses:
                #     #      if(last_pose_result.label == item.label):
                #     #         previous_pose = last_pose_result.pose                              
                #     #         position_diff = np.linalg.norm(pose[:3, 3] - previous_pose[:3, 3])
                #     #         rotation_diff = np.degrees(np.arccos((np.trace(np.dot(previous_pose[:3, :3].T, pose[:3, :3])) - 1) / 2))
                #     # Ensure pose and previous_pose are reshaped to 4x4 matrices if they are 1D arrays of length 16

                #     for last_pose_result in self.last_obj_poses:
                #          if(last_pose_result.label == item.label):
                #             previous_pose = last_pose_result.pose 
                #             pose = pose.reshape(4, 4) if pose.ndim == 1 and pose.size == 16 else pose
                #             previous_pose = previous_pose.reshape(4, 4) if previous_pose is not None and previous_pose.ndim == 1 and previous_pose.size == 16 else previous_pose

                #             # Calculate position and angle difference only if pose and previous_pose are 4x4 matrices
                #             if previous_pose is not None and previous_pose.shape == (4, 4) and pose.shape == (4, 4):
                #                 position_diff = np.linalg.norm(pose[:3, 3] - previous_pose[:3, 3])
                #                 rotation_diff = np.degrees(np.arccos((np.trace(np.dot(previous_pose[:3, :3].T, pose[:3, :3])) - 1) / 2))
                #             else:
                #                 # If previous_pose is None or dimensions are incorrect, force re-estimation
                #                 position_diff, rotation_diff = float('inf'), float('inf')
                #     if (position_diff > position_threshold or rotation_diff > angle_threshold):
                #         mask = reader.get_mask(0).astype(bool)
                #         # Run initial pose estimation
                #         start_register_time = time.time()
                #         pose = self.objects[item.label]["pose_estimator"].register(
                #             K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter
                #         )
                #         register_time = time.time() - start_register_time
                #         print(f"Frame {item.label} initial registration time: {register_time:.2f} seconds")
                #         # Update last detection time
                                               
                
                # 可视化代码
                self.debug=2
                if self.debug >= 1:
                    # 将物体姿态保存到文本文件
                    os.makedirs(f'{self.debug_dir}/ob_in_cam', exist_ok=True)
                    np.savetxt(f'{self.debug_dir}/ob_in_cam/{item.label}.txt', pose.reshape(4,4))
                    
                    water = True
                    if(water):
                        # 加载 T_robot_camera 和 T_base_robot 矩阵
                        T_robot_camera_path = '/home/catkin_ws/src/naviai_manipulation/naviai_manip_main/camera_info/cam2torso.npy'
                        T_robot_camera = np.load(T_robot_camera_path)

                        if T_robot_camera.shape != (4, 4):
                            raise ValueError("Loaded T_robot_camera is not a 4x4 matrix.")
                        if pose.shape != (4, 4):
                            raise ValueError("Loaded pose is not a 4x4 matrix.")

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

                        # 设置旋转矩阵为单位矩阵，以保持与 SACRUM 一致的角度
                        pose_in_robot[:3, :3] = np.eye(3)

                        # 计算物品在相机坐标系下的完整 4x4 位姿矩阵
                        T_robot_camera_inv = np.linalg.inv(T_robot_camera)
                        pose = np.dot(T_robot_camera_inv, pose_in_robot)
                    np.save(f'{self.debug_dir}/ob_in_cam/{item.label}.npy', pose.reshape(4,4)) 
                # 如果调试级别设置为 2 或更高，保存跟踪可视化
                if self.debug >= 2:                
                    os.makedirs(f'{self.debug_dir}/track_vis', exist_ok=True)
                    vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                    imageio.imwrite(f'{self.debug_dir}/track_vis/{item.label}.png', vis)
                
            else:
                # Skip processing if label is not in self.objects
                continue


            pose_result = ObjPose()
            pose_result.label = item.label
            pose_result.pose = pose.flatten().astype(np.float32)
            obj_poses.append(pose_result) 

        # 返回响应
        self.last_obj_poses = obj_poses
        self.last_detection_time = time.time()
        return PoseEstResponse(obj_poses=obj_poses)

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = PoseEstimatorNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass