import numpy as np

# 读取cook和pan的位姿矩阵
T_cook = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/pan.npy')
T_pan = np.load('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/spatula.npy')

# 计算T_cook的逆矩阵
T_cook_inv = np.linalg.inv(T_cook)

# 计算pan在cook坐标系下的位姿矩阵
T_pan_in_cook = np.dot(T_cook_inv, T_pan)

# 输出或保存结果
print("pan在cook坐标系下的位姿矩阵:\n", T_pan_in_cook)
np.save('/home/catkin_ws/src/fdpose/scripts/debug/ob_in_cam/spatula_in_pan.npy', T_pan_in_cook)