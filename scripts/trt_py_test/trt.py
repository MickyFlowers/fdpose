import torch
from build import pyTRTEng

test_model = pyTRTEng.TensorRTEngine(2,2)
test_model.loadEngineModel("/home/catkin_ws/src/fdpose/FoundationPose/trt_test/test_1st_dynamic.eng")

import numpy as np
inputA = np.fromfile("/home/catkin_ws/src/fdpose/FoundationPose/trt_test/test_1st_input_a_252.bin", dtype = np.float32)
inputB = np.fromfile("/home/catkin_ws/src/fdpose/FoundationPose/trt_test/test_1st_input_b_252.bin", dtype = np.float32)
output_ref_trans = np.fromfile("/home/catkin_ws/src/fdpose/FoundationPose/trt_test/test_1st_output_trans_252.bin", dtype = np.float32)
output_ref_rot = np.fromfile("/home/catkin_ws/src/fdpose/FoundationPose/trt_test/test_1st_output_rot_252.bin", dtype = np.float32)

input_tensor_A = torch.from_numpy(inputA).cuda()
input_tensor_B = torch.from_numpy(inputB).cuda()
output_res_trans = torch.from_numpy(output_ref_trans)
output_res_trans = torch.zeros_like(output_res_trans).cuda()
output_res_rot = torch.from_numpy(output_ref_rot)
output_res_rot = torch.zeros_like(output_res_rot).cuda()

test_model.inferEngineModel([input_tensor_A, input_tensor_B], [output_res_trans, output_res_rot])

diff = output_ref_trans - output_res_trans.cpu().numpy()
print(abs(diff).sum(), abs(output_ref_trans).sum())