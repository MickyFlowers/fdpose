#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>
#include <vector>

void gpuTestKorniaPerspective(float* dst, float* src, float* M, int batch, int channel, int src_h, int src_w, int dst_h, int dst_w, cudaStream_t stream);

void cpuTestKorniaPerspective(float* dst, float* src, float* M, int batch, int channel, int src_h, int src_w, int dst_h, int dst_w);
//only support foudationpose current scene