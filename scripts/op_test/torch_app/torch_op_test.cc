#include "op_test.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

bool korniaGeometryPerspectiveFillZeroNoAlignNearestBCHW(at::Tensor &dst, at::Tensor &src, at::Tensor &M){
    if (dst.dim() != 4 || src.dim() != 4 || M.dim() != 3){
        std::cout<< "output & input tensor dim should be BCHW, transform matrix tensor dim should be Bx3x3"<<std::endl;
        return false;
    }
    if (!dst.is_cuda() || !src.is_cuda() || !M.is_cuda()){
        std::cout<<"only support gpu device"<<std::endl;
        return false;
    }
    if (dst.size(0) != src.size(0) || dst.size(0) != M.size(0)){
        std::cout<<"tensors' batch (dim 0) not unified"<<std::endl;
        return false;
    }
    int batch = dst.size(0);
    if (dst.size(1) != src.size(1)){
        std::cout<< "input & output (BCHW) channel not unified"<<std::endl;
        return false;
    }
    int channel = dst.size(1);
    int dst_h = dst.size(2);
    int dst_w = dst.size(3);
    int src_h = src.size(2);
    int src_w = src.size(3);
    if ((dst_h * dst_w) % 128 != 0){
        std::cout<<"output size must be multiple of 128, e.g. H=16,W=16 --> 16*16%128=0"<<std::endl;
        return false;
    }
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    gpuTestKorniaPerspective(reinterpret_cast<float*>(dst.data_ptr()), 
                            reinterpret_cast<float*>(src.data_ptr()), 
                            reinterpret_cast<float*>(M.data_ptr()),
                            batch, channel, src_h, src_w, dst_h, dst_w, torch_stream);
    cudaStreamSynchronize(torch_stream);
    return true;
}


PYBIND11_MODULE(customOps, m){
    m.def("korniaGeometryPerspectiveFillZeroNoAlignNearestBCHW", &korniaGeometryPerspectiveFillZeroNoAlignNearestBCHW,
        "kornia.geometry.transform.perspective fill=zero align_coner=false method=nearest cuda");
}