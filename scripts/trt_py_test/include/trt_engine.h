#ifndef TRT_ENGINE_H
#define TRT_ENGINE_H
#include <NvInfer.h>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

namespace TrtEngine
{

class Logger : public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kWARNING) std::cout<<msg<<std::endl;
    }
};

Logger logger;
class TensorRTEngine{
public:
    TensorRTEngine(){}
    ~TensorRTEngine(){}
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<nvinfer1::Dims> tensor_dims_;
    std::vector<std::string> tensor_names_;
public:
    void cudaCheck(cudaError_t ret, std::string cur_func_pos){
        if (ret != cudaSuccess) std::cout<<cur_func_pos<<": "<<cudaGetErrorString(ret)<<std::endl;
    }

    bool loadEngineModel(const std::string &eng_path);
    
    bool inferEngineModel(void* input0_p, void* input1_p, void* output0_p, void* output1_p, int input_size, int output_size);
    //参数需要改动统一
};



} // namespace TrtEngine

#endif