#include <NvInfer.h>
// #include <torch/torch.h>
#include <torch/extension.h>
// #include <pybind11/pybind11.h>
#include <c10/cuda/CUDAStream.h>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

class Logger : public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kWARNING) std::cout<<msg<<std::endl;
    }
};

Logger logger;
class TensorRTEngine{
public:
    TensorRTEngine() = delete;
    TensorRTEngine(const int &inputs_num, const int &outputs_num);
    ~TensorRTEngine(){}
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<nvinfer1::Dims> tensor_dims_;
    std::vector<std::string> tensor_names_;
    cudaStream_t stream_;
    int inputs_num_;
    int outputs_num_;
    std::vector<void*> inputs_ptr_;
    std::vector<void*> outputs_ptr_;
public:
    void cudaCheck(cudaError_t ret, std::string cur_func_pos){
        if (ret != cudaSuccess) std::cout<<cur_func_pos<<": "<<cudaGetErrorString(ret)<<std::endl;
    }

    bool loadEngineModel(const std::string &eng_path);
    
    bool inferEngineModel(std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &outputs);
};

TensorRTEngine::TensorRTEngine(const int &inputs_num, const int &outputs_num){
    cudaStreamCreate(&stream_);
    inputs_num_ = inputs_num;
    outputs_num_ = outputs_num;
    inputs_ptr_.resize(inputs_num);
    outputs_ptr_.resize(outputs_num);
}

bool TensorRTEngine::loadEngineModel(const std::string &eng_path){
    std::cout<<"loading trt engine"<<std::endl;
    std::ifstream eng_file(eng_path, std::ios::binary);
    if (!eng_file.good()){
        std::cout<<"engine file path wrong: "<<eng_path<<" no such file"<<std::endl;
        return false;
    }
    eng_file.seekg(0, eng_file.end);
    int64_t end_pos = eng_file.tellg();
    eng_file.seekg(0, eng_file.beg);
    int64_t beg_pos = eng_file.tellg();
    int64_t file_length = end_pos - beg_pos;
    std::vector<char> data(file_length);
    eng_file.read(data.data(), file_length);
    eng_file.close();

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (engine_ == nullptr) return false;
    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (context_ == nullptr) return false;

    int num_inputs = engine_->getNbIOTensors();
    for (int i = 0; i < num_inputs; i++){
        tensor_names_.emplace_back(engine_->getIOTensorName(i));
        tensor_dims_.emplace_back(engine_->getTensorShape(tensor_names_[i].c_str()));
    }
    cudaDeviceSynchronize();
    // std::cout<<"well1"<<std::endl;
    return true;
    return true;
}
    
bool TensorRTEngine::inferEngineModel(std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &outputs){
    if (inputs.size() != inputs_num_){
        std::cout<< "model has "<<inputs_num_<<" input tensors but get "<<inputs.size()<<std::endl;
        return false;
    }
    if (outputs.size() != outputs_num_){
        std::cout<< "model has "<<outputs_num_<<" output tensors but get "<<outputs.size()<<std::endl;
        return false;
    }
    for (int i = 0; i < inputs_num_; i++) context_->setInputTensorAddress(engine_->getIOTensorName(i), inputs[i].data_ptr()); 
    for (int i = 0; i < outputs_num_; i++) context_->setTensorAddress(engine_->getIOTensorName(inputs_num_ + i), outputs[i].data_ptr());
    cudaStreamSynchronize(stream_);
    // auto t0 = std::chrono::system_clock::now();
    context_->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);
    // auto t1 = std::chrono::system_clock::now();
    // double duration = std::chrono::duration<double>(t1 - t0).count();
    // duration *= 1000.0f;
    // std::cout<<"time cost: "<<duration<<" ms"<<std::endl;
    return true;
}

PYBIND11_MODULE(pyTRTEng, m){
    pybind11::class_<TensorRTEngine>(m, "TensorRTEngine")
            .def(pybind11::init<const int &, const int &>())
            .def("loadEngineModel", &TensorRTEngine::loadEngineModel)
            .def("inferEngineModel", &TensorRTEngine::inferEngineModel);
}