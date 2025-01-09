#include "trt_engine.h"
#include <chrono>

namespace TrtEngine{

    bool TensorRTEngine::loadEngineModel(const std::string &eng_path){
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
    }

    bool TensorRTEngine::inferEngineModel(void* input0_p, void* input1_p, void* output0_p, void* output1_p, int input_size, int output_size){
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        context_->setInputTensorAddress(engine_->getIOTensorName(0), input0_p);
        context_->setInputTensorAddress(engine_->getIOTensorName(1), input1_p);
        context_->setTensorAddress(engine_->getIOTensorName(2), output0_p);
        context_->setTensorAddress(engine_->getIOTensorName(3), output1_p);
        cudaStreamSynchronize(stream);
        auto t0 = std::chrono::system_clock::now();
        context_->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::system_clock::now();
        double duration = std::chrono::duration<double>(t1 - t0).count();
        duration *= 1000.0f;
        std::cout<<"time cost: "<<duration<<" ms"<<std::endl;
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return true;
    }
}