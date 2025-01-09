#include <iostream>
#include "trt_engine.h"
// #include <stdlib.h>
#include <dlfcn.h>

template <typename Src, typename Aim>
std::vector<Aim> loadDataFromFile(std::string file_path){
    std::ifstream file(file_path, std::ios::binary);
    if (!file.good()) return std::vector<Aim>(0);
    file.seekg(0, file.end);
    int64_t end_byte = file.tellg();
    file.seekg(0, file.beg);
    int64_t beg_byte = file.tellg();
    int64_t byte_length = end_byte - beg_byte;
    int64_t num_element = byte_length / sizeof(Src);
    std::vector<Aim> ret(num_element);
    std::vector<Src> load(num_element);
    file.read(reinterpret_cast<char*>(load.data()), byte_length);
    for (int i = 0; i < num_element; i++) ret[i] = static_cast<Aim>(load[i]);
    return ret;
}

template <typename ResT, typename RefT>
void maxErrCheck(ResT* res, RefT* ref, int head, int check, int tail){
    std::vector<RefT> max_err(check, static_cast<RefT>(0));
    std::vector<RefT> err_ref(check, static_cast<RefT>(0));
    std::vector<RefT> err_res(check, static_cast<RefT>(0));
    std::vector<RefT> err_sum(check, static_cast<RefT>(0));
    for (int h = 0; h < head; h++){
        for (int c = 0; c < check; c++){
            for (int t = 0; t < tail; t++){
                int idx = h*check*tail + c*tail + t;
                RefT err = std::abs(static_cast<RefT>(res[idx] - ref[idx]));
                err_sum[c] += err;
                if (err > max_err[c]){
                    max_err[c] = err;
                    err_ref[c] = ref[idx];
                    err_res[c] = res[idx];
                }
            }
        }
    }
    for (int i = 0; i < 3; i++) std::cout << max_err[i]<<' ';
    std::cout<<std::endl;
    for (int i = 0; i < 3; i++) std::cout << err_res[i]<<' ';
    std::cout<<std::endl;
    for (int i = 0; i < 3; i++) std::cout << err_ref[i]<<' ';
    std::cout<<std::endl;
    for (int i = 0; i < 3; i++) std::cout << err_sum[i]<<' ';
    std::cout<<std::endl;
}

void loadPluginLib(std::string path){
    void* handle = dlopen(path.c_str(), RTLD_LAZY);
    if (handle == nullptr)
    {
        std::cout << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
    }
}

int main(){
    // loadPluginLib("/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8");
    // loadPluginLib("/home/yjy/WS/custom_plugin/build/libcustom_plugin.so");
    std::string file_path = "/home/catkin_ws/src/fdpose/FoundationPose/trt_test/test_1st_dynamic.eng";
    std::vector<float> input0 = loadDataFromFile<float, float>("/home/catkin_ws/src/fdpose/FoundationPose/trt_test/test_1st_input_a_252.bin");
    std::vector<float> input1 = loadDataFromFile<float, float>("/home/catkin_ws/src/fdpose/FoundationPose/trt_test/test_1st_input_b_252.bin");
    std::vector<float> output0 = loadDataFromFile<float, float>("/home/catkin_ws/src/fdpose/FoundationPose/trt_test/test_1st_output_trans_252.bin");
    std::vector<float> output1 = loadDataFromFile<float, float>("/home/catkin_ws/src/fdpose/FoundationPose/trt_test/test_1st_output_rot_252.bin");
    float sum = 0.0f;
    for (int i = 0; i < input0.size(); i++) sum += input0[i];
    std::cout<<sum<<std::endl;
    void* input0_p = nullptr;
    void* input1_p = nullptr;
    void* output0_p = nullptr;
    void* output1_p = nullptr;
    cudaMalloc(&input0_p, 4*252*6*160*160);
    cudaMalloc(&input1_p, 4*252*6*160*160);
    cudaMemcpy(input0_p, input0.data(), sizeof(float)*input0.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(input1_p, input1.data(), sizeof(float)*input1.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&output0_p, 4*252*3);
    cudaMalloc(&output1_p, 4*252*3);
    TrtEngine::TensorRTEngine test_eng;
    test_eng.loadEngineModel(file_path);
    for (int i = 0; i < 10; i++) test_eng.inferEngineModel(input0_p, input1_p, output0_p, output1_p, 4*252*6*160*160, 4*252*3);
    std::vector<float> out_trans(output0.size());
    std::vector<float> out_rot(output1.size());
    cudaMemcpy(out_trans.data(), output0_p, sizeof(float)*out_trans.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_rot.data(), output1_p, sizeof(float)*out_rot.size(), cudaMemcpyDeviceToHost);
    float sum1 = 0.0f;
    for (int i = 0; i < out_trans.size(); i++) sum1 += std::abs(out_trans[i]);
    std::cout<<sum1<<std::endl;
    float sum2 = 0.0f;
    for (int i = 0; i < output0.size(); i++) sum2 += std::abs(output0[i]);
    std::cout<<sum2<<std::endl;
    maxErrCheck<float, float>(out_trans.data(), output0.data(), 252, 3, 1);

    // for (int i = 0; i < 252*3; i++) std::cout<<out_trans[i]<<' '<<output0[i]<<std::endl;
    return 0;
}