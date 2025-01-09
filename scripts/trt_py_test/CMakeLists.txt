set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "-Xcompiler -fPIC ${CMAKE_CUDA_FLAGS}\
                    -gencode arch=compute_86,code=compute_86 -std=c++17")

cmake_minimum_required(VERSION 3.16)
project(trt)
enable_language(CXX CUDA)

set(SRC_FILE src/trt_engine.cc
            plugins/common/checkMacrosPlugin.cpp  
            plugins/common/plugin.cpp  
            plugins/layerNormPlugin/layerNormKernel.cu
            plugins/layerNormPlugin/layerNormPlugin.cpp
)

set(INCLUDE_PATHS include/
                plugins/common/
                plugins/layerNormPlugin/
                /usr/local/cuda-11.4/include/
                /usr/include/aarch64-linux-gnu/
)
set(LINK_LIBS /usr/local/cuda-11.4/lib64/libcudart.so
            /usr/lib/aarch64-linux-gnu/libnvinfer.so
            /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
)

add_library(trt_lib SHARED ${SRC_FILE})
target_include_directories(trt_lib PUBLIC ${INCLUDE_PATHS})
target_link_libraries(trt_lib PUBLIC ${LINK_LIBS})

add_executable(0_trt trt.cpp)
target_include_directories(0_trt PUBLIC ${INCLUDE_PATHS})
target_link_libraries(0_trt PUBLIC trt_lib -ldl)

set(PYBIND_SRC src/trt_engine_pybind.cc
            plugins/common/checkMacrosPlugin.cpp  
            plugins/common/plugin.cpp  
            plugins/layerNormPlugin/layerNormKernel.cu
            plugins/layerNormPlugin/layerNormPlugin.cpp
)
list(APPEND CMAKE_PREFIX_PATH "/root/miniforge3/envs/fdpose/lib/python3.8/site-packages/torch")
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
#list(APPEND CMAKE_PREFIX_PATH "/root/miniforge3/envs/fdpose/lib/python3.8/site-packages/pybind11")
find_package(pybind11 REQUIRED PATHS /root/miniforge3/envs/fdpose/lib/python3.8/site-packages/pybind11
			 NO_DEFAULT_PATH)
set(PYBIND11_CPP_STANDARD -std=c++17)
pybind11_add_module(pyTRTEng ${PYBIND_SRC})
target_include_directories(pyTRTEng PRIVATE ${INCLUDE_PATHS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(pyTRTEng PRIVATE ${LINK_LIBS} ${TORCH_LIBRARIES} pybind11::module 
                        /root/miniforge3/envs/fdpose/lib/libpython3.8.so
                        /root/miniforge3/envs/fdpose/lib/python3.8/site-packages/torch/lib/libtorch_python.so)

# add_library(pyTRTEng SHARED ${PYBIND_SRC})
# target_include_directories(pyTRTEng PRIVATE ${INCLUDE_PATHS} ${TORCH_INCLUDE_DIRS} 
#                         "/home/yjy/.local/lib/python3.8/site-packages/torch/include/pybind11"
#                         "/usr/include/python3.8")
# target_link_libraries(pyTRTEng PRIVATE ${LINK_LIBS} ${TORCH_LIBRARIES}
#                             /home/yjy/.local/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
target_compile_definitions(pyTRTEng PRIVATE TORCH_EXTENSION_NAME=pyTRTEng)

message(********* ${TORCH_INCLUDE_DIRS})
message(********* ${TORCH_LIBRARIES})