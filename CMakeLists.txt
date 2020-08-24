
cmake_minimum_required(VERSION 2.8)
project( jetson_proc )
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
list( APPEND CUDA_NVCC_FLAGS "-std=c++11" )
cuda_add_executable( jetson_proc video.cu )
target_link_libraries( jetson_proc -lpthread ${OpenCV_LIBS} )
