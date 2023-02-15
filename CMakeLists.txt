# 
# Copyright (c) 2020 - 2021, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
#set(CMAKE_CUDA_COMPILER  /usr/local/cuda-12.0/bin/nvcc)

project(pynvjpeg LANGUAGES CXX CUDA)


# find pybind11
set(pybind11_DIR extern/pybind11)
find_package(pybind11 REQUIRED)

set(NVJPEG2K_PATH libnvjpeg)
if(NOT NVJPEG2K_PATH)
message(FATAL_ERROR, "NVJPEG2K_PATH not set")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include_directories(
  ${NVJPEG2K_PATH}/include
  SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

find_library(NVJPEG_LIBRARY nvjpeg ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

find_library(NVJPEG2K_LIB
    NAMES nvjpeg2k
    PATHS ${NVJPEG2K_PATH}/lib)

find_library(CUDART_LIB 
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

if(UNIX)
  set(FILESYS -lstdc++fs)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

  
pybind11_add_module(
  ${PROJECT_NAME} 
  SHARED
  "src/pynvjpeg.cpp" 
  "src/nvjpeg2k_encode.cpp" 
  "src/nvjpeg2k_decode.cpp" 
  "src/dicom.cpp" 
  "src/helpers.cpp"
)
target_link_libraries(${PROJECT_NAME} PUBLIC ${NVJPEG2K_LIB} ${CUDART_LIB})




