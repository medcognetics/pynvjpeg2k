# 
# Copyright (c) 2020 - 2021, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(pynvjpeg LANGUAGES CXX CUDA)

message(STATUS "Python Executable: ${Python_EXECUTABLE}")
message(STATUS "Python Include Dirs: ${Python_INCLUDE_DIRS}")

# find pybind11
find_package(pybind11 REQUIRED PATHS ${PYBIND_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include_directories(
  SYSTEM 
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Provided by nvidia-nvjpeg2k-cuX
set(NVJPEG2K_LIB ${Python_INCLUDE_DIRS}/nvidia/nvjpeg2k/lib/libnvjpeg2k.so.0)
message(STATUS "NVJPEG2K_LIBRARY at: ${NVJPEG2K_LIB}")
set(NVJPEG2K_INCLUDE ${Python_INCLUDE_DIRS}/nvidia/nvjpeg2k/include)
message(STATUS "NVJPEG2K_INCLUDE at: ${NVJPEG2K_INCLUDE}")

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
target_include_directories(${PROJECT_NAME} PRIVATE ${NVJPEG2K_INCLUDE})
target_link_libraries(${PROJECT_NAME} PRIVATE ${NVJPEG2K_LIB} ${CUDART_LIB})

# Add the directory to the rpath
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH ${Python_INCLUDE_DIRS}/nvidia/nvjpeg2k/lib
    INSTALL_RPATH ${Python_INCLUDE_DIRS}/nvidia/nvjpeg2k/lib
)
