#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>
#include <nvjpeg.h>


namespace py = pybind11;


#define NUM_COMPONENTS 1

#define CHECK_CUDA(call)                                                                                          \
    {                                                                                                             \
        cudaError_t _e = (call);                                                                                  \
        if (_e != cudaSuccess)                                                                                    \
        {                                                                                                         \
            std::cerr << "CUDA Runtime failure: '#" << _e <<"' " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                                     \
        }                                                                                                         \
    }

#define CHECK_NVJPEG2K(call)                                                                                \
    {                                                                                                       \
        nvjpeg2kStatus_t _e = (call);                                                                       \
        if (_e != NVJPEG2K_STATUS_SUCCESS)                                                                  \
        {                                                                                                   \
            std::cerr << "NVJPEG2K failure: '#" << _e << "' " << nvjpegGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                            \
        }                                                                                                   \
    }


#define CHECK_NVJPEG(call)                                                      \
    {                                                                           \
        nvjpegStatus_t _e = (call);                                             \
        if (_e != NVJPEG_STATUS_SUCCESS)                                        \
        {                                                                       \
            std::cerr << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }


#define CHECK_CUDA_NO_RETURN(call)                                                                                          \
    {                                                                                                             \
        cudaError_t _e = (call);                                                                                  \
        if (_e != cudaSuccess)                                                                                    \
        {                                                                                                         \
            std::stringstream msg; \
            msg << "CUDA Runtime failure: '#" << _e <<"' " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(msg.str());\
        }                                                                                                         \
    }

#define CHECK_NVJPEG2K_NO_RETURN(call)                                                                                \
    {                                                                                                       \
        nvjpeg2kStatus_t _e = (call);                                                                       \
        if (_e != NVJPEG2K_STATUS_SUCCESS)                                                                  \
        {                                                                                                   \
            std::stringstream msg; \
            msg << "NVJPEG2K failure: '#" << _e << "' " << nvjpegGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(msg.str());\
        }                                                                                                   \
    }


#define CHECK_NVJPEG_NO_RETURN(call)                                                                                \
    {                                                                                                       \
        nvjpegStatus_t _e = (call);                                                                       \
        if (_e != NVJPEG_STATUS_SUCCESS)                                                                  \
        {                                                                                                   \
            std::stringstream msg; \
            msg << "NVJPEG failure: '#" << _e << "' " << nvjpegGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(msg.str());\
        }                                                                                                   \
    }



namespace py = pybind11;

namespace pynvjpeg {


std::string nvjpegGetErrorString(nvjpeg2kStatus_t e);


template <typename D>
cudaError_t deviceToHostCopy(
  unsigned char* deviceBuf,
  const size_t deviceBufPitchInBytes,
  D* hostBuf,
  const size_t rows,
  const size_t cols,
  cudaStream_t *stream = NULL
); 


template <typename D>
cudaError_t hostToDeviceCopy(
  unsigned char* deviceBuf,
  const size_t deviceBufPitchInBytes,
  D* hostBuf,
  const size_t rows,
  const size_t cols,
  cudaStream_t *stream = NULL
);


template <typename D>
cudaError_t deviceMalloc(
  unsigned char **buffer,
  size_t *pitchInBytes,
  const size_t rows,
  const size_t cols,
  const size_t batchSize = 1
);


/*
 * Seeks to a given frame index from the start of the buffer
 */
template <typename D>
D* seekToFrameNumber(
  D* ptr,
  const size_t pitchInBytes,
  const size_t rows,
  const size_t targetFrameIndex
);




}
