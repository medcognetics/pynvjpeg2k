#ifndef PYNVJPEG_HELPERS
#define PYNVJPEG_HELPERS
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>
#include <nvjpeg.h>

namespace py = pybind11;

namespace pynvjpeg {

std::string nvjpegGetErrorString(nvjpeg2kStatus_t e) {
  switch(e) {
    case NVJPEG2K_STATUS_SUCCESS: return std::string("NVJPEG2K_STATUS_SUCCESS");
    case NVJPEG2K_STATUS_NOT_INITIALIZED: return std::string("NVJPEG2K_STATUS_NOT_INITIALIZED");               
    case NVJPEG2K_STATUS_INVALID_PARAMETER: return std::string("NVJPEG2K_STATUS_INVALID_PARAMETER");             
    case NVJPEG2K_STATUS_BAD_JPEG: return std::string("NVJPEG2K_STATUS_BAD_JPEG");                      
    case NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED: return std::string("NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED");            
    case NVJPEG2K_STATUS_ALLOCATOR_FAILURE: return std::string("NVJPEG2K_STATUS_ALLOCATOR_FAILURE");             
    case NVJPEG2K_STATUS_EXECUTION_FAILED: return std::string("NVJPEG2K_STATUS_EXECUTION_FAILED");              
    case NVJPEG2K_STATUS_ARCH_MISMATCH: return std::string("NVJPEG2K_STATUS_ARCH_MISMATCH");                 
    case NVJPEG2K_STATUS_INTERNAL_ERROR: return std::string("NVJPEG2K_STATUS_INTERNAL_ERROR");                
    case NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED: return std::string("NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED");  
  }
  throw std::invalid_argument("Invalid nvjpeg2kStatus_t");
}


template <typename D>
cudaError_t deviceToHostCopy(
  unsigned char* deviceBuf,
  const size_t deviceBufPitchInBytes,
  D* hostBuf,
  const size_t rows,
  const size_t cols,
  cudaStream_t *stream
) {
  const size_t hostBufPitchInBytes = cols * sizeof(D);
  const size_t copySize = cols * sizeof(D);
  if (stream == NULL) {
    return cudaMemcpy2D(hostBuf, hostBufPitchInBytes, (D *)deviceBuf, deviceBufPitchInBytes, copySize, rows, cudaMemcpyDeviceToHost);
  }
  else {
    return cudaMemcpy2DAsync(hostBuf, hostBufPitchInBytes, (D *)deviceBuf, deviceBufPitchInBytes, copySize, rows, cudaMemcpyDeviceToHost, *stream);
  }
}


template <typename D>
cudaError_t hostToDeviceCopy(
  unsigned char* deviceBuf,
  const size_t deviceBufPitchInBytes,
  D* hostBuf,
  const size_t rows,
  const size_t cols,
  cudaStream_t *stream
) {
  const size_t hostBufPitchInBytes = cols * sizeof(D);
  const size_t copySize = cols * sizeof(D);
  if (stream == NULL) {
    return cudaMemcpy2D((D *)deviceBuf, deviceBufPitchInBytes, hostBuf, hostBufPitchInBytes, copySize, rows, cudaMemcpyHostToDevice);
  }
  else {
    return cudaMemcpy2DAsync((D *)deviceBuf, deviceBufPitchInBytes, hostBuf, hostBufPitchInBytes, copySize, rows, cudaMemcpyHostToDevice, *stream);
  }
}


template <typename D>
cudaError_t deviceMalloc(
  unsigned char **buffer,
  size_t *pitchInBytes,
  const size_t rows,
  const size_t cols,
  const size_t batchSize
) {
  return cudaMallocPitch((void**)buffer, pitchInBytes, cols * sizeof(D), batchSize *rows);
}


/*
 * Seeks to a given frame index from the start of the buffer
 */
template <typename D>
D* seekToFrameNumber(
  D* ptr,
  const size_t pitchInBytes,
  const size_t rows,
  const size_t targetFrameIndex
) {
  size_t offset = (rows * pitchInBytes) * targetFrameIndex;
  return ptr + offset;
}


// Template definitions for linker

template cudaError_t deviceToHostCopy<uint16_t>(
  unsigned char* deviceBuf,
  const size_t deviceBufPitchInBytes,
  uint16_t* hostBuf,
  const size_t rows,
  const size_t cols,
  cudaStream_t *stream = NULL
); 


template cudaError_t deviceMalloc<uint16_t>(
  unsigned char **buffer,
  size_t *pitchInBytes,
  const size_t rows,
  const size_t cols,
  const size_t batchSize = 1
);


template cudaError_t hostToDeviceCopy<uint16_t>(
  unsigned char* deviceBuf,
  const size_t deviceBufPitchInBytes,
  uint16_t* hostBuf,
  const size_t rows,
  const size_t cols,
  cudaStream_t *stream = NULL
);

template uint16_t* seekToFrameNumber<uint16_t>(
  uint16_t* ptr,
  const size_t pitchInBytes,
  const size_t rows,
  const size_t targetFrameIndex
);

template  unsigned char* seekToFrameNumber<unsigned char>(
  unsigned char* ptr,
  const size_t pitchInBytes,
  const size_t rows,
  const size_t targetFrameIndex
);


}


#endif
