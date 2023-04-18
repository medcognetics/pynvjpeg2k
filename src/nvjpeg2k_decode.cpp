#ifndef PYNVJPEG2K_DECODE
#define PYNVJPEG2K_DECODE
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>
#include <nvjpeg.h>

#include "helpers.h"
#include "dicom.h"
#include "nvjpeg2k_decode.h"

namespace py = pybind11;

namespace pynvjpeg {
namespace jpeg2k {


int checkStream(
    const char* buffer,
    const size_t size
) {
  // Initialize stream and handle
  nvjpeg2kHandle_t handle;
  nvjpeg2kStream_t stream;
  CHECK_NVJPEG2K(nvjpeg2kCreateSimple(&handle));
  CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&stream));

  // Parse stream and get image info
  nvjpeg2kImageInfo_t imageInfo;
  CHECK_NVJPEG2K(nvjpeg2kStreamParse(handle, (unsigned char*)buffer, size, 0, 0, stream));
  CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(stream, &imageInfo));
  return EXIT_SUCCESS;
}


bool isValid(
    const char* buffer,
    const size_t size
) {
  return checkStream(buffer, size) == EXIT_SUCCESS;
}


int _async_frame_decode(
    const unsigned char *srcBuffer, 
    const std::size_t srcBufSize, 
    DecodeParams_t *params,
    unsigned char *devBuffer, 
    size_t pitchInBytes,
    cudaStream_t *stream = NULL
) {
  // build struct expected by nvjpeg
  nvjpeg2kImage_t output_image;
  output_image.pixel_data = (void**)&devBuffer;
  output_image.pixel_type = NVJPEG2K_UINT16;
  output_image.pitch_in_bytes = &pitchInBytes;
  output_image.num_components = 1;

  // Run decode
  if (stream == NULL) {
    CHECK_NVJPEG2K(nvjpeg2kDecode(*params->handle, params->decodeState, params->jpegStream, &output_image, 0)); 
  }
  else {
    CHECK_NVJPEG2K(nvjpeg2kDecode(*params->handle, params->decodeState, params->jpegStream, &output_image, *stream)); 
  }

  return EXIT_SUCCESS;
}


int _decode_frames(
    std::vector<const char*> frameBuffers,
    std::vector<size_t> bufferSizes,
    const std::size_t rows,
    const std::size_t cols,
    uint16_t *outBuffer,
    size_t batchSize 
) {
  const size_t numFrames = frameBuffers.size();
  batchSize = std::min(numFrames, batchSize);
  if (frameBuffers.size() != bufferSizes.size()) {
    throw std::invalid_argument("frameBuffers length should match bufferSizes length");
  }
  if (numFrames <= 0) {
    throw std::invalid_argument("frameBuffers must have nonzero length");
  }
  if (batchSize < 1) {
    throw std::invalid_argument("batchSize must be >= 1");
  }
  if (numFrames % batchSize != 0) {
    throw std::invalid_argument("numFrames must be evenly divisble by batchSize");
  }

  // init all pipeline stages
  const size_t PIPELINE_STAGES = batchSize;
  assert(PIPELINE_STAGES > 0);
  DecodeParams_t stageParams[PIPELINE_STAGES];
  nvjpeg2kHandle_t handle;
  nvjpeg2kCreateSimple(&handle);
  for (size_t p=0; p < PIPELINE_STAGES; p++) {
    stageParams[p].rows = rows;
    stageParams[p].cols = cols;
    stageParams[p].handle = &handle;
    nvjpeg2kDecodeStateCreate(*stageParams[p].handle, &stageParams[p].decodeState);
    nvjpeg2kStreamCreate(&stageParams[p].jpegStream);
    cudaStreamCreateWithFlags(&stageParams[p].cudaStream, cudaStreamNonBlocking);
  }

  // Allocate GPU memory to receive decode.
  // Allocated size will be 2D array of size (batchSize * Rows x Pitch)
  unsigned char* devBuffer;
  size_t pitchInBytes;
  CHECK_CUDA(deviceMalloc<uint16_t>(&devBuffer, &pitchInBytes, batchSize * rows, cols));

  // Loop over frames to be decoded
  int err = EXIT_SUCCESS;
  for (size_t frameIndex = 0; frameIndex < numFrames; frameIndex++) {
    // Get buffer info for this frame
    const unsigned char* buffer = reinterpret_cast<const unsigned char*>(frameBuffers.at(frameIndex));
    size_t size = bufferSizes.at(frameIndex);

    // Get decode params for the pipeline stage we want to use
    const size_t stage = frameIndex % PIPELINE_STAGES;
    DecodeParams_t *params = &stageParams[stage];

    // Seek devBuffer to the index of the frame we are decoding within the batch
    unsigned char* devBufferThisFrame = seekToFrameNumber<unsigned char>(devBuffer, pitchInBytes, rows, frameIndex % batchSize);

    // Ensure the previous stage has finished
    if(frameIndex >= PIPELINE_STAGES) {
      CHECK_CUDA(cudaStreamSynchronize(params->cudaStream));
    }
    
    // Parse the stream
    CHECK_NVJPEG2K(nvjpeg2kStreamParse(*params->handle, buffer, size, 0, 0, params->jpegStream));

    // Read image info
    nvjpeg2kImageInfo_t imageInfo;
    CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(params->jpegStream, &imageInfo));

    if (imageInfo.num_components != 1) {
      throw std::invalid_argument("Only single channel images are supported");
    }


    // Update component info based on JPEG2K header
    std::vector<nvjpeg2kImageComponentInfo_t> imageComponentInfo;
    imageComponentInfo.resize(imageInfo.num_components);
    for (uint32_t c = 0; c < imageInfo.num_components; c++)
    {
        CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(params->jpegStream, &imageComponentInfo[c], c));
        if (imageComponentInfo[c].sgn != 0) {
          throw std::invalid_argument("Only unsigned images are supported");
        }
    }

    // Submit decode job
    err = _async_frame_decode(buffer, size, params, devBufferThisFrame, pitchInBytes, &params->cudaStream);
    if (err != EXIT_SUCCESS) {
      std::cerr << "Error decoding frame at index " << frameIndex << std::endl;
      break;
    }

    // Copy decoded result back to host
    uint16_t* outBufferThisFrame = seekToFrameNumber<uint16_t>(outBuffer, cols, rows, frameIndex);
    CHECK_CUDA(deviceToHostCopy<uint16_t>(devBufferThisFrame, pitchInBytes, outBufferThisFrame, params->rows, params->cols, &params->cudaStream));
  }

  // Free all resources
  for (size_t p=0; p < PIPELINE_STAGES; p++) {
    DecodeParams_t *params = &stageParams[p];
    CHECK_CUDA(cudaStreamSynchronize(params->cudaStream));
    CHECK_CUDA(cudaStreamDestroy(stageParams[p].cudaStream));
    CHECK_NVJPEG2K(nvjpeg2kStreamDestroy(stageParams[p].jpegStream));
    CHECK_NVJPEG2K(nvjpeg2kDecodeStateDestroy(stageParams[p].decodeState));
  }
  CHECK_CUDA(cudaFree(devBuffer));
  CHECK_NVJPEG2K(nvjpeg2kDestroy(handle));
  CHECK_CUDA(cudaDeviceReset());
  return err;
}


/*
 * Run decode on a single frame
 *
 * Args:
 *    buffer - Data buffer
 *    inBufSize - Size of data buffer
 *    rows - Rows in output image
 *    cols - Cols in output image
 *    
 * Returns:
 *  2D array of decoded pixel data
 */
py::array_t<uint16_t> decode(
    const char* buffer, 
    const size_t inBufSize, 
    const size_t rows, 
    const size_t cols
) {
  // Allocate array to receive decoded data
  py::array_t<uint16_t> outBuffer(
    {rows, cols}, 
    {cols*sizeof(uint16_t), sizeof(uint16_t)}
  );

  // Wrap single frame in vector and run frame decode
  std::vector<const char*> frameBuffers({buffer});
  std::vector<size_t> bufferSizes({inBufSize});
  int err = _decode_frames(
      frameBuffers,
      bufferSizes,
      rows, 
      cols,
      (uint16_t *)outBuffer.data(),
      1
  );

  if (err) {
    cudaDeviceReset();
    throw std::invalid_argument("error");
  }
  return outBuffer;
}


/*
 * Run batched decode on multiple frames
 *
 * Args:
 *    buffer - Data buffer of multiple frames
 *    inBufSize - Size of data buffer
 *    rows - Rows in output image
 *    cols - Cols in output image
 *    batchSize - Batch size for decoding
 *    
 * Returns:
 *  3D array of decoded pixel data
 */
py::array_t<uint16_t> decode_frames(
    const char* buffer,
    size_t size,
    const size_t rows, 
    const size_t cols,
    const int batchSize
) {
  // Scan PixelData buffer and to find frame offsets and sizes
  std::vector<dicom::FrameInfo_t> frameInfo = dicom::getFrameInfo(buffer, size);
  std::vector<const char*> frameBuffers;
  std::vector<size_t> bufferSizes;
  for (auto x : frameInfo) {
      const char* frameBuffer = buffer + x.offset;
      frameBuffers.push_back(frameBuffer);
      bufferSizes.push_back(x.length);
  }
  const size_t numFrames = frameBuffers.size();

  // Allocate output array
  py::array_t<uint16_t> outBuffer(
    {numFrames, rows, cols}, 
    {rows*cols*sizeof(uint16_t), cols*sizeof(uint16_t), sizeof(uint16_t)}
  );

  // Run decode over frames
  int err = _decode_frames(
      frameBuffers,
      bufferSizes,
      rows, 
      cols,
      (uint16_t *)outBuffer.data(),
      batchSize
  );

  if (err) {
    std::stringstream ss;
    ss << "Decode error " << err;
    throw std::runtime_error(ss.str());
  }
  return outBuffer;
}


/*
 * Run batched decode on multiple frames with offsets
 *
 * Args:
 *    frameBuffers - Vector of frame buffers
 *    bufferSizes - Size of each frame buffer
 *    rows - Rows in output image
 *    cols - Cols in output image
 *    batchSize - Batch size for decoding
 *    
 * Returns:
 *  3D array of decoded pixel data
 */
py::array_t<uint16_t> decode_framelist(
    const char* buffer,
    std::vector<size_t> bufferSizes,
    const size_t rows, 
    const size_t cols,
    const int batchSize
) {
  const size_t numFrames = bufferSizes.size();

  // Allocate output array
  py::array_t<uint16_t> outBuffer(
    {numFrames, rows, cols}, 
    {rows*cols*sizeof(uint16_t), cols*sizeof(uint16_t), sizeof(uint16_t)}
  );

  // Pybind11 has trouble with std::vector<const char*> as an argument
  // so we need to convert the const char* to a vector of const char*
  std::vector<const char*> frameBuffers;
  size_t totalOffset = 0;
  for (auto x : bufferSizes) {
    const char* frameBuffer = buffer + totalOffset;
    frameBuffers.push_back(frameBuffer);
    totalOffset += x;
  }

  // Run decode over frames
  int err = _decode_frames(
      frameBuffers,
      bufferSizes,
      rows, 
      cols,
      (uint16_t *)outBuffer.data(),
      batchSize
  );

  if (err) {
    std::stringstream ss;
    ss << "Decode error " << err;
    throw std::runtime_error(ss.str());
  }
  return outBuffer;
}


void pybind_init_dec(py::module &m) {
  m.def(
    "is_valid_jpeg2k", 
    &isValid,
    "validate"
  );
  m.def(
    "get_image_info_jpeg2k", 
    &getImageInfo,
    "read"
  );
  m.def(
    "decode_jpeg2k", 
    &decode,
    "Run decode"
  );
  m.def(
    "decode_frames_jpeg2k", 
    &decode_frames,
    "Run decode on frames"
  );
  m.def(
    "decode_framelist_jpeg2k", 
    &decode_framelist,
    "Run decode on sequence of frames"
  );
}


}
}
#endif
