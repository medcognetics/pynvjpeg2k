#ifndef PYNVJPEG2K_ENCODE
#define PYNVJPEG2K_ENCODE
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
#include "nvjpeg2k_encode.h"


namespace py = pybind11;

namespace pynvjpeg {
namespace jpeg2k {


py::dict getImageInfo(
    const char* buffer, 
    const size_t inBufSize
) {
  // Initialize stream and handle
  nvjpeg2kHandle_t handle;
  nvjpeg2kStream_t stream;
  CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kCreateSimple(&handle));
  CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kStreamCreate(&stream));

  // Parse stream and get image info
  nvjpeg2kImageInfo_t imageInfo;
  CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kStreamParse(handle, (unsigned char*)buffer, inBufSize, 0, 0, stream));
  CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kStreamGetImageInfo(stream, &imageInfo));
  
  // Assign image info to dictionary
  py::dict result;
  result["width"] = imageInfo.image_width;
  result["height"] = imageInfo.image_height;
  result["tile_height"] = imageInfo.tile_height;
  result["tile_width"] = imageInfo.tile_width;
  result["num_tiles_x"] = imageInfo.num_tiles_x;
  result["num_tiles_y"] = imageInfo.num_tiles_y;
  result["num_components"] = imageInfo.num_components;

  // Read component info
  std::vector<nvjpeg2kImageComponentInfo_t> imageComponentInfo;
  imageComponentInfo.resize(imageInfo.num_components);
  for (uint32_t c = 0; c < imageInfo.num_components; c++)
  {
      CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kStreamGetImageComponentInfo(stream, &imageComponentInfo[c], c));
  }

  // Assign component info to dictionary
  py::list componentInfoResult;
  for (auto comp : imageComponentInfo) {
    py::dict infoThisComp;
    infoThisComp["width"] = comp.component_width;
    infoThisComp["height"] = comp.component_height;
    infoThisComp["precision"] = comp.precision;
    infoThisComp["sign"] = comp.sgn;
    componentInfoResult.append(infoThisComp);
  }
  result["component_info"] = componentInfoResult;

  // Cleanup and return
  CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kDestroy(handle));
  CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kStreamDestroy(stream));
  return result;
}


nvjpeg2kEncodeConfig_t getEncodeConfig(
  nvjpeg2kImageComponentInfo_t *imageComponentInfo,
  size_t rows,
  size_t cols,
  size_t codeBlockH,
  size_t codeBlockW,
  size_t numResolutions
) {
  // Create component info
  imageComponentInfo->component_width = cols;
  imageComponentInfo->component_height = rows;
  imageComponentInfo->precision = 16;
  imageComponentInfo->sgn = 0;

  // Create encode config
  // NOTE: pydicom expects NVJPEG2K_STREAM_J2K
  nvjpeg2kEncodeConfig_t encodeConfig;
  memset(&encodeConfig, 0, sizeof(encodeConfig));
  encodeConfig.stream_type      =  NVJPEG2K_STREAM_J2K; 
  encodeConfig.color_space      =  NVJPEG2K_COLORSPACE_GRAY; 
  encodeConfig.image_width      =  cols;
  encodeConfig.image_height     =  rows;
  encodeConfig.num_components   =  NUM_COMPONENTS;
  encodeConfig.image_comp_info  =  imageComponentInfo;
  encodeConfig.code_block_w     =  64;
  encodeConfig.code_block_h     =  64;
  encodeConfig.irreversible     =  0;
  encodeConfig.mct_mode         =  0;
  encodeConfig.prog_order       =  NVJPEG2K_LRCP;
  encodeConfig.num_resolutions  =  6;

  return encodeConfig;
}


/*
std::vector<unsigned char> encode2(
    const char* buffer, 
    const size_t inBufSize,
    size_t rows,
    size_t cols,
) {
  nvjpeg2kEncoder_t enc_handle;
  nvjpeg2kEncodeState_t enc_state;
  nvjpeg2kEncodeParams_t enc_params;
  nvjpeg2kEncoderCreateSimple(&enc_handle);
  nvjpeg2kEncodeStateCreate(&enc_state);
  nvjpeg2kEncodeParamsCreate(&enc_params);

  nvjpeg2kImageComponentInfo_t image_comp_info[NUM_COMPONENTS];

  unsigned char *pixel_data[NUM_COMPONENTS];
  size_t *pitch_in_bytes[NUM_COMPONENTS];


  int32_t image_width  =  cols;
  uint32_t image_height =  rows

  for (int c = 0; c < image_info.num_components; c++)
  {
      image_comp_info[c].component_width  = image_width;
      image_comp_info[c].component_height = image_height;
      image_comp_info[c].precision        = 8;
      image_comp_info[c].sgn              = 0;
  }

  nvjpeg2kImage_t input_image;
  for (int c = 0; c < NUM_COMPONENTS; c++)
  {
      cudaMallocPitch(&pixel_data[c], &pitch_in_bytes[c], image_comp_info[c].comp_width, image_comp_info[c].comp_height);
      // cudaMallocPitch is used to let cuda deterimine the pitch. cudaMalloc can be used if required.
  }

  // Copy the image to the device buffers.

  input_image.pixel_data = pixel_data;
  input_image.pixel_type = NVJPEG2K_UINT8;
  input_image.pitch_in_bytes = pitch_in_bytes;
}
*/


std::vector<EncodedFrame> encode(
    const char* buffer, 
    const size_t inBufSize,
    size_t rows,
    size_t cols,
    size_t numFrames,
    size_t batchSize
) {
  batchSize = std::min(numFrames, batchSize);
  if (numFrames <= 0) {
    throw std::invalid_argument("numFrames must be >= 0");
  }
  if (batchSize < 1) {
    throw std::invalid_argument("batchSize must be >= 1");
  }
  if (numFrames % batchSize != 0) {
    throw std::invalid_argument("numFrames must be evenly divisble by batchSize");
  }

  // init all pipeline stages
  const size_t PIPELINE_STAGES = std::min(numFrames, (size_t)10);
  assert(PIPELINE_STAGES > 0);
  EncodeParams_t stageParams[PIPELINE_STAGES];
  nvjpeg2kEncoder_t handle;
  CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncoderCreateSimple(&handle));
  for (size_t p=0; p < PIPELINE_STAGES; p++) {
    stageParams[p].handle = &handle;
    CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeStateCreate(handle, &stageParams[p].encodeState));
    CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeParamsCreate(&stageParams[p].params));
    cudaStreamCreateWithFlags(&stageParams[p].cudaStream, cudaStreamNonBlocking);
  }

  // create config for desired encoding output
  nvjpeg2kImageComponentInfo_t imageComponentInfo;
  nvjpeg2kEncodeConfig_t encodeConfig = getEncodeConfig(&imageComponentInfo, rows, cols);
  std::cout << "Rows: " << encodeConfig.image_height << " Cols " << encodeConfig.image_width << std::endl;

  // Allocate GPU memory to hold frames for encode
  // Allocated size will be 2D array of size (batchSize * Rows x Pitch)
  unsigned char* devBuffer;
  size_t pitchInBytes;
  CHECK_CUDA_NO_RETURN(deviceMalloc<uint16_t>(&devBuffer, &pitchInBytes, batchSize * rows, cols));
  std::cout << "Allocated with pitch " << pitchInBytes << std::endl;

  // Loop over frames to be encoded
  std::vector<EncodedFrame> output;
  for (size_t frameIndex = 0; frameIndex < numFrames; frameIndex++) {
    // Get encode params for the pipeline stage we want to use
    const size_t stage = frameIndex % PIPELINE_STAGES;
    EncodeParams_t *params = &stageParams[stage];

    // Seek hostBuffer to the start of this frame
    uint16_t* hostBufferThisFrame = seekToFrameNumber<uint16_t>((uint16_t*)buffer, cols, rows, frameIndex);
    std::cout << "Host buffer " << (void*)hostBufferThisFrame << " from start " << (void*)buffer << std::endl;

    // Seek devBuffer to the index of the frame we are encoding within the batch
    unsigned char* devBufferThisFrame = seekToFrameNumber<unsigned char>(devBuffer, pitchInBytes, rows, frameIndex % batchSize);
    std::cout << "Dev buffer " << (void*)devBufferThisFrame << " from start " << (void*)devBuffer << std::endl;

    // Ensure the previous stage has finished
    if(frameIndex >= PIPELINE_STAGES) {
      CHECK_CUDA_NO_RETURN(cudaStreamSynchronize(params->cudaStream));
    }

    // Additional config
    nvjpeg2kImage_t inputImage;
    inputImage.pixel_data = (void**)&devBufferThisFrame;
    inputImage.pixel_type = NVJPEG2K_UINT16;
    inputImage.pitch_in_bytes = &pitchInBytes;
    inputImage.num_components = NUM_COMPONENTS;
    CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeParamsSetEncodeConfig(params->params, &encodeConfig));

    // Copy frame to device
    CHECK_CUDA_NO_RETURN(hostToDeviceCopy<uint16_t>(devBufferThisFrame, pitchInBytes, hostBufferThisFrame, rows, cols, &params->cudaStream));

    // Run encode
    CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncode(handle, params->encodeState, params->params, &inputImage, params->cudaStream));

    // Check size of encoded stream and allocate a buffer
    size_t compressed_size;
    CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeRetrieveBitstream(handle, params->encodeState, NULL, &compressed_size, params->cudaStream));
    EncodedFrame encodedFrame = EncodedFrame(compressed_size);

    // Get the encoded data in host memory
    CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeRetrieveBitstream(handle, params->encodeState, encodedFrame.data(), &compressed_size, params->cudaStream));
    output.push_back(encodedFrame);

  }

  // free all resources
  CHECK_CUDA_NO_RETURN(cudaDeviceSynchronize());
  CHECK_CUDA_NO_RETURN(cudaFree(devBuffer));
  CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncoderDestroy(handle));
  for (size_t p=0; p < PIPELINE_STAGES; p++) {
    CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeStateDestroy(stageParams[p].encodeState));
    CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeParamsDestroy(stageParams[p].params));
    CHECK_CUDA_NO_RETURN(cudaStreamDestroy(stageParams[p].cudaStream));
  }

  return output;
}


std::vector<py::bytes> encode_frames(
    py::array_t<uint16_t> array,
    size_t batchSize
) {
  if (array.ndim() != 3) {
    throw std::invalid_argument("`array` must have 3 dimensions (DxHxW)");
  }

  // Get raw data and size in bytes
  const char* buffer = (char*)array.data();
  size_t bufferSize = array.size() * sizeof(uint16_t);

  // Get dimensions
  const size_t numFrames = array.shape(0);
  const size_t rows = array.shape(1);
  const size_t cols = array.shape(2);

  // Encode all frames
  std::vector<EncodedFrame> result = encode(
      buffer,
      bufferSize,
      rows,
      cols,
      numFrames,
      batchSize
  );

  std::vector<py::bytes> resultBytes;
  for(auto frame: result) {
    std::cout << "OUT SIZE " << frame.size() << std::endl;
    resultBytes.push_back(py::bytes(reinterpret_cast<const char*>(frame.data()), frame.size()));
    //for(int i = 0; i < 100; i++) {
    //  std::cout << std::hex << (int)frame.at(i);
    //}
  }

  return resultBytes;
}


void pybind_init_enc(py::module &m) {
  m.def(
    "encode_jpeg2k", 
    &encode_frames,
    "Run decode on frames"
  );
}

}
}
#endif
