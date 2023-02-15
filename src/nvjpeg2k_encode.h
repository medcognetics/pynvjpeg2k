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

namespace pynvjpeg {
namespace jpeg2k{

struct EncodeParams_t
{
    nvjpeg2kEncoder_t *handle;
    nvjpeg2kEncodeState_t encodeState;
    nvjpeg2kEncodeParams_t params;
    cudaStream_t cudaStream;
};

typedef std::vector<unsigned char> EncodedFrame;

py::dict getImageInfo(const char* buffer,  const size_t inBufSize);

nvjpeg2kEncodeConfig_t getEncodeConfig(
  nvjpeg2kImageComponentInfo_t *imageComponentInfo,
  size_t rows,
  size_t cols,
  size_t codeBlockH = 64,
  size_t codeBlockW = 64,
  size_t numResolutions = 6
); 


std::vector<EncodedFrame> encode(
    const char* buffer, 
    const size_t inBufSize,
    size_t rows,
    size_t cols,
    size_t numFrames,
    size_t batchSize = 4
);


std::vector<py::bytes> encode_frames(
    py::array_t<uint16_t> array,
    size_t batchSize = 4
); 


void pybind_init_enc(py::module &m);

}
}
