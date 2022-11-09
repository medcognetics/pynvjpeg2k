#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>


namespace py = pybind11;

namespace dicom {


template <typename D>
D read(const char* buffer, size_t offset = 0);
uint16_t readLE(const char* buffer, size_t offset = 0);
uint32_t readLE32(const char* buffer, size_t offset = 0);

bool hasBot(const char* buffer);

typedef struct
{
    size_t offset;
    size_t length;
} FrameInfo_t;

std::vector<FrameInfo_t> getFrameInfo(const char* buffer, size_t size);

size_t getNumberOfFragments(const char* buffer, size_t size);

void pybind_init(py::module &m);

}
