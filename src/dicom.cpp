#ifndef DICOM
#define DICOM

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>
#include <nvjpeg.h>

#include "dicom.h"

namespace py = pybind11;

namespace dicom {

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


template <typename D>
D read(const char* buffer, size_t offset) {
  D result = *reinterpret_cast<const D*>(buffer + offset);
  return result;
}


uint16_t readLE(const char* buffer, size_t offset) {
  uint16_t result = read<uint16_t>(buffer, offset);
  return __builtin_bswap16(result);
}


uint32_t readLE32(const char* buffer, size_t offset) {
  uint16_t r1 = read<uint16_t>(buffer, offset);
  uint16_t r2 = read<uint16_t>(buffer, offset + sizeof(uint16_t));
  return ((uint32_t)r1 << 16) | r2;
}


bool hasBot(const char* buffer) {
  constexpr uint32_t BOT_TAG = 0xfffee000;
  uint32_t tag = readLE32(buffer, 0);
  uint32_t size = readLE32(buffer, 4);
  return (tag == BOT_TAG) && size;
}


std::vector<FrameInfo_t> getFrameInfo(const char* buffer, size_t size) {
  constexpr uint32_t LENGTH_PREAMBLE = 0xfffee000;
  constexpr uint32_t DELIM = 0xfffee0dd;

  std::vector<FrameInfo_t> result;
  for(size_t offset = sizeof(uint32_t) * 2; offset < size;) {
    uint32_t tag = readLE32(buffer, offset);

    // If a length is given, advance that far ahead
    if (tag == LENGTH_PREAMBLE) {
      uint32_t length = read<uint32_t>(buffer, offset + 4);
      FrameInfo_t frameInfo = {offset + 8, length};
      result.push_back(frameInfo);
      offset += length + 8;
    }
    else if (tag == DELIM){
      break;
    }
    else {
      std::stringstream ss;
      ss << "Unexpected tag: " << std::hex << tag;
      throw std::runtime_error(ss.str());
    }
  }
  return result;
}


size_t getNumberOfFragments(const char* buffer, size_t size) {
  return getFrameInfo(buffer, size).size();
}


py::list getFramePositions(const char* buffer, size_t size) {
  std::vector<FrameInfo_t> frameInfo =  getFrameInfo(buffer, size);
  py::list result;
  for (auto& frame : frameInfo) {
    result.append(py::make_tuple(frame.offset, frame.length));
  }
  return result;
}


void pybind_init(py::module &m) {
  m.def(
    "_has_bot", 
    &hasBot,
    "Has BOT"
  );
  m.def(
    "_read_uint32_le", 
    &readLE32,
    "read"
  );
  m.def(
    "_get_num_fragments", 
    &getNumberOfFragments,
    "read"
  );
  m.def(
    "get_frame_offsets", 
    &getFramePositions,
    "Get the frame offsets"
  );
}

}
#endif
