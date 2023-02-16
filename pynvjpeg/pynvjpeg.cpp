#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>
#include <nvjpeg.h>

#include "dicom.h"
#include "nvjpeg2k_encode.h"
#include "nvjpeg2k_decode.h"

namespace pynvjpeg {

PYBIND11_MODULE(pynvjpeg, m) {
  m.doc() = "Python Bindings for nvjpeg2k";
  pynvjpeg::jpeg2k::pybind_init_enc(m);
  pynvjpeg::jpeg2k::pybind_init_dec(m);
  dicom::pybind_init(m);
};

}
