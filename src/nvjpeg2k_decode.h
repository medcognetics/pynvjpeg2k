#include <assert.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvjpeg.h>
#include <nvjpeg2k.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pynvjpeg
{
namespace jpeg2k
{

struct DecodeParams_t
{
	nvjpeg2kHandle_t *handle;
	nvjpeg2kDecodeState_t decodeState;
	cudaStream_t cudaStream;
	nvjpeg2kStream_t jpegStream;
	size_t rows;
	size_t cols;
};

int _decode_frames(std::vector<const char *> frameBuffers,
	std::vector<size_t> bufferSizes,
	const std::size_t rows,
	const std::size_t cols,
	uint16_t *outBuffer,
	size_t batchSize = 4);

py::dict getImageInfo(const char *buffer, const size_t size);

int checkStream(const char *buffer, const size_t size);

bool isValid(const char *buffer, const size_t size);

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
py::array_t<uint16_t> decode(const char *buffer,
	const size_t inBufSize,
	const std::size_t rows,
	const std::size_t cols);

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
py::array_t<uint16_t> decode_frames(const char *buffer,
	size_t size,
	const std::size_t rows,
	const std::size_t cols,
	const int batchSize = 4);

/*
 * Run batched decode on multiple frames with offsets
 *
 * Args:
 *    buffer - Data buffer of multiple frames
 *    bufferSizes - Size of each frame buffer
 *    rows - Rows in output image
 *    cols - Cols in output image
 *    batchSize - Batch size for decoding
 *
 * Returns:
 *  3D array of decoded pixel data
 */
py::array_t<uint16_t> decode_framelist(const char *buffer,
	std::vector<size_t> bufferSizes,
	const size_t rows,
	const size_t cols,
	const int batchSize);

void pybind_init_dec(py::module &m);

} // namespace jpeg2k
} // namespace pynvjpeg
