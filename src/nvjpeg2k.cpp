#include <assert.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvjpeg.h>
#include <nvjpeg2k.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NUM_COMPONENTS 1

#define CHECK_CUDA(call)                                                                                               \
	{                                                                                                                  \
		cudaError_t _e = (call);                                                                                       \
		if (_e != cudaSuccess)                                                                                         \
		{                                                                                                              \
			std::cerr << "CUDA Runtime failure: '#" << _e << "' " << cudaGetErrorString(_e) << " at " << __FILE__      \
					  << ":" << __LINE__ << std::endl;                                                                 \
			return EXIT_FAILURE;                                                                                       \
		}                                                                                                              \
	}

#define CHECK_NVJPEG2K(call)                                                                                           \
	{                                                                                                                  \
		nvjpeg2kStatus_t _e = (call);                                                                                  \
		if (_e != NVJPEG2K_STATUS_SUCCESS)                                                                             \
		{                                                                                                              \
			std::cerr << "NVJPEG failure: '#" << _e << "' " << nvjpegGetErrorString(_e) << " at " << __FILE__ << ":"   \
					  << __LINE__ << std::endl;                                                                        \
			return EXIT_FAILURE;                                                                                       \
		}                                                                                                              \
	}

#define CHECK_CUDA_NO_RETURN(call)                                                                                     \
	{                                                                                                                  \
		cudaError_t _e = (call);                                                                                       \
		if (_e != cudaSuccess)                                                                                         \
		{                                                                                                              \
			std::stringstream msg;                                                                                     \
			msg << "CUDA Runtime failure: '#" << _e << "' " << cudaGetErrorString(_e) << " at " << __FILE__ << ":"     \
				<< __LINE__ << std::endl;                                                                              \
			throw std::runtime_error(msg.str());                                                                       \
		}                                                                                                              \
	}

#define CHECK_NVJPEG2K_NO_RETURN(call)                                                                                 \
	{                                                                                                                  \
		nvjpeg2kStatus_t _e = (call);                                                                                  \
		if (_e != NVJPEG2K_STATUS_SUCCESS)                                                                             \
		{                                                                                                              \
			std::stringstream msg;                                                                                     \
			msg << "NVJPEG failure: '#" << _e << "' " << nvjpegGetErrorString(_e) << " at " << __FILE__ << ":"         \
				<< __LINE__ << std::endl;                                                                              \
			throw std::runtime_error(msg.str());                                                                       \
		}                                                                                                              \
	}

namespace py = pybind11;

namespace pynvjpeg2k
{

std::string nvjpegGetErrorString(nvjpeg2kStatus_t e)
{
	switch (e)
	{
	case NVJPEG2K_STATUS_SUCCESS:
		return std::string("NVJPEG2K_STATUS_SUCCESS");
	case NVJPEG2K_STATUS_NOT_INITIALIZED:
		return std::string("NVJPEG2K_STATUS_NOT_INITIALIZED");
	case NVJPEG2K_STATUS_INVALID_PARAMETER:
		return std::string("NVJPEG2K_STATUS_INVALID_PARAMETER");
	case NVJPEG2K_STATUS_BAD_JPEG:
		return std::string("NVJPEG2K_STATUS_BAD_JPEG");
	case NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED:
		return std::string("NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED");
	case NVJPEG2K_STATUS_ALLOCATOR_FAILURE:
		return std::string("NVJPEG2K_STATUS_ALLOCATOR_FAILURE");
	case NVJPEG2K_STATUS_EXECUTION_FAILED:
		return std::string("NVJPEG2K_STATUS_EXECUTION_FAILED");
	case NVJPEG2K_STATUS_ARCH_MISMATCH:
		return std::string("NVJPEG2K_STATUS_ARCH_MISMATCH");
	case NVJPEG2K_STATUS_INTERNAL_ERROR:
		return std::string("NVJPEG2K_STATUS_INTERNAL_ERROR");
	case NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
		return std::string("NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED");
	}
	throw std::invalid_argument("Invalid nvjpeg2kStatus_t");
}

/*
	NVJPEG2K_STATUS_NOT_INITIALIZED               = 1,
	NVJPEG2K_STATUS_INVALID_PARAMETER             = 2,
	NVJPEG2K_STATUS_BAD_JPEG                      = 3,
	NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED            = 4,
	NVJPEG2K_STATUS_ALLOCATOR_FAILURE             = 5,
	NVJPEG2K_STATUS_EXECUTION_FAILED              = 6,
	NVJPEG2K_STATUS_ARCH_MISMATCH                 = 7,
	NVJPEG2K_STATUS_INTERNAL_ERROR                = 8,
	NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED  = 9,
*/

template <typename D> D read(const char *buffer, size_t offset = 0)
{
	D result = *reinterpret_cast<const D *>(buffer + offset);
	return result;
}

uint16_t readLE(const char *buffer, size_t offset = 0)
{
	uint16_t result = read<uint16_t>(buffer, offset);
	return __builtin_bswap16(result);
}

uint32_t readLE32(const char *buffer, size_t offset = 0)
{
	uint16_t r1 = read<uint16_t>(buffer, offset);
	uint16_t r2 = read<uint16_t>(buffer, offset + sizeof(uint16_t));
	return ((uint32_t)r1 << 16) | r2;
}

bool hasBot(const char *buffer)
{
	constexpr uint32_t BOT_TAG = 0xfffee000;
	uint32_t tag = readLE32(buffer, 0);
	uint32_t size = readLE32(buffer, 4);
	return (tag == BOT_TAG) && size;
}

typedef struct
{
	size_t offset;
	size_t length;
} FrameInfo_t;

std::vector<FrameInfo_t> getFrameInfo(const char *buffer, size_t size)
{
	constexpr uint32_t LENGTH_PREAMBLE = 0xfffee000;
	constexpr uint32_t DELIM = 0xfffee0dd;

	std::vector<FrameInfo_t> result;
	for (size_t offset = sizeof(uint32_t) * 2; offset < size;)
	{
		uint32_t tag = readLE32(buffer, offset);

		// If a length is given, advance that far ahead
		if (tag == LENGTH_PREAMBLE)
		{
			uint32_t length = read<uint32_t>(buffer, offset + 4);
			FrameInfo_t frameInfo = {offset + 8, length};
			result.push_back(frameInfo);
			offset += length + 8;
		}
		else if (tag == DELIM)
		{
			break;
		}
		else
		{
			throw std::invalid_argument("Unexpected tag");
		}
	}
	return result;
}

size_t getNumberOfFragments(const char *buffer, size_t size)
{
	return getFrameInfo(buffer, size).size();
}

struct DecodeParams_t
{
	nvjpeg2kHandle_t *handle;
	nvjpeg2kDecodeState_t decodeState;
	cudaStream_t cudaStream;
	nvjpeg2kStream_t jpegStream;
	size_t rows;
	size_t cols;
};

struct EncodeParams_t
{
	nvjpeg2kEncoder_t *handle;
	nvjpeg2kEncodeState_t encodeState;
	nvjpeg2kEncodeParams_t params;
	cudaStream_t cudaStream;
};

template <typename D>
cudaError_t deviceToHostCopy(unsigned char *deviceBuf,
	const size_t deviceBufPitchInBytes,
	D *hostBuf,
	const size_t rows,
	const size_t cols,
	cudaStream_t *stream = NULL)
{
	const size_t hostBufPitchInBytes = cols * sizeof(D);
	const size_t copySize = cols * sizeof(D);
	if (stream == NULL)
	{
		return cudaMemcpy2D(hostBuf,
			hostBufPitchInBytes,
			(D *)deviceBuf,
			deviceBufPitchInBytes,
			copySize,
			rows,
			cudaMemcpyDeviceToHost);
	}
	else
	{
		return cudaMemcpy2DAsync(hostBuf,
			hostBufPitchInBytes,
			(D *)deviceBuf,
			deviceBufPitchInBytes,
			copySize,
			rows,
			cudaMemcpyDeviceToHost,
			*stream);
	}
}

template <typename D>
cudaError_t hostToDeviceCopy(unsigned char *deviceBuf,
	const size_t deviceBufPitchInBytes,
	D *hostBuf,
	const size_t rows,
	const size_t cols,
	cudaStream_t *stream = NULL)
{
	const size_t hostBufPitchInBytes = cols * sizeof(D);
	const size_t copySize = cols * sizeof(D);
	if (stream == NULL)
	{
		return cudaMemcpy2D((D *)deviceBuf,
			deviceBufPitchInBytes,
			hostBuf,
			hostBufPitchInBytes,
			copySize,
			rows,
			cudaMemcpyHostToDevice);
	}
	else
	{
		return cudaMemcpy2DAsync((D *)deviceBuf,
			deviceBufPitchInBytes,
			hostBuf,
			hostBufPitchInBytes,
			copySize,
			rows,
			cudaMemcpyHostToDevice,
			*stream);
	}
}

template <typename D>
cudaError_t deviceMalloc(unsigned char **buffer,
	size_t *pitchInBytes,
	const size_t rows,
	const size_t cols,
	const size_t batchSize = 1)
{
	return cudaMallocPitch((void **)buffer, pitchInBytes, cols * sizeof(D), batchSize * rows);
}

/*
 * Seeks to a given frame index from the start of the buffer
 */
template <typename D>
inline D *seekToFrameNumber(D *ptr, const size_t pitchInBytes, const size_t rows, const size_t targetFrameIndex)
{
	size_t offset = (rows * pitchInBytes) * targetFrameIndex;
	return ptr + offset;
}

int _async_frame_decode(const unsigned char *srcBuffer,
	const std::size_t srcBufSize,
	DecodeParams_t *params,
	unsigned char *devBuffer,
	size_t pitchInBytes,
	cudaStream_t *stream = NULL)
{
	// build struct expected by nvjpeg
	nvjpeg2kImage_t output_image;
	output_image.pixel_data = (void **)&devBuffer;
	output_image.pixel_type = NVJPEG2K_UINT16;
	output_image.pitch_in_bytes = &pitchInBytes;
	output_image.num_components = 1;

	// Run decode
	if (stream == NULL)
	{
		CHECK_NVJPEG2K(nvjpeg2kDecode(*params->handle, params->decodeState, params->jpegStream, &output_image, 0));
	}
	else
	{
		CHECK_NVJPEG2K(
			nvjpeg2kDecode(*params->handle, params->decodeState, params->jpegStream, &output_image, *stream));
	}

	return EXIT_SUCCESS;
}

int _decode_frames(std::vector<const char *> frameBuffers,
	std::vector<size_t> bufferSizes,
	const std::size_t rows,
	const std::size_t cols,
	uint16_t *outBuffer,
	size_t batchSize = 4)
{
	const size_t numFrames = frameBuffers.size();
	batchSize = std::min(numFrames, batchSize);
	if (frameBuffers.size() != bufferSizes.size())
	{
		throw std::invalid_argument("frameBuffers length should match bufferSizes length");
	}
	if (numFrames <= 0)
	{
		throw std::invalid_argument("frameBuffers must have nonzero length");
	}
	if (batchSize < 1)
	{
		throw std::invalid_argument("batchSize must be >= 1");
	}
	if (numFrames % batchSize != 0)
	{
		throw std::invalid_argument("numFrames must be evenly divisble by batchSize");
	}

	// init all pipeline stages
	const size_t PIPELINE_STAGES = std::min(numFrames, (size_t)10);
	assert(PIPELINE_STAGES > 0);
	DecodeParams_t stageParams[PIPELINE_STAGES];
	nvjpeg2kHandle_t handle;
	nvjpeg2kCreateSimple(&handle);
	for (size_t p = 0; p < PIPELINE_STAGES; p++)
	{
		stageParams[p].rows = rows;
		stageParams[p].cols = cols;
		stageParams[p].handle = &handle;
		nvjpeg2kDecodeStateCreate(*stageParams[p].handle, &stageParams[p].decodeState);
		nvjpeg2kStreamCreate(&stageParams[p].jpegStream);
		cudaStreamCreateWithFlags(&stageParams[p].cudaStream, cudaStreamNonBlocking);
	}

	// Allocate GPU memory to receive decode.
	// Allocated size will be 2D array of size (batchSize * Rows x Pitch)
	unsigned char *devBuffer;
	size_t pitchInBytes;
	CHECK_CUDA(deviceMalloc<uint16_t>(&devBuffer, &pitchInBytes, batchSize * rows, cols));

	// Loop over frames to be decoded
	int err = EXIT_SUCCESS;
	for (size_t frameIndex = 0; frameIndex < numFrames; frameIndex++)
	{
		const unsigned char *buffer = reinterpret_cast<const unsigned char *>(frameBuffers.at(frameIndex));
		size_t size = bufferSizes.at(frameIndex);
		const size_t stage = frameIndex % PIPELINE_STAGES;

		// Get decode params for the pipeline stage we want to use
		DecodeParams_t *params = &stageParams[stage];

		// Seek devBuffer to the index of the frame we are decoding within the batch
		unsigned char *devBufferThisFrame =
			seekToFrameNumber<unsigned char>(devBuffer, pitchInBytes, rows, frameIndex % batchSize);

		// Submit decode job
		CHECK_NVJPEG2K(nvjpeg2kStreamParse(*params->handle, buffer, size, 0, 0, params->jpegStream));
		err = _async_frame_decode(buffer, size, params, devBufferThisFrame, pitchInBytes, &params->cudaStream);

		if (err)
		{
			std::cerr << "Error decoding frame at index " << frameIndex << std::endl;
			break;
		}

		// Once a full batch has been submitted, read that batch from GPU memory -> Host memory
		if (frameIndex % batchSize == batchSize - 1)
		{
			// Seek pointer backwards to the start of last batch in host memory
			uint16_t *outBufferLastBatch =
				seekToFrameNumber<uint16_t>(outBuffer, cols, rows, frameIndex / batchSize * batchSize);

			// Perform the copy
			CHECK_CUDA(deviceToHostCopy<uint16_t>(devBuffer,
				pitchInBytes,
				outBufferLastBatch,
				params->rows * batchSize,
				params->cols,
				&params->cudaStream));
			CHECK_CUDA(cudaDeviceSynchronize());
		}
	}

	// free all resources
	CHECK_CUDA(cudaFree(devBuffer));
	CHECK_NVJPEG2K(nvjpeg2kDestroy(handle));
	for (size_t p = 0; p < PIPELINE_STAGES; p++)
	{
		CHECK_NVJPEG2K(nvjpeg2kStreamDestroy(stageParams[p].jpegStream));
		CHECK_NVJPEG2K(nvjpeg2kDecodeStateDestroy(stageParams[p].decodeState));
		CHECK_CUDA(cudaStreamDestroy(stageParams[p].cudaStream));
	}
	return err;
}

std::map<std::string, uint32_t> getImageInfo(const char *buffer, const size_t inBufSize)
{
	nvjpeg2kHandle_t handle;
	nvjpeg2kStream_t stream;
	nvjpeg2kCreateSimple(&handle);
	nvjpeg2kStreamCreate(&stream);

	nvjpeg2kImageInfo_t imageInfo;
	CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kStreamParse(handle, (unsigned char *)buffer, inBufSize, 0, 0, stream));
	CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kStreamGetImageInfo(stream, &imageInfo));

	std::map<std::string, uint32_t> result{{"width", imageInfo.image_width},
		{"height", imageInfo.image_height},
		{"tile_height", imageInfo.tile_height},
		{"tile_width", imageInfo.tile_width},
		{"num_tiles_x", imageInfo.num_tiles_x},
		{"num_tiles_y", imageInfo.num_tiles_y},
		{"num_components", imageInfo.num_components}};
	return result;
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
py::array_t<uint16_t> decode(const char *buffer, const size_t inBufSize, const size_t rows, const size_t cols)
{
	// Allocate array to receive decoded data
	py::array_t<uint16_t> outBuffer({rows, cols}, {cols * sizeof(uint16_t), sizeof(uint16_t)});

	// Wrap single frame in vector and run frame decode
	std::vector<const char *> frameBuffers({buffer});
	std::vector<size_t> bufferSizes({inBufSize});
	int err = _decode_frames(frameBuffers, bufferSizes, rows, cols, (uint16_t *)outBuffer.data());

	if (err)
	{
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
py::array_t<uint16_t> decode_frames(const char *buffer,
	size_t size,
	const size_t rows,
	const size_t cols,
	const int batchSize = 4)
{
	// Scan PixelData buffer and to find frame offsets and sizes
	std::vector<FrameInfo_t> frameInfo = getFrameInfo(buffer, size);
	std::vector<const char *> frameBuffers;
	std::vector<size_t> bufferSizes;
	for (auto x : frameInfo)
	{
		const char *frameBuffer = buffer + x.offset;
		frameBuffers.push_back(frameBuffer);
		bufferSizes.push_back(x.length);
	}
	const size_t numFrames = frameBuffers.size();

	// Allocate output array
	py::array_t<uint16_t> outBuffer({numFrames, rows, cols},
		{rows * cols * sizeof(uint16_t), cols * sizeof(uint16_t), sizeof(uint16_t)});

	// Run decode over frames
	int err = _decode_frames(frameBuffers, bufferSizes, rows, cols, (uint16_t *)outBuffer.data(), batchSize);

	if (err)
	{
		throw std::invalid_argument("error");
	}
	return outBuffer;
}

nvjpeg2kEncodeConfig_t getEncodeConfig(nvjpeg2kImageComponentInfo_t *imageComponentInfo,
	size_t rows,
	size_t cols,
	size_t codeBlockH = 64,
	size_t codeBlockW = 64,
	size_t numResolutions = 6)
{
	// Create component info
	imageComponentInfo->component_width = cols;
	imageComponentInfo->component_height = rows;
	imageComponentInfo->precision = 16;
	imageComponentInfo->sgn = 0;

	// Create encode config
	nvjpeg2kEncodeConfig_t encodeConfig;
	memset(&encodeConfig, 0, sizeof(encodeConfig));
	encodeConfig.stream_type = NVJPEG2K_STREAM_J2K;
	encodeConfig.color_space = NVJPEG2K_COLORSPACE_GRAY;
	encodeConfig.image_width = cols;
	encodeConfig.image_height = rows;
	encodeConfig.num_components = NUM_COMPONENTS;
	encodeConfig.image_comp_info = imageComponentInfo;
	encodeConfig.code_block_w = 64;
	encodeConfig.code_block_h = 64;
	encodeConfig.irreversible = 0;
	encodeConfig.mct_mode = 0;
	encodeConfig.prog_order = NVJPEG2K_LRCP;
	encodeConfig.num_resolutions = 6;

	return encodeConfig;
}

std::vector<unsigned char> encode(const char *buffer,
	const size_t inBufSize,
	size_t rows,
	size_t cols,
	size_t numFrames,
	size_t batchSize = 4)
{
	batchSize = std::min(numFrames, batchSize);
	if (numFrames <= 0)
	{
		throw std::invalid_argument("numFrames must be >= 0");
	}
	if (batchSize < 1)
	{
		throw std::invalid_argument("batchSize must be >= 1");
	}
	if (numFrames % batchSize != 0)
	{
		throw std::invalid_argument("numFrames must be evenly divisble by batchSize");
	}
	// const size_t frameSize = rows * cols * sizeof(uint16_t);
	std::cout << "In buffer " << inBufSize << std::endl;

	// init all pipeline stages
	const size_t PIPELINE_STAGES = std::min(numFrames, (size_t)10);
	assert(PIPELINE_STAGES > 0);
	EncodeParams_t stageParams[PIPELINE_STAGES];
	nvjpeg2kEncoder_t handle;
	CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncoderCreateSimple(&handle));
	for (size_t p = 0; p < PIPELINE_STAGES; p++)
	{
		std::cout << "SET" << std::endl;
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
	unsigned char *devBuffer;
	size_t pitchInBytes;
	CHECK_CUDA_NO_RETURN(deviceMalloc<uint16_t>(&devBuffer, &pitchInBytes, batchSize * rows, cols));
	std::cout << "Allocated with pitch " << pitchInBytes << std::endl;

	// Loop over frames to be encoded
	std::vector<unsigned char *> outBuffers(numFrames);
	std::vector<size_t> outBufSizes(numFrames);
	for (size_t frameIndex = 0; frameIndex < numFrames; frameIndex++)
	{
		// Get encode params for the pipeline stage we want to use
		const size_t stage = frameIndex % PIPELINE_STAGES;
		EncodeParams_t *params = &stageParams[stage];

		// Seek hostBuffer to the start of this frame
		uint16_t *hostBufferThisFrame = seekToFrameNumber<uint16_t>((uint16_t *)buffer, cols, rows, frameIndex);
		std::cout << "Host buffer " << (void *)hostBufferThisFrame << " from start " << (void *)buffer << std::endl;

		// Seek devBuffer to the index of the frame we are encoding within the batch
		unsigned char *devBufferThisFrame =
			seekToFrameNumber<unsigned char>(devBuffer, pitchInBytes, rows, frameIndex % batchSize);
		std::cout << "Dev buffer " << (void *)devBufferThisFrame << " from start " << (void *)devBuffer << std::endl;

		// Copy frame to device
		CHECK_CUDA_NO_RETURN(
			hostToDeviceCopy<uint16_t>(devBufferThisFrame, pitchInBytes, hostBufferThisFrame, rows, cols));
		CHECK_CUDA_NO_RETURN(cudaDeviceSynchronize());

		std::cout << "Start encode" << std::endl;

		// Additional config
		unsigned char *pixelData[NUM_COMPONENTS];
		pixelData[0] = devBufferThisFrame;

		nvjpeg2kImage_t inputImage;
		inputImage.pixel_data = (void **)&devBufferThisFrame;
		inputImage.pixel_type = NVJPEG2K_UINT16;
		inputImage.pitch_in_bytes = &pitchInBytes;
		inputImage.num_components = NUM_COMPONENTS;

		nvjpeg2kEncodeParams_t enc_params;
		nvjpeg2kEncodeState_t encState;

		CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeStateCreate(handle, &encState));
		CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeParamsCreate(&enc_params));

		CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeParamsSetEncodeConfig(enc_params, &encodeConfig));

		// Run encode
		CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncode(handle, encState, enc_params, &inputImage, NULL));

		// Check size of encoded stream and allocate a buffer
		size_t compressed_size;
		CHECK_NVJPEG2K_NO_RETURN(
			nvjpeg2kEncodeRetrieveBitstream(handle, params->encodeState, NULL, &compressed_size, NULL));
		unsigned char *compressed_data = new unsigned char[compressed_size];

		// Get the encoded data in host memory
		CHECK_NVJPEG2K_NO_RETURN(
			nvjpeg2kEncodeRetrieveBitstream(handle, params->encodeState, compressed_data, &compressed_size, NULL));

		cudaDeviceSynchronize();
		std::cout << "Compressed size " << compressed_size << std::endl;
		outBuffers.push_back(compressed_data);
		outBufSizes.push_back(compressed_size);

		// Once a full batch has been submitted, synchronize and wait for full encode to complete
		if (frameIndex % batchSize == batchSize - 1)
		{
			CHECK_CUDA_NO_RETURN(cudaDeviceSynchronize());
		}
	}

	// free all resources
	CHECK_CUDA_NO_RETURN(cudaFree(devBuffer));
	CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncoderDestroy(handle));
	for (size_t p = 0; p < PIPELINE_STAGES; p++)
	{
		CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeStateDestroy(stageParams[p].encodeState));
		CHECK_NVJPEG2K_NO_RETURN(nvjpeg2kEncodeParamsDestroy(stageParams[p].params));
		CHECK_CUDA_NO_RETURN(cudaStreamDestroy(stageParams[p].cudaStream));
	}

	// Compute total size of output bytestream
	size_t totalSize = 0;
	for (auto it : outBufSizes)
	{
		totalSize += it;
	}
	std::cout << "Total size " << totalSize << std::endl;

	// copy frames into contiguous byte stream
	std::vector<unsigned char> result(totalSize);
	size_t currentOffset = 0;
	for (size_t i = 0; i < numFrames; i++)
	{
		size_t sizeToCopy = outBufSizes.at(i);
		unsigned char *src = outBuffers.at(i);
		unsigned char *dest = result.data() + currentOffset;
		std::cout << "Copying " << sizeToCopy << std::endl;
		std::copy(dest, dest + sizeToCopy, src);
		currentOffset += sizeToCopy;
	}

	return result;
}

py::bytes encode_frames(py::array_t<uint16_t> array, size_t batchSize = 4)
{
	if (array.ndim() != 3)
	{
		throw std::invalid_argument("`array` must have 3 dimensions (DxHxW)");
	}

	// Get raw data and size in bytes
	const char *buffer = (char *)array.data();
	size_t bufferSize = array.size() * sizeof(uint16_t);

	// Get dimensions
	const size_t numFrames = array.shape(0);
	const size_t rows = array.shape(1);
	const size_t cols = array.shape(2);

	// Encode all frames
	std::vector<unsigned char> encodedFrames = encode(buffer, bufferSize, rows, cols, numFrames, batchSize);

	return py::bytes(buffer);
}

PYBIND11_MODULE(pynvjpeg2k, m)
{
	m.doc() = "Python Bindings for nvjpeg2k";

	m.def("get_image_info", &getImageInfo, "read");

	m.def("decode", &decode, "Run decode");
	m.def("decode_frames", &decode_frames, "Run decode on frames");
	m.def("encode", &encode_frames, "Run decode on frames");
	m.def("_has_bot", &hasBot, "Has BOT");
	m.def("_read_uint32_le", &readLE32, "read");
	m.def("_get_num_fragments", &getNumberOfFragments, "read");
}

} // namespace pynvjpeg2k
