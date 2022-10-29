# PyNVJPEG2K

PyNVJPEG2K is a work in progress Python library to accelerate decompression and compression of JPEG2000 encoded images.
Python bindings have not yet been implemented, but there is support for accelerated decoding using a standaclone C++ program.
Only decompression of single channel inputs has been implemented.

This library depends on `libnvjpeg2k`, which is provided only to developers in the Nvidia developer program. A copy
of `libnvjpeg2k` has been provided with this repository as PyNVJPEG2K is currently a private MedCognetics repository.

## Usage

1. Run `make build`
2. See build program under `nvjpeg2k/build/nvjpeg2k_dec_pipelined`

## Implementation Notes

[This](https://github.com/NVIDIA/CUDALibrarySamples.git) example repo served as the starting point for PyNVJPEG2K. In order to
quickly develop a proof of concept, the implementation provided by that repo was modified to facilitate exchange to/from Python
without directly implementing Python bindings.

The exchange is as follows:
1. One or more compressed images are written to temporary files in a named directory
2. C++ script is called using the named directory as a parameter
3. C++ script performs accelerated decompression of input images in batches
4. C++ script writes the resulting decompressed data to stdout
5. Calling Python script captures stdout and constructs a decompressed array

Currently the C++ script will return additional data if batch padding was needed. It is currently the responsibility of the calling
Python script to account for this additional data and truncate if needed. The data exchange implementation in C++ was rewritten, as the
existing implementation of writing to files or looping over `cout` calls was too slow.

## TODO
* Implement a real Python binding that exchanges data without temporary files
* Implement compression
