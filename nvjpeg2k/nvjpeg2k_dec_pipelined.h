/*
 * Copyright (c) 2020 - 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include <string.h> // strcmpi

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <windows.h>
#include <filesystem>
const std::string separator = "\\";
namespace fs = std::filesystem;
#else
#include <sys/time.h> // timings
#include <experimental/filesystem>
const std::string separator = "/";
namespace fs = std::experimental::filesystem::v1;
#endif

#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

#define CHECK_CUDA(call)                                                                                          \
    {                                                                                                             \
        cudaError_t _e = (call);                                                                                  \
        if (_e != cudaSuccess)                                                                                    \
        {                                                                                                         \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                                     \
        }                                                                                                         \
    }

#define CHECK_NVJPEG2K(call)                                                                                \
    {                                                                                                       \
        nvjpeg2kStatus_t _e = (call);                                                                       \
        if (_e != NVJPEG2K_STATUS_SUCCESS)                                                                  \
        {                                                                                                   \
            std::cout << "NVJPEG failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                            \
        }                                                                                                   \
    }

constexpr int PIPELINE_STAGES = 10;
constexpr int NUM_COMPONENTS = 4;

//#define USE8BITOUTPUT

#ifdef USE8BITOUTPUT
constexpr int MAX_PRECISION = 8;
typedef struct
{
    uint16_t num_comps;
    unsigned char *component[NUM_COMPONENTS];
    size_t    pitch_in_bytes[NUM_COMPONENTS];
} nvjpeg2ksample_img;

#else
constexpr int MAX_PRECISION = 16;
typedef struct
{
    uint16_t num_comps;
    unsigned short *component[NUM_COMPONENTS];
    size_t    pitch_in_bytes[NUM_COMPONENTS];
} nvjpeg2ksample_img;
#endif




typedef struct
{
    size_t    comp_sz[NUM_COMPONENTS];
} nvjpeg2ksample_img_sz;

int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }

int dev_free(void *p) { return (int)cudaFree(p); }

int host_malloc(void **p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }

int host_free(void *p) { return (int)cudaFreeHost(p); }

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char>> FileData;

struct decode_params_t
{
    std::string input_dir;
    int batch_size;
    int total_images;
    int dev;
    int warmup;

    nvjpeg2kDecodeState_t nvjpeg2k_decode_states[PIPELINE_STAGES];
    nvjpeg2kHandle_t nvjpeg2k_handle;
    cudaStream_t stream[PIPELINE_STAGES];
    std::vector<nvjpeg2kStream_t> jpeg2k_streams;
    bool verbose;
    bool write_decoded;
    std::string output_dir;
};



int read_next_batch(FileNames &image_names, int batch_size,
                    FileNames::iterator &cur_iter, FileData &raw_data,
                    std::vector<size_t> &raw_len, FileNames &current_names, bool verbose)
{
    int counter = 0;

    while (counter < batch_size)
    {
        if (cur_iter == image_names.end())
        {
            if(verbose)
            {
                std::cerr << "Image list is too short to fill the batch, adding files "
                         "from the beginning of the image list"
                         << std::endl;
            }
            cur_iter = image_names.begin();
        }

        if (image_names.size() == 0)
        {
            std::cerr << "No valid images left in the input list, exit" << std::endl;
            return EXIT_FAILURE;
        }

        // Read an image from disk.
        std::ifstream input(cur_iter->c_str(),
                            std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open()))
        {
            std::cerr << "Cannot open image: " << *cur_iter
                      << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }

        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (raw_data[counter].size() < static_cast<size_t>(file_size))
        {
            raw_data[counter].resize(file_size);
        }
        if (!input.read(raw_data[counter].data(), file_size))
        {
            std::cerr << "Cannot read from file: " << *cur_iter
                      << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }
        raw_len[counter] = file_size;

        current_names[counter] = *cur_iter;

        counter++;
        cur_iter++;
    }
    return EXIT_SUCCESS;
}

double Wtime(void)
{
#if defined(_WIN32)
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer)
    {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer)
    {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else
    {
        return (double)GetTickCount() / 1000.0;
    }
#else
    struct timespec tp;
    int rv = clock_gettime(CLOCK_MONOTONIC, &tp);

    if (rv)
        return 0;

    return tp.tv_nsec / 1.0E+9 + (double)tp.tv_sec;

#endif
}
// *****************************************************************************
// reading input directory to file list
// -----------------------------------------------------------------------------
int readInput(const std::string &sInputPath, std::vector<std::string> &filelist)
{
    
    if( fs::is_regular_file(sInputPath))
    {
        filelist.push_back(sInputPath);
    }
    else if (fs::is_directory(sInputPath))
    { 
        fs::recursive_directory_iterator iter(sInputPath);
        for(auto& p: iter)
        {
           if( fs::is_regular_file(p))
           {
                filelist.push_back(p.path().string());
           }
        }
        std::sort(filelist.begin(), filelist.end());

        /* Print sorted file list to stderr
        for (auto f : filelist) {
              std::cerr << f << std::endl;
        }
        */

    }
    else
    {
        std::cout<<"unable to open input"<<std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// *****************************************************************************
// check for inputDirExists
// -----------------------------------------------------------------------------
int inputDirExists(const char *pathname)
{
    struct stat info;
    if (stat(pathname, &info) != 0)
    {
        return 0; // Directory does not exists
    }
    else if (info.st_mode & S_IFDIR)
    {
        // is a directory
        return 1;
    }
    else
    {
        // is not a directory
        return 0;
    }
}

// *****************************************************************************
// check for getInputDir
// -----------------------------------------------------------------------------
int getInputDir(std::string &input_dir, const char *executable_path)
{
    int found = 0;
    if (executable_path != 0)
    {
        std::string executable_name = std::string(executable_path);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        // Windows path delimiter
        size_t delimiter_pos = executable_name.find_last_of('\\');
        executable_name.erase(0, delimiter_pos + 1);

        if (executable_name.rfind(".exe") != std::string::npos)
        {
            // we strip .exe, only if the .exe is found
            executable_name.resize(executable_name.size() - 4);
        }
#else
        // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of('/');
        executable_name.erase(0, delimiter_pos + 1);
#endif

        // Search in default paths for input images.
        std::string pathname = "";
        const char *searchPath[] = {
            "./images"};

        for (unsigned int i = 0; i < sizeof(searchPath) / sizeof(char *); ++i)
        {
            std::string pathname(searchPath[i]);
            size_t executable_name_pos = pathname.find("<executable_name>");

            // If there is executable_name variable in the searchPath
            // replace it with the value
            if (executable_name_pos != std::string::npos)
            {
                pathname.replace(executable_name_pos, strlen("<executable_name>"),
                                 executable_name);
            }

            if (inputDirExists(pathname.c_str()))
            {
                input_dir = pathname + "/";
                found = 1;
                break;
            }
        }
    }
    return found;
}


// *****************************************************************************
// parse parameters
// -----------------------------------------------------------------------------
int findParamIndex(const char **argv, int argc, const char *parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++)
    {
        if (strncmp(argv[i], parm, 100) == 0)
        {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1)
    {
        return index;
    }
    else
    {
        std::cout << "Error, parameter " << parm
                  << " has been specified more than once, exiting\n"
                  << std::endl;
        return -1;
    }

    return -1;
}


// write PGM, input - single channel, device
template <typename D>
int writeRaw(const D *pSrc, size_t nSrcStep, int nWidth, int nHeight, uint8_t precision)
{
    // allocate host buffer and pointer hpSrc to receive data from GPU
    const size_t bufSize = nHeight * (nSrcStep / sizeof(D));
    std::vector<D> img(bufSize);
    D *hpSrc = img.data();

    // GPU -> host copy
    CHECK_CUDA(cudaMemcpy2D(hpSrc, nSrcStep, pSrc, nSrcStep, nWidth * sizeof(D), nHeight, cudaMemcpyDeviceToHost));

    std::cout.write(reinterpret_cast<char*>(hpSrc), bufSize * 2);
    return 0;
}
