#include <cmath>
#include <iostream>

#include <hip/hip_runtime.h>

#include "hipcheck.h"

// generate histogram counts
__global__ void histogram_gpu_counts(int *grid, int *gpuhistogram, size_t width, size_t height)
{
    // get exact location in array
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < width && y < height)
    {
        // uses atomic addition to ensure that counts are correct
        atomicAdd(&gpuhistogram[grid[y * width + x]], 1);
    }
}

// finalize colors
__global__ void histogram_gpu_finalize(int *grid, int *gpuhistogram, size_t width, size_t height, size_t total, double color_scaler)
{
    // get exact location in array
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < width && y < height)
    {
        // index to array
        size_t index = y * width + x;

        // color value
        int color = grid[index];

        // initialize hue
        int hue = 0;

        // compute hue based on cumulative histogram counts
        for (size_t i = 0; i <= color; i++)
        {
            hue += gpuhistogram[i];
        }

        // set final color
        grid[index] = int(double(hue) * color_scaler);
    }
}

void histogramGPU(int *gputemp, int *gpugrid, dim3 dimGrid, dim3 dimBlock, size_t width, size_t height)
{
    std::cerr << "calculating histogram color scheme on GPU ..." << std::endl
              << std::endl;

    size_t histogram_size = 256 * sizeof(int);

    size_t image_size = width * height * sizeof(int);

    size_t total = width * height;

    double color_scaler = 1.0 / double(total) * 255.0;

    int *gpuhistogram;

    int *cpuhistogram;

    // allocate required memory
    cpuhistogram = (int *)malloc(histogram_size);

    if (cpuhistogram != nullptr)
    {
        // initialize histogram on cpu
        std::memset(cpuhistogram, 0, histogram_size);

        // allocate histogram on gpu
        HIP_CHECK(hipMalloc((void **)&gpuhistogram, histogram_size));

        // copy cpu histogram to gpu
        HIP_CHECK(hipMemcpy(gpuhistogram, cpuhistogram, histogram_size, hipMemcpyHostToDevice));

        // compute histogram
        hipLaunchKernelGGL(histogram_gpu_counts, dimGrid, dimBlock, 0, 0, gputemp, gpuhistogram, width, height);

        // finalize colors
        hipLaunchKernelGGL(histogram_gpu_finalize, dimGrid, dimBlock, 0, 0, gputemp, gpuhistogram, width, height, total, color_scaler);

        // copy color-mapped image from gpu
        HIP_CHECK(hipMemcpy(gpugrid, gputemp, image_size, hipMemcpyDeviceToHost));

        // free allocated arrays
        HIP_CHECK(hipFree(gpuhistogram));

        free(cpuhistogram);
    }
    else
    {
        std::cerr << "Unable to allocate RAM on host!" << std::endl;

        exit(1);
    }
}

void histogram(int *grid, size_t width, size_t height)
{
    std::cerr << "calculating histogram color scheme on CPU ..." << std::endl
              << std::endl;

    // compute total counts
    size_t total = width * height;

    int NumIterationsPerPixel[256];

    int *hue = (int *)malloc(total * sizeof(int));

    std::memset(NumIterationsPerPixel, 0, 256 * sizeof(int));

    if (hue != nullptr)
    {
        // pass 1: compute histogram
        for (size_t i = 0; i < width * height; i++)
        {
            NumIterationsPerPixel[grid[i]]++;

            hue[i] = 0;
        }

        auto color_scaler = 1.0 / double(total) * 255.0;

        // pass 2: compute hue and map to color
        for (size_t i = 0; i < width * height; i++)
        {
            for (size_t j = 0; j <= grid[i]; j++)
            {
                hue[i] += NumIterationsPerPixel[j];
            }

            grid[i] = int(double(hue[i]) * color_scaler);
        }

        free(hue);
    }
    else
    {
        std::cerr << "Unable to allocate RAM on host!" << std::endl;

        exit(1);
    }
}
