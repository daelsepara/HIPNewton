#ifndef __HIP_HISTOGRAM_H__
#define __HIP_HISTOGRAM_H__

#include <hip/hip_runtime.h>

#include "hipcheck.h"

__global__ void histogram_gpu_finalize(int *grid, int *gpuhistogram, size_t width, size_t height, size_t total, double color_scaler);

__global__ void histogram_gpu_counts(int* grid, int* gpuhistogram, size_t width, size_t height);

void histogramGPU(int* gputemp, int* gpugrid, dim3 dimGrid, dim3 dimBlock, size_t width, size_t height);

void histogram(int *grid, size_t width, size_t height);

#endif