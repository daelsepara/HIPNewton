#ifndef __HIP_HISTOGRAM_H__
#define __HIP_HISTOGRAM_H__

#include <hip/hip_runtime.h>

#include "hipcheck.h"

void histogramGPU(int *gputemp, int *gpugrid, dim3 dimGrid, dim3 dimBlock, size_t width, size_t height);

void histogram(int *grid, size_t width, size_t height);

#endif