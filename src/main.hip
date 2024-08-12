#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>

#include <hip/hip_runtime.h>

#include "hipcheck.h"
#include "histogram.h"

__host__ __device__ double mag2(double zx, double zy)
{
    return zx * zx + zy * zy;
}

__host__ __device__ int escape_time(size_t x, size_t y, const size_t height, const double dx, const double dy, const double tolerance, const size_t max_iter)
{
    // initialize escape time
    size_t t = 0;

    // calculate initial pixel zx, zy coordinate on complex plane
    double zx = -2.5 + (double)x * dx;

    double zy = -2.5 + (double)(height - y - 1) * dy;

    // initialize other variables
    double diff = std::numeric_limits<double>::infinity();

    double oldx = 0.0;

    double oldy = 0.0;

    double zxx = 0.0;

    double zyy = 0.0;

    double tmp = 0.0;

    double tmp3 = 0.0;

    // generate escape time fractal
    while (diff > tolerance && t < max_iter)
    {
        oldx = zx;

        oldy = zy;

        zxx = zx * zx;

        zyy = zy * zy;

        tmp = (zxx + zyy) * (zxx + zyy);

        tmp3 = 3.0 * tmp;

        zx = ((zx + zx) * tmp + zxx - zyy) / tmp3;

        zy = ((zy + zy) * (tmp - oldx)) / tmp3;

        diff = mag2(zx - oldx, zy - oldy);

        t++;
    }

    // set color for trapped
    if (t == max_iter)
    {
        t = 0;
    }

    return (int)t;
}

// device (kernel) function
__global__ void newtonGPU(int *grid, const size_t width, const size_t height, const double dx, const double dy, const double tolerance, const size_t max_iter)
{
    // get exact location in array
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < width && y < height)
    {
        int t = escape_time(x, y, height, dx, dy, tolerance, max_iter);

        grid[y * width + x] = t;
    }

    __syncthreads();
}

// host (kernel) function
void newtonCPU(int *grid, const size_t width, const size_t height, const double dx, const double dy, const double tolerance, const size_t max_iter)
{
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            int t = escape_time(x, y, height, dx, dy, tolerance, max_iter);

            grid[y * width + x] = t;
        }
    }
}

void generate_image_gray(int *grid, size_t WIDTH, size_t HEIGHT, int brightness, const char *filename)
{
    std::ofstream ofs(filename);

    std::cerr << "saving GPU image to " << filename << " ..." << std::endl;

    ofs << "P2" << std::endl
        << WIDTH << ' ' << HEIGHT << std::endl
        << "255" << std::endl;

    for (size_t j = 0; j < HEIGHT; ++j)
    {
        for (size_t i = 0; i < WIDTH; ++i)
        {
            ofs << grid[j * WIDTH + i] % 256 * brightness << ' ';
        }
    }

    ofs.close();
}

void generate_image_rgb(int *grid, size_t WIDTH, size_t HEIGHT, int brightness, const char *filename)
{
    unsigned char colors[768] = {
        0, 0, 0, 0, 36, 0, 0, 71, 0, 0, 108, 0, 0, 145, 0, 0, 181,
        0, 0, 218, 0, 0, 255, 0, 84, 0, 0, 84, 36, 0, 84, 71, 0, 84,
        108, 0, 84, 145, 0, 84, 181, 0, 84, 218, 0, 84, 255, 0, 170,
        0, 0, 170, 36, 0, 170, 71, 0, 170, 108, 0, 170, 145, 0, 170,
        181, 0, 170, 218, 0, 170, 255, 0, 255, 0, 0, 255, 36, 0,
        255, 71, 0, 255, 108, 0, 255, 145, 0, 255, 181, 0, 255, 218,
        0, 255, 255, 0, 0, 0, 36, 0, 36, 36, 0, 71, 36, 0, 108, 36,
        0, 145, 36, 0, 181, 36, 0, 218, 36, 0, 255, 36, 84, 0, 36,
        84, 36, 36, 84, 71, 36, 84, 108, 36, 84, 145, 36, 84, 181,
        36, 84, 218, 36, 84, 255, 36, 170, 0, 36, 170, 36, 36, 170,
        71, 36, 170, 108, 36, 170, 145, 36, 170, 181, 36, 170, 218,
        36, 170, 255, 36, 255, 0, 36, 255, 36, 36, 255, 71, 36, 255,
        108, 36, 255, 145, 36, 255, 181, 36, 255, 218, 36, 255, 255,
        36, 0, 0, 71, 0, 36, 71, 0, 71, 71, 0, 108, 71, 0, 145, 71,
        0, 181, 71, 0, 218, 71, 0, 255, 71, 84, 0, 71, 84, 36, 71,
        84, 71, 71, 84, 108, 71, 84, 145, 71, 84, 181, 71, 84, 218,
        71, 84, 255, 71, 170, 0, 71, 170, 36, 71, 170, 71, 71, 170,
        108, 71, 170, 145, 71, 170, 181, 71, 170, 218, 71, 170, 255,
        71, 255, 0, 71, 255, 36, 71, 255, 71, 71, 255, 108, 71, 255,
        145, 71, 255, 181, 71, 255, 218, 71, 255, 255, 71, 0, 0,
        108, 0, 36, 108, 0, 71, 108, 0, 108, 108, 0, 145, 108, 0,
        181, 108, 0, 218, 108, 0, 255, 108, 84, 0, 108, 84, 36, 108,
        84, 71, 108, 84, 108, 108, 84, 145, 108, 84, 181, 108, 84,
        218, 108, 84, 255, 108, 170, 0, 108, 170, 36, 108, 170, 71,
        108, 170, 108, 108, 170, 145, 108, 170, 181, 108, 170, 218,
        108, 170, 255, 108, 255, 0, 108, 255, 36, 108, 255, 71, 108,
        255, 108, 108, 255, 145, 108, 255, 181, 108, 255, 218, 108,
        255, 255, 108, 0, 0, 145, 0, 36, 145, 0, 71, 145, 0, 108,
        145, 0, 145, 145, 0, 181, 145, 0, 218, 145, 0, 255, 145, 84,
        0, 145, 84, 36, 145, 84, 71, 145, 84, 108, 145, 84, 145,
        145, 84, 181, 145, 84, 218, 145, 84, 255, 145, 170, 0, 145,
        170, 36, 145, 170, 71, 145, 170, 108, 145, 170, 145, 145,
        170, 181, 145, 170, 218, 145, 170, 255, 145, 255, 0, 145,
        255, 36, 145, 255, 71, 145, 255, 108, 145, 255, 145, 145,
        255, 181, 145, 255, 218, 145, 255, 255, 145, 0, 0, 181, 0,
        36, 181, 0, 71, 181, 0, 108, 181, 0, 145, 181, 0, 181, 181,
        0, 218, 181, 0, 255, 181, 84, 0, 181, 84, 36, 181, 84, 71,
        181, 84, 108, 181, 84, 145, 181, 84, 181, 181, 84, 218, 181,
        84, 255, 181, 170, 0, 181, 170, 36, 181, 170, 71, 181, 170,
        108, 181, 170, 145, 181, 170, 181, 181, 170, 218, 181, 170,
        255, 181, 255, 0, 181, 255, 36, 181, 255, 71, 181, 255, 108,
        181, 255, 145, 181, 255, 181, 181, 255, 218, 181, 255, 255,
        181, 0, 0, 218, 0, 36, 218, 0, 71, 218, 0, 108, 218, 0, 145,
        218, 0, 181, 218, 0, 218, 218, 0, 255, 218, 84, 0, 218, 84,
        36, 218, 84, 71, 218, 84, 108, 218, 84, 145, 218, 84, 181,
        218, 84, 218, 218, 84, 255, 218, 170, 0, 218, 170, 36, 218,
        170, 71, 218, 170, 108, 218, 170, 145, 218, 170, 181, 218,
        170, 218, 218, 170, 255, 218, 255, 0, 218, 255, 36, 218,
        255, 71, 218, 255, 108, 218, 255, 145, 218, 255, 181, 218,
        255, 218, 218, 255, 255, 218, 0, 0, 255, 0, 36, 255, 0, 71,
        255, 0, 108, 255, 0, 145, 255, 0, 181, 255, 0, 218, 255, 0,
        255, 255, 84, 0, 255, 84, 36, 255, 84, 71, 255, 84, 108,
        255, 84, 145, 255, 84, 181, 255, 84, 218, 255, 84, 255, 255,
        170, 0, 255, 170, 36, 255, 170, 71, 255, 170, 108, 255, 170,
        145, 255, 170, 181, 255, 170, 218, 255, 170, 255, 255, 255,
        0, 255, 255, 36, 255, 255, 71, 255, 255, 108, 255, 255, 145,
        255, 255, 181, 255, 255, 218, 255, 255, 255, 255};

    std::ofstream ofs(filename);

    std::cerr << "saving GPU image to " << filename << " ..." << std::endl;

    ofs << "P6" << std::endl
        << WIDTH << ' ' << HEIGHT << std::endl
        << "255" << std::endl;

    for (size_t j = 0; j < HEIGHT; ++j)
    {
        for (size_t i = 0; i < WIDTH; ++i)
        {
            int index = grid[j * WIDTH + i] % 256 * 3;

            char r = (char)(colors[index] * brightness % 256);
            char g = (char)(colors[index + 1] * brightness % 256);
            char b = (char)(colors[index + 2] * brightness % 256);

            ofs << (char)g << (char)b << (char)r;
        }
    }

    ofs.close();
}

bool check_argument(char **argv, int i, const char *param)
{
    std::string arg = std::string(argv[i]);

    std::transform(arg.begin(), arg.end(), arg.begin(), ::toupper);

    return arg == param;
}

int main(int argc, char **argv)
{
    size_t WIDTH = 2048;

    size_t HEIGHT = 2048;

    size_t BYTES = sizeof(int);

    bool gray = false;

    bool cpuhist = false;

    if (argc < 2)
    {
        std::cerr << "To use:" << std::endl
                  << std::endl
                  << argv[0] << " [WIDTH] [HEIGHT]" << std::endl;

        exit(1);
    }
    else if (argc < 3)
    {
        WIDTH = std::atoi(argv[1]);

        HEIGHT = WIDTH;
    }
    else if (argc < 4)
    {
        WIDTH = std::atoi(argv[1]);

        HEIGHT = std::atoi(argv[2]);
    }
    else if (argc < 5)
    {
        WIDTH = std::atoi(argv[1]);

        HEIGHT = std::atoi(argv[2]);

        gray = check_argument(argv, 3, "/GRAY");

        cpuhist = check_argument(argv, 3, "/HISTCPU");
    }
    else
    {
        WIDTH = std::atoi(argv[1]);

        HEIGHT = std::atoi(argv[2]);

        gray = check_argument(argv, 3, "/GRAY");

        cpuhist = check_argument(argv, 4, "/HISTCPU");
    }

    // get device properties
    hipDeviceProp_t devProp;

    HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

    std::cerr << std::endl
              << "Device: " << std::string(devProp.name) << std::endl;

    // check if there is sufficient GPU memory
    size_t requiredMem = (size_t(WIDTH) * size_t(HEIGHT)) * size_t(BYTES);

    size_t totalMem = devProp.totalGlobalMem;

    size_t memLimit = size_t(size_t(32768) * size_t(65536) * BYTES);

    std::cerr << std::endl;

    std::cerr << "Estimated GPU RAM requirements: " << requiredMem << " bytes" << std::endl;

    std::cerr << "Total GPU RAM: " << totalMem << " bytes" << std::endl;

    std::cerr << "Program GPU RAM limit: " << memLimit << " bytes" << std::endl;

    std::cerr << "Size of int: " << BYTES << " bytes" << std::endl;

    if (requiredMem >= memLimit || requiredMem >= totalMem)
    {
        std::cerr << std::endl
                  << "Unable to run this test!" << std::endl
                  << std::endl;

        exit(1);
    }

    // complex plane (array placeholders)
    int *gputemp = nullptr;

    int *gpugrid = nullptr;

    int *cpugrid = nullptr;

    // gpu run configuration
    const size_t NUM = WIDTH * HEIGHT;

    const size_t THREADS_PER_BLOCK_X = 16;

    const size_t THREADS_PER_BLOCK_Y = 16;

    const size_t THREADS_PER_BLOCK_Z = 1;

    const size_t GRID_X = std::ceil(WIDTH / double(THREADS_PER_BLOCK_X));

    const size_t GRID_Y = std::ceil(HEIGHT / double(THREADS_PER_BLOCK_Y));

    dim3 dimGrid(GRID_X, GRID_Y);

    dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

    // allocate memory on GPU and CPU
    HIP_CHECK(hipMalloc((void **)&gputemp, size_t(NUM * BYTES)));

    cpugrid = (int *)malloc(NUM * BYTES);

    gpugrid = (int *)malloc(NUM * BYTES);

    if (cpugrid == nullptr || gpugrid == nullptr)
    {
        std::cerr << "Unable to allocate RAM on host!" << std::endl;

        if (cpugrid != nullptr)
        {
            free(cpugrid);
        }

        if (gpugrid != nullptr)
        {
            free(gpugrid);
        }

        exit(1);
    }

    // initialize the input data
    std::memset(cpugrid, 0, NUM * BYTES);

    std::memset(gpugrid, 0, NUM * BYTES);

    // newton paramemters
    double dx = 5.0 / (double)WIDTH;

    double dy = 5.0 / (double)HEIGHT;

    double tolerance = 1e-6;

    size_t max_iter = 255;

    std::cerr << std::endl;

    // generate newton using GPU
    std::cerr << "generating newton set using GPU ..." << std::endl;

    // elapsed time
    float gpu_elapsed = 0.0;

    // create gpu kernel start and stop events
    hipEvent_t gpuinit;

    hipEvent_t gpustop;

    HIP_CHECK(hipEventCreate(&gpuinit));

    HIP_CHECK(hipEventCreate(&gpustop));

    // start recording
    HIP_CHECK(hipEventRecord(gpuinit, 0));

    // copy data to GPU
    HIP_CHECK(hipMemcpy(gputemp, gpugrid, NUM * BYTES, hipMemcpyHostToDevice));

    // run GPU newton generator

    // preferred launch method
    hipLaunchKernelGGL(newtonGPU, dimGrid, dimBlock, 0, 0, gputemp, WIDTH, HEIGHT, dx, dy, tolerance, max_iter);

    // copy results from GPU
    HIP_CHECK(hipMemcpy(gpugrid, gputemp, NUM * BYTES, hipMemcpyDeviceToHost));

    // stop recording and synchronize
    HIP_CHECK(hipEventRecord(gpustop, 0));

    HIP_CHECK(hipEventSynchronize(gpustop));

    // compute GPU elapsed time
    HIP_CHECK(hipEventElapsedTime(&gpu_elapsed, gpuinit, gpustop));

    std::cerr << "GPU elapsed time: " << gpu_elapsed << " ms" << std::endl;

    // destory events
    HIP_CHECK(hipEventDestroy(gpuinit));

    HIP_CHECK(hipEventDestroy(gpustop));

    // Use system clock for CPU newton generator
    using clock = std::chrono::system_clock;

    // use millisecond resolution
    using msec = std::chrono::duration<double, std::milli>;

    std::cerr << "generating newton set using CPU ..." << std::endl;

    // start recording
    std::chrono::time_point<std::chrono::system_clock> cpuinit = clock::now();

    // run CPU newton generator
    newtonCPU(cpugrid, WIDTH, HEIGHT, dx, dy, tolerance, max_iter);

    // compute CPU elapsed time
    double cpu_elapsed = ((msec)(clock::now() - cpuinit)).count();

    std::cerr << "CPU elapsed time: " << cpu_elapsed << " ms" << std::endl;

    std::cerr << std::endl;

    // check for differences
    size_t diff = 0;

    for (size_t i = 0; i < NUM; ++i)
    {
        if (std::abs(gpugrid[i] - cpugrid[i]) != 0)
        {
            diff++;
        }
    }

    if (gpu_elapsed > cpu_elapsed)
    {
        std::cerr << gpu_elapsed / cpu_elapsed << "x slowdown";
    }
    else
    {
        std::cerr << cpu_elapsed / gpu_elapsed << "x speedup";
    }

    std::cerr << ", diff: " << diff;

    std::cerr << std::endl
              << std::endl;

    int brightness = 1;

    if (WIDTH * HEIGHT <= 16384 * 16384)
    {
        if (cpuhist)
        {
            histogram(gpugrid, WIDTH, HEIGHT);
        }
        else
        {
            histogramGPU(gputemp, gpugrid, dimGrid, dimBlock, WIDTH, HEIGHT);
        }

        if (gray)
        {
            generate_image_gray(gpugrid, WIDTH, HEIGHT, brightness, "newton.pgm");
        }
        else
        {
            generate_image_rgb(gpugrid, WIDTH, HEIGHT, brightness, "newton.ppm");
        }
    }

    // free gpu memory
    HIP_CHECK(hipFree(gputemp));

    // free cpu memory
    free(gpugrid);

    free(cpugrid);

    return 0;
}