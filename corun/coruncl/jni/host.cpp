// ./corun_kernel 1048576 1000 256 64

// host.cpp
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <inttypes.h>

#define CHECK_ERR(err, msg) \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "%s failed (err=%d)\n", msg, err); \
        std::exit(EXIT_FAILURE); \
    }

#define ERT_FLOP 2

const char* KERNEL_PATH = "corun_kernel.cl";

static char* loadSource(const char* filename, size_t* outSize) {
    FILE* fp = std::fopen(filename, "rb");
    if (!fp) { perror("fopen"); std::exit(EXIT_FAILURE); }
    std::fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char* src = (char*)std::malloc(size + 1);
    fread(src, 1, size, fp);
    src[size] = '\0';
    fclose(fp);
    *outSize = size;
    return src;
}

int main(int argc, char** argv) {
    const size_t nsize   = (argc > 1 ? std::atoi(argv[1]) : (1<<20));
    const size_t ntrials = (argc > 2 ? std::atoi(argv[2]) : 1000);
    const size_t threads = (argc > 3 ? std::atoi(argv[3]) : 256);
    const size_t blocks  = (argc > 4 ? std::atoi(argv[4]) : 64);

    cl_int err;
    cl_uint numPlat = 0;
    CHECK_ERR(clGetPlatformIDs(0, nullptr, &numPlat), "clGetPlatformIDs");
    std::vector<cl_platform_id> plats(numPlat);
    CHECK_ERR(clGetPlatformIDs(numPlat, plats.data(), nullptr), "clGetPlatformIDs");

    cl_device_id device = nullptr;
    for (auto &plat : plats) {
        cl_uint numDev = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDev) == CL_SUCCESS
            && numDev > 0) {
            std::vector<cl_device_id> devs(numDev);
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, numDev, devs.data(), nullptr);
            device = devs[0];
            break;
        }
    }
    if (!device) {
        fprintf(stderr, "GPU device not found\n");
        return EXIT_FAILURE;
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_ERR(err, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERR(err, "clCreateCommandQueue");

    size_t srcSize;  
    char* srcStr = loadSource(KERNEL_PATH, &srcSize);
    cl_program program = clCreateProgramWithSource(
        context, 1, (const char**)&srcStr, &srcSize, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device,
            CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device,
            CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        fprintf(stderr, "Build log:\n%s\n", log.data());
        CHECK_ERR(err, "clBuildProgram");
    }
    std::free(srcStr);

    cl_kernel kernel = clCreateKernel(program, "block_stride", &err);
    CHECK_ERR(err, "clCreateKernel");

    size_t bufferBytes = nsize * sizeof(float);
    cl_mem bufA    = clCreateBuffer(
        context, CL_MEM_READ_WRITE, bufferBytes, nullptr, &err);
    CHECK_ERR(err, "clCreateBuffer A");
    cl_mem bufBPE  = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);
    CHECK_ERR(err, "clCreateBuffer BPE");
    cl_mem bufMPE  = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);
    CHECK_ERR(err, "clCreateBuffer MPE");

    std::vector<float> hostA(nsize, 0.5f);
    CHECK_ERR(clEnqueueWriteBuffer(
        queue, bufA, CL_TRUE, 0, bufferBytes, hostA.data(),
        0, nullptr, nullptr), "clEnqueueWriteBuffer");

    CHECK_ERR(clSetKernelArg(kernel, 0, sizeof(bufA),    &bufA),    "clSetKernelArg 0");
    CHECK_ERR(clSetKernelArg(kernel, 1, sizeof(ntrials), &ntrials),"clSetKernelArg 1");
    CHECK_ERR(clSetKernelArg(kernel, 2, sizeof(nsize),   &nsize),  "clSetKernelArg 2");
    CHECK_ERR(clSetKernelArg(kernel, 3, sizeof(bufBPE),  &bufBPE),  "clSetKernelArg 3");
    CHECK_ERR(clSetKernelArg(kernel, 4, sizeof(bufMPE),  &bufMPE),  "clSetKernelArg 4");

    size_t global = threads * blocks;
    size_t local  = threads;
    cl_event prof_event;
    CHECK_ERR(clEnqueueNDRangeKernel(
        queue, kernel, 1, nullptr,
        &global, &local, 0, nullptr,
        &prof_event), "clEnqueueNDRangeKernel");
    CHECK_ERR(clWaitForEvents(1, &prof_event), "clWaitForEvents");

    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(prof_event,
        CL_PROFILING_COMMAND_START,
        sizeof(t0), &t0, nullptr);
    clGetEventProfilingInfo(prof_event,
        CL_PROFILING_COMMAND_END,
        sizeof(t1), &t1, nullptr);
    double kernel_time_s = double(t1 - t0) * 1e-9;

    int bytesPerElem = 0, memsPerElem = 0;
    CHECK_ERR(clEnqueueReadBuffer(
        queue, bufBPE, CL_TRUE, 0,
        sizeof(int), &bytesPerElem,
        0, nullptr, nullptr), "clEnqueueReadBuffer BPE");
    CHECK_ERR(clEnqueueReadBuffer(
        queue, bufMPE, CL_TRUE, 0,
        sizeof(int), &memsPerElem,
        0, nullptr, nullptr), "clEnqueueReadBuffer MPE");

    uint64_t total_bytes = uint64_t(ntrials)
                         * uint64_t(nsize)
                         * uint64_t(bytesPerElem)
                         * uint64_t(memsPerElem);
    uint64_t total_flops = uint64_t(ntrials)
                         * uint64_t(nsize)
                         * uint64_t(ERT_FLOP);

    double bandwidth_GBs = double(total_bytes)
                         / kernel_time_s
                         / 1024.0 / 1024.0 / 1024.0;
    double gflops        = double(total_flops)
                         / kernel_time_s
                         / 1e9;

    printf("Time         : %.6f s\n",   kernel_time_s);
    printf("Bytes moved  : %" PRIu64 "\n", total_bytes);
    printf("Bandwidth    : %.3f GB/s\n", bandwidth_GBs);
    printf("Flops done   : %" PRIu64 "\n", total_flops);
    printf("GFLOPS       : %.3f\n",      gflops);

    clReleaseEvent(prof_event);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufBPE);
    clReleaseMemObject(bufMPE);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
