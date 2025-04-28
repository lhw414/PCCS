/* ──────────── host.cpp ────────────
 *  OpenCL host program replicating the CUDA ERT-style benchmark
 *  Build:  g++ host.cpp -lOpenCL -o corun_host
 ******************************************** */

 #include <CL/cl.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <sys/time.h>
 #include <inttypes.h>
 
 #define ERT_FLOP 2
 #define GBUNIT   (1024 * 1024 * 1024)
 
 /*         kernel launch geometry (CUDA → OpenCL)              */
 static const int GPU_BLOCKS  = 512;
 static const int GPU_THREADS = 512;
 
 /* host-side helpers */
 static double getTime()
 {
     struct timeval tv;
     gettimeofday(&tv, nullptr);
     return tv.tv_sec + tv.tv_usec / 1e6;
 }
 
 static void initialize(uint64_t n, float *A, float val)
 {
     for (uint64_t i = 0; i < n; ++i) A[i] = val;
 }
 
 /* very small error-checking wrapper */
 #define CLCHK(err, msg)                                   \
     if (err != CL_SUCCESS) {                              \
         fprintf(stderr, "%s (%d)\n", msg, err); exit(-1); \
     }
 
 int main()
 {
     const uint64_t TSIZE = 1ULL << 28;        /* 256 MiB */
     const int      nprocs = 1, nthreads = 1;
     const uint64_t PSIZE  = TSIZE / nprocs;
 
     float *buf = (float*) malloc(PSIZE);
     if (!buf) { fprintf(stderr,"OOM\n"); return -1; }
 
     cl_int  err;
     cl_uint num_plat;
     cl_platform_id platform;
     CLCHK(clGetPlatformIDs(1, &platform, &num_plat), "clGetPlatformIDs");
 
     cl_uint num_dev;
     cl_device_id device;
     CLCHK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_dev),
           "clGetDeviceIDs");
 
     cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
     CLCHK(err, "clCreateContext");
 
     cl_command_queue q =
         clCreateCommandQueueWithProperties(ctx, device, 0, &err);
     CLCHK(err, "clCreateCommandQueue");
 
     FILE *fp = fopen("corun_kernel.cl", "rb");
     fseek(fp, 0, SEEK_END);
     long fsz = ftell(fp);
     rewind(fp);
     char *src = (char*) malloc(fsz + 1);
     fread(src, 1, fsz, fp); src[fsz] = '\0'; fclose(fp);
 
     const char *sources[] = { src };
     cl_program prog = clCreateProgramWithSource(ctx, 1, sources, nullptr, &err);
     CLCHK(err, "clCreateProgramWithSource");
     err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0", nullptr, nullptr);
     if (err != CL_SUCCESS) {
         size_t logsz; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
         char *log = (char*) malloc(logsz);
         clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, logsz, log, nullptr);
         fprintf(stderr, "%s\n", log); free(log);
         exit(-1);
     }
     cl_kernel krnl = clCreateKernel(prog, "block_stride", &err);
     CLCHK(err, "clCreateKernel");
     free(src);
 
     uint64_t nsize = PSIZE / sizeof(float);
     nsize &= ~(uint64_t)(64 - 1);   /* 64-byte align */
     initialize(nsize, buf, 1.0f);
 
     cl_mem d_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                   nsize * sizeof(float), nullptr, &err);
     CLCHK(err, "clCreateBuffer");
 
     uint64_t n = 1ULL << 25;
     while (n <= nsize) {
         uint64_t max_trials = 600;
         for (uint64_t t = 1; t <= max_trials; ++t) {
             CLCHK(clEnqueueWriteBuffer(q, d_buf, CL_TRUE,
                                        0, n * sizeof(float), buf, 0, nullptr, nullptr),
                   "clEnqueueWriteBuffer");
 
             uint64_t ntrials = t;
             CLCHK(clSetKernelArg(krnl, 0, sizeof(cl_ulong), &ntrials), "arg0");
             CLCHK(clSetKernelArg(krnl, 1, sizeof(cl_ulong), &n),      "arg1");
             CLCHK(clSetKernelArg(krnl, 2, sizeof(cl_mem),   &d_buf),   "arg2");
 
             size_t local_size  = GPU_THREADS;
             size_t global_size = (size_t)GPU_BLOCKS * GPU_THREADS;
 
             double t0 = getTime();
             CLCHK(clEnqueueNDRangeKernel(q, krnl, 1,
                          nullptr, &global_size, &local_size, 0, nullptr, nullptr),
                   "clEnqueueNDRangeKernel");
             CLCHK(clFinish(q), "clFinish");
             double t1 = getTime();
 
             uint64_t working_set_size = n;        
             uint64_t bytes_per_elem   = sizeof(float);
             uint64_t total_bytes = t * working_set_size * bytes_per_elem * 2; 
             uint64_t total_flops = t * working_set_size * ERT_FLOP;
 
             printf("%12" PRIu64 " %12" PRIu64 " %15.3lf %12" PRIu64 " %12" PRIu64 "\n",
                    working_set_size * bytes_per_elem,
                    t,
                    (t1 - t0) * 1e6,
                    total_bytes,
                    total_flops);
             printf("BW: %15.3lf GiB/s\n",
                    total_bytes / (t1 - t0) / GBUNIT);
 
             CLCHK(clEnqueueReadBuffer(q, d_buf, CL_TRUE,
                                       0, n * sizeof(float), buf, 0, nullptr, nullptr),
                   "clEnqueueReadBuffer");
         }
         break;  
     }
 
     clReleaseMemObject(d_buf);
     clReleaseKernel(krnl);
     clReleaseProgram(prog);
     clReleaseCommandQueue(q);
     clReleaseContext(ctx);
     free(buf);
 
     puts("\nMETA_DATA");
     printf("FLOPS          %d\n", ERT_FLOP);
     printf("GPU_BLOCKS     %d\n", GPU_BLOCKS);
     printf("GPU_THREADS    %d\n", GPU_THREADS);
     return 0;
 }
 