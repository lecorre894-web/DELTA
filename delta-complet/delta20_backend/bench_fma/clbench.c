#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#define NUM_INVOCATIONS (256 * 4096 * 4)
#define ITERS 200000
#define FMA_FLOPS 2

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static const char *kernel_src =
"__kernel void bench(__global float4 *data) {\n"
"    int idx = get_global_id(0);\n"
"    float4 a = data[idx];\n"
"    float4 b = (float4)(1.000001f);\n"
"    float4 c = (float4)(0.0000001f);\n"
"    for (int i = 0; i < 200000; ++i) {\n"
"        a = a * b + c;\n"
"    }\n"
"    data[idx] = a;\n"
"}\n";

static void check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s failed: %d\n", msg, err);
        exit(1);
    }
}

int main(void) {
    cl_uint numPlatforms = 0;
    check(clGetPlatformIDs(0, NULL, &numPlatforms), "clGetPlatformIDs count");
    if (numPlatforms == 0) { fprintf(stderr, "No OpenCL platforms\n"); return 1; }
    cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * numPlatforms);
    check(clGetPlatformIDs(numPlatforms, platforms, NULL), "clGetPlatformIDs list");

    cl_device_id device = NULL;
    cl_platform_id chosenPlatform = NULL;
    for (cl_uint p = 0; p < numPlatforms; p++) {
        cl_uint numDevices = 0;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices) == CL_SUCCESS && numDevices > 0) {
            cl_device_id *devs = malloc(sizeof(cl_device_id) * numDevices);
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devs, NULL);
            device = devs[0];
            chosenPlatform = platforms[p];
            free(devs);
            break;
        }
    }
    if (!device) { fprintf(stderr, "No OpenCL device found\n"); return 1; }

    char nameBuf[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(nameBuf), nameBuf, NULL);
    printf("Using device: %s\n", nameBuf);
    char platBuf[256];
    clGetPlatformInfo(chosenPlatform, CL_PLATFORM_NAME, sizeof(platBuf), platBuf, NULL);
    printf("Platform: %s\n", platBuf);

    cl_device_type devType;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(devType), &devType, NULL);
    printf("Device type: %s%s%s%s\n",
        (devType & CL_DEVICE_TYPE_CPU) ? "CPU " : "",
        (devType & CL_DEVICE_TYPE_GPU) ? "GPU " : "",
        (devType & CL_DEVICE_TYPE_ACCELERATOR) ? "ACCELERATOR " : "",
        (devType == 0) ? "UNKNOWN" : "");

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check(err, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    check(err, "clCreateCommandQueue");

    size_t bufSize = NUM_INVOCATIONS * sizeof(float);
    float *hostData = malloc(bufSize);
    for (int i = 0; i < NUM_INVOCATIONS; i++) hostData[i] = 1.0f;

    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufSize, hostData, &err);
    check(err, "clCreateBuffer");

    cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    check(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *log = malloc(logSize + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        log[logSize] = 0;
        fprintf(stderr, "Build log:\n%s\n", log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "bench", &err);
    check(err, "clCreateKernel");
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    check(err, "clSetKernelArg");

    size_t globalSize = NUM_INVOCATIONS / 4;
    size_t localSize = 256;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    check(err, "warmup clEnqueueNDRangeKernel");
    clFinish(queue);

    double t0 = now_ms();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    check(err, "clEnqueueNDRangeKernel");
    clFinish(queue);
    double t1 = now_ms();

    double elapsed_ms = t1 - t0;
    double total_flops = (double)NUM_INVOCATIONS * ITERS * FMA_FLOPS;
    double gflops = total_flops / (elapsed_ms / 1000.0) / 1e9;

    printf("Invocations (scalar elements): %d, iters/invocation: %d\n", NUM_INVOCATIONS, ITERS);
    printf("Elapsed: %.3f ms\n", elapsed_ms);
    printf("Estimated: %.2f GFLOPS\n", gflops);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buf);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(hostData);
    free(platforms);
    return 0;
}
