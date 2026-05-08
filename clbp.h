/*
 * clbp.h - Header-only OpenCL library
 * Just #include this file and use init(), run(), cleanup()
 * 
 * Usage:
 *   #include "clbp.h"
 */

#ifndef CLPB_H
#define CLPB_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <CL/cl.h>
#ifdef _WIN32
/* Forward-declare GetModuleFileNameA to avoid pulling in all of windows.h
   (which conflicts with raylib's Rectangle, CloseWindow, etc.) */
extern __declspec(dllimport) unsigned long __stdcall
    GetModuleFileNameA(void* hModule, char* lpFilename, unsigned long nSize);
#endif

#define OPENCL_MAX_KERNELS 16

typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_program programs[OPENCL_MAX_KERNELS];
    cl_kernel kernels[OPENCL_MAX_KERNELS];
    int num_kernels;
} OpenCLContext;

static OpenCLContext* _opencl_ctx = NULL;

/* Error checking macro */
#define OPENCL_CHECK(err, msg) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL Error %d: %s\n", err, msg); \
        exit(1); \
    }

/* Read file into string */
static char* _opencl_read_file(const char* filename) {
    FILE* f = fopen(filename, "rb");  /* binary mode — ftell matches fread on Windows */
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    size_t size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);

    char* source = (char*)malloc(size + 1);
    size_t n = fread(source, 1, size, f);
    source[n] = '\0';  /* null-terminate at actual bytes read, not ftell size */
    fclose(f);

    return source;
}

/* Initialize OpenCL context */
static inline void clbpinit() {
    if (_opencl_ctx) return;  // Already initialized
    
    _opencl_ctx = (OpenCLContext*)malloc(sizeof(OpenCLContext));
    _opencl_ctx->num_kernels = 0;
    
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &_opencl_ctx->device, NULL);
    
    _opencl_ctx->context = clCreateContext(NULL, 1, &_opencl_ctx->device, NULL, NULL, &err);
    OPENCL_CHECK(err, "Failed to create context");
    
    _opencl_ctx->queue = clCreateCommandQueue(_opencl_ctx->context, _opencl_ctx->device, 0, &err);
    OPENCL_CHECK(err, "Failed to create queue");
    
    printf("✓ OpenCL initialized\n");
}

/* Resolve a .cl filename relative to the running exe's directory.
   Falls back to the bare name (CWD) if the exe path cannot be determined. */
static char* _opencl_resolve_path(const char* cl_file) {
    /* If already absolute, use as-is */
    if (cl_file[0] == '/' || cl_file[0] == '\\' ||
        (cl_file[0] && cl_file[1] == ':')) {
        char* p = (char*)malloc(strlen(cl_file) + 1);
        strcpy(p, cl_file);
        return p;
    }
#ifdef _WIN32
    char exe_path[260];  /* MAX_PATH without pulling in windows.h */
    unsigned long len = GetModuleFileNameA(NULL, exe_path, 260);
    if (len > 0) {
        /* Strip the exe filename to get its directory */
        char* last_sep = NULL;
        for (char* p = exe_path; *p; p++)
            if (*p == '\\' || *p == '/') last_sep = p;
        if (last_sep) {
            size_t dir_len  = (size_t)(last_sep - exe_path) + 1; /* include separator */
            size_t file_len = strlen(cl_file);
            char*  result   = (char*)malloc(dir_len + file_len + 1);
            memcpy(result, exe_path, dir_len);
            memcpy(result + dir_len, cl_file, file_len + 1);
            return result;
        }
    }
#endif
    /* Fallback: use bare name (CWD) */
    char* p = (char*)malloc(strlen(cl_file) + 1);
    strcpy(p, cl_file);
    return p;
}

/* Load a kernel from file */
static inline int loadkernel(const char* cl_file, const char* kernel_name) {
    if (!_opencl_ctx) clbpinit();

    if (_opencl_ctx->num_kernels >= OPENCL_MAX_KERNELS) {
        fprintf(stderr, "Maximum kernels (%d) reached\n", OPENCL_MAX_KERNELS);
        return -1;
    }

    char* resolved = _opencl_resolve_path(cl_file);
    char* source   = _opencl_read_file(resolved);
    free(resolved);
    if (!source) return -1;

    cl_int err;
    
    int idx = _opencl_ctx->num_kernels;
    
    _opencl_ctx->programs[idx] = clCreateProgramWithSource(_opencl_ctx->context, 1,
                                                           (const char**)&source, NULL, &err);
    OPENCL_CHECK(err, "Failed to create program");
    
    err = clBuildProgram(_opencl_ctx->programs[idx], 1, &_opencl_ctx->device,
                         "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(_opencl_ctx->programs[idx], _opencl_ctx->device,
                            CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(_opencl_ctx->programs[idx], _opencl_ctx->device,
                            CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error for %s:\n%s\n", cl_file, log);
        free(log);
        free(source);
        return -1;
    }
    
    _opencl_ctx->kernels[idx] = clCreateKernel(_opencl_ctx->programs[idx], kernel_name, &err);
    OPENCL_CHECK(err, "Failed to create kernel");
    
    free(source);
    
    printf("✓ Loaded kernel '%s' from %s (ID: %d)\n", kernel_name, cl_file, idx);
    _opencl_ctx->num_kernels++;
    return idx;
}

/* Run kernel with arrays (simple 3-array version) - OPTIMIZED */
static inline void runkernel(int kernel_id, int N, float* a, float* b, float* out) {
    // Fast validation
    if (!_opencl_ctx || kernel_id < 0 || kernel_id >= _opencl_ctx->num_kernels) {
        fprintf(stderr, "Invalid kernel ID: %d\n", kernel_id);
        return;
    }
    
    cl_int err;
    cl_kernel kernel = _opencl_ctx->kernels[kernel_id];
    size_t buffer_size = N * sizeof(float);
    
    // Create all buffers at once - better cache locality
    cl_mem a_buf = clCreateBuffer(_opencl_ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 buffer_size, a, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error %d: Failed to create buffer a\n", err);
        return;
    }
    
    cl_mem b_buf = clCreateBuffer(_opencl_ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 buffer_size, b, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(a_buf);
        fprintf(stderr, "OpenCL Error %d: Failed to create buffer b\n", err);
        return;
    }
    
    cl_mem out_buf = clCreateBuffer(_opencl_ctx->context, CL_MEM_WRITE_ONLY,
                                   buffer_size, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(a_buf);
        clReleaseMemObject(b_buf);
        fprintf(stderr, "OpenCL Error %d: Failed to create buffer out\n", err);
        return;
    }
    
    // Set all arguments - batch operations
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_buf);
    
    // Execute
    size_t global_size = N;
    err = clEnqueueNDRangeKernel(_opencl_ctx->queue, kernel, 1, NULL,
                                &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error %d: Failed to execute kernel\n", err);
        goto cleanup;
    }
    
    // Read results - blocking read ensures completion
    err = clEnqueueReadBuffer(_opencl_ctx->queue, out_buf, CL_TRUE, 0,
                             buffer_size, out, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error %d: Failed to read buffer\n", err);
    }
    
cleanup:
    // Cleanup all buffers
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(out_buf);
}

/* Advanced: Run kernel with custom arguments - OPTIMIZED */
static inline void advrunkernel(int kernel_id, size_t global_size, int num_args, ...) {
    // Fast validation - single check
    if (!_opencl_ctx || kernel_id < 0 || kernel_id >= _opencl_ctx->num_kernels) {
        fprintf(stderr, "Invalid kernel ID: %d\n", kernel_id);
        return;
    }
    
    cl_int err;
    cl_kernel kernel = _opencl_ctx->kernels[kernel_id];
    
    // Set all arguments - batch error checking
    va_list args;
    va_start(args, num_args);
    
    int first_error = CL_SUCCESS;
    for (int i = 0; i < num_args; i++) {
        void* arg = va_arg(args, void*);
        size_t arg_size = va_arg(args, size_t);
        err = clSetKernelArg(kernel, i, arg_size, arg);
        
        // Record only first error, continue setting args
        if (err != CL_SUCCESS && first_error == CL_SUCCESS) {
            first_error = err;
        }
    }
    
    va_end(args);
    
    for (int i = 0; i < num_args; i++) {
        void* arg = va_arg(args, void*);
        size_t arg_size = va_arg(args, size_t);
        err = clSetKernelArg(kernel, i, arg_size, arg);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "OpenCL Error %d: arg index %d size %zu\n", err, i, arg_size);
            first_error = err;
        }
    }
    
    // Check errors after all args set (faster than checking each time)
    if (first_error != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error %d: Failed to set kernel arg\n", first_error);
        return;
    }
    
    // Execute kernel — no clFinish here; the blocking readbuf in fb.sync()
    // is in the same in-order queue and implicitly waits for the kernel.
    err = clEnqueueNDRangeKernel(_opencl_ctx->queue, kernel, 1, NULL,
                                &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error %d: Failed to execute kernel\n", err);
    }
}

/* Helper: Create GPU buffer from CPU array */
static inline cl_mem loadbuf(float* data, int N, int read_only) {
    if (!_opencl_ctx) {
        fprintf(stderr, "OpenCL not initialized\n");
        return NULL;
    }

    cl_int err;
    if (read_only && data) {
        return clCreateBuffer(_opencl_ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            N * sizeof(float), data, &err);
    } else {
        return clCreateBuffer(_opencl_ctx->context, CL_MEM_WRITE_ONLY,
                            N * sizeof(float), NULL, &err);
    }
}

/* Helper: Create a persistent READ_WRITE GPU buffer and upload data.
   Used by ff_list_pin so the buffer works as both input and output. */
static inline cl_mem pinbuf(float* data, int N) {
    if (!_opencl_ctx) {
        fprintf(stderr, "OpenCL not initialized\n");
        return NULL;
    }
    cl_int err;
    cl_mem buf = clCreateBuffer(_opencl_ctx->context, CL_MEM_READ_WRITE,
                                N * sizeof(float), NULL, &err);
    if (buf && data && N > 0) {
        clEnqueueWriteBuffer(_opencl_ctx->queue, buf, CL_TRUE, 0,
                             N * sizeof(float), data, 0, NULL, NULL);
    }
    return buf;
}

/* Helper: Read GPU buffer to CPU */
static inline void readbuf(cl_mem buffer, float* data, int N) {
    if (!_opencl_ctx) return;
    clEnqueueReadBuffer(_opencl_ctx->queue, buffer, CL_TRUE, 0,
                       N * sizeof(float), data, 0, NULL, NULL);
}

/* Helper: Free GPU buffer */
static inline void freebuf(cl_mem buffer) {
    clReleaseMemObject(buffer);
}

/* Cleanup OpenCL */
static inline void cleanupkernel() {
    if (!_opencl_ctx) return;
    
    for (int i = 0; i < _opencl_ctx->num_kernels; i++) {
        clReleaseKernel(_opencl_ctx->kernels[i]);
        clReleaseProgram(_opencl_ctx->programs[i]);
    }
    
    clReleaseCommandQueue(_opencl_ctx->queue);
    clReleaseContext(_opencl_ctx->context);
    free(_opencl_ctx);
    _opencl_ctx = NULL;
    
    printf("✓ OpenCL cleaned up\n");
}

#endif /* CLBP */