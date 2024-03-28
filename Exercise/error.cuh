// error.cuh

#pragma once
#include <stdio.h>

#define CHECK_cuda(call)                              \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

#define CHECK_cublas(call)                            \
do                                                    \
{                                                     \
    const cublasStatus_t error_code = call;         \
    if (error_code != CUBLAS_STATUS_SUCCESS)        \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        exit(1);                                      \
    }                                                 \
} while (0)

#define CHECK_cusolver(call)                          \
do                                                    \
{                                                     \
    const cusolverStatus_t error_code = call;         \
    if (error_code != CUSOLVER_STATUS_SUCCESS)        \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        exit(1);                                      \
    }                                                 \
} while (0)

#define CHECK_cusparse(call)                          \
do                                                    \
{                                                     \
    const cusparseStatus_t error_code = call;         \
    if (error_code != CUSPARSE_STATUS_SUCCESS)        \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cusparseGetErrorString(error_code));      \
        exit(1);                                      \
    }                                                 \
} while (0)