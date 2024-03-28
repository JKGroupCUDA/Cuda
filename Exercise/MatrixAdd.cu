#include "error.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void MatrixAdd(double* a, double* b, double* c, int m, int n) 

{ 

    int i = threadIdx.x + blockIdx.x * blockDim.x; 
    int j = threadIdx.y + blockIdx.y * blockDim.y; 

    if (i < m && j < n) 
    { 
        c[i*n+j] = a[i*n+j] + b[i*n+j]; 
    } 
} 

int main() 
{ 
    int m = 33;
    int n = 33;
    double *a, *b, *c;
    // 分配内存
    a = (double*)malloc(m*n*sizeof(double));
    b = (double*)malloc(m*n*sizeof(double));
    c = (double*)malloc(m*n*sizeof(double));
    for (int i = 0; i < m*n; i++)
    {
        a[i] = 0.1*i;
        b[i] = 0.2*i;
    }

    // 每块32*32
    dim3 block(32,32);
    dim3 grid((m -1)/block.x +1, (n -1)/block.y +1);
    // 分配显存
    double *a_g, *b_g, *c_g;
    CHECK_cuda(cudaMalloc((void**)&a_g, sizeof(double)*m*n));
    CHECK_cuda(cudaMalloc((void**)&b_g, sizeof(double)*m*n));
    CHECK_cuda(cudaMalloc((void**)&c_g, sizeof(double)*m*n));
    CHECK_cuda(cudaMemcpy(a_g, a, sizeof(double)*m*n, cudaMemcpyHostToDevice));
    CHECK_cuda(cudaMemcpy(b_g, b, sizeof(double)*m*n, cudaMemcpyHostToDevice));

    MatrixAdd <<<grid, block>>>(a_g, b_g, c_g, m, n); 

    CHECK_cuda(cudaMemcpy(c, c_g, sizeof(double)*m*n, cudaMemcpyDeviceToHost)); 

    for (int i = 0; i < m; i++) 
    { 
        for (int j = 0; j < n; j++) 
            printf("%.1f ", c[i*n+j]);
        printf("\n"); 
    }

    free(a);
    free(b);
    free(c);
    CHECK_cuda(cudaFree(a_g));
    CHECK_cuda(cudaFree(b_g));
    CHECK_cuda(cudaFree(c_g));

    return 0;
}

// nvcc -o run MatrixAdd.cu
