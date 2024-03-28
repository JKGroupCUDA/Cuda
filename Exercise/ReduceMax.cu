#include "error.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
using namespace std;


// 归约求最大值
__global__ void reduceMax(double* input, double* output, int n)
{
    __shared__ double sha_partialMax[1024];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n)
    {
        sha_partialMax[threadIdx.x] = input[tid];
    }
    else
    {
        sha_partialMax[threadIdx.x] = 0.0;
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
        double max = 0.0;
        for(int i = 0; i < 1024; i++)
        {
            if(sha_partialMax[i] > max)
            {
                max = sha_partialMax[i];
            }
        }
        output[blockIdx.x] = max;
    }
}


int main()
{
    int n = 1025;
    double *a;
    a = (double*)malloc(n*sizeof(double));
    for (int i = 0; i < n; i++)
    {
        a[i] = 0.1*i;
    }

    // 分2块，每块1024
    dim3 block(1024,1);
    dim3 grid((n -1)/block.x +1,1);

    double *a_g, *b_g;
    CHECK_cuda(cudaMalloc((void**)&a_g, sizeof(double)*n));
    CHECK_cuda(cudaMalloc((void**)&b_g, sizeof(double)*grid.x));
    CHECK_cuda(cudaMemcpy(a_g, a, sizeof(double)*n, cudaMemcpyHostToDevice));

    reduceMax<<<grid, block>>>(a_g, b_g, n);

    double *b;
    b = (double*)malloc(grid.x*sizeof(double));
    CHECK_cuda(cudaMemcpy(b, b_g, sizeof(double)*grid.x, cudaMemcpyDeviceToHost));

    cout << "b:";
    for (int i = 0; i < grid.x; i++)
    {
        cout <<  b[i] << " ";
    }
    cout << endl;
    
    double maxvalue = 0.0;
    for (int i = 0; i < grid.x; i++)
    {
        if (maxvalue < b[i])
        {
            maxvalue = b[i];
        }
    }
    cout << "Max value: " << maxvalue << endl;

    free(a);
    free(b);
    CHECK_cuda(cudaFree(a_g));
    CHECK_cuda(cudaFree(b_g));
    return 0;
}

// nvcc -o run ReduceMax.cu
