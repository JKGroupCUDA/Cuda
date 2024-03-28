#include "error.cuh"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#define pi 3.14159265358979323846

__global__ void set_data(cufftDoubleComplex *data, int size, double a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    data[i].x = a*sin((1.0*i)/(1.0*size)*2.0*pi);
    data[i].y = 0.0;
}

void fft(cufftHandle plan, cufftDoubleComplex *in, cufftDoubleComplex *out, size_t size)
{
    // 执行 fft, 正向变换
    cufftExecZ2Z(plan, in, out, CUFFT_FORWARD);
}


void ifft(cufftHandle plan, cufftDoubleComplex *in, cufftDoubleComplex *out, size_t size)
{
    // 执行 ifft, 逆向变换
    cufftExecZ2Z(plan, in, out, CUFFT_INVERSE);
}


// 定义卷积函数
__global__ void convolve(cufftDoubleComplex *u1_f,
                         cufftDoubleComplex *u2_f,
                         cufftDoubleComplex *u3_f,
                         cufftDoubleComplex *rslt_f,
                         int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }

    cufftDoubleComplex tmp2;
    cufftDoubleComplex tmp3;

    // tmp2 = u1_f .* u2_f
    tmp2.x = u2_f[i].x * u1_f[i].x - u2_f[i].y * u1_f[i].y;
    tmp2.y = u2_f[i].x * u1_f[i].y + u2_f[i].y * u1_f[i].x;
    // tmp3 = tmp2 .* u3_f
    tmp3.x = tmp2.x * u3_f[i].x - tmp2.y * u3_f[i].y;
    tmp3.y = tmp2.x * u3_f[i].y + tmp2.y * u3_f[i].x;

    // rslt_f = (tmp2 + tmp3)/N
    rslt_f[i].x = (tmp2.x + tmp3.x)/(1.0*size);
    rslt_f[i].y = (tmp2.y + tmp3.y)/(1.0*size);
}


int main() {
    const int size = 16;
    cufftDoubleComplex *u1,   *u2,   *u3,   *rslt,   *output;
    cufftDoubleComplex *u1_f, *u2_f, *u3_f, *rslt_f;
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_Z2Z, 1);
    
    // GPU 分配内存
    CHECK_cuda(cudaMalloc((void**)&u1,   sizeof(cufftDoubleComplex) * size));
    CHECK_cuda(cudaMalloc((void**)&u2,   sizeof(cufftDoubleComplex) * size));
    CHECK_cuda(cudaMalloc((void**)&u3,   sizeof(cufftDoubleComplex) * size));
    CHECK_cuda(cudaMalloc((void**)&rslt, sizeof(cufftDoubleComplex) * size));
    CHECK_cuda(cudaMalloc((void**)&u1_f, sizeof(cufftDoubleComplex) * size));
    CHECK_cuda(cudaMalloc((void**)&u2_f, sizeof(cufftDoubleComplex) * size));
    CHECK_cuda(cudaMalloc((void**)&u3_f, sizeof(cufftDoubleComplex) * size));
    CHECK_cuda(cudaMalloc((void**)&rslt_f, sizeof(cufftDoubleComplex) * size));

    // CPU分配内存
    output = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * size);


    // 每块16
    dim3 block(16,1);
    dim3 grid((size -1)/block.x +1, 1);

    set_data<<<grid, block>>>(u1, size, 1.0);
    set_data<<<grid, block>>>(u2, size, 2.0);
    set_data<<<grid, block>>>(u3, size, 3.0);


    // fft
    fft(plan, u1, u1_f, size);
    fft(plan, u2, u2_f, size);
    fft(plan, u3, u3_f, size);

    // 执行卷积, 乘积, 线性求和, /N
    convolve<<<grid, block>>>(u1_f, u2_f, u3_f, rslt_f, size);


    // ifft
    ifft(plan, rslt_f, rslt, size);

    // 输出结果
    CHECK_cuda(cudaDeviceSynchronize());
    CHECK_cuda(cudaMemcpy(output, rslt,sizeof(cufftDoubleComplex)*size,cudaMemcpyDeviceToHost));

    // 打印卷积结果
    for (int i = 0; i < size; i++) {
        printf("output[%d] = %lf\n", i, output[i].x);
    }

    cufftDestroy(plan);
    CHECK_cuda(cudaFree(u1));
    CHECK_cuda(cudaFree(u2));
    CHECK_cuda(cudaFree(u3));
    CHECK_cuda(cudaFree(rslt));
    CHECK_cuda(cudaFree(u1_f));
    CHECK_cuda(cudaFree(u2_f));
    CHECK_cuda(cudaFree(u3_f));
    CHECK_cuda(cudaFree(rslt_f));
    free(output);
    return 0;
}

    // 编译
    // nvcc -o run Convolve1D.cu -lcufft

    /* matlab 验证 */
    /* % matlab 参考链接：https://zhuanlan.zhihu.com/p/504286492
    % 理论推导：https://www.zhihu.com/question/340004682
    size = 16;
    x = [0:1:size-1]/size*2*pi;
    u1 = sin(x);
    u2 = 2*sin(x);
    u3 = 3*sin(x);
    u1_f = fft(u1);
    u2_f = fft(u2);
    u3_f = fft(u3);
    u = ifft(u1_f .* u2_f + u1_f .* u2_f .* u3_f);
    */
    
