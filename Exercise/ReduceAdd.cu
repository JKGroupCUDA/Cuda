#include "error.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void set_indata(double *idata_g,int n){

    int i=blockDim.x*blockIdx.x+threadIdx.x;

    if(i<n){
        idata_g[i]=1.0;
    }
}

__global__ void reduce2(double *idata_g, int n) {
    extern __shared__ double sdata[];
    // each thread loads one element from global to shared mem
    // tid = 0 ~ 1023 每一块的局部指标
    unsigned int tid = threadIdx.x;
    // i = 0 ~ (n-1) 大数组的全局指标
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    // 每一块赋给 sdata
    if(i<n){
        sdata[tid] = idata_g[i];
    }else{
        sdata[tid]=0.0;
    }
    __syncthreads();
    // do reduction in shared mem
    // s=512, tid=0~511; s=256, tid=0~255
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0){ 
        idata_g[blockIdx.x] = sdata[0];
    }
}

int main(){
    int n0 = pow(2,20);
    double *idata_g;
    CHECK_cuda(cudaMalloc((void**)&idata_g,sizeof(double)*n0));


    // 每块1024
    dim3 block(1024,1);
    dim3 grid((n0 -1)/block.x +1,1);
    set_indata<<<grid, block>>>(idata_g, n0);

    // 用于归约求和的临时变量
    // Copy idata_g to tmp_g
    int n = n0;
    double *tmp_g;
    CHECK_cuda(cudaMalloc((void**)&tmp_g,sizeof(double)*n));
    CHECK_cuda(cudaMemcpy(tmp_g,idata_g,sizeof(double)*n,cudaMemcpyDeviceToDevice));

    // 循环归约求和
    while (n > 1){
        // 分 n1 块，每块1024个线程
        int n1 = (n-1)/1024+1;
        // 归约求和
        reduce2<<<n1,1024,2048*sizeof(double)>>>(tmp_g,n);
        // 归约求和后的结果存入 tmp_g
        // 替换变量，进入下一次循环
        n = n1;
    }

    CHECK_cuda(cudaDeviceSynchronize());
    double rslt;
    CHECK_cuda(cudaMemcpy(&rslt,tmp_g,sizeof(double)*1,cudaMemcpyDeviceToHost));
    printf("n: %d\n",n0);
    printf("sum: %.1f\n",rslt);

    CHECK_cuda(cudaFree(idata_g));
    CHECK_cuda(cudaFree(tmp_g));
    return 0;
}

// nvcc -o run ReduceAdd.cu
