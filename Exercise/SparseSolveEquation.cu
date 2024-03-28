#include "error.cuh"
#include <time.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusparse.h>
#define pi 3.14159265358979323846

// Dense 转 CSR
// 输入：
//      h_A_dense: 稠密矩阵
//      m: 矩阵行数
//      n: 矩阵列数
// 输出：
//      h_A: CSR 矩阵
//      h_A_RowIndices: CSR 矩阵的行索引
//      h_A_ColIndices: CSR 矩阵的列索引
// 返回值：
//      nnz: 非零元个数
int DenseToCSR(double *h_A_dense, int m, int n,
                double *h_A, int *h_A_RowIndices, int *h_A_ColIndices)
{
    int nnz = 0;
    h_A_RowIndices[0] = 0;
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            if (h_A_dense[i*n+j] != 0){
                nnz++;
                h_A[nnz-1] = h_A_dense[i*n+j];
                h_A_ColIndices[nnz-1] = j;
            }
        }
        h_A_RowIndices[i+1] = nnz;
    }
    return nnz;
}

int main() {
    // --- Initialize cuSOLVER
    cusolverSpHandle_t cusolver_H;
    CHECK_cusolver(cusolverSpCreate(&cusolver_H));


    const int m = 16;
    const int n = 16;
    double *h_A_dense, *h_B, *h_X, *h_A;
    int *h_A_RowIndices, *h_A_ColIndices;
    h_A_dense      = (double *)malloc(m*n*sizeof(double));
    h_A            = (double *)malloc(m*n*sizeof(double));
    h_A_RowIndices = (int *)malloc((m+1)*sizeof(int));
    h_A_ColIndices = (int *)malloc(m*n*sizeof(int));
    h_B            = (double *)malloc(m*sizeof(double));
    h_X            = (double *)malloc(m*sizeof(double));

    // A 稠密矩阵赋值
    memset(h_A_dense,0,sizeof(double)*m*n);
    for (int i=1; i<m-1; i++){
            h_A_dense[i*n+i] = 2.0;
            h_A_dense[i*n+i+1] = -1.0;
            h_A_dense[i*n+i-1] = -1.0;
    }
    h_A_dense[0*n+0] = 2.0;         h_A_dense[0*n+1] = -1.0;
    h_A_dense[(m-1)*n+(n-1)] = 2.0; h_A_dense[(m-1)*n+(n-2)] = -1.0;
    
    // 右端项 b 赋值
    for (int i=0; i<m; i++){
        h_B[i] = sin(2.0*pi*(1.0*i)/(1.0*(n-1))) * (2.0*pi/(1.0*(n-1))) * (2.0*pi/(1.0*(n-1)));
    }

    // CPU 稠密矩阵稀疏化 CSR
    int nnz = DenseToCSR(h_A_dense, m, n,
                         h_A, h_A_RowIndices, h_A_ColIndices);

    // A_CSR, b ---- CPU to GPU
    // 分配显存
    double *d_A, *d_B, *d_X;
    int *d_A_RowIndices, *d_A_ColIndices;
    CHECK_cuda(cudaMalloc((void**)&d_A, nnz * sizeof(double)));
    CHECK_cuda(cudaMalloc((void**)&d_A_RowIndices, (m + 1) * sizeof(int)));
    CHECK_cuda(cudaMalloc((void**)&d_A_ColIndices, nnz * sizeof(int)));
    CHECK_cuda(cudaMalloc((void**)&d_B, m * sizeof(double)));
    CHECK_cuda(cudaMalloc((void**)&d_X, m * sizeof(double)));
    // Copy A,b to GPU
    CHECK_cuda(cudaMemcpy(d_A, h_A, nnz*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_cuda(cudaMemcpy(d_A_RowIndices, h_A_RowIndices, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_cuda(cudaMemcpy(d_A_ColIndices, h_A_ColIndices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_cuda(cudaMemcpy(d_B, h_B, m*sizeof(double), cudaMemcpyHostToDevice));

    // --- Descriptor for sparse matrix A (稀疏矩阵 A 的描述符)
    // 
    cusparseMatDescr_t descrA;
    CHECK_cusparse(cusparseCreateMatDescr(&descrA));
    CHECK_cusparse(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_cusparse(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    
    // 打印稀疏矩阵元素
/*     for (int i = 0; i < nnz; ++i) 
        printf("A[%i] = %.0f ", i, h_A[i]); printf("\n");
    for (int i = 0; i < (m + 1); ++i)
        printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");
    for (int i = 0; i < nnz; ++i)
        printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]); */
    
    
    // 官方链接: https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cusolver-api
    int singularity;
    /* --- Using QR factorization --- */
    // CHECK_cusolver(cusolver_status = cusolverSpDcsrlsvqr(cusolver_H, m, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_B, 0.000001, 0, d_X, &singularity));
    /* --- Using Cholesky factorization --- */
    CHECK_cusolver(cusolverSpDcsrlsvchol(cusolver_H, m, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_B, 0.000001, 0, d_X, &singularity));

    // 打印结果
    CHECK_cuda(cudaDeviceSynchronize());
    CHECK_cuda(cudaMemcpy(h_X, d_X, m * sizeof(double), cudaMemcpyDeviceToHost));
    printf("Showing the results...\n");
    for (int i = 0; i < m; i++)
        printf("%f\n", h_X[i]);
     
    
    // 释放内存
    CHECK_cusolver(cusolverSpDestroy(cusolver_H));
    CHECK_cusparse(cusparseDestroyMatDescr(descrA));
    // host dense
    free(h_A_dense);
    free(h_B);
    free(h_X);
    // host sparse
    free(h_A);
    free(h_A_RowIndices);
    free(h_A_ColIndices);
    // device sparse
    CHECK_cuda(cudaFree(d_A));
    CHECK_cuda(cudaFree(d_A_RowIndices));
    CHECK_cuda(cudaFree(d_A_ColIndices));
    CHECK_cuda(cudaFree(d_B));
    CHECK_cuda(cudaFree(d_X));

    return 0;
}

// 编译：nvcc -o run SparseSolveEquation.cu -lcusolver -lcusparse
// matlab 验证：
/* n = 16;
i = [0:1:n-1].';
h = 2*pi/(n-1);
x = h*i;
f = sin(x) * h *h;
a = 2*ones(n,1);
b = -1*ones(n-1,1);
A = diag(a,0) + diag(b,-1) + diag(b,1);
u = A\f;
plot(x,u); */
