#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <float.h>

#include "tsp_2opt.cuh"
#include "util.cuh"


__global__ void two_opt_kernel(const int* x, const int* y, int* return_best, int n, int* lock) {
    int i = threadIdx.x + blockIdx.x * blockDim.x+1;
    int j = blockIdx.y * blockDim.x;
    __shared__ int shared_x[65];
    __shared__ int shared_y[65];
    __shared__ int shared_best[64];
    __shared__ int shared_j[64];
    shared_best[threadIdx.x] = 0;
    shared_j[threadIdx.x] = 0;
    if (j+threadIdx.x < n) {
        shared_x[threadIdx.x] = x[j+threadIdx.x];
        shared_y[threadIdx.x] = y[j+threadIdx.x];
    }
    if (threadIdx.x == blockDim.x-1) {
        shared_x[blockDim.x] = x[j+threadIdx.x+1];
        shared_y[blockDim.x] = y[j+threadIdx.x+1];
    }
    if (j+blockDim.x+1 <= i || i >= n-2)
        return;
    __syncthreads();
    int best = 0, best_j = 0;
    if (i > 0) {
        int xi = x[i], xim = x[i-1], yi = y[i], yim = y[i-1];
        int i_dist = dist(xi, yi, xim, yim);
        if (j <= i || j + blockDim.x >= n) {
            for (int k = 0; k < blockDim.x; ++k) {
                if (j+k > i && j+k < n-1) {
                    
                    int k_dist = i_dist + dist(shared_x[k], shared_y[k], shared_x[k+1], shared_y[k+1]) - 
                    (dist(xi, yi, shared_x[k+1], shared_y[k+1]) + dist(xim, yim, shared_x[k], shared_y[k]));
                    if (k_dist > best) {
                        best = k_dist;
                        best_j = j+k;
                    }
                }
            }
        } else {
            for (int k = 0; k < blockDim.x; ++k) {
                int k_dist = i_dist + dist(shared_x[k], shared_y[k], shared_x[k+1], shared_y[k+1]) - 
                (dist(xi, yi, shared_x[k+1], shared_y[k+1]) + dist(xim, yim, shared_x[k], shared_y[k]));
                if (k_dist > best) {
                    best = k_dist;
                    best_j = j+k;
                }
            }
        }
    }

    int out_w = divupg(n, 64);
    shared_best[threadIdx.x] = best;
    shared_j[threadIdx.x] = best_j;
    int best_i = i;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i2 = 0; i2 < blockDim.x; ++i2) {
            if (i + i2 >= n-2)
                continue;
            if (shared_best[i2] > best) {
                best_i = i+i2;
                best = shared_best[i2];
                best_j = shared_j[i2];
            }
        }
        if (best > return_best[0]) {
            while (atomicExch(&lock[0], 1) != 0);
            if (best > return_best[0]) {
                return_best[0] = best;
                return_best[1] = best_i;
                return_best[2] = best_j;
            }
            lock[0] = 0;
            __threadfence();
        }
    }
}

__global__ void two_opt_swap_kernel(int* x, int* y, int i, int j) {
    int k = blockIdx.x;
    int tmp = x[i+k];
    x[i+k] = x[j-k];
    x[j-k] = tmp;
    tmp = y[i+k];
    y[i+k] = y[j-k];
    y[j-k] = tmp;
}

void run_gpu_2opt(int* x, int* y, int n) {
    int* xGPU = NULL;
    cudaMalloc((void**)&xGPU, n * sizeof(int));
    cudaMemcpy(xGPU, x, n * sizeof(int), cudaMemcpyHostToDevice);
    int* yGPU = NULL;
    cudaMalloc((void**)&yGPU, n * sizeof(int));
    cudaMemcpy(yGPU, y, n * sizeof(int), cudaMemcpyHostToDevice);
    int* bestGPU = NULL;
    cudaMalloc((void**)&bestGPU, 3 * sizeof(int));
    int* lock = NULL;
    cudaMalloc((void**)&lock, 1 * sizeof(int));

    dim3 dimBlock(64, 1);
    dim3 dimGrid(divup(n, 64), divup(n, 64));
    printf("Block (%d %d), Grid(%d, %d)\n", 61, 1, divup(n, 64), divup(n, 64));
    int steps = 0;
    do {
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaMemset(bestGPU, 0, 1*sizeof(int));
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaMemset(lock, 0, 1*sizeof(int));
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        two_opt_kernel<<<dimGrid, dimBlock>>>(xGPU, yGPU, bestGPU, n, lock);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        int* best = (int*)malloc(3 * sizeof(int));
        cudaMemcpy(best, bestGPU, 3 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (best[0] == 0)
            break;
        printf("Improvement %d %d %d\n", best[0], best[1], best[2]);
        two_opt_swap_kernel<<<dim3((best[2]-best[1]+1)/2, 1), dim3(1, 1)>>>(xGPU, yGPU, best[1], best[2]);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();

        ++steps;
        //int tmp;
        //std::cin >> tmp;
    } while(true);
    cudaFree(xGPU);
    cudaFree(yGPU);
}