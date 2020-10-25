#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#include "tsp_2opt.cuh"

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

__device__ static inline int divupg(int a, int b) {
    return (a + b - 1)/b;
}

__device__ inline int roundupg(int a, int b) {
    return divup(a, b) * b;
}

__device__ static inline int dist(int x1, int y1, int x2, int y2) {
    int dx = x1-x2;
    int dy = y1-y2;
    return dx*dx+dy*dy;
}

__global__ void two_opt_kernel_slow(const int* x, const int* y, int* best, int* best_j, int n) {
    int i = blockIdx.x;
    if (i >= n-2)
        return;
    if (i >= 1) {
        int bi = i, bj, bimpr = 0;
        for (int j = i+1; j < n-1; ++j) {
            int impr = dist(x[i], y[i], x[i-1], y[i-1]) + dist(x[j+1], y[j+1], x[j], y[j]) - 
            (dist(x[i], y[i], x[j+1], y[j+1]) + dist(x[i-1], y[i-1], x[j], y[j]));
            if (impr > bimpr) {
                bimpr = impr;
                bj = j;
            }
        }
        best[i] = bimpr;
        best_j[i] = bj;
    }
    __syncthreads();
    if (i == 0) {
        best[0] = 0;
        for (int i2 = 1; i2 < n-2; ++i2) {
            if (best[i2] > best[0]) {
                best[0] = best[i2];
                best_j[0] = best_j[i2];
                best_j[1] = i2;
            }
        }
    }
}


__global__ void reduce_kernel(int* best, int* best_i, int* best_j, int n) {
    int local_best = 0, local_best_i = 0, local_best_j = 0;
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        for (int k = 0; k < divupg(n, 64); ++k) {
            if (best[blockIdx.x*blockDim.x + k] > local_best) {
                local_best = best[blockIdx.x * blockDim.x + k];
                local_best_i = best_i[blockIdx.x * blockDim.x + k];
                local_best_j = best_j[blockIdx.x * blockDim.x + k];
            }
        }
        printf("paras2 %d %d %d\n", local_best, local_best_i, local_best_j);
        printf("paras %d %d %d %d, %d\n", local_best, threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.x * blockDim.x);
        best[blockIdx.x * blockDim.x] = local_best;
        best_i[blockIdx.x * blockDim.x] = local_best_i;
        best_j[blockIdx.x * blockDim.x] = local_best_j;
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0) {
        for (int k = 0; k < divupg(n, 64); ++k) {
            printf("lol %d %d\n", best[k*blockDim.x], k*blockDim.x);
            if (best[k*blockDim.x] > local_best) {
                local_best = best[k * blockDim.x];
                local_best_i = best_i[k * blockDim.x];
                local_best_j = best_j[k * blockDim.x];
            }
        }
        best[0] = local_best;
        best_i[0] = local_best_i;
        best_j[0] = local_best_j;
    }
}

__global__ void two_opt_kernel(const int* x, const int* y, int* return_best, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x+1;
    int j = blockIdx.y * blockDim.x;
    //printf("lol1 %d %d %d\n", threadIdx.x, blockIdx.x, blockIdx.y);
    if (j+threadIdx.x >= n)
        return;
    
    //printf("lol %d %d %d\n", i, j, n);
    __shared__ int shared_x[65];
    __shared__ int shared_y[65];
    __shared__ int shared_best[64];
    __shared__ int shared_j[64];
    shared_best[threadIdx.x] = 0;
    shared_j[threadIdx.x] = 0;
    shared_x[threadIdx.x] = x[j+threadIdx.x];
    //printf("ladataan %d %d %d %d %d %d\n", j+blockDim.x+1, i, n-2, x[j+threadIdx.x], j, j+threadIdx.x);
    shared_y[threadIdx.x] = y[j+threadIdx.x];
    if (threadIdx.x == blockDim.x-1) {
        shared_x[blockDim.x] = x[j+threadIdx.x+1];
        shared_y[blockDim.x] = y[j+threadIdx.x+1];
    }
    //printf("iffi %d %d %d\n", j+blockDim.x+1, i, n-2);
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
                    int k_dist = dist(shared_x[k], shared_y[k], shared_x[k+1], shared_y[k+1]) - 
                    (dist(xi, yi, shared_x[k+1], shared_y[k+1]) + dist(xim, yim, shared_x[k], shared_y[k]));
                    if (k_dist > best) {
                        best = k_dist;
                        best_j = j+k;
                    }
                }
            }
        } else {
            for (int k = 0; k < blockDim.x; ++k) {
                int k_dist = dist(shared_x[k], shared_y[k], shared_x[k+1], shared_y[k+1]) - 
                (dist(xi, yi, shared_x[k+1], shared_y[k+1]) + dist(xim, yim, shared_x[k], shared_y[k]));
                if (k_dist > best) {
                    best = k_dist;
                    best_j = j+k;
                }
            }
        }
        best = best + i_dist;
    }
    //printf("parase %d %d %d %d\n", best, i, best_j, blockIdx.x * blockDim.x + blockIdx.y);
    
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
        return_best[blockIdx.x * blockDim.x + blockIdx.y] = best;
        printf("paras2 %d %d %d %d\n", best, best_i, best_j, blockIdx.x * blockDim.x + blockIdx.y);
    }
    
}

__global__ void two_opt_swap_kernel(const int* x, const int* y, int i, int j) {
    printf("lol\n");

}

void two_opt_loop(const int* x, const int* y, int n) {
    int* xGPU = NULL;
    cudaMalloc((void**)&xGPU, n * sizeof(int));
    cudaMemcpy(xGPU, x, n * sizeof(int), cudaMemcpyHostToDevice);
    int* yGPU = NULL;
    cudaMalloc((void**)&yGPU, n * sizeof(int));
    cudaMemcpy(yGPU, y, n * sizeof(int), cudaMemcpyHostToDevice);
    int* bestGPU = NULL;
    cudaMalloc((void**)&bestGPU, divup(n, 64) * divup(n, 64) * sizeof(int));

    dim3 dimBlock(64, 1);
    dim3 dimGrid(divup(n, 64), divup(n, 64));
    printf("Block (%d %d), Grid(%d, %d)\n", 61, 1, divup(n, 64), divup(n, 64));
    do {
        two_opt_kernel<<<dimGrid, dimBlock>>>(xGPU, yGPU, bestGPU, n);
        //two_opt_swap_kernel<<<dimGrid, dimBlock>>>(xGPU, yGPU, 0, 0);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        int* best = (int*)malloc(3 * sizeof(int));
        cudaMemcpy(best, bestGPU, 3 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        printf("ohi %d %d %d\n", best[0], best[1], best[2]);
        break;
        
    } while(true);
    cudaFree(xGPU);
    cudaFree(yGPU);
}

void run_gpu_2opt(int* x, int* y, int n) {
    
    two_opt_loop(x, y, n);
    

}