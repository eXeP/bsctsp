#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <float.h>

#include "cuda_2opt.cuh"
#include "util.cuh"


struct best_struct {
    float best;
    int i;
    int j;
};

__global__ void two_opt_kernel(const float* x, const float* y, best_struct* return_best, int n, int* lock) {
    int i = threadIdx.x + blockIdx.x * blockDim.x+1;
    int j = blockIdx.y * blockDim.x;
    __shared__ float shared_x[65];
    __shared__ float shared_y[65];
    __shared__ float shared_best[64];
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
    float best = 0;
    int best_j = 0;
    if (i > 0) {
        float xi = x[i], xim = x[i-1], yi = y[i], yim = y[i-1];
        float i_dist = dist(xi, yi, xim, yim);
        if (j <= i || j + blockDim.x >= n) {
            for (int k = 0; k < blockDim.x; ++k) {
                if (j+k > i && j+k < n-1) {
                    float k_dist = i_dist + dist(shared_x[k], shared_y[k], shared_x[k+1], shared_y[k+1]) - 
                    (dist(xi, yi, shared_x[k+1], shared_y[k+1]) + dist(xim, yim, shared_x[k], shared_y[k]));
                    if (k_dist > best) {
                        best = k_dist;
                        best_j = j+k;
                    }
                }
            }
        } else {
            for (int k = 0; k < blockDim.x; ++k) {
                float k_dist = i_dist + dist(shared_x[k], shared_y[k], shared_x[k+1], shared_y[k+1]) - 
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
        if (best > return_best[0].best) {
            while (atomicExch(&lock[0], 1) != 0);
            if (best > return_best[0].best) {
                return_best[0].best = best;
                return_best[0].i = best_i;
                return_best[0].j = best_j;
            }
            lock[0] = 0;
            __threadfence();
        }
    }
}

__global__ void two_opt_swap_kernel(float* x, float* y, int i, int j) {
    int k = blockIdx.x;
    float tmp = x[i+k];
    x[i+k] = x[j-k];
    x[j-k] = tmp;
    tmp = y[i+k];
    y[i+k] = y[j-k];
    y[j-k] = tmp;
}

void run_gpu_2opt(float* x, float* y, int n) {
    float* xGPU = NULL;
    cudaMalloc((void**)&xGPU, n * sizeof(float));
    cudaMemcpy(xGPU, x, n * sizeof(float), cudaMemcpyHostToDevice);
    float* yGPU = NULL;
    cudaMalloc((void**)&yGPU, n * sizeof(float));
    cudaMemcpy(yGPU, y, n * sizeof(float), cudaMemcpyHostToDevice);
    best_struct* bestGPU = NULL;
    cudaMalloc((void**)&bestGPU, sizeof(best_struct));
    int* lock = NULL;
    cudaMalloc((void**)&lock, 1 * sizeof(int));

    dim3 dimBlock(64, 1);
    dim3 dimGrid(divup(n, 64), divup(n, 64));
    do {
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaMemset(bestGPU, 0, sizeof(best_struct));
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaMemset(lock, 0, 1*sizeof(int));
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        two_opt_kernel<<<dimGrid, dimBlock>>>(xGPU, yGPU, bestGPU, n, lock);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        best_struct* best = (best_struct*)malloc(sizeof(best_struct));
        cudaMemcpy(best, bestGPU, sizeof(best_struct), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (abs(best[0].best) < 0.000001)
            break;
        //printf("Improvement %f %d %d\n", best[0].best, best[0].i, best[0].j);
        two_opt_swap_kernel<<<dim3((best[0].j-best[0].i+1)/2, 1), dim3(1, 1)>>>(xGPU, yGPU, best[0].i, best[0].j);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
    } while(true);

    cudaMemcpy(x, xGPU, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, yGPU, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(xGPU);
    cudaFree(yGPU);
    cudaFree(bestGPU);
    cudaFree(lock);
}

__global__ void two_opt_swap_kernel_id(float* x, float* y, int* id, int i, int j) {
    int k = blockIdx.x;
    float tmp1 = x[i+k];
    x[i+k] = x[j-k];
    x[j-k] = tmp1;
    float tmp2 = y[i+k];
    y[i+k] = y[j-k];
    y[j-k] = tmp2;
    int tmp3 = id[i+k];
    id[i+k] = id[j-k];
    id[j-k] = tmp3;
}

__global__ void id_map_calculate(int* id, int* id_map) {
    int k = blockIdx.x;
    id_map[id[k]] = k;
}

__global__ void two_opt_kernel_restricted(const float* x, const float* y, const int* moves, const int* id, const int* id_map, best_struct* return_best, int n, const int allowed_moves, int* lock) {
    int i = blockIdx.x;
    const int* i_moves = moves + i * allowed_moves;
    float best = 0;
    int best_j = 0;
    if (i > 0) {
        float xi = x[i], xim = x[i-1], yi = y[i], yim = y[i-1];
        float i_dist = dist(xi, yi, xim, yim);
        int id_i = id[i];
        for (int k = 0; k < allowed_moves; ++k) {
            int id_j = i_moves[k];
            int j = id_map[id_j];
            if (j != 0 && j != n-1) {
                float k_dist = i_dist + dist(x[j], y[j], x[j+1], y[j+1]) - 
                (dist(xi, yi, x[j+1], y[j+1]) + dist(xim, yim, x[j], y[j]));
                if (k_dist > best) {
                    best = k_dist;
                    best_j = j+k;
                }
            }
        }
    }

    __syncthreads();
    if (best > return_best[0].best) {
        while (atomicExch(&lock[0], 1) != 0);
        if (best > return_best[0].best) {
            return_best[0].best = best;
            return_best[0].i = i;
            return_best[0].j = best_j;
        }
        lock[0] = 0;
        __threadfence();
    }
}

void run_gpu_2opt_restricted(float* x, float* y, int* id, int* moves, int n, int allowed_moves) {
    float* xGPU = NULL;
    cudaMalloc((void**)&xGPU, n * sizeof(float));
    cudaMemcpy(xGPU, x, n * sizeof(float), cudaMemcpyHostToDevice);
    float* yGPU = NULL;
    cudaMalloc((void**)&yGPU, n * sizeof(float));
    cudaMemcpy(yGPU, y, n * sizeof(float), cudaMemcpyHostToDevice);
    best_struct* bestGPU = NULL;
    cudaMalloc((void**)&bestGPU, sizeof(best_struct));
    int* lock = NULL;
    cudaMalloc((void**)&lock, 1 * sizeof(int));

    int* Gmoves = NULL;
    cudaMalloc((void**)&Gmoves, n * allowed_moves * sizeof(int));
    cudaMemcpy(Gmoves, moves, n * allowed_moves * sizeof(float), cudaMemcpyHostToDevice);

    int* Gid = NULL;
    cudaMalloc((void**)&Gid, n * sizeof(int));
    cudaMemcpy(Gid, id, n * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<int> id_map(n);
    for (int i = 0; i < n; ++i)
        id_map[id[i]] = i;

    int* Gid_map = NULL;
    cudaMalloc((void**)&Gid_map, n * sizeof(int));
    cudaMemcpy(Gid_map, id_map.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(64, 1);
    dim3 dimGrid(divup(n, 64), divup(n, 64));
    do {
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaMemset(bestGPU, 0, sizeof(best_struct));
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaMemset(lock, 0, 1*sizeof(int));
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        two_opt_kernel_restricted<<<dimGrid, dimBlock>>>(xGPU, yGPU, Gmoves, Gid, Gid_map, bestGPU, n, allowed_moves, lock);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        best_struct* best = (best_struct*)malloc(sizeof(best_struct));
        cudaMemcpy(best, bestGPU, sizeof(best_struct), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (abs(best[0].best) < 0.000001)
            break;
        //printf("Improvement %f %d %d\n", best[0].best, best[0].i, best[0].j);
        two_opt_swap_kernel_id<<<dim3((best[0].j-best[0].i+1)/2, 1), dim3(1, 1)>>>(xGPU, yGPU, Gid, best[0].i, best[0].j);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        id_map_calculate<<<dim3(n, 1), dim3(1, 1)>>>(Gid, Gid_map);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
    } while(true);

    cudaMemcpy(x, xGPU, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, yGPU, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(id, Gid, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(xGPU);
    cudaFree(yGPU);
    cudaFree(bestGPU);
    cudaFree(lock);
    cudaFree(Gmoves);
    cudaFree(Gid);
}