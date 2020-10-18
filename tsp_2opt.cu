#include <stdio.h>
#include "tsp_2opt.cuh"
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

__global__ void two_opt_kernel(const int* x, const int* y, int* return_best, int* return_i, int* return_j, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x+1;
    int j = blockIdx.y * blockDim.x;
    //printf("lol1 %d %d %d\n", threadIdx.x, blockIdx.x, blockIdx.y);
    if (j+blockDim.x+1 <= i || i >= n-2)
        return;
    //printf("lol %d %d %d\n", i, j, n);
    __shared__ int shared_x[65];
    __shared__ int shared_y[65];
    __shared__ int shared_best[64];
    __shared__ int shared_j[64];
    shared_best[threadIdx.x] = 0;
    shared_j[threadIdx.x] = 0;
    shared_x[threadIdx.x] = x[j+threadIdx.x];
    shared_y[threadIdx.x] = y[j+threadIdx.x];
    if (threadIdx.x == blockDim.x-1) {
        shared_x[blockDim.x] = x[j+threadIdx.x+1];
        shared_y[blockDim.x] = y[j+threadIdx.x+1];
    }
    __syncthreads();
    int best = -1<<30, best_j = 0;
    if (i > 0) {
        int xi = x[i], xim = x[i-1], yi = y[i], yim = y[i-1];
        int i_dist = dist(xi, yi, xim, yim);
        if (j <= i || j + blockDim.x >= n) {
            for (int k = 0; k < blockDim.x; ++k) {
                if (j+k > i && j+k < n-1) {
                    int k_dist = dist(shared_x[k], shared_y[k], shared_x[k+1], shared_y[k+1]) - 
                    (dist(xi, xi, shared_x[k+1], shared_y[k+1]) + dist(xim, yim, shared_x[k], shared_y[k]));
                    if (k_dist > best) {
                        best = k_dist;
                        best_j = k;
                    }
                }
            }
        } else {
            for (int k = 0; k < blockDim.x; ++k) {
                int k_dist = dist(shared_x[k], shared_y[k], shared_x[k+1], shared_y[k+1]) - 
                (dist(xi, xi, shared_x[k+1], shared_y[k+1]) + dist(xim, yim, shared_x[k], shared_y[k]));
                if (k_dist > best) {
                    best = k_dist;
                    best_j = k;
                }
            }
        }
        best = best + i_dist;
    }
    
    
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
        return_i[blockIdx.x * blockDim.x + blockIdx.y] = best_i;
        return_j[blockIdx.x * blockDim.x + blockIdx.y] = best_j;
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        for (int k = 0; k < divupg(n, 64); ++k) {
            if (return_best[blockIdx.x*blockDim.x + k] > best) {
                best = return_best[blockIdx.x * blockDim.x + k];
                best_i = return_i[blockIdx.x * blockDim.x + k];
                best_j = return_j[blockIdx.x * blockDim.x + k];
            }
        }
        printf("paras %d %d %d %d, %d\n", best, threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.x * blockDim.x);
        return_best[blockIdx.x * blockDim.x] = best;
        return_i[blockIdx.x * blockDim.x] = best_i;
        return_j[blockIdx.x * blockDim.x] = best_j;
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0) {
        for (int k = 0; k < divupg(n, 64); ++k) {
            printf("lol %d %d\n", return_best[k*blockDim.x], k*blockDim.x);
            if (return_best[k*blockDim.x] > best) {
                best = return_best[k * blockDim.x];
                best_i = return_i[k * blockDim.x];
                best_j = return_j[k * blockDim.x];
            }
        }
        return_best[0] = best;
        return_i[0] = best_i;
        return_j[0] = best_j;
    }
}



__global__ void two_opt_swap_kernel(const int* x, const int* y, int i, int j) {
    
}

void two_opt_loop(const int* x, const int* y, int n) {
    int* xGPU = NULL;
    cudaMalloc((void**)&xGPU, n * sizeof(int));
    cudaMemcpy(xGPU, x, n * sizeof(int), cudaMemcpyHostToDevice);
    int* yGPU = NULL;
    cudaMalloc((void**)&yGPU, n * sizeof(int));
    cudaMemcpy(yGPU, y, n * sizeof(int), cudaMemcpyHostToDevice);
    int* bestGPU = NULL;
    cudaMalloc((void**)&bestGPU, n*n * sizeof(int));
    int* best_jGPU = NULL;
    cudaMalloc((void**)&best_jGPU, n*n * sizeof(int));
    int* best_iGPU = NULL;
    cudaMalloc((void**)&best_iGPU, n*n * sizeof(int));

    dim3 dimBlock(64, 1);
    dim3 dimGrid(divup(n, 64), divup(n, 64));
    do {
        two_opt_kernel<<<dimGrid, dimBlock>>>(xGPU, yGPU, bestGPU, best_iGPU, best_jGPU, n);
        cudaDeviceSynchronize();
        int* best = (int*)malloc(sizeof(int));
        cudaMemcpy(best, bestGPU, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        int* besti = (int*)malloc(sizeof(int));
        cudaMemcpy(besti, best_iGPU, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        int* bestj = (int*)malloc(sizeof(int));
        cudaMemcpy(bestj, best_jGPU, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        printf("paras %d %d %d\n", best[0], besti[0], bestj[0]);
        if (best[0] == 0)
            break;
        two_opt_swap_kernel<<<dimGrid, dimBlock>>>(x, y, best_jGPU[1], best_jGPU[0]);
    } while(true);
    cudaFree(xGPU);
    cudaFree(yGPU);
}

void run_gpu_2opt(int* x, int* y, int n) {
    
    two_opt_loop(x, y, n);
    

}