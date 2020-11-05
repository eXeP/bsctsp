#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <float.h>
#include <bits/stdc++.h>

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

static __device__ int lock_d;
static __global__ void Init()
{
  lock_d = 0;
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

__global__ void two_opt_kernel(const int* x, const int* y, int* return_best, int n, int* lock, bool dbg) {
    int i = threadIdx.x + blockIdx.x * blockDim.x+1;
    int j = blockIdx.y * blockDim.x;
    //printf("lol1 %d %d %d\n", threadIdx.x, blockIdx.x, blockIdx.y);
    //printf("lol %d %d %d\n", i, j, n);
    __shared__ int shared_x[65];
    __shared__ int shared_y[65];
    __shared__ int shared_best[64];
    __shared__ int shared_j[64];
    shared_best[threadIdx.x] = 0;
    shared_j[threadIdx.x] = 0;
    if (j+threadIdx.x < n) {
        shared_x[threadIdx.x] = x[j+threadIdx.x];
        //printf("ladataan %d %d %d %d %d %d\n", j+blockDim.x+1, i, n-2, x[j+threadIdx.x], j, j+threadIdx.x);
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
        //return_best[3 * (blockIdx.x * out_w + blockIdx.y) + 0] = best;
        //return_best[3 * (blockIdx.x * out_w + blockIdx.y) + 1] = best_i;
        //return_best[3 * (blockIdx.x * out_w + blockIdx.y) + 2] = best_j;
        //printf("paras2 %d %d %d %d %d\n", best, best_i, best_j, blockIdx.x * out_w + blockIdx.y, out_w);
        if (best > return_best[0]) {
            while (atomicExch(&lock_d, 1) != 0);
            if (best > return_best[0]) {
                return_best[0] = best;
                return_best[1] = best_i;
                return_best[2] = best_j;
            }
            lock_d = 0;  // release
            __threadfence();
        }
    }
}

__global__ void two_opt_reduce_kernel(int* best_global, int n) {
    //printf("%d %d\n", threadIdx.x, blockDim.x);
    int i = threadIdx.x;
    int best = 0, best_i, best_j;
    for (int j = 0; j < n; ++j) {
        if (best_global[3 * (i * blockDim.x + j)] > best) {
            best = best_global[3 * (i * blockDim.x  + j)];
            best_i = best_global[3 * (i * blockDim.x + j) + 1];
            best_j = best_global[3 * (i * blockDim.x + j) + 2];
        }
    }
    int out_w = blockDim.x;
    best_global[3 * (i * out_w)] = best;
    best_global[3 * (i * out_w) + 1] = best_i;
    best_global[3 * (i * out_w) + 2] = best_j;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i2 = 0; i2 < n; ++i2) {
            if (best_global[3 * (i2 * blockDim.x)] > best) {
                best = best_global[3 * (i2 * blockDim.x)];
                best_i = best_global[3 * (i2 * blockDim.x) + 1];
                best_j = best_global[3 * (i2 * blockDim.x) + 2];
            }
        }
        best_global[0] = best;
        best_global[1] = best_i;
        best_global[2] = best_j;
    }
}


__global__ void two_opt_swap_kernel(int* x, int* y, int i, int j) {
    int k = blockIdx.x;
    //printf("swapping %d %d %d %d %d %d\n", i+k, j-k, x[i+k], x[j-k], y[i+k], y[j-k]);
    int tmp = x[i+k];
    x[i+k] = x[j-k];
    x[j-k] = tmp;
    tmp = y[i+k];
    y[i+k] = y[j-k];
    y[j-k] = tmp;
}

void two_opt_loop(const int* x, const int* y, int n) {
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
        Init<<<1, 1>>>();
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaMemset(bestGPU, 0, 1*sizeof(int));
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaMemset(lock, 0, 1*sizeof(int));
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        two_opt_kernel<<<dimGrid, dimBlock>>>(xGPU, yGPU, bestGPU, n, lock, steps == 198);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        /*two_opt_reduce_kernel<<<dim3(1, 1), dim3(divup(n, 64), 1)>>>(bestGPU, divup(n, 64));
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();*/
        int* best = (int*)malloc(3 * sizeof(int));
        cudaMemcpy(best, bestGPU, 3 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (best[0] == 0)
            break;
        printf("Improvement %d %d %d\n", best[0], best[1], best[2]);
        two_opt_swap_kernel<<<dim3((best[2]-best[1]+1)/2, 1), dim3(1, 1)>>>(xGPU, yGPU, best[1], best[2]);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        /*int* xdbg = (int*)malloc(n * sizeof(int));
        cudaMemcpy(xdbg, xGPU, n * sizeof(int), cudaMemcpyDeviceToHost);
        if (steps == 197) {
            for (int i = 0; i < n; ++i)
                std::cout << xdbg[i] << " ";
            std::cout << std::endl;
        }*/

        ++steps;
        //int tmp;
        //std::cin >> tmp;
    } while(true);
    cudaFree(xGPU);
    cudaFree(yGPU);
}

void run_gpu_2opt(int* x, int* y, int n) {
    two_opt_loop(x, y, n);
}

__global__ void boruvka_smallest_kernel(int n, float* x, float* y, float* pi, int* component, float* component_best, int* component_best_i, int* component_best_j, int* component_lock, int excluded_vertex) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = blockIdx.y * blockDim.x;
    __shared__ float shared_x[64];
    __shared__ float shared_y[64];
    __shared__ float shared_pi[64];
    __shared__ float shared_best[64];
    __shared__ float shared_best_j[64];
    __shared__ float shared_component[64];
    __shared__ float shared_component_i[64];
    if (j+threadIdx.x < n) {
        shared_x[threadIdx.x] = x[j+threadIdx.x];
        shared_y[threadIdx.x] = y[j+threadIdx.x];
        shared_component[threadIdx.x] = component[j+threadIdx.x];
        shared_pi[threadIdx.x] = pi[j+threadIdx.x];
    }
    if (i < n)
        shared_component_i[threadIdx.x] = component[i];
    shared_best[threadIdx.x] = FLT_MAX;
    shared_best_j[threadIdx.x] = 0;
    __syncthreads();
    if (i >= n)
        return;
    float best = FLT_MAX;
    int best_j = -1;
    int component_i = component[i];
    if (!(i == excluded_vertex || i >= n)) {
        float pi_i = pi[i], x_i = x[i], y_i = y[i];
        if (j + blockDim.x > n || (excluded_vertex >= j && excluded_vertex < j + blockDim.x) || (i >= j && i < j + blockDim.x)) {
            for (int k = 0; k < 64; ++k) {
                if (j+k == excluded_vertex || j+k >= n || shared_component[k] == component_i || j+k == i)
                    continue;
                float d_ij = pi_i + shared_pi[k];
                d_ij += (x_i - shared_x[k]) * (x_i - shared_x[k]) + (y_i - shared_y[k]) * (y_i - shared_y[k]);
                if (d_ij < best) {
                    best = d_ij;
                    best_j = j+k;
                }
            }
        } else {
            for (int k = 0; k < 64; ++k) {
                if (shared_component[k] == component_i)
                    continue;
                float d_ij = pi_i + shared_pi[k];
                d_ij += (x_i - shared_x[k]) * (x_i - shared_x[k]) + (y_i - shared_y[k]) * (y_i - shared_y[k]);
                if (d_ij < best) {
                    best = d_ij;
                    best_j = j+k;
                }
            }
        }
    }
    shared_best[threadIdx.x] = best;
    shared_best_j[threadIdx.x] = best_j;
    __syncthreads();
    //printf("paras %d %d %f comp %d %d\n", i, best_j, best, component_i, component[best_j]);
    if (threadIdx.x == 0) {
        for (int k = 0; k < 64; ++k) {
            int tmp_i = blockIdx.x * blockDim.x + k;
            if (tmp_i >= n || tmp_i == excluded_vertex)
                continue;
            component_i = shared_component_i[k];
            best = shared_best[k];
            best_j = shared_best_j[k];
            //Mita jos komponentit osoittavat toisiina ja floatin takia indeksit eri, vaikka pitäisi symmetrisyyden takia olla samat?
            //jätetään myöhemmäksi tämän miettiminen
            if (best < component_best[component_i] || best == component_best[component_i]) {
                while (atomicExch(&component_lock[component_i], 1) != 0);
                if (best < component_best[component_i]) {
                    //printf("laitetaan %d %d %.2f %d\n", i, component_i, best, best_j);
                    component_best[component_i] = best;
                    component_best_i[component_i] = tmp_i;
                    component_best_j[component_i] = best_j;
                } else if (best == component_best[component_i]) {
                    if (tmp_i < component_best_i[component_i]) {
                        component_best_i[component_i] = tmp_i;
                        component_best_j[component_i] = best_j;
                    } else if (tmp_i == component_best_i[component_i] && best_j < component_best_j[component_i]) {
                        component_best_i[component_i] = tmp_i;
                        component_best_j[component_i] = best_j;
                    }
                }
                component_lock[component_i] = 0;  // release
                __threadfence();
            }
        }
    }
}

__global__ void boruvka_update_components(int n, 
    int* component, int* successor, float* component_best, int* component_best_i, int* component_best_j, int* component_lock, 
    int* degrees, float* L_T, int* components, int excluded_vertex) {
    int i = blockIdx.x;
    if (i == excluded_vertex)
        return;
    int component_i = component[i];
    int component_j = component[component_best_j[component_i]];
    
    int component_min = min(component_i, component_j);
    component[i] = successor[i];
    //printf("update %d %d->%d %d\n", i, component_i, component[i], component_j);
    int vertex_ii = component_best_i[component_i];
    int vertex_ji = component_best_i[component_j];
    int vertex_ij = component_best_j[component_i];
    int vertex_jj = component_best_j[component_j];
    //printf("yhdist %d %d %d %d\n", component_i, component_j, vertex_ii, vertex_ij);
    if (i == vertex_ii) {
        //printf("se vertex %d %d\n", i, vertex_jj);
        //Cycle!
        if (vertex_jj == i) {
            if (i < vertex_ij) {
                //printf("valitaan %d-%d %.2f %d-%d\n", i, vertex_ij, component_best[component_i], blockIdx.x, blockIdx.y, component_i, component_j);
                atomicSub(components, 1);
                atomicAdd(L_T, component_best[component_i]);
            }
        } else {
            //printf("valitaan %d-%d %.2f %d-%d\n", i, vertex_ij, component_best[component_i], blockIdx.x, blockIdx.y, component_i, component_j);
            atomicSub(components, 1);
            atomicAdd(L_T, component_best[component_i]);
            atomicAdd(&degrees[i], 1);
            atomicAdd(&degrees[vertex_ij], 1);
        }
        __threadfence();
    }
}

__global__ void boruvka_remove_cycles(int n, 
    int* component, float* component_best, int* component_best_i, int* component_best_j, int* component_lock, 
    int* degrees, float* L_T, int* components, int excluded_vertex) {
    int i = blockIdx.x;
    if (i == excluded_vertex)
        return;
    int component_i = component[i];
    int component_j = component[component_best_j[component_i]];

    int vertex_ii = component_best_i[component_i];
    int vertex_ji = component_best_i[component_j];
    int vertex_ij = component_best_j[component_i];
    int vertex_jj = component_best_j[component_j];
    if (i == vertex_ii) {
        if (vertex_jj == i) {
            if (i < vertex_ij) {
                component_best_j[component_i] = i;
            }
        }
    }
}

__global__ void boruvka_pointer_doubling(int n, 
    int* component, float* component_best, int* component_best_i, int* component_best_j, int* component_lock, 
    int* degrees, float* L_T, int* components, int excluded_vertex) {
    int i = blockIdx.x;
    if (i == excluded_vertex)
        return;
    component[i] = component[component_best_j[component[i]]];
    component[i] = component[component_best_j[component[i]]];
}

__global__ void excluded_vertex_add(int n, 
    float* x, float* y, float* pi, int excluded_vertex, float* closest, int* closest_i, int* lock) {
    int i = blockIdx.x;
    if (i == excluded_vertex)
        return;
    float d_ij = pi[excluded_vertex] + pi[i];
    d_ij += (x[excluded_vertex] - x[i]) * (x[excluded_vertex] - x[i]);
    d_ij += (y[excluded_vertex] - y[i]) * (y[excluded_vertex] - y[i]);
    if (d_ij < closest[0] || d_ij < closest[1]) {
        while (atomicExch(&lock[0], 1) != 0);
        if (d_ij < closest[0] && d_ij < closest[1]) {
            closest[1] = closest[0];
            closest_i[1] = closest_i[0];
            closest[0] = d_ij;
            closest_i[0] = i;
        } else if (d_ij < closest[1]) {
            closest[1] = d_ij;
            closest_i[1] = i;
        }
        lock[0] = 0;
    }
}

__global__ void excluded_vertex_set(float* closest, int* closest_i, int* degrees, float* L_T, int excluded_vertex) {
    degrees[excluded_vertex] += 2;
    degrees[closest_i[0]]++;
    degrees[closest_i[1]]++;
    //printf("valitaan %d-%d %.2f\n", excluded_vertex, closest_i[0], closest[0]);
    //printf("valitaan %d-%d %.2f\n", excluded_vertex, closest_i[1], closest[1]);
    L_T[0] += closest[0] + closest[1];
}


std::pair<float, std::vector<int>> gpu_prim_onetree(int n, float* Gx, float* Gy, float* Gpi) {
    int* Gcomponent = NULL;
    cudaMalloc((void**)&Gcomponent, n * sizeof(int));
    std::vector<int> super_init;
    for (int i = 0; i < n; ++i)
        super_init.push_back(i);
    cudaMemcpy(Gcomponent, super_init.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int* Gsuccessor = NULL;
    cudaMalloc((void**)&Gsuccessor, n * sizeof(int));
    cudaMemcpy(Gsuccessor, super_init.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int* Gvertex_lock = NULL;
    cudaMalloc((void**)&Gvertex_lock, n * sizeof(int));
    cudaMemset(Gvertex_lock, 0, n*sizeof(int));

    float* Gsmallest_add = NULL;
    cudaMalloc((void**)&Gsmallest_add, n * sizeof(float));


    int* Gsmallest_i = NULL;
    cudaMalloc((void**)&Gsmallest_i, n * sizeof(int));
    int* Gsmallest_j = NULL;
    cudaMalloc((void**)&Gsmallest_j, n * sizeof(int));

    int* Gdegrees = NULL;
    cudaMalloc((void**)&Gdegrees, n * sizeof(int));
    cudaMemset(Gdegrees, 0, n*sizeof(int));

    float* GL_T = NULL;
    cudaMalloc((void**)&GL_T, 1 * sizeof(float));
    cudaMemset(GL_T, 0, 1*sizeof(float));

    int* Gcomponents = NULL;
    cudaMalloc((void**)&Gcomponents, 1 * sizeof(int));
    int tmp = n-1;
    cudaMemcpy(Gcomponents, &tmp, 1 * sizeof(int), cudaMemcpyHostToDevice);

    int excluded_vertex = 0;
    int components = n-1;
    dim3 dimBlock(64, 1);
    dim3 dimGrid(divup(n, 64), divup(n, 64));

    std::vector<float> inf(n, std::numeric_limits<float>::max());
    while (components > 1) {
        cudaMemset(Gsmallest_i, 0, n*sizeof(int));
        cudaMemset(Gsmallest_j, 0, n*sizeof(int));
        cudaMemcpy(Gsmallest_add, inf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        boruvka_smallest_kernel<<<dimGrid, dimBlock>>>(n, Gx, Gy, Gpi, Gcomponent, Gsmallest_add, Gsmallest_i, Gsmallest_j, Gvertex_lock, excluded_vertex);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        boruvka_remove_cycles<<<dim3(n, 1), dim3(1, 1)>>>(n, Gcomponent, Gsmallest_add, Gsmallest_i, Gsmallest_j, Gvertex_lock, Gdegrees, GL_T, Gcomponents, excluded_vertex);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        int n_pd = components;
        while (n_pd > 0) {
            boruvka_pointer_doubling<<<dim3(n, 1), dim3(1, 1)>>>(n, Gsuccessor, Gsmallest_add, Gsmallest_i, Gsmallest_j, Gvertex_lock, Gdegrees, GL_T, Gcomponents, excluded_vertex);
            CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
            n_pd /= 2;
        }
        boruvka_update_components<<<dim3(n, 1), dim3(1, 1)>>>(n, Gcomponent, Gsuccessor, Gsmallest_add, Gsmallest_i, Gsmallest_j, Gvertex_lock, Gdegrees, GL_T, Gcomponents, excluded_vertex);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaMemcpy(&components, Gcomponents, 1 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //int tmp;
        //std::cin >> tmp;
    }
    float* Gclosest = NULL;
    cudaMalloc((void**)&Gclosest, 2 * sizeof(float));
    cudaMemcpy(Gclosest, inf.data(), 2 * sizeof(float), cudaMemcpyHostToDevice);
    int* Gclosest_idx = NULL;
    cudaMalloc((void**)&Gclosest_idx, 2 * sizeof(int));
    excluded_vertex_add<<<dim3(n, 1), dim3(1, 1)>>>(n, Gx, Gy, Gpi, excluded_vertex, Gclosest, Gclosest_idx, Gvertex_lock);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    excluded_vertex_set<<<dim3(1, 1), dim3(1, 1)>>>(Gclosest, Gclosest_idx, Gdegrees, GL_T, excluded_vertex);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    float return_length;
    cudaMemcpy(&return_length, GL_T, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::vector<int> d(n);
    cudaMemcpy(d.data(), Gdegrees, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return {return_length, d};
}

std::vector<float> gpu_subgradient_opt_alpha(float* x, float* y, int n) {
    printf("\n----GPU----\n");
    std::vector<float> pi(n, 0);
    float* Gpi = NULL;
    cudaMalloc((void**)&Gpi, n * sizeof(float));
    cudaMemset(Gpi, 0, n*sizeof(float));
    float* Gx = NULL;
    cudaMalloc((void**)&Gx, n * sizeof(float));
    cudaMemcpy(Gx, x, n * sizeof(float), cudaMemcpyHostToDevice);
    float* Gy = NULL;
    cudaMalloc((void**)&Gy, n * sizeof(float));
    cudaMemcpy(Gy, y, n * sizeof(float), cudaMemcpyHostToDevice);

    float W = -1<<28;
    float t = 1.0;
    int period = n/2;
    int np = 4;
    while (true) {
        const auto& [length, d] = gpu_prim_onetree(n, Gx, Gy, Gpi);
        float w = length;
        for (int i = 0; i < n; ++i)
            w -= pi[i];
        W = std::max(W, w);
        bool is_tour = true;
        std::vector<int> v(n);
        for (int i = 0; i < n; ++i) {
            v[i] = d[i] - 2;
            is_tour &= v[i] == 0;
        }
        for (int i = 0; i < n; ++i) {
            pi[i] = pi[i] + t * v[i];
            //std::cout << v[i] << " ";
        }
        //std::cout << std::endl;
        period--;
        if (period == 0) {
            t *= 0.5;
            period = n/np;
            np *= 2;
        }
        std::cout << is_tour << " " << t << " " << period << " " << length << std::endl;
        if (is_tour || t < 0.001 || period == 0) 
            break;
        cudaMemcpy(Gpi, pi.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    std::cout << "Done, pi:" << std::endl;
    for (int i = 0; i < n; ++i)
        std::cout << pi[i] << " ";
    std::cout << std::endl;
    return pi;
}
