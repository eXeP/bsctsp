#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <float.h>

#include "cuda_preprocessing.cuh"
#include "util.cuh"


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
                d_ij += (x_i - shared_x[k]) * (x_i - shared_x[k]);
                d_ij += (y_i - shared_y[k]) * (y_i - shared_y[k]);
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
                d_ij += (x_i - shared_x[k]) * (x_i - shared_x[k]);
                d_ij += (y_i - shared_y[k]) * (y_i - shared_y[k]);
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
    if (threadIdx.x == 0) {
        for (int k = 0; k < 64; ++k) {
            int tmp_i = blockIdx.x * blockDim.x + k;
            if (tmp_i >= n || tmp_i == excluded_vertex)
                continue;
            component_i = shared_component_i[k];
            best = shared_best[k];
            best_j = shared_best_j[k];
            if (best < component_best[component_i] || best == component_best[component_i]) {
                while (atomicExch(&component_lock[component_i], 1) != 0);
                if (best < component_best[component_i]) {
                    component_best[component_i] = best;
                    component_best_i[component_i] = tmp_i;
                    component_best_j[component_i] = best_j;
                } else if (abs(best - component_best[component_i]) < 0.0000001) {
                    int mi_c = min(tmp_i, best_j), ma_c = max(tmp_i, best_j);
                    int mi_o = min(component_best_i[component_i], component_best_j[component_i]), ma_o = max(component_best_i[component_i], component_best_j[component_i]);
                    if (mi_c < mi_o) {
                        component_best_i[component_i] = tmp_i;
                        component_best_j[component_i] = best_j;
                    } else if (mi_c == mi_o && ma_c < ma_o) {
                        component_best_i[component_i] = tmp_i;
                        component_best_j[component_i] = best_j;
                    }
                }
                component_lock[component_i] = 0;
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
    
    component[i] = successor[i];
    int vertex_ii = component_best_i[component_i];
    int vertex_ij = component_best_j[component_i];
    int vertex_jj = component_best_j[component_j];
    if (i == vertex_ii) {
        if (vertex_jj == i) {
            if (i < vertex_ij) {
                atomicSub(components, 1);
                atomicAdd(L_T, component_best[component_i]);
            }
        } else {
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
        __threadfence();
    }
}

__global__ void excluded_vertex_set(float* closest, int* closest_i, int* degrees, float* L_T, int excluded_vertex) {
    degrees[excluded_vertex] += 2;
    degrees[closest_i[0]]++;
    degrees[closest_i[1]]++;
    L_T[0] += closest[0] + closest[1];
}


std::pair<float, std::vector<int>> gpu_boruvka_onetree(int n, float* Gx, float* Gy, float* Gpi) {
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

    cudaFree(Gcomponent);
    cudaFree(Gsuccessor);
    cudaFree(Gvertex_lock);
    cudaFree(Gsmallest_add);
    cudaFree(Gsmallest_i);
    cudaFree(Gsmallest_j);
    cudaFree(Gdegrees);
    cudaFree(GL_T);
    cudaFree(Gcomponents);
    cudaFree(Gclosest);
    cudaFree(Gclosest_idx);
    return {return_length, d};
}

std::vector<float> gpu_subgradient_opt_alpha(float* x, float* y, int n) {
    printf("\n----Subgradient opt. for alpha-nearness on GPU----\n");
    float* Gpi = NULL;
    cudaMalloc((void**)&Gpi, n * sizeof(float));
    cudaMemset(Gpi, 0, n*sizeof(float));
    float* Gx = NULL;
    cudaMalloc((void**)&Gx, n * sizeof(float));
    cudaMemcpy(Gx, x, n * sizeof(float), cudaMemcpyHostToDevice);
    float* Gy = NULL;
    cudaMalloc((void**)&Gy, n * sizeof(float));
    cudaMemcpy(Gy, y, n * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> pi(n, 0), best_pi(n, 0);
    auto [init_w, init_d] = gpu_boruvka_onetree(n, Gx, Gy, Gpi);
    float best_w = init_w;
    std::vector<int> last_v(n), v(n);
    bool is_tour = true;
    for (int i = 0; i < n; ++i) {
        last_v[i] = init_d[i] - 2;
        v[i] = last_v[i];
        is_tour &= (last_v[i] == 0);
    }
    bool initial_phase = true;
    int initial_period = max(n/2, 100);
    int period = initial_period;
    for (float t = 1.f; t > 0; t /= 2.f, period /= 2) {
        for (int p = 1; t > 0 && p <= period; ++p) {
            for (int i = 0; i < n; ++i) {
                pi[i] += t * ( 0.7f * v[i] + 0.3f * last_v[i]);
            }
            cudaMemcpy(Gpi, pi.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            last_v = v;
            auto [w, d] = gpu_boruvka_onetree(n, Gx, Gy, Gpi);
            is_tour = true;
            for (int i = 0; i < n; ++i) {
                v[i] = d[i] - 2;
                is_tour &= (v[i] == 0);
            }
            if (w > best_w) {
                best_w = w;
                best_pi = pi;
                if (initial_phase)
                    t *= 2.f;
                if (p == period)
                    period *= 2;
            } else if (initial_phase && p > initial_period / 2) {
                initial_phase = false;
                p = 0;
                t = 0.75f * t;
            }
        }
    }
    cudaFree(Gpi);
    cudaFree(Gx);
    cudaFree(Gy);
    return best_pi;
}
