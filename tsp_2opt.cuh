#ifndef CUDA_TSP_2OPT_H
#define CUDA_TSP_2OPT_H
#include <vector>

void run_gpu_2opt(int* x, int* y, int n);

std::vector<float> gpu_subgradient_opt_alpha(float* x, float* y, int n);
#endif