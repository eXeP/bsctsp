#pragma once
#include <vector>

void run_gpu_2opt(float* x, float* y, int n);

void run_gpu_2opt_restricted(float* x, float* y, int* id, int* moves, int n, int allowed_moves);