#pragma once

#include <bits/stdc++.h>

static inline float rand01(){
    return ((float) rand() / (RAND_MAX));
}

std::vector<std::vector<float>> read_graph(char* tsp_name);

static inline float distance(std::vector<std::vector<float>>& coords, int i, int j) {
    float d = 0;
    for (int k = 0; k < coords.size(); ++k) {
        d += (coords[k][i] - coords[k][j]) * (coords[k][i] - coords[k][j]);
    }
    return d;
}

static inline float d_ij(std::vector<std::vector<float>>& coords, std::vector<float>& pi, int i, int j) {
    return pi[i] + pi[j] + distance(coords, i, j);
}

float calculate_dist(std::vector<std::vector<float>>& coords, std::vector<int>& path);

float read_optimal(std::vector<std::vector<float>>& coords, char* tsp_name);