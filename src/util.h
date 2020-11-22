#pragma once

#include <vector>
#include <math.h> 


static inline float rand01(){
    return ((float) rand() / (RAND_MAX));
}

std::vector<std::vector<float>> read_graph(char* tsp_name);

static inline float sqrdistance(std::vector<std::vector<float>>& coords, int i, int j) {
    float d = 0;
    for (int k = 0; k < coords.size(); ++k) {
        d += (coords[k][i] - coords[k][j]) * (coords[k][i] - coords[k][j]);
    }
    return std::sqrt(d);
}

static inline std::vector<std::vector<float>> random_graph(int n) {
    const int NDIM = 2;
    std::vector<std::vector<float>> p(2);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < NDIM; ++j) {
            p[j].push_back(rand()%2000+rand01());
        }
    }
    return p;
}

static inline int tour_cost(std::vector<std::vector<float>>& coords) {
    int n = coords[0].size();
    float c = 0;
    for (int i = 0; i < n; ++i) {
        c += sqrdistance(coords, i, (i+1)%n);
    }
    return c;
}

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