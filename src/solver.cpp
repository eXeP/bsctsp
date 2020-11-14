#include <bits/stdc++.h>
#include "tsp_2opt.cuh"
#include "cuda_preprocessing.cuh"
#include "util.h"
#include "preprocessing.h"


int main() {
    int n = 100;
    std::vector<std::vector<float>> coords(2);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 2; ++j) {
            coords[j].push_back(rand()%2000+((float) rand() / (RAND_MAX)));
        }
    }
}