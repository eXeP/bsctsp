#include <vector>
#include <iostream>

#include "cuda_2opt.cuh"
#include "cuda_preprocessing.cuh"
#include "util.h"
#include "preprocessing.h"

int main(int argc, char** argv) {
    std::cout << std::setprecision(20);
    int seed = 42;
    std::srand(seed);
    int n = std::stoi(argv[1]);
    
    const int NDIM = 2;
    
    std::vector<std::vector<float>> p;
    for (int tests = 0; tests < 10000; ++tests) {
        p.clear();
        for (int i = 0; i < NDIM; ++i) {
            p.push_back(std::vector<float>());
        }
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < NDIM; ++j) {
                p[j].push_back(rand()%2000+((float) rand() / (RAND_MAX)));
            }
        }
        auto piCPU = subgradient_opt_alpha(p);

        auto piGPU = gpu_subgradient_opt_alpha(p[0].data(), p[1].data(), n);
        std::cout << "\n------CHECK------\n";
        std::cout << "seed: " << seed << std::endl;
        bool diff = false;
        for (int i = 0; i < n; ++i) {
            if (abs(piCPU[i]-piGPU[i]) > 0.001) {
                diff = true;
                std::cout << tests <<" Diff: " << i << " " << piCPU[i] << " vs " << piGPU[i] << std::endl;
            }
        }
        if (diff) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < NDIM; ++j) {
                    std::cout << p[j][i] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            return 0;
        } 
    }
    
    
}