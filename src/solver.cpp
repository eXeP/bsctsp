#include <vector>
#include <iostream>

#include "cuda_2opt.cuh"
#include "2opt.h"
#include "cuda_preprocessing.cuh"
#include "util.h"
#include "preprocessing.h"
#include "timer.h"

void shuffle(std::vector<std::vector<float>>& coords) {
    int swaps = coords[0].size();
    int n = coords[0].size();
    while(swaps--) {
        int si = rand()%n;
        int sj = rand()%n;
        std::swap(coords[0][si], coords[0][sj]);
        std::swap(coords[1][si], coords[1][sj]);
    }
}

int cost(std::vector<std::vector<float>>& coords) {
    int n = coords[0].size();
    float c = 0;
    for (int i = 0; i < n; ++i) {
        c += distance(coords, i, (i+1)%n);
    }
    return c;
}

void solve_instance_gpu_random(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    std::srand(42);
    while(permutations--) {
        shuffle(coords);
        Timer timer;
        timer.start();
        run_gpu_2opt(coords[0].data(), coords[1].data(), n);
        timer.stop();
        std::cout << "Iteration took " << timer.elapsedMilliseconds() << std::endl;
        float new_cost = cost(coords);
        if (new_cost < best) {
            best = new_cost;
            best_coords = coords;
        }
    }
    std::cout << "GPU best was: " << best << std::endl; 
}

void solve_instance_cpu_random(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    std::srand(42);
    while(permutations--) {
        shuffle(coords);
        Timer timer;
        timer.start();
        auto [x, y] = two_opt_best(coords[0], coords[1]);
        timer.stop();
        std::cout << "Iteration took " << timer.elapsedMilliseconds() << std::endl;
        coords[0] = x;
        coords[1] = y;
        float new_cost = cost(coords);
        if (new_cost < best) {
            best = new_cost;
            best_coords = coords;
        }
    }
    std::cout << "CPU best was: " << best << std::endl; 
}

int main(int argc, char** argv) {
    std::cout << std::setprecision(10);
    int seed = 42;
    std::srand(seed);
    int n = std::stoi(argv[1]);
    
    const int NDIM = 2;
    
    std::vector<std::vector<float>> p(2);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < NDIM; ++j) {
            p[j].push_back(rand()%2000+rand01());
            std::cout << p[j].back() << " ";
        }
    }
    std::cout << std::endl;
    solve_instance_gpu_random(p);
    solve_instance_cpu_random(p);
}