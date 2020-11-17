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
    while(permutations--) {
        shuffle(coords);
        Timer timer;
        timer.start();
        run_gpu_2opt(coords[0].data(), coords[1].data(), n);
        timer.stop();
        //std::cout << "Iteration took " << timer.elapsedMilliseconds() << std::endl;

        float new_cost = cost(coords);
        if (new_cost < best) {
            best = new_cost;
            best_coords = coords;
        }
    }
    std::cout << "GPU (random) best was: " << best << std::endl; 
}

std::vector<std::vector<float>> shuffle_alpha(std::vector<std::vector<float>>& coords, std::vector<std::vector<float>>& alpha, int max_candidates) {
    int n = coords[0].size();
    std::vector<std::vector<std::pair<float, int>>> closest(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            closest[i].push_back({alpha[i][j], j});
        }
        sort(closest[i].begin(), closest[i].end());
    }
    bool done = false;
    std::vector<std::vector<float>> shuffled(2);
    std::vector<bool> picked(n, false);
    std::function<bool (int, int)> dfs = [&closest, max_candidates, &picked, &coords, &shuffled, &dfs, n](int i, int nodes) {
        picked[i] = true;
        shuffled[0].push_back(coords[0][i]);
        shuffled[1].push_back(coords[1][i]);
        if (nodes == 0) {
            return true;
        }
        //std::cout << "valittu " << i << " " <<nodes << std::endl;
        int ne = rand()%max_candidates%closest[i].size();
        if (picked[closest[i][ne].second]) {
            for (int ne2 = (ne+1)%closest[i].size(); ne2 != ne; ne2 = (ne2+1)%closest[i].size()) {
                if (!picked[closest[i][ne2].second]) {
                    ne = ne2;
                    break;
                }
            }
        }
        if (picked[closest[i][ne].second]) {
            return false;
        } else {
            return dfs(closest[i][ne].second, nodes-1);
        }
    };
    while (!done) {
        //std::cout << "nollataan" << std::endl;
        picked = std::vector<bool>(n, false);
        shuffled[0].clear();
        shuffled[1].clear();
        done = dfs(rand()%n, n-1);
    }
    return shuffled;
}

void solve_instance_gpu_alpha(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;

    auto pi = gpu_subgradient_opt_alpha(coords[0].data(), coords[1].data(), n);

    auto onetree = prim_onetree_edges(coords, pi);

    auto alpha = calculate_alpha(coords, pi, onetree);

    while(permutations--) {
        auto shuffled_coords = shuffle_alpha(coords, alpha, std::min(n-1, 5));
        Timer timer;
        timer.start();
        run_gpu_2opt(shuffled_coords[0].data(), shuffled_coords[1].data(), n);
        timer.stop();
        //std::cout << "Iteration took " << timer.elapsedMilliseconds() << std::endl;
        float new_cost = cost(shuffled_coords);
        if (new_cost < best) {
            best = new_cost;
            best_coords = shuffled_coords;
        }
    }
    std::cout << "GPU (alpha) best was: " << best << std::endl; 
}

void solve_instance_cpu_alpha(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;

    auto pi = subgradient_opt_alpha(coords);

    auto onetree = prim_onetree_edges(coords, pi);

    auto alpha = calculate_alpha(coords, pi, onetree);

    while(permutations--) {
        auto shuffled_coords = shuffle_alpha(coords, alpha, std::min(n-1, 5));
        Timer timer;
        timer.start();
        run_gpu_2opt(shuffled_coords[0].data(), shuffled_coords[1].data(), n);
        timer.stop();
        //std::cout << "Iteration took " << timer.elapsedMilliseconds() << std::endl;
        float new_cost = cost(shuffled_coords);
        if (new_cost < best) {
            best = new_cost;
            best_coords = shuffled_coords;
        }
    }
    std::cout << "CPU (alpha) best was: " << best << std::endl; 
}

std::vector<std::vector<float>> shuffle_shortest(std::vector<std::vector<float>>& coords, int max_candidates) {
    int n = coords[0].size();
    std::vector<std::vector<std::pair<float, int>>> closest(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            closest[i].push_back({distance(coords, i, j), j});
        }
        sort(closest[i].begin(), closest[i].end());
    }
    bool done = false;
    std::vector<std::vector<float>> shuffled(2);
    std::vector<bool> picked(n, false);
    std::function<bool (int, int)> dfs = [&closest, max_candidates, &picked, &coords, &shuffled, &dfs, n](int i, int nodes) {
        picked[i] = true;
        shuffled[0].push_back(coords[0][i]);
        shuffled[1].push_back(coords[1][i]);
        if (nodes == 0) {
            return true;
        }
        //std::cout << "valittu " << i << " " <<nodes << std::endl;
        int ne = rand()%max_candidates%closest[i].size();
        if (picked[closest[i][ne].second]) {
            for (int ne2 = (ne+1)%closest[i].size(); ne2 != ne; ne2 = (ne2+1)%closest[i].size()) {
                if (!picked[closest[i][ne2].second]) {
                    ne = ne2;
                    break;
                }
            }
        }
        if (picked[closest[i][ne].second]) {
            return false;
        } else {
            return dfs(closest[i][ne].second, nodes-1);
        }
    };
    while (!done) {
        //std::cout << "nollataan" << std::endl;
        picked = std::vector<bool>(n, false);
        shuffled[0].clear();
        shuffled[1].clear();
        done = dfs(rand()%n, n-1);
    }
    return shuffled;
}

void solve_instance_gpu_shortest(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;

    while(permutations--) {
        auto shuffled_coords = shuffle_shortest(coords, std::min(n-1, 10));
        Timer timer;
        timer.start();
        run_gpu_2opt(shuffled_coords[0].data(), shuffled_coords[1].data(), n);
        timer.stop();
        //std::cout << "Iteration took " << timer.elapsedMilliseconds() << std::endl;
        float new_cost = cost(shuffled_coords);
        if (new_cost < best) {
            best = new_cost;
            best_coords = shuffled_coords;
        }
    }
    std::cout << "GPU (shortest) best was: " << best << std::endl; 
}

void solve_instance_cpu_random(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
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
    solve_instance_gpu_alpha(p);
    solve_instance_cpu_alpha(p);
    solve_instance_gpu_shortest(p);
}