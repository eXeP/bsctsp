#include <vector>
#include <iostream>
#include <limits>
#include <algorithm> 
#include <iomanip>


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


void shuffle_id(std::vector<std::vector<float>>& coords, std::vector<int>& id) {
    int swaps = coords[0].size();
    int n = coords[0].size();
    while(swaps--) {
        int si = rand()%n;
        int sj = rand()%n;
        std::swap(coords[0][si], coords[0][sj]);
        std::swap(coords[1][si], coords[1][sj]);
        std::swap(id[si], id[sj]);
    }
}


std::pair<std::vector<std::vector<float>>, std::vector<int>> initial_tour_alpha(std::vector<std::vector<float>>& coords, std::vector<std::vector<float>>& alpha) {
    int n = coords[0].size();
    std::vector<std::vector<std::pair<std::pair<float, float>, int>>> closest(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            closest[i].push_back({{alpha[i][j], distance(coords, i, j)}, j});
        }
        sort(closest[i].begin(), closest[i].end());
    }
    
    std::vector<std::vector<float>> initial_coords(2);
    std::vector<int> initial_id;
    std::vector<bool> picked(n);
    auto pick_vertex = [&initial_coords, &initial_id, &picked, &coords](int i) {
        initial_coords[0].push_back(coords[0][i]);
        initial_coords[1].push_back(coords[1][i]);
        initial_id.push_back(i);
        picked[i] = true;
        //std::cout << i << " ";
    };
    
    int start = rand()%n;
    pick_vertex(start);
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < closest[initial_id.back()].size(); ++j) {
            if (!picked[closest[initial_id.back()][j].second]) {
                pick_vertex(closest[initial_id.back()][j].second);
                break;
            }
        }
    }
    //std::cout << std::endl;
    return {initial_coords, initial_id};
}

std::pair<std::vector<std::vector<float>>, std::vector<int>> initial_tour_alpha(std::vector<std::vector<float>>& coords, std::vector<std::vector<std::pair<float, int>>>& alpha) {
    int n = coords[0].size();
    std::vector<std::vector<float>> initial_coords(2);
    std::vector<int> initial_id;
    std::vector<bool> picked(n);
    auto pick_vertex = [&initial_coords, &initial_id, &picked, &coords](int i) {
        initial_coords[0].push_back(coords[0][i]);
        initial_coords[1].push_back(coords[1][i]);
        initial_id.push_back(i);
        picked[i] = true;
        //std::cout << i << " ";
    };
    
    int start = rand()%n;
    pick_vertex(start);
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < alpha[initial_id.back()].size(); ++j) {
            if (!picked[alpha[initial_id.back()][j].second]) {
                pick_vertex(alpha[initial_id.back()][j].second);
                break;
            }
        }
        for (int j = 0; j < n; ++j) {
            if (!picked[j]) {
                pick_vertex(j);
                break;
            }
        }
    }
    //std::cout << std::endl;
    return {initial_coords, initial_id};
}

std::vector<int> calculate_allowed_alpha_gpu(std::vector<std::vector<std::pair<float, int>>>& alpha) {
    int n = alpha.size();
    std::vector<int> allowed(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < alpha[i].size(); ++j)
            allowed.push_back(alpha[i][j].second);
    }
    return allowed;
}

std::vector<int> calculate_allowed_alpha_gpu(std::vector<std::vector<float>>& alpha, const int MAX_EDGES) {
    int n = alpha.size();
    std::vector<std::vector<std::pair<float, int>>> tmp(n);
    std::vector<int> allowed(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j)
                tmp[i].push_back({alpha[i][j], j});
        }
        std::sort(tmp[i].begin(), tmp[i].end());
        for (int j = 0; j < MAX_EDGES; ++j)
            allowed.push_back(tmp[i][j].second);
    }
    return allowed;
}


void solve_instance_gpu_alpha(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    Timer timer;
    timer.start();
    auto pi = gpu_subgradient_opt_alpha(coords[0].data(), coords[1].data(), n);
    auto onetree = prim_onetree_edges(coords, pi);
    const int MAX_EDGES = std::min(20, n-1);
    auto candidate_set = candidate_generation_alpha(coords, pi, onetree, MAX_EDGES);
    auto allowed = calculate_allowed_alpha_gpu(candidate_set);
    timer.stop();
    std::cout << "GPU alpha preprocessing took " << timer.elapsedMilliseconds() << std::endl;
    timer.start();
    while(permutations--) {
        auto [shuffled_coords, shuffled_id] = initial_tour_alpha(coords, candidate_set);
        run_gpu_2opt_restricted(shuffled_coords[0].data(), shuffled_coords[1].data(), shuffled_id.data(), allowed.data(), n, MAX_EDGES);
        run_gpu_2opt(shuffled_coords[0].data(), shuffled_coords[1].data(), n);
        float new_tour_cost = tour_cost(shuffled_coords);
        if (new_tour_cost < best) {
            best = new_tour_cost;
            best_coords = shuffled_coords;
        }
    }
    timer.stop();
    std::cout << 100 << " perms took " << timer.elapsedMilliseconds() << std::endl;
    std::cout << "GPU (alpha) best was: " << best << std::endl; 
}


std::vector<std::vector<int>> calculate_allowed_alpha(std::vector<std::vector<float>> alpha, const int MAX_EDGES) {
    int n = alpha.size();
    std::vector<std::vector<std::pair<float, int>>> tmp(n);
    std::vector<std::vector<int>> allowed(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j)
                tmp[i].push_back({alpha[i][j], j});
        }
        std::sort(tmp[i].begin(), tmp[i].end());
        for (int j = 0; j < MAX_EDGES; ++j)
            allowed[i].push_back(tmp[i][j].second);
    }
    return allowed;
}

std::vector<std::vector<int>> calculate_allowed_alpha(std::vector<std::vector<std::pair<float, int>>> alpha) {
    int n = alpha.size();
    std::vector<std::vector<int>> allowed(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < alpha[i].size(); ++j)
            allowed[i].push_back(alpha[i][j].second);
    }
    return allowed;
}


void solve_instance_cpu_alpha(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    Timer timer;
    timer.start();
    auto pi = subgradient_opt_alpha(coords);
    auto onetree = prim_onetree_edges(coords, pi);
    const int MAX_EDGES = std::min(20, n-1);
    auto candidate_set = candidate_generation_alpha(coords, pi, onetree, MAX_EDGES);
    auto allowed = calculate_allowed_alpha(candidate_set);
    timer.stop();
    std::cout << "CPU alpha preprocessing took " << timer.elapsedMilliseconds() << std::endl;

    std::vector<int> id(n);
    for (int i= 0; i < n; ++i)
        id[i] = i;
    timer.start();
    while(permutations--) {
        auto [shuffled_coords, shuffled_id] = initial_tour_alpha(coords, candidate_set);
        auto [x, y, id2] = two_opt_best_restricted(shuffled_coords[0], shuffled_coords[1], shuffled_id, allowed);
        shuffled_coords[0] = x;
        shuffled_coords[1] = y;
        auto [x2, y2] = two_opt_best(shuffled_coords[0], shuffled_coords[1]);
        shuffled_coords[0] = x2;
        shuffled_coords[1] = y2;
        float new_tour_cost = tour_cost(shuffled_coords);
        if (new_tour_cost < best) {
            best = new_tour_cost;
            best_coords = shuffled_coords;
        }
    }
    timer.stop();
    std::cout << 100 << " perms took " << timer.elapsedMilliseconds() << std::endl;
    std::cout << "CPU (alpha) best was: " << best << std::endl; 
}


std::vector<std::vector<int>> calculate_allowed_shortest(std::vector<std::vector<float>>& coords, const int MAX_EDGES) {
    int n = coords[0].size();
    std::vector<std::vector<std::pair<float, int>>> tmp(n);
    std::vector<std::vector<int>> allowed(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j)
                tmp[i].push_back({distance(coords, i, j), j});
        }
        std::sort(tmp[i].begin(), tmp[i].end());
        for (int j = 0; j < MAX_EDGES; ++j)
            allowed[i].push_back(tmp[i][j].second);
    } 
    return allowed;
}


void solve_instance_cpu_shortest(std::vector<std::vector<float>> coords) {
    int permutations = 1;
    int n = coords[0].size();
    Timer timer;
    timer.start();
    std::vector<int> id(n);
    for (int i = 0; i < n; ++i)
        id[i] = i;
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    auto allowed = calculate_allowed_shortest(coords, std::min(80, n-1));
    while(permutations--) {
        shuffle_id(coords, id);
        auto [x, y, id2] = two_opt_best_restricted(coords[0], coords[1], id, allowed);
        coords[0] = x;
        coords[1] = y;
        id = id2;
        float new_tour_cost = tour_cost(coords);
        if (new_tour_cost < best) {
            best = new_tour_cost;
            best_coords = coords;
        }
    }
    timer.stop();
    std::cout << 100 << " perms took " << timer.elapsedMilliseconds() << std::endl;
    std::cout << "CPU (shortest) best was: " << best << std::endl; 
}


void solve_instance_gpu_random(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    Timer timer;
    timer.start();
    while(permutations--) {
        shuffle(coords);
        run_gpu_2opt(coords[0].data(), coords[1].data(), n);
        float new_tour_cost = tour_cost(coords);
        if (new_tour_cost < best) {
            best = new_tour_cost;
            best_coords = coords;
        }
    }
    timer.stop();
    std::cout << 100 << " perms took " << timer.elapsedMilliseconds() << std::endl;
    std::cout << "GPU (random) best was: " << best << std::endl; 
}


void solve_instance_cpu_random(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    Timer timer;
    timer.start();
    while(permutations--) {
        shuffle(coords);
        auto [x, y] = two_opt_best(coords[0], coords[1]);
        coords[0] = x;
        coords[1] = y;
        float new_tour_cost = tour_cost(coords);
        if (new_tour_cost < best) {
            best = new_tour_cost;
            best_coords = coords;
        }
    }
    timer.stop();
    std::cout << "100 perms took " << timer.elapsedMilliseconds() << std::endl;
    std::cout << "CPU best was: " << best << std::endl; 
}

int main(int argc, char** argv) {
    std::cout << std::setprecision(10);
    int seed = 43;
    std::srand(time(nullptr));

    auto p = read_graph(argv[1]);
    //int n = std::stoi(argv[1]);
    //auto p = random_graph(n);

    solve_instance_gpu_alpha(p);
    solve_instance_cpu_alpha(p);
    //solve_instance_cpu_shortest(p);
    solve_instance_gpu_random(p);
    //solve_instance_cpu_random(p);
}