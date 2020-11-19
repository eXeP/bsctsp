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
    Timer timer;
    timer.start();
    while(permutations--) {
        shuffle(coords);
        
        run_gpu_2opt(coords[0].data(), coords[1].data(), n);
        

        float new_cost = cost(coords);
        if (new_cost < best) {
            best = new_cost;
            best_coords = coords;
        }
    }
    timer.stop();
    std::cout << 100 << " perms took " << timer.elapsedMilliseconds() << std::endl;
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

std::pair<std::vector<std::vector<float>>, std::vector<int>> shuffle_alpha_id(std::vector<std::vector<float>>& coords, std::vector<std::vector<float>>& alpha, int max_candidates) {
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
    std::vector<int> shuffled_id;
    std::vector<bool> picked(n, false);
    std::function<bool (int, int)> dfs = [&closest, max_candidates, &picked, &coords, &shuffled, &shuffled_id, &dfs, n](int i, int nodes) {
        picked[i] = true;
        shuffled[0].push_back(coords[0][i]);
        shuffled[1].push_back(coords[1][i]);
        shuffled_id.push_back(i);
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
        shuffled_id.clear();
        done = dfs(rand()%n, n-1);
    }
    return {shuffled, shuffled_id};
}

void solve_instance_gpu_alpha(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;

    auto pi = gpu_subgradient_opt_alpha(coords[0].data(), coords[1].data(), n);

    auto onetree = prim_onetree_edges(coords, pi);

    auto alpha = calculate_alpha(coords, pi, onetree);
    Timer timer;
    timer.start();
    while(permutations--) {
        auto shuffled_coords = shuffle_alpha(coords, alpha, std::min(n-1, 5));
        run_gpu_2opt(shuffled_coords[0].data(), shuffled_coords[1].data(), n);
        float new_cost = cost(shuffled_coords);
        if (new_cost < best) {
            best = new_cost;
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
    std::cout << "ALPHA\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << alpha[i][j] << " ";
            if (i != j)
                tmp[i].push_back({alpha[i][j], j});
        }
        std::cout << std::endl;
        std::sort(tmp[i].begin(), tmp[i].end());
        for (int j = 0; j < MAX_EDGES; ++j)
            allowed[i].push_back(tmp[i][j].second);
    }
    std::cout << std::endl;
    return allowed;
}

void solve_instance_cpu_alpha(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;

    auto pi = subgradient_opt_alpha(coords);
    auto onetree = prim_onetree_edges(coords, pi);
    auto alpha = calculate_alpha(coords, pi, onetree);
    const int MAX_EDGES = std::min(10, n-1);
    auto allowed = calculate_allowed_alpha(alpha, MAX_EDGES);
    Timer timer;
    timer.start();
    while(permutations--) {
        auto [shuffled_coords, shuffled_id] = shuffle_alpha_id(coords, alpha, std::min(n-1, 5));
        auto [x, y, id] = two_opt_best_restricted(shuffled_coords[0], shuffled_coords[1], shuffled_id, allowed);
        shuffled_coords[0] = x;
        shuffled_coords[1] = y;
        float new_cost = cost(shuffled_coords);
        if (new_cost < best) {
            best = new_cost;
            best_coords = shuffled_coords;
        }
    }
    timer.stop();
    std::cout << 100 << " perms took " << timer.elapsedMilliseconds() << std::endl;
    std::cout << "CPU (alpha) best was: " << best << std::endl; 
}

std::pair<std::vector<std::vector<float>>, std::vector<int>> shuffle_shortest(std::vector<std::vector<float>>& coords, int max_candidates) {
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
    std::vector<int> shuffled_id;
    std::vector<bool> picked(n, false);
    std::function<bool (int, int)> dfs = [&closest, max_candidates, &picked, &coords, &shuffled, &shuffled_id, &dfs, n](int i, int nodes) {
        picked[i] = true;
        shuffled[0].push_back(coords[0][i]);
        shuffled[1].push_back(coords[1][i]);
        shuffled_id.push_back(i);
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
        shuffled_id.clear();
        done = dfs(rand()%n, n-1);
    }
    return {shuffled, shuffled_id};
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
    std::vector<int> id(n);
    for (int i = 0; i < n; ++i)
        id[i] = i;
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    auto allowed = calculate_allowed_shortest(coords, std::min(40, n-1));
    Timer timer;
    timer.start();
    while(permutations--) {
        shuffle_id(coords, id);
        auto [x, y, id2] = two_opt_best_restricted(coords[0], coords[1], id, allowed);
        coords[0] = x;
        coords[1] = y;
        id = id2;
        float new_cost = cost(coords);
        if (new_cost < best) {
            best = new_cost;
            best_coords = coords;
        }
    }
    timer.stop();
    std::cout << 100 << " perms took " << timer.elapsedMilliseconds() << std::endl;
    std::cout << "CPU (shortest) best was: " << best << std::endl; 
}

void solve_instance_gpu_shortest(std::vector<std::vector<float>> coords) {
    int permutations = 100;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;

    while(permutations--) {
        auto [shuffled_coords, shuffled_id] = shuffle_shortest(coords, std::min(n-1, 10));
        Timer timer;
        timer.start();
        run_gpu_2opt(shuffled_coords[0].data(), shuffled_coords[1].data(), n);
        timer.stop();
        //std::cout << "Iteration took " << timer.elapsedMilliseconds() << std::endl;2
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
    
    //solve_instance_gpu_alpha(p);
    solve_instance_cpu_alpha(p);
    //solve_instance_gpu_shortest(p);
    solve_instance_cpu_shortest(p);
    solve_instance_gpu_random(p);
}