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

std::pair<std::vector<std::vector<float>>, std::vector<int>> shuffle_id_oof(std::vector<std::vector<float>> coords, std::vector<int> id) {
    int swaps = coords[0].size();
    int n = coords[0].size();
    while(swaps--) {
        int si = rand()%n;
        int sj = rand()%n;
        std::swap(coords[0][si], coords[0][sj]);
        std::swap(coords[1][si], coords[1][sj]);
        std::swap(id[si], id[sj]);
    }
    return {coords, id};
}

int cost(std::vector<std::vector<float>>& coords) {
    int n = coords[0].size();
    float c = 0;
    for (int i = 0; i < n; ++i) {
        c += sqrdistance(coords, i, (i+1)%n);
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
    int permutations = 3;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    Timer timer;
    timer.start();
    auto pi = gpu_subgradient_opt_alpha(coords[0].data(), coords[1].data(), n);
    //auto pi = subgradient_opt_alpha(coords);
    auto onetree = prim_onetree_edges(coords, pi);
    auto alpha = calculate_alpha(coords, pi, onetree);
    const int MAX_EDGES = std::min(10, n-1);
    auto allowed = calculate_allowed_alpha_gpu(alpha, MAX_EDGES);
    timer.stop();
    std::cout << "GPU alpha preprocessing took " << timer.elapsedMilliseconds() << std::endl;
    timer.start();
    while(permutations--) {
        auto [shuffled_coords, shuffled_id] = shuffle_alpha_id(coords, alpha, std::min(n-1, 1));
        run_gpu_2opt_restricted(shuffled_coords[0].data(), shuffled_coords[1].data(), shuffled_id.data(), allowed.data(), n, MAX_EDGES);
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
    //std::cout << "ALPHA\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            //std::cout << alpha[i][j] << " ";
            if (i != j)
                tmp[i].push_back({alpha[i][j], j});
        }
        //std::cout << std::endl;
        std::sort(tmp[i].begin(), tmp[i].end());
        for (int j = 0; j < MAX_EDGES; ++j)
            allowed[i].push_back(tmp[i][j].second);
    }
    //std::cout << std::endl;
    return allowed;
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
    return {initial_coords, initial_id};
}

void solve_instance_cpu_alpha(std::vector<std::vector<float>> coords) {
    int permutations = 3;
    int n = coords[0].size();
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    Timer timer;
    timer.start();
    auto pi = subgradient_opt_alpha(coords);
    auto onetree = prim_onetree_edges(coords, pi);
    auto alpha = calculate_alpha(coords, pi, onetree);
    auto exact_alpha = calculate_exact_alpha(coords, pi);
    //for (int i = 0; i < n; ++i) {
    //    for (int j = 0; j < n; ++j) {
    //        std::cout << i << ", " << j << ": " << alpha[i][j] << " vs " << exact_alpha[i][j] << std::endl;
    //    }
    //}
    const int MAX_EDGES = std::min(80, n-1);
    auto allowed = calculate_allowed_alpha(exact_alpha, MAX_EDGES);
    timer.stop();
    std::cout << "CPU alpha preprocessing took " << timer.elapsedMilliseconds() << std::endl;
    std::vector<int> id(n);
    for (int i= 0; i < n; ++i)
        id[i] = i;
    timer.start();
    while(permutations--) {
        auto [shuffled_coords, shuffled_id] = initial_tour_alpha(coords, exact_alpha);
        auto [x, y, id2] = two_opt_best_restricted(shuffled_coords[0], shuffled_coords[1], shuffled_id, allowed);
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
    Timer timer;
    timer.start();
    std::vector<int> id(n);
    for (int i = 0; i < n; ++i)
        id[i] = i;
    float best = std::numeric_limits<float>::max();
    std::vector<std::vector<float>> best_coords;
    auto allowed = calculate_allowed_shortest(coords, std::min(10, n-1));
    
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
    int seed = 43;
    std::srand(time(nullptr));
    
    //auto p = read_graph(argv[1]);
    int n = std::stoi(argv[1]);
    auto p = random_graph(n);

    solve_instance_gpu_alpha(p);
    solve_instance_cpu_alpha(p);
    //solve_instance_gpu_shortest(p);
    //solve_instance_cpu_shortest(p);
    solve_instance_gpu_random(p);
}