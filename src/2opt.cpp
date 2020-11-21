#include "2opt.h"
#include "util.h"

#include <algorithm>

void swap_2opt_arr(std::vector<float>& t, int i, int j) {
    std::reverse(t.begin()+i, t.begin()+j+1);
}

std::pair<std::vector<float>, std::vector<float>> two_opt_best(std::vector<float> x, std::vector<float> y) {
    int n = x.size();
    auto dist = [&x, &y](int i, int j) {
        float d1 = (x[i]-x[i-1])*(x[i]-x[i-1]) + (y[i]-y[i-1])*(y[i]-y[i-1]);
        float d2 = (x[j]-x[j+1])*(x[j]-x[j+1]) + (y[j]-y[j+1])*(y[j]-y[j+1]);
        float d3 = (x[i]-x[j+1])*(x[i]-x[j+1]) + (y[i]-y[j+1])*(y[i]-y[j+1]);
        float d4 = (x[j]-x[i-1])*(x[j]-x[i-1]) + (y[j]-y[i-1])*(y[j]-y[i-1]);
        return d1+d2-(d3+d4);
    };
    float best = 0;
    int best_i = 0, best_j = 0;
    while (true) {
        best = 0.f; best_i = 0; best_j = 0;
        #pragma omp parallel for schedule(static,1)
        for (int i = 1; i < n-2; ++i) {
            for (int j = i+1; j < n-1; ++j) {
                float new_impr = dist(i, j);
                if (new_impr > best) {
                    #pragma omp critical 
                    {
                        if (new_impr > best) {
                            best = new_impr;
                            best_i = i;
                            best_j = j;
                        }
                    }
                }
            }
        }
        if (best == 0.f)
            break;
        //std::cout << "Improvement " << best << " " << best_i << " " << best_j << std::endl;
        swap_2opt_arr(x, best_i, best_j);
        swap_2opt_arr(y, best_i, best_j);
    }
    return {x, y};
}

void swap_2opt_arr(std::vector<int>& t, int i, int j) {
    std::reverse(t.begin()+i, t.begin()+j+1);
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<int>> two_opt_best_restricted(std::vector<float> x, std::vector<float> y, std::vector<int> id, std::vector<std::vector<int>> allowed) {
    int n = x.size();
    auto dist = [&x, &y](int i, int j) {
        return (x[i]-x[j])*(x[i]-x[j]) + (y[i]-y[j])*(y[i]-y[j]);
    };
    auto g = [&x, &y](int i, int j) {
        float d1 = (x[i]-x[i-1])*(x[i]-x[i-1]) + (y[i]-y[i-1])*(y[i]-y[i-1]);
        float d2 = (x[j]-x[j+1])*(x[j]-x[j+1]) + (y[j]-y[j+1])*(y[j]-y[j+1]);
        float d3 = (x[i]-x[j+1])*(x[i]-x[j+1]) + (y[i]-y[j+1])*(y[i]-y[j+1]);
        float d4 = (x[j]-x[i-1])*(x[j]-x[i-1]) + (y[j]-y[i-1])*(y[j]-y[i-1]);
        return d1+d2-(d3+d4);
    };
    std::vector<int> id_map(n);
    for (int i = 0; i < n; ++i)
        id_map[id[i]] = i;
    float best = 0;
    int best_i = 0, best_j = 0;
    while (true) {
        best = 0.f; best_i = 0; best_j = 0;
        #pragma omp parallel for schedule(static,1)
        for (int i = 1; i < n-2; ++i) {
            for (auto id_j : allowed[id[i-1]]) {
                int j = id_map[id_j];
                //std::cout << "kokeillaan " << i << ", " << j  << " " << dist(i, j)<< std::endl;
                if (j == n-1 || j == 0)
                    continue;
                int jp = id_map[j+1];
                for (auto id_jp : allowed[id[i]]) {
                    if (jp == id_jp) {
                        float new_impr = g(std::min(i, j), std::max(i, j));
                        //std::cout << "oli " << i << ", " << j << ": " << new_impr << std::endl;
                        if (new_impr > best) {
                            #pragma omp critical 
                            {
                                if (new_impr > best) {
                                    best = new_impr;
                                    best_i = std::min(i, j);
                                    best_j = std::max(i, j);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (best == 0.f)
            break;
        //std::cout << "Improvement " << best << " " << best_i << " " << best_j << std::endl;
        swap_2opt_arr(x, best_i, best_j);
        swap_2opt_arr(y, best_i, best_j);
        swap_2opt_arr(id, best_i, best_j);
        for (int i = 0; i < n; ++i)
            id_map[id[i]] = i;
    }
    return {x, y, id};
}