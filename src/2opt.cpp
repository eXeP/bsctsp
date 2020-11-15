#include "2opt.h"

#include "util.h"

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