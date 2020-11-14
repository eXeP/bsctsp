#include "preprocessing.h"

#include <bits/stdc++.h>
#include "tsp_2opt.cuh"
#include "tsp_preprocessing.cuh"
#include "util.h"

static inline float distance(std::vector<std::vector<float>>& coords, int i, int j) {
    float d = 0;
    for (int k = 0; k < coords.size(); ++k) {
        d += (coords[k][i] - coords[k][j]) * (coords[k][i] - coords[k][j]);
    }
    return sqrt(d);
}

static inline float d_ij(std::vector<std::vector<float>>& coords, std::vector<float>& pi, int i, int j) {
    return pi[i] + pi[j] + distance(coords, i, j);
}

float calculate_dist(std::vector<std::vector<float>>& coords, std::vector<int>& path) {
    float dist = 0;
    for(int i = 0; i < coords[0].size(); ++i)
        dist += distance(coords, path[i], path[(i+1)%coords[0].size()]);
    return dist;
}

std::vector<std::vector<float>> calculate_alpha(std::vector<std::vector<float>>& coords, std::vector<float>& pi, one_tree& onetree) {
    int n = coords[0].size();
    std::vector<std::vector<float>> alpha = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) 
                alpha[i][j] = std::numeric_limits<float>::max();
            //Case (a) in paper
            for (auto f : onetree.edges[i]) {
                if (f == j) {
                    alpha[i][j] = 0.0f;
                    goto next_j;
                }
            }
            //Case (b) in paper
            if (i == 0 || j == 0) {
                float one_max_edge = 0.0f;
                for (auto f : onetree.edges[0]) {
                    one_max_edge = std::max(one_max_edge, d_ij(coords, pi, 0, f));
                }
                alpha[i][j] = d_ij(coords, pi, i, j) - one_max_edge;
                goto next_j;
            }
        }
        next_j:
        1==1;
    }
    std::vector<int> mark = std::vector<int>(n);
    std::vector<float> b = std::vector<float>(n);
    for (int i = 1; i < n; ++i)
        mark[i] = 0;
    for (int i = 1; i < n; ++i) {
        int node_id = onetree.topo[i-1];
        b[node_id] = std::numeric_limits<float>::min();
        int j = 0;
        for (int k = node_id; k != onetree.topo[0]; k = j) {
            j = onetree.dad[k];
            b[j] = std::max(b[k], d_ij(coords, pi, k, j));
            mark[j] = node_id;
        }
        for (j = 1; j < n; ++j) {
            if (j != node_id) {
                if (mark[j] != node_id) {
                    b[j] = std::max(b[onetree.dad[j]], d_ij(coords, pi, j, onetree.dad[j]));
                    alpha[node_id][j] = d_ij(coords, pi, j, node_id) - b[j];
                }
            }
        }
    }
    return alpha;
}

one_tree prim_onetree_edges(std::vector<std::vector<float>>& p, std::vector<float>& pi) {
    int n = p[0].size();
    auto c = [&p, &pi](int i, int j) {
        float c_ij = pi[i] + pi[j];
        for (size_t k = 0; k < p.size(); ++k) {
            float c_k = p[k][i]-p[k][j];
            c_ij += c_k * c_k;
        }
        return c_ij;
    };
    std::vector<bool> picked = std::vector<bool>(n, false);
    std::vector<std::pair<float, int>> value = std::vector<std::pair<float, int>>(n, {std::numeric_limits<float>::max(), -1});
    int excluded_vertex = 0;
    int start_vertex = (excluded_vertex+1)%n;
    value[start_vertex] = {0, start_vertex};
    std::set<std::pair<float, int>> pq;
    pq.insert({0, start_vertex});
    float length = 0;
    std::vector<int> degrees(n);
    std::vector<std::vector<int>> edges(n, std::vector<int>());
    std::vector<int> topo, dad(n);
    while (!pq.empty()) {
        auto [current_value, current_vertex] = *pq.begin();
        pq.erase(pq.begin());
        if (picked[current_vertex]) {
            continue;
        }
        if (current_vertex != value[current_vertex].second) {
            dad[current_vertex] = value[current_vertex].second;
            degrees[current_vertex]++;
            degrees[value[current_vertex].second]++;
            length += value[current_vertex].first;
            edges[current_vertex].push_back(value[current_vertex].second);
            edges[value[current_vertex].second].push_back(current_vertex);
        }
        topo.push_back(current_vertex);
        picked[current_vertex] = true;
        for (int i = 0; i < n; ++i) {
            if (i == current_vertex || picked[i] || i == excluded_vertex)
                continue;
            float new_len = c(i, current_vertex);
            if (new_len < value[i].first) {
                auto old = pq.find({value[i].first, i});
                if (old != pq.end())
                    pq.erase(old);
                pq.insert({new_len, i});
                value[i] = {new_len, current_vertex};
            }
            
        }
    }
    std::pair<int, float> edge_lens[2] = {{-1, std::numeric_limits<float>::max()}, {-1, std::numeric_limits<float>::max()}};
    for (int i = 0; i < n; ++i) {
        if (i == excluded_vertex)
            continue;
        float len = c(excluded_vertex, i);
        if (len < edge_lens[1].second && len < edge_lens[0].second) {
            edge_lens[1] = edge_lens[0];
            edge_lens[0] = {i, len};
        } else if (len < edge_lens[1].second) {
            edge_lens[1] = {i, len};
        }
    }
    edges[excluded_vertex].push_back(edge_lens[0].first);
    edges[value[edge_lens[0].first].second].push_back(excluded_vertex);
    edges[excluded_vertex].push_back(edge_lens[1].first);
    edges[value[edge_lens[1].first].second].push_back(excluded_vertex);
    length += edge_lens[0].second + edge_lens[1].second;
    degrees[edge_lens[0].first]++;
    degrees[edge_lens[1].first]++;
    degrees[excluded_vertex] += 2;

    one_tree onetree;
    onetree.length = length;
    onetree.degrees = degrees;
    onetree.edges = edges;
    onetree.dad = dad;
    onetree.topo = topo;

    return onetree;
}

std::pair<float, std::vector<int>> prim_onetree(std::vector<std::vector<float>>& p, std::vector<float>& pi) {
    int n = p[0].size();
    auto c = [&p, &pi](int i, int j) {
        float c_ij = pi[i] + pi[j];
        for (size_t k = 0; k < p.size(); ++k) {
            float c_k = p[k][i]-p[k][j];
            c_ij += c_k * c_k;
        }
        return c_ij;
    };
    std::vector<bool> picked = std::vector<bool>(n, false);
    std::vector<std::pair<float, int>> value = std::vector<std::pair<float, int>>(n, {std::numeric_limits<float>::max(), -1});
    int excluded_vertex = 0;
    int start_vertex = (excluded_vertex+1)%n;
    value[start_vertex] = {0, start_vertex};
    std::set<std::pair<float, int>> pq;
    pq.insert({0, start_vertex});
    float length = 0;
    std::vector<int> degrees(n);
    while (!pq.empty()) {
        auto [current_value, current_vertex] = *pq.begin();
        pq.erase(pq.begin());
        if (picked[current_vertex]) {
            continue;
        }
        if (current_vertex != value[current_vertex].second) {
            degrees[current_vertex]++;
            degrees[value[current_vertex].second]++;
            length += value[current_vertex].first;
        }
        picked[current_vertex] = true;
        for (int i = 0; i < n; ++i) {
            if (i == current_vertex || picked[i] || i == excluded_vertex)
                continue;
            float new_len = c(i, current_vertex);
            if (new_len < value[i].first) {
                auto old = pq.find({value[i].first, i});
                if (old != pq.end())
                    pq.erase(old);
                pq.insert({new_len, i});
                value[i] = {new_len, current_vertex};
            }
            
        }
    }
    std::pair<int, float> edge_lens[2] = {{-1, std::numeric_limits<float>::max()}, {-1, std::numeric_limits<float>::max()}};
    for (int i = 0; i < n; ++i) {
        if (i == excluded_vertex)
            continue;
        float len = c(excluded_vertex, i);
        if (len < edge_lens[1].second && len < edge_lens[0].second) {
            edge_lens[1] = edge_lens[0];
            edge_lens[0] = {i, len};
        } else if (len < edge_lens[1].second) {
            edge_lens[1] = {i, len};
        }
    }
    length += edge_lens[0].second + edge_lens[1].second;
    degrees[edge_lens[0].first]++;
    degrees[edge_lens[1].first]++;
    degrees[excluded_vertex] += 2;
    return {length, degrees};
}

std::vector<float> subgradient_opt_alpha(std::vector<std::vector<float>>& coord) {
    int n = coord[0].size();
    std::vector<float> pi(n, 0), best_pi(n, 0);
    auto [init_w, init_d] = prim_onetree(coord, pi);
    float best_w = init_w;
    std::vector<int> last_v(n), v(n);
    bool is_tour = true;
    for (int i = 0; i < n; ++i) {
        last_v[i] = init_d[i] - 2;
        v[i] = last_v[i];
        is_tour &= (last_v[i] == 0);
    }
    bool initial_phase = true;
    int initial_period = std::max(n/2, 100);
    int period = initial_period;
    for (float t = 1.f; t > 0; t /= 2.f, period /= 2) {
        for (int p = 1; t > 0 && p <= period; ++p) {
            for (int i = 0; i < n; ++i) {
                pi[i] += t * ( 0.7f * v[i] + 0.3f * last_v[i]);
            }
            last_v = v;
            auto [w, d] = prim_onetree(coord, pi);
            is_tour = true;
            for (int i = 0; i < n; ++i) {
                v[i] = d[i] - 2;
                is_tour &= (v[i] == 0);
            }
            if (w > best_w) {
                best_w = w;
                best_pi = pi;
                if (initial_phase)
                    t *= 2.f;
                if (p == period)
                    period *= 2;
            } else if (initial_phase && p > initial_period / 2) {
                initial_phase = false;
                p = 0;
                t = 0.75f * t;
            }
        }
    }
    return best_pi;
}


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