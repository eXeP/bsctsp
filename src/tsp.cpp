#include <bits/stdc++.h>
#include "tsp_2opt.cuh"
#include "tsp_preprocessing.cuh"

struct Graph {
    int nodes;
    std::vector<std::vector<std::pair<int, float>>> edges;
    std::vector<int> topo, dad;
    

    Graph (int n): nodes(n), edges(std::vector<std::vector<std::pair<int, float>>>(n+1)), dad(std::vector<int>(n+1, 0)) {}

    void add_edge(int i, int j, float w) {
        edges[i].push_back({j, w});
    }

    std::vector<std::vector<float>> w;
    void floyd_warshall() {
        w = std::vector<std::vector<float>>(nodes, std::vector<float>(nodes, std::numeric_limits<float>::max()/10.0));
        for (int i = 0; i < nodes; ++i)
            for (auto& f: edges[i]) 
                w[i][f.first] = std::min(w[i][f.first], f.second);
        for (int i = 0; i < nodes; ++i) {
            for (int j = 0; j < nodes; ++j) {
                for (int k = 0; k < nodes; ++k) {
                    w[i][j] = std::min(w[i][j], w[i][k]+w[k][j]);
                }
            }
        } 
    }

    float dist(int i, int j) {
        return w[i][j];
    }
};

struct TSP_Graph {
    int dimension;
    std::vector<std::array<int, 2>> coordinates;
    std::vector<int> x;
    std::vector<int> y;

    TSP_Graph (int dimension): dimension(dimension), coordinates(std::vector<std::array<int, 2>>(dimension+1)), x(std::vector<int>(dimension+1)), y(std::vector<int>(dimension+1)) {}
    
    void add_node(int i, int x_coord, int y_coord) {
        coordinates[i] = {x_coord, y_coord};
        x[i] = x_coord;
        y[i] = y_coord;
    }

    int dist(int i, int j) {
        return 
        ((int) coordinates[i][0]-coordinates[j][0])*((int) coordinates[i][0]-coordinates[j][0])
        +((int) coordinates[i][1]-coordinates[j][1])*((int) coordinates[i][1]-coordinates[j][1]);
    } 
};

struct TSP_Solution {
    float cost;
    std::vector<int> solution;
    TSP_Solution (float cost, std::vector<int> solution): cost(cost), solution(solution) {}
};

float rand01(){
    return ((float) rand() / (RAND_MAX));
}

TSP_Graph read_graph(char* tsp_name) {
    std::string tsp_file_name = "tsplib/" + std::string(tsp_name) + ".tsp";
    std::ifstream tsp_file(tsp_file_name);
    std::string sink;
    int dimension;
    std::getline(tsp_file, sink);
    std::getline(tsp_file, sink);
    std::getline(tsp_file, sink);
    tsp_file >> sink >> dimension;
    std::getline(tsp_file, sink);
    std::getline(tsp_file, sink);
    std::getline(tsp_file, sink);

    TSP_Graph g = TSP_Graph(dimension);
    for (int i = 0; i < dimension; ++i) {
        int id, x, y;
        tsp_file >> id >> x >> y;
        g.add_node(id, x, y);
    }
    return g;
}

float calculate_dist(TSP_Graph& g, std::vector<int>& path) {
    float dist = 0;
    for(int i = 0; i < g.dimension; ++i)
        dist += g.dist(path[i], path[(i+1)%g.dimension]);
    return dist;
}

float read_optimal(TSP_Graph& g, char* tsp_name) {
    std::ifstream tsp_file("tsplib/" + std::string(tsp_name) + ".opt.tour");
    std::string sink;
    std::getline(tsp_file, sink);
    std::getline(tsp_file, sink);
    std::getline(tsp_file, sink);
    std::getline(tsp_file, sink);
    int node;
    tsp_file >> node;
    std::vector<int> tour;
    do {
        tour.push_back(node);
        tsp_file >> node;
    } while (node != -1);
    return calculate_dist(g, tour);
}


Graph prim_1tree(TSP_Graph& g) {
    std::vector<bool> picked = std::vector<bool>(g.dimension+1, false);
    std::vector<std::pair<float, int>> value = std::vector<std::pair<float, int>>(g.dimension+1, {std::numeric_limits<float>::max(), -1});
    value[2] = {0, 2};
    std::set<std::pair<float, int>> pq;
    pq.insert({0, 2});
    Graph onetree = Graph(g.dimension);
    while (!pq.empty()) {
        auto current = *pq.begin();
        pq.erase(pq.begin());
        if (picked[current.second]) {
            continue;
        }
        onetree.topo.push_back(current.second);
        onetree.add_edge(current.second, value[current.second].second, value[current.second].first-value[value[current.second].second].first);
        onetree.add_edge(value[current.second].second, current.second, value[current.second].first-value[value[current.second].second].first);
        picked[current.second] = true;
        for (int i = 2; i <= g.dimension; ++i) {
            if (i == current.second || picked[i])
                continue;
            float new_len = g.dist(i, current.second);
            if (current.first+new_len < value[i].first) {
                auto old = pq.find({value[i].first, i});
                if (old != pq.end())
                    pq.erase(old);
                pq.insert({current.first+new_len, i});
                value[i] = {current.first+new_len, current.second};
                onetree.dad[i] = current.second;
            }
        }
    }

    std::vector<std::pair<float, int>> edges1;
    for (int i = 2; i <= g.dimension; ++i)
        edges1.push_back({g.dist(1, i), i});
    std::sort(edges1.begin(), edges1.end());
    onetree.add_edge(1, edges1[0].first, edges1[0].second);
    onetree.add_edge(edges1[0].first, 1, edges1[0].second);
    onetree.add_edge(1, edges1[1].first, edges1[1].second);
    onetree.add_edge(edges1[1].first, 1, edges1[1].second);
    
    return onetree;
}

std::vector<std::vector<float>> calculate_alpha(TSP_Graph& g, Graph one_tree) {
    float LT = 0.0f;
    for (int i = 1; i <= g.dimension; ++i) {
        for (auto f: one_tree.edges[i]) {
            LT += f.second;
        }
    }
    LT /= 2.0f;
    std::vector<std::vector<float>> alpha = std::vector<std::vector<float>>(g.dimension+1, std::vector<float>(g.dimension+1, 0));
    for (int i = 1; i <= g.dimension; ++i) {
        for (int j = i; j <= g.dimension; ++j) {
            if (i == j) 
                alpha[i][j] = std::numeric_limits<float>::max();
            //Case (a) in paper
            for (auto f : one_tree.edges[i]) {
                if (f.first == j) {
                    alpha[i][j] = 0.0f;
                    goto next_j;
                }
            }
            //Case (b) in paper
            if (i == 1 || j == 1) {
                float one_max_edge = 0.0f;
                for (auto f : one_tree.edges[1])
                    one_max_edge = std::max(one_max_edge, f.second);
                alpha[i][j] = g.dist(i, j) - one_max_edge;
                goto next_j;
            }
        }
        next_j:
        1==1;
    }
    std::vector<int> mark = std::vector<int>(g.dimension+1);
    std::vector<float> b = std::vector<float>(g.dimension+1);
    for (int i = 2; i <= g.dimension; ++i)
        mark[i] = 0;
    for (int i = 2; i <= g.dimension; ++i) {
        int node_id = one_tree.topo[i-2];
        b[node_id] = -1<<28;
        int j = 0;
        for (int k = node_id; k != 2; k = j) {
            j = one_tree.dad[k];
            b[j] = std::max(b[k], (float)g.dist(k, j));
            mark[j] = node_id;
        }
        for (j = 2; j <= g.dimension; ++j) {
            if (j != node_id) {
                if (mark[j] != node_id) {
                    b[j] = std::max(b[one_tree.dad[j]], (float)g.dist(j, one_tree.dad[j]));
                    alpha[node_id][j] = g.dist(j, node_id)-b[j];
                }
            }
        }
    }
    return alpha;
}

std::pair<float, std::vector<int>> prim_onetree(std::vector<std::vector<float>>& p, std::vector<float>& pi) {
    int NDIM = p.size();
    int n = p[0].size();
    auto c = [&p, &pi](int i, int j) {
        float c_ij = pi[i] + pi[j];
        for (int k = 0; k < p.size(); ++k) {
            float c_k = p[k][i]-p[k][j];
            c_ij += c_k * c_k;
        }
        return c_ij;
    };

    auto cd = [&p, &pi](int i, int j) {
        float c_ij = pi[i] + pi[j];
        
        for (int k = 0; k < p.size(); ++k) {
            float c_k = p[k][i]-p[k][j];
            c_ij += c_k * c_k;
        }
        //printf("wtf %d %d %f %f %f %f %f %f %f %f %f %f %f\n", i, j, pi[i], pi[j], c_ij, p[0][i], p[0][j], p[1][i], p[1][j], (p[0][i] - p[0][j]), (p[1][i] - p[1][j]), (p[0][i] - p[0][j])*(p[0][i] - p[0][j]), (p[1][i] - p[1][j])*(p[1][i] - p[1][j]));
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
        
        std::cout << "valitaan " << std::min(current_vertex, value[current_vertex].second) << "-" << std::max(current_vertex, value[current_vertex].second) << " "  << value[current_vertex].first << std::endl;
        picked[current_vertex] = true;
        for (int i = 0; i < n; ++i) {
            if (i == current_vertex || picked[i] || i == excluded_vertex)
                continue;
            float new_len = cd(i, current_vertex);
            if (new_len < value[i].first) {
                auto old = pq.find({value[i].first, i});
                if (old != pq.end())
                    pq.erase(old);
                pq.insert({new_len, i});
                //std::printf("lisataan %d %d %.2f\n", current_vertex, i, new_len);
                value[i] = {new_len, current_vertex};
            }
            
        }
    }
    std::pair<int, float> edge_lens[2] = {{-1, std::numeric_limits<float>::max()}, {-1, std::numeric_limits<float>::max()}};
    for (int i = 0; i < n; ++i) {
        if (i == excluded_vertex)
            continue;
        float len = cd(excluded_vertex, i);
        if (len < edge_lens[1].second && len < edge_lens[0].second) {
            edge_lens[1] = edge_lens[0];
            edge_lens[0] = {i, len};
        } else if (len < edge_lens[1].second) {
            edge_lens[1] = {i, len};
        }
    }
    std::printf("valitaan2 %d-%d %.2f\n", excluded_vertex, edge_lens[0].first, edge_lens[0].second);
    std::printf("valitaan2 %d-%d %.2f\n", excluded_vertex, edge_lens[1].first, edge_lens[1].second);
    length += edge_lens[0].second + edge_lens[1].second;
    degrees[edge_lens[0].first]++;
    degrees[edge_lens[1].first]++;
    degrees[excluded_vertex] += 2;
    return {length, degrees};
}

std::vector<float> subgradient_opt_alpha(std::vector<std::vector<float>>& coord) {
    int NDIM = coord.size();
    int n = coord[0].size();
    std::vector<float> pi(n, 0), best_pi(n, 0);
    const auto& [init_w, init_d] = prim_onetree(coord, pi);
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
                std::cout << pi[i] << " ";
            }
            std::cout << std::endl;
            last_v = v;
            const auto& [w, d] = prim_onetree(coord, pi);
            is_tour = true;
            for (int i = 0; i < n; ++i) {
                v[i] = d[i] - 2;
                is_tour &= (v[i] == 0);
            }
            std::cout << is_tour << " " << t << " " << period << " " << p << " " << w << std::endl;
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
    std::cout << "Done, best pi:" << std::endl;
    for (int i = 0; i < n; ++i)
        std::cout << best_pi[i] << " ";
    std::cout << std::endl;
    return best_pi;
}


int main(int argc, char** argv) {
    //std::srand(42);
    std::cout << std::setprecision(20);
    int seed = 42;
    std::srand(seed);
    int n = std::stoi(argv[1]);
    
    const int NDIM = 2;
    
    std::vector<std::vector<float>> p;
    for (int tests; tests < 10000; ++tests) {
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
                std::cout << tests <<" Eroaa: " << i << " " << piCPU[i] << " vs " << piGPU[i] << std::endl;
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