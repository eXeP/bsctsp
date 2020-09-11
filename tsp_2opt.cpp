#include <bits/stdc++.h>

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

    TSP_Graph (int dimension): dimension(dimension), coordinates(std::vector<std::array<int, 2>>(dimension+1)) {}
    
    void add_node(int i, int x, int y) {
        coordinates[i] = {x, y};
    }

    float dist(int i, int j) {
        return 
        std::sqrt(((float) coordinates[i][0]-coordinates[j][0])*((float) coordinates[i][0]-coordinates[j][0])
        +((float) coordinates[i][1]-coordinates[j][1])*((float) coordinates[i][1]-coordinates[j][1]));
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
    float dist = 0.0f;
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





struct Tour {
    int dimension;
    float length = 0.0f;
    std::vector<int> tour;
    Tour (int dimension): dimension(dimension), tour(std::vector<int>(dimension)){}
    
};

void shuffle_tour(TSP_Graph& g, Tour& t) {
    for (int swaps = 0; swaps < g.dimension/3; ++swaps) {
        int a = rand()%g.dimension, b = rand()%g.dimension;
        if (a == ((b-1+t.dimension)%t.dimension) || a == ((b+1+t.dimension)%t.dimension) || a == b) {
            continue;
        }
        int ai = t.tour[a], bi = t.tour[b];
        auto an = {t.tour[(a-1+t.dimension)%t.dimension], t.tour[(a+1+t.dimension)%t.dimension]}, 
        bn = {t.tour[(b-1+t.dimension)%t.dimension], t.tour[(b+1+t.dimension)%t.dimension]};
        
        for (auto f : an) {
            t.length -= g.dist(ai, f);
        }
        for (auto f : bn) {
            t.length -= g.dist(bi, f);
        }
        for (auto f : an) {
            t.length += g.dist(bi, f);
        }
        for (auto f : bn) {
            t.length += g.dist(ai, f);
        }
        t.tour[a] = bi;
        t.tour[b] = ai;
    }
}

void swap_2opt(Tour& t, int i, int j) {
    std::reverse(t.tour.begin()+i, t.tour.begin()+j+1);
}

void tsp_2opt(TSP_Graph& g) {
    Tour tour = Tour(g.dimension);
    for (int i = 1; i <= g.dimension; ++i) {
        tour.tour[i-1] = i;
    }
    tour.length = calculate_dist(g, tour.tour);
    Tour best = tour;
    for (int n_shuffles = 0; n_shuffles < 20; ++n_shuffles) {
        Tour best_local = tour;
        shuffle_tour(g, best_local);
        float better = false;
        do {
            better = false;
            std::pair<int, int> best_ij;
            float best_impr = 0.0f;
            for (int i = 1; i < g.dimension-2; ++i) {
                for (int j = i+1; j < g.dimension-1; ++j) {
                    float impr = g.dist(best_local.tour[i], best_local.tour[i-1]) + g.dist(best_local.tour[j+1], best_local.tour[j]) - 
                    (g.dist(best_local.tour[i], best_local.tour[j+1]) + g.dist(best_local.tour[i-1], best_local.tour[j]));
                    if (impr > 0.0f) {
                        better = true;
                        if (impr > best_impr) {
                            best_ij = {i, j};
                            best_impr = impr;
                        }
                    }
                }
            }
            if (better) {
                swap_2opt(best_local, best_ij.first, best_ij.second);
                best_local.length -= best_impr;
            }
            
        } while (better);
        if (best_local.length < best.length) {
            best = best_local;
        }
    }
    std::cout << "Smallest route: " << best.length << std::endl;
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
        b[node_id] = 0;
        int j = 0;
        for (int k = node_id; k != 2; k = j) {
            j = one_tree.dad[k];
            b[j] = std::max(b[k], g.dist(k, j));
            mark[j] = node_id;
        }
        for (j = 2; j <= g.dimension; ++j) {
            if (j != node_id) {
                if (mark[j] != node_id) {
                    b[j] = std::max(b[one_tree.dad[j]], g.dist(j, one_tree.dad[j]));
                    alpha[node_id][j] = g.dist(j, node_id)-b[j];
                }
            }
        }
    }
    return alpha;
}





void tsp_naive(TSP_Graph& g) {
    std::vector<int> perm = std::vector<int>(g.dimension);
    for (int i = 0; i < g.dimension; ++i)
        perm[i] = i;
    TSP_Solution best = TSP_Solution(calculate_dist(g, perm), perm);
    while (std::next_permutation(perm.begin(), perm.end())) {
        float new_cost = calculate_dist(g, perm);
        if (new_cost < best.cost) {
            best = TSP_Solution(new_cost, perm);
        }
    }
    std::cout << "Real best: " << best.cost << std::endl;
}

int main(int argc, char** argv) {
    std::srand(std::time(nullptr));
    TSP_Graph graph = read_graph(argv[1]);
    //tsp_2opt(graph);
    auto one_tree = prim_1tree(graph);
    auto alpha = calculate_alpha(graph, one_tree);
    std::cout << "Best: " << read_optimal(graph, argv[1]) << std::endl;
    //tsp_naive(graph);
}