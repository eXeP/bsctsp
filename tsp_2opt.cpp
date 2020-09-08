#include <bits/stdc++.h>

struct Graph {
    int dimension;
    std::vector<std::array<int, 2>> coordinates;

    Graph (int dimension): dimension(dimension), coordinates(std::vector<std::array<int, 2>>(dimension+1)) {}
    
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

Graph read_graph(char* tsp_name) {
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

    Graph g = Graph(dimension);
    for (int i = 0; i < dimension; ++i) {
        int id, x, y;
        tsp_file >> id >> x >> y;
        g.add_node(id, x, y);
    }
    return g;
}

float calculate_dist(Graph& g, std::vector<int>& path) {
    float dist = 0.0f;
    for(int i = 0; i < g.dimension; ++i)
        dist += g.dist(path[i], path[(i+1)%g.dimension]);
    return dist;
}

float read_optimal(Graph& g, char* tsp_name) {
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
    std::vector<std::array<int, 2>> edges;
    Tour (int dimension): dimension(dimension), edges(std::vector<std::array<int, 2>>(dimension+1, {0, 0})){}
    void remove_edge(int i, int j, Graph& g) {
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            if (edges[i][k] == j)
                edges[i][k] = 0;
            if (edges[j][k] == i)
                edges[j][k] = 0;
        }
        length -= g.dist(i, j);
    }
    void add_edge(int i, int j, Graph& g) {
        for (int k = 0; k < 2; ++k) {
            if (edges[i][k] == 0) {
                edges[i][k] = j;
                break;
            }
        }
        for (int k = 0; k < 2; ++k) {
            if (edges[j][k] == 0) {
                edges[j][k] = i;
                break;
            }
        }
        length += g.dist(i, j);
    }
};

void shuffle_tour(Graph& g, Tour& t) {
    for (int swaps = 0; swaps < g.dimension/3; ++swaps) {
        int a = rand()%g.dimension+1, b = rand()%g.dimension+1;
        auto an = t.edges[a], bn = t.edges[b];
        for (auto f : an) {
            t.remove_edge(a, f, g);
        }
        for (auto f : bn) {
            t.remove_edge(b, f, g);
        }
        for (auto f : an) {
            t.add_edge(b, f, g);
        }
        for (auto f : bn) {
            t.add_edge(a, f, g);
        }
    }
}


void tsp_2opt(Graph& g) {
    Tour tour = Tour(g.dimension);
    for (int i = 1; i <= g.dimension; ++i)
        tour.add_edge(i, (i+1)%g.dimension+1, g);
    Tour best = tour;
    for (int n_shuffles = 0; n_shuffles < 1; ++n_shuffles) {
        Tour best_local = tour;
        shuffle_tour(g, best_local);
        float better = false;
        do {
            better = false;
            std::pair<int, int> best_ij;
            float ij_impr = 9999999999.0f;
            for (int i = 2; i < g.dimension-2; ++i) {
                for (int j = i+1; j < g.dimension-1; ++j) {
                    if (g.dist(i, i-1) + g.dist(j+1, j) > g.dist(i, j+1) + g.dist(i-1, j)) {
                        better = true;
                        if (g.dist(i, j+1) + g.dist(i-1, j) < ij_impr) {
                            best_ij = {i, j};
                            ij_impr = g.dist(i, j+1) + g.dist(i-1, j);
                        }
                    }
                    
                }
            }
            if (better) {
                std::cout << "Parempi " << best_ij.first << " " << best_ij.second << std::endl;
                best_local.remove_edge(best_ij.first, best_ij.first-1, g);
                best_local.remove_edge(best_ij.second, best_ij.second+1, g);
                best_local.add_edge(best_ij.first, best_ij.second+1, g);
                best_local.add_edge(best_ij.first-1, best_ij.second, g);
            }
            //std::cout << "Paras: " << best_local.length << std::endl;
        } while (better);
        if (best_local.length < best.length) {
            best = best_local;
        }
    }
    std::cout << "Smallest route: " << best.length << std::endl;
}

void prim_1tree(Graph& g) {
    std::vector<bool> picked = std::vector<bool>(g.dimension, false);
    std::vector<std::pair<float, int>> value = std::vector<std::pair<float, int>>(g.dimension, {std::numeric_limits<float>::max()/10.0, -1});
    value[1] = {0, 1};
    std::set<std::pair<float, int>> pq;
    pq.insert({0, 1});
    Graph onetree = Graph(g.dimension);
    onetree.dimension = g.dimension;
    while (!pq.empty()) {
        auto current = *pq.begin();
        pq.erase(pq.begin());
        if (picked[current.second]) {
            continue;
        }
        /*onetree.add_edge(current.second, value[current.second].second, value[current.second].first-value[value[current.second].second].first);
        onetree.add_edge(value[current.second].second, current.second, value[current.second].first-value[value[current.second].second].first);
        picked[current.second] = true;
        for (auto& f: g.edges[current.second]) {
            if (current.first+f.first < value[current.second].first) {
                pq.erase(pq.find({value[current.second].first, f.second}));
                pq.insert({current.first+f.first, f.second});
                value[f.second] = {current.first+f.first, current.second};
            }
        }
        */
    }
    /*
    std::sort(g.edges[0].begin(), g.edges[0].end());
    onetree.add_edge(0, g.edges[0][0].second, g.edges[0][0].first);
    onetree.add_edge(g.edges[0][0].second, 0, g.edges[0][0].first);
    onetree.add_edge(0, g.edges[0][1].second, g.edges[0][1].first);
    onetree.add_edge(g.edges[0][1].second, 0, g.edges[0][1].first);
    */
}



void tsp_naive(Graph& g) {
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
    Graph graph = read_graph(argv[1]);
    tsp_2opt(graph);
    std::cout << "Best: " << read_optimal(graph, argv[1]) << std::endl;
    //tsp_naive(graph);
}