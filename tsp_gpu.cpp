#include <bits/stdc++.h>
#include "tsp_2opt.cuh"

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

    TSP_Graph (int dimension): dimension(dimension), coordinates(std::vector<std::array<int, 2>>(dimension+1)), x(std::vector<int>(dimension)), y(std::vector<int>(dimension)) {}
    
    void add_node(int i, int x_coord, int y_coord) {
        coordinates[i] = {x_coord, y_coord};
        x[i-1] = x_coord;
        y[i-1] = y_coord;
    }

    float dist(int i, int j) {
        return 
        std::sqrt(((float) coordinates[i][0]-coordinates[j][0])*((float) coordinates[i][0]-coordinates[j][0])
        +((float) coordinates[i][1]-coordinates[j][1])*((float) coordinates[i][1]-coordinates[j][1]));
    }

    void shufflexy() {
        for (int swaps = 0; swaps < dimension/3; ++swaps) {
            int a = rand()%dimension, b = rand()%dimension;
            std::swap(x[a], x[b]);
            std::swap(y[a], y[b]);
        }
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
        bool better = false;
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

std::pair<int, int> two_opt_best(std::vector<int>& x, std::vector<int>& y) {
    int n = x.size();
    int best = 0, best_i = 0, best_j = 0;
    auto dist = [&x, &y](int i, int j) {
        int d1 = (x[i]-x[i-1])*(x[i]-x[i-1]) + (y[i]-y[i-1])*(y[i]-y[i-1]);
        int d2 = (x[j]-x[j+1])*(x[j]-x[j+1]) + (y[j]-y[j+1])*(y[j]-y[j+1]);
        int d3 = (x[i]-x[j+1])*(x[i]-x[j+1]) + (y[i]-y[j+1])*(y[i]-y[j+1]);
        int d4 = (x[j]-x[i-1])*(x[j]-x[i-1]) + (y[j]-y[i-1])*(y[j]-y[i-1]);
        return d1+d2-(d3+d4);
    };
    for (int i = 1; i < n-2; ++i) {
        for (int j = i+1; j < n-1; ++j) {
            int new_impr = dist(i, j);
            if (new_impr > best) {
                best = new_impr;
                best_i = i;
                best_j = j;
            }
        }
    }
    auto dist2 = [&x, &y](int i, int j) {
        int d1 = (x[i]-x[i-1])*(x[i]-x[i-1]) + (y[i]-y[i-1])*(y[i]-y[i-1]);
        int d2 = (x[j]-x[j+1])*(x[j]-x[j+1]) + (y[j]-y[j+1])*(y[j]-y[j+1]);
        int d3 = (x[i]-x[j+1])*(x[i]-x[j+1]) + (y[i]-y[j+1])*(y[i]-y[j+1]);
        int d4 = (x[j]-x[i-1])*(x[j]-x[i-1]) + (y[j]-y[i-1])*(y[j]-y[i-1]);
        std::cout << d1 << " " << d2 << " " << d3 << " " << d4 << std::endl;
        return d1+d2-(d3+d4);
    };
    dist2(best_i, best_j);
    std::cout << "CPU paras " << best << " " << best_i << " " << best_j << " " << x[best_i] << " " << x[best_j] << std::endl;
    return {best_i, best_j};
}

int main(int argc, char** argv) {
    std::srand(42);

    std::vector<int> x, y;
    int n = 1000;
    for (int i = 0; i < n; ++i) {
        x.push_back(rand()%200);
        y.push_back(rand()%200);
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;
    two_opt_best(x, y);
    run_gpu_2opt(x.data(), y.data(), n);

}