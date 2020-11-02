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
    auto c = [&p](int i, int j) {
        float c_ij = 0;
        for (int k = 0; k < p.size(); ++k) {
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
        const auto& [current_value, current_vertex] = *pq.begin();
        pq.erase(pq.begin());
        if (picked[current_vertex]) {
            continue;
        }
        degrees[current_vertex]++;
        degrees[value[current_vertex].second]++;
        picked[current_vertex] = true;
        for (int i = 0; i < n; ++i) {
            if (i == current_vertex || picked[i] || i == excluded_vertex)
                continue;
            float new_len = c(i, current_vertex);
            if (current_value+new_len < value[i].first) {
                auto old = pq.find({value[i].first, i});
                if (old != pq.end())
                    pq.erase(old);
                pq.insert({current_value+new_len, i});
                value[i] = {current_value+new_len, current_vertex};
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
    return {length, degrees};
}

void subgradient_opt_alpha(std::vector<std::vector<float>>& p) {
    int NDIM = p.size();
    int n = p[0].size();
    std::vector<float> pi(n, 0);
    float W = -1<<28;
    float t = 1.0;
    int period = n/2;
    int np = 4;
    while (true) {
        const auto& [length, d] = prim_onetree(p, pi);
        float w = length;
        for (int i = 0; i < n; ++i)
            w -= pi[i];
        W = std::max(W, w);
        bool is_tour = true;
        std::vector<int> v(n);
        for (int i = 0; i < n; ++i) {
            v[i] = d[i] - 2;
            is_tour &= v[i] == 0;
        }
        for (int i = 0; i < n; ++i)
            pi[i] = pi[i] + t * v[i];
        period--;
        if (period == 0) {
            t *= 0.5;
            period = n/np;
            np *= 2;
        }
        std::cout << is_tour << " " << t << " " << period << std::endl;
        if (is_tour || t < 0.001 || period == 0) 
            break;
    }
    std::cout << "Done, pi:" << std::endl;
    for (int i = 0; i < n; ++i)
        std::cout << pi[i] << " ";
    std::cout << std::endl;
}


int main(int argc, char** argv) {
    std::srand(42);
    //TSP_Graph graph = read_graph(argv[1]);
    std::vector<std::vector<float>> p;
    const int NDIM = 2;
    for (int i = 0; i < NDIM; ++i) {
        p.push_back(std::vector<float>());
    }
    int n = std::stoi(argv[1]);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < NDIM; ++j) {
            p[j].push_back(rand()%200);
        }
    }
    subgradient_opt_alpha(p);

    //auto one_tree = prim_1tree(graph);
    //auto alpha = calculate_alpha(graph, one_tree);

    //std::cout << "Best: " << read_optimal(graph, argv[1]) << std::endl;
    //tsp_naive(graph);
}