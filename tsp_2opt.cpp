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

Graph read_graph() {
    std::string sink;
    int dimension;
    std::getline(std::cin, sink);
    std::getline(std::cin, sink);
    std::getline(std::cin, sink);
    std::cin >> sink >> dimension;
    std::getline(std::cin, sink);
    std::getline(std::cin, sink);

    Graph g = Graph(dimension);
    for (int i = 0; i < dimension; ++i) {
        int id, x, y;
        std::cin >> id >> x >> y;
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

std::vector<int> two_opt_swap(std::vector<int> route, int i, int k) {
    std::reverse(route.begin()+i, route.begin()+k+1);
    return route;
}

void tsp_2opt(Graph& g) {
    std::vector<int> perm = std::vector<int>(g.dimension);
    for (int i = 0; i < g.dimension; ++i)
        perm[i] = i;
    TSP_Solution best = TSP_Solution(calculate_dist(g, perm), perm);
    for (int n_shuffles = 0; n_shuffles < 80; ++n_shuffles) {
        std::random_shuffle(perm.begin(), perm.end());
        TSP_Solution best_local = TSP_Solution(calculate_dist(g, perm), perm);
        float better = false;
        do {
            better = false;
            for (int i = 1; i < g.dimension-1; ++i) {
                for (int k = i+1; k < g.dimension; ++k) {
                    auto new_route = two_opt_swap(perm, i, k);
                    float new_route_cost = calculate_dist(g, new_route);
                    if (new_route_cost < best_local.cost) {
                        best_local = TSP_Solution(new_route_cost, new_route);
                        better = true;
                    }
                }
            }
        } while (better);
        if (best_local.cost < best.cost) {
            best = best_local;
        }
    }
    std::cout << "Smallest route: " << best.cost << std::endl;
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

int main() {
    Graph graph = read_graph();
    tsp_2opt(graph);
    //tsp_naive(graph);
}