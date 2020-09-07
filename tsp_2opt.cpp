#include <bits/stdc++.h>

struct Graph {
    int nodes;
    std::vector<std::vector<std::pair<int, float>>> edges;
    

    Graph (int n): nodes(n), edges(std::vector<std::vector<std::pair<int, float>>>(n)){}

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

struct TSP_Solution {
    float cost;
    std::vector<int> solution;
    TSP_Solution (float cost, std::vector<int> solution): cost(cost), solution(solution) {}
};

float rand01(){
    return ((float) rand() / (RAND_MAX));
}

Graph random_graph(int n) {
    Graph g = Graph(n);
    for (int i = 0; i < n; ++i) {
        float len = rand01();
        g.add_edge(i, (i+1)%n, len);
        g.add_edge((i+1)%n, i, len);
        for (int j = 0; j < 7; ++j) {
            len = rand01();
            int rand_node = rand()%g.nodes;
            g.add_edge(i, rand_node, len);
            g.add_edge(rand_node, i, len);
        }
    }
    g.floyd_warshall();
    return g;
}

float calculate_dist(Graph& g, std::vector<int>& path) {
    float dist = 0.0f;
    for(int i = 0; i < g.nodes; ++i)
        dist += g.dist(path[i], path[(i+1)%g.nodes]);
    return dist;
}

std::vector<int> two_opt_swap(std::vector<int> route, int i, int k) {
    std::reverse(route.begin()+i, route.begin()+k+1);
    return route;
}

void tsp_2opt(Graph& g) {
    std::vector<int> perm = std::vector<int>(g.nodes);
    for (int i = 0; i < g.nodes; ++i)
        perm[i] = i;
    TSP_Solution best = TSP_Solution(calculate_dist(g, perm), perm);
    for (int n_shuffles = 0; n_shuffles < 2000; ++n_shuffles) {
        std::random_shuffle(perm.begin(), perm.end());
        TSP_Solution best_local = TSP_Solution(calculate_dist(g, perm), perm);
        float better = false;
        do {
            better = false;
            for (int i = 1; i < g.nodes-1; ++i) {
                for (int k = i+1; k < g.nodes; ++k) {
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
    std::vector<bool> picked = std::vector<bool>(g.nodes, false);
    std::vector<std::pair<float, int>> value = std::vector<std::pair<float, int>>(g.nodes, {std::numeric_limits<float>::max()/10.0, -1});
    value[1] = {0, 1};
    std::set<std::pair<float, int>> pq;
    pq.insert({0, 1});
    Graph onetree = Graph(g.nodes);
    onetree.nodes = g.nodes;
    while (!pq.empty()) {
        auto current = *pq.begin();
        pq.erase(pq.begin());
        if (picked[current.second]) {
            continue;
        }
        onetree.add_edge(current.second, value[current.second].second, value[current.second].first-value[value[current.second].second].first);
        onetree.add_edge(value[current.second].second, current.second, value[current.second].first-value[value[current.second].second].first);
        picked[current.second] = true;
        for (auto& f: g.edges[current.second]) {
            if (current.first+f.first < value[current.second].first) {
                pq.erase(pq.find({value[current.second].first, f.second}));
                pq.insert({current.first+f.first, f.second});
                value[f.second] = {current.first+f.first, current.second};
            }
        }
    }
    std::sort(g.edges[0].begin(), g.edges[0].end());
    onetree.add_edge(0, g.edges[0][0].second, g.edges[0][0].first);
    onetree.add_edge(g.edges[0][0].second, 0, g.edges[0][0].first);
    onetree.add_edge(0, g.edges[0][1].second, g.edges[0][1].first);
    onetree.add_edge(g.edges[0][1].second, 0, g.edges[0][1].first);
}



void tsp_naive(Graph& g) {
    std::vector<int> perm = std::vector<int>(g.nodes);
    for (int i = 0; i < g.nodes; ++i)
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
    Graph graph = random_graph(11);
    tsp_2opt(graph);
    tsp_naive(graph);
}