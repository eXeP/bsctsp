#pragma once
#include <vector>
#include <unordered_set>

struct one_tree {
    float length;
    std::vector<int> degrees;
    std::vector<std::vector<int>> edges;
    std::vector<int> dad;
    std::vector<int> topo;
    std::vector<float> next_best;
    int first_node;
    int special_node;
};

std::vector<std::vector<float>> calculate_alpha(std::vector<std::vector<float>>& coords, std::vector<float>& pi, one_tree& onetree);
std::vector<std::vector<float>> calculate_exact_alpha(std::vector<std::vector<float>>& coords, std::vector<float>& pi);
std::vector<std::vector<std::pair<float, int>>> candidate_generation_alpha(std::vector<std::vector<float>>& coords, std::vector<float>& pi, one_tree& onetree, const int MAX_EDGES);

one_tree prim_onetree_edges(std::vector<std::vector<float>>& p, std::vector<float>& pi);

std::vector<float> subgradient_opt_alpha(std::vector<std::vector<float>>& coord);