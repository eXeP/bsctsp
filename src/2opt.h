#pragma once

#include <vector>

std::pair<std::vector<float>, std::vector<float>> two_opt_best(std::vector<float> x, std::vector<float> y);

std::tuple<std::vector<float>, std::vector<float>, std::vector<int>> two_opt_best_restricted(std::vector<float> x, std::vector<float> y, std::vector<int> id, std::vector<std::vector<int>> allowed);