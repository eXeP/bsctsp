#include "preprocessing.h"

#include <vector>
#include <iostream>
#include <set>
#include <limits>
#include <math.h> 
#include <algorithm> 
#include <unordered_set>
#include "util.h"


std::vector<std::vector<float>> calculate_alpha(std::vector<std::vector<float>>& coords, std::vector<float>& pi, one_tree& onetree) {
    int n = coords[0].size();
    std::vector<std::vector<float>> alpha = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));
    std::vector<int> mark = std::vector<int>(n, -1);
    std::vector<float> b = std::vector<float>(n);
    for (int i = 0; i < n; ++i) {
        int from = onetree.topo[i];
        int to = 0;
        b[from] = -std::numeric_limits<float>::max()/2.f;
        if (from != onetree.first_node) {
            for (int to = from; to != onetree.first_node; to = onetree.dad[to]) {
                b[onetree.dad[to]] = std::max(b[to], d_ij(coords, pi, to, onetree.dad[to]));
                mark[to] = from;
            }
        }
        for (int j = 0; j < n; ++j) {
            to = onetree.topo[j];
            if (to == from)
                continue;
            if (from == onetree.special_node) {
                alpha[from][to] = (d_ij(coords, pi, from, to) <=  onetree.next_best[from]) ? 0.f : d_ij(coords, pi, from, to) - onetree.next_best[from];
            } else if (to == onetree.special_node) {
                alpha[from][to] = (d_ij(coords, pi, from, to) <=  onetree.next_best[to]) ? 0.f : d_ij(coords, pi, from, to) - onetree.next_best[to];
            } else {
                if (mark[to] != from) {
                    b[to] = std::max(b[onetree.dad[to]], d_ij(coords, pi, to, onetree.dad[to]));
                }
                //max for numerical errors.
                alpha[from][to] = std::max(d_ij(coords, pi, to, from) - b[to], 0.f);
            }
        }
    }
    return alpha;
}

std::vector<std::vector<std::pair<float, int>>> candidate_generation_alpha(std::vector<std::vector<float>>& coords, std::vector<float>& pi, one_tree& onetree, const int MAX_EDGES) {
    int n = coords[0].size();
    std::vector<std::vector<std::pair<std::pair<float, float>, int>>> alpha(n);
    std::vector<int> mark = std::vector<int>(n, -1);
    std::vector<float> b = std::vector<float>(n);

    auto add_alpha = [&alpha, MAX_EDGES, &coords](int i, int j, float value) {
        alpha[i].push_back({{value, distance(coords, i, j)}, j});
        std::sort(alpha[i].begin(), alpha[i].end());
        if (alpha[i].size() > MAX_EDGES)
            alpha[i].pop_back();
    };

    for (int i = 0; i < n; ++i) {
        int from = onetree.topo[i];
        int to = 0;
        b[from] = -std::numeric_limits<float>::max()/2.f;
        if (from != onetree.first_node) {
            for (int to = from; to != onetree.first_node; to = onetree.dad[to]) {
                b[onetree.dad[to]] = std::max(b[to], d_ij(coords, pi, to, onetree.dad[to]));
                mark[to] = from;
            }
        }
        for (int j = 0; j < n; ++j) {
            to = onetree.topo[j];
            if (to == from)
                continue;
            if (from == onetree.special_node) {
                add_alpha(from, to, (d_ij(coords, pi, from, to) <=  onetree.next_best[from]) ? 0.f : d_ij(coords, pi, from, to) - onetree.next_best[from]);
            } else if (to == onetree.special_node) {
                add_alpha(from, to, (d_ij(coords, pi, from, to) <=  onetree.next_best[to]) ? 0.f : d_ij(coords, pi, from, to) - onetree.next_best[to]);
            } else {
                if (mark[to] != from) {
                    b[to] = std::max(b[onetree.dad[to]], d_ij(coords, pi, to, onetree.dad[to]));
                }
                //max for numerical errors.
                add_alpha(from, to, std::max(d_ij(coords, pi, to, from) - b[to], 0.f));
            }
        }
    }

    std::vector<std::vector<std::pair<float, int>>> ret(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < MAX_EDGES; ++j) {
            ret[i].push_back({alpha[i][j].first.first, alpha[i][j].second});
        }
    }
    return ret;
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
    int start_vertex = rand()%n;
    value[start_vertex] = {0, start_vertex};
    std::set<std::pair<float, int>> pq;
    pq.insert({0, start_vertex});
    float length = 0;
    std::vector<int> degrees(n);
    std::vector<std::vector<int>> edges(n, std::vector<int>());
    std::vector<int> topo, dad(n, -1);
    std::vector<float> next_best(n);
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
            if (i == current_vertex || picked[i])
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
    int best_i = 0, best_j = 0;
    float second_longest = std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i) {
        if (degrees[i] == 1 && dad[i] != -1) {
            std::vector<std::pair<float, int>> lens;
            for (int j = 0; j < n; ++j) {
                if (i != j)
                    lens.push_back({c(i, j), j});
            }
            std::sort(lens.begin(), lens.end());
            next_best[i] = lens[1].first;
            if (lens[1].first > second_longest) {
                best_i = i;
                best_j = lens[1].second;
                second_longest = lens[1].first;
            }
        }
    }
    length += second_longest;
    degrees[best_i]++;
    degrees[best_j]++;

    one_tree onetree;
    onetree.length = length;
    onetree.degrees = degrees;
    onetree.edges = edges;
    onetree.dad = dad;
    onetree.topo = topo;
    onetree.first_node = start_vertex;
    onetree.special_node = best_i;
    onetree.next_best = next_best;

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
    int start_vertex = rand()%n;
    value[start_vertex] = {0, start_vertex};
    std::set<std::pair<float, int>> pq;
    pq.insert({0, start_vertex});
    float length = 0;
    std::vector<int> degrees(n);
   std::vector<int> dad(n, -1);
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
            //std::cout << "e " << std::min(current_vertex, value[current_vertex].second) << "-" << std::max(current_vertex, value[current_vertex].second) << " " << value[current_vertex].first << std::endl;
        }
        picked[current_vertex] = true;
        for (int i = 0; i < n; ++i) {
            if (i == current_vertex || picked[i])
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
    int best_i = 0, best_j = 0;
    float second_longest = std::numeric_limits<float>::min();
    for (int i = 0; i < n; ++i) {
        if (degrees[i] == 1) {
            std::vector<std::pair<float, int>> lens;
            for (int j = 0; j < n; ++j) {
                if (i != j)
                    lens.push_back({c(i, j), j});
            }
            std::sort(lens.begin(), lens.end());
            if (lens[1].first > second_longest) {
                best_i = i;
                best_j = lens[1].second;
                second_longest = lens[1].first;
            }
        }
    }
    //std::cout << "s " << std::min(best_i, best_j) << "-" << std::max(best_i, best_j) << " " << second_longest << std::endl;
    length += second_longest;
    degrees[best_i]++;
    degrees[best_j]++;
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
            //std::cout << pi[i] << " ";
            }
            //std::cout << std::endl;
            last_v = v;
            auto [w, d] = prim_onetree(coord, pi);
            for (int i = 0; i < n; ++i)
                w -= 2 * pi[i];
            is_tour = true;
            for (int i = 0; i < n; ++i) {
                v[i] = d[i] - 2;
                is_tour &= (v[i] == 0);
            }
            //printf("opt %.3f %.3f %.3f %d %d\n", w, best_w, t, p, period, initial_period);
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