#include "util.h"

#include <string>
#include <iostream>
#include <fstream>

std::vector<std::vector<float>> read_graph(char* tsp_name) {
    std::string tsp_file_name = "tsplib/" + std::string(tsp_name) + ".tsp";
    std::ifstream tsp_file(tsp_file_name);
    std::string sink;
    int dimension;
    while(true) {
        std::getline(tsp_file, sink);
        if (sink.find("DIMENSION") != std::string::npos) {
            dimension = std::stoi(sink.substr(sink.find(":")+1));
            break;
        }
    }
    while(true) {
        std::getline(tsp_file, sink);
        if (sink.find("NODE_COORD_SECTION") != std::string::npos)
            break;
    }

    std::vector<std::vector<float>> coords;
    for (int i = 0; i < 2; ++i)
        coords.push_back(std::vector<float>());
    for (int i = 0; i < dimension; ++i) {
        float id, x, y;
        tsp_file >> id >> x >> y;
        coords[0].push_back(x);
        coords[1].push_back(y);
    }
    return coords;
}

float calculate_dist(std::vector<std::vector<float>>& coords, std::vector<int>& path) {
    float dist = 0;
    for(int i = 0; i < coords[0].size(); ++i)
        dist += distance(coords, path[i], path[(i+1)%coords[0].size()]);
    return dist;
}

float read_optimal(std::vector<std::vector<float>>& coords, char* tsp_name) {
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
    return calculate_dist(coords, tour);
}