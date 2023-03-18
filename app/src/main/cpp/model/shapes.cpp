//
// Created by samuel on 18/03/2023.
//


#include "shapes.h"

#include "../utils/asset.h"
#include "../utils/string_utils.h"

ObjShape::ObjShape(AAssetManager *mgr, const std::string &obj_file_path) {
    std::vector<std::tuple<float, float, float>> vertices_ref;
    std::vector<std::tuple<float, float, float>> normals_ref;

    std::vector<int> vertices_order;
    std::vector<int> normals_order;

    std::string file_content = read_asset(mgr, obj_file_path);
    std::vector<std::string> lines = split_string(file_content, '\n');

    for (const auto &line: lines) {
        std::vector<std::string> split_line = split_string(line, ' ');

        if (split_line[0] == "vn") {
            normals_ref.emplace_back(
                    std::stof(split_line[1]),
                    std::stof(split_line[2]),
                    std::stof(split_line[3])
            );
        } else if (split_line[0] == "v") {
            vertices_ref.emplace_back(
                    std::stof(split_line[1]),
                    std::stof(split_line[2]),
                    std::stof(split_line[3])
            );

        } else if (split_line[0] == "f") {
            vertices_order.push_back(std::stoi(split_string(split_line[1], '/')[0]));
            vertices_order.push_back(std::stoi(split_string(split_line[2], '/')[0]));
            vertices_order.push_back(std::stoi(split_string(split_line[3], '/')[0]));

            normals_order.push_back(std::stoi(split_string(split_line[1], '/')[2]));
            normals_order.push_back(std::stoi(split_string(split_line[2], '/')[2]));
            normals_order.push_back(std::stoi(split_string(split_line[3], '/')[2]));
        }
    }

    for (int i = 0; i < vertices_order.size(); i++) {
        vertices.push_back(vertices_ref[vertices_order[i] - 1]);
        normals.push_back(normals_ref[normals_order[i] - 1]);
    }
}

std::vector<std::tuple<float, float, float>> ObjShape::get_vertices() {
    return vertices;
}

std::vector<std::tuple<float, float, float>> ObjShape::get_normals() {
    return normals;
}