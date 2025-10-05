//
// Created by samuel on 18/03/2023.
//

#include <utility>

#include <phyvr_model/shapes.h>
#include <phyvr_utils/cache.h>
#include <phyvr_utils/singleton.h>
#include <phyvr_utils/string_utils.h>

/*
 * Obj Shape
 */

ObjShape::ObjShape(
    const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &obj_file_path) {

    auto cache = Singleton<Cache<std::shared_ptr<Shape>>>::get_singleton();
    if (cache->exists(obj_file_path)) {
        auto shape = cache->get(obj_file_path);
        shape_id = shape->get_id();
        vertices = shape->get_vertices();
        normals = shape->get_normals();
        return;
    }

    shape_id = obj_file_path;

    std::vector<std::tuple<float, float, float>> vertices_ref;
    std::vector<std::tuple<float, float, float>> normals_ref;

    std::vector<int> vertices_order;
    std::vector<int> normals_order;

    std::string file_content = file_reader->read_text(obj_file_path);
    std::vector<std::string> lines = split_string(file_content, '\n');

    for (const auto &line: lines) {
        std::vector<std::string> split_line = split_string(line, ' ');

        if (split_line[0] == "vn") {
            normals_ref.emplace_back(
                std::stof(split_line[1]), std::stof(split_line[2]), std::stof(split_line[3]));
        } else if (split_line[0] == "v") {
            vertices_ref.emplace_back(
                std::stof(split_line[1]), std::stof(split_line[2]), std::stof(split_line[3]));

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

    cache->add(obj_file_path, std::make_shared<FromMeshShape>(shape_id, vertices, normals));
}

std::vector<std::tuple<float, float, float>> ObjShape::get_vertices() { return vertices; }

std::vector<std::tuple<float, float, float>> ObjShape::get_normals() { return normals; }

std::string ObjShape::get_id() { return shape_id; }

// FromMeshShape

FromMeshShape::FromMeshShape(
    const std::string &shape_id, std::vector<std::tuple<float, float, float>> vertices,
    std::vector<std::tuple<float, float, float>> normals)
    : shape_id(shape_id), vertices(std::move(vertices)), normals(std::move(normals)) {}

std::vector<std::tuple<float, float, float>> FromMeshShape::get_vertices() { return vertices; }

std::vector<std::tuple<float, float, float>> FromMeshShape::get_normals() { return normals; }

std::string FromMeshShape::get_id() { return shape_id; }
