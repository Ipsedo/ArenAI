//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_SHAPES_H
#define PHYVR_SHAPES_H

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <phyvr_utils/file_reader.h>

class Shape {
public:
    virtual ~Shape() = default;

    virtual std::vector<std::tuple<float, float, float>> get_vertices() = 0;

    virtual std::vector<std::tuple<float, float, float>> get_normals() = 0;

    virtual std::string get_id() = 0;
};

class ObjShape final : public Shape {
private:
    std::string shape_id;
    std::vector<std::tuple<float, float, float>> vertices;
    std::vector<std::tuple<float, float, float>> normals;

public:
    explicit ObjShape(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &obj_file_path);

    std::vector<std::tuple<float, float, float>> get_vertices() override;

    std::vector<std::tuple<float, float, float>> get_normals() override;

    std::string get_id() override;

    ~ObjShape() override;
};

class FromMeshShape final : public Shape {
private:
    std::string shape_id;
    std::vector<std::tuple<float, float, float>> vertices;
    std::vector<std::tuple<float, float, float>> normals;

public:
    FromMeshShape(
        const std::string &shape_id, std::vector<std::tuple<float, float, float>> vertices,
        std::vector<std::tuple<float, float, float>> normals);

    std::vector<std::tuple<float, float, float>> get_vertices() override;

    std::vector<std::tuple<float, float, float>> get_normals() override;

    std::string get_id() override;

    ~FromMeshShape() override;
};

#endif// PHYVR_SHAPES_H
