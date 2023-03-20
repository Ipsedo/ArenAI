//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_SHAPES_H
#define PHYVR_SHAPES_H

#include <string>
#include <tuple>
#include <vector>
#include <android/asset_manager.h>

class Shape {
public:
    virtual std::vector<std::tuple<float, float, float>> get_vertices() = 0;

    virtual std::vector<std::tuple<float, float, float>> get_normals() = 0;
};

class ObjShape : public Shape {
private:
    std::vector<std::tuple<float, float, float>> vertices;
    std::vector<std::tuple<float, float, float>> normals;

public:
    explicit ObjShape(AAssetManager *mgr, const std::string &obj_file_path);

    std::vector<std::tuple<float, float, float>> get_vertices() override;

    std::vector<std::tuple<float, float, float>> get_normals() override;
};

class FromMeshShape : public Shape {
private:
    std::vector<std::tuple<float, float, float>> vertices;
    std::vector<std::tuple<float, float, float>> normals;
public:
    FromMeshShape(std::vector<std::tuple<float, float, float>> vertices,
                  std::vector<std::tuple<float, float, float>> normals);

    std::vector<std::tuple<float, float, float>> get_vertices() override;

    std::vector<std::tuple<float, float, float>> get_normals() override;
};

#endif //PHYVR_SHAPES_H
