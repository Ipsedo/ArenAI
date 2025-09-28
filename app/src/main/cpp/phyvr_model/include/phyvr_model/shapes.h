//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_SHAPES_H
#define PHYVR_SHAPES_H

#include <android/asset_manager.h>
#include <string>
#include <tuple>
#include <vector>

#include <phyvr_utils/file_reader.h>

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
  explicit ObjShape(const std::shared_ptr<AbstractFileReader> &text_reader,
                    const std::string &obj_file_path);

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

#endif // PHYVR_SHAPES_H
